"""Checkpoint loading, event inputs, and RGB evaluation for ENVC."""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from PIL import Image
from models import ENVC
from models.image_model import IntraNoAR
from utils.envc_stream import get_padding_size
from utils.event_voxel import NUM_BINS, build_voxel
from utils.metrics import (
    compute_psnr, compute_psnr_y, compute_ssim,
    compute_ms_ssim, compute_lpips, avg_frame_metrics,
)

ENVC_MIN_DIV = 64

def _load_state_dict(ckpt_path: str) -> dict:
    """Load an ENVC/DCVC-style checkpoint into a plain state dict."""
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict):
        for key in ("state_dict", "net", "model", "model_state_dict", "p_net"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                ckpt = value
                break

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint {ckpt_path} did not contain a PyTorch state dict."
        )

    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt

def _require_state_keys(state: dict, required: set[str], ckpt_path: str, label: str) -> None:
    missing = sorted(required.difference(state.keys()))
    if not missing:
        return

    sample = ", ".join(list(state.keys())[:8])
    raise RuntimeError(
        f"{label} checkpoint does not look compatible with ENVC.\n"
        f"Path: {ckpt_path}\n"
        f"Missing keys: {', '.join(missing)}\n"
        f"First checkpoint keys: {sample}"
    )

def load_i_frame_net(device: torch.device, ckpt_path: str):
    """Load the IntraNoAR I-frame model (frozen DCVC-DC intra)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"I-frame checkpoint not found: {ckpt_path}")
    state = _load_state_dict(ckpt_path)
    _require_state_keys(state, {"q_scale_enc", "q_scale_dec"}, ckpt_path, "I-frame")
    net = IntraNoAR(inplace=True)
    net.load_state_dict(state)
    net = net.to(device)
    net.eval()
    return net, ckpt_path


def load_p_frame_net(device, ckpt_path):
    state = _load_state_dict(ckpt_path)
    _require_state_keys(state, {
        "y_q_scale_enc", "y_q_scale_dec", "mv_y_q_scale_enc", "mv_y_q_scale_dec",
        "front_end.image_encoder.pyramid1.0.0.weight",
    }, ckpt_path, "ENVC P-frame")
    net = ENVC(inplace=True)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval(), str(ckpt_path)


def event_path(npz_dir, interval_idx):
    candidates = [Path(npz_dir) / f"{interval_idx:05d}.npz",
                  Path(npz_dir) / f"{interval_idx:06d}.npz"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one event interval {interval_idx} in {npz_dir}; found {found}")
    return found[0]


def load_event_voxel(npz_dir, interval_idx, width, height, device):
    path = event_path(npz_dir, interval_idx)
    with np.load(path) as data:
        n_events = int(data["x"].shape[0])
    voxel = build_voxel(path, width, height, NUM_BINS)
    return torch.from_numpy(voxel).unsqueeze(0).to(device), n_events / float(width * height)


def compress_sequence(
    p_net,
    i_net,
    frames: list,
    npz_dir: str,
    i_q_index: int,
    p_q_index: int,
    device: torch.device,
    width: int,
    height: int,
    intra_period: int = -1,
    lpips_fn=None,
):
    """
    Compress and reconstruct a list of RGB frames using ENVC.

    intra_period = -1 : single I-frame at frame 0, rest are P-frames.
    intra_period = N  : I-frame every N frames.

    q_in_ckpt=True for the four embedded rate points; frame_idx % 4 keeps the
    P-frame model's context adaptor on the four-frame context cycle. P-frame t uses
    the event voxel of interval t-1 -> t (npz file t-1).
    """
    if not frames:
        return []
    if intra_period == 0 or intra_period < -1:
        raise ValueError("intra_period must be -1 or a positive integer.")

    h, w = frames[0].shape[1], frames[0].shape[2]
    pixels = h * w
    pad_l, pad_r, pad_t, pad_b = get_padding_size(h, w, ENVC_MIN_DIV)

    frame_results: list[dict] = []
    dpb = None

    with torch.no_grad():
        for frame_idx, frame in enumerate(frames):
            x = frame.unsqueeze(0).to(device)  # (1, C, H, W), float32 [0, 1]
            x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")

            is_intra = (frame_idx == 0) or (
                intra_period > 0 and frame_idx % intra_period == 0
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if is_intra:
                result = i_net.encode_decode(
                    x_pad,
                    q_in_ckpt=True,
                    q_index=i_q_index,
                    output_path=None,
                    pic_height=h,
                    pic_width=w,
                )
                rec_pad = result["x_hat"].clamp(0.0, 1.0)
                bpp = result["bit"] / pixels
                bpp_parts = {}

                dpb = {
                    "ref_frame": rec_pad,
                    "ref_feature": None,
                    "ref_y": None,
                }
            else:
                if dpb is None:
                    raise RuntimeError("ENVC P-frame encountered before DPB initialisation.")
                voxel, events_per_pixel = load_event_voxel(
                    npz_dir, frame_idx - 1, w, h, device
                )
                # Events outside the frame do not exist: zero-pad, never
                # replicate (replication would fabricate border events).
                voxel_pad = F.pad(voxel, (pad_l, pad_r, pad_t, pad_b),
                                  mode="constant", value=0.0)
                result = p_net.encode_decode(
                    x_pad,
                    dpb,
                    voxel_pad,
                    q_in_ckpt=True,
                    q_index=p_q_index,
                    output_path=None,
                    pic_height=h,
                    pic_width=w,
                    frame_idx=frame_idx % 4,
                )
                dpb = result["dpb"]
                rec_pad = dpb["ref_frame"].clamp(0.0, 1.0)
                bpp = result["bit"] / pixels
                # Separate transmitted residual-motion and image-residual rates.
                bpp_parts = {
                    "bpp_mv": round((result["bit_mv_y"] + result["bit_mv_z"]) / pixels, 6),
                    "bpp_res": round((result["bit_y"] + result["bit_z"]) / pixels, 6),
                    "events_per_pixel": round(events_per_pixel, 6),
                }
                del voxel, voxel_pad

            del x_pad, x

            rec = rec_pad[:, :, pad_t:pad_t + h, pad_l:pad_l + w]
            orig_b = frame.unsqueeze(0)
            rec_b = rec.cpu()

            psnr = compute_psnr(orig_b, rec_b)
            psnr_y = compute_psnr_y(orig_b, rec_b)
            ssim = compute_ssim(orig_b, rec_b)
            ms_ssim = compute_ms_ssim(orig_b, rec_b)
            lp = compute_lpips(orig_b.to(device), rec_b.to(device), lpips_fn)

            frame_dict: dict = {
                "frame_idx": frame_idx,
                "is_intra": is_intra,
                "psnr": round(psnr, 4),
                "psnr_y": round(psnr_y, 4),
                "bpp": round(bpp, 6),
                **bpp_parts,
            }
            if ssim is not None:
                frame_dict["ssim"] = round(ssim, 6)
            if ms_ssim is not None:
                frame_dict["ms_ssim"] = round(ms_ssim, 6)
            if lp is not None:
                frame_dict["lpips"] = round(lp, 6)

            frame_results.append(frame_dict)

    return frame_results

def load_png_frames(paths: list[str]) -> tuple[list[torch.Tensor], int, int]:
    """CHW float32 RGB tensors in [0, 1], matching yuv_io.read_yuv_frames()'s
    output contract exactly, so compress_sequence() needs no changes."""
    frames = []
    width = height = None
    for path in paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if width is None:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError(f"{path} is {w}x{h}, expected {width}x{height}")
        arr = torch.from_numpy(
            __import__("numpy").array(img, dtype="float32") / 255.0
        )
        frames.append(arr.permute(2, 0, 1).contiguous())
    return frames, width, height
