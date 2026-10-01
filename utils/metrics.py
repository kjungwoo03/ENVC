#!/usr/bin/env python3
"""
Quality metrics for image/video compression evaluation.

All functions accept BCHW float32 tensors in [0, 1], channel order RGB.

Optional dependencies (graceful degradation when missing):
  pip install pytorch-msssim   # SSIM / MS-SSIM
  pip install lpips            # LPIPS
"""

import math
from pathlib import Path
import shutil

from utils.paths import WEIGHTS_ROOT

import torch

# ---------------------------------------------------------------------------
# Optional library availability flags
# ---------------------------------------------------------------------------

try:
    from pytorch_msssim import ssim as _ssim_fn, ms_ssim as _ms_ssim_fn
    _HAVE_MSSSIM = True
except ImportError:
    _HAVE_MSSSIM = False

try:
    import lpips as _lpips_module
    _HAVE_LPIPS = True
except ImportError:
    _HAVE_LPIPS = False


# ---------------------------------------------------------------------------
# Frame-level metrics  (BCHW float [0, 1])
# ---------------------------------------------------------------------------

def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """RGB PSNR between two BCHW float tensors in [0, 1]."""
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def compute_psnr_y(a: torch.Tensor, b: torch.Tensor) -> float:
    """Luma (Y) PSNR — the standard metric in video compression papers.

    BT.709 full-range coefficients, matching the RGB evaluation profile.
    Both tensors must be BCHW float in [0, 1], channel order RGB.
    """
    weights = a.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    ya = (a * weights).sum(dim=1, keepdim=True)
    yb = (b * weights).sum(dim=1, keepdim=True)
    mse = torch.mean((ya - yb) ** 2).item()
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def compute_ssim(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """SSIM via pytorch_msssim. Returns None if library not installed."""
    if not _HAVE_MSSSIM:
        return None
    return _ssim_fn(a, b, data_range=1.0).item()


def compute_ms_ssim(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """MS-SSIM via pytorch_msssim.

    Requires spatial size ≥ 160 px (5 pooling scales × 2^4 stride).
    Returns None if the library is unavailable or the input is too small.
    """
    if not _HAVE_MSSSIM:
        return None
    if a.shape[-2] < 160 or a.shape[-1] < 160:
        return None
    return _ms_ssim_fn(a, b, data_range=1.0).item()


def compute_lpips(a: torch.Tensor, b: torch.Tensor, lpips_fn) -> float | None:
    """LPIPS perceptual distance (lower = more similar).

    Accepts BCHW float tensors in [0, 1]. lpips_fn is the model returned by
    init_lpips(); it must be on the same device as a and b.
    Returns None if lpips_fn is None.
    """
    if lpips_fn is None:
        return None
    a_n = a * 2.0 - 1.0
    b_n = b * 2.0 - 1.0
    with torch.no_grad():
        val = lpips_fn(a_n, b_n)
    return val.mean().item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_lpips(device: torch.device, net: str = "alex"):
    """Initialise and return an LPIPS model.

    Args:
        device: Target device (CPU / CUDA).
        net:    Backbone — "alex" (faster, default) or "vgg" (more accurate).

    Returns:
        Loaded LPIPS model or None if the lpips library is not installed.
    """
    if not _HAVE_LPIPS:
        return None
    metric_root = WEIGHTS_ROOT / "lpips"
    calibration = metric_root / "weights" / "v0.1" / f"{net}.pth"
    if not calibration.is_file():
        bundled = Path(_lpips_module.__file__).parent / "weights" / "v0.1" / f"{net}.pth"
        calibration.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled, calibration)
    # torchvision uses the torch hub cache for its ImageNet backbone. Keep
    # those weights under the same central root, then restore the global setting.
    previous_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(metric_root))
    try:
        fn = _lpips_module.LPIPS(net=net, model_path=str(calibration)).to(device)
    finally:
        torch.hub.set_dir(previous_hub)
    fn.eval()
    return fn


def avg_frame_metrics(frame_results: list[dict]) -> dict:
    """Average per-frame metric dicts into a single summary dict.

    Keys present in only a subset of frames are still averaged over available
    (non-None) values. Keys absent from every frame are omitted.
    The special key "frame_idx" is always excluded from the average.
    """
    if not frame_results:
        return {}
    all_keys = {k for r in frame_results for k in r if k != "frame_idx"}
    out: dict = {}
    for key in all_keys:
        vals = [r[key] for r in frame_results if key in r and r[key] is not None]
        if vals:
            out[key] = round(sum(vals) / len(vals), 6)
    return out


def warn_missing_libs() -> None:
    """Print one-time warnings for unavailable optional metric libraries."""
    if not _HAVE_MSSSIM:
        print(
            "[WARN] pytorch_msssim not installed — SSIM/MS-SSIM will be skipped.\n"
            "       Install with: pip install pytorch-msssim"
        )
    if not _HAVE_LPIPS:
        print(
            "[WARN] lpips not installed — LPIPS will be skipped.\n"
            "       Install with: pip install lpips"
        )
