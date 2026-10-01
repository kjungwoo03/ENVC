"""Read 8-bit YUV420p as BT.709 full-range RGB tensors."""
import os
import numpy as np
import scipy.ndimage
import torch

RGB_EVAL_PROFILE = "rgb_bt709_full_bilinear_8bit"
BT709_WEIGHTS = (0.2126, 0.7152, 0.0722)

def validate_yuv420_dimensions(width: int, height: int) -> None:
    """Raise ValueError if the dimensions are not valid for YUV420p."""
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}×{height}.")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            f"YUV420p requires even width/height, got {width}×{height}."
        )

def read_yuv_frames(
    yuv_path: str,
    width: int,
    height: int,
    num_frames: int | None = None,
) -> list[torch.Tensor]:
    """Read raw YUV420p frames and return a list of CHW float32 RGB tensors in [0, 1].

    Each frame on disk: Y plane (H×W) + U plane (H/2×W/2) + V plane (H/2×W/2).
    This exactly mirrors the RGB-content preparation used by the official
    DCMVC/DCVC-DC/DCVC-FM repositories: full-range normalisation (/255),
    BT.709 conversion, bilinear 4:2:0 chroma upsampling, and RGB8 rounding.

    Args:
        yuv_path:   Path to the raw .yuv file.
        width:      Frame width in pixels (must be even).
        height:     Frame height in pixels (must be even).
        num_frames: Maximum frames to read (None → all frames in the file).

    Returns:
        List of CHW float32 tensors in [0, 1], one per decoded frame.

    Raises:
        ValueError: If the file size is not an integer multiple of one frame.
    """
    luma_size   = width * height
    chroma_size = (width // 2) * (height // 2)
    frame_bytes = luma_size + 2 * chroma_size

    file_size = os.path.getsize(yuv_path)
    if file_size % frame_bytes != 0:
        raise ValueError(
            f"{yuv_path} size ({file_size} bytes) is not divisible by one frame "
            f"({frame_bytes} bytes) for {width}×{height} YUV420p."
        )

    total = file_size // frame_bytes
    n = total if num_frames is None else min(num_frames, total)

    frames: list[torch.Tensor] = []
    with open(yuv_path, "rb") as fh:
        for _ in range(n):
            y = np.frombuffer(fh.read(luma_size),   dtype=np.uint8).reshape(height,      width)
            u = np.frombuffer(fh.read(chroma_size), dtype=np.uint8).reshape(height // 2, width // 2)
            v = np.frombuffer(fh.read(chroma_size), dtype=np.uint8).reshape(height // 2, width // 2)
            frames.append(_yuv420_to_rgb_tensor(y, u, v))
    return frames

def _yuv420_to_rgb_tensor(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> torch.Tensor:
    """Convert YUV420 to the official BT.709 RGB-content representation."""
    y_f = y.astype(np.float32)[None, ...] / 255.0
    uv = np.stack([u, v], axis=0).astype(np.float32) / 255.0
    uv_up = scipy.ndimage.zoom(uv, (1, 2, 2), order=1)

    cb = uv_up[0:1]
    cr = uv_up[1:2]
    Kr, Kg, Kb = BT709_WEIGHTS
    r = y_f + (2 - 2 * Kr) * (cr - 0.5)
    b = y_f + (2 - 2 * Kb) * (cb - 0.5)
    g = (y_f - Kr * r - Kb * b) / Kg

    # Official test_data_to_png.py writes uint8 PNGs with np.rint(), then the
    # model reader normalises them by 255. Reproduce that without disk I/O.
    rgb = np.concatenate([r, g, b], axis=0)
    rgb = np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.float32) / 255.0
    return torch.from_numpy(rgb)
