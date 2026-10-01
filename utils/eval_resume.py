"""Signed, sequence-level resume for the RGB evaluator."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

METRIC_NAMES = ('bpp', 'psnr', 'psnr_y', 'ssim', 'ms_ssim', 'lpips')
ROOT = Path(__file__).resolve().parent


def add_resume_args(parser):
    parser.add_argument('--resume', action='store_true',
                        help='Reuse only complete sequences with matching provenance and input stat.')
    parser.add_argument('--sequence', help='Evaluate one YUV filename (including .yuv).')
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU thread count (default: 4).')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_run(args, codec, checkpoint_paths, bitrate_source, source_roots=()):
    from utils import metrics

    if args.intra_period != -1 and args.intra_period < 1:
        raise ValueError('intra_period must be -1 or positive')
    if args.num_frames is not None and args.num_frames < 1:
        raise ValueError('num_frames must be positive')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if not metrics._HAVE_MSSSIM or not metrics._HAVE_LPIPS:
        raise RuntimeError('pytorch-msssim and lpips are required for a complete benchmark')
    ignored = {'resume', 'sequence', 'threads', 'save_dir', 'save_root', 'run_name',
               'eval_provenance', 'run_signature'}
    parameters = {key: value for key, value in vars(args).items() if key not in ignored}
    parameters['dataset'] = str(Path(args.dataset).resolve())
    checkpoints = {str(Path(path).resolve()): _sha256(path) for path in checkpoint_paths}
    code_paths = {Path(__file__).resolve(), ROOT/'metrics.py', ROOT/'yuv_io.py', ROOT/'utils.py', ROOT/'paths.py'}
    script = Path(sys.argv[0])
    if script.is_file():
        code_paths.add(script.resolve())
    suffixes = {'.py', '.cpp', '.cc', '.c', '.cu', '.h', '.hpp', '.cuh'}
    for source_root in source_roots:
        code_paths.update(path.resolve() for path in Path(source_root).rglob('*')
                          if path.is_file() and path.suffix in suffixes)
    code_hash = hashlib.sha256()
    for path in sorted(code_paths):
        code_hash.update(str(path).encode())
        code_hash.update(_sha256(path).encode())
    torch_module = sys.modules.get('torch')
    provenance = {
        'protocol_version': 'envc_rgb_signed_v1', 'codec': codec,
        'eval_profile': args.eval_profile, 'bitrate_source': bitrate_source,
        'checkpoint_sha256': checkpoints, 'code_sha256': code_hash.hexdigest(),
        'run_parameters': parameters,
        'torch_version': getattr(torch_module, '__version__', None),
        'cuda_version': getattr(getattr(torch_module, 'version', None), 'cuda', None),
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    args.run_signature = signature
    args.eval_provenance = {**provenance, 'run_signature': signature}
    atomic_json(Path(args.save_dir)/'provenance.json', args.eval_provenance)
    return args.eval_provenance


def sequence_provenance(args, yuv_path):
    path = Path(yuv_path)
    stat = path.stat()
    return {**args.eval_provenance, 'source_path': str(path.resolve()),
            'source_stat': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}


def _finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def valid_sequence(data, args, provenance):
    """Reject short/stale/corrupt or differently configured outputs."""
    if not isinstance(data, dict):
        return False
    for key in ('run_signature', 'source_path', 'source_stat', 'eval_profile'):
        if data.get(key) != provenance.get(key):
            return False
    for key in ('width', 'height', 'intra_period', 'num_frames_requested'):
        if data.get(key) != getattr(args, 'num_frames' if key == 'num_frames_requested' else key):
            return False
    if data.get('quality', data.get('q_index')) != args.quality:
        return False
    frame_bytes = args.width * args.height * 3 // 2
    size = provenance['source_stat']['size']
    if size % frame_bytes:
        return False
    count = args.num_frames if args.num_frames is not None else size // frame_bytes
    if count < 1 or size // frame_bytes < count or data.get('num_frames_coded') != count:
        return False
    frames = data.get('frames')
    if not isinstance(frames, list) or len(frames) != count:
        return False
    for idx, frame in enumerate(frames):
        intra = idx == 0 or (args.intra_period > 0 and idx % args.intra_period == 0)
        if not isinstance(frame, dict) or frame.get('frame_idx') != idx or frame.get('is_intra') is not intra:
            return False
        if any(not _finite_number(frame.get(key)) for key in METRIC_NAMES) or frame['bpp'] < 0:
            return False
    for key in METRIC_NAMES:
        average = data.get('avg_' + key)
        if not _finite_number(average):
            return False
        if abs(average - sum(frame[key] for frame in frames) / count) > 2e-6:
            return False
    return data['avg_bpp'] > 0


def load_cached_sequence(args, yuv_path, seq_name):
    if not args.resume:
        return None
    try:
        data = json.loads((Path(args.save_dir)/(seq_name + '.json')).read_text())
        return data if valid_sequence(data, args, sequence_provenance(args, yuv_path)) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None
