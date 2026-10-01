#!/usr/bin/env python3
"""Evaluate ENVC using entropy-estimated rates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--layout', choices=('native', 'pairs', 'sequences', 'yuv'), default='native',
                        help='Explicit frame protocol; native=color PNGs, pairs=simulator frames, sequences=images/events, yuv=baseline RGB conversion')
    parser.add_argument('--frames-subdir', default='frames_color_native')
    parser.add_argument('--seq-subdir', default='sequences')
    parser.add_argument('--event-root', type=Path)
    parser.add_argument('--event-subdir', default='')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sequences', nargs='+')
    parser.add_argument('--quality', type=int, choices=range(4), default=1)
    parser.add_argument('--num-frames', type=int, default=96, help='<=0: all frames')
    parser.add_argument('--intra-period', type=int, default=32)
    parser.add_argument('--p-frame-model-path', type=Path, required=True)
    parser.add_argument('--i-frame-model-path', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, help='Default: results/envc/<dataset>/<layout>/q<quality>-ip<period>')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-inputs', action='store_true', help='Check frames, event intervals, and weight paths without inference')
    args = parser.parse_args(argv)
    if args.intra_period == 0 or args.intra_period < -1:
        parser.error('--intra-period must be -1 or positive')
    if args.threads < 1:
        parser.error('--threads must be positive')
    if (args.width is None) != (args.height is None):
        parser.error('--width and --height must be supplied together')
    if args.width is not None and min(args.width, args.height) < 1:
        parser.error('Dimensions must be positive')
    if args.resume and (args.layout != 'yuv' or args.num_frames < 1):
        parser.error('--resume requires YUV input and a positive frame count')
    from utils.datasets import is_mcl_jcv, resolve_eval_dataset
    args.dataset = resolve_eval_dataset(args.dataset, args.width, args.height).resolve()
    args.dataset_name = "MCL-JCV" if is_mcl_jcv(args.dataset) else args.dataset.name
    if args.save_dir is None:
        args.save_dir = ROOT/'results/envc'/args.dataset_name/args.layout/f'q{args.quality}-ip{args.intra_period}'
    return args


def collect_inputs(args):
    from utils.envc_eval import event_path
    dataset = args.dataset
    folder = dataset / {'native': args.frames_subdir, 'pairs': 'pairs',
                        'sequences': args.seq_subdir, 'yuv': 'yuv'}[args.layout]
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing {args.layout} input directory: {folder}; select the intended --layout explicitly')
    if args.layout == 'yuv':
        entries = {p.stem: p for p in sorted(folder.glob('*.yuv')) if p.is_file()}
    else:
        entries = {p.name: p for p in sorted(folder.iterdir()) if p.is_dir()}
    if args.sequences:
        missing = set(args.sequences) - entries.keys()
        if missing:
            raise ValueError(f'Unknown sequences: {sorted(missing)}')
        entries = {name: entries[name] for name in dict.fromkeys(args.sequences)}
    if not entries:
        raise ValueError(f'No input sequences in {folder}')
    inputs = []
    for name, entry in entries.items():
        width = height = None
        if args.layout == 'yuv':
            match = re.search(r'(\d+)x(\d+)', name)
            if args.width is not None:
                width, height = args.width, args.height
                if match and (width, height) != tuple(map(int, match.groups())):
                    raise ValueError(f'Explicit dimensions disagree with {entry}')
            elif match:
                width, height = map(int, match.groups())
            else:
                raise ValueError(f'Missing WxH in {entry}; pass --width and --height')
            if width % 2 or height % 2:
                raise ValueError(f'YUV420 requires even dimensions: {entry}')
            count, remainder = divmod(entry.stat().st_size, width * height * 3 // 2)
            if remainder:
                raise ValueError(f'Invalid YUV420 file size: {entry}')
            paths = entry
        else:
            if args.layout == 'pairs':
                paths = [dataset/'frames'/name/Path(line.split('\t')[0]).name
                         for line in (entry/'pairs.txt').read_text().splitlines()
                         if line.strip() and not line.startswith('#')]
            else:
                paths = sorted((entry/'images' if args.layout == 'sequences' else entry).glob('*.png'))
            count = len(paths)
        if count < 1 or (args.num_frames > 0 and count < args.num_frames):
            raise ValueError(f'{name}: {count} frames available, requested {args.num_frames}')
        count = min(count, args.num_frames) if args.num_frames > 0 else count
        if isinstance(paths, list):
            paths = paths[:count]
            for p in paths:
                if not p.is_file():
                    raise FileNotFoundError(p)
        events = (args.event_root/name/args.event_subdir if args.event_root else
                  entry/'events' if args.layout == 'sequences' else dataset/'event_voxel_npz'/name)
        for index in range(1, count):
            if args.intra_period < 0 or index % args.intra_period:
                event_path(events, index - 1)
        inputs.append((name, paths, events, count, width, height))
    for p in (args.i_frame_model_path, args.p_frame_model_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    return inputs


def main(argv=None):
    args = parse_args(argv)
    inputs = collect_inputs(args)
    for name, _, events, count, _, _ in inputs:
        print(f'[INPUT] {name}: {count} frames, events={events}', flush=True)
    if args.check_inputs:
        print('Input paths and required event intervals OK; model inference was not run.')
        return

    import cupy
    import torch
    from utils.envc_eval import load_i_frame_net, load_p_frame_net, load_png_frames, compress_sequence
    from utils.metrics import avg_frame_metrics, init_lpips, warn_missing_libs
    from utils.utils import set_seed
    from utils.yuv_io import read_yuv_frames
    from utils.eval_resume import (atomic_json, prepare_run, sequence_provenance,
                                   load_cached_sequence, valid_sequence)
    from utils.envc_eval import event_path

    if not torch.cuda.is_available():
        raise RuntimeError('ENVC softsplat requires CUDA. Run from your allocated GPU shell.')
    if not hasattr(cupy.cuda, 'compile_with_cache'):
        raise RuntimeError(f'The softsplat compiler is unavailable in CuPy {cupy.__version__}. Check the documented CUDA/CuPy dependencies.')
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    i_net, _ = load_i_frame_net(device, args.i_frame_model_path)
    p_net, _ = load_p_frame_net(device, args.p_frame_model_path)
    warn_missing_libs()
    lpips_fn = init_lpips(device)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(model='ENVC', rate_mode='entropy_estimate',
                    event_bits_included=False, feature_reset_threshold=5.0,
                    deterministic_softsplat=os.environ.get('EVCOM_DETERMINISTIC_SOFTSPLAT', '0'),
                    deterministic_softsplat_block=os.environ.get('EVCOM_DETERMINISTIC_SOFTSPLAT_BLOCK', '32'))
    signed = args.layout == 'yuv' and args.num_frames > 0
    run_args = None
    if signed:
        dimensions = {(row[4], row[5]) for row in inputs}
        if len(dimensions) != 1:
            raise ValueError('A signed dataset evaluation requires uniform dimensions')
        width, height = dimensions.pop()
        event_inputs = {}
        for name, _, events, count, _, _ in inputs:
            intervals = []
            for index in range(1, count):
                if args.intra_period > 0 and index % args.intra_period == 0:
                    continue
                path = event_path(events, index - 1)
                stat = path.stat()
                intervals.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
            event_inputs[name] = intervals
        run_args = argparse.Namespace(**metadata)
        run_args.width, run_args.height = width, height
        run_args.eval_profile = 'rgb_bt709_full_bilinear_8bit'
        run_args.event_inputs = event_inputs
        provenance = prepare_run(run_args, 'envc',
                                 [args.p_frame_model_path, args.i_frame_model_path],
                                 'entropy_estimate', source_roots=(ROOT/'models', ROOT/'utils'))
        metadata.update(provenance, width=width, height=height,
                        num_frames_requested=args.num_frames, sequence_filter=args.sequences)
    atomic_json(args.save_dir/'args.json', metadata)
    summaries = {}
    for name, paths, events, count, width, height in inputs:
        cached = load_cached_sequence(run_args, paths, name) if signed else None
        if cached is not None:
            summaries[name] = {key: value for key, value in cached.items() if key.startswith('avg_')}
            print(f'[RESUME] {name}', flush=True)
            continue
        print(f'[EVAL] {name} q{args.quality}', flush=True)
        if args.layout == 'yuv':
            frames = read_yuv_frames(str(paths), width, height, count)
        else:
            frames, width, height = load_png_frames(paths)
        metrics = compress_sequence(p_net, i_net, frames, str(events), args.quality, args.quality,
                                    device, width, height, args.intra_period, lpips_fn=lpips_fn)
        averages = {f'avg_{key}': value for key, value in avg_frame_metrics(metrics).items()}
        result = dict(metadata, sequence=name, width=width, height=height,
                      num_frames_coded=len(metrics), num_intra_frames=sum(r['is_intra'] for r in metrics),
                      **averages, frames=metrics)
        if signed:
            provenance = sequence_provenance(run_args, paths)
            result.update(provenance)
            if not valid_sequence(result, run_args, provenance):
                raise RuntimeError(f'{name}: incomplete or invalid evaluation metrics')
        atomic_json(args.save_dir/f'{name}.json', result)
        summaries[name] = averages
        del frames
        print(f"[DONE] {name}: PSNR={averages['avg_psnr']:.4f}, bpp={averages['avg_bpp']:.6f}", flush=True)
    summary_name = 'summary.partial.json' if args.sequences else 'summary.json'
    means = {key.replace('avg_', 'mean_', 1): sum(row[key] for row in summaries.values()) / len(summaries)
             for key in next(iter(summaries.values()))}
    atomic_json(args.save_dir/summary_name,
                dict(metadata, num_sequences=len(summaries), sequences=summaries, **means))



if __name__ == '__main__':
    main()
