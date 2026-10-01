"""Canonical dataset selection shared by evaluation and plotting."""
from pathlib import Path


def resolve_eval_dataset(path, width=None, height=None):
    """Resolve the canonical MCL-JCV input without exposing a variant label."""
    path = Path(path).expanduser().absolute()
    parts = path.parts
    index = next((i for i, part in enumerate(parts) if part.lower() == 'mcl-jcv'), None)
    if index is None:
        return path
    tail = list(parts[index + 1:])
    if tail and tail[0].lower() in {'720p', '480p', '360p', '1440p', '4k', '2k'}:
        raise ValueError('This MCL-JCV variant is excluded from evaluation and plotting')
    if (width is not None and width != 1920) or (height is not None and height != 1080):
        raise ValueError('MCL-JCV requires width=1920 and height=1080')
    if tail and tail[0].lower() == '1080p':
        tail.pop(0)
    return Path(*parts[:index + 1]) / '1080p' / Path(*tail)


def is_mcl_jcv(path):
    return any(part.lower() == 'mcl-jcv' for part in Path(path).parts)
