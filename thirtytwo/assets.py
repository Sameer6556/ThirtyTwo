"""Model weights: cusp_assets.zip from this repository's GitHub release (or a Hugging Face model repo), downloaded and
unpacked at first start. Only the files the app needs are extracted."""
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import config as C

NEEDED = sorted({C.TOOTH['weights']} | {c['weights'] for c in C.CLASSIFIERS} | {d['weights'] for d in C.DETECTORS})


def missing(root=None):
    root = Path(root or C.ASSETS)
    return [f for f in NEEDED if not (root / f).is_file()]


def _download(dest):
    if C.ASSETS_HF_REPO:
        from huggingface_hub import hf_hub_download
        print(f'[thirtytwo] downloading cusp_assets.zip from the Hugging Face model repo {C.ASSETS_HF_REPO}',
              file=sys.stderr, flush=True)
        p = hf_hub_download(C.ASSETS_HF_REPO, 'cusp_assets.zip', repo_type='model', local_dir=str(dest.parent))
        if Path(p) != dest:
            shutil.move(p, dest)
        return dest
    print(f'[thirtytwo] downloading {C.ASSETS_URL}', file=sys.stderr, flush=True)
    part = dest.with_suffix('.part')
    with urllib.request.urlopen(C.ASSETS_URL, timeout=60) as r, open(part, 'wb') as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    part.replace(dest)
    return dest


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ensure_assets(root=None):
    """the folder that holds every weight file the app needs; downloads and unpacks them if any is missing"""
    import dentex_utils as U
    root = Path(root or C.ASSETS)
    if not missing(root):
        return root
    root.mkdir(parents=True, exist_ok=True)
    z = root / 'cusp_assets.zip'
    if not z.is_file():
        _download(z)
    if C.ASSETS_SHA256 and _sha256(z) != C.ASSETS_SHA256:
        bad = z.with_suffix('.zip.bad')
        z.replace(bad)
        raise RuntimeError(f'{z.name} does not have the expected SHA-256 (moved to {bad.name}). If you replaced the '
                           'release asset on purpose, set THIRTYTWO_ASSETS_SHA256 to the new hash, or to an empty '
                           'string to skip the check.')
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
    absent = [f for f in NEEDED if f not in names]
    if absent:
        raise RuntimeError(f'{z.name} lacks {absent}')
    U.safe_extract(z, root, skip=(), keep=lambda n: n in NEEDED or n == 'MANIFEST.txt')
    if os.environ.get('THIRTYTWO_KEEP_ZIP', '') != '1':
        z.unlink()
    still = missing(root)
    if still:
        raise RuntimeError(f'could not unpack {still}')
    return root


if __name__ == '__main__':
    print(ensure_assets())
