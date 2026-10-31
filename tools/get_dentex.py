"""Download DENTEX the way the v15 notebook's data cell does (cell 11): hf_hub_download of the four files, then
dentex_utils.safe_extract. The self-training pool (unlabelled and quadrant-only X-rays) is skipped, as the notebook
does when it does not need it (POOL_SKIP = ('unlabel', '/quadrant/')). Needs about 17 GB free while it runs, keeps
about 6 GB in data/dentex_raw/x.

    python tools/get_dentex.py
"""
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')     # no second copy of the 11 GB zip in a cache

from common import ROOT, U  # noqa: E402

REPO = 'ibrahimhamamci/DENTEX'
FILES = ['DENTEX/validation_triple.json', 'DENTEX/validation_data.zip', 'DENTEX/test_data.zip',
         'DENTEX/training_data.zip']
POOL_SKIP = ('unlabel', '/quadrant/')
RAW = ROOT / 'data' / 'dentex_raw'


def main():
    from huggingface_hub import hf_hub_download
    (RAW / 'x').mkdir(parents=True, exist_ok=True)
    if (RAW / '.done').exists():
        print('DENTEX is already in', RAW / 'x')
        return
    for f in FILES:
        t0 = time.time()
        p = hf_hub_download(REPO, f, repo_type='dataset', local_dir=str(RAW / 'dl'), token=os.environ.get('HF_TOKEN'))
        print(f'downloaded {f}: {os.path.getsize(p) / 1e9:.2f} GB in {time.time() - t0:.0f}s', flush=True)
        if p.endswith('.zip'):
            print(f'  extracted {U.safe_extract(p, RAW / "x", skip=POOL_SKIP)} files', flush=True)
            os.remove(p)
        else:
            shutil.copy(p, RAW / 'x' / Path(p).name)
    (RAW / '.done').touch()


if __name__ == '__main__':
    main()
