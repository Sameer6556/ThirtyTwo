"""Shared by the tools and tests: DENTEX records (parsed by the notebook's own dentex_utils.discover) and the
protocol-A keys from assets/keys."""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dentex_utils as U  # noqa: E402
from thirtytwo import config as C  # noqa: E402

DENTEX = Path(os.environ.get('THIRTYTWO_DENTEX', ROOT / 'data' / 'dentex_raw' / 'x'))


def have_dentex():
    return (DENTEX / 'validation_triple.json').is_file()


@functools.lru_cache(maxsize=1)
def sets():
    """U.discover of DENTEX; the LabelMe test labels are dropped as in the notebook (cell 12)"""
    s = U.discover(DENTEX, verbose=False)
    s.pop('dis_test', None)
    return s


def keys(name):
    return (C.ASSETS / 'keys' / f'{name}.txt').read_text().split()


def records():
    return {r['key']: r for k in ('dis_train', 'dis_val') for r in sets()[k]['images']}
