import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from thirtytwo import config as C  # noqa: E402
from thirtytwo.assets import missing  # noqa: E402
import common  # noqa: E402

SAMPLES = ROOT / 'thirtytwo' / 'static' / 'samples'
needs_assets = pytest.mark.skipif(bool(missing()), reason='model weights not in assets/ (python -m thirtytwo.assets)')
needs_references = pytest.mark.skipif(not (C.ASSETS / 'teeth_clean.pkl').is_file(),
                                      reason='the saved references of cusp_assets.zip are not in assets/')
needs_dentex = pytest.mark.skipif(not common.have_dentex(), reason='DENTEX is not in data/ (see README)')


@pytest.fixture(scope='session')
def model():
    from thirtytwo.pipeline import ThirtyTwo
    return ThirtyTwo(device='cpu')


@pytest.fixture(scope='session')
def sample_bytes():
    return (SAMPLES / 'val_14.png').read_bytes()
