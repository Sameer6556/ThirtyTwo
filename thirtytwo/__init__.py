"""Thirty Two: caries, deep caries, periapical lesions and impacted teeth on panoramic dental X-rays.

The deployed system is protocol A of the v15 notebook (dentex_twinarch_kaggle_2xT4.ipynb). The notebook's own helper
modules, dentex_utils.py and train_arch.py at the repository root, are imported as they are, never copied.
"""
import os
import sys
from pathlib import Path

# No telemetry and no surprise downloads: ultralytics sends usage events when it is online (settings 'sync'), and
# YOLO_OFFLINE turns that off before ultralytics is imported. The weights come from assets/ only.
os.environ.setdefault('YOLO_OFFLINE', '1')
os.environ.setdefault('YOLO_VERBOSE', 'False')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:      # dentex_utils.py and train_arch.py live at the repository root
    sys.path.insert(0, str(ROOT))

__version__ = '1.0.0'
