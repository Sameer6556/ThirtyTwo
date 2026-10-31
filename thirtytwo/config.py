"""Settings of the deployed system: protocol A of the v15 notebook (trained on 635 X-rays, every setting picked on the
70 held-out ones). Each value comes from assets/results_protocol_A_v15.csv, a model's own training log, or the
notebook cell named next to it."""
import json
import os
from pathlib import Path

from . import ROOT

# ----------------------------------------------------------------------------- assets (weights are not in git)
ASSETS = Path(os.environ.get('THIRTYTWO_ASSETS', ROOT / 'assets'))
ASSETS_URL = os.environ.get(
    'THIRTYTWO_ASSETS_URL',
    'https://github.com/Sameer6556/ThirtyTwo/releases/download/assets-v15/cusp_assets.zip')
ASSETS_HF_REPO = os.environ.get('THIRTYTWO_ASSETS_HF_REPO', '')   # optional Hugging Face model repo instead
ASSETS_SHA256 = os.environ.get('THIRTYTWO_ASSETS_SHA256',
                               '2819760bf22bac3dfa7116c5cf0ed46c7c0b2d6800cdbf0a4c2da99c8515fc06')
SETTINGS = Path(__file__).resolve().parent / 'settings'

# ----------------------------------------------------------------------------- preprocessing (cell 17, U.resize_long)
LONG_SIDE = 1280        # every detector reads this JPEG export ...
LONG_SIDE_HI = 1536     # ... except the round-3 student, which reads this one (ensure_dis_images_hi)
JPEG_QUALITY = 95

# ----------------------------------------------------------------------------- 1. clean tooth detector (cells 16, 18)
TOOTH = dict(weights='weights/tooth_clean.pt', imgsz=1024, conf=0.10, iou=0.5, agnostic_nms=True, max_det=64)

# ----------------------------------------------------------------------------- 2. crops (CFG CROP_H / CROP_W)
CROP_H, CROP_W = 384, 224

# ----------------------------------------------------------------------------- 3-4. tooth classifiers
# Architecture from each run's log.json. Box source and conf_pow were picked on validation (results rows
# 'crop only', seed 0, and 'self-trained crop only', picked=True).
CLASSIFIER_ARCH = dict(backbone='convnext_nano.in12k_ft_in1k', variant='single', dim=256, layers=2, heads=4,
                       drop_path=0.1, box_src='both', max_teeth=48, img_bs=2)
CLASSIFIERS = [
    dict(name='crop only', weights='cls/v9_A_single/best.pt', kind='c',
         boxes='median adapter', conf_pow=0.0),
    dict(name='self-trained crop only', weights='cls/v11_A_single_st_t4tta/best.pt', kind='cs',
         boxes='median adapter', conf_pow=0.0),
]

# ----------------------------------------------------------------------------- 5-6. disease detectors (YOLO)
# Boxes exactly as ultralytics' validator makes them (cell 18, DET_EVAL). Listed in the final fusion's member order
# (cell 26, fusion_members): the three plain YOLOs, then the self-trained students of rounds 1, 2 and 3.
DET_EVAL = dict(conf=0.001, iou=0.7, max_det=300)
DETECTORS = [
    dict(name='plain YOLO', weights='weights/disA.pt', imgsz=1024, export=LONG_SIDE, kind='d?'),
    dict(name='plain YOLO, seed 1', weights='weights/disA_s1.pt', imgsz=1024, export=LONG_SIDE, kind='d?'),
    dict(name='plain YOLO at 1280', weights='weights/disA_r1280.pt', imgsz=1280, export=LONG_SIDE, kind='d?'),
    dict(name='round-1 self-trained YOLO11m', weights='weights/disA_st1280_t4tta.pt', imgsz=1280, export=LONG_SIDE,
         kind='ds'),
    dict(name='round-2 self-trained YOLO11m', weights='weights/disA_st1280_r2t6tta.pt', imgsz=1280,
         export=LONG_SIDE, kind='ds'),
    dict(name='round-2 self-trained YOLO26m', weights='weights/disA_st1280_r2t6tta_y26.pt', imgsz=1280,
         export=LONG_SIDE, kind='ds'),
    dict(name='round-3 self-trained YOLO26m', weights='weights/disA_st1536_r3t8tta_hi.pt', imgsz=1536,
         export=LONG_SIDE_HI, kind='ds'),
]
QUICK_DETECTOR = 'round-1 self-trained YOLO11m'   # Quick mode: 48.08 AP on the official 50 (results row
                                                  # 'self-trained YOLO (1280)', no TTA)

# ----------------------------------------------------------------------------- 7. final fusion
# Results row 'ours v14: classifiers + self-trained YOLOs incl. round 3 (+ plain YOLOs), fused' without TTA,
# chosen 'WBF 1:1:1:1:1:1:1:1:1 iou 0.6, detector boxes'; the headline row says 'round 3, no TTA' (48.77 AP).
FUSION = dict(weights=[1.0] * 9, iou=0.6, detector_boxes=True,
              donor_iou=0.55)   # DONOR_IOU (cell 24): the YOLOs' plain WBF whose boxes the classifiers borrow
FUSION_KINDS = [c['kind'] for c in CLASSIFIERS] + [d['kind'] for d in DETECTORS]

# ----------------------------------------------------------------------------- output
DISEASES = [   # U.DIS order
    dict(key='caries', name='Caries', code='C'),
    dict(key='deep_caries', name='Deep caries', code='DC'),
    dict(key='periapical_lesion', name='Periapical lesion', code='PL'),
    dict(key='impacted', name='Impacted tooth', code='IT'),
]
MIN_TEETH = 10                  # fewer teeth found: probably not a panoramic X-ray
ASPECT_RANGE = (1.5, 3.2)       # width / height of a panoramic X-ray (DENTEX: see README)


def load_json(name):
    return json.loads((SETTINGS / name).read_text())


def load_adapter():
    """the median box adapter as U.adapt_box wants it: {(disease, upper jaw): [dx1, dy1, dx2, dy2]}"""
    return {(int(r['disease']), bool(r['upper'])): [float(v) for v in r['offsets']]
            for r in load_json('adapter.json')['table']}


def load_thresholds():
    """display thresholds per mode and disease key (best F1 at IoU 0.5 on the 70 protocol-A validation X-rays)"""
    t = load_json('thresholds.json')
    return {m: {d['key']: float(t[m][d['key']]) for d in DISEASES} for m in ('full', 'quick')}
