"""ThirtyTwo: every model loaded once; predict(image, mode) runs the deployed system on one X-ray.

full : clean tooth detector -> tooth crops -> 2 classifiers, 7 YOLOs -> classifier findings moved onto the YOLOs'
       boxes for their tooth -> WBF of the nine (weights 1, IoU 0.6). 48.77 AP on the official 50.
quick: clean tooth detector (tooth numbers only) + the round-1 self-trained YOLO11m. 48.08 AP.
"""
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C, images
from .classifier import KEY, ToothClassifier, tooth_rows
from .detectors import DiseaseDetector, ToothDetector
from .fusion import assign_tooth, make_det_preds, make_tooth_preds, wbf_members, with_detector_boxes

MODES = ('full', 'quick')


def _to_gray(image):
    """bytes of an upload, a file path, or an already decoded 2-D grey array"""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return images.decode_upload(bytes(image))
    if isinstance(image, (str, Path)):
        return images.decode_upload(Path(image).read_bytes())
    a = np.asarray(image)
    if a.ndim != 2 or a.dtype != np.uint8:
        raise images.ImageError('expected a 2-D uint8 grey image')
    return a


def panoramic_warning(w, h, teeth=None):
    """a plain sentence if the image does not look like a panoramic X-ray, else None"""
    lo, hi = C.ASPECT_RANGE
    if not lo <= w / h <= hi:
        return ('This does not look like a panoramic X-ray. Panoramic X-rays are about twice as wide as they are '
                'tall, and this image is not.')
    if teeth is not None and len(teeth) < C.MIN_TEETH:
        return (f'Only {len(teeth)} teeth were found, so this may not be a panoramic X-ray. The system expects a '
                'full panoramic X-ray of the jaws.')
    return None


class ThirtyTwo:
    def __init__(self, device=None, assets=None):
        from .assets import ensure_assets
        root = ensure_assets(assets)
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.adapter = C.load_adapter()
        try:
            self.thresholds = C.load_thresholds()
        except FileNotFoundError:          # before the accuracy check has written them
            self.thresholds = None
        self.tooth = ToothDetector(root / C.TOOTH['weights'], self.device)
        self.classifiers = [ToothClassifier(root / c['weights'], self.device) for c in C.CLASSIFIERS]
        self.detectors = {d['name']: DiseaseDetector(root / d['weights'], d['imgsz'], self.device)
                          for d in C.DETECTORS}

    # ------------------------------------------------------------------ the system
    def members(self, gray, teeth, exports):
        """the nine fusion members' predictions (notebook dicts), in the fusion's order"""
        rows, cache = tooth_rows(gray, teeth)
        out = [make_tooth_preds(rows, clf(rows, cache), c['conf_pow'], self.adapter if c['boxes'] == 'median adapter'
                                else None) for clf, c in zip(self.classifiers, C.CLASSIFIERS)]
        for d in C.DETECTORS:
            bgr, s = exports[d['export']]
            out.append(make_det_preds(self.detectors[d['name']](bgr, s), teeth))
        return out, rows

    def run(self, gray, mode):
        """(fused predictions as the notebook's dicts, teeth, warning); the accuracy check scores these"""
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        h, w = gray.shape[:2]
        warning = panoramic_warning(w, h)
        if warning:
            return [], [], warning
        exports = {C.LONG_SIDE: images.export(gray, C.LONG_SIDE)}
        teeth = self.tooth(*exports[C.LONG_SIDE])
        warning = panoramic_warning(w, h, teeth)
        if warning:
            return [], teeth, warning
        if mode == 'quick':
            bgr, s = exports[C.LONG_SIDE]
            return make_det_preds(self.detectors[C.QUICK_DETECTOR](bgr, s), teeth), teeth, None
        exports[C.LONG_SIDE_HI] = images.export(gray, C.LONG_SIDE_HI)
        members, rows = self.members(gray, teeth, exports)
        if C.FUSION['detector_boxes']:
            members = with_detector_boxes(members, C.FUSION_KINDS, {KEY: rows}, C.FUSION['donor_iou'])
        return wbf_members(members, C.FUSION['weights'], C.FUSION['iou']), teeth, None

    # ------------------------------------------------------------------ the public result
    def predict(self, image, mode='quick'):
        """image: upload bytes, a file path or a 2-D uint8 grey array. Returns a JSON-ready dict with the image size,
        every finding (class, score, box in original pixels, FDI tooth number or None), the teeth and the time."""
        t0 = time.perf_counter()
        gray = _to_gray(image)
        preds, teeth, warning = self.run(gray, mode)
        findings = []
        for p in sorted(preds, key=lambda z: -z['score']):
            d = C.DISEASES[int(p['d'])]
            j = assign_tooth(p, teeth)
            findings.append(dict(id=len(findings), cls=d['key'], name=d['name'], code=d['code'], score=p['score'],
                                 box=list(p['box']), tooth=None if j is None else int(teeth[j]['fdi'])))
        return dict(mode=mode, image=dict(width=int(gray.shape[1]), height=int(gray.shape[0])), warning=warning,
                    findings=findings,
                    teeth=[dict(fdi=int(t['fdi']), box=list(t['box']), score=float(t['conf']))
                           for t in sorted(teeth, key=lambda t: t['fdi'])],
                    thresholds=(self.thresholds or {}).get(mode),
                    time_s=round(time.perf_counter() - t0, 3))


def to_json(result):
    """the result with rounded numbers (boxes to 0.1 px, scores to 5 decimals) and 'class' as the field name"""
    def box(b):
        return [round(float(v), 1) for v in b]
    out = dict(result)
    out['findings'] = [dict(id=f['id'], **{'class': f['cls']}, name=f['name'], code=f['code'],
                            score=round(float(f['score']), 5), box=box(f['box']), tooth=f['tooth'])
                       for f in result['findings']]
    out['teeth'] = [dict(fdi=t['fdi'], box=box(t['box']), score=round(t['score'], 5)) for t in result['teeth']]
    return out
