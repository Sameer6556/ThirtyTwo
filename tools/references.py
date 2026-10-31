"""The saved protocol-A outputs in assets/ (made on Kaggle by the v15 notebook), as the notebook used them.

clean_rows rebuilds the crop ids of the clean crops (cell 20: tasks of 50 X-rays, 2,000 ids each from 30,000,000), so
the saved classifier scores (cls/*_E/pred_*.csv) can be matched to teeth without any image. replay() rebuilds the final
fusion from the saved member outputs with the app's own fusion code.
"""
import functools
import pickle

import numpy as np
import pandas as pd

from common import C, U, keys, sets
from thirtytwo.fusion import wbf_members, with_detector_boxes

SAVED_CLS = {'crop only': 'v9_A_single_E', 'self-trained crop only': 'v11_A_single_st_t4tta_E'}
SAVED_BOXES = {
    'plain YOLO': 'disbox_A_validator.pkl',
    'plain YOLO, seed 1': 'disbox_A_s1_validator.pkl',
    'plain YOLO at 1280': 'disbox_A_r1280_validator.pkl',
    'round-1 self-trained YOLO11m': 'disbox_disA_st1280_t4tta_validator.pkl',
    'round-2 self-trained YOLO11m': 'disbox_disA_st1280_r2t6tta_validator.pkl',
    'round-2 self-trained YOLO26m': 'disbox_disA_st1280_r2t6tta_y26_validator.pkl',
    'round-3 self-trained YOLO26m': 'disbox_disA_st1536_r3t8tta_hi_validator.pkl',
}
REPORT = {   # results_protocol_A_v15.csv rows (variant, boxes) of every member, no TTA
    'crop only': ('crop only', 'median adapter'),
    'self-trained crop only': ('self-trained crop only', 'median adapter'),
    'plain YOLO': ('plain YOLO', 'detector'),
    'plain YOLO, seed 1': ('plain YOLO, seed 1 (control)', 'detector'),
    'plain YOLO at 1280': ('plain YOLO at 1280', 'detector'),
    'round-1 self-trained YOLO11m': ('self-trained YOLO (1280)', 'detector'),
    'round-2 self-trained YOLO11m': ('round-2 self-trained YOLO (1280)', 'detector'),
    'round-2 self-trained YOLO26m': ('round-2 self-trained YOLO26m (1280)', 'detector'),
    'round-3 self-trained YOLO26m': ('round-3 self-trained YOLO26m (1536)', 'detector'),
    'final fusion': ('ours v14: classifiers + self-trained YOLOs incl. round 3 (+ plain YOLOs), fused', 'fused'),
}


@functools.lru_cache(maxsize=None)
def teeth_clean():
    return pickle.loads((C.ASSETS / 'teeth_clean.pkl').read_bytes())


@functools.lru_cache(maxsize=1)
def clean_rows():
    """{image key: crop rows} of the clean crops, crop ids as the notebook assigned them"""
    out, task, T = {}, 0, teeth_clean()
    for split in ('dis_train', 'dis_val'):
        recs = sets()[split]['images']
        for i in range(0, len(recs), 50):
            cid = 30_000_000 + task * 2_000
            for r in recs[i:i + 50]:
                for t in T.get(r['key'], []):
                    x1, y1, x2, y2 = t['box']
                    out.setdefault(r['key'], []).append(dict(
                        crop_id=cid, image_key=r['key'], q=t['q'], n=t['n'], fdi=t['fdi'], x1=x1, y1=y1, x2=x2,
                        y2=y2, det_conf=t['conf'], gt_only=0))
                    cid += 1
            task += 1
    return out


@functools.lru_cache(maxsize=None)
def saved_probs(member, part):
    d = pd.read_csv(C.ASSETS / 'cls' / SAVED_CLS[member] / f'pred_{part}.csv')
    return {int(c): np.array(v, float) for c, v in zip(d.crop_id, d[['p_caries', 'p_deep', 'p_peri', 'p_impacted']].values)}


@functools.lru_cache(maxsize=None)
def saved_boxes(member):
    return pickle.loads((C.ASSETS / SAVED_BOXES[member]).read_bytes())[0]


def report_row(member):
    T = pd.read_csv(C.ASSETS / 'results_protocol_A_v15.csv')
    variant, boxes = REPORT[member]
    rows = T[(T.variant == variant) & (T.boxes == boxes)]
    seed0 = rows[rows.seed.astype(str) == '0']       # the baseline row has seed '-'
    r = (seed0 if len(seed0) else rows).iloc[0]
    return dict(AP=float(r.diag_AP), AP50=float(r.diag_AP50), val_AP=float(r.val_diag_AP), chosen=r.chosen)


def adapter():
    return U.fit_box_adapter(sets()['dis_train']['images'], teeth_clean(), keys('A_train'))[0]


def saved_members(part):
    """the nine members' predictions on 'val' (A_val) or 'test' (A_test) from the saved outputs, fusion order"""
    ks = keys('A_val' if part == 'val' else 'A_test')
    rows = clean_rows()
    ad = adapter()
    out = []
    for c in C.CLASSIFIERS:
        probs = saved_probs(c['name'], part)
        out.append(U.preds_from_teeth([r for k in ks for r in rows.get(k, [])], probs, c['conf_pow'], ad, None, None))
    for d in C.DETECTORS:
        b = saved_boxes(d['name'])
        out.append(U.preds_from_disdet({k: b[k] for k in ks}, teeth_clean()))
    return out


def replay(part='test'):
    """the final fusion rebuilt from the saved member outputs with the app's fusion code"""
    m = saved_members(part)
    m = with_detector_boxes(m, C.FUSION_KINDS, clean_rows(), C.FUSION['donor_iou'])
    return wbf_members(m, C.FUSION['weights'], C.FUSION['iou'])


def score(preds, part='test'):
    """DENTEX / COCO diagnosis AP and AP50 (pycocotools, IoU 0.50:0.95), as the notebook's score_row"""
    ks = keys('A_val' if part == 'val' else 'A_test')
    gt = sets()['dis_train' if part == 'val' else 'dis_val']['images']
    d = U.dentex_eval(gt, preds, ks)['diagnosis']
    return dict(AP=round(d['AP'], 2), AP50=round(d['AP50'], 2), AP75=round(d['AP75'], 2),
                per_class={U.DIS[c - 1]: round(v, 2) for c, v in d['per_class_AP'].items()})
