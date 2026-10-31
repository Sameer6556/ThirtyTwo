"""The notebook's glue between models, for one X-ray.

make_tooth_preds and make_det_preds are the 'tooth' and 'det' modes of make_preds (cell 24); with_detector_boxes and
wbf_members are copied from cell 24 with the val/test loop dropped (everything they call works per X-ray: WBF groups
by X-ray and disease, transplant_boxes by X-ray). assign_tooth is the rule build_crops uses to put an annotated disease
box on a tooth. The helpers themselves are dentex_utils', unchanged.
"""
import dentex_utils as U

from .classifier import KEY


def make_tooth_preds(rows, probs, conf_pow, adapter):
    """every found tooth -> 4 candidate boxes (one per disease), boxes from the median adapter"""
    return U.preds_from_teeth(rows, probs, conf_pow, adapter, None, None)


def make_det_preds(boxes, teeth):
    """a YOLO's boxes, FDI number taken from the matched tooth"""
    return U.preds_from_disdet({KEY: boxes}, {KEY: teeth})


def with_detector_boxes(members, kinds, rows_by_key, donor_iou):
    """every classifier member ('c...' kind) moved onto the detector members' boxes for its tooth; detectors unchanged.
    The donors are the plain WBF (equal weights) of ALL detector members, also those a setting leaves out."""
    dets = [m for m, k in zip(members, kinds) if k[0] == 'd']
    donors = U.wbf(dets, iou_thr=donor_iou)
    return [U.transplant_boxes(m, donors, rows_by_key) if k[0] == 'c' else m for m, k in zip(members, kinds)]


def wbf_members(members, weights, iou_thr):
    """WBF of the members with a weight above 0"""
    keep = [i for i, x in enumerate(weights) if x > 0]
    return U.wbf([members[i] for i in keep], weights=[weights[i] for i in keep], iou_thr=iou_thr)


def assign_tooth(finding, teeth, min_iou=0.3):
    """index of the tooth a finding sits on, or None. build_crops' rule for an annotated box: the tooth with the same
    tooth number if their IoU is at least 0.1, else the tooth it overlaps most if that IoU is at least min_iou."""
    if not teeth:
        return None
    ious = U.iou_matrix([finding['box']], [t['box'] for t in teeth])[0]
    same = [k for k, t in enumerate(teeth) if t['q'] == finding.get('q') and t['n'] == finding.get('n')]
    if same and ious[same].max() >= 0.1:
        return same[int(ious[same].argmax())]
    if ious.max() >= min_iou:
        return int(ious.argmax())
    return None
