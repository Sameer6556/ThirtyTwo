"""The app against the notebook's own functions (dentex_utils, train_arch, ultralytics' validator), and against the
saved protocol-A outputs in assets/."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import SAMPLES, needs_assets, needs_dentex, needs_references
import dentex_utils as U
import train_arch as TA
from thirtytwo import config as C, images
from thirtytwo.classifier import KEY, tooth_rows
from thirtytwo.fusion import assign_tooth

SAMPLE_PATHS = [SAMPLES / 'val_14.png', SAMPLES / 'val_28.png']


# ----------------------------------------------------------------------------- preprocessing
@pytest.mark.parametrize('long_side', [C.LONG_SIDE, C.LONG_SIDE_HI])
@pytest.mark.parametrize('path', SAMPLE_PATHS, ids=lambda p: p.stem)
def test_export_bytes_match_resize_long(tmp_path, path, long_side):
    gray = images.decode_upload(path.read_bytes())
    assert np.array_equal(gray, U.read_gray(path))
    jpeg, meta = images.resize_long(gray, long_side)
    ref = U.resize_long(path, tmp_path / 'x.jpg', long_side)
    assert meta == ref
    assert jpeg == (tmp_path / 'x.jpg').read_bytes()
    from ultralytics.utils.patches import imread
    assert np.array_equal(images.decode_like_ultralytics(jpeg), imread(str(tmp_path / 'x.jpg')))


def test_sniff_and_dicom(sample_bytes):
    import io
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    gray = images.decode_upload(sample_bytes)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = SecondaryCaptureImageStorage, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = gray.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = gray.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    data = buf.getvalue()
    assert images.sniff(data) == 'DICOM' and images.sniff(sample_bytes) == 'PNG'
    back = images.decode_upload(data)
    assert back.shape == gray.shape and np.abs(back.astype(int) - gray.astype(int)).max() <= 1   # min-max stretch
    with pytest.raises(images.ImageError):
        images.decode_upload(b'not an image at all')


# ----------------------------------------------------------------------------- crops and the classifier
def _teeth(gray):
    """a fixed set of teeth (the saved clean boxes of val_14 when present, else a grid) for the crop tests"""
    from references import teeth_clean
    try:
        return teeth_clean()['dis_val/val_14']
    except Exception:
        h, w = gray.shape
        return [dict(q=q, n=n, fdi=10 * q + n, conf=0.9,
                     box=[w * (0.1 + 0.05 * n), h * (0.2 + 0.3 * (q > 2)), w * (0.14 + 0.05 * n), h * (0.45 + 0.3 * (q > 2))])
                for q in (1, 2, 3, 4) for n in range(1, 9)]


def test_crops_match_build_crops(tmp_path, sample_bytes):
    gray = images.decode_upload(sample_bytes)
    teeth = _teeth(gray)
    rows, cache = tooth_rows(gray, teeth)
    rec = dict(key=KEY, path=str(SAMPLES / 'val_14.png'), anns=[])
    ref, _ = U.build_crops([rec], {KEY: teeth}, tmp_path, 'upload', C.CROP_H, C.CROP_W, add_gt_only=False)
    assert len(ref) == len(rows) == len(teeth)
    for a, b in zip(rows, ref):
        for f in ('crop_id', 'q', 'n', 'fdi', 'x1', 'y1', 'x2', 'y2', 'det_conf', 'gt_only'):
            assert a[f] == b[f]
        assert np.array_equal(cache[a['crop_path']], cv2.imread(b['crop_path'], cv2.IMREAD_GRAYSCALE))


@needs_assets
def test_classifier_matches_train_arch_on_saved_crops(tmp_path, sample_bytes, model):
    """our in-memory scores == train_arch.predict reading the PNG crops build_crops wrote (the notebook's path)"""
    gray = images.decode_upload(sample_bytes)
    teeth = _teeth(gray)
    rows, cache = tooth_rows(gray, teeth)
    ref, _ = U.build_crops([dict(key=KEY, path=str(SAMPLES / 'val_14.png'), anns=[])], {KEY: teeth}, tmp_path,
                           'upload', C.CROP_H, C.CROP_W, add_gt_only=False)
    clf = model.classifiers[0]
    mine = clf(rows, cache)
    ids, probs, _, _ = TA.predict(clf.core, pd.DataFrame(ref), clf.args, clf.device, False, None, tta=1)
    for c, p in zip(ids, probs):
        assert np.array_equal(np.asarray(mine[int(c)], np.float32), p)


# ----------------------------------------------------------------------------- detectors
@needs_assets
@pytest.mark.parametrize('name', ['plain YOLO', 'round-3 self-trained YOLO26m'])
def test_disease_detector_matches_ultralytics_validator(tmp_path, sample_bytes, model, name):
    """the validator copy == ultralytics' own model.val(batch=1) on the same JPEG, read back as validator_boxes does"""
    from ultralytics import YOLO
    d = next(x for x in C.DETECTORS if x['name'] == name)
    gray = images.decode_upload(sample_bytes)
    jpeg, (_, _, s) = images.resize_long(gray, d['export'])
    mine = model.detectors[name](images.decode_like_ultralytics(jpeg), s)
    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'images' / 'x.jpg').write_bytes(jpeg)
    (tmp_path / 'labels' / 'x.txt').write_text('')
    (tmp_path / 'list.txt').write_text(str(tmp_path / 'images' / 'x.jpg') + '\n')
    yaml = tmp_path / 'd.yaml'
    yaml.write_text(f'path: {tmp_path}\ntrain: {tmp_path / "list.txt"}\nval: {tmp_path / "list.txt"}\nnames:\n'
                    + ''.join(f'  {i}: "{n}"\n' for i, n in enumerate(U.DIS)))
    m = YOLO(str(C.ASSETS / d['weights'])).val(data=str(yaml), split='val', imgsz=d['imgsz'], batch=1, device='cpu',
                                               plots=False, save_json=True, verbose=False, project=str(tmp_path),
                                               name='v', exist_ok=True, **C.DET_EVAL)
    pj = json.loads((tmp_path / 'v' / 'predictions.json').read_text())
    ref = [dict(box=[p['bbox'][0] / s, p['bbox'][1] / s, (p['bbox'][0] + p['bbox'][2]) / s,
                     (p['bbox'][1] + p['bbox'][3]) / s], cls=int(p['category_id']) - 1, conf=float(p['score']))
           for p in pj]
    assert m is not None and len(ref) == len(mine) > 0
    assert mine == ref


@needs_assets
def test_tooth_detector_matches_predict_on_the_file(tmp_path, sample_bytes, model):
    gray = images.decode_upload(sample_bytes)
    jpeg, (_, _, s) = images.resize_long(gray, C.LONG_SIDE)
    (tmp_path / 'x.jpg').write_bytes(jpeg)
    mine = model.tooth(images.decode_like_ultralytics(jpeg), s)
    t = C.TOOTH
    r = model.tooth.model.predict(source=[str(tmp_path / 'x.jpg')], imgsz=t['imgsz'], conf=t['conf'], iou=t['iou'],
                                  agnostic_nms=True, max_det=t['max_det'], rect=False, device='cpu',
                                  verbose=False)[0]
    ref = U.teeth_from_yolo(r.boxes.xyxy.cpu().numpy() / s, r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy())
    assert mine == ref and len(mine) >= C.MIN_TEETH


# ----------------------------------------------------------------------------- fusion, adapter, tooth rule
@needs_references
@needs_dentex
def test_fusion_code_reproduces_the_saved_fusion():
    """the app's fusion code on the saved member outputs gives the reported 48.77 / 69.40 (val 48.69)"""
    import references as R
    assert R.score(R.replay('test'))['AP'] == pytest.approx(48.77, abs=0.005)
    assert R.score(R.replay('test'))['AP50'] == pytest.approx(69.40, abs=0.005)
    assert R.score(R.replay('val'), 'val')['AP'] == pytest.approx(48.69, abs=0.005)


@needs_references
@needs_dentex
def test_adapter_json_rebuilds_and_scores_32_21():
    import references as R
    from build_adapter import fit, table
    adapter, counts = fit()
    shipped = json.loads((C.SETTINGS / 'adapter.json').read_text())['table']
    assert shipped == json.loads(json.dumps(table(adapter, counts)))
    assert C.load_adapter() == adapter
    keys = R.keys('A_test')
    rows = [r for k in keys for r in R.clean_rows().get(k, [])]
    preds = U.preds_from_teeth(rows, R.saved_probs('crop only', 'test'), 0.0, C.load_adapter(), None, None)
    assert R.score(preds)['AP'] == pytest.approx(32.21, abs=0.005)


@needs_references
@needs_dentex
def test_assign_tooth_follows_build_crops(tmp_path):
    """for every annotated disease box of the official 50, assign_tooth picks the tooth build_crops labels"""
    import references as R
    recs = R.sets()['dis_val']['images']
    for rec in recs:
        teeth = R.teeth_clean()[rec['key']]
        ref, _ = U.build_crops([rec], {rec['key']: teeth}, tmp_path, 'x', 8, 8,
                               add_gt_only=False)                    # tiny crops: only the labels matter here
        want = {(i, d) for i, r in enumerate(ref) for d, c in enumerate(U.LABEL_COLS) if r[c]}
        got = set()
        for a in rec['anns']:
            if 'd' in a:
                j = assign_tooth(dict(box=U.xywh_to_xyxy(a['box']), q=a.get('q'), n=a.get('n')), teeth)
                if j is not None:
                    got.add((j, a['d']))
        assert got == want, rec['key']


@needs_assets
def test_quick_mode_is_the_round1_student(sample_bytes, model):
    gray = images.decode_upload(sample_bytes)
    trace = {}
    model.run(gray, 'full', trace)
    quick, teeth, _ = model.run(gray, 'quick')
    assert teeth == trace['teeth']
    j = 2 + [d['name'] for d in C.DETECTORS].index(C.QUICK_DETECTOR)
    assert quick == trace['members'][j]


@needs_assets
def test_predict_output(sample_bytes, model):
    r = model.predict(sample_bytes, 'quick')
    assert r['image'] == dict(width=2728, height=1316) and r['warning'] is None
    assert len(r['teeth']) >= C.MIN_TEETH and r['findings']
    fdis = {t['fdi'] for t in r['teeth']}
    for f in r['findings']:
        assert f['cls'] in {d['key'] for d in C.DISEASES} and 0 < f['score'] <= 1 and len(f['box']) == 4
        assert f['tooth'] is None or f['tooth'] in fdis
    assert [f['score'] for f in r['findings']] == sorted((f['score'] for f in r['findings']), reverse=True)
    square = cv2.resize(images.decode_upload(sample_bytes), (900, 900))
    r = model.predict(square, 'full')
    assert r['warning'] and r['findings'] == []
