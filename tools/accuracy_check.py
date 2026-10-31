"""Step 2: run the deployed system exactly as the app does (thirtytwo.pipeline on the PNG bytes) on protocol A's
X-rays, score it with pycocotools, and compare every member with the saved outputs in assets/.

    python tools/accuracy_check.py              # runs what is not cached yet, then scores and writes the report
    python tools/accuracy_check.py --score-only # only scores the cached runs

* official 50 (keys/A_test.txt): full mode must give 48.8 AP and 69.4 AP50 (+-0.3), Quick 48.1 AP
* 70 protocol-A validation X-rays (keys/A_val.txt): display thresholds, best F1 at IoU 0.5 per disease and mode
  (never tuned on the official 50) -> thirtytwo/settings/thresholds.json
Report: docs/accuracy-check.md. Runs are cached in data/accuracy/ (one pickle per X-ray and part).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from common import C, ROOT, U, keys, records
import references as R

CACHE = ROOT / 'data' / 'accuracy'
TARGET = dict(AP=48.8, AP50=69.4, quick_AP=48.1, tol=0.3)


def relabel(preds, key):
    return [dict(p, image_key=key) for p in preds]


def run_all(parts):
    from thirtytwo import images
    from thirtytwo.pipeline import ThirtyTwo
    model = None
    recs = records()
    for part in parts:
        ks = keys('A_test' if part == 'test' else 'A_val')
        (CACHE / part).mkdir(parents=True, exist_ok=True)
        todo = [k for k in ks if not (CACHE / part / f'{k.replace("/", "__")}.pkl').exists()]
        for i, k in enumerate(todo):
            if model is None:
                t0 = time.time()
                model = ThirtyTwo(device='cpu')
                print(f'models loaded in {time.time() - t0:.0f}s', flush=True)
            gray = images.decode_upload(Path(recs[k]['path']).read_bytes())
            trace = {}
            t0 = time.perf_counter()
            fused, teeth, warning = model.run(gray, 'full', trace)
            t_full = time.perf_counter() - t0
            t0 = time.perf_counter()
            quick, teeth_q, warning_q = model.run(gray, 'quick')
            t_quick = time.perf_counter() - t0
            out = dict(trace, key=k, fused=fused, quick=quick, teeth=teeth, teeth_quick=teeth_q, warning=warning,
                       warning_quick=warning_q, t_full=t_full, t_quick=t_quick)
            (CACHE / part / f'{k.replace("/", "__")}.pkl').write_bytes(pickle.dumps(out))
            print(f'[{part} {i + 1}/{len(todo)}] {k}: full {t_full:.1f}s, quick {t_quick:.1f}s, {len(teeth)} teeth, '
                  f'{len(fused)} fused boxes{", WARNING " + warning if warning else ""}', flush=True)


def load(part):
    ks = keys('A_test' if part == 'test' else 'A_val')
    return {k: pickle.loads((CACHE / part / f'{k.replace("/", "__")}.pkl').read_bytes()) for k in ks}


# ----------------------------------------------------------------------------- comparisons with the saved outputs
def box_match(a, b):
    """a, b: [dict(box, cls, conf)] of one X-ray -> (identical, IoUs of a's boxes to the best same-class box of b,
    |conf diffs|)"""
    A = np.array([x['box'] + [x['cls'], x['conf']] for x in a], float).reshape(-1, 6)
    B = np.array([x['box'] + [x['cls'], x['conf']] for x in b], float).reshape(-1, 6)
    same = A.shape == B.shape and np.array_equal(A, B)
    ious, dc = [], []
    for x in A:
        m = B[B[:, 4] == x[4]]
        if not len(m):
            ious.append(0.0)
            dc.append(x[5])
            continue
        iou = U.iou_matrix([x[:4]], m[:, :4])[0]
        j = int(np.argmax(iou))
        ious.append(float(iou[j]))
        dc.append(abs(m[j, 5] - x[5]))
    return same, np.array(ious), np.array(dc), len(A) - len(B)


def teeth_diff(runs):
    T = R.teeth_clean()
    same_set, ious, dconf, max_px, identical = 0, [], [], [], 0
    for k, r in runs.items():
        mine = {t['fdi']: t for t in r['teeth']}
        ref = {t['fdi']: t for t in T[k]}
        same_set += set(mine) == set(ref)
        common = sorted(set(mine) & set(ref))
        px = [float(np.max(np.abs(np.array(mine[f]['box']) - np.array(ref[f]['box'])))) for f in common]
        ious += [float(U.iou_matrix([mine[f]['box']], [ref[f]['box']])[0, 0]) for f in common]
        dconf += [abs(mine[f]['conf'] - ref[f]['conf']) for f in common]
        max_px.append(max(px) if px else 0.0)
        identical += set(mine) == set(ref) and (max(px) if px else 0) < 1e-6
    ious = np.array(ious)
    return dict(xrays=len(runs), identical=identical, same_tooth_numbers=same_set, teeth=len(ious),
                median_iou=float(np.median(ious)), min_iou=float(ious.min()),
                max_box_diff_px=float(max(max_px)), max_conf_diff=float(max(dconf)))


def cls_diff(runs, part):
    """our probabilities vs the saved pred_*.csv, tooth by tooth (matched by X-ray and tooth number)"""
    rows = R.clean_rows()
    out = {}
    for j, c in enumerate(C.CLASSIFIERS):
        saved = R.saved_probs(c['name'], part)
        d = []
        for k, r in runs.items():
            ref = {x['fdi']: saved[x['crop_id']] for x in rows.get(k, []) if x['crop_id'] in saved}
            mine = {x['fdi']: r['probs'][j][x['crop_id']] for x in r['rows'] if x['crop_id'] in r['probs'][j]}
            d += [np.abs(mine[f] - ref[f]) for f in set(mine) & set(ref)]
        d = np.array(d)
        out[c['name']] = dict(teeth=len(d), median_abs_diff=float(np.median(d)), p99_abs_diff=float(np.percentile(d, 99)),
                              max_abs_diff=round(float(d.max()), 4))
    return out


def det_diff(runs):
    out = {}
    for d in C.DETECTORS:
        saved = R.saved_boxes(d['name'])
        same, ious, dcs, dn, hi = 0, [], [], [], []
        for k, r in runs.items():
            s, iou, dc, n = box_match(r['boxes'][d['name']], saved[k])
            same += s
            ious.append(iou)
            dcs.append(dc)
            dn.append(n)
            conf = np.array([x['conf'] for x in r['boxes'][d['name']]])
            hi.append(iou[conf >= 0.1])
        ious, dcs, hi = np.concatenate(ious), np.concatenate(dcs), np.concatenate(hi)
        out[d['name']] = dict(identical_xrays=same, xrays=len(runs), boxes=len(ious),
                              box_count_diff=int(np.sum(np.abs(dn))), median_iou=round(float(np.median(ious)), 5),
                              boxes_iou_below_0_9=int((ious < 0.9).sum()),
                              confident_boxes=len(hi), confident_iou_below_0_9=int((hi < 0.9).sum()),
                              median_conf_diff=float(np.median(dcs)), max_conf_diff=round(float(dcs.max()), 4))
    return out


def member_aps(runs, part):
    names = [c['name'] for c in C.CLASSIFIERS] + [d['name'] for d in C.DETECTORS]
    out = {}
    for j, n in enumerate(names):
        preds = [p for k, r in runs.items() for p in relabel(r['members'][j], k)]
        out[n] = R.score(preds, part)
    return out


def thresholds(runs):
    """best F1 at IoU 0.5 per disease on the 70 A_val X-rays (U.f1_threshold), for both modes"""
    gt = R.sets()['dis_train']['images']
    ks = list(runs)
    out, detail = {}, {}
    for mode, field in (('full', 'fused'), ('quick', 'quick')):
        preds = [p for k, r in runs.items() for p in relabel(r[field], k)]
        out[mode], detail[mode] = {}, {}
        for d, dis in enumerate(C.DISEASES):
            t = U.f1_threshold(gt, preds, ks, d)
            out[mode][dis['key']] = round(t['thr'], 5)
            detail[mode][dis['key']] = {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()}
    return out, detail


def write_report(res, path):
    """docs/accuracy-check.md from the scored results"""
    f, q, rep = res['full'], res['quick'], res['report']
    ok = lambda b: 'pass' if b else 'FAIL'
    L = ['# Accuracy check', '',
         'The deployed system (`thirtytwo.pipeline`, exactly as the app runs it, one X-ray at a time on the PNG bytes) '
         'on protocol A of the v15 notebook, scored with pycocotools (COCO AP over IoU 0.50 to 0.95, all boxes) by the '
         "notebook's own `dentex_utils.dentex_eval`. CPU only, fp32. Reproduce with `python tools/accuracy_check.py`.",
         '', '## Official 50 DENTEX validation X-rays', '',
         '| | AP | AP50 | AP75 | report AP | report AP50 | check |', '|---|---|---|---|---|---|---|',
         f"| Full system | {f['AP']:.2f} | {f['AP50']:.2f} | {f['AP75']:.2f} | {rep['full']['AP']:.2f} | "
         f"{rep['full']['AP50']:.2f} | {ok(res['pass']['full_AP'] and res['pass']['full_AP50'])} (target 48.8 / 69.4 +-0.3) |",
         f"| Quick mode | {q['AP']:.2f} | {q['AP50']:.2f} | {q['AP75']:.2f} | {rep['quick']['AP']:.2f} | "
         f"{rep['quick']['AP50']:.2f} | {ok(res['pass']['quick_AP'])} (target 48.1 +-0.3) |",
         f"| Saved Kaggle outputs, fused by the app's fusion code | {res['replay_of_saved_outputs']['AP']:.2f} | "
         f"{res['replay_of_saved_outputs']['AP50']:.2f} | {res['replay_of_saved_outputs']['AP75']:.2f} | | | |", '',
         'Per disease (full system, AP): ' + ', '.join(f'{k} {v:.1f}' for k, v in f['per_class'].items()) + '.', '']
    if 'val' in res:
        v = res['val']
        L += [f"On the 70 protocol-A validation X-rays: full {v['full']['AP']:.2f} AP (report {v['report_val_AP']['full']:.2f}),"
              f" Quick {v['quick']['AP']:.2f} AP (report {v['report_val_AP']['quick']:.2f}).", '']
    L += ['## Every member against the saved outputs', '', '### AP of each member alone (official 50)', '',
          '| member | AP | AP50 | report AP | report AP50 | difference |', '|---|---|---|---|---|---|']
    for n, m in res['members_test'].items():
        L.append(f"| {n} | {m['ours']['AP']:.2f} | {m['ours']['AP50']:.2f} | {m['report']['AP']:.2f} | "
                 f"{m['report']['AP50']:.2f} | {m['ours']['AP'] - m['report']['AP']:+.2f} |")
    for part in ('test', 'val'):
        if f'diff_teeth_{part}' not in res:
            continue
        name = 'official 50' if part == 'test' else '70 A_val'
        t = res[f'diff_teeth_{part}']
        L += ['', f'### Tooth detector vs teeth_clean.pkl ({name})', '',
              f"{t['same_tooth_numbers']} of {t['xrays']} X-rays have the same tooth numbers ({t['identical']} bit for "
              f"bit). Over {t['teeth']} matched teeth: lowest IoU {t['min_iou']:.6f}, largest box edge difference "
              f"{t['max_box_diff_px']:.1e} px, largest confidence difference {t['max_conf_diff']:.1e} (GPU vs CPU "
              f"floating point).", '',
              f'### Classifiers vs cls/*_E/pred_*.csv ({name})', '',
              '| classifier | teeth | median abs diff | 99th percentile | max |', '|---|---|---|---|---|']
        for n, c in res[f'diff_classifiers_{part}'].items():
            L.append(f"| {n} | {c['teeth']} | {c['median_abs_diff']:.1e} | {c['p99_abs_diff']:.1e} | {c['max_abs_diff']} |")
        L += ['', f'### YOLOs vs disbox_*_validator.pkl ({name})', '',
              '| YOLO | identical X-rays | boxes | count diff | median IoU | boxes IoU < 0.9 | of them with conf >= 0.1 | '
              'median conf diff | max conf diff |', '|---|---|---|---|---|---|---|---|---|']
        for n, d in res[f'diff_detectors_{part}'].items():
            L.append(f"| {n} | {d['identical_xrays']} / {d['xrays']} | {d['boxes']} | {d['box_count_diff']} | "
                     f"{d['median_iou']} | {d['boxes_iou_below_0_9']} | {d['confident_iou_below_0_9']} of "
                     f"{d['confident_boxes']} | {d['median_conf_diff']:.1e} | {d['max_conf_diff']} |")
    if 'thresholds' in res:
        L += ['', '## Display thresholds (70 protocol-A validation X-rays, best F1 at IoU 0.5)', '',
              '| disease | full | F1 | quick | F1 |', '|---|---|---|---|---|']
        for d in C.DISEASES:
            a, b = res['thresholds_detail']['full'][d['key']], res['thresholds_detail']['quick'][d['key']]
            L.append(f"| {d['name']} | {a['thr']:.4f} | {a['f1']:.3f} | {b['thr']:.4f} | {b['f1']:.3f} |")
    t = res['time_s_test']
    L += ['', '## Time per X-ray on this machine (4 CPU cores)', '',
          f"Full {t['full_mean']} s on average (slowest {t['full_max']} s), Quick {t['quick_mean']} s "
          f"(slowest {t['quick_max']} s), models already loaded.", '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--parts', default='test,val')
    a = ap.parse_args()
    parts = a.parts.split(',')
    if not a.score_only:
        run_all(parts)
    res = {}
    test = load('test')
    fused = [p for k, r in test.items() for p in relabel(r['fused'], k)]
    quick = [p for k, r in test.items() for p in relabel(r['quick'], k)]
    res['full'] = R.score(fused)
    res['quick'] = R.score(quick)
    res['report'] = dict(full=R.report_row('final fusion'), quick=R.report_row('round-1 self-trained YOLO11m'))
    res['replay_of_saved_outputs'] = R.score(R.replay('test'))
    res['pass'] = dict(full_AP=abs(res['full']['AP'] - TARGET['AP']) <= TARGET['tol'],
                       full_AP50=abs(res['full']['AP50'] - TARGET['AP50']) <= TARGET['tol'],
                       quick_AP=abs(res['quick']['AP'] - TARGET['quick_AP']) <= TARGET['tol'])
    res['members_test'] = {n: dict(ours=v, report=R.report_row(n)) for n, v in member_aps(test, 'test').items()}
    res['diff_teeth_test'] = teeth_diff(test)
    res['diff_classifiers_test'] = cls_diff(test, 'test')
    res['diff_detectors_test'] = det_diff(test)
    res['warnings_test'] = {k: r['warning'] for k, r in test.items() if r['warning']}
    res['time_s_test'] = dict(full_mean=round(float(np.mean([r['t_full'] for r in test.values()])), 2),
                              quick_mean=round(float(np.mean([r['t_quick'] for r in test.values()])), 2),
                              full_max=round(float(np.max([r['t_full'] for r in test.values()])), 2),
                              quick_max=round(float(np.max([r['t_quick'] for r in test.values()])), 2))
    if 'val' in parts:
        val = load('val')
        res['val'] = dict(full=R.score([p for k, r in val.items() for p in relabel(r['fused'], k)], 'val'),
                          quick=R.score([p for k, r in val.items() for p in relabel(r['quick'], k)], 'val'),
                          report_val_AP=dict(full=res['report']['full']['val_AP'],
                                             quick=res['report']['quick']['val_AP']))
        res['diff_teeth_val'] = teeth_diff(val)
        res['diff_classifiers_val'] = cls_diff(val, 'val')
        res['diff_detectors_val'] = det_diff(val)
        thr, detail = thresholds(val)
        res['thresholds'], res['thresholds_detail'] = thr, detail
        doc = dict(description='Display thresholds: for each disease, the score with the best F1 at IoU 0.5 '
                               '(dentex_utils.f1_threshold) on the 70 protocol-A validation X-rays (keys/A_val.txt), '
                               'for each mode. Never tuned on the official 50.', **thr)
        (C.SETTINGS / 'thresholds.json').write_text(json.dumps(doc, indent=1) + '\n')
    (CACHE / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    write_report(res, ROOT / 'docs' / 'accuracy-check.md')
    print(json.dumps({k: res[k] for k in ('full', 'quick', 'report', 'replay_of_saved_outputs', 'pass')}, indent=1,
                     default=float))
    return res


if __name__ == '__main__':
    sys.exit(0 if all(main()['pass'].values()) else 1)
