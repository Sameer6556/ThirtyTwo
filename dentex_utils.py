# dentex_utils.py
# Helpers for the DENTEX symmetry project: data discovery, YOLO conversion,
# tooth crops with contralateral twins, prediction building and DENTEX-style COCO evaluation.
import os, json, math, random, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

DIS = ['Caries', 'Deep Caries', 'Periapical Lesion', 'Impacted']   # our fixed order
LABEL_COLS = ['y_caries', 'y_deep', 'y_peri', 'y_impacted']
IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
# Order used in the HierarchicalDet/DENTEX json files, only used if categories_3 is missing.
DEFAULT_DIS_IDS = {0: 'Impacted', 1: 'Caries', 2: 'Periapical Lesion', 3: 'Deep Caries'}
TWIN_Q = {1: 2, 2: 1, 3: 4, 4: 3}


# ----------------------------------------------------------------------------- labels
_TR = str.maketrans({'ç': 'c', 'Ç': 'c', 'ü': 'u', 'Ü': 'u', 'ö': 'o', 'Ö': 'o', 'ğ': 'g', 'Ğ': 'g',
                     'ş': 's', 'Ş': 's', 'ı': 'i', 'İ': 'i'})


def norm_dis(name):
    """English training names and the Turkish names used in the official test labels
    (cürük = caries, derin cürük = deep caries, gömülü = impacted, periapikal lezyon = periapical lesion)."""
    n = str(name).translate(_TR).lower().strip()
    if 'impact' in n or 'gomulu' in n:
        return 'Impacted'
    if 'periap' in n or 'lezyon' in n or 'lesion' in n:
        return 'Periapical Lesion'
    if 'deep' in n or 'derin' in n:
        return 'Deep Caries'
    if 'carie' in n or 'curuk' in n:
        return 'Caries'
    return None


def fdi_to_cls(q, n):
    return (int(q) - 1) * 8 + (int(n) - 1)


def cls_to_fdi(c):
    c = int(c)
    return c // 8 + 1, c % 8 + 1


def twin_of(q, n):
    return TWIN_Q[int(q)], int(n)


# ----------------------------------------------------------------------------- download / unzip
def safe_extract(zpath, dest, skip=('unlabel',), keep=None):
    """Extract a zip but skip members whose path contains any of `skip` (case-insensitive), and, if `keep` is
    given, members for which keep(name) is False."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            name = m.filename
            if any(s in name.lower() for s in skip) or (keep is not None and not keep(name)):
                continue
            target = (dest / name).resolve()
            if not str(target).startswith(str(root)):
                continue  # path traversal guard
            z.extract(m, dest)
            n += 1
    return n


# ----------------------------------------------------------------------------- discovery
def index_image_dirs(root):
    dirs = {}
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            dirs.setdefault(p.parent, set()).add(p.name)
    return dirs


def _common_prefix_len(a, b):
    a, b = Path(a).resolve().parts, Path(b).resolve().parts
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def resolve_image_dir(json_path, file_names, img_dirs):
    """Pick the image folder that contains most of the json's file names; ties go to the folder
    closest to the json file. Returns (dir, hit_rate)."""
    names = {Path(f).name for f in file_names}
    best, best_score = None, (-1.0, -1)
    for d, s in img_dirs.items():
        hit = len(names & s) / max(1, len(names))
        score = (round(hit, 4), _common_prefix_len(d, json_path))
        if score > best_score:
            best, best_score = d, score
    return best, best_score[0]


def classify_json(js):
    if not isinstance(js, dict) or 'images' not in js or 'annotations' not in js:
        return None
    anns = js.get('annotations') or []
    keys = set()
    for a in anns[:200]:
        keys |= set(a.keys())
    if 'category_id_3' in keys:
        return 'disease'
    if 'category_id_2' in keys:
        return 'enum'
    if 'category_id_1' in keys or 'category_id' in keys:
        return 'quad'
    return None


def _cat_map(js, key):
    return {int(c['id']): c.get('name', c['id']) for c in (js.get(key) or [])}


def _as_int(v, default):
    try:
        return int(str(v).strip())
    except Exception:
        return default


def parse_json(json_path, img_dirs, tag):
    js = json.loads(Path(json_path).read_text(encoding='utf-8'))
    kind = classify_json(js)
    if kind is None:
        return None
    imgs = js['images']
    d, hit = resolve_image_dir(json_path, [im['file_name'] for im in imgs], img_dirs)
    c1, c2, c3 = _cat_map(js, 'categories_1'), _cat_map(js, 'categories_2'), _cat_map(js, 'categories_3')
    warn = []
    if kind == 'disease' and not c3:
        warn.append('categories_3 missing, using default disease id order ' + str(DEFAULT_DIS_IDS))
        c3 = DEFAULT_DIS_IDS
    by_id = {}
    for im in imgs:
        stem = Path(im['file_name']).stem
        by_id[im['id']] = dict(key=f'{tag}/{stem}', file=im['file_name'],
                               path=str(Path(d) / Path(im['file_name']).name) if d else None,
                               w=int(im.get('width', 0)), h=int(im.get('height', 0)), anns=[])
    for a in js['annotations']:
        rec = by_id.get(a['image_id'])
        if rec is None:
            continue
        x, y, w, h = [float(v) for v in a['bbox']]
        if w <= 1 or h <= 1:
            continue
        ann = dict(box=[x, y, w, h])
        if 'category_id_1' in a:
            cid = int(a['category_id_1'])
            ann['q'] = _as_int(c1.get(cid, cid + 1), cid + 1)
        elif 'category_id' in a:
            cid = int(a['category_id'])
            ann['q'] = _as_int(c1.get(cid, cid + 1), cid + 1)
        if 'category_id_2' in a:
            cid = int(a['category_id_2'])
            ann['n'] = _as_int(c2.get(cid, cid + 1), cid + 1)
        if 'category_id_3' in a:
            cid = int(a['category_id_3'])
            nm = norm_dis(c3.get(cid, ''))
            if nm is None:
                warn.append(f'unknown disease category id {cid}')
                continue
            ann['d'] = DIS.index(nm)
        rec['anns'].append(ann)
    return dict(kind=kind, json=str(json_path), image_dir=str(d), hit=hit, images=list(by_id.values()),
                cats=dict(c1=c1, c2=c2, c3=c3), warn=warn)


def parse_labelme_label(label):
    """'1-çürük-33' -> ('çürük', 33); also accepts 'çürük-33'."""
    parts = [p.strip() for p in str(label).split('-')]
    fdi = int(parts[-1])
    name = '-'.join(parts[1:-1]) if len(parts) >= 3 else parts[0]
    return name, fdi


def parse_labelme_dir(label_dir, files, img_dirs, tag):
    """Official DENTEX test labels: one LabelMe json per X-ray, shape label '<n>-<diagnosis in Turkish>-<FDI>'."""
    import base64
    items, unknown, names_seen = [], {}, {}
    for jp in sorted(files):
        js = json.loads(Path(jp).read_text(encoding='utf-8'))
        fname = Path(js.get('imagePath') or (Path(jp).stem + '.png')).name
        anns = []
        for sh in js.get('shapes', []):
            try:
                name, fdi = parse_labelme_label(sh['label'])
            except Exception:
                unknown[str(sh.get('label'))] = unknown.get(str(sh.get('label')), 0) + 1
                continue
            names_seen[name] = names_seen.get(name, 0) + 1
            nm = norm_dis(name)
            q, n = fdi // 10, fdi % 10
            if nm is None or not (1 <= q <= 4 and 1 <= n <= 8):
                unknown[str(sh['label'])] = unknown.get(str(sh['label']), 0) + 1
                continue
            xs = [float(p[0]) for p in sh['points']]
            ys = [float(p[1]) for p in sh['points']]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            anns.append(dict(box=[x1, y1, x2 - x1, y2 - y1], q=q, n=n, d=DIS.index(nm)))
        items.append(dict(fname=fname, w=int(js.get('imageWidth') or 0), h=int(js.get('imageHeight') or 0),
                          anns=anns, data=js.get('imageData'), json=str(jp)))
    d, hit = resolve_image_dir(label_dir, [it['fname'] for it in items], img_dirs)
    if hit < 0.5:   # images only embedded in the json files: decode them once
        d = Path(label_dir).parent / '_decoded_images'
        d.mkdir(exist_ok=True)
        for it in items:
            if it['data'] and not (d / it['fname']).exists():
                (d / it['fname']).write_bytes(base64.b64decode(it['data']))
        hit = 1.0
    images = [dict(key=f'{tag}/{Path(it["fname"]).stem}', file=it['fname'], path=str(Path(d) / it['fname']),
                   w=it['w'], h=it['h'], anns=it['anns']) for it in items]
    warn = [f'label names found: {names_seen}']
    if unknown:
        warn.append(f'UNRECOGNISED labels skipped (tell Claude): {unknown}')
    return dict(kind='disease', json=str(label_dir), image_dir=str(d), hit=hit, images=images,
                cats=dict(c1={}, c2={}, c3={'labelme': 'Turkish names mapped by norm_dis'}), warn=warn)


def _disease_split(json_path, n_images):
    p = str(json_path).lower()
    if 'valid' in p:
        return 'dis_val'
    if 'test' in p:
        return 'dis_test'
    if 'train' in p:
        return 'dis_train'
    sizes = {'dis_train': 705, 'dis_val': 50, 'dis_test': 250}
    return min(sizes, key=lambda k: abs(sizes[k] - n_images))


def discover(raw_dir, verbose=True):
    """Find every DENTEX json under raw_dir and return {'quad','enum','dis_train','dis_val','dis_test'}."""
    raw_dir = Path(raw_dir)
    img_dirs = index_image_dirs(raw_dir)
    out, seen, labelme = {}, set(), {}
    for jp in sorted(raw_dir.rglob('*.json')):
        try:
            js = json.loads(jp.read_text(encoding='utf-8'))
        except Exception:
            continue
        kind = classify_json(js)
        if kind is None:
            if isinstance(js, dict) and 'shapes' in js:
                labelme.setdefault(jp.parent, []).append(jp)
            continue
        sig = (kind, len(js['images']), len(js['annotations']))
        if sig in seen:          # same file copied twice
            continue
        seen.add(sig)
        if kind == 'disease':
            split = _disease_split(jp, len(js['images']))
        else:
            split = kind
        rec = parse_json(jp, img_dirs, tag=split)
        if rec is None:
            continue
        if split in out:          # keep the bigger one if two candidates map to the same split
            if len(rec['images']) <= len(out[split]['images']):
                continue
        out[split] = rec
        if verbose:
            n_ann = sum(len(i['anns']) for i in rec['images'])
            print(f'[{split}] {jp}  images={len(rec["images"])} anns={n_ann} '
                  f'image_dir={rec["image_dir"]} hit={rec["hit"]:.2f}')
            for w in rec['warn'][:5]:
                print('   WARN', w)
    for d, files in labelme.items():   # official test labels come as one LabelMe file per X-ray
        if len(files) < 10:
            continue
        split = 'dis_test' if 'test' in str(d).lower() else _disease_split(d, len(files))
        if split in out and len(out[split]['images']) >= len(files):
            continue
        rec = parse_labelme_dir(d, files, img_dirs, tag=split)
        out[split] = rec
        if verbose:
            n_ann = sum(len(i['anns']) for i in rec['images'])
            print(f'[{split}] LabelMe {d}  images={len(rec["images"])} anns={n_ann} '
                  f'image_dir={rec["image_dir"]} hit={rec["hit"]:.2f}')
            for w in rec['warn']:
                print('   ', w)
    return out


# ----------------------------------------------------------------------------- geometry
def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


# ----------------------------------------------------------------------------- images
def read_gray(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im


def resize_long(path, dst, long_side=1280, quality=95):
    """Save a grayscale JPEG copy with the long side capped at long_side. Returns (W, H, scale)."""
    import cv2
    im = read_gray(path)
    h, w = im.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1.0:
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return w, h, s


# ----------------------------------------------------------------------------- YOLO export
def export_yolo(records, root, cls_fn, long_side=1280):
    """Write every record once as root/images/all/<key>.jpg (long side <= long_side) and
    root/labels/all/<key>.txt. Splits are plain text lists of image paths (see write_list), so
    the 80/20 seeds of protocol B reuse the same files. Returns {key: (W, H, scale, jpg_path)}."""
    root = Path(root)
    (root / 'images' / 'all').mkdir(parents=True, exist_ok=True)
    (root / 'labels' / 'all').mkdir(parents=True, exist_ok=True)
    meta = {}
    for r in records:
        safe = r['key'].replace('/', '__')
        jpg = root / 'images' / 'all' / f'{safe}.jpg'
        if jpg.exists() and r.get('w') and r.get('h'):
            w, h = r['w'], r['h']
            s = min(1.0, long_side / max(w, h))
        else:
            w, h, s = resize_long(r['path'], jpg, long_side)
        r['w'], r['h'], r['scale'] = w, h, s
        meta[r['key']] = (w, h, s, str(jpg))
        lines = []
        for a in r['anns']:
            c = cls_fn(a)
            if c is None:
                continue
            x, y, bw, bh = a['box']
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f'{c} {min(max(cx, 0), 1):.6f} {min(max(cy, 0), 1):.6f} '
                         f'{min(bw / w, 1):.6f} {min(bh / h, 1):.6f}')
        (root / 'labels' / 'all' / f'{safe}.txt').write_text('\n'.join(lines))
    return meta


def write_list(root, name, keys, meta):
    p = Path(root) / f'{name}.txt'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(meta[k][3] for k in keys) + '\n')
    return str(p)


def write_yolo_yaml(root, names, train_txt, val_txt, yaml_name='data.yaml'):
    root = Path(root)
    lines = [f'path: {root}', f'train: {train_txt}', f'val: {val_txt}', 'names:']
    for i, n in enumerate(names):
        lines.append(f'  {i}: "{n}"')
    p = root / yaml_name
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


def tooth_cls(a):
    if 'q' in a and 'n' in a and 1 <= a['q'] <= 4 and 1 <= a['n'] <= 8:
        return fdi_to_cls(a['q'], a['n'])
    return None


def disease_cls(a):
    return a.get('d')


TOOTH_NAMES = [str(10 * (c // 8 + 1) + (c % 8 + 1)) for c in range(32)]


# ----------------------------------------------------------------------------- detector outputs
def teeth_from_yolo(xyxy, cls, conf):
    """Keep the most confident box per FDI number. xyxy in original pixels."""
    best = {}
    for b, c, s in zip(np.asarray(xyxy), np.asarray(cls).astype(int), np.asarray(conf)):
        if c not in best or s > best[c][1]:
            best[c] = (b, float(s))
    out = []
    for c, (b, s) in best.items():
        q, n = cls_to_fdi(c)
        out.append(dict(q=q, n=n, fdi=10 * q + n, box=[float(v) for v in b], conf=s))
    return out


def assign_tooth(box, teeth, min_iou=0.3):
    """Best matching tooth for a box (IoU, else nearest centre)."""
    if not teeth:
        return None, 0.0
    ious = iou_matrix([box], [t['box'] for t in teeth])[0]
    j = int(np.argmax(ious))
    if ious[j] >= min_iou:
        return teeth[j], float(ious[j])
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    d = [((t['box'][0] + t['box'][2]) / 2 - cx) ** 2 + ((t['box'][1] + t['box'][3]) / 2 - cy) ** 2 for t in teeth]
    j = int(np.argmin(d))
    return teeth[j], float(ious[j])


# ----------------------------------------------------------------------------- crops
def crop_tooth(img, box, q, H, W, mx=0.15, my_crown=0.08, my_apex=0.20):
    """Letterboxed crop with extra margin on the root-apex side (up for upper jaw, down for lower)."""
    import cv2
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 2), max(y2 - y1, 2)
    upper = int(q) in (1, 2)
    top = my_apex if upper else my_crown
    bot = my_crown if upper else my_apex
    X1 = int(max(0, math.floor(x1 - mx * w)))
    X2 = int(min(img.shape[1], math.ceil(x2 + mx * w)))
    Y1 = int(max(0, math.floor(y1 - top * h)))
    Y2 = int(min(img.shape[0], math.ceil(y2 + bot * h)))
    c = img[Y1:Y2, X1:X2]
    if c.size == 0:
        return np.zeros((H, W), np.uint8)
    ch, cw = c.shape[:2]
    s = min(H / ch, W / cw)
    nh, nw = max(1, int(round(ch * s))), max(1, int(round(cw * s)))
    c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((H, W), np.uint8)
    oy, ox = (H - nh) // 2, (W - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = c
    return out


def build_crops(records, teeth_by_key, out_dir, split, H, W, add_gt_only, start_id=0, min_iou=0.3):
    """One crop per detected tooth (plus GT-only teeth if add_gt_only). Returns (rows, next_id).
    Each row carries multi-hot disease labels from overlapping GT boxes and the crop id of its
    contralateral twin (same tooth number, mirrored quadrant) or -1."""
    import cv2
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cid = [], start_id
    for r in records:
        teeth = [dict(t) for t in teeth_by_key.get(r['key'], [])]
        for t in teeth:
            t['labels'] = [0, 0, 0, 0]
            t['gt_only'] = 0
        gts = [a for a in r['anns'] if 'd' in a]
        extra = []
        if gts:
            gboxes = [xywh_to_xyxy(a['box']) for a in gts]
            ious = iou_matrix(gboxes, [t['box'] for t in teeth]) if teeth else np.zeros((len(gts), 0))
            for gi, a in enumerate(gts):
                j = -1
                if teeth:
                    same = [k for k, t in enumerate(teeth)
                            if 'q' in a and 'n' in a and t['q'] == a['q'] and t['n'] == a['n']]
                    if same and ious[gi, same].max() >= 0.1:
                        j = same[int(np.argmax(ious[gi, same]))]
                    elif ious[gi].max() >= min_iou:
                        j = int(np.argmax(ious[gi]))
                if j >= 0:
                    teeth[j]['labels'][a['d']] = 1
                elif add_gt_only and 'q' in a and 'n' in a:
                    lab = [0, 0, 0, 0]
                    lab[a['d']] = 1
                    # merge with an existing GT-only tooth that covers the same tooth
                    merged = False
                    for e in extra:
                        if e['q'] == a['q'] and e['n'] == a['n'] and \
                                iou_matrix([e['box']], [gboxes[gi]])[0, 0] > 0.5:
                            e['labels'][a['d']] = 1
                            merged = True
                    if not merged:
                        extra.append(dict(q=a['q'], n=a['n'], fdi=10 * a['q'] + a['n'], box=gboxes[gi],
                                          conf=1.0, labels=lab, gt_only=1))
        allt = teeth + extra
        if not allt:
            continue
        img = read_gray(r['path'])
        safe = r['key'].replace('/', '__')
        ids_by_fdi = {}
        for t in allt:
            t['crop_id'] = cid
            cid += 1
            crop = crop_tooth(img, t['box'], t['q'], H, W)
            p = out_dir / f'{safe}_{t["fdi"]}_{t["crop_id"]}.png'
            cv2.imwrite(str(p), crop)
            t['crop_path'] = str(p)
            if not t['gt_only']:
                ids_by_fdi[t['fdi']] = t['crop_id']
        for t in allt:
            tq, tn = twin_of(t['q'], t['n'])
            twin = ids_by_fdi.get(10 * tq + tn, -1)
            x1, y1, x2, y2 = t['box']
            rows.append(dict(crop_id=t['crop_id'], image_key=r['key'], split=split, q=t['q'], n=t['n'],
                             fdi=t['fdi'], x1=x1, y1=y1, x2=x2, y2=y2, det_conf=t['conf'],
                             gt_only=t['gt_only'], **{c: t['labels'][i] for i, c in enumerate(LABEL_COLS)},
                             crop_path=t['crop_path'], twin_crop_id=twin))
    return rows, cid


# ----------------------------------------------------------------------------- predictions
def match_gt_to_teeth(record, teeth, min_iou=0.3):
    """(gt index, disease, gt box xyxy, tooth index) for every GT disease box whose tooth was found:
    same tooth number first (as when labelling the crops), IoU >= min_iou so wrong-tooth pairs are dropped"""
    out = []
    if not teeth:
        return out
    for gi, a in enumerate(record['anns']):
        if 'd' not in a or 'q' not in a or 'n' not in a:
            continue
        g = xywh_to_xyxy(a['box'])
        same = [j for j, t in enumerate(teeth) if t['q'] == a['q'] and t['n'] == a['n']]
        if not same:
            continue
        ious = iou_matrix([g], [teeth[j]['box'] for j in same])[0]
        k = int(np.argmax(ious))
        if ious[k] >= min_iou:
            out.append((gi, int(a['d']), g, same[k]))
    return out


def fit_box_adapter(records, teeth_by_key, keys, min_count=20):
    """Box adapter, learned on TRAINING images only. The tooth finder learned the box style of the
    enumeration set, while the disease boxes were drawn separately. For each (disease, upper/lower jaw)
    we take the median offset of every box edge from the found tooth box to the annotated disease box,
    in units of tooth width/height. Returns ({(d, upper): [dx1, dy1, dx2, dy2]}, counts)."""
    keys = set(keys)
    offs = {}
    for r in records:
        if r['key'] not in keys:
            continue
        teeth = teeth_by_key.get(r['key'], [])
        for _, d, g, j in match_gt_to_teeth(r, teeth):
            t = teeth[j]['box']
            w, h = max(t[2] - t[0], 1.0), max(t[3] - t[1], 1.0)
            offs.setdefault((d, int(teeth[j]['q']) in (1, 2)), []).append(
                [(g[0] - t[0]) / w, (g[1] - t[1]) / h, (g[2] - t[2]) / w, (g[3] - t[3]) / h])
    adapter = {k: np.median(np.array(v), 0).tolist() for k, v in offs.items() if len(v) >= min_count}
    return adapter, {k: len(v) for k, v in offs.items()}


def adapt_box(box, d, q, adapter):
    if not adapter:
        return list(box)
    m = adapter.get((int(d), int(q) in (1, 2)))
    if m is None:
        return list(box)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [x1 + m[0] * w, y1 + m[1] * h, x2 + m[2] * w, y2 + m[3] * h]


def box_iou_stats(records, teeth_by_key, keys, adapter=None):
    """IoU between found tooth boxes and the annotated disease boxes they match (raw or adapted)"""
    keys = set(keys)
    vals = []
    for r in records:
        if r['key'] not in keys:
            continue
        teeth = teeth_by_key.get(r['key'], [])
        for _, d, g, j in match_gt_to_teeth(r, teeth, min_iou=0.1):
            b = adapt_box(teeth[j]['box'], d, teeth[j]['q'], adapter)
            vals.append(float(iou_matrix([g], [b])[0, 0]))
    v = np.array(vals) if vals else np.zeros(1)
    return dict(pairs=len(vals), median_iou=round(float(np.median(v)), 3),
                below_050=round(float((v < 0.5).mean()), 3), below_075=round(float((v < 0.75).mean()), 3))


def oracle_boxes(records, teeth_by_key, keys):
    """ANALYSIS ONLY: for each found tooth and disease, the annotated box it matches (uses test labels)"""
    keys = set(keys)
    out = {}
    for r in records:
        if r['key'] not in keys:
            continue
        teeth = teeth_by_key.get(r['key'], [])
        for _, d, g, j in match_gt_to_teeth(r, teeth, min_iou=0.1):
            out[(r['key'], int(teeth[j]['fdi']), d)] = list(g)
    return out


def _rows_as_teeth(rows):
    return [dict(q=int(r['q']), n=int(r['n']), box=[r['x1'], r['y1'], r['x2'], r['y2']], crop_id=int(r['crop_id']))
            for r in rows]


def box_targets(records, man, min_iou=0.3):
    """Training targets for the learned box head: for every found tooth and every disease it has, the offset of
    each edge of the annotated disease box from the tooth box, in tooth widths/heights (original image frame)."""
    rows_by_key = {}
    for r in man[man.gt_only == 0].to_dict('records'):
        rows_by_key.setdefault(r['image_key'], []).append(r)
    out = {}
    for rec in records:
        teeth = _rows_as_teeth(rows_by_key.get(rec['key'], []))
        for _, d, g, j in match_gt_to_teeth(rec, teeth, min_iou):
            t = teeth[j]['box']
            w, h = max(t[2] - t[0], 1.0), max(t[3] - t[1], 1.0)
            row = out.setdefault(teeth[j]['crop_id'], {'crop_id': teeth[j]['crop_id']})
            iou = float(iou_matrix([g], [t])[0, 0])
            if row.get(f'iou{d}', -1) >= iou:   # two boxes of one disease on one tooth: keep the better match
                continue
            row.update({f'iou{d}': iou, f'm{d}': 1.0, f't{d}_x1': (g[0] - t[0]) / w, f't{d}_y1': (g[1] - t[1]) / h,
                        f't{d}_x2': (g[2] - t[2]) / w, f't{d}_y2': (g[3] - t[3]) / h})
    cols = ['crop_id'] + [f'm{d}' for d in range(4)] + [f't{d}_{k}' for d in range(4) for k in ('x1', 'y1', 'x2', 'y2')]
    df = pd.DataFrame(list(out.values()))
    for c in cols:
        if c not in df:
            df[c] = 0.0
    return df[cols].fillna(0.0)


def refined_box(r, off):
    w, h = r['x2'] - r['x1'], r['y2'] - r['y1']
    return [r['x1'] + off[0] * w, r['y1'] + off[1] * h, r['x2'] + off[2] * w, r['y2'] + off[3] * h]


def refined_iou_stats(records, rows_by_key, keys, refine):
    """IoU between the learned per-disease boxes and the annotated disease boxes they match"""
    keys = set(keys)
    vals = []
    for rec in records:
        if rec['key'] not in keys:
            continue
        rows = rows_by_key.get(rec['key'], [])
        teeth = _rows_as_teeth(rows)
        for _, d, g, j in match_gt_to_teeth(rec, teeth, min_iou=0.1):
            off = refine.get(teeth[j]['crop_id'])
            b = refined_box(rows[j], off[d]) if off is not None else teeth[j]['box']
            vals.append(float(iou_matrix([g], [b])[0, 0]))
    v = np.array(vals) if vals else np.zeros(1)
    return dict(pairs=len(vals), median_iou=round(float(np.median(v)), 3),
                below_050=round(float((v < 0.5).mean()), 3), below_075=round(float((v < 0.75).mean()), 3))


def preds_from_teeth(tooth_rows, probs_by_crop, conf_pow=0.5, adapter=None, oracle=None, refine=None):
    """Every detected tooth becomes 4 candidate boxes (one per disease). refine = learned per-disease box offsets
    from the classifier; adapter = median offsets; oracle (analysis only) swaps in the matched annotated box."""
    out = []
    for r in tooth_rows:
        if r['gt_only']:
            continue
        p = probs_by_crop.get(r['crop_id'])
        if p is None:
            continue
        w = max(r['det_conf'], 1e-6) ** conf_pow
        base = [r['x1'], r['y1'], r['x2'], r['y2']]
        off = refine.get(r['crop_id']) if refine is not None else None
        for d in range(4):
            box = refined_box(r, off[d]) if off is not None else adapt_box(base, d, r['q'], adapter)
            if oracle is not None:
                box = oracle.get((r['image_key'], int(r['fdi']), d), box)
            out.append(dict(image_key=r['image_key'], box=box, q=r['q'], n=r['n'], d=d,
                            score=float(p[d]) * w, src='tooth', cid=int(r['crop_id'])))
    return out


def _iou_pairs(A, B):
    """IoU of A[i] with B[i] (two aligned (n, 4) arrays of x1, y1, x2, y2)"""
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    iw = np.clip(np.minimum(A[:, 2], B[:, 2]) - np.maximum(A[:, 0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(A[:, 3], B[:, 3]) - np.maximum(A[:, 1], B[:, 1]), 0, None)
    inter = iw * ih
    ua = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def transplant_boxes(cls_preds, donors, rows_by_key, min_cont=0.5, min_iou=0.5):
    """Our tooth-level predictions moved onto the detectors' box for the same tooth: the detectors place the box,
    our classifier scores it (detector boxes + classifier scores, as in DCR, Cheng et al. 2018). The DENTEX disease
    boxes are boxes around the diseased tooth, so a detector box on a tooth is a good box for any disease of it.
    For a prediction (tooth t, disease d) the donor is the highest-scoring detector box of disease d lying mostly inside
    t (containment >= min_cont), else the highest-scoring detector box of any disease on t. It is used only if it
    overlaps our own box with IoU >= min_iou; otherwise our own box stays. Scores never change.
    cls_preds: from preds_from_teeth (they carry the crop id 'cid'); donors: fused detector boxes."""
    best = {}   # (crop id, disease or -1 for any) -> (score, box)
    by_img = {}
    for p in donors:
        by_img.setdefault(p['image_key'], []).append(p)
    for key, ps in by_img.items():
        rows = [r for r in rows_by_key.get(key, []) if not r.get('gt_only', 0)]
        if not rows or not ps:
            continue
        idx = assign_by_containment([p['box'] for p in ps], [[r['x1'], r['y1'], r['x2'], r['y2']] for r in rows],
                                    min_cont)
        for p, j in zip(ps, idx):
            if j < 0:
                continue
            cid, s = int(rows[j]['crop_id']), float(p['score'])
            for k in (int(p['d']), -1):
                cur = best.get((cid, k))
                if cur is None or s > cur[0]:
                    best[(cid, k)] = (s, p['box'])
    cand = []
    for p in cls_preds:
        c = p.get('cid')
        b = None if c is None else (best.get((int(c), int(p['d']))) or best.get((int(c), -1)))
        cand.append(None if b is None else b[1])
    hit = [i for i, b in enumerate(cand) if b is not None]
    ok = np.zeros(len(cls_preds), bool)
    if hit:
        iou = _iou_pairs([cand[i] for i in hit], [cls_preds[i]['box'] for i in hit])
        ok[np.asarray(hit)[iou >= min_iou]] = True
    out = []
    for i, p in enumerate(cls_preds):
        if ok[i]:
            q = dict(p)
            q['box'] = [float(v) for v in cand[i]]
            q['src'] = 'tooth, detector box'
            out.append(q)
        else:
            out.append(p)
    return out


def preds_from_disdet(dis_by_key, teeth_by_key, probs_by_key_fdi=None, mix=0.0, min_iou=0.3):
    """Plain detector boxes, FDI taken from the matched tooth. If probs are given the score is
    det_conf^(1-mix) * p_class^mix (symmetry-aware rescoring)."""
    out = []
    for key, boxes in dis_by_key.items():
        teeth = teeth_by_key.get(key, [])
        for b in boxes:
            t, _ = assign_tooth(b['box'], teeth, min_iou)
            q, n = (t['q'], t['n']) if t else (1, 1)
            s = b['conf']
            if probs_by_key_fdi is not None and mix > 0 and t is not None:
                p = probs_by_key_fdi.get((key, t['fdi']))
                if p is not None:
                    s = (max(b['conf'], 1e-6) ** (1 - mix)) * (max(float(p[b['cls']]), 1e-6) ** mix)
            out.append(dict(image_key=key, box=list(b['box']), q=q, n=n, d=int(b['cls']), score=float(s),
                            src='det'))
    return out


def nms_merge(preds, iou_thr=0.6):
    """Class-wise NMS per image on a list of predictions."""
    out = []
    groups = {}
    for p in preds:
        groups.setdefault((p['image_key'], p['d']), []).append(p)
    for _, ps in groups.items():
        ps = sorted(ps, key=lambda z: -z['score'])
        boxes = np.array([p['box'] for p in ps])
        keep = []
        alive = np.ones(len(ps), bool)
        for i in range(len(ps)):
            if not alive[i]:
                continue
            keep.append(ps[i])
            if i + 1 < len(ps):
                ious = iou_matrix(boxes[i:i + 1], boxes[i + 1:])[0]
                alive[i + 1:] &= ious < iou_thr
        out.extend(keep)
    return out


def top_k_per_image(preds, k=100):
    by = {}
    for p in preds:
        by.setdefault(p['image_key'], []).append(p)
    out = []
    for v in by.values():
        out.extend(sorted(v, key=lambda z: -z['score'])[:k])
    return out


# ----------------------------------------------------------------------------- evaluation
def coco_eval(gt_items, pred_items, cat_ids, image_keys, verbose=False):
    """gt_items: dict(image_key, box xyxy, cat). pred_items: dict(image_key, box xyxy, cat, score).
    Returns AP, AP50, AP75, AR (COCO AR@100, averaged over IoU 0.5:0.95), per-class AP and
    best F1 per class at IoU 0.5."""
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    kid = {k: i + 1 for i, k in enumerate(sorted(set(image_keys)))}
    anns = []
    for i, g in enumerate(gt_items):
        x1, y1, x2, y2 = g['box']
        anns.append(dict(id=i + 1, image_id=kid[g['image_key']], category_id=int(g['cat']),
                         bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0))
    ds = dict(images=[dict(id=v) for v in kid.values()], annotations=anns,
              categories=[dict(id=int(c)) for c in cat_ids])
    res = []
    for p in pred_items:
        if p['image_key'] not in kid:
            continue
        x1, y1, x2, y2 = p['box']
        res.append(dict(image_id=kid[p['image_key']], category_id=int(p['cat']),
                        bbox=[x1, y1, x2 - x1, y2 - y1], score=float(p['score'])))
    out = dict(AP=0.0, AP50=0.0, AP75=0.0, AR=0.0, per_class_AP={}, best_F1_50={})
    if not res or not anns:
        return out
    ctx = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    with ctx:
        gt = COCO()
        gt.dataset = ds
        gt.createIndex()
        dt = gt.loadRes(res)
        E = COCOeval(gt, dt, 'bbox')
        E.params.maxDets = [1, 10, 100]
        E.evaluate()
        E.accumulate()
        E.summarize()
    st = E.stats
    out.update(AP=100 * max(st[0], 0), AP50=100 * max(st[1], 0), AP75=100 * max(st[2], 0),
               AR=max(st[8], 0))
    prec = E.eval['precision']            # [T, R, K, A, M]
    rec_thr = np.asarray(E.params.recThrs)
    for k, c in enumerate(E.params.catIds):
        p_all = prec[:, :, k, 0, -1]
        v = p_all[p_all > -1]
        out['per_class_AP'][int(c)] = 100 * float(v.mean()) if v.size else float('nan')
        p50 = prec[0, :, k, 0, -1]
        ok = p50 > -1
        if ok.any():
            pp, rr = p50[ok], rec_thr[ok]
            f1 = np.where(pp + rr > 0, 2 * pp * rr / np.maximum(pp + rr, 1e-9), 0)
            out['best_F1_50'][int(c)] = float(f1.max())
    return out


def dentex_eval(gt_records, preds, image_keys=None):
    """DENTEX-style: the same predicted boxes are scored three times, once with quadrant labels,
    once with enumeration labels and once with diagnosis labels (COCO AP, AP50, AP75, AR)."""
    keys = image_keys or [r['key'] for r in gt_records]
    keyset = set(keys)
    gts = []
    for r in gt_records:
        if r['key'] not in keyset:
            continue
        for a in r['anns']:
            if 'd' in a:
                gts.append(dict(image_key=r['key'], box=xywh_to_xyxy(a['box']),
                                q=a.get('q', 1), n=a.get('n', 1), d=a['d']))
    res = {}
    for task, field, cats in (('quadrant', 'q', [1, 2, 3, 4]), ('enumeration', 'n', list(range(1, 9))),
                              ('diagnosis', 'd', [1, 2, 3, 4])):
        off = 1 if field == 'd' else 0
        g = [dict(image_key=x['image_key'], box=x['box'], cat=x[field] + off) for x in gts]
        p = [dict(image_key=x['image_key'], box=x['box'], cat=x[field] + off, score=x['score']) for x in preds]
        res[task] = coco_eval(g, p, cats, keys)
    return res


def fast_ap(gt_items, pred_items, cat_ids, image_keys):
    """COCO AP and AP50 only (all areas, 100 detections): same numbers as coco_eval, ~4x faster (for bootstraps)"""
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    kid = {k: i + 1 for i, k in enumerate(sorted(set(image_keys)))}
    anns = [dict(id=i + 1, image_id=kid[g['image_key']], category_id=int(g['cat']),
                 bbox=[g['box'][0], g['box'][1], g['box'][2] - g['box'][0], g['box'][3] - g['box'][1]],
                 area=(g['box'][2] - g['box'][0]) * (g['box'][3] - g['box'][1]), iscrowd=0) for i, g in enumerate(gt_items)]
    res = [dict(image_id=kid[x['image_key']], category_id=int(x['cat']), score=float(x['score']),
                bbox=[x['box'][0], x['box'][1], x['box'][2] - x['box'][0], x['box'][3] - x['box'][1]])
           for x in pred_items if x['image_key'] in kid]
    if not res or not anns:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = dict(images=[dict(id=v) for v in kid.values()], annotations=anns,
                          categories=[dict(id=int(c)) for c in cat_ids])
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(res), 'bbox')
        E.params.areaRng, E.params.areaRngLbl, E.params.maxDets = [[0, 1e10]], ['all'], [100]
        E.evaluate()
        E.accumulate()
    p = E.eval['precision'][:, :, :, 0, 0]
    ap = p[p > -1].mean() if (p > -1).any() else 0.0
    p50 = p[0]
    ap50 = p50[p50 > -1].mean() if (p50 > -1).any() else 0.0
    return 100 * float(ap), 100 * float(ap50)


COCO_IOUS = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)   # as pycocotools
COCO_RECS = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)


def coco_iou(d, g):
    """pycocotools bbIou for xywh boxes without crowd, with its exact float operations. D x G."""
    dx, dy, dw, dh = (d[:, i:i + 1] for i in range(4))
    gx, gy, gw, gh = (g[None, :, i] for i in range(4))
    w = np.minimum(dw + dx, gw + gx) - np.maximum(dx, gx)
    h = np.minimum(dh + dy, gh + gy) - np.maximum(dy, gy)
    i = w * h
    with np.errstate(divide='ignore', invalid='ignore'):
        o = i / (dw * dh + gw * gh - i)
    return np.where((w > 0) & (h > 0), o, 0.0)


class CocoBoot:
    """Diagnosis AP / AP50 exactly as fast_ap (pycocotools, area 'all', best 100 per X-ray and class), but fast
    for bootstraps: every X-ray is matched once (matching only looks inside one X-ray), and a resample of X-rays
    only concatenates the cached matches, in the order pycocotools would see them."""

    def __init__(self, gt_records, preds, keys, cats=(1, 2, 3, 4), max_det=100):
        self.keys, self.cats = list(keys), tuple(cats)
        ks = set(self.keys)
        gt_by = {r['key']: r for r in gt_records if r['key'] in ks}
        P = {}
        for x in preds:
            if x['image_key'] in ks:
                P.setdefault((x['image_key'], int(x['d']) + 1), []).append(x)
        self.cache = {}
        for k in ks:
            anns = [a for a in gt_by[k]['anns'] if 'd' in a] if k in gt_by else []
            for c in self.cats:
                gb = [xywh_to_xyxy(a['box']) for a in anns if a['d'] + 1 == c]   # same round trip as fast_ap
                g = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in gb], float).reshape(-1, 4)
                ps = P.get((k, c), [])
                d = np.array([[p['box'][0], p['box'][1], p['box'][2] - p['box'][0], p['box'][3] - p['box'][1]]
                              for p in ps], float).reshape(-1, 4)
                s = np.array([float(p['score']) for p in ps], float)
                self.cache[(k, c)] = self._match(g, d, s, max_det)

    @staticmethod
    def _match(g, d, s, max_det):
        """pycocotools evaluateImg: detections by score (stable), best max_det, then per IoU threshold each
        detection takes the unmatched box with the highest IoU >= threshold (the last one on ties)"""
        order = np.argsort(-s, kind='mergesort')[:max_det]
        s, d = s[order], d[order]
        m = np.zeros((len(COCO_IOUS), len(s)), bool)
        if len(s) and len(g):
            ious = coco_iou(d, g)
            for t, thr in enumerate(COCO_IOUS):
                thr = min(thr, 1 - 1e-10)
                free = np.ones(len(g), bool)
                for j in range(len(s)):
                    ok = free & (ious[j] >= thr)
                    if ok.any():
                        cand = np.flatnonzero(ok)
                        v = ious[j, cand]
                        free[cand[np.flatnonzero(v == v.max())[-1]]] = False
                        m[t, j] = True
        return s, m, len(g)

    def ap(self, sample=None):
        """(AP, AP50) in % over the X-rays sample (indices into keys, repeats allowed; None = all of them)"""
        idx = range(len(self.keys)) if sample is None else sample
        named = sorted((f'{self.keys[j]}#{i}', self.keys[j]) for i, j in enumerate(idx))   # pycocotools image order
        ks = [k for _, k in named]
        per_cat = []
        for c in self.cats:
            parts = [self.cache[(k, c)] for k in ks]
            ngt = sum(p[2] for p in parts)
            if ngt == 0:
                continue                      # no box of this class in the resample: left out, as pycocotools
            S = np.concatenate([p[0] for p in parts])
            q = np.zeros((len(COCO_IOUS), len(COCO_RECS)))
            if len(S):
                M = np.concatenate([p[1] for p in parts], axis=1)[:, np.argsort(-S, kind='mergesort')]
                tp = np.cumsum(M, axis=1).astype(float)
                fp = np.cumsum(~M, axis=1).astype(float)
                rc = tp / ngt
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                nd = pr.shape[1]
                for t in range(len(COCO_IOUS)):
                    ii = np.searchsorted(rc[t], COCO_RECS, side='left')
                    ok = ii < nd
                    q[t, ok] = pr[t, ii[ok]]
            per_cat.append(q)
        if not per_cat:
            return 0.0, 0.0
        P = np.stack(per_cat)
        return 100 * float(P.mean()), 100 * float(P[:, 0].mean())


def paired_bootstrap(gt_records, preds_by_name, keys, ref, n_boot=500, seed=0):
    """Image-level paired bootstrap of the diagnosis AP: every resample draws len(keys) X-rays with replacement
    and scores ALL models on that same set, so model differences are paired. Returns, per model, the observed
    AP/AP50 difference to `ref` and its 95% interval. Same resamples for the same keys and seed."""
    keys = list(keys)
    boots = {n: CocoBoot(gt_records, P, keys) for n, P in preds_by_name.items()}
    obs = {n: b.ap() for n, b in boots.items()}
    rng = np.random.default_rng(seed)
    diffs = {n: [] for n in preds_by_name if n != ref}
    for _ in range(n_boot):
        sample = rng.integers(0, len(keys), len(keys))
        r0 = boots[ref].ap(sample)
        for n in diffs:
            a = boots[n].ap(sample)
            diffs[n].append((a[0] - r0[0], a[1] - r0[1]))
    out = {}
    for n, v in diffs.items():
        v = np.array(v)
        out[n] = dict(AP=round(float(obs[n][0]), 2), dAP=round(float(obs[n][0] - obs[ref][0]), 2),
                      dAP_ci=f'[{np.percentile(v[:, 0], 2.5):+.1f}, {np.percentile(v[:, 0], 97.5):+.1f}]',
                      dAP_lo=round(float(np.percentile(v[:, 0], 2.5)), 2), dAP_hi=round(float(np.percentile(v[:, 0], 97.5)), 2),
                      AP50=round(float(obs[n][1]), 2), dAP50=round(float(obs[n][1] - obs[ref][1]), 2),
                      dAP50_ci=f'[{np.percentile(v[:, 1], 2.5):+.1f}, {np.percentile(v[:, 1], 97.5):+.1f}]',
                      p_better=round(float((v[:, 0] > 0).mean()), 3))
    return out


# ----------------------------------------------------------------------------- ultralytics-style metric
# The YOLO-based DENTEX papers report the numbers printed by ultralytics' validator, which differ from COCO:
# predictions are matched to boxes in order of IoU (COCO: in order of confidence), all of them count
# (COCO: best 100 per image) and AP is the area under a linearly interpolated precision envelope.
UL_IOUV = np.linspace(0.5, 0.95, 10).astype(np.float32).astype(float)   # torch.linspace in float32, as ultralytics


def ul_match(pred_cls, true_cls, iou):
    """ultralytics BaseValidator.match_predictions (default, non-scipy). iou: labels x predictions."""
    correct = np.zeros((len(pred_cls), len(UL_IOUV)), bool)
    iou = iou * (np.asarray(true_cls)[:, None] == np.asarray(pred_cls)[None, :])
    for i, thr in enumerate(UL_IOUV):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def ul_compute_ap(recall, precision, new_formula=True):
    """ultralytics compute_ap. new_formula: version >= 8.4.37 (precision drops to 0 after the last recall).
    Old formula (YOLOv5 ... ultralytics 8.4.36, what the 2024-2026 papers used): a straight line from the last
    point to (recall 1, precision 0), which adds area whenever recall stays below 1."""
    if new_formula:
        mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    else:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return float(trap(np.interp(x, mrec, mpre), x))


def ul_ap_per_class(tp, conf, pred_cls, target_cls, new_formula=True, eps=1e-16):
    """AP per class (classes present in the labels) x 10 IoU thresholds, as ultralytics ap_per_class"""
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    classes, nt = np.unique(target_cls, return_counts=True)
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, c in enumerate(classes):
        k = pred_cls == c
        if k.sum() == 0 or nt[ci] == 0:
            continue
        fpc = (1 - tp[k]).cumsum(0)
        tpc = tp[k].cumsum(0)
        recall = tpc / (nt[ci] + eps)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j] = ul_compute_ap(recall[:, j], precision[:, j], new_formula)
    return classes, ap


def ul_eval(gt_records, preds, image_keys):
    """Diagnosis mAP50-95 / mAP50 computed the way ultralytics' validator does, on the given X-rays.
    Keys without suffix: current ultralytics (>= 8.4.37, what model.val() prints on Kaggle now);
    '_old': the AP formula of YOLOv5 ... ultralytics 8.4.36, i.e. how the peer papers' numbers were made."""
    keys = sorted(set(image_keys))
    keyset = set(keys)
    gt_by = {r['key']: [(xywh_to_xyxy(a['box']), int(a['d'])) for a in r['anns'] if 'd' in a]
             for r in gt_records if r['key'] in keyset}
    p_by = {}
    for p in preds:
        if p['image_key'] in keyset:
            p_by.setdefault(p['image_key'], []).append(p)
    tps, confs, pcls, tcls = [], [], [], []
    for k in keys:
        g = gt_by.get(k, [])
        gc = np.array([c for _, c in g], int)
        tcls.append(gc)
        P = p_by.get(k, [])
        if not P:
            continue
        pc = np.array([int(p['d']) for p in P], int)
        if len(g):
            tp = ul_match(pc, gc, iou_matrix(np.array([b for b, _ in g], float), np.array([p['box'] for p in P], float)))
        else:
            tp = np.zeros((len(P), len(UL_IOUV)), bool)
        tps.append(tp); confs.append(np.array([float(p['score']) for p in P])); pcls.append(pc)
    tc = np.concatenate(tcls) if tcls else np.zeros(0, int)
    out = {}
    if not tps or not len(tc):
        for tag in ('', '_old'):
            out.update({'mAP' + tag: 0.0, 'mAP50' + tag: 0.0, 'per_class_AP50' + tag: {}})
        return out
    tp, cf, pc = np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls)
    for tag, new in (('', True), ('_old', False)):
        classes, ap = ul_ap_per_class(tp, cf, pc, tc, new)
        out['mAP' + tag] = 100 * float(ap.mean())
        out['mAP50' + tag] = 100 * float(ap[:, 0].mean())
        out['per_class_AP50' + tag] = {DIS[c]: round(100 * float(ap[i, 0]), 1) for i, c in enumerate(classes)}
    return out


# ----------------------------------------------------------------------------- self-training pool
POOL_TAGS = ('enum', 'quad', 'unl')
POOL_EXPECTED = {'enum': 634, 'quad': 693, 'unl': 1571}   # DENTEX X-rays without diagnosis labels


def pool_tag(rel):
    """which DENTEX subset an image belongs to, from its path inside the data folder: the X-rays without diagnosis
    labels are the enumeration set, the quadrant-only set and the unlabelled set (None = anything else)"""
    s = '/' + str(rel).lower().replace('\\', '/')
    if any(t in s for t in ('valid', '/test', 'disease', '_decoded', '__macosx', '/._')):
        return None
    if 'unlabel' in s:
        return 'unl'
    if 'enumeration' in s:
        return 'enum'
    if 'quadrant' in s:
        return 'quad'
    return None


def find_pool(roots):
    """{tag: sorted image paths} of the X-rays without diagnosis labels; each subset is taken from the first
    folder in `roots` that has it, so a subset found twice is not used twice"""
    out = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        here = {}
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                t = pool_tag(p.relative_to(root))
                if t:
                    here.setdefault(t, set()).add(str(p))
        for t, v in here.items():
            out.setdefault(t, sorted(v))
    return out


def thumb_of(im):
    """64x32 grey thumbnail used to find copies of the same X-ray"""
    import cv2
    return cv2.resize(im, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def thumb_file(path):
    return thumb_of(read_gray(path))


def export_pool_image(task):
    """(source, jpg, long_side) -> (W, H, scale, thumbnail), or None if the file is not a readable image.
    The jpg is made exactly like resize_long's."""
    import cv2
    src, dst, long_side = task
    try:
        im = read_gray(src)
    except Exception:
        return None
    h, w = im.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if not Path(dst).exists():
        small = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else im
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return w, h, s, thumb_of(im)


def build_crops_chunk(task):
    """build_crops for one chunk of X-rays (a picklable task for pmap); returns the rows"""
    return build_crops(*task)[0]


def pmap(fn, tasks, workers=4, chunksize=4):
    """map fn over tasks in worker processes ('spawn': safe in a notebook that already uses CUDA and threads);
    falls back to one process if that fails"""
    tasks = list(tasks)
    if workers > 1 and len(tasks) > 8:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
                return list(ex.map(fn, tasks, chunksize=chunksize))
        except Exception as e:
            print('   parallel map failed, running in one process:', str(e)[:200])
    return [fn(t) for t in tasks]


def _standardize(x):
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-6)


def rms_nearest(a, b=None):
    """RMS difference between standardised thumbnails (zero mean, unit variance, so a copy saved with other
    brightness or contrast still counts) from each one in a to its nearest one in b (to the nearest EARLIER one in a
    when b is None). sqrt(2 - 2 * correlation): ~0.05 for the same X-ray saved twice. Returns (distance, index)."""
    A = _standardize(a)
    B = A if b is None else _standardize(b)
    D = A.shape[1]
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), np.inf), np.full(len(A), -1)
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    d2 = np.maximum(d2, 0) / D
    if b is None:
        d2[np.triu_indices(len(A))] = np.inf     # only earlier images count
    j = d2.argmin(1)
    return np.sqrt(d2[np.arange(len(A)), j]), j


def assign_by_containment(boxes, tooth_boxes, min_cont=None):
    """index of the tooth each box belongs to: the tooth box holding the largest part of it. A box outside every
    tooth box goes to the tooth with the nearest centre, or, with min_cont, every box with less than that fraction
    inside its best tooth box gets -1 (also -1 when there are no teeth)"""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    T = np.asarray(tooth_boxes, float).reshape(-1, 4)
    if len(T) == 0:
        return np.full(len(boxes), -1)
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1 = np.maximum(boxes[:, None, 0], T[None, :, 0]); y1 = np.maximum(boxes[:, None, 1], T[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], T[None, :, 2]); y2 = np.minimum(boxes[:, None, 3], T[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    cont = inter / area[:, None]
    j = cont.argmax(1)
    c = (boxes[:, :2] + boxes[:, 2:]) / 2
    tc = (T[:, :2] + T[:, 2:]) / 2
    near = ((c[:, None, :] - tc[None]) ** 2).sum(-1).argmin(1)
    if min_cont is not None:
        return np.where(cont.max(1) >= min_cont, j, -1)
    return np.where(cont.max(1) > 0, j, near)


def tooth_scores(preds, rows_by_key, min_cont=0.3):
    """{crop_id: [4 scores]}: per tooth and disease, the best score of the predicted boxes that belong to that tooth
    (assign_by_containment; a box with less than min_cont of its area inside any tooth box is ignored, e.g. an impacted
    tooth the tooth detector missed). preds: dict(image_key, box, d, score) as made by preds_from_disdet."""
    out = {}
    by = {}
    for p in preds:
        by.setdefault(p['image_key'], []).append(p)
    for key, rows in rows_by_key.items():
        rows = [r for r in rows if not r.get('gt_only', 0)]
        for r in rows:
            out[int(r['crop_id'])] = np.zeros(4)
        ps = by.get(key, [])
        if not ps or not rows:
            continue
        idx = assign_by_containment([p['box'] for p in ps], [[r['x1'], r['y1'], r['x2'], r['y2']] for r in rows],
                                    min_cont)
        for p, j in zip(ps, idx):
            if j < 0:
                continue
            v = out[int(rows[j]['crop_id'])]
            v[p['d']] = max(v[p['d']], float(p['score']))
    return out


def f1_threshold(gt_records, preds, keys, d=None, iou=0.5):
    """score threshold with the best F1 at IoU >= iou (each annotated box matched once, highest scores first, per
    X-ray and disease). d = one disease, None = all four pooled. Returns dict(thr, f1, p, r, n_gt)."""
    keys = set(keys)
    gts = {}
    for r in gt_records:
        if r['key'] not in keys:
            continue
        for a in r['anns']:
            if 'd' in a and (d is None or a['d'] == d):
                gts.setdefault((r['key'], a['d']), []).append(xywh_to_xyxy(a['box']))
    n_gt = sum(len(v) for v in gts.values())
    cand = sorted([p for p in preds if p['image_key'] in keys and (d is None or p['d'] == d)], key=lambda z: -z['score'])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(cand))
    for i, p in enumerate(cand):
        g = gts.get((p['image_key'], p['d']))
        if not g:
            continue
        ious = iou_matrix([p['box']], g)[0]
        ious[used[(p['image_key'], p['d'])]] = -1
        j = int(np.argmax(ious))
        if ious[j] >= iou:
            used[(p['image_key'], p['d'])][j] = True
            tp[i] = 1
    if not len(cand) or n_gt == 0:
        return dict(thr=1.0, f1=0.0, p=0.0, r=0.0, n_gt=n_gt)
    ctp = np.cumsum(tp)
    prec, rec = ctp / np.arange(1, len(cand) + 1), ctp / n_gt
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
    k = int(np.argmax(f1))
    return dict(thr=float(cand[k]['score']), f1=float(f1[k]), p=float(prec[k]), r=float(rec[k]), n_gt=n_gt)


def yolo_label_lines(boxes, W, H):
    """YOLO label lines 'cls cx cy w h' (normalised) for (d, x1, y1, x2, y2) boxes in original pixels"""
    out = []
    for d, x1, y1, x2, y2 in boxes:
        x1, x2 = min(max(x1, 0), W), min(max(x2, 0), W)
        y1, y2 = min(max(y1, 0), H), min(max(y2, 0), H)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(f'{int(d)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
    return out


# ----------------------------------------------------------------------------- box fusion
try:   # numba is in the Kaggle image; without it the same algorithm runs in numpy (~6x slower, same result)
    from numba import njit as _njit
except Exception:
    _njit = None


def _wbf_group_np(S, B, thr):
    """one X-ray and disease: boxes B (sorted by weighted score S, highest first) -> clusters as running sums"""
    n = len(S)
    fused, acc = np.empty((n, 4)), np.empty((n, 4))
    ssum, cnt, area, top = np.empty(n), np.zeros(n, np.int64), np.empty(n), np.empty(n, np.int64)
    k = 0
    for i in range(n):
        b = B[i]
        j = -1
        ab = (b[2] - b[0]) * (b[3] - b[1])
        if k:
            f = fused[:k]
            iw = np.minimum(f[:, 2], b[2]) - np.maximum(f[:, 0], b[0])
            ih = np.minimum(f[:, 3], b[3]) - np.maximum(f[:, 1], b[1])
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            iou = inter / np.maximum(area[:k] + ab - inter, 1e-9)
            j = int(np.argmax(iou))
            if iou[j] <= thr:
                j = -1
        s = S[i]
        if j >= 0:
            acc[j] += s * b
            ssum[j] += s
            cnt[j] += 1
            fb = acc[j] / max(ssum[j], 1e-12)
            fused[j] = fb
            area[j] = (fb[2] - fb[0]) * (fb[3] - fb[1])
        else:
            fused[k], acc[k], ssum[k], cnt[k], area[k], top[k] = b, s * b, s, 1, ab, i
            k += 1
    return fused[:k], ssum[:k], cnt[:k], top[:k]


if _njit is not None:
    @_njit(cache=False)
    def _wbf_group_nb(S, B, thr):
        n = S.shape[0]
        fused = np.empty((n, 4))
        acc = np.empty((n, 4))
        ssum = np.empty(n)
        cnt = np.zeros(n, np.int64)
        area = np.empty(n)
        top = np.empty(n, np.int64)
        k = 0
        for i in range(n):
            bx1, by1, bx2, by2 = B[i, 0], B[i, 1], B[i, 2], B[i, 3]
            ab = (bx2 - bx1) * (by2 - by1)
            best = -1.0
            j = -1
            for c in range(k):
                iw = min(fused[c, 2], bx2) - max(fused[c, 0], bx1)
                ih = min(fused[c, 3], by2) - max(fused[c, 1], by1)
                inter = max(iw, 0.0) * max(ih, 0.0)
                den = area[c] + ab - inter
                if den < 1e-9:
                    den = 1e-9
                iou = inter / den
                if iou > best:
                    best = iou
                    j = c
            if j >= 0 and best <= thr:
                j = -1
            s = S[i]
            if j >= 0:
                for q in range(4):
                    acc[j, q] += s * B[i, q]
                ssum[j] += s
                cnt[j] += 1
                den = ssum[j] if ssum[j] > 1e-12 else 1e-12
                for q in range(4):
                    fused[j, q] = acc[j, q] / den
                area[j] = (fused[j, 2] - fused[j, 0]) * (fused[j, 3] - fused[j, 1])
            else:
                for q in range(4):
                    fused[k, q] = B[i, q]
                    acc[k, q] = s * B[i, q]
                ssum[k] = s
                cnt[k] = 1
                area[k] = ab
                top[k] = i
                k += 1
        return fused[:k].copy(), ssum[:k].copy(), cnt[:k].copy(), top[:k].copy()
else:
    _wbf_group_nb = None


def wbf(pred_lists, weights=None, iou_thr=0.55):
    """Weighted boxes fusion (Solovyev et al. 2021; same result as ensemble_boxes 1.0.9 with conf_type='avg'),
    per X-ray and disease. Boxes of the models are clustered (IoU > iou_thr with the running fused box), the fused
    box is the score-weighted mean of its cluster and the score is the mean weighted score times
    min(#models, #boxes) / sum(weights), so a box that only one model found keeps a lower score.
    The FDI labels come from the best box of the cluster."""
    w = np.ones(len(pred_lists)) if weights is None else np.asarray(weights, float)
    groups = {}
    for t, P in enumerate(pred_lists):
        wt = w[t]
        for p in P:
            groups.setdefault((p['image_key'], int(p['d'])), []).append((float(p['score']) * wt, p))
    global _wbf_group_nb
    fn = _wbf_group_nb if _wbf_group_nb is not None else _wbf_group_np
    out = []
    n_models, wsum = len(w), float(w.sum())
    for (key, d), items in groups.items():
        items.sort(key=lambda z: -z[0])
        n = len(items)
        S = np.fromiter((x[0] for x in items), float, n)
        B = np.array([x[1]['box'] for x in items], float).reshape(n, 4)
        try:
            fused, ssum, cnt, top = fn(S, B, float(iou_thr))
        except Exception:
            if fn is _wbf_group_np:
                raise
            print('   numba WBF failed, using the numpy version (same result, slower)')
            _wbf_group_nb, fn = None, _wbf_group_np
            fused, ssum, cnt, top = fn(S, B, float(iou_thr))
        for j in range(len(ssum)):
            t = items[top[j]][1]
            out.append(dict(image_key=key, box=[float(v) for v in fused[j]], q=t.get('q', 1), n=t.get('n', 1),
                            d=d, score=float(ssum[j] / cnt[j] * min(n_models, cnt[j]) / wsum), src='wbf'))
    return out


def diag_eval_fast(gt_records, preds, image_keys):
    """diagnosis AP (0.5:0.95), AP50 and AP per disease {1..4: AP}: the same numbers as
    dentex_eval(...)['diagnosis'] (pycocotools, area 'all', best 100 per X-ray and disease), about 4x faster"""
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    keys = set(image_keys)
    kid = {k: i + 1 for i, k in enumerate(sorted(keys))}
    anns = []
    for r in gt_records:
        if r['key'] not in keys:
            continue
        for a in r['anns']:
            if 'd' in a:
                x, y, w, h = a['box']
                anns.append(dict(id=len(anns) + 1, image_id=kid[r['key']], category_id=int(a['d']) + 1,
                                 bbox=[x, y, w, h], area=w * h, iscrowd=0))
    res = [dict(image_id=kid[p['image_key']], category_id=int(p['d']) + 1, score=float(p['score']),
                bbox=[p['box'][0], p['box'][1], p['box'][2] - p['box'][0], p['box'][3] - p['box'][1]])
           for p in preds if p['image_key'] in kid]
    out = dict(AP=0.0, AP50=0.0, per_class_AP={})
    if not res or not anns:
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = dict(images=[dict(id=v) for v in kid.values()], annotations=anns,
                          categories=[dict(id=c) for c in (1, 2, 3, 4)])
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(res), 'bbox')
        E.params.areaRng, E.params.areaRngLbl, E.params.maxDets = [[0, 1e10]], ['all'], [100]
        E.evaluate()
        E.accumulate()
    p = E.eval['precision'][:, :, :, 0, 0]
    v = p[p > -1]
    out['AP'] = 100 * float(v.mean()) if v.size else 0.0
    p50 = p[0][p[0] > -1]
    out['AP50'] = 100 * float(p50.mean()) if p50.size else 0.0
    for k, c in enumerate(E.params.catIds):
        pk = p[:, :, k]
        pk = pk[pk > -1]
        out['per_class_AP'][int(c)] = 100 * float(pk.mean()) if pk.size else float('nan')
    return out


def summarize(res, name=''):
    d = res['diagnosis']
    pc = {DIS[c - 1]: round(v, 1) for c, v in d['per_class_AP'].items()}
    f1 = {DIS[c - 1]: round(v, 3) for c, v in d['best_F1_50'].items()}
    return dict(model=name,
                diag_AP=round(d['AP'], 2), diag_AP50=round(d['AP50'], 2), diag_AP75=round(d['AP75'], 2),
                diag_AR=round(d['AR'], 4),
                enum_AP=round(res['enumeration']['AP'], 2), enum_AP50=round(res['enumeration']['AP50'], 2),
                quad_AP=round(res['quadrant']['AP'], 2), quad_AP50=round(res['quadrant']['AP50'], 2),
                per_class_AP=pc, best_F1_iou50=f1)

# ----------------------------------------------------------------------------- v15 audit
def orb_inliers(task):
    """(path_a, path_b) -> number of ORB keypoint matches between two X-rays (1024 px, CLAHE) that one RANSAC
    homography explains. The same X-ray gives hundreds; two X-rays of the same patient (the same teeth, fillings and
    crowns) usually many; different patients few. -1 if an image cannot be read."""
    import cv2
    pa, pb = task

    def load(p):
        im = read_gray(p)
        s = 1024 / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(im)
    try:
        a, b = load(pa), load(pb)
        orb = cv2.ORB_create(nfeatures=1500)
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None or len(ka) < 10 or len(kb) < 10:
            return 0
        good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            return 0
        src = np.float32([ka[g.queryIdx].pt for g in good])
        dst = np.float32([kb[g.trainIdx].pt for g in good])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)
        return int(mask.sum()) if mask is not None else 0
    except Exception:
        return -1


def tooth_ious(gt_teeth, pred_teeth):
    """per annotated tooth (dicts with q, n and an xyxy box): IoU with the predicted tooth of the same FDI number,
    0 when the prediction has no such tooth"""
    pred = {(int(t['q']), int(t['n'])): t['box'] for t in pred_teeth}
    out = []
    for t in gt_teeth:
        b = pred.get((int(t['q']), int(t['n'])))
        out.append(float(iou_matrix([t['box']], [b])[0, 0]) if b is not None else 0.0)
    return out


def pair_montage(pairs, out_path, h=200):
    """[(path_a, path_b, label)] -> one JPEG with each pair side by side and its label (to check near copies by eye)"""
    import cv2
    rows = []
    for pa, pb, label in pairs:
        ims = []
        for p in (pa, pb):
            try:
                im = read_gray(p)
                im = cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
            except Exception:
                im = np.zeros((h, 2 * h), np.uint8)
            ims.append(im)
        wmax = 2 * h * 2
        canvas = np.full((h + 28, wmax + 10), 255, np.uint8)
        x = 0
        for im in ims:
            w = min(im.shape[1], wmax // 2)
            canvas[28:28 + h, x:x + w] = im[:, :w]
            x += wmax // 2 + 10
        cv2.putText(canvas, str(label)[:120], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
        rows.append(canvas)
    if not rows:
        return None
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(out_path)
