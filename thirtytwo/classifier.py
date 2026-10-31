"""The tooth classifiers (train_arch.TwinArch, variant 'single') on in-memory tooth crops.

Crops are cut from the decoded original X-ray with dentex_utils.crop_tooth, as dentex_utils.build_crops does (which
saves them as PNG, a lossless round trip). They are handed to train_arch.predict through ArchDS's crop cache, so the
mirroring of quadrants 2 and 3, the normalisation and the forward pass are train_arch's own code.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import config as C

KEY = 'upload'      # every prediction is about one X-ray; the notebook's functions want an image key


def tooth_rows(gray, teeth):
    """one row per found tooth, as build_crops writes them (crop ids 0..n-1 in tooth order), and the crops by path"""
    import dentex_utils as U
    rows, cache = [], {}
    for cid, t in enumerate(teeth):
        path = f'memory://crop/{cid}'
        cache[path] = U.crop_tooth(gray, t['box'], t['q'], C.CROP_H, C.CROP_W)
        x1, y1, x2, y2 = t['box']
        rows.append(dict(crop_id=cid, image_key=KEY, split='upload', q=t['q'], n=t['n'], fdi=t['fdi'],
                         x1=x1, y1=y1, x2=x2, y2=y2, det_conf=t['conf'], gt_only=0,
                         **{c: 0 for c in U.LABEL_COLS}, crop_path=path, twin_crop_id=-1))
    return rows, cache


class ToothClassifier:
    def __init__(self, weights, device):
        import train_arch as TA
        a = C.CLASSIFIER_ARCH
        self.TA = TA
        self.device = device
        self.core = TA.TwinArch(a['backbone'], a['variant'], pretrained=False, dim=a['dim'], layers=a['layers'],
                                heads=a['heads'], drop_path=a['drop_path'], box_src=a['box_src']).to(device)
        self.core.load_state_dict(torch.load(weights, map_location=device))     # as run_predict_only
        self.core.eval()
        self.args = SimpleNamespace(crop_h=C.CROP_H, crop_w=C.CROP_W, max_teeth=a['max_teeth'], img_bs=a['img_bs'],
                                    workers=0)
        self.use_amp = device.type == 'cuda'     # train_arch.main: use_amp = cuda (the references ran on GPU)

    def __call__(self, rows, cache):
        """{crop_id: 4 probabilities}, like the notebook's load_probs of a pred_*.csv"""
        if not rows:
            return {}
        ids, probs, _, _ = self.TA.predict(self.core, pd.DataFrame(rows), self.args, self.device, self.use_amp,
                                           cache, tta=1)
        return {int(c): np.array(p, float) for c, p in zip(ids, probs)}
