#!/usr/bin/env python
"""TwinArch: contralateral-aware dental arch reasoning for tooth-level diagnosis.

One X-ray = one set of teeth. Every tooth crop is encoded ONCE by a small CNN. Crops from the
image-right side (quadrants 2 and 3) are mirrored before encoding, so a tooth and its contralateral
twin (16<->26, 36<->46, ...) end up in the same orientation and their features can be compared
directly. Then:

  single : head(f_i + FDI_i)                                  no context (plain crop classifier)
  twin   : head([f_i + FDI_i, |f_i - f_twin|, has_twin])      pairwise asymmetry only
  arch   : relation-biased transformer over all teeth          arch context, no explicit asymmetry
  full   : arch + asymmetry token input + asymmetry in head    OURS

The transformer's attention gets a learned bias per tooth relation (self, twin, neighbour,
antagonist, same quadrant, other), so anatomy is built into the attention instead of being learned
from scratch on 705 X-rays.

2 GPUs:  torchrun --standalone --nproc_per_node=2 train_arch.py ...
1 GPU :  python train_arch.py ...
"""
import os, json, math, time, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

LABELS = ['y_caries', 'y_deep', 'y_peri', 'y_impacted']
MEAN, STD = 0.449, 0.226
TWIN_Q = {1: 2, 2: 1, 3: 4, 4: 3}
ANTAG_Q = {1: 4, 4: 1, 2: 3, 3: 2}
REL_SELF, REL_TWIN, REL_NEIGH, REL_ANTAG, REL_QUAD, REL_OTHER = range(6)
N_REL = 6


# ----------------------------------------------------------------------------- args / ddp
def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--train-keys', default='', help='image keys to train on (required unless --predict-only)')
    ap.add_argument('--val-keys', default='', help='image keys for checkpoint selection (required unless --predict-only)')
    ap.add_argument('--pred-keys', nargs='*', default=[], help='name=path pairs')
    ap.add_argument('--variant', default='full', choices=['single', 'twin', 'arch', 'full'])
    ap.add_argument('--backbone', default='convnext_nano.in12k_ft_in1k')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--img-bs', type=int, default=2, help='X-rays per GPU per step')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--head-lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--max-teeth', type=int, default=48)
    ap.add_argument('--crop-h', type=int, default=384)
    ap.add_argument('--crop-w', type=int, default=224)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--max-images', type=int, default=0, help='smoke test: cap X-rays per split')
    ap.add_argument('--no-cache', action='store_true', help='read crops from disk every time')
    ap.add_argument('--drop-path', type=float, default=0.1, help='stochastic depth in the encoder')
    ap.add_argument('--ema', type=float, default=0.997, help='weight averaging decay, 0 = off')
    ap.add_argument('--box-targets', default='', help='csv from the notebook: annotated-box offsets per tooth')
    ap.add_argument('--box-w', type=float, default=4.0, help='weight of the learned-box loss')
    ap.add_argument('--box-src', default='both', choices=['token', 'map', 'both'],
                    help='learned boxes from the pooled tooth token, from the spatial feature map, or both '
                         '(two detached heads trained side by side; pred files get b* and s* columns)')
    ap.add_argument('--predict-only', action='store_true', help='no training: load --weights and write predictions')
    ap.add_argument('--weights', default='', help='state dict for --predict-only (a best.pt of an earlier run)')
    ap.add_argument('--tta', type=int, default=1, help='test-time augmentation: average over this many fixed views '
                                                        '(1 = off, up to 5: identity, zoom in/out, rotate +/-3 deg)')
    ap.add_argument('--pool-ratio', type=float, default=1.0,
                    help='self-training: pseudo-labelled X-rays (manifest rows with w < 1) drawn per annotated X-ray in '
                         'each epoch; 0 = the whole pool every epoch')
    a = ap.parse_args(argv)
    if not a.predict_only and not (a.train_keys and a.val_keys):
        ap.error('--train-keys and --val-keys are required for training')
    if a.predict_only and not a.weights:
        ap.error('--predict-only needs --weights')
    a.tta = max(1, min(int(a.tta), len(TTA_VIEWS)))
    return a


def setup_ddp():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        local = int(os.environ.get('LOCAL_RANK', 0))
        try:
            if torch.cuda.is_available():
                dist.init_process_group(backend=backend, device_id=torch.device(f'cuda:{local}'))
            else:
                dist.init_process_group(backend=backend)
        except TypeError:          # older torch without device_id
            dist.init_process_group(backend=backend)
        rank, local = dist.get_rank(), int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def barrier(world):
    if world > 1:
        dist.barrier()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_keys(p):
    return set(x.strip() for x in Path(p).read_text().splitlines() if x.strip())


# ----------------------------------------------------------------------------- numpy helpers
def load_crop(path, H, W):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return np.zeros((H, W), np.float32)
    if im.shape != (H, W):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0


def preload(paths, H, W):
    """read every crop once into RAM as uint8; DataLoader workers share it after fork"""
    cache = {}
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            im = np.zeros((H, W), np.uint8)
        elif im.shape != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        cache[str(p)] = im
    return cache


def sample_photo(rng):
    """exposure settings drawn ONCE per X-ray and shared by all its teeth. Exposure belongs to the image,
    and drawing it per tooth would fake left-right density differences that TwinArch then learns to ignore."""
    use_cb = rng.random() < 0.8
    return dict(contrast=rng.uniform(0.8, 1.25) if use_cb else 1.0,
                bright=rng.uniform(-0.08, 0.08) if use_cb else 0.0,
                gamma=rng.uniform(0.75, 1.35) if rng.random() < 0.3 else None,
                blur=rng.uniform(0.3, 1.0) if rng.random() < 0.2 else None,
                noise=0.02 if rng.random() < 0.2 else 0.0)


def augment(a, rng, photo):
    """returns (augmented crop, 2x3 affine used or None)"""
    H, W = a.shape
    M = None
    if rng.random() < 0.9:     # small geometric jitter stays per tooth, like box noise from the tooth detector
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-7, 7), rng.uniform(0.9, 1.1))
        M[:, 2] += (rng.uniform(-0.05, 0.05) * W, rng.uniform(-0.05, 0.05) * H)
        a = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if photo['contrast'] != 1.0 or photo['bright'] != 0.0:
        a = np.clip((a - 0.5) * photo['contrast'] + 0.5 + photo['bright'], 0, 1)
    if photo['gamma'] is not None:
        a = np.clip(a, 0, 1) ** photo['gamma']
    if photo['blur'] is not None:
        a = cv2.GaussianBlur(a, (3, 3), photo['blur'])
    if photo['noise'] > 0:     # sensor noise is independent per pixel anyway
        a = np.clip(a + rng.normal(0, photo['noise'], a.shape).astype(np.float32), 0, 1)
    return a.astype(np.float32), M


TTA_VIEWS = [None, (0.0, 1.06), (0.0, 0.94), (3.0, 1.0), (-3.0, 1.0)]   # (rotation deg, scale), inside training jitter


def tta_matrix(view, H, W):
    return None if view is None else cv2.getRotationMatrix2D((W / 2, H / 2), view[0], view[1])


def tooth_frame(w, h, upper, H, W, mx=0.15, my_crown=0.08, my_apex=0.20):
    """where the found tooth box sits inside its letterboxed crop (same geometry as dentex_utils.crop_tooth)"""
    top, bot = (my_apex, my_crown) if upper else (my_crown, my_apex)
    RW, RH = w * (1 + 2 * mx), h * (1 + top + bot)
    s = min(H / RH, W / RW)
    ox, oy = (W - RW * s) / 2, (H - RH * s) / 2
    fx1, fy1 = ox + mx * w * s, oy + top * h * s
    return fx1, fy1, fx1 + w * s, fy1 + h * s


def flip_offsets(o):
    """edge offsets [dx1, dy1, dx2, dy2] under a left-right mirror of the crop (its own inverse)"""
    return np.stack([-o[..., 2], o[..., 1], -o[..., 0], o[..., 3]], -1)


def warp_offsets(o, frame, M):
    """move the annotated box with the same affine as the crop, measured against the fixed tooth frame"""
    fx1, fy1, fx2, fy2 = frame
    FW, FH = max(fx2 - fx1, 1e-3), max(fy2 - fy1, 1e-3)
    x1, y1, x2, y2 = fx1 + o[0] * FW, fy1 + o[1] * FH, fx2 + o[2] * FW, fy2 + o[3] * FH
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], np.float64) @ np.asarray(M, np.float64).T
    nx1, ny1 = pts[:, 0].min(), pts[:, 1].min()
    nx2, ny2 = pts[:, 0].max(), pts[:, 1].max()
    return np.array([(nx1 - fx1) / FW, (ny1 - fy1) / FH, (nx2 - fx2) / FW, (ny2 - fy2) / FH], np.float32)


def twin_index(qs, ns, gt_only):
    """index of the contralateral twin inside the same X-ray (prefer detected teeth), else -1"""
    T = len(qs)
    out = np.full(T, -1, np.int64)
    for i in range(T):
        cand = [j for j in range(T) if j != i and qs[j] == TWIN_Q[int(qs[i])] and ns[j] == ns[i]]
        if cand:
            cand.sort(key=lambda j: gt_only[j])
            out[i] = cand[0]
    return out


def relations(qs, ns):
    qs = np.asarray(qs, np.int64)
    ns = np.asarray(ns, np.int64)
    qi, qj = qs[:, None], qs[None, :]
    ni, nj = ns[:, None], ns[None, :]
    twin_q = np.vectorize(TWIN_Q.get)(qs)[:, None] if len(qs) else qs[:, None]
    antag_q = np.vectorize(ANTAG_Q.get)(qs)[:, None] if len(qs) else qs[:, None]
    R = np.full((len(qs), len(qs)), REL_OTHER, np.int64)
    R[qi == qj] = REL_QUAD
    R[(qj == antag_q) & (nj == ni)] = REL_ANTAG
    R[(qi == qj) & (np.abs(ni - nj) == 1)] = REL_NEIGH
    R[(qj == twin_q) & (nj == ni)] = REL_TWIN
    np.fill_diagonal(R, REL_SELF)
    return R


# ----------------------------------------------------------------------------- dataset
class ArchDS(Dataset):
    """one item = one X-ray with all its tooth crops"""
    def __init__(self, df, H, W, train, max_teeth=48, seed=0, cache=None, boxes=None, view=None):
        self.groups = [g.reset_index(drop=True) for _, g in df.groupby('image_key', sort=True)]
        self.H, self.W, self.train, self.max_teeth, self.seed = H, W, train, max_teeth, seed
        self.cache = cache
        self.boxes = boxes    # {crop_id: (offsets [4 diseases, 4 edges], mask [4])} in the original image frame
        self.view = tta_matrix(view, H, W) if not train else None   # fixed test-time view
        # X-rays whose teeth carry the teacher's soft labels (weight w < 1) instead of annotations
        self.is_pool = [bool('w' in g and (g['w'] < 1).any()) for g in self.groups]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        g = self.groups[i]
        if len(g) > self.max_teeth:
            g = g.sort_values(['gt_only', 'det_conf'], ascending=[False, False]).head(self.max_teeth)
            g = g.reset_index(drop=True)
        rng = np.random.default_rng() if self.train else None
        photo = sample_photo(rng) if self.train else None
        qs, ns = g['q'].values.astype(int), g['n'].values.astype(int)
        crops = []
        T = len(g)
        bt, bm = np.zeros((T, 4, 4), np.float32), np.zeros((T, 4), np.float32)
        ws, hs = (g['x2'] - g['x1']).values, (g['y2'] - g['y1']).values
        for i_t, (p, q, cid) in enumerate(zip(g['crop_path'], qs, g['crop_id'].values)):
            if self.cache is not None and p in self.cache:
                a = self.cache[p].astype(np.float32) / 255.0
            else:
                a = load_crop(p, self.H, self.W)
            tb = self.boxes.get(int(cid)) if self.boxes is not None else None
            if tb is not None:
                bt[i_t], bm[i_t] = tb
            if q in (2, 3):                 # canonical orientation: mirror the image-right side
                a = np.ascontiguousarray(a[:, ::-1])
                bt[i_t] = flip_offsets(bt[i_t])
            if self.train:
                a, M = augment(a, rng, photo)
                if M is not None and bm[i_t].any():
                    fr = tooth_frame(max(ws[i_t], 1.0), max(hs[i_t], 1.0), q in (1, 2), self.H, self.W)
                    for d in np.flatnonzero(bm[i_t]):
                        bt[i_t, d] = warp_offsets(bt[i_t, d], fr, M)
            elif self.view is not None:
                a = cv2.warpAffine(a, self.view, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            crops.append((a - MEAN) / STD)
        return dict(x=np.stack(crops)[:, None].astype(np.float32), fdi=((qs - 1) * 8 + (ns - 1)).astype(np.int64),
                    twin=twin_index(qs, ns, g['gt_only'].values), rel=relations(qs, ns),
                    y=g[LABELS].values.astype(np.float32), ids=g['crop_id'].values.astype(np.int64),
                    gt_only=g['gt_only'].values.astype(np.int64), bt=bt, bm=bm * (1 - g['gt_only'].values[:, None]),
                    w=(g['w'].values if 'w' in g else np.ones(T)).astype(np.float32))


def collate(batch):
    B = len(batch)
    T = max(len(b['fdi']) for b in batch)
    H, W = batch[0]['x'].shape[-2:]
    x = np.zeros((B, T, 1, H, W), np.float32)
    fdi = np.zeros((B, T), np.int64)
    twin = np.full((B, T), -1, np.int64)
    rel = np.full((B, T, T), REL_OTHER, np.int64)
    y = np.zeros((B, T, 4), np.float32)
    ids = np.full((B, T), -1, np.int64)
    gto = np.ones((B, T), np.int64)
    mask = np.zeros((B, T), bool)
    bt, bm = np.zeros((B, T, 4, 4), np.float32), np.zeros((B, T, 4), np.float32)
    w = np.zeros((B, T), np.float32)
    for i, b in enumerate(batch):
        t = len(b['fdi'])
        x[i, :t], fdi[i, :t], twin[i, :t], rel[i, :t, :t] = b['x'], b['fdi'], b['twin'], b['rel']
        y[i, :t], ids[i, :t], gto[i, :t], mask[i, :t] = b['y'], b['ids'], b['gt_only'], True
        bt[i, :t], bm[i, :t], w[i, :t] = b['bt'], b['bm'], b['w']
    return {k: torch.from_numpy(v) for k, v in
            dict(x=x, fdi=fdi, twin=twin, rel=rel, y=y, ids=ids, gt_only=gto, mask=mask, bt=bt, bm=bm, w=w).items()}


class MixSampler(torch.utils.data.Sampler):
    """Self-training epochs: every annotated X-ray once plus `ratio` times as many pseudo-labelled ones, shuffled
    together and split evenly over the GPUs. The pseudo-labelled X-rays are taken in order from one random permutation
    of the pool after another, so all of them are used before any repeats."""
    def __init__(self, is_pool, ratio, world, rank, seed):
        is_pool = np.asarray(is_pool, bool)
        self.lab, self.pool = np.flatnonzero(~is_pool), np.flatnonzero(is_pool)
        n = len(self.pool) if ratio <= 0 else int(round(ratio * len(self.lab)))
        self.n_pool = min(len(self.pool), n)
        self.world, self.rank, self.seed, self.epoch = world, rank, seed, 0
        self.total = (len(self.lab) + self.n_pool) // world * world   # same number of X-rays on every GPU

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total // self.world

    def pool_part(self):
        out, start = [], self.epoch * self.n_pool
        while len(out) < self.n_pool:
            k, off = divmod(start + len(out), len(self.pool))
            perm = np.random.default_rng(self.seed + 7919 * (k + 1)).permutation(self.pool)
            out.extend(perm[off: off + self.n_pool - len(out)].tolist())
        return np.asarray(out, np.int64)

    def __iter__(self):
        idx = np.concatenate([self.lab, self.pool_part()]) if self.n_pool else self.lab.copy()
        np.random.default_rng(self.seed + 1000 * (self.epoch + 1)).shuffle(idx)   # identical on every rank
        return iter(idx[:self.total][self.rank::self.world].tolist())


# ----------------------------------------------------------------------------- model
class RelAttention(nn.Module):
    def __init__(self, dim, heads, drop=0.1):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Embedding(N_REL, heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.drop = nn.Dropout(drop)

    def forward(self, x, rel, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att + self.rel_bias(rel).permute(0, 3, 1, 2).to(att.dtype)
        att = att.masked_fill(~mask[:, None, None, :], torch.finfo(att.dtype).min)
        att = self.drop(att.softmax(-1))
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim, heads, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = RelAttention(dim, heads, drop)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * dim, dim))
        for lin in (self.attn.proj, self.mlp[-1]):   # block starts as identity
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, rel, mask):
        x = x + self.attn(self.n1(x), rel, mask)
        return x + self.mlp(self.n2(x))


def make_head(in_d, n_cls, drop, zero_last=False):
    head = nn.Sequential(nn.LayerNorm(in_d), nn.Dropout(drop), nn.Linear(in_d, 256), nn.GELU(),
                         nn.Dropout(drop), nn.Linear(256, n_cls))
    if zero_last:
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    return head


class TwinArch(nn.Module):
    def __init__(self, backbone, variant, pretrained=True, dim=256, layers=2, heads=4, n_cls=4, drop=0.2,
                 drop_path=0.0, box_src='token'):
        super().__init__()
        import timm
        self.variant = variant
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=1,
                                     drop_path_rate=drop_path)
        self.pad_to = 16   # encoder always sees 16, 32, 48 or 64 crops, so GPU kernels are set up once
        self.proj = nn.Sequential(nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.fdi = nn.Embedding(32, dim)
        nn.init.normal_(self.fdi.weight, std=0.02)
        use_diff = variant in ('twin', 'full')
        if use_diff:
            self.missing = nn.Parameter(torch.zeros(dim))
        if variant == 'full':
            self.diff_in = nn.Linear(dim + 1, dim)
            nn.init.zeros_(self.diff_in.weight)
            nn.init.zeros_(self.diff_in.bias)
        if variant in ('arch', 'full'):
            self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
            self.norm = nn.LayerNorm(dim)
        self.head = make_head(dim, n_cls, drop)
        self.n_cls = n_cls
        # learned boxes: edge offsets per disease, read off the tooth features WITHOUT changing them (detached),
        # so the diagnosis is trained exactly as without boxes. 'token' reads the pooled tooth token, 'map' the
        # encoder's spatial feature map (where in the crop things are), 'both' trains the two side by side.
        self.box_src = box_src
        self.box_head = self.box_map = None
        if box_src in ('token', 'both'):
            self.box_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, n_cls * 4))
            nn.init.zeros_(self.box_head[-1].weight)
            nn.init.zeros_(self.box_head[-1].bias)
        if box_src in ('map', 'both'):
            C = self.enc.num_features
            self.box_reduce = nn.Sequential(nn.Conv2d(C, 64, 1), nn.GELU(), nn.AdaptiveAvgPool2d((6, 4)), nn.Flatten())
            m_in = 64 * 6 * 4 + dim
            self.box_map = nn.Sequential(nn.LayerNorm(m_in), nn.Linear(m_in, 256), nn.GELU(), nn.Linear(256, n_cls * 4))
            nn.init.zeros_(self.box_map[-1].weight)
            nn.init.zeros_(self.box_map[-1].bias)
        if use_diff:   # asymmetry evidence is added on top of the crop evidence, starting at zero
            self.head_asym = make_head(dim + 1, n_cls, drop, zero_last=True)

    def forward(self, x, fdi, twin, rel, mask):
        B, T = mask.shape
        xs = x[mask]                                           # each tooth encoded once
        n = xs.shape[0]
        pad = (-n) % self.pad_to if self.pad_to > 1 else 0
        if pad:   # blank crops; ConvNeXt has no batch statistics, so they do not touch real teeth
            xs = torch.cat([xs, xs.new_zeros((pad,) + tuple(xs.shape[1:]))], 0)
        fm = self.enc.forward_features(xs)                      # [n(+pad), C, h, w]
        f = self.proj(self.enc.forward_head(fm, pre_logits=True)[:n])
        P = f.new_zeros(B, T, f.shape[-1])
        P[mask] = f
        tok = P + self.fdi(fdi).to(P.dtype)
        if self.variant in ('twin', 'full'):
            idx = twin.clamp(min=0).unsqueeze(-1).expand(-1, -1, P.shape[-1])
            has = (twin >= 0).to(P.dtype).unsqueeze(-1)
            diff = has * (P - torch.gather(P, 1, idx)).abs() + (1 - has) * self.missing.to(P.dtype)
            asym = torch.cat([diff, has], -1)
        if self.variant == 'full':
            tok = tok + self.diff_in(asym)
        if self.variant in ('arch', 'full'):
            for blk in self.blocks:
                tok = blk(tok, rel, mask)
            tok = self.norm(tok)
        out = self.head(tok)
        if self.variant in ('twin', 'full'):
            out = out + self.head_asym(asym)
        # detach: in v4 the box loss trained the encoder too and pulled the diagnosis down (AP50 64 -> 56)
        boxes = {}
        if self.box_head is not None:
            boxes['token'] = self.box_head(tok.detach()).view(B, T, self.n_cls, 4)
        if self.box_map is not None:
            sp = self.box_reduce(fm[:n].detach())
            S = sp.new_zeros(B, T, sp.shape[-1])
            S[mask] = sp
            boxes['map'] = self.box_map(torch.cat([S, tok.detach()], -1)).view(B, T, self.n_cls, 4)
        return out, boxes                                       # [B, T, 4], {source: [B, T, 4, 4]}


# ----------------------------------------------------------------------------- helpers
class WeightEMA:
    """running average of the weights; validation, checkpoint and predictions use the averaged copy"""
    def __init__(self, model, decay):
        import copy
        self.decay, self.n = decay, 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))   # short warm-up so early weights are not frozen in
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1 - d)
            else:
                v.copy_(src[k])


def amp_ctx(enabled):
    if enabled:
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def macro_ap(y, p):
    from sklearn.metrics import average_precision_score
    aps = [average_precision_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if y[:, k].sum() > 0]
    return (float(np.mean(aps)) if aps else 0.0), aps


def to_dev(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def predict(core, df, args, device, use_amp, cache=None, tta=1):
    """probabilities per tooth; with tta > 1 they are averaged over that many fixed views (boxes: identity view)"""
    out = predict_view(core, df, args, device, use_amp, cache, None)
    if tta > 1 and len(out[0]):
        ps = [out[1]]
        for view in TTA_VIEWS[1:tta]:
            o = predict_view(core, df, args, device, use_amp, cache, view)
            assert np.array_equal(o[0], out[0])
            ps.append(o[1])
        out = (out[0], np.mean(ps, 0), out[2], out[3])
    return out


@torch.no_grad()
def predict_view(core, df, args, device, use_amp, cache=None, view=None):
    ds = ArchDS(df, args.crop_h, args.crop_w, False, args.max_teeth, cache=cache, view=view)
    dl = DataLoader(ds, batch_size=args.img_bs * 2, shuffle=False, num_workers=args.workers, collate_fn=collate)
    core.eval()
    ids, probs, ys, boxes = [], [], [], {}
    for b in dl:
        b = to_dev(b, device)
        with amp_ctx(use_amp):
            logit, bxs = core(b['x'], b['fdi'], b['twin'], b['rel'], b['mask'])
        p = torch.sigmoid(logit.float())
        keep = b['mask'] & (b['gt_only'] == 0)
        q = (b['fdi'][keep].cpu().numpy() // 8) + 1
        flip = np.isin(q, (2, 3))
        for src, box in bxs.items():
            bx = box.float()[keep].cpu().numpy()                     # canonical frame
            bx[flip] = flip_offsets(bx[flip])                        # back to the original image frame
            boxes.setdefault(src, []).append(bx)
        ids.append(b['ids'][keep].cpu().numpy())
        probs.append(p[keep].cpu().numpy())
        ys.append(b['y'][keep].cpu().numpy())
    if not ids:
        return np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros((0, 4)), {}
    return (np.concatenate(ids), np.concatenate(probs), np.concatenate(ys),
            {k: np.concatenate(v) for k, v in boxes.items()})


def pred_frame(ids, probs, bxs):
    """one row per tooth: probabilities, and the learned box offsets (b* = token head, s* = spatial map head)"""
    cols = dict(crop_id=ids, p_caries=probs[:, 0], p_deep=probs[:, 1], p_peri=probs[:, 2], p_impacted=probs[:, 3])
    for src, pre in (('token', 'b'), ('map', 's')):
        if src in bxs:
            for d in range(4):
                for k, e in enumerate(('x1', 'y1', 'x2', 'y2')):
                    cols[f'{pre}{d}_{e}'] = bxs[src][:, d, k]
    return pd.DataFrame(cols)


def pred_keys_of(spec, args):
    name, kp = spec.split('=', 1)
    keys = sorted(read_keys(kp))
    return name, (keys[:args.max_images] if args.max_images else keys)


def run_predict_only(core, df, args, device, use_amp, rank, world, out):
    """score every --pred-keys set with the weights in --weights; each GPU takes every world-th X-ray"""
    core.load_state_dict(torch.load(args.weights, map_location=device))
    for spec in args.pred_keys:
        name, keys = pred_keys_of(spec, args)
        mine = set(keys[rank::world])
        dfp = df[df['image_key'].isin(mine) & (df['gt_only'] == 0)]
        t0 = time.time()
        ids, probs, _, bxs = predict(core, dfp, args, device, use_amp, None, tta=args.tta)
        part = out / (f'pred_{name}.part{rank}.csv' if world > 1 else f'pred_{name}.csv')
        pred_frame(ids, probs, bxs).to_csv(part, index=False)
        print(f'[train_arch] rank {rank}: {name} {len(ids)} teeth of {len(mine)} X-rays, {args.tta} view(s), '
              f'{time.time() - t0:.0f}s', flush=True)
    barrier(world)
    if rank == 0:
        for spec in args.pred_keys:
            name = spec.split('=', 1)[0]
            if world > 1:
                parts = [out / f'pred_{name}.part{r}.csv' for r in range(world)]
                pd.concat([pd.read_csv(q) for q in parts], ignore_index=True).to_csv(out / f'pred_{name}.csv', index=False)
                for q in parts:
                    q.unlink()
            print(f'[train_arch] wrote {out / f"pred_{name}.csv"}', flush=True)
        (out / 'log.json').write_text(json.dumps(dict(args=vars(args), predict_only=True), indent=1))


@torch.no_grad()
def profile(core, args, device, use_amp, n_teeth=28):
    """params, GFLOPs and latency for the classifier part on one X-ray with n_teeth teeth"""
    core.eval()
    pad_was, core.pad_to = core.pad_to, 1
    T = n_teeth
    x = torch.randn(1, T, 1, args.crop_h, args.crop_w, device=device)
    fdi = torch.arange(T, device=device).clamp(max=31).unsqueeze(0)
    twin = torch.full((1, T), -1, dtype=torch.long, device=device)
    rel = torch.full((1, T, T), REL_OTHER, dtype=torch.long, device=device)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    out = dict(params_M=sum(p.numel() for p in core.parameters()) / 1e6, teeth=T)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            core(x, fdi, twin, rel, mask)
        out['GFLOPs_per_xray'] = fc.get_total_flops() / 1e9
    except Exception as e:
        out['GFLOPs_per_xray'] = None
        out['flop_error'] = str(e)[:200]
    if device.type == 'cuda':
        for _ in range(3):
            with amp_ctx(use_amp):
                core(x, fdi, twin, rel, mask)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            with amp_ctx(use_amp):
                core(x, fdi, twin, rel, mask)
        torch.cuda.synchronize()
        out['ms_per_xray'] = (time.time() - t0) / 20 * 1000
    core.pad_to = pad_was
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = get_args(argv)
    rank, world, local = setup_ddp()
    cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{local}' if cuda else 'cpu')
    use_amp = cuda
    torch.backends.cudnn.benchmark = False   # batch shapes change every step (teeth per X-ray)
    seed_all(args.seed + rank)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    barrier(world)

    df = pd.read_csv(args.manifest)
    if 'w' in df:      # per-tooth loss weight: 1 = annotated, < 1 = the teacher's soft labels (self-training)
        df['w'] = df['w'].fillna(1.0)
    if not os.path.exists(df['crop_path'].iloc[0]):   # crops unpacked somewhere else: <manifest dir>/<split>/<file>
        base = Path(args.manifest).parent
        df['crop_path'] = [str(base / Path(p).parent.name / Path(p).name) for p in df['crop_path']]
    missing = [p for p in df['crop_path'].iloc[::max(1, len(df) // 200)] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} sampled crops are missing, e.g. {missing[0]}')
    kw = dict(pretrained=not (args.no_pretrained or args.predict_only), dim=args.dim, layers=args.layers,
              heads=args.heads, drop_path=args.drop_path, box_src=args.box_src)
    if args.predict_only:
        core = TwinArch(args.backbone, args.variant, **kw).to(device)
        run_predict_only(core, df, args, device, use_amp, rank, world, out)
        barrier(world)
        if world > 1:
            dist.destroy_process_group()
        return

    tr_keys, va_keys = read_keys(args.train_keys), read_keys(args.val_keys)
    if args.max_images:
        tr_keys = set(sorted(tr_keys)[:args.max_images])
        va_keys = set(sorted(va_keys)[:args.max_images])
    tr = df[df['image_key'].isin(tr_keys)]
    va = df[df['image_key'].isin(va_keys) & (df['gt_only'] == 0)]   # GT-only crops are for training only
    weighted = 'w' in tr and bool((tr['w'] < 1).any())
    lab = tr[tr['w'] >= 1] if weighted else tr       # class balance from the annotated teeth only
    pos = lab[LABELS].values.sum(0)
    pw = np.clip(np.sqrt((len(lab) - pos) / np.maximum(pos, 1)), 1.0, 10.0)
    if rank == 0:
        print(f'[train_arch] variant={args.variant} backbone={args.backbone} world={world} '
              f'train_xrays={lab.image_key.nunique()} teeth={len(lab)} val_xrays={va.image_key.nunique()} '
              f'pos={np.round(pos, 1).tolist()} pos_weight={np.round(pw, 2).tolist()}', flush=True)
        if weighted:
            pl = tr[tr['w'] < 1]
            print(f'[train_arch] + {pl.image_key.nunique()} pseudo-labelled X-rays ({len(pl)} teeth, weight '
                  f'{pl.w.mean():.2f}, mean soft label {np.round(pl[LABELS].values.mean(0), 3).tolist()}), '
                  f'{args.pool_ratio:g} per annotated X-ray each epoch', flush=True)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

    if rank == 0:
        model = TwinArch(args.backbone, args.variant, **kw)
    barrier(world)
    if rank != 0:
        model = TwinArch(args.backbone, args.variant, **kw)
    model = model.to(device)
    core = model
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None)
        core = model.module

    cache = None
    if not args.no_cache:
        t_c = time.time()
        keys_all = set(tr_keys) | set(va_keys)
        for spec in args.pred_keys:
            keys_all |= read_keys(spec.split('=', 1)[1])
        cdf = df[df['image_key'].isin(keys_all)]
        if 'w' in cdf:      # pseudo-labelled X-rays are too many for RAM: they are read from disk
            cdf = cdf[cdf['w'] >= 1]
        cache = preload(cdf['crop_path'].unique(), args.crop_h, args.crop_w)
        if rank == 0:
            print(f'[train_arch] cached {len(cache)} crops in RAM in {time.time() - t_c:.0f}s', flush=True)
    boxes = None
    if args.box_targets and os.path.exists(args.box_targets):
        bt_df = pd.read_csv(args.box_targets)
        tcols = [f't{d}_{k}' for d in range(4) for k in ('x1', 'y1', 'x2', 'y2')]
        mcols = [f'm{d}' for d in range(4)]
        boxes = {int(c): (t.reshape(4, 4).astype(np.float32), mk.astype(np.float32))
                 for c, t, mk in zip(bt_df['crop_id'], bt_df[tcols].values, bt_df[mcols].values)}
        if rank == 0:
            print(f'[train_arch] box targets for {len(boxes)} teeth', flush=True)
    ds_tr = ArchDS(tr, args.crop_h, args.crop_w, True, args.max_teeth, seed=args.seed + 100 * rank, cache=cache,
                   boxes=boxes)
    if any(ds_tr.is_pool):
        sampler = MixSampler(ds_tr.is_pool, args.pool_ratio, world, rank, args.seed)
        if rank == 0:
            print(f'[train_arch] each epoch: {len(sampler.lab)} annotated + {sampler.n_pool} pseudo-labelled X-rays',
                  flush=True)
    else:
        sampler = DistributedSampler(ds_tr, num_replicas=world, rank=rank, shuffle=True, seed=args.seed,
                                     drop_last=True) if world > 1 else None
    dl_tr = DataLoader(ds_tr, batch_size=args.img_bs, sampler=sampler, shuffle=sampler is None,
                       num_workers=args.workers, pin_memory=cuda, drop_last=True, collate_fn=collate,
                       persistent_workers=args.workers > 0)
    enc_ids = {id(p) for p in core.enc.parameters()}
    # box heads are clipped on their own: a large box gradient must not shrink the diagnosis update
    box_ids = {id(p) for nm in ('box_head', 'box_reduce', 'box_map') if getattr(core, nm, None) is not None
               for p in getattr(core, nm).parameters()}
    main_params = [p for p in core.parameters() if id(p) not in box_ids]
    box_params = [p for p in core.parameters() if id(p) in box_ids]
    opt = torch.optim.AdamW([{'params': list(core.enc.parameters()), 'lr': args.lr},
                             {'params': [p for p in core.parameters() if id(p) not in enc_ids], 'lr': args.head_lr}],
                            weight_decay=args.wd)
    steps, warm = max(1, args.epochs * len(dl_tr)), max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = make_scaler(use_amp)
    ema = WeightEMA(core, args.ema) if (args.ema > 0 and rank == 0) else None   # ranks hold identical weights

    best, log = -1.0, []
    for ep in range(args.epochs):
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        tot, nb = 0.0, 0
        for b in dl_tr:
            b = to_dev(b, device)
            with amp_ctx(use_amp):
                logits, boxes = model(b['x'], b['fdi'], b['twin'], b['rel'], b['mask'])
            m = b['mask']
            x, y = logits.float()[m], b['y'][m]
            if weighted:
                # annotated teeth: class-balanced BCE as before. Pseudo-labelled teeth: plain BCE on the teacher's
                # soft label (it is already class-balanced) times their weight w
                wm = b['w'][m]
                is_lab = (wm >= 1).float()[:, None]
                pwm = is_lab * pos_weight[None, :] + (1 - is_lab)
                bce = pwm * y * F.softplus(-x) + (1 - y) * F.softplus(x)
                loss = (bce.mean(-1) * wm).sum() / wm.sum().clamp(min=1e-6)
            else:
                loss = F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)
            bmk = b['bm'] * m.unsqueeze(-1).float()                     # only real teeth that have that disease
            for box in boxes.values():                                  # detached heads: no effect on diagnosis
                l1 = F.smooth_l1_loss(box.float(), b['bt'], beta=0.02, reduction='none').sum(-1)
                loss = loss + args.box_w * (l1 * bmk).sum() / bmk.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            if box_params:
                torch.nn.utils.clip_grad_norm_(box_params, 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale_before:   # a skipped step lowers the scale; keep the schedule in sync
                sched.step()
            if ema is not None:
                ema.update(core)
            tot += float(loss.detach())
            nb += 1
        if rank == 0:
            evm = ema.module if ema is not None else core
            ids, probs, ys, _ = predict(evm, va, args, device, use_amp, cache)
            mAP, per = macro_ap(ys, probs) if len(ids) else (0.0, [])
            if mAP > best:
                best = mAP
                torch.save(evm.state_dict(), out / 'best.pt')
            rec = dict(epoch=ep, loss=round(tot / max(nb, 1), 4), val_macro_ap=round(mAP, 4),
                       per_class=[round(v, 4) for v in per], sec=round(time.time() - t0, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)
        barrier(world)

    if rank == 0:
        if not (out / 'best.pt').exists():
            torch.save(core.state_dict(), out / 'best.pt')
        core.load_state_dict(torch.load(out / 'best.pt', map_location=device))
        prof = profile(core, args, device, use_amp)
        print('[train_arch] efficiency', json.dumps(prof), flush=True)
        for spec in args.pred_keys:
            name, keys = pred_keys_of(spec, args)
            dfp = df[df['image_key'].isin(set(keys)) & (df['gt_only'] == 0)]
            ids, probs, _, bxs = predict(core, dfp, args, device, use_amp, cache, tta=args.tta)
            pred_frame(ids, probs, bxs).to_csv(out / f'pred_{name}.csv', index=False)
            print(f'[train_arch] wrote {out / f"pred_{name}.csv"} rows={len(ids)}', flush=True)
        (out / 'log.json').write_text(json.dumps(dict(args=vars(args), best_val_macro_ap=best, efficiency=prof,
                                                      log=log), indent=1))
    barrier(world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
