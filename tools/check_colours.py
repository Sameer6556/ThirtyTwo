"""Are the four class colours distinguishable? CIEDE2000 between every pair, for normal vision and simulated
protanopia, deuteranopia and tritanopia (Machado, Oliveira & Fernandes 2009, severity 1), plus WCAG contrast of each
colour on the viewer's near-black and of the label text (#111) on each colour.

    python tools/check_colours.py
"""
import itertools
import math

import numpy as np

COLOURS = {'caries': '#E69F00', 'deep caries': '#CC79A7', 'periapical lesion': '#56B4E9', 'impacted tooth': '#F0E442'}
VIEWER, LABEL_TEXT = '#0E0F10', '#111111'
MACHADO = {
    'protanopia': [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    'deuteranopia': [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    'tritanopia': [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def srgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def lab(c):
    x, y, z = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ to_lin(c)
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    fx, fy, fz = f(x / 0.95047), f(y), f(z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def de2000(l1, l2):
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    G = 0.5 * (1 - math.sqrt(((C1 + C2) / 2) ** 7 / (((C1 + C2) / 2) ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p, h2p = math.degrees(math.atan2(b1, a1p)) % 360, math.degrees(math.atan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    dh = 0 if C1p * C2p == 0 else (h2p - h1p if abs(h2p - h1p) <= 180 else h2p - h1p - 360 * math.copysign(1, h2p - h1p))
    dH = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh / 2))
    Lm, Cm = (L1 + L2) / 2, (C1p + C2p) / 2
    hm = (h1p + h2p) / 2 if abs(h1p - h2p) <= 180 else (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    if C1p * C2p == 0:
        hm = h1p + h2p
    T = 1 - 0.17 * math.cos(math.radians(hm - 30)) + 0.24 * math.cos(math.radians(2 * hm)) \
        + 0.32 * math.cos(math.radians(3 * hm + 6)) - 0.20 * math.cos(math.radians(4 * hm - 63))
    SL = 1 + 0.015 * (Lm - 50) ** 2 / math.sqrt(20 + (Lm - 50) ** 2)
    SC, SH = 1 + 0.045 * Cm, 1 + 0.015 * Cm * T
    RT = -2 * math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)) * math.sin(math.radians(60 * math.exp(-((hm - 275) / 25) ** 2)))
    return math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH))


def luminance(c):
    return float(np.array([0.2126, 0.7152, 0.0722]) @ to_lin(c))


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def main():
    worst = {}
    for vision in ['normal'] + list(MACHADO):
        cols = {k: srgb(v) for k, v in COLOURS.items()}
        if vision != 'normal':
            cols = {k: to_srgb(np.array(MACHADO[vision]) @ to_lin(c)) for k, c in cols.items()}
        pairs = [(a, b, de2000(lab(cols[a]), lab(cols[b]))) for a, b in itertools.combinations(cols, 2)]
        a, b, d = min(pairs, key=lambda t: t[2])
        worst[vision] = d
        print(f'{vision:12s} closest pair: {a} / {b}, CIEDE2000 {d:5.1f}   (all: '
              + ', '.join(f'{x[:4]}-{y[:4]} {z:.0f}' for x, y, z in pairs) + ')')
    for k, v in COLOURS.items():
        print(f'{k:18s} {v}: contrast on the viewer {contrast(srgb(v), srgb(VIEWER)):4.1f}:1, '
              f'label text {LABEL_TEXT} on it {contrast(srgb(LABEL_TEXT), srgb(v)):4.1f}:1')
    return worst


if __name__ == '__main__':
    main()
