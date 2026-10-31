"""Uploads in memory: decoding, and the notebook's JPEG export without touching the disk.

read_gray / resize_long below copy dentex_utils.read_gray and dentex_utils.resize_long step for step, with
cv2.imdecode / cv2.imencode in place of the file reads and writes. The JPEG bytes are then decoded the way ultralytics
reads an image file (cv2.imdecode with IMREAD_COLOR), because the detectors only ever saw those JPEGs.
tests/test_parity.py checks the bytes against resize_long itself at both long sides.
"""
import io

import cv2
import numpy as np

from . import config as C

FORMATS = ('JPEG', 'PNG', 'DICOM')


class ImageError(ValueError):
    """the upload is not an image we can read"""


def sniff(data: bytes):
    """'JPEG', 'PNG', 'DICOM' or None, from the first bytes"""
    if data[:3] == b'\xff\xd8\xff':
        return 'JPEG'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'PNG'
    if len(data) > 132 and data[128:132] == b'DICM':
        return 'DICOM'
    return None


def read_gray(data: bytes) -> np.ndarray:
    """dentex_utils.read_gray on bytes: cv2.imread(path, IMREAD_GRAYSCALE) -> cv2.imdecode(bytes, IMREAD_GRAYSCALE)"""
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ImageError('could not decode the image')
    return im


def read_dicom(data: bytes) -> np.ndarray:
    """a DICOM file as an 8-bit grey image (not part of the notebook, which only saw PNGs): modality and VOI LUT as
    the file asks, scaled to 0-255, MONOCHROME1 inverted"""
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        a = ds.pixel_array
    except Exception as e:     # compressed transfer syntaxes without a decoder, broken files
        raise ImageError(f'could not read this DICOM file ({type(e).__name__})') from e
    if a.ndim == 3 and a.shape[-1] in (3, 4) and str(getattr(ds, 'PhotometricInterpretation', '')).startswith(
            ('RGB', 'YBR')):
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif a.ndim == 3:           # several frames: the first one
        a = a[0]
    a = apply_voi_lut(apply_modality_lut(a, ds), ds).astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / max(hi - lo, 1e-9) * 255.0
    if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        a = 255.0 - a
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def decode_upload(data: bytes) -> np.ndarray:
    """the original X-ray as a 2-D uint8 grey image"""
    kind = sniff(data)
    if kind is None:
        raise ImageError('not a JPEG, PNG or DICOM file')
    gray = read_dicom(data) if kind == 'DICOM' else read_gray(data)
    if gray.ndim != 2 or min(gray.shape) < 32:
        raise ImageError('the image is too small')
    return gray


def resize_long(gray: np.ndarray, long_side: int, quality: int = C.JPEG_QUALITY):
    """dentex_utils.resize_long in memory. Returns (JPEG bytes, (W, H, scale)) like resize_long's (W, H, scale)."""
    im = gray
    h, w = im.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1.0:
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageError('could not encode the export')
    return buf.tobytes(), (w, h, s)


def decode_like_ultralytics(jpeg: bytes) -> np.ndarray:
    """what ultralytics.utils.patches.imread returns for that JPEG file: a BGR image"""
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def export(gray: np.ndarray, long_side: int):
    """(BGR image the detectors read, scale from original to export pixels)"""
    jpeg, (_, _, s) = resize_long(gray, long_side)
    return decode_like_ultralytics(jpeg), s
