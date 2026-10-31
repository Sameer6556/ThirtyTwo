"""The FastAPI app with the real models: endpoints, limits, formats, in-memory uploads, one prediction at a time."""
import io
import tempfile
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import needs_assets
from thirtytwo import server

pytestmark = needs_assets


@pytest.fixture(scope='module')
def client():
    with TestClient(server.app) as c:
        t0 = time.time()
        while c.get('/api/health').json()['status'] == 'loading' and time.time() - t0 < 600:
            time.sleep(0.5)
        assert c.get('/api/health').json()['status'] == 'ok'
        yield c


def post(client, data, mode='quick', name='x.png', ctype='image/png'):
    return client.post('/api/predict', files={'image': (name, data, ctype)}, data={'mode': mode})


def test_health_and_pages(client):
    h = client.get('/api/health').json()
    assert h['status'] == 'ok' and h['default_mode'] in ('quick', 'full') and h['max_upload_mb'] == 20
    page = client.get('/')
    assert page.status_code == 200 and 'Thirty Two' in page.text and 'id="drop"' in page.text
    assert "default-src 'self'" in page.headers['content-security-policy']
    for path, heading in (('/how-it-works', 'How it works'), ('/accuracy', 'Accuracy'), ('/about', 'About')):
        r = client.get(path)
        assert r.status_code == 200 and f'<h1>{heading}</h1>' in r.text and 'aria-current="page"' in r.text
        assert "default-src 'self'" in r.headers['content-security-policy']
    assert 'Sameer' not in client.get('/about').text          # the About page is about the project


@pytest.mark.parametrize('mode', ['quick', 'full'])
def test_predict(client, sample_bytes, mode):
    r = post(client, sample_bytes, mode)
    assert r.status_code == 200 and r.headers['cache-control'] == 'no-store'
    j = r.json()
    assert j['mode'] == mode and j['image'] == {'width': 2728, 'height': 1316} and j['warning'] is None
    assert len(j['teeth']) >= 10 and j['findings'] and j['time_s'] > 0
    assert set(j['thresholds']) == {'caries', 'deep_caries', 'periapical_lesion', 'impacted'}
    f = j['findings'][0]
    assert set(f) == {'id', 'class', 'name', 'code', 'score', 'box', 'tooth'}
    assert f['class'] in j['thresholds'] and f['code'] in ('C', 'DC', 'PL', 'IT') and len(f['box']) == 4


def test_jpeg_and_dicom(client, sample_bytes):
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    gray = cv2.imdecode(np.frombuffer(sample_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    ok, jpg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert post(client, jpg.tobytes(), name='x.jpg', ctype='image/jpeg').status_code == 200
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
    r = post(client, buf.getvalue(), name='x.dcm', ctype='application/dicom')
    assert r.status_code == 200 and len(r.json()['teeth']) >= 10


def test_uploads_stay_in_memory(client, sample_bytes, monkeypatch):
    """a 3.5 MB upload must not roll over to a temporary file (Starlette's default spools above 1 MB)"""
    def boom(*a, **k):
        raise AssertionError('an upload was about to be written to disk')
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, 'rollover', boom)
    monkeypatch.setattr(tempfile, 'mkstemp', boom)
    monkeypatch.setattr(tempfile, 'NamedTemporaryFile', boom)
    assert len(sample_bytes) > 3_000_000
    assert post(client, sample_bytes).status_code == 200


def test_rejections(client, sample_bytes):
    big = b'\x89PNG\r\n\x1a\n' + bytes(21 * 1024 * 1024)
    assert post(client, big).status_code == 413
    assert post(client, b'just some text', name='x.txt', ctype='text/plain').status_code == 415
    assert post(client, sample_bytes, mode='fast').status_code == 422
    assert client.post('/api/predict', data={'mode': 'quick'}).status_code in (400, 415)
    assert client.post('/api/predict', content=sample_bytes, headers={'content-type': 'image/png'}).status_code == 415
    assert post(client, b'\x89PNG\r\n\x1a\n' + b'broken').status_code == 422


def test_not_a_panoramic(client, sample_bytes):
    gray = cv2.imdecode(np.frombuffer(sample_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    ok, png = cv2.imencode('.png', cv2.resize(gray, (800, 800)))
    j = post(client, png.tobytes()).json()
    assert j['warning'] and j['findings'] == [] and 'panoramic' in j['warning']


def test_one_prediction_at_a_time(client, sample_bytes, monkeypatch):
    """requests queue behind one lock; beyond MAX_QUEUE waiting ones the server answers 503"""
    real = server.S.model
    active, peak = [0], [0]

    class Slow:
        device = real.device

        def predict(self, data, mode):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            time.sleep(0.3)
            active[0] -= 1
            return real.predict(np.zeros((500, 1000), np.uint8), mode)   # wrong shape: returns at once

    monkeypatch.setattr(server.S, 'model', Slow())
    monkeypatch.setattr(server, 'MAX_QUEUE', 3)
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(post(client, sample_bytes).status_code)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
    assert codes.count(200) >= 3 and set(codes) <= {200, 503}
