"""The web app: POST /api/predict, GET /api/health, and the static frontend.

    uvicorn thirtytwo.server:app --host 0.0.0.0 --port 7860

Uploads stay in memory: the body is read with a size cap and parsed by a multipart parser whose buffer never rolls
over to a temporary file, and nothing about an upload is logged. One prediction runs at a time; up to MAX_QUEUE more
wait in line, after that the server answers 503.
"""
import asyncio
import base64
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from statistics import median

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException, MultiPartParser

from . import __version__
from .images import ImageError, sniff

MAX_BYTES = 20 * 1024 * 1024
MAX_BODY = MAX_BYTES + 64 * 1024                   # room for the multipart framing and the mode field
MAX_QUEUE = int(os.environ.get('THIRTYTWO_MAX_QUEUE', '4'))
DEFAULT_MODE = os.environ.get('THIRTYTWO_DEFAULT_MODE', 'quick')
EXPECTED_S = {'full': 10.0, 'quick': 1.0}          # 10.5 s and 1.2 s on average on a 4-core CPU (README); replaced
                                                   # by this server's own recent runs once it has some
STATIC = Path(__file__).resolve().parent / 'static'
log = logging.getLogger('thirtytwo')


class InMemoryMultiPartParser(MultiPartParser):
    """Starlette's parser spools file parts to disk above 1 MB; this one keeps them in memory up to the body cap"""
    spool_max_size = MAX_BODY + 1


class TooLarge(Exception):
    pass


class State:
    def __init__(self):
        self.model = None
        self.error = None
        self.lock = asyncio.Lock()          # one prediction at a time
        self.waiting = 0                    # requests holding or waiting for the lock
        self.times = {'full': [], 'quick': []}
        self.warm = set()                   # modes that have run once (a first, cold run is not a typical time)

    def expected(self, mode):
        t = self.times[mode][-20:]
        return round(median(t), 1) if t else EXPECTED_S[mode]


S = State()


def _load():
    try:
        from .pipeline import ThirtyTwo
        t0 = time.time()
        model = ThirtyTwo()
        sample = STATIC / 'samples' / 'val_14.png'
        if sample.is_file():                # one Quick run while starting, so no visitor waits for the warm-up
            try:
                model.predict(sample.read_bytes(), 'quick')
                S.warm.add('quick')
            except Exception:
                log.exception('thirtytwo: warm-up failed')
        S.model = model
        log.warning('thirtytwo: models loaded in %.0fs on %s', time.time() - t0, S.model.device)
    except Exception as e:                  # reported by /api/health
        S.error = f'{type(e).__name__}: {e}'
        log.exception('thirtytwo: could not load the models')


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load, daemon=True).start()   # health answers while the models load
    yield


app = FastAPI(title='Thirty Two', version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
              openapi_url=None)


@app.middleware('http')
async def headers(request: Request, call_next):
    response = await call_next(request)
    # the page may only talk to this server: no outside scripts, fonts, images or requests
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'")
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


def error(status, message):
    return JSONResponse({'error': message}, status_code=status)


@app.get('/api/health')
async def health():
    status = 'ok' if S.model is not None else ('error' if S.error else 'loading')
    return dict(status=status, version=__version__, device=str(S.model.device) if S.model else None,
                default_mode=DEFAULT_MODE, max_upload_mb=MAX_BYTES // (1024 * 1024), queue=S.waiting,
                expected_s={m: S.expected(m) for m in ('full', 'quick')}, error=S.error)


async def _read_body(request: Request):
    n = int(request.headers.get('content-length') or 0)
    if n > MAX_BODY:
        raise TooLarge()
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY:
            raise TooLarge()
        yield chunk


@app.post('/api/predict')
async def predict(request: Request):
    if not request.headers.get('content-type', '').startswith('multipart/form-data'):
        return error(415, 'Send the X-ray as multipart/form-data with an "image" field.')
    try:
        form = await InMemoryMultiPartParser(request.headers, _read_body(request), max_files=1, max_fields=4).parse()
    except TooLarge:
        return error(413, 'The file is larger than 20 MB.')
    except MultiPartException as e:
        return error(400, f'Could not read the upload: {e.message}')
    upload, mode = form.get('image'), str(form.get('mode') or DEFAULT_MODE)
    if upload is None or isinstance(upload, str):
        return error(400, 'No file in the "image" field.')
    if mode not in ('full', 'quick'):
        return error(422, 'mode must be "full" or "quick".')
    data = await upload.read()
    await upload.close()
    if len(data) > MAX_BYTES:
        return error(413, 'The file is larger than 20 MB.')
    if sniff(data) is None:
        return error(415, 'Use a JPEG, PNG or DICOM file.')
    if S.model is None:
        return error(503, 'The models are still loading. Try again in a minute.' if not S.error
                     else 'The models could not be loaded.')
    if S.waiting > MAX_QUEUE:                # one running + MAX_QUEUE waiting
        return error(503, 'The server is busy. Try again in a minute.')
    S.waiting += 1
    try:
        async with S.lock:
            from .pipeline import to_json
            t0 = time.perf_counter()
            result = await run_in_threadpool(S.model.predict, data, mode)
            if not result['warning']:
                if mode in S.warm:
                    S.times[mode].append(time.perf_counter() - t0)
                S.warm.add(mode)
    except ImageError as e:
        return error(422, f'Could not read the image: {e}.')
    except Exception:
        log.exception('thirtytwo: prediction failed')          # the traceback holds no image data
        return error(500, 'Something went wrong while reading this X-ray.')
    finally:
        S.waiting -= 1
    out = to_json(result)
    if sniff(data) == 'DICOM':               # browsers cannot show DICOM: send the 1280 export the detectors read
        from .images import decode_upload, resize_long
        jpeg, _ = resize_long(decode_upload(data), 1280)
        out['preview'] = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()
    del data
    return out


def _page(name):
    async def page():
        return FileResponse(STATIC / f'{name}.html')
    return page


for _name in ('how-it-works', 'accuracy', 'about'):          # the three text pages, at clean URLs
    app.add_api_route(f'/{_name}', _page(_name), methods=['GET'], include_in_schema=False)

app.mount('/', StaticFiles(directory=STATIC, html=True), name='static')
