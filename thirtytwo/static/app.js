// Thirty Two: upload, call /api/predict, draw the findings. No libraries.
'use strict';

const CLASSES = [
  { key: 'caries', name: 'Caries', code: 'C', colour: '#E69F00' },
  { key: 'deep_caries', name: 'Deep caries', code: 'DC', colour: '#CC79A7' },
  { key: 'periapical_lesion', name: 'Periapical lesion', code: 'PL', colour: '#56B4E9' },
  { key: 'impacted', name: 'Impacted tooth', code: 'IT', colour: '#F0E442' },
];
const BY_KEY = Object.fromEntries(CLASSES.map((c) => [c.key, c]));
const $ = (id) => document.getElementById(id);

const S = {
  file: null, name: '', url: null, results: {}, measured: {}, mode: 'quick', expected: { full: 12, quick: 1.5 },
  shown: Object.fromEntries(CLASSES.map((c) => [c.key, true])), factor: 1, allTeeth: false,
  hover: null, selected: null, view: { k: 1, x: 0, y: 0, fit: 1 }, busy: false,
};

// ------------------------------------------------------------------ helpers
function show(el, on) { el.hidden = !on; }
function plural(n, one, many) { return `${n} ${n === 1 ? one : many}`; }
function seconds(s) { return s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`; }
function current() { return S.results[S.mode]; }
function message(text) { $('message').textContent = text || ''; show($('message'), !!text); }

function visible() {
  const r = current();
  if (!r) return [];
  const thr = r.thresholds || {};
  return r.findings.filter((f) => S.shown[f.class] && f.score >= (thr[f.class] ?? 0.3) * S.factor);
}

// ------------------------------------------------------------------ upload
async function health() {
  try {
    const h = await (await fetch('/api/health')).json();
    if (h.expected_s) S.expected = h.expected_s;
    if (!S.file && (h.default_mode === 'full' || h.default_mode === 'quick')) S.mode = h.default_mode;
  } catch (e) { /* the page still works; predict will report the problem */ }
  document.querySelectorAll('input[name="mode"]').forEach((i) => { i.checked = i.value === S.mode; });
  modeLabels();
}

function useFile(file, name) {
  if (S.busy) return;
  message('');
  if (file.size > 20 * 1024 * 1024) { message('The file is larger than 20 MB.'); return; }
  if (S.url) URL.revokeObjectURL(S.url);
  S.file = file; S.name = name || file.name || 'x-ray'; S.url = URL.createObjectURL(file);
  S.results = {}; S.measured = {}; S.selected = null; S.hover = null;
  analyse(S.mode);
}

async function analyse(mode) {
  S.busy = true;
  const previous = S.mode;
  // switching modes on a shown X-ray: the results stay, the progress line sits in the mode block (where the user is)
  const switching = !!S.results[previous] && !$('results').hidden;
  S.mode = mode;
  const wait = S.expected[mode] || 5;
  const text = mode === 'full'
    ? `Running the full system. This usually takes about ${seconds(wait)}.`
    : `Running Quick mode. This usually takes about ${seconds(wait)}.`;
  const box = switching ? $('mode-progress') : $('progress');
  const fill = switching ? $('mode-fill') : $('progress-fill');
  if (switching) {
    $('mode-hint').textContent = text;
    document.querySelectorAll('input[name="mode"]').forEach((i) => { i.disabled = true; });
  } else {
    show($('drop'), false); show($('results'), false); show($('warning'), false);
    $('progress-text').textContent = text;
  }
  fill.style.transition = 'none'; fill.style.width = '0';
  show(box, true);
  void fill.offsetWidth;
  fill.style.transition = `width ${Math.max(wait, 0.5)}s linear`; fill.style.width = '90%';
  const body = new FormData();
  body.append('image', S.file, S.name);
  body.append('mode', mode);
  let r;
  try {
    const res = await fetch('/api/predict', { method: 'POST', body });
    r = await res.json();
    if (!res.ok) throw new Error(r.error || `The server answered ${res.status}.`);
  } catch (e) {
    S.busy = false;
    show(box, false);
    document.querySelectorAll('input[name="mode"]').forEach((i) => { i.disabled = false; });
    const text = e.message && !e.message.startsWith('Failed to fetch') && !e.message.includes('JSON')
      ? e.message : 'The server could not be reached. Try again in a moment.';
    if (S.results[previous]) {
      S.mode = previous;
      show($('results'), true);
      syncMode();
      $('mode-hint').textContent = text;
    } else {
      show($('drop'), true);
      message(text);
    }
    return;
  }
  fill.style.transition = 'width 0.2s'; fill.style.width = '100%';
  S.busy = false;
  S.results[mode] = r;
  S.measured[mode] = r.time_s;
  show(box, false);
  document.querySelectorAll('input[name="mode"]').forEach((i) => { i.disabled = false; });
  if (r.warning) {
    $('warning-text').textContent = r.warning;
    show($('warning'), true);
    return;
  }
  await showResults();
}

// ------------------------------------------------------------------ results
async function showResults() {
  const r = current();
  const img = $('xray');
  const src = r.preview || S.url;            // DICOM: the server sends a JPEG preview
  const fresh = img.dataset.src !== src;
  if (fresh) {
    img.dataset.src = src;
    img.src = src;
    try { await img.decode(); } catch (e) { /* drawn when it loads */ }
    img.style.width = `${img.naturalWidth}px`;
    img.style.height = `${img.naturalHeight}px`;
  }
  show($('results'), true);
  syncMode();
  if (fresh) fit();                          // a mode switch keeps the current zoom
  panel();
}

function syncMode() {
  document.querySelectorAll('input[name="mode"]').forEach((i) => { i.checked = i.value === S.mode; });
  modeLabels();
}

function modeLabels() {
  for (const m of ['quick', 'full']) {
    $(`time-${m}`).textContent = S.measured[m] != null ? seconds(S.measured[m]) : `about ${seconds(S.expected[m])}`;
  }
  $('mode-hint').textContent = S.mode === 'quick'
    ? `Full mode runs all nine models and takes about ${seconds(S.measured.full ?? S.expected.full)}.`
    : 'Quick mode runs one detector and is many times faster.';
}

function panel() {
  const r = current();
  const vis = visible();
  const teeth = new Set(vis.filter((f) => f.tooth != null).map((f) => f.tooth));
  $('summary').textContent = vis.length
    ? `${plural(vis.length, 'finding', 'findings')} on ${plural(teeth.size, 'tooth', 'teeth')}`
    : 'No findings at this sensitivity';
  const groups = $('groups');
  groups.replaceChildren();
  for (const c of CLASSES) {
    const fs = vis.filter((f) => f.class === c.key);
    if (!fs.length) continue;
    const g = document.createElement('div');
    g.className = 'group';
    const h = document.createElement('h3');
    h.append(swatch(c.colour), `${c.name} (${fs.length})`);
    const ul = document.createElement('ul');
    for (const f of fs) {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'row';
      b.dataset.id = f.id;
      const name = document.createElement('span');
      name.textContent = c.name;
      const tooth = document.createElement('span');
      tooth.className = 'tooth';
      tooth.textContent = f.tooth != null ? `Tooth ${f.tooth}` : 'No tooth number';
      const pct = document.createElement('span');
      pct.className = 'pct num';
      pct.textContent = `${Math.round(f.score * 100)}%`;
      b.append(swatch(c.colour), name, tooth, pct);
      b.setAttribute('aria-label', `${c.name}, ${tooth.textContent.toLowerCase()}, ${pct.textContent} confidence`);
      b.addEventListener('mouseenter', () => { S.hover = f.id; draw(); });
      b.addEventListener('mouseleave', () => { S.hover = null; draw(); });
      b.addEventListener('focus', () => { S.hover = f.id; draw(); });
      b.addEventListener('blur', () => { S.hover = null; draw(); });
      b.addEventListener('click', () => { S.selected = S.selected === f.id ? null : f.id; markRows(); draw(); });
      li.append(b);
      ul.append(li);
    }
    g.append(h, ul);
    groups.append(g);
  }
  if (!vis.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = r.findings.length ? 'Move the sensitivity towards "More findings" to see weaker ones.' : '';
    if (p.textContent) groups.append(p);
  }
  markRows();
  draw();
}

function swatch(colour) {
  const s = document.createElement('span');
  s.className = 'swatch';
  s.style.background = colour;
  return s;
}

function markRows() {
  document.querySelectorAll('.row').forEach((b) => b.classList.toggle('on', Number(b.dataset.id) === S.selected));
}

function toggles() {
  const box = $('toggles');
  for (const c of CLASSES) {
    const l = document.createElement('label');
    const i = document.createElement('input');
    i.type = 'checkbox';
    i.checked = true;
    i.addEventListener('change', () => { S.shown[c.key] = i.checked; panel(); });
    l.append(i, swatch(c.colour), c.name);
    box.append(l);
  }
}

// ------------------------------------------------------------------ viewer
const viewer = $('viewer');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');

function scale() {                                  // displayed image pixels per original pixel
  const r = current();
  return r ? $('xray').naturalWidth / r.image.width : 1;
}

function fit() {
  const img = $('xray');
  const w = viewer.clientWidth;
  const h = viewer.clientHeight;
  if (!img.naturalWidth || !w) return;
  const k = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  S.view = { k, fit: k, x: (w - img.naturalWidth * k) / 2, y: (h - img.naturalHeight * k) / 2 };
  apply();
}

function apply() {
  const v = S.view;
  $('xray').style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.k})`;
  draw();
}

function zoomAt(px, py, factor) {
  const v = S.view;
  const k = Math.min(Math.max(v.k * factor, v.fit * 0.5), v.fit * 16);
  const f = k / v.k;
  v.x = px - (px - v.x) * f;
  v.y = py - (py - v.y) * f;
  v.k = k;
  apply();
}

function filters() {
  $('xray').style.filter = `brightness(${$('brightness').value}%) contrast(${$('contrast').value}%)`;
}

function label(c, text, x, y, colour, size) {
  c.font = `600 ${size}px system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif`;
  const w = c.measureText(text).width + size * 0.6;
  const h = size * 1.45;
  const ty = y - h >= 0 ? y - h : y;
  c.fillStyle = colour;
  c.fillRect(x, ty, w, h);
  c.fillStyle = '#111111';
  c.textBaseline = 'middle';
  c.fillText(text, x + size * 0.3, ty + h / 2 + 0.5);
}

// draws the boxes onto a 2D context; `to` maps original pixels to that context's pixels
function paint(c, to, opts) {
  const r = current();
  if (!r) return;
  if (S.allTeeth) {
    c.lineWidth = opts.thin;
    for (const t of r.teeth) {
      const [x, y, w, h] = to(t.box);
      c.strokeStyle = 'rgba(250, 250, 249, 0.35)';
      c.strokeRect(x, y, w, h);
      c.font = `500 ${opts.small}px system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif`;
      c.fillStyle = 'rgba(250, 250, 249, 0.7)';
      c.textBaseline = 'top';
      c.fillText(String(t.fdi), x + 3, y + 2);
    }
  }
  const hl = opts.highlight;
  const vis = visible().sort((a, b) => (a.id === hl) - (b.id === hl));
  for (const f of vis) {
    const cls = BY_KEY[f.class];
    const [x, y, w, h] = to(f.box);
    c.globalAlpha = hl != null && f.id !== hl ? 0.35 : 1;
    c.lineWidth = f.id === hl ? opts.line * 2 : opts.line;
    c.strokeStyle = cls.colour;
    c.strokeRect(x, y, w, h);
    label(c, f.tooth != null ? `${f.tooth} ${cls.code}` : cls.code, x, y, cls.colour, opts.font);
  }
  c.globalAlpha = 1;
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = viewer.clientWidth;
  const h = viewer.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const v = S.view;
  const s = scale() * v.k;
  paint(ctx, (b) => [v.x + b[0] * s, v.y + b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s],
    { line: 1.5, thin: 1, font: 11, small: 10, highlight: S.hover ?? S.selected });
}

function hit(px, py) {                              // the smallest visible box under the point
  const v = S.view;
  const s = scale() * v.k;
  const x = (px - v.x) / s;
  const y = (py - v.y) / s;
  let best = null;
  for (const f of visible()) {
    const [x1, y1, x2, y2] = f.box;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      const a = (x2 - x1) * (y2 - y1);
      if (!best || a < best.a) best = { f, a };
    }
  }
  return best && best.f;
}

const pointers = new Map();
let moved = 0;
let pinch = null;
viewer.addEventListener('pointerdown', (e) => {
  viewer.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
  moved = 0;
  viewer.classList.add('dragging');
});
viewer.addEventListener('pointermove', (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const q = { x: e.offsetX, y: e.offsetY };
  if (pointers.size === 1) {
    S.view.x += q.x - p.x;
    S.view.y += q.y - p.y;
    moved += Math.abs(q.x - p.x) + Math.abs(q.y - p.y);
    pointers.set(e.pointerId, q);
    apply();
    return;
  }
  pointers.set(e.pointerId, q);
  const [a, b] = [...pointers.values()];
  const d = Math.hypot(a.x - b.x, a.y - b.y);
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  if (pinch) {
    S.view.x += mx - pinch.mx;
    S.view.y += my - pinch.my;
    zoomAt(mx, my, d / pinch.d);
  }
  pinch = { d: Math.max(d, 1), mx, my };
  moved += 10;
});
function release(e) {
  const tap = pointers.size === 1 && moved < 4;
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = null;
  if (!pointers.size) viewer.classList.remove('dragging');
  if (tap && e.type === 'pointerup') {
    const f = hit(e.offsetX, e.offsetY);
    S.selected = f ? f.id : null;
    markRows();
    draw();
    if (f) {
      const row = document.querySelector(`.row[data-id="${f.id}"]`);
      if (row) { row.scrollIntoView({ block: 'nearest' }); row.focus({ preventScroll: true }); }
    }
  }
}
viewer.addEventListener('pointerup', release);
viewer.addEventListener('pointercancel', release);
viewer.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
}, { passive: false });
viewer.addEventListener('keydown', (e) => {
  const w = viewer.clientWidth / 2;
  const h = viewer.clientHeight / 2;
  const moves = { ArrowLeft: [40, 0], ArrowRight: [-40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] };
  if (e.key === '+' || e.key === '=') zoomAt(w, h, 1.25);
  else if (e.key === '-' || e.key === '_') zoomAt(w, h, 0.8);
  else if (e.key === '0') fit();
  else if (moves[e.key]) { S.view.x += moves[e.key][0]; S.view.y += moves[e.key][1]; apply(); }
  else return;
  e.preventDefault();
});
new ResizeObserver(() => { if (current() && !$('results').hidden) fit(); }).observe(viewer);

// ------------------------------------------------------------------ downloads
function save(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function baseName() { return S.name.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '-') || 'x-ray'; }

$('dl-png').addEventListener('click', () => {
  const img = $('xray');
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const x = c.getContext('2d');
  x.filter = img.style.filter || 'none';
  x.drawImage(img, 0, 0);
  x.filter = 'none';
  const s = scale();
  const u = Math.max(1, c.width / 1400);
  paint(x, (b) => [b[0] * s, b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s],
    { line: 1.5 * u, thin: u, font: Math.round(11 * u * 1.3), small: Math.round(10 * u * 1.3), highlight: null });
  c.toBlob((blob) => save(blob, `${baseName()}-thirtytwo.png`), 'image/png');
});

$('dl-json').addEventListener('click', () => {
  const r = current();
  const thr = Object.fromEntries(CLASSES.map((c) => [c.key, +((r.thresholds?.[c.key] ?? 0.3) * S.factor).toFixed(5)]));
  const out = {
    app: 'Thirty Two', note: 'Research prototype. Not a medical device. Do not use it to diagnose or treat anyone.',
    file: S.name, mode: r.mode, image: r.image, time_s: r.time_s, sensitivity: S.factor, thresholds: thr,
    shown: CLASSES.filter((c) => S.shown[c.key]).map((c) => c.key),
    findings: visible().map(({ id, ...f }) => f), teeth: r.teeth,
  };
  save(new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' }), `${baseName()}-thirtytwo.json`);
});

// ------------------------------------------------------------------ wiring
function reset() {
  if (S.busy) return;
  show($('results'), false); show($('warning'), false); show($('drop'), true);
  message('');
  $('file').value = '';
  $('choose').focus();
}

$('choose').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => { if (e.target.files[0]) useFile(e.target.files[0]); });
const drop = $('drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  if (e.dataTransfer.files[0]) useFile(e.dataTransfer.files[0]);
});
document.querySelectorAll('[data-sample]').forEach((b) => b.addEventListener('click', async () => {
  try {
    const blob = await (await fetch(b.dataset.sample)).blob();
    const name = `DENTEX ${b.dataset.sample.split('/').pop()}`;
    useFile(new File([blob], name, { type: 'image/png' }), name);
  } catch (e) { message('The sample could not be loaded.'); }
}));
document.querySelectorAll('[data-again]').forEach((b) => b.addEventListener('click', reset));
document.querySelectorAll('input[name="mode"]').forEach((i) => i.addEventListener('change', () => {
  if (!i.checked || S.busy) return;
  if (S.results[i.value]) { S.mode = i.value; S.selected = null; syncMode(); panel(); } else analyse(i.value);
}));
$('sensitivity').addEventListener('input', (e) => {
  S.factor = 2 ** (Number(e.target.value) / 100);
  $('sens-value').textContent = `×${S.factor.toFixed(2).replace(/0$/, '')}`;
  e.target.setAttribute('aria-valuetext', `thresholds times ${S.factor.toFixed(2)}`);
  panel();
});
$('brightness').addEventListener('input', filters);
$('contrast').addEventListener('input', filters);
$('all-teeth').addEventListener('change', (e) => { S.allTeeth = e.target.checked; draw(); });
$('reset').addEventListener('click', () => {
  $('brightness').value = 100;
  $('contrast').value = 100;
  filters();
  fit();
});

toggles();
health();
