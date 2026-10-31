// How it works: show the same expected times as the tool (this server's recent runs, from /api/health)
'use strict';
const about = (s) => `about ${Math.max(1, Math.round(s))} s`;
fetch('/api/health').then((r) => r.json()).then((h) => {
  for (const el of document.querySelectorAll('[data-time]')) {
    const v = h.expected_s && h.expected_s[el.dataset.time];
    if (v) el.textContent = about(v);
  }
}).catch(() => { /* the page keeps its default text */ });
