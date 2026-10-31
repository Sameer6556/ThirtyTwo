// Screenshots of the running app at 375 x 812 and 1440 x 900, before and after a sample, with checks:
// console errors, requests to other hosts, horizontal scroll, elements sticking out of the page.
//   node tools/screenshots.js http://127.0.0.1:7860 <output folder>
const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:7860';
  const out = process.argv[3] || 'screenshots';
  fs.mkdirSync(out, { recursive: true });
  const host = new URL(base).host;
  const browser = await chromium.launch();
  const problems = [];
  const outside = new Set();

  const layout = (page, label) => page.evaluate((label) => {
    const p = [];
    const W = window.innerWidth;
    if (document.documentElement.scrollWidth > W) p.push(`${label}: page scrolls sideways (${document.documentElement.scrollWidth} > ${W})`);
    for (const el of document.querySelectorAll('body *')) {
      if (el.closest('.viewer') || el.closest('[hidden]') || !el.getClientRects().length) continue;
      const r = el.getBoundingClientRect();
      if (r.right > W + 0.5 || r.left < -0.5) p.push(`${label}: <${el.tagName.toLowerCase()} class="${el.className}"> spans ${Math.round(r.left)}..${Math.round(r.right)} of ${W}`);
      if (['BUTTON', 'LABEL', 'P', 'SPAN', 'A', 'LI', 'H3'].includes(el.tagName) && el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflow !== 'visible')
        p.push(`${label}: text cut off in <${el.tagName.toLowerCase()}> "${el.textContent.trim().slice(0, 40)}"`);
    }
    return p;
  }, label);

  for (const [w, h] of [[375, 812], [1440, 900]]) {
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: w < 500 ? 2 : 1 });
    page.on('request', (r) => {
      const u = new URL(r.url());
      if (!['data:', 'blob:'].includes(u.protocol) && u.host !== host) outside.add(r.url());
    });
    page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) problems.push(`${w}px console ${m.type()}: ${m.text()}`); });
    page.on('pageerror', (e) => problems.push(`${w}px page error: ${e.message}`));
    page.on('response', (r) => { if (r.status() >= 400) problems.push(`${w}px HTTP ${r.status()} for ${r.url()}`); });

    await page.goto(base, { waitUntil: 'networkidle' });
    problems.push(...await layout(page, `${w}px start`));
    await page.screenshot({ path: `${out}/${w}-start.png` });
    await page.screenshot({ path: `${out}/${w}-page.png`, fullPage: true });

    await page.click('[data-sample="samples/val_28.png"]');
    await page.waitForSelector('#results:not([hidden])', { timeout: 180000 });
    await page.waitForTimeout(400);
    problems.push(...await layout(page, `${w}px results`));
    const row = await page.$('.row');
    if (row) await row.hover();
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${out}/${w}-results.png` });
    await page.screenshot({ path: `${out}/${w}-results-page.png`, fullPage: true });

    await page.check('#all-teeth');
    await page.locator('#viewer').hover();
    await page.mouse.wheel(0, -600);
    await page.waitForTimeout(200);
    await page.screenshot({ path: `${out}/${w}-zoom-teeth.png` });
    await page.uncheck('#all-teeth');

    // keyboard: a findings row takes focus (visible ring) and highlights its box
    await page.click('#reset');
    await page.focus('.row');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Shift+Tab');
    const focused = await page.evaluate(() => document.activeElement && document.activeElement.className);
    if (focused !== 'row' && focused !== 'row on') problems.push(`${w}px: keyboard focus not on a findings row (${focused})`);
    await page.screenshot({ path: `${out}/${w}-keyboard.png` });

    if (w > 1000) {     // switch to Full mode: runs the nine models and updates the measured time
      await page.click('input[name="mode"][value="full"]');
      await page.waitForSelector('#mode-progress:not([hidden])');
      await page.waitForTimeout(1500);
      await page.screenshot({ path: `${out}/${w}-loading.png` });
      await page.waitForSelector('#mode-progress', { state: 'hidden', timeout: 300000 });
      if (await page.isHidden('#results')) problems.push(`${w}px: results gone after switching to Full mode`);
      await page.waitForTimeout(400);
      problems.push(...await layout(page, `${w}px full`));
      await page.screenshot({ path: `${out}/${w}-full.png` });
    }
    await page.close();
  }
  await browser.close();
  if (outside.size) problems.push(`requests to other hosts: ${[...outside].join(', ')}`);
  console.log(JSON.stringify({ problems }, null, 1));
  process.exit(problems.length ? 1 : 0);
})();
