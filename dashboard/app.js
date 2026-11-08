/* =======================================================================
   Дашборд комплекса синтеза фазового ДОЭ.
   Пять вкладок: физика · тест · алгоритм · модель · запуск.

   Тяжёлый счёт выполняется заранее (scripts/build_dashboard.py) либо
   локальным пультом (scripts/serve_dashboard.py). Здесь — рисование готовых
   массивов на офскрин-канвасах 128×128 и анимации; при открытии index.html
   как файла (офлайн) серверные функции просто не активны.
   ======================================================================= */
'use strict';

const N = 128, NP = 4;
const S = {
  tab: 'physics',
  variant: 'nn',
  plane: 0,
  quant: true,
  z: 0, playing: true, speed: 1, loop: true, autopause: true,
  planeMode: 'swipe', swipe: 0.5,
  protocol: 'primary',
  physStep: 0,
  curve: 'ssim',
  trainIdx: 0, trainPlaying: false,
  algoStep: 0, algoPlaying: true, algoSpeed: 1, algoT: 0,
  actIdx: 0, actPlaying: false, actT: 0,
  netT: 0, netPlaying: false,
  fps: 0,
};

const D = {};                    // данные бандла
const CACHE = { frames: {}, small: {} };
const RUN = { mode: 'off', available: false, sets: [], stages: [], pipelines: [],
              poll: null, since: 0, state: {}, files: {}, metrics: null,
              graph: null, act: null, history: null, currentSet: null, testResult: null };

/* ------------------------------------------------------------------ утилиты */
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const fmt = (v, d = 3) => (v === undefined || v === null || Number.isNaN(v)) ? '—' : (+v).toFixed(d);
const dpr = () => window.devicePixelRatio || 1;

function toCanvas(data, offset) {
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(N, N);
  const px = img.data;
  for (let i = 0; i < N * N; i++) {
    const o = i * 4;
    const v = data[offset + i];
    px[o] = px[o + 1] = px[o + 2] = v;
    px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function drawScaled(ctx, src, w, h) {
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(src, 0, 0, w, h);
}

/** Фаза → цветная картинка (циклическая палитра). */
function phaseCanvas(codes, offset = 0) {
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(N, N);
  const px = img.data;
  for (let i = 0; i < N * N; i++) {
    const t = codes[offset + i] / 255;
    const [r, g, b] = cyclic(t);
    const o = i * 4;
    px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function cyclic(t) {
  const a = 2 * Math.PI * t;
  return [Math.round(128 + 127 * Math.sin(a)),
          Math.round(128 + 127 * Math.sin(a + 2.094)),
          Math.round(128 + 127 * Math.sin(a + 4.188))];
}

function targetFrame(i) {
  const key = 't' + i;
  if (!CACHE.frames[key]) CACHE.frames[key] = toCanvas(D.targets, i * N * N);
  return CACHE.frames[key];
}

function reconFrame(i, variant = S.variant) {
  const src = D.recon[variant];
  const key = 'r' + variant + i;
  if (!CACHE.frames[key]) CACHE.frames[key] = toCanvas(src, i * N * N);
  return CACHE.frames[key];
}

function phaseCodes() {
  return S.variant === 'gs' ? D.phase.gs : D.phase.nn;
}

/* ------------------------------------------------------------------ загрузка */
async function loadBundle() {
  if (window.HDOE_BUNDLE) {
    const b = window.HDOE_BUNDLE;
    const bufs = {};
    for (const key of Object.keys(b.b64 || {})) bufs[key] = await b64ToBufGz(b.b64[key]);
    const u8 = key => new Uint8Array(bufs[key] || new ArrayBuffer(0));
    const f32 = key => new Float32Array(bufs[key] || new ArrayBuffer(0));
    const unDelta = (arr, npx) => {
      const n = Math.floor(arr.length / npx);
      for (let k = 1; k < n; k++) {
        const o = k * npx, p = o - npx;
        for (let i = 0; i < npx; i++) arr[o + i] = (arr[o + i] + arr[p + i]) & 255;
      }
      return arr;
    };
    D.manifest = b.manifest;
    D.targets = u8('targets');
    D.phase = { nn: f32('phase_nn'), gs: f32('phase_gs') };
    D.doe = { nn: u8('doe_nn'), gs: u8('doe_gs') };
    D.zstack = { nn: unDelta(u8('zstack_nn'), N * N), gs: unDelta(u8('zstack_gs'), N * N) };
    D.recon = { nn: f32('recon_nn'), gs: f32('recon_gs') };
    D.ssimMap = { nn: f32('ssim_map_nn'), gs: f32('ssim_map_gs') };
    D.zgrid = f32('zgrid');
    D.bitdepth = { nn: u8('bitdepth_nn'), gs: u8('bitdepth_gs') };
    D.zprobe = { nn: f32('zprobe_nn'), gs: f32('zprobe_gs') };
    D.physSteps = u8('phys_steps');
    D.sourceSteps = u8('source_steps');
    D.trainSteps = (b.json.train_steps || {}).steps || [];
    D.trainLog = b.json.train_log || [];
    D.tests = b.json.tests || {};
    D.trainMasks = u8('train_masks');
    D.trainRecon = u8('train_recon');
    D.offline = true;
    return;
  }
  const load = p => fetch(p).then(r => { if (!r.ok) throw new Error(p); return r.arrayBuffer(); });
  const j = p => fetch(p).then(r => r.json());
  const u8 = async p => new Uint8Array(await load('data/' + p));
  const f32 = async p => new Float32Array(await load('data/' + p));
  D.targets = await u8('targets.u8.bin');
  D.phase = { nn: await f32('phase_nn.f32.bin'), gs: await f32('phase_gs.f32.bin') };
  D.doe = { nn: await u8('doe_nn.u8.bin'), gs: await u8('doe_gs.u8.bin') };
  D.zstack = { nn: await u8('zstack_nn.u8.bin'), gs: await u8('zstack_gs.u8.bin') };
  D.recon = { nn: await f32('recon_nn.f32.bin'), gs: await f32('recon_gs.f32.bin') };
  D.ssimMap = { nn: await f32('ssim_map_nn.f32.bin'), gs: await f32('ssim_map_gs.f32.bin') };
  D.zgrid = await f32('zgrid.f32.bin');
  D.bitdepth = { nn: await u8('bitdepth_nn.u8.bin'), gs: await u8('bitdepth_gs.u8.bin') };
  D.zprobe = { nn: await f32('zprobe_nn.f32.bin'), gs: await f32('zprobe_gs.f32.bin') };
  D.physSteps = await u8('phys_steps.u8.bin');
  D.sourceSteps = await u8('source_steps.u8.bin');
  D.trainSteps = (await j('data/train_steps.json')).steps || [];
  D.trainLog = await j('data/train_log.json');
  D.tests = await j('data/tests.json');
  D.trainMasks = await u8('train_masks.u8.bin');
  D.trainRecon = await u8('train_recon.u8.bin');
  D.manifest = await j('data/manifest.json');
  D.offline = false;
}

/* gunzip + base64 (офлайн-бандл) */
async function b64ToBufGz(str) {
  const bin = atob(str);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (typeof DecompressionStream !== 'function') {
    throw new Error('браузер без DecompressionStream: запустите python scripts/serve_dashboard.py');
  }
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return await new Response(stream).arrayBuffer();
}

/* ------------------------------------------------------------------ шапка */
function renderHeader() {
  const c = D.manifest.constants;
  const m = D.manifest.metrics || {};
  const mean = (m.mean) || {};
  $('#params').innerHTML = [
    `SSIM <b>${fmt(mean.ssim, 3)}</b>`,
    `Pearson <b>${fmt(mean.pcc, 3)}</b>`,
    `K = <b>${(m.phys || {}).steps || '—'}</b>`,
    `N <b>${c.N}</b>`,
    `seed <b>${c.seed}</b>`,
  ].map(t => `<span class="chip">${t}</span>`).join('');

  const t = D.tests || {};
  const items = [['8.1', 'Парсеваль', t.parseval], ['8.2', 'Линза', t.lens],
    ['8.3', 'Квантование', t.quantization], ['8.4', 'Градиент', t.gradient],
    ['8.5', 'Кросс-схема', t.cross_scheme], ['8.6', 'Граница', t.boundary],
    ['8.7', 'Форматы', t.formats], ['8.9', 'Физика в сети', t.netphys]];
  const passed = items.filter(([, , r]) => r && r.passed !== false).length;
  $('#badges').innerHTML = items.map(([num, name, r]) => {
    const ok = r && r.passed !== false;
    return `<span class="badge ${ok ? '' : 'fail'}" data-test="${num}">${ok ? '✓' : '×'} ${num}</span>`;
  }).join('') + `<span class="badge info">${passed}/${items.length}</span>`;
  $$('.badge[data-test]').forEach(b => b.onclick = () => showTest(b.dataset.test));
}

const TEST_TITLES = {
  parseval: '8.1 Парсеваль: Σ|U₀|²Δx² = Σ|U_z|²Δx′²',
  lens: '8.2 Линза: фокус в центральном пикселе',
  quantization: '8.3 Квантование 8 бит: ошибка ≤ δ/2',
  gradient: '8.4 Градиент (gradcheck, complex128)',
  cross_scheme: '8.5 Кросс-схема: одно-БПФ vs угловой спектр',
  boundary: '8.6 Граница применимости z_crit',
  formats: '8.7 Форматы PNG: 128×128, 8 бит, режим L',
  netphys: '8.9 Физика внутри сети: один прямой проход',
};
const TEST_KEYS = { '8.1': 'parseval', '8.2': 'lens', '8.3': 'quantization', '8.4': 'gradient',
  '8.5': 'cross_scheme', '8.6': 'boundary', '8.7': 'formats', '8.9': 'netphys' };

function showTest(num) {
  const key = TEST_KEYS[num];
  const r = (D.tests || {})[key] || {};
  let body = `<div class="formula">${TEST_TITLES[key] || ''}</div>`;
  body += `<table>${Object.entries(r).map(([k, v]) =>
    `<tr><td>${k}</td><td>${typeof v === 'number' ? (+v).toPrecision(6) : escapeHtml(JSON.stringify(v))}</td></tr>`).join('')}</table>`;
  openModal(`Тест ${num}`, body);
}

function openModal(title, html) {
  $('#modal-title').textContent = title;
  $('#modal-body').innerHTML = html;
  $('#modal').classList.remove('hidden');
}
const escapeHtml = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

/* ------------------------------------------------------------------ вкладки */
function setTab(tab) {
  S.tab = tab;
  $$('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  ['physics', 'test', 'algo', 'model', 'run'].forEach(t =>
    $('#tab-' + t).classList.toggle('hidden', t !== tab));
  if (tab === 'algo') renderAlgo();
  if (tab === 'model') { ensureGraph(); ensureAct(); }
  if (tab === 'test') { renderTestTab(); }
  if (tab === 'run') renderRunTab();
  if (tab === 'physics') { renderDoe(); renderPlanes(); }
  requestAnimationFrame(tick);
}

/* =================================================================== 1. ФИЗИКА */
function renderSamples() {
  const grid = $('#samples');
  if (!grid) return;
  grid.innerHTML = '';
  for (let i = 0; i < NP; i++) {
    const cell = document.createElement('div');
    cell.className = 'target-cell' + (i === S.plane ? ' active' : '');
    cell.dataset.plane = i;
    const c = document.createElement('canvas');
    c.width = c.height = N;
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(targetFrame(i), 0, 0);
    cell.appendChild(c);
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.textContent = 'z' + (i + 1);
    cell.appendChild(tag);
    cell.onclick = () => setPlane(i);
    grid.appendChild(cell);
  }
}

function setPlane(i) {
  S.plane = i;
  renderSamples();
  renderPlanes();
  renderMetrics();
  renderProfiles();
  drawSsim();
}

function renderDoe() {
  const codes = S.quant ? (S.variant === 'gs' ? D.doe.gs : D.doe.nn) : null;
  const cv = $('#doe-phase');
  const ctx = cv.getContext('2d');
  const src = codes ? phaseCanvas(codes) : toCanvas(D.phase[S.variant], 0);
  drawScaled(ctx, src, cv.width, cv.height);
  const g = $('#doe-gray');
  drawScaled(g.getContext('2d'), codes ? toCanvas(codes, 0)
    : toCanvas(D.phase[S.variant], 0), N, N);
  drawHistogram(codes || new Uint8Array(0), D.phase[S.variant]);
  const ph = D.phase[S.variant];
  $('#doe-probe').innerHTML =
    `φ ∈ [${fmt(Math.min.apply(null, ph), 2)}; ${fmt(Math.max.apply(null, ph), 2)}] · ` +
    `δ ${(2 * Math.PI / 255).toFixed(4)} рад`;
}

function drawHistogram(codes, phf) {
  const cv = $('#doe-hist');
  const W = cv.width = cv.clientWidth * dpr();
  const H = cv.height = 80 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const bins = 64, hist = new Array(bins).fill(0);
  if (codes && codes.length) {
    for (let i = 0; i < codes.length; i++) hist[Math.min(bins - 1, (codes[i] / 256 * bins) | 0)]++;
  } else if (phf) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < phf.length; i++) { if (phf[i] < lo) lo = phf[i]; if (phf[i] > hi) hi = phf[i]; }
    for (let i = 0; i < phf.length; i++) {
      const t = (phf[i] - lo) / (hi - lo || 1);
      hist[Math.min(bins - 1, (t * bins) | 0)]++;
    }
  }
  const mx = Math.max(...hist, 1);
  const bw = W / bins;
  for (let i = 0; i < bins; i++) {
    const h = (hist[i] / mx) * (H - 6);
    ctx.fillStyle = codes && codes.length ? '#4dd6c1' : '#7aa2ff';
    ctx.fillRect(i * bw + 0.5, H - h, bw - 1, h);
  }
}

/* --------------------------------------------------------------- z-свип */
function renderPropStatic() {
  const c = D.manifest.constants;
  const track = $('#zscale-track');
  const z0 = D.zgrid[0], z1 = D.zgrid[D.zgrid.length - 1];
  const pos = z => ((z - z0) / (z1 - z0)) * 100;
  let html = '';
  c.zs_mm.forEach((z, i) => {
    html += `<div class="marker" style="left:${pos(z).toFixed(2)}%" data-plane="${i}">z${i + 1}</div>`;
  });
  html += `<div class="crit" style="left:${pos(c.z_crit_mm * 1e-3).toFixed(2)}%">z<sub>crit</sub></div>`;
  track.innerHTML = html;
  track.querySelectorAll('.marker').forEach(el => el.onclick = () => {
    const i = +el.dataset.plane;
    const z = c.zs_mm[i] * 1e-3;
    let best = 0, bd = 1e9;
    D.zgrid.forEach((zz, k) => { const d = Math.abs(zz - z); if (d < bd) { bd = d; best = k; } });
    setZ(best);
  });
  const cone = $('#cone');
  cone.innerHTML = '';
  const H = 320, W = 200;
  cone.innerHTML += `<defs><linearGradient id="beam" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#4dd6c1" stop-opacity=".55"/>
    <stop offset="1" stop-color="#4dd6c1" stop-opacity="0"/></linearGradient></defs>`;
  c.zs_mm.forEach((z, i) => {
    const y = 20 + (i / (c.zs_mm.length - 1)) * (H - 60);
    const spread = 26 + i * 12;
    cone.innerHTML += `<polygon points="${W / 2},${y} ${W / 2 - spread},${y + 18} ${W / 2 + spread},${y + 18}"
      fill="url(#beam)" opacity=".5"/>`;
    cone.innerHTML += `<rect x="6" y="${y + 16}" width="${W - 12}" height="2" rx="1" fill="#3a5680"/>`;
    cone.innerHTML += `<text x="${W - 8}" y="${y + 12}" fill="#8ea1c2" font-size="10" text-anchor="end">z${i + 1}</text>`;
  });
}

function setZ(k) {
  S.z = clamp(k, 0, D.zgrid.length - 1);
  const cv = $('#prop-canvas');
  const ctx = cv.getContext('2d');
  const stack = D.zstack[S.variant];
  drawScaled(ctx, toCanvas(stack, S.z * N * N), cv.width, cv.height);
  const z = D.zgrid[S.z];
  const c = D.manifest.constants;
  $('#prop-probe').innerHTML = `z = <b>${(z * 1e3).toFixed(2)} мм</b> · кадр ${S.z + 1}/${D.zgrid.length}` +
    (z < c.z_crit_mm * 1e-3 ? ' · z &lt; z<sub>crit</sub>' : '');
  $('#prop-slider').value = Math.round((S.z / (D.zgrid.length - 1)) * 100);
}

function tick(now) {
  const dt = _lastT ? Math.min(50, now - _lastT) : 16;
  _lastT = now;
  if (S.playing && S.tab === 'physics') {
    S.z += Math.max(1, Math.round(S.speed));
    if (S.z >= D.zgrid.length) {
      if (S.loop) S.z = 0; else S.playing = false;
    }
    if (S.autopause) {
      const c = D.manifest.constants;
      const z = D.zgrid[S.z];
      for (let i = 0; i < c.zs_mm.length; i++) {
        if (Math.abs(z - c.zs_mm[i] * 1e-3) < 0.01e-3) { /* у плоскости — кадр держится */ }
      }
    }
    setZ(S.z);
  }
  if (S.tab === 'algo') stepAlgo(dt);
  if (S.tab === 'model') { stepNet(dt); stepAct(dt); }
  if (S.trainPlaying) stepTrain();
  _frames++;
  if (now - _fpsT > 800) {
    S.fps = Math.round((_frames * 1000) / (now - _fpsT));
    $('#fps').textContent = S.fps + ' к/с';
    _frames = 0; _fpsT = now;
  }
  requestAnimationFrame(tick);
}
let _lastT = 0, _frames = 0, _fpsT = 0;

/* --------------------------------------------------------------- плоскости */
function renderPlanes() {
  const grid = $('#planes-grid');
  grid.innerHTML = '';
  for (let i = 0; i < NP; i++) {
    const cell = document.createElement('div');
    cell.className = 'plane-cell' + (i === S.plane ? ' active' : '');
    const cv = document.createElement('canvas');
    cv.width = cv.height = N;
    const ctx = cv.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    if (S.planeMode === 'side') {
      ctx.drawImage(targetFrame(i), 0, 0, N, N);
      ctx.globalAlpha = 0.5;
      ctx.drawImage(reconFrame(i), 0, 0, N, N);
      ctx.globalAlpha = 1;
    } else if (S.planeMode === 'diff') {
      const img = ctx.createImageData(N, N);
      const t = D.targets, r = D.recon[S.variant];
      for (let k = 0; k < N * N; k++) {
        const d = Math.abs(t[i * N * N + k] / 255 - r[i * N * N + k] / 255);
        const v = clamp(d * 255 * 2, 0, 255);
        img.data[k * 4] = v; img.data[k * 4 + 1] = v * 0.35; img.data[k * 4 + 2] = v * 0.2;
        img.data[k * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    } else {
      ctx.drawImage(reconFrame(i), 0, 0);
      const cut = Math.round(N * S.swipe);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, cut, N); ctx.clip();
      ctx.drawImage(targetFrame(i), 0, 0);
      ctx.restore();
      ctx.strokeStyle = '#4dd6c1'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cut, 0); ctx.lineTo(cut, N); ctx.stroke();
    }
    cell.appendChild(cv);
    const tag = document.createElement('div');
    tag.className = 'tag';
    const m = (D.manifest.file_metrics || {})[S.variant];
    tag.textContent = `z${i + 1} · ${m ? fmt(m[i].ssim, 3) : ''}`;
    cell.appendChild(tag);
    cell.onclick = () => setPlane(i);
    cell.onmousemove = e => {
      if (S.planeMode !== 'swipe') return;
      const r = cv.getBoundingClientRect();
      S.swipe = clamp((e.clientX - r.left) / r.width, 0, 1);
      renderPlanes();
    };
    grid.appendChild(cell);
  }
  const row = (D.manifest.file_metrics || {})[S.variant]?.[S.plane];
  const c = D.manifest.constants;
  $('#target-caption').innerHTML = row
    ? `z${S.plane + 1} = <b>${row.z_mm.toFixed(1)} мм</b> · Δx′ ${(c.out_dx[S.plane] * 1e6).toFixed(2)} мкм · SSIM <b>${fmt(row.ssim, 3)}</b> · Pearson <b>${fmt(row.pcc, 3)}</b>`
    : '';
}

function renderBitbar() {
  const bar = $('#bitbar');
  bar.innerHTML = `<span class="plot-title">битность</span>` +
    [8, 6, 4, 2].map((b, i) => `<button data-bit="${b}" class="${b === 8 ? 'active' : ''}">${b}</button>`).join('');
  $$('#bitbar button').forEach(btn => btn.onclick = () => {
    const idx = [8, 6, 4, 2].indexOf(+btn.dataset.bit);
    $$('#bitbar button').forEach(x => x.classList.toggle('active', x === btn));
    S.bits = +btn.dataset.bit;
    const src = D.bitdepth[S.variant];
    const cell = $('#planes-grid').children[S.plane];
    if (!cell) return;
    const cv = cell.querySelector('canvas');
    if (idx === 0) { renderPlanes(); return; }
    drawScaled(cv.getContext('2d'), toCanvas(src, idx * N * N), N, N);
  });
}

/* ------------------------------------------------------------- профили */
function renderProfiles() {
  const rec = D.recon[S.variant];
  const off = S.plane * N * N;
  const row = [], col = [];
  for (let i = 0; i < N; i++) { row.push(rec[off + 64 * N + i]); col.push(rec[off + i * N + 64]); }
  drawSeries($('#prof-h'), row, '#4dd6c1');
  drawSeries($('#prof-v'), col, '#7aa2ff');
  const zp = D.zprobe[S.variant];
  const nz = D.zgrid.length;
  const curve = [];
  for (let i = 0; i < nz; i++) curve.push(zp[i]);
  drawSeries($('#zprobe'), curve, '#ffb347');
  // маркер текущей плоскости на кривой
  const cvz = $('#zprobe');
  if (cvz) {
    const ctx = cvz.getContext('2d');
    let lo = Infinity, hi = -Infinity;
    curve.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; });
    const x = (S.z / (nz - 1)) * cvz.width;
    ctx.strokeStyle = 'rgba(77,214,193,.85)';
    ctx.lineWidth = 1.2 * dpr();
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cvz.height); ctx.stroke();
    void lo; void hi;
  }
  $('#prof-probe').innerHTML = `строка/столбец 64 · плоскость z${S.plane + 1}`;
}

function drawSeries(cv, vals, color) {
  if (!cv) return;
  const W = cv.width = cv.clientWidth * dpr();
  const H = cv.height = (cv.getAttribute('height') | 0) * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  let lo = Infinity, hi = -Infinity;
  vals.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; });
  if (!(hi > lo)) { hi = lo + 1; }
  ctx.beginPath();
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * W;
    const y = H - 4 - ((v - lo) / (hi - lo)) * (H - 8);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 1.6 * dpr(); ctx.stroke();
}

function drawSsim() {
  const cv = $('#ssim-canvas');
  if (!cv) return;
  const src = D.ssimMap[S.variant];
  if (!src || !src.length) return;
  drawScaled(cv.getContext('2d'), toCanvas(src, S.plane * N * N), cv.width, cv.height);
}

/* =================================================================== 2. ТЕСТ */
function renderTestTab() {
  const sel = $('#test-set');
  const sets = RUN.sets || [];
  const sig = sets.map(s => s.id).join(',');
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.innerHTML = sets.map(s => {
      const label = s.id === 'default' ? 'сдаваемый набор' : s.id;
      const m = s.mean ? ` · SSIM ${fmt(s.mean.ssim, 3)}` : '';
      return `<option value="${s.id}">${label}${m}</option>`;
    }).join('') || '<option value="">нет комплектов</option>';
    if (!sets.some(s => s.id === RUN.currentSet)) RUN.currentSet = sets.length ? sets[sets.length - 1].id : null;
    sel.value = RUN.currentSet || '';
    sel.onchange = () => { RUN.currentSet = sel.value || null; renderTestTab(); loadSetIntoPhysics(sel.value); };
  }
  const cur = RUN.currentSet;
  const info = sets.find(s => s.id === cur);
  $('#test-thumbs').innerHTML = '';
  for (let i = 0; i < NP; i++) {
    const d = document.createElement('div');
    d.className = 'thumb';
    const c = document.createElement('canvas');
    c.width = c.height = N;
    c.getContext('2d').drawImage(targetFrame(i), 0, 0);
    d.appendChild(c);
    d.innerHTML += `<span>z${i + 1}</span>`;
    $('#test-thumbs').appendChild(d);
  }
  const m = info && info.mean ? info.mean : (D.manifest.metrics.mean || {});
  const pear = m.pearson ?? m.pcc ?? (D.manifest.metrics.mean || {}).pcc;
  $('#test-probe').innerHTML = cur
    ? `${cur === 'default' ? 'сдаваемый набор' : 'комплект <b>' + cur + '</b>'}` +
      ` · SSIM <b>${fmt(m.ssim, 3)}</b> · Pearson <b>${fmt(pear, 3)}</b>` +
      `${info && info.has_doe ? ' · DOE готов' : ' · DOE не посчитан'}`
    : `сдаваемый набор · SSIM <b>${fmt(m.ssim, 3)}</b> · Pearson <b>${fmt(pear, 3)}</b>`;
  // DOE из комплекта либо из сдаваемого набора
  const doeBox = $('#test-doe');
  doeBox.innerHTML = '';
  const url = cur ? `api/asset?set=${encodeURIComponent(cur)}&name=DOE.png` : null;
  if (url && info && info.has_doe) {
    const img = document.createElement('img');
    img.src = url + '&t=' + Date.now();
    img.className = 'doe-img';
    doeBox.appendChild(img);
    doeBox.insertAdjacentHTML('beforeend', '<div class="plot-title">DOE.png 128×128 · 8 бит</div>');
    $('#test-download').onclick = () => {
      const a = document.createElement('a');
      a.href = img.src;
      a.download = 'DOE.png';
      document.body.appendChild(a);
      a.click();
      a.remove();
    };
  } else {
    doeBox.innerHTML = `<div class="plot-title">DOE.png сдаваемого набора</div>`;
    const c = document.createElement('canvas');
    c.width = c.height = N;
    drawScaled(c.getContext('2d'), toCanvas(D.doe[S.variant], 0), N, N);
    c.className = 'doe-img';
    doeBox.appendChild(c);
    $('#test-download').onclick = () => downloadCanvas(c, 'DOE.png');
  }
  renderTestRecon(cur, info);
  renderMetrics();
  renderCurves();
  renderSetsTable();
}

/** Таблица по всем комплектам: типы картинок, SSIM и Pearson. */
function renderSetsTable() {
  const box = $('#sets-table');
  if (!box) return;
  const sum = RUN.summary;
  const rows = (sum && sum.sets) || [];
  if (!rows.length) {
    box.innerHTML = `<div class="hint">сводка появится после
      <code>python scripts/report_sets.py</code></div>`;
    return;
  }
  box.innerHTML = `<table><tr><th>комплект</th><th>тип</th><th>SSIM</th>
      <th>Pearson</th><th>z₁…z₄</th></tr>` +
    rows.map(r => `<tr data-set="${r.id}"><td>${shortId(r.id)}</td>
      <td class="kind">${r.kind || ''}</td>
      <td>${fmt(r.ssim, 3)}</td><td>${fmt(r.pcc, 3)}</td>
      <td class="planes">${(r.ssim_planes || []).map(v => v.toFixed(2)).join(' ')}</td></tr>`).join('') +
    `</table>`;
  box.querySelectorAll('tr[data-set]').forEach(tr => tr.onclick = () => {
    const id = tr.dataset.set;
    if (!id.startsWith('sample-')) return;
    RUN.currentSet = id;
    if ($('#test-set')) $('#test-set').value = id;
    loadSetIntoPhysics(id);
    renderTestTab();
  });
}

const shortId = id => String(id).replace('sample-', '').replace('data/targets', 'сдаваемый');

function downloadCanvas(cv, name) {
  cv.toBlob(b => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = name;
    a.click();
  });
}

function renderTestRecon(cur, info) {
  const box = $('#test-recon');
  box.innerHTML = '';
  const metrics = RUN.testResult && RUN.testResult.planes;
  for (let i = 0; i < NP; i++) {
    const cell = document.createElement('div');
    cell.className = 'recon-cell';
    const c = document.createElement('canvas');
    c.width = c.height = N;
    const ctx = c.getContext('2d');
    if (cur && info && info.has_recon) {
      const img = new Image();
      img.onload = () => { ctx.imageSmoothingEnabled = false; ctx.drawImage(img, 0, 0); };
      img.src = `api/asset?set=${encodeURIComponent(cur)}&name=recon_${i + 1}.png&t=${Date.now()}`;
    } else {
      ctx.drawImage(reconFrame(i), 0, 0);
    }
    cell.appendChild(c);
    const m = metrics ? metrics[i] : ((D.manifest.file_metrics || {})[S.variant] || [])[i];
    cell.insertAdjacentHTML('beforeend',
      `<span>z${i + 1} · SSIM ${m ? fmt(m.ssim, 3) : '—'}<br>Pearson ${m ? fmt(m.pearson || m.pcc, 3) : '—'}</span>`);
    box.appendChild(cell);
  }
}

function renderMetrics() {
  const key = r => (S.protocol === 'primary' ? r.ssim : (r.ssim_raw ?? r.ssim));
  const rows = (D.manifest.file_metrics || {})[S.variant] || [];
  const active = rows[S.plane] || {};
  const mean = (D.manifest.metrics || {}).mean || {};
  const gs = ((D.manifest.metrics || {}).gs || {}).mean || {};
  $('#c-ssim').textContent = fmt(key(active), 3);
  $('#c-pearson').textContent = fmt(active.pcc, 3);
  $('#c-gs').textContent = fmt(gs.ssim, 3);
  $('#c-planes').textContent = '4';
  const bars = $('#metrics-bars');
  bars.innerHTML = rows.map((r, i) => `
    <div class="bar-row ${i === S.plane ? 'active' : ''}" data-plane="${i}">
      <span class="bar-label">z${i + 1}</span>
      <div class="bar"><i style="width:${(clamp(key(r), 0, 1) * 100).toFixed(1)}%"></i></div>
      <span class="bar-val">${fmt(key(r), 3)}</span>
    </div>`).join('') +
    `<div class="bar-row"><span class="bar-label">∅</span>
      <div class="bar alt"><i style="width:${(clamp(key(mean), 0, 1) * 100).toFixed(1)}%"></i></div>
      <span class="bar-val">${fmt(key(mean), 3)}</span></div>`;
  bars.querySelectorAll('.bar-row[data-plane]').forEach(el => el.onclick = () => setPlane(+el.dataset.plane));
  const t = $('#metrics-tables');
  t.innerHTML = `<table><tr><th>z</th><th>мм</th><th>SSIM</th><th>Pearson</th><th>SSIM без норм.</th></tr>` +
    rows.map(r => `<tr data-plane="${r.plane - 1}"><td>z${r.plane}</td><td>${r.z_mm.toFixed(1)}</td>
      <td>${fmt(r.ssim, 4)}</td><td>${fmt(r.pcc, 4)}</td><td>${fmt(r.ssim_raw, 4)}</td></tr>`).join('') +
    `<tr class="sum"><td>∅</td><td></td><td>${fmt(mean.ssim, 4)}</td><td>${fmt(mean.pcc, 4)}</td>
      <td>${fmt(mean.ssim_raw, 4)}</td></tr></table>`;
  t.querySelectorAll('tr[data-plane]').forEach(el => el.onclick = () => setPlane(+el.dataset.plane));
}

/* ------------------------------------------------------------- кривые */
function renderCurves() {
  const cv = $('#curve-canvas');
  if (!cv) return;
  const W = cv.width = cv.clientWidth * dpr();
  const H = cv.height = 220 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const hist = RUN.history || {};
  const key = S.curve;
  const series = [];
  if (hist.train && hist.train.length) {
    series.push({ name: 'обучение', color: '#4dd6c1',
      pts: hist.train.filter(r => r[key === 'ssim' ? 'train_ssim' : 'train_pearson'] !== null)
        .map(r => ({ x: r.step, y: r[key === 'ssim' ? 'train_ssim' : 'train_pearson'] })) });
  }
  if (hist.val && hist.val.length) {
    series.push({ name: 'валидация', color: '#7aa2ff',
      pts: hist.val.map(r => ({ x: r.step, y: r[key] })) });
  }
  const test = hist.test || {};
  const mean = (D.manifest.metrics || {}).mean || {};
  const testVal = key === 'ssim' ? mean.ssim : mean.pcc;
  const all = series.flatMap(s => s.pts.map(p => p.y)).concat([testVal]);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.15 || 0.05;
  lo -= pad; hi += pad;
  const X = x => 34 + (x / 1600) * (W - 50);
  const Y = y => H - 22 - ((y - lo) / (hi - lo)) * (H - 40);
  // сетка
  ctx.strokeStyle = 'rgba(120,150,190,.18)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yy = H - 22 - (i / 4) * (H - 40);
    ctx.beginPath(); ctx.moveTo(34, yy); ctx.lineTo(W - 16, yy); ctx.stroke();
    ctx.fillStyle = 'rgba(160,185,215,.75)';
    ctx.font = `${9 * dpr()}px ui-monospace, monospace`;
    ctx.fillText((lo + (hi - lo) * (i / 4)).toFixed(2), 4, yy + 3);
  }
  series.forEach(s => {
    ctx.beginPath();
    s.pts.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.8 * dpr(); ctx.stroke();
  });
  // тестовая точка
  ctx.setLineDash([4 * dpr(), 4 * dpr()]);
  ctx.beginPath(); ctx.moveTo(X(1000), Y(testVal)); ctx.lineTo(W - 16, Y(testVal)); ctx.strokeStyle = '#ffb347';
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#ffb347'; ctx.beginPath(); ctx.arc(X(1000), Y(testVal), 3.4 * dpr(), 0, 7); ctx.fill();
  $('#curve-legend').innerHTML = series.map(s => `<span style="color:${s.color}">■ ${s.name}</span>`)
    .join(' ') + `<span style="color:#ffb347">▬ тест ${fmt(testVal, 3)}</span>`;
  renderSplits();
}

/** Три набора одной моделью: тренировка / валидация / тест. */
function renderSplits() {
  const box = $('#split-row');
  if (!box) return;
  // под пультом — свежие данные с сервера, офлайн — из бандла
  const sp = (RUN.history && RUN.history.splits) ||
    (D.manifest && D.manifest.splits) || null;
  if (!sp || !sp.train || !sp.val || !sp.test) {
    box.innerHTML = `<span class="hint">протокол train / val / test:
      <code>python scripts/eval_splits.py</code></span>`;
    return;
  }
  const cell = (key, label, note) => {
    const s = sp[key];
    return `<div class="split-cell">
      <div class="split-k">${label}<span>n = ${s.n}</span></div>
      <div class="split-v">${fmt(s.ssim, 3)}<small>SSIM</small></div>
      <div class="split-p">Pearson ${fmt(s.pcc, 3)}</div>
      <div class="split-n">${note}</div></div>`;
  };
  box.innerHTML =
    cell('train', 'тренировка', 'четвёрки корпуса из обучения') +
    cell('val', 'валидация', 'восемь четвёрок, не видевших обучения') +
    cell('test', 'тест', 'сдаваемый набор и пять комплектов');
}

/* =================================================================== 3. АЛГОРИТМ */
const ALGO_STEPS = [
  { t: '1. Вход', d: 'Комплект из четырёх изображений 128×128 подаётся в сеть как 4 канала.' },
  { t: '2. U-Net', d: 'Энкодер сжимает вход ×16, декодер со skip-связями собирает карту (c, s).' },
  { t: '3. Фаза', d: 'φ = atan2(s, c) — начальная фаза маски, диапазон (−π, π].' },
  { t: '4. Физика', d: 'K шагов спуска по физическому лоссу: градиент через пропагатор Френеля.' },
  { t: '5. Квантование', d: '8 бит: 0 ↔ −π, 255 ↔ +π, шаг δ = 2π/255.' },
  { t: '6. Дифракция', d: 'Одно-БПФ-схема: чирп → БПФ → чирп для каждого z_i.' },
  { t: '7. Метрики', d: 'SSIM и Pearson по четырём восстановленным изображениям.' },
];

function renderAlgo() {
  drawAlgoSvg();
  drawWeights();
  renderIterGrid();
  $('#algo-code').textContent =
`φ₀ = U-Net(t)                      # один прямой проход
φ  = PhysicsRefiner(φ₀, t, K)      # K шагов по ∇ L(A(φ))
φ_q = round(255·(φ+π)/2π)          # 8 бит
I_i = |F⁻¹{ Q_out · F{ e^{iφ_q}·Q_in } }|²
SSIM_i, Pearson_i = метрика(t_i, I_i)`;
  $('#algo-play').onclick = () => { S.algoPlaying = !S.algoPlaying; $('#algo-play').textContent = S.algoPlaying ? '⏸' : '▶'; };
  $('#algo-speed').oninput = e => { S.algoSpeed = +e.target.value; };
}

function drawAlgoSvg() {
  const svg = $('#algo-svg');
  const boxes = ALGO_STEPS.map((s, i) => ({ ...s, i }));
  const w = 96, h = 46, gap = 8;
  let html = '';
  boxes.forEach((b, i) => {
    const row = i < 4 ? 0 : 1;
    const col = i < 4 ? i : i - 4;
    const x = 12 + col * (w + gap) + (row ? 40 : 0);
    const y = row ? 170 : 40;
    html += `<g class="algo-node ${b.i === S.algoStep ? 'on' : ''}" data-i="${b.i}">
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/>
      <text x="${x + w / 2}" y="${y + h / 2 + 4}" text-anchor="middle">${b.t}</text></g>`;
    if (col < 3 || (i >= 4 && col < 6)) {
      const nx = x + w + 2;
      html += `<path d="M${nx} ${y + h / 2} l${gap - 4} 0" stroke="#3a5680" stroke-width="1.5" marker-end="url(#ar)"/>`;
    }
    if (i === 3) {
      html += `<path d="M${x + w / 2} ${y + h} C ${x + w / 2} ${y + h + 40}, 40 ${170}, 52 ${170}"
        stroke="#3a5680" stroke-width="1.5" fill="none" marker-end="url(#ar)"/>`;
    }
  });
  html = `<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 z" fill="#3a5680"/></marker></defs>` + html;
  svg.innerHTML = html;
  $$('#algo-svg .algo-node').forEach(n => n.onclick = () => { S.algoStep = +n.dataset.i; S.algoT = 0; drawAlgoSvg(); showAlgoStep(); });
  showAlgoStep();
}

function showAlgoStep() {
  const st = ALGO_STEPS[S.algoStep];
  $('#algo-step').innerHTML = `<b>${st.t}</b> — ${st.d}`;
}

function stepAlgo(dt) {
  S.algoT += dt * S.algoSpeed;
  if (S.algoT > 1500) {
    S.algoT = 0;
    S.algoStep = (S.algoStep + 1) % ALGO_STEPS.length;
    drawAlgoSvg();
  }
  drawWeights();
  stepIter(dt);
}

function drawWeights() {
  const cv = $('#weights-canvas');
  if (!cv) return;
  const W = cv.width = cv.clientWidth * dpr();
  const H = cv.height = 240 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const g = RUN.graph;
  if (!g || !g.layers || !g.layers.length) {
    ctx.fillStyle = 'rgba(160,185,215,.7)';
    ctx.font = `${11 * dpr()}px ui-monospace, monospace`;
    ctx.fillText('данные о весах появятся под пультом', 10, 20);
    return;
  }
  // «живые» веса: колонки по слоям, яркость — |w|, пульсация по шагу алгоритма
  const cols = Math.min(g.layers.length, 60);
  const step = Math.max(1, Math.floor(g.layers.length / cols));
  const bw = W / cols;
  const phase = S.algoStep / ALGO_STEPS.length;
  for (let i = 0; i < cols; i++) {
    const l = g.layers[i * step];
    const amp = clamp(l.abs_mean * 6, 0.05, 1);
    const pul = 0.6 + 0.4 * Math.sin(2 * Math.PI * (phase + i / cols));
    ctx.fillStyle = `rgba(77,214,193,${(amp * pul).toFixed(3)})`;
    const h = (H - 20) * amp;
    ctx.fillRect(i * bw + 0.5, H - 10 - h, bw - 1, h);
  }
  ctx.fillStyle = 'rgba(160,185,215,.75)';
  ctx.font = `${9 * dpr()}px ui-monospace, monospace`;
  ctx.fillText(`слоёв ${g.layers.length} · параметров ${(g.params / 1e6).toFixed(2)} М`, 8, 12);
  $('#weights-probe').innerHTML = `средний |w| ${fmt(g.layers.reduce((a, l) => a + l.abs_mean, 0) / g.layers.length, 4)}`;
}

function renderIterGrid() {
  const box = $('#iter-grid');
  if (!box) return;
  box.innerHTML = '';
  const row = D.manifest.phys_steps || [];
  if (!row.length) return;
  const picks = [0, Math.floor(row.length / 3), Math.floor(2 * row.length / 3), row.length - 1];
  picks.forEach((p, i) => {
    const cell = document.createElement('div');
    cell.className = 'iter-cell' + (i === S.physStep ? ' active' : '');
    const cv = document.createElement('canvas');
    cv.width = cv.height = 96;
    drawScaled(cv.getContext('2d'), phaseCanvas(D.physSteps, p * N * N), 96, 96);
    cell.appendChild(cv);
    cell.insertAdjacentHTML('beforeend',
      `<span>K ${row[p].step}<b>${fmt(row[p].pcc, 3)}</b></span>`);
    cell.onclick = () => { S.physStep = p; renderIterGrid(); };
    box.appendChild(cell);
  });
  $('#iter-info').textContent = `K = ${D.manifest.phys_total || '—'}`;
}

function stepIter(dt) {
  const row = D.manifest.phys_steps || [];
  if (!row.length) return;
  S.physT = (S.physT || 0) + dt;
  if (S.physT > 400) { S.physT = 0; S.physStep = (S.physStep + 1) % row.length; renderIterGrid(); }
}

/* =================================================================== 4. МОДЕЛЬ */
async function ensureGraph() {
  if (RUN.graph || !RUN.available) return drawNetGraph();
  try { RUN.graph = await (await fetch('api/model/graph', { cache: 'no-store' })).json(); }
  catch (e) { RUN.graph = null; }
  drawNetGraph();
  drawWeights();
}

async function ensureAct() {
  if (RUN.act || !RUN.available) return drawAct();
  const url = `api/model/act${RUN.currentSet ? '?set=' + encodeURIComponent(RUN.currentSet) : ''}`;
  try {
    RUN.act = await (await fetch(url, { cache: 'no-store' })).json();
  } catch (e) { RUN.act = null; }
  if (!RUN.act || !RUN.act.stages || !RUN.act.stages.length) {
    // первый расчёт на CPU занимает несколько секунд — пробуем ещё раз
    await new Promise(r => setTimeout(r, 1500));
    try {
      RUN.act = await (await fetch(url, { cache: 'no-store' })).json();
    } catch (e) { /* остаёмся без активаций */ }
  }
  drawAct();
}

function drawNetGraph() {
  const cv = $('#net-canvas');
  if (!cv) return;
  const W = cv.width = cv.clientWidth * dpr();
  const H = cv.height = 420 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const g = RUN.graph;
  if (!g || !g.layers || !g.layers.length) {
    ctx.fillStyle = 'rgba(160,185,215,.7)';
    ctx.font = `${12 * dpr()}px ui-monospace, monospace`;
    ctx.fillText('граф весов доступен под пультом: python scripts/serve_dashboard.py', 12, 26);
    return;
  }
  // узлы по группам: энкодер (вниз), декодер (вверх), голова
  const groups = { encoder: [], decoder: [], head: [] };
  g.layers.forEach(l => (groups[l.group] || groups.head).push(l));
  const stage = [];
  groups.encoder.forEach((l, i) => stage.push({ l, x: 0.08 + 0.22 * i / Math.max(1, groups.encoder.length - 1), y: 0.18 }));
  groups.decoder.forEach((l, i) => stage.push({ l, x: 0.08 + 0.22 * i / Math.max(1, groups.decoder.length - 1), y: 0.66 }));
  groups.head.forEach((l, i) => stage.push({ l, x: 0.78 + 0.16 * i / Math.max(1, groups.head.length - 1), y: 0.42 }));
  const R = 5.5 * dpr();
  stage.forEach((n, i) => {
    const x = n.x * W, y = n.y * H;
    const act = Math.sin(2 * Math.PI * (S.netT / 1000 + i / stage.length));
    const glow = S.netPlaying ? 0.45 + 0.55 * Math.max(0, act) : 0.5;
    ctx.beginPath();
    ctx.fillStyle = `rgba(77,214,193,${(0.35 + 0.6 * clamp(n.l.abs_mean * 8, 0, 1) * glow).toFixed(3)})`;
    ctx.arc(x, y, R * (1 + 0.5 * n.l.numel / Math.max(...g.layers.map(l => l.numel))), 0, 7);
    ctx.fill();
    if (i % Math.ceil(stage.length / 14) === 0) {
      ctx.fillStyle = 'rgba(170,195,220,.8)';
      ctx.font = `${8 * dpr()}px ui-monospace, monospace`;
      ctx.fillText(n.l.name.replace(/^net\./, '').slice(0, 16), x - 24, y - 10 * dpr());
    }
  });
  // рёбра внутри группы
  (g.edges || []).forEach(([a, b]) => {
    const A = stage.find(s => s.l.name === a), B = stage.find(s => s.l.name === b);
    if (!A || !B) return;
    ctx.beginPath(); ctx.moveTo(A.x * W, A.y * H); ctx.lineTo(B.x * W, B.y * H);
    ctx.strokeStyle = 'rgba(90,130,175,.25)'; ctx.lineWidth = 1 * dpr(); ctx.stroke();
  });
  ctx.fillStyle = 'rgba(160,185,215,.65)';
  ctx.font = `${9 * dpr()}px ui-monospace, monospace`;
  ctx.fillText('энкодер', 8, H * 0.14);
  ctx.fillText('декодер', 8, H * 0.62);
  ctx.fillText('голова', W * 0.76, H * 0.38);
  $('#net-probe').innerHTML = `параметров <b>${(g.params / 1e6).toFixed(2)} М</b> · слоёв ${g.layers.length}`;
}

function stepNet(dt) {
  if (!S.netPlaying) return;
  S.netT += dt;
  drawNetGraph();
}

function drawAct() {
  const box = $('#act-grid');
  if (!box) return;
  const act = RUN.act;
  if (!act || !act.stages || !act.stages.length) {
    box.innerHTML = `<div class="hint">активации доступны под пультом</div>`;
    return;
  }
  if (box.children.length !== act.stages.length) {
    box.innerHTML = '';
    act.stages.forEach((st, i) => {
      const d = document.createElement('div');
      d.className = 'act-cell';
      d.innerHTML = `<canvas width="64" height="64"></canvas><span>${st.name}</span>`;
      d.onclick = () => { S.actIdx = i; drawAct(); };
      box.appendChild(d);
    });
  }
  [...box.children].forEach((d, i) => {
    const st = act.stages[i];
    const cv = d.querySelector('canvas');
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(24, 24);
    for (let k = 0; k < 24 * 24; k++) {
      const v = st.data[k];
      const on = i <= S.actIdx;
      img.data[k * 4] = on ? v * 0.35 : 12;
      img.data[k * 4 + 1] = on ? v : 16;
      img.data[k * 4 + 2] = on ? v * 0.85 : 24;
      img.data[k * 4 + 3] = 255;
    }
    const t = document.createElement('canvas');
    t.width = t.height = 24;
    t.getContext('2d').putImageData(img, 0, 0);
    drawScaled(ctx, t, 64, 64);
    d.classList.toggle('on', i <= S.actIdx);
  });
}

function stepAct(dt) {
  if (!S.actPlaying || !RUN.act || !RUN.act.stages) return;
  S.actT += dt;
  if (S.actT > 320) {
    S.actT = 0;
    S.actIdx = (S.actIdx + 1) % RUN.act.stages.length;
    drawAct();
  }
}

/* --------------------------------------------------------- обучение (панель) */
function renderTrain() {
  const steps = D.trainSteps || [];
  const sl = $('#train-slider');
  sl.max = Math.max(0, steps.length - 1);
  sl.value = Math.min(S.trainIdx, sl.max);
  $('#train-play').onclick = () => {
    S.trainPlaying = !S.trainPlaying;
    $('#train-play').textContent = S.trainPlaying ? '⏸' : '▶';
  };
  sl.oninput = e => { S.trainIdx = +e.target.value; renderTrain(); };
  drawTrainFrame();
  drawTrainCurve();
}

function drawTrainFrame() {
  const steps = D.trainSteps || [];
  if (!steps.length) return;
  const i = clamp(S.trainIdx, 0, steps.length - 1);
  $('#train-mask').getContext('2d');
  drawScaled($('#train-mask').getContext('2d'), phaseCanvas(D.trainMasks, i * N * N), 150, 150);
  drawScaled($('#train-recon').getContext('2d'), toCanvas(D.trainRecon, i * N * N), 150, 150);
  $('#train-step-label').textContent = `шаг ${steps[i]}`;
}

function stepTrain() {
  const steps = D.trainSteps || [];
  if (!steps.length) return;
  S.trainIdx = (S.trainIdx + 1) % steps.length;
  $('#train-slider').value = S.trainIdx;
  drawTrainFrame();
}

function drawTrainCurve() {
  const cv = $('#train-loss');
  if (!cv) return;
  const W = cv.width = cv.clientWidth * dpr();
  const H = cv.height = 130 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const log = D.trainLog || [];
  if (!log.length) return;
  const losses = log.map(r => r.loss).filter(v => typeof v === 'number');
  let lo = Math.min(...losses), hi = Math.max(...losses);
  ctx.beginPath();
  losses.forEach((v, i) => {
    const x = (i / Math.max(1, losses.length - 1)) * (W - 8) + 4;
    const y = H - 6 - ((v - lo) / (hi - lo || 1)) * (H - 12);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = '#4dd6c1'; ctx.lineWidth = 1.6 * dpr(); ctx.stroke();
}

/* =================================================================== 5. ЗАПУСК */
function renderRunTab() {
  const pip = $('#rp-pipelines');
  if (!pip) return;
  if (!RUN.available) {
    pip.innerHTML = '';
    $('#rp-stages').innerHTML = `<div class="hint">запуск задач доступен при работе через пульт:
      <code>python scripts/serve_dashboard.py</code> (в Docker — сразу)</div>`;
    $('#rp-log').textContent = 'офлайн-режим: только просмотр уже посчитанных данных';
    return;
  }
  pip.innerHTML = RUN.pipelines.map(p =>
    `<button data-pl="${p.id}" title="${p.stages.join(' → ')}">${p.title}</button>`).join('');
  pip.querySelectorAll('button').forEach(b => b.onclick = () => runStart({ pipeline: b.dataset.pl }));
  const groups = {};
  RUN.stages.forEach(s => (groups[s.group] = groups[s.group] || []).push(s));
  $('#rp-stages').innerHTML = Object.entries(groups).map(([g, items]) =>
    `<div class="plot-title">${g}</div>` + items.map(s =>
      `<button class="rp-stage" data-stage="${s.id}"><b>${s.title}</b><span>${s.hint}</span></button>`).join('')).join('');
  $('#rp-stages').querySelectorAll('.rp-stage').forEach(b =>
    b.onclick = () => runStart({ stages: [b.dataset.stage] }));
  markDone();
  refreshRunState();
}

function markDone() {
  const F = RUN.files || {};
  const map = { evaluate: 'doe', tests: 'tests', report: 'report', video: 'video',
                bundle: 'bundle', train: 'ckpt', sweep: 'sweep' };
  Object.entries(map).forEach(([stage, key]) => {
    const el = document.querySelector(`.rp-stage[data-stage="${stage}"]`);
    if (!el || !F[key]) return;
    el.insertAdjacentHTML('beforeend', `<small>готово · ${F[key].mtime_str}</small>`);
  });
}

async function refreshRunState() {
  if (!RUN.available) return;
  try {
    const js = await (await fetch('api/state', { cache: 'no-store' })).json();
    RUN.state = js.state; RUN.files = js.files; RUN.metrics = js.metrics;
    RUN.sets = js.sets || []; RUN.summary = js.summary || RUN.summary;
  } catch (e) { return; }
  const st = RUN.state || {};
  const busy = !!(st.running || st.title);
  document.querySelectorAll('.rp-stage').forEach(b =>
    b.classList.toggle('active', b.dataset.stage === st.running));
  const badge = $('#runner-badge');
  badge.classList.toggle('hidden', !BUSY_LATCH && !busy);
  badge.textContent = busy ? `⏳ ${st.title || st.running}` : 'пульт';
  if (busy) flashStatus('выполняется', 'busy');
  else if (st.returncode === 0) flashStatus('готово', 'ok');
  else if (st.returncode === -15 || st.returncode === -9) flashStatus('остановлено', 'ok');
  else if (st.returncode) flashStatus(`код ${st.returncode}`, 'fail');
  else flashStatus('готов', 'ok');
  const stop = $('#rp-stop');
  if (stop) stop.disabled = !busy;
  BUSY_LATCH = busy;
}

function flashStatus(text, kind) {
  const el = $('#rp-status');
  if (!el) return;
  el.textContent = text;
  el.className = 'rp-status' + (kind === 'busy' ? ' busy' : kind === 'fail' ? ' fail' : '');
}

let BUSY_LATCH = false;

async function runStart(payload) {
  if (!RUN.available) { setTab('run'); return; }
  setTab('run');
  $('#rp-log').textContent = '';
  RUN.since = 0;
  try {
    const r = await fetch('api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const js = await r.json();
    if (!js.ok) { flashStatus(js.message || 'не удалось', 'fail'); return; }
    flashStatus('выполняется', 'busy');
    startLogPoll();
    refreshRunState();
  } catch (e) {
    flashStatus('нет связи с пультом', 'fail');
  }
}

function startLogPoll() {
  if (RUN.poll) return;
  if (location.protocol === 'file:') return;   // офлайн: пульта нет, не стучимся
  const step = async () => {
    let js = null;
    try {
      js = await (await fetch(`api/log?since=${RUN.since}`, { cache: 'no-store' })).json();
    } catch (e) { RUN.poll = setTimeout(step, 1200); return; }
    if (js.lines && js.lines.length) {
      const pre = $('#rp-log');
      pre.insertAdjacentHTML('beforeend', escapeHtml(js.lines.join('')));
      if (pre.textContent.length > 60000) pre.textContent = pre.textContent.slice(-40000);
      RUN.since = js.total;
      pre.scrollTop = pre.scrollHeight;
    }
    if (js.running) { RUN.poll = setTimeout(step, 700); refreshRunState(); }
    else {
      RUN.poll = null;
      await refreshRunState();
      $('#rp-log').insertAdjacentHTML('beforeend', '\n— завершено —\n');
      await afterJob();
    }
  };
  RUN.poll = setTimeout(step, 300);
}

/** После этапа: перечитать бандл/комплекты/метрики, если они изменились. */
async function afterJob() {
  await refreshBucket();
}

async function refreshBucket() {
  try {
    const st = await (await fetch('api/state', { cache: 'no-store' })).json();
    RUN.sets = st.sets || [];
    RUN.summary = st.summary || RUN.summary;
    const before = JSON.stringify(RUN.files || {});
    RUN.files = st.files || {};
    if (before !== JSON.stringify(RUN.files)) await reloadBundle();
    RUN.history = await (await fetch('api/history', { cache: 'no-store' })).json();
    RUN.graph = null; RUN.act = null;
    renderTestTab();
    renderCurves();
  } catch (e) { /* тихо */ }
}

async function reloadBundle() {
  try {
    const r = await fetch('data_bundle.js', { cache: 'no-store' });
    const code = await r.text();
    if (!code || code.length < 1000) return;
    const el = document.createElement('script');
    el.textContent = code;
    document.head.appendChild(el);
    CACHE.frames = {}; CACHE.small = {};
    await loadBundle();
    renderHeader(); renderSamples(); renderDoe(); renderPropStatic(); setZ(S.z);
    renderPlanes(); renderBitbar(); renderProfiles(); renderMetrics(); renderCurves();
    renderTrain(); renderIterGrid();
  } catch (e) { /* остаёмся на прежних данных */ }
}

/* --------------------------------------------------------- загрузка картинок */
async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const fd = new FormData();
  [...fileList].slice(0, 4).forEach((f, i) => fd.append(`file${i}`, f, f.name));
  hint('загрузка…');
  try {
    const r = await fetch('api/sets', { method: 'POST', body: fd });
    const js = await r.json();
    if (!js.ok) { hint(js.message || 'не удалось', true); return; }
    RUN.currentSet = js.id;
    await refreshBucket();
    const st = await (await fetch('api/state')).json();
    RUN.sets = st.sets || [];
    hint(`комплект ${js.id} готов · ${js.info.sources.length} файл(ов) · жмите «прогнать сеть»`);
    renderSetSelect();
    setTab('test');
    await loadSetIntoPhysics(js.id);
  } catch (e) { hint('нет связи с пультом', true); }
}

async function loadSetIntoPhysics(setId) {
  // подтягиваем цели комплекта в панель «физика» и в бандл-данные
  const imgs = await Promise.all([1, 2, 3, 4].map(i => new Promise(res => {
    const im = new Image();
    // страховка: битая или не отданная картинка не должна вешать интерфейс
    const timer = setTimeout(() => res(null), 4000);
    im.onload = () => { clearTimeout(timer); res(im); };
    im.onerror = () => { clearTimeout(timer); res(null); };
    im.src = `api/asset?set=${encodeURIComponent(setId)}&name=target_${i}.png&t=${Date.now()}`;
  })));
  if (imgs.some(i => !i)) {
    // картинки комплекта не дошли — показываем прежние цели, но вкладку не ломаем
    setTab('test');
    return;
  }
  const arr = new Uint8Array(N * N * 4);
  imgs.forEach((im, k) => {
    const c = document.createElement('canvas');
    c.width = c.height = N;
    const ctx = c.getContext('2d');
    ctx.drawImage(im, 0, 0);
    const d = ctx.getImageData(0, 0, N, N).data;
    for (let i = 0; i < N * N; i++) arr[k * N * N + i] = d[i * 4];
  });
  D.targets = arr;
  CACHE.frames = {};
  renderSamples();
  renderTestTab();
}

function hint(text, bad) {
  const el = $('#load-hint');
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('bad', !!bad);
}

function renderSetSelect() {
  const sel = $('#set-select');
  if (!sel) return;
  const sets = RUN.sets || [];
  sel.innerHTML = `<option value="">сдаваемый набор</option>` +
    sets.map(s => `<option value="${s.id}">${s.id}</option>`).join('');
  sel.value = RUN.currentSet || '';
  sel.onchange = () => { RUN.currentSet = sel.value || null; if (sel.value) loadSetIntoPhysics(sel.value); };
  $('#set-delete').onclick = async () => {
    if (!RUN.currentSet) return;
    if (!confirm(`удалить комплект ${RUN.currentSet}?`)) return;
    await fetch(`api/sets/delete?id=${encodeURIComponent(RUN.currentSet)}`, { method: 'POST' });
    RUN.currentSet = null;
    await refreshBucket();
    renderSetSelect();
    hint('комплект удалён');
  };
}

/* --------------------------------------------------------- анимация загрузки */
function playFlow() {
  const cap = $('#flow-cap');
  const stages = ['исходник', 'яркость', 'кроп и 128×128', 'полярность и 8 бит'];
  let i = 0;
  const cv = ['#flow-in', '#flow-gray', '#flow-crop', '#flow-target'].map(s => $(s));
  cv.forEach((c, k) => { c.style.opacity = k === 0 ? '1' : '.15'; });
  cap.textContent = stages[0];
  const t = setInterval(() => {
    i++;
    if (i >= stages.length) { clearInterval(t); return; }
    cv[i].style.opacity = '1';
    cap.textContent = stages[i];
  }, 550);
}

function drawFlow() {
  // четыре кадра пути подготовки (их считает hdoe/prepare.py: исходник →
  // яркость → кроп → цель 128×128 в 8 битах). Для своего комплекта берём цели.
  const cells = ['#flow-in', '#flow-gray', '#flow-crop', '#flow-target'];
  const steps = D.sourceSteps;
  cells.forEach((sel, i) => {
    const cv = $(sel);
    if (!cv) return;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, 96, 96);
    if (steps && steps.length >= 4 * N * N) {
      drawScaled(ctx, toCanvas(steps, i * N * N), 96, 96);
    } else {
      drawScaled(ctx, targetFrame(Math.min(i, NP - 1)), 96, 96);
    }
    ctx.strokeStyle = 'rgba(90,130,175,.4)';
    ctx.strokeRect(0.5, 0.5, 95, 95);
  });
}

/* =================================================================== запуск */
async function probeRunner() {
  if (location.protocol === 'file:') return false;
  try {
    const r = await fetch('api/state', { cache: 'no-store' });
    if (!r.ok) return false;
    const js = await r.json();
    RUN.available = true;
    RUN.stages = js.stages || [];
    RUN.pipelines = js.pipelines || [];
    RUN.sets = js.sets || [];
    RUN.state = js.state || {};
    RUN.files = js.files || {};
    RUN.metrics = js.metrics || null;
    RUN.summary = js.summary || null;
    RUN.python = js.python;
    const badge = $('#runner-badge');
    if (badge) badge.classList.toggle('hidden', false);
    return true;
  } catch (e) { return false; }
}

function wire() {
  $$('#tabs button').forEach(b => b.onclick = () => setTab(b.dataset.tab));
  $$('#variant-switch button').forEach(b => b.onclick = () => {
    S.variant = b.dataset.variant;
    $$('#variant-switch button').forEach(x => x.classList.toggle('active', x === b));
    renderDoe(); renderPlanes(); renderMetrics(); renderProfiles(); drawSsim(); renderTrain();
  });
  $$('#plane-mode button').forEach(b => b.onclick = () => {
    S.planeMode = b.dataset.mode;
    $$('#plane-mode button').forEach(x => x.classList.toggle('active', x === b));
    renderPlanes();
  });
  $$('#protocol-switch button').forEach(b => b.onclick = () => {
    S.protocol = b.dataset.proto;
    $$('#protocol-switch button').forEach(x => x.classList.toggle('active', x === b));
    renderMetrics();
  });
  $$('#curve-switch button').forEach(b => b.onclick = () => {
    S.curve = b.dataset.curve;
    $$('#curve-switch button').forEach(x => x.classList.toggle('active', x === b));
    renderCurves();
  });
  $('#prop-play').onclick = () => {
    S.playing = !S.playing;
    $('#prop-play').textContent = S.playing ? '⏸' : '▶';
  };
  $('#prop-speed').oninput = e => { S.speed = +e.target.value; $('#prop-speed-val').textContent = '×' + S.speed; };
  $('#prop-loop').onchange = e => { S.loop = e.target.checked; };
  $('#prop-autopause').onchange = e => { S.autopause = e.target.checked; };
  $('#prop-slider').oninput = e => {
    S.playing = false; $('#prop-play').textContent = '▶';
    setZ(Math.round((+e.target.value / 100) * (D.zgrid.length - 1)));
  };
  $('#doe-quant').onchange = e => { S.quant = e.target.checked; renderDoe(); };
  $('#doe-flash').onclick = () => { S.quant = !S.quant; $('#doe-quant').checked = S.quant; renderDoe(); };
  $('#pick-files').onclick = () => $('#file-input').click();
  $('#file-input').onchange = e => uploadFiles(e.target.files);
  $('#pick-demo').onclick = () => { RUN.currentSet = null; hint('показан сдаваемый набор'); renderSetSelect(); };
  $('#flow-play').onclick = playFlow;
  $('#test-run').onclick = () => {
    if (!RUN.available) { flashStatus('нужен пульт', 'fail'); return; }
    if (!RUN.currentSet) { setTab('physics'); hint('сначала загрузите комплект', true); return; }
    fetch('api/infer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ set: RUN.currentSet }),
    }).then(r => r.json()).then(js => {
      if (!js.ok) { flashStatus(js.message, 'fail'); return; }
      startLogPoll();
    });
  };
  $('#act-play').onclick = () => {
    S.actPlaying = !S.actPlaying;
    $('#act-play').textContent = S.actPlaying ? '⏸' : '▶';
    if (S.actPlaying) { S.actIdx = 0; }
  };
  $('#net-play').onclick = () => {
    S.netPlaying = !S.netPlaying;
    $('#net-play').textContent = S.netPlaying ? '⏸' : '▶ сигнал';
  };
  $('#net-speed').oninput = e => { S.netSpeed = +e.target.value; };
  $('#rp-stop').onclick = async () => {
    flashStatus('останавливаю…', 'busy');
    try {
      const r = await fetch('api/stop', { method: 'POST' });
      const js = await r.json();
      flashStatus(js.stopped ? 'остановлено' : 'нечего останавливать', js.stopped ? 'ok' : 'fail');
      RUN.poll = null;
      await refreshRunState();
    } catch (e) { flashStatus('нет связи', 'fail'); }
  };
  $('#rp-copy').onclick = () => {
    const t = $('#rp-log').textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject())
      .then(() => flashStatus('скопировано', 'ok')).catch(() => flashStatus('нет доступа', 'fail'));
  };
  const closeModal = () => $('#modal').classList.add('hidden');
  $('#modal-close').onclick = closeModal;
  $('#modal').onclick = e => { if (e.target.id === 'modal') $('#modal').classList.add('hidden'); };
  // drag&drop на панель изображений
  const drop = $('#tab-physics');
  ['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); $('#load-hint').classList.add('hover');
  }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); $('#load-hint').classList.remove('hover');
  }));
  drop.addEventListener('drop', e => {
    if (e.dataTransfer && e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === ' ') { e.preventDefault(); $('#prop-play').click(); }
    if (e.key === 'ArrowRight') setZ(S.z + 1);
    if (e.key === 'ArrowLeft') setZ(S.z - 1);
    if (e.key === 'n' || e.key === 'N') $$('#variant-switch button')[0].click();
    if (e.key === 'g' || e.key === 'G') $$('#variant-switch button')[1].click();
    if (e.key === '1') setTab('physics');
    if (e.key === '2') setTab('test');
    if (e.key === '3') setTab('algo');
    if (e.key === '4') setTab('model');
    if (e.key === '5') setTab('run');
  });
  window.addEventListener('resize', () => {
    renderDoe(); renderPlanes(); renderProfiles(); renderMetrics(); renderCurves();
    drawTrainCurve(); drawNetGraph(); drawWeights(); drawHistMap();
  });
}

/* ------------------------------------------------------- гистограммы весов */
function drawHistMap() {
  const cv = $('#hist-canvas');
  if (!cv) return;
  const W = cv.width = cv.clientWidth * dpr();
  const H = cv.height = 220 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const g = RUN.graph;
  if (!g || !g.layers || !g.layers.length) {
    ctx.fillStyle = 'rgba(160,185,215,.7)';
    ctx.font = `${11 * dpr()}px ui-monospace, monospace`;
    ctx.fillText('гистограммы весов доступны под пультом', 10, 20);
    return;
  }
  const rows = Math.min(g.layers.length, 28);
  const rh = H / rows;
  for (let r = 0; r < rows; r++) {
    const l = g.layers[Math.floor(r * g.layers.length / rows)];
    l.hist.forEach((v, i) => {
      const x = (i / l.hist.length) * W;
      ctx.fillStyle = `rgba(122,162,255,${(0.08 + 0.9 * v).toFixed(3)})`;
      ctx.fillRect(x, r * rh + 1, W / l.hist.length, rh - 2);
    });
  }
  $('#hist-probe').innerHTML = `${rows} слоёв × ${g.layers[0].hist.length} бинов`;
}

/* =================================================================== main */
async function main() {
  try {
    await loadBundle();
  } catch (err) {
    document.body.insertAdjacentHTML('afterbegin',
      `<div class="bundle-err"><b>Не удалось загрузить данные.</b>
       Откройте index.html двойным кликом (рядом должен лежать data_bundle.js)
       или запустите <code>python scripts/serve_dashboard.py</code>.
       <br><small>${escapeHtml(err)}</small></div>`);
    return;
  }
  await probeRunner();
  if (RUN.available) {
    try { RUN.history = await (await fetch('api/history', { cache: 'no-store' })).json(); } catch (e) { RUN.history = null; }
  }
  renderHeader();
  renderSamples();
  renderDoe();
  renderPropStatic();
  setZ(S.z);
  renderPlanes();
  renderBitbar();
  renderProfiles();
  renderMetrics();
  renderTrain();
  renderIterGrid();
  renderCurves();
  renderRunTab();
  renderSetSelect();
  drawFlow();
  wire();
  $('#runner-badge').classList.toggle('hidden', !RUN.available);
  $('#footer-info').textContent = RUN.available
    ? `пульт · ${(RUN.python || '').split('/').slice(-3).join('/')}`
    : `офлайн · seed ${D.manifest.constants.seed}`;
  setTab('physics');
  requestAnimationFrame(tick);
}

window.HDOE = { S, D, RUN, setTab, renderMetrics, renderCurves, uploadFiles, refreshBucket };
document.addEventListener('DOMContentLoaded', main);
