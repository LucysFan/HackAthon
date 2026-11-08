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

const D = {};                    // данные активного комплекта
let DEMO = null;                 // неизменный сдаваемый набор из бандла
const CACHE = { frames: {}, small: {} };
const RUN = { mode: 'off', available: false, sets: [], stages: [], pipelines: [],
              poll: null, since: 0, state: {}, files: {}, metrics: null,
              graph: null, act: null, actSet: null, history: null, currentSet: null,
              loadSeq: 0, starting: false, graphError: null, actError: null, job: null };

/* ------------------------------------------------------------------ утилиты */
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const fmt = (v, d = 3) => (v === undefined || v === null || Number.isNaN(v)) ? '—' : (+v).toFixed(d);
const dpr = () => window.devicePixelRatio || 1;

function toCanvas(data, offset, scale = 1) {
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(N, N);
  const px = img.data;
  for (let i = 0; i < N * N; i++) {
    const o = i * 4;
    const v = data[offset + i] * scale;
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
  if (!CACHE.frames[key]) CACHE.frames[key] = toCanvas(src, i * N * N, 255);
  return CACHE.frames[key];
}

function phaseCodes() {
  return S.variant === 'gs' ? D.phase.gs : D.phase.nn;
}

function hasResult() { return !!(D.doe.nn && D.doe.nn.length); }
function activeRows() { return (D.manifest.file_metrics || {})[S.variant] || []; }
function activeMean() { const m = D.manifest.metrics || {}; return (S.variant === 'gs' ? (m.gs || {}).mean : m.mean) || {}; }
function rememberDemo() { DEMO = { ...D }; }
function clearCanvas(cv, text) {
  if (!cv) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!text) return;
  ctx.fillStyle = '#8ea1c2'; ctx.font = `${11 * dpr()}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  const words = text.split(' '), lines = []; let line = '';
  for (const word of words) {
    if ((line + ' ' + word).length > 24 && line) { lines.push(line); line = word; }
    else line = line ? line + ' ' + word : word;
  }
  if (line) lines.push(line);
  lines.forEach((l, i) => ctx.fillText(l, cv.width / 2, cv.height / 2 + (i - (lines.length - 1) / 2) * 16));
  ctx.textAlign = 'left';
}
function notify(text, kind = 'ok') {
  const el = $('#job-notice');
  if (!el) return;
  el.textContent = text;
  el.className = 'job-notice ' + kind;
  el.classList.toggle('hidden', !text);
}
async function fetchJSON(url, options) {
  const r = await fetch(url, options);
  let js;
  try { js = await r.json(); } catch (e) { throw new Error(`сервер вернул не JSON (${r.status})`); }
  if (!r.ok || js.ok === false) throw new Error(js.message || js.error || `ошибка сервера ${r.status}`);
  return js;
}
function updateJobButtons() {
  const busy = !!(RUN.state && RUN.state.running) || RUN.starting;
  const inferBusy = busy && (RUN.state.kind === 'infer' || RUN.starting);
  for (const id of ['test-run', 'physics-run']) {
    const b = $('#' + id);
    if (!b) continue;
    b.disabled = busy || !RUN.available;
    b.textContent = inferBusy ? '⏳ считаю…' : 'прогнать сеть';
  }
  const stop = $('#rp-stop');
  if (stop) stop.disabled = !busy;
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
    `K = <b>${m.steps || (m.phys || {}).steps || '—'}</b>`,
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
  if (!hasResult()) {
    clearCanvas($('#doe-phase'), 'Маска появится после запуска сети');
    clearCanvas($('#doe-gray'), 'Нет маски');
    drawHistogram(new Uint8Array(0), null);
    $('#doe-probe').textContent = 'Комплект загружен, результат ещё не рассчитан';
    return;
  }
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
  if (!stack || !stack.length) {
    clearCanvas(cv, 'Прогнать сеть → дифракция выбранной картинки');
    $('#prop-probe').textContent = 'Это панель результата. Старая демо-дифракция скрыта.';
    $('#prop-slider').value = 0;
    return;
  }
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
    if (!hasResult()) {
      ctx.drawImage(targetFrame(i), 0, 0);
    } else if (S.planeMode === 'side') {
      ctx.drawImage(targetFrame(i), 0, 0, N, N);
      ctx.globalAlpha = 0.5;
      ctx.drawImage(reconFrame(i), 0, 0, N, N);
      ctx.globalAlpha = 1;
    } else if (S.planeMode === 'diff') {
      const img = ctx.createImageData(N, N);
      const t = D.targets, r = D.recon[S.variant];
      for (let k = 0; k < N * N; k++) {
        const d = Math.abs(t[i * N * N + k] / 255 - r[i * N * N + k]);
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
    tag.textContent = !hasResult() ? `z${i + 1} · цель · ожидает запуска` : `z${i + 1} · ${m && m[i] ? fmt(m[i].ssim, 3) : ''}`;
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
    : (!hasResult() ? 'Цели выбранного комплекта. Нажмите «прогнать сеть» для расчёта маски и дифракции.' : '');
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
    if (!src || !src.length) return;
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
  if (!rec || !rec.length) {
    ['prof-h', 'prof-v', 'zprobe'].forEach(id => clearCanvas($('#' + id)));
    $('#prof-probe').textContent = 'Профили появятся после расчёта';
    return;
  }
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
    const x = (i / Math.max(1, vals.length - 1)) * W;
    const y = H - 4 - ((v - lo) / (hi - lo)) * (H - 8);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 1.6 * dpr(); ctx.stroke();
}

function drawSsim() {
  const cv = $('#ssim-canvas');
  if (!cv) return;
  const src = D.ssimMap[S.variant];
  if (!src || !src.length) { clearCanvas(cv, 'Нет результата'); return; }
  const mapped = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) mapped[i] = (src[S.plane * N * N + i] + 1) * 127.5;
  drawScaled(cv.getContext('2d'), toCanvas(mapped, 0), cv.width, cv.height);
}

/* =================================================================== 2. ТЕСТ */
function setLabel(id) { return id === 'default' || !id ? 'сдаваемый набор' : id; }
function renderTestTab() {
  const sel = $('#test-set');
  const sets = RUN.sets || [];
  const sig = sets.map(s => `${s.id}:${s.mtime || 0}`).join(',');
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.innerHTML = sets.map(s => {
      const m = s.mean ? ` · SSIM ${fmt(s.mean.ssim, 3)}` : '';
      return `<option value="${escapeHtml(s.id)}">${escapeHtml(setLabel(s.id))}${m}</option>`;
    }).join('') || '<option value="default">сдаваемый набор</option>';
  }
  if (!RUN.currentSet) RUN.currentSet = sets.some(s => s.id === 'default') ? 'default' : null;
  sel.value = RUN.currentSet || 'default';
  sel.onchange = () => selectSet(sel.value);
  $('#test-thumbs').innerHTML = '';
  for (let i = 0; i < NP; i++) {
    const d = document.createElement('div');
    d.className = 'thumb';
    const c = document.createElement('canvas');
    c.width = c.height = N;
    c.getContext('2d').drawImage(targetFrame(i), 0, 0);
    d.appendChild(c);
    d.insertAdjacentHTML('beforeend', `<span>z${i + 1}</span>`);
    $('#test-thumbs').appendChild(d);
  }
  const mean = activeMean();
  $('#test-probe').innerHTML = `комплект <b>${escapeHtml(setLabel(RUN.currentSet))}</b>` +
    ` · SSIM <b>${fmt(mean.ssim, 3)}</b> · Pearson <b>${fmt(mean.pcc, 3)}</b>` +
    (hasResult() ? ' · DOE готов' : ' · ожидает запуска сети');
  const doeBox = $('#test-doe');
  doeBox.innerHTML = '';
  const download = $('#test-download');
  download.disabled = !hasResult();
  if (hasResult()) {
    const c = document.createElement('canvas');
    c.width = c.height = N;
    drawScaled(c.getContext('2d'), toCanvas(D.doe[S.variant], 0), N, N);
    c.className = 'doe-img';
    doeBox.appendChild(c);
    doeBox.insertAdjacentHTML('beforeend', '<div class="plot-title">DOE.png этого комплекта · 128×128 · 8 бит</div>');
    download.onclick = () => downloadCanvas(c, 'DOE.png');
  } else {
    doeBox.innerHTML = '<div class="hint">Маска ещё не рассчитана. Нажмите «прогнать сеть».</div>';
    download.onclick = null;
  }
  renderTestRecon();
  renderMetrics();
  renderCurves();
  renderSetsTable();
  updateJobButtons();
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
    selectSet(id);
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

function renderTestRecon() {
  const box = $('#test-recon');
  box.innerHTML = '';
  const rows = activeRows();
  for (let i = 0; i < NP; i++) {
    const cell = document.createElement('div');
    cell.className = 'recon-cell';
    const c = document.createElement('canvas');
    c.width = c.height = N;
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(hasResult() ? reconFrame(i) : targetFrame(i), 0, 0);
    cell.appendChild(c);
    const m = rows[i];
    cell.insertAdjacentHTML('beforeend', hasResult()
      ? `<span>z${i + 1} · SSIM ${fmt(m && m.ssim, 3)}<br>Pearson ${fmt(m && m.pcc, 3)}</span>`
      : `<span>z${i + 1} · цель<br>ожидает запуска</span>`);
    box.appendChild(cell);
  }
}

function renderMetrics() {
  const key = r => (S.protocol === 'primary' ? r.ssim : (r.ssim_raw ?? r.ssim));
  const rows = activeRows();
  const active = rows[S.plane] || {};
  const mean = activeMean();
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
    </div>`).join('') + (rows.length ? `
    <div class="bar-row"><span class="bar-label">∅</span>
      <div class="bar alt"><i style="width:${(clamp(key(mean), 0, 1) * 100).toFixed(1)}%"></i></div>
      <span class="bar-val">${fmt(key(mean), 3)}</span></div>` : '<div class="hint">Метрики появятся после запуска сети.</div>');
  bars.querySelectorAll('.bar-row[data-plane]').forEach(el => el.onclick = () => setPlane(+el.dataset.plane));
  const t = $('#metrics-tables');
  t.innerHTML = rows.length ? `<table><tr><th>z</th><th>мм</th><th>SSIM</th><th>Pearson</th><th>SSIM без норм.</th></tr>` +
    rows.map(r => `<tr data-plane="${r.plane - 1}"><td>z${r.plane}</td><td>${fmt(r.z_mm, 1)}</td>
      <td>${fmt(r.ssim, 4)}</td><td>${fmt(r.pcc, 4)}</td><td>${fmt(r.ssim_raw, 4)}</td></tr>`).join('') +
    `<tr class="sum"><td>∅</td><td></td><td>${fmt(mean.ssim, 4)}</td><td>${fmt(mean.pcc, 4)}</td>
      <td>${fmt(mean.ssim_raw, 4)}</td></tr></table>` : '';
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
  const mean = ((DEMO || D).manifest.metrics || {}).mean || {};
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
  if (!row.length) { box.innerHTML = '<div class="hint">Траектория физического слоя показана для сдаваемого набора.</div>'; return; }
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
  if (!RUN.available) { drawNetGraph(); drawHistMap(); return; }
  if (RUN.graph && RUN.graph.layers && RUN.graph.layers.length) {
    drawNetGraph(); drawWeights(); drawHistMap(); return;
  }
  if (RUN.graphPromise) return RUN.graphPromise;
  $('#model-status').textContent = 'Читаю веса outputs/best.pt…';
  RUN.graphPromise = (async () => {
    try {
      RUN.graph = await fetchJSON('api/model/graph', { cache: 'no-store' });
      RUN.graphError = null;
      populateLayerSelect();
    } catch (e) {
      RUN.graph = null; RUN.graphError = String(e.message || e);
    } finally {
      RUN.graphPromise = null;
      drawNetGraph(); drawWeights(); drawHistMap();
    }
  })();
  return RUN.graphPromise;
}

function populateLayerSelect() {
  const sel = $('#hist-layer');
  if (!sel || !RUN.graph) return;
  sel.innerHTML = RUN.graph.layers.map((l, i) =>
    `<option value="${i}">${escapeHtml(l.name.replace(/^net\./, ''))} · ${l.shape.join('×')}</option>`).join('');
  sel.onchange = () => { S.histLayer = +sel.value; drawHistMap(); };
  S.histLayer = clamp(S.histLayer || 0, 0, RUN.graph.layers.length - 1);
  sel.value = S.histLayer;
}

async function ensureAct() {
  const sid = RUN.currentSet || 'default';
  if (!RUN.available) { drawAct(); return; }
  if (RUN.act && RUN.actSet === sid && RUN.act.stages.length) { drawAct(); return; }
  if (RUN.actPromise) return RUN.actPromise;
  RUN.act = null; RUN.actError = null; S.actIdx = 0; drawAct();
  RUN.actPromise = (async () => {
    try {
      const data = await fetchJSON(`api/model/act?set=${encodeURIComponent(sid)}`, { cache: 'no-store' });
      if (sid === (RUN.currentSet || 'default')) { RUN.act = data; RUN.actSet = sid; }
    } catch (e) { RUN.actError = String(e.message || e); }
    finally { RUN.actPromise = null; drawAct(); }
  })();
  return RUN.actPromise;
}

function drawNetGraph() {
  const cv = $('#net-canvas');
  if (!cv) return;
  const W = cv.width = Math.max(320, cv.clientWidth) * dpr();
  const H = cv.height = 420 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const g = RUN.graph;
  const status = $('#model-status');
  if (!g || !g.layers || !g.layers.length) {
    const text = RUN.graphError ? `Не удалось прочитать веса: ${RUN.graphError}` :
      RUN.available ? 'Загрузка весов…' : 'Веса доступны через сервер, не в офлайн-файле';
    clearCanvas(cv, text);
    if (status) status.textContent = text;
    $('#net-probe').textContent = RUN.graphError || 'outputs/best.pt';
    return;
  }
  // Один узел — блок сети, а не 67 тензоров в одной точке.
  const blocks = g.blocks || [];
  const enc = blocks.filter(b => b.group === 'encoder');
  const dec = blocks.filter(b => b.group === 'decoder');
  const stage = [];
  enc.forEach((l, i) => stage.push({ l, x: 0.09 + 0.72 * i / Math.max(1, enc.length - 1), y: 0.27 }));
  dec.forEach((l, i) => stage.push({ l, x: 0.70 - 0.51 * i / Math.max(1, dec.length - 1), y: 0.70 }));
  const head = blocks.find(b => b.group === 'head');
  if (head) stage.push({ l: head, x: 0.91, y: 0.70 });
  const byName = Object.fromEntries(stage.map(n => [n.l.name, n]));
  const maxN = Math.max(...blocks.map(b => b.numel), 1);
  // Рёбра до узлов; skip-связи пунктиром.
  (g.edges || []).forEach(([a, b, kind]) => {
    const A = byName[a], B = byName[b];
    if (!A || !B) return;
    ctx.beginPath(); ctx.moveTo(A.x * W, A.y * H); ctx.lineTo(B.x * W, B.y * H);
    ctx.setLineDash(kind === 'skip' ? [5 * dpr(), 5 * dpr()] : []);
    ctx.strokeStyle = kind === 'skip' ? 'rgba(255,179,71,.55)' : 'rgba(122,162,255,.6)';
    ctx.lineWidth = 1.5 * dpr(); ctx.stroke();
  });
  ctx.setLineDash([]);
  stage.forEach((n, i) => {
    const x = n.x * W, y = n.y * H;
    const r = (10 + 10 * Math.sqrt(n.l.numel / maxN)) * dpr();
    const pulse = S.netPlaying ? 0.65 + 0.35 * Math.sin(S.netT / 350 - i) : 0.85;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = n.l.group === 'decoder' ? `rgba(122,162,255,${pulse})` : `rgba(77,214,193,${pulse})`;
    ctx.fill();
    ctx.strokeStyle = '#203b55'; ctx.lineWidth = 2 * dpr(); ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillStyle = '#dce9fa';
    ctx.font = `${10 * dpr()}px system-ui, sans-serif`;
    ctx.fillText(n.l.label, x, y - r - 8 * dpr());
    ctx.fillStyle = '#8ea1c2'; ctx.font = `${9 * dpr()}px ui-monospace, monospace`;
    ctx.fillText((n.l.numel / 1e6).toFixed(2) + ' М', x, y + r + 13 * dpr());
  });
  ctx.textAlign = 'left'; ctx.fillStyle = '#8ea1c2'; ctx.font = `${10 * dpr()}px system-ui, sans-serif`;
  ctx.fillText('Энкодер: сжатие признаков', 12 * dpr(), 20 * dpr());
  ctx.fillText('Декодер + skip-связи (пунктир)', 12 * dpr(), H * 0.52);
  cv.onclick = e => {
    const rect = cv.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * W, y = (e.clientY - rect.top) / rect.height * H;
    let best = null, dist = Infinity;
    stage.forEach(n => { const d = Math.hypot(x - n.x * W, y - n.y * H); if (d < dist) { dist = d; best = n; } });
    if (!best || dist > 38 * dpr()) return;
    const idx = g.layers.findIndex(l => l.block === best.l.name);
    if (idx >= 0) { S.histLayer = idx; $('#hist-layer').value = idx; drawHistMap(); }
  };
  const mb = g.checkpoint ? (g.checkpoint.size / 1048576).toFixed(1) : '—';
  if (status) status.textContent = `✓ outputs/best.pt загружен · ${mb} МБ`;
  $('#net-probe').innerHTML = `<b>${(g.params / 1e6).toFixed(2)} М</b> параметров · ${g.layers.length} тензоров · ${blocks.length} блоков`;
}

function stepNet(dt) {
  if (!S.netPlaying) return;
  S.netT += dt * (S.netSpeed || 1);
  drawNetGraph();
}

function drawAct() {
  const box = $('#act-grid');
  if (!box) return;
  const act = RUN.act;
  if (!act || !act.stages || !act.stages.length) {
    box.innerHTML = `<div class="hint">${escapeHtml(RUN.actError || (RUN.available ? 'Считаю активации выбранного комплекта…' : 'Активации доступны через сервер'))}</div>`;
    return;
  }
  if (box.dataset.sig !== RUN.actSet + ':' + act.stages.length) {
    box.dataset.sig = RUN.actSet + ':' + act.stages.length;
    box.innerHTML = '';
    act.stages.forEach((st, i) => {
      const d = document.createElement('div');
      d.className = 'act-cell';
      d.innerHTML = `<canvas width="64" height="64"></canvas><span>${escapeHtml(st.name)}</span>`;
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
  if (!S.actPlaying || !RUN.act || !RUN.act.stages || !RUN.act.stages.length) return;
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
  let js;
  try { js = await fetchJSON('api/state', { cache: 'no-store' }); }
  catch (e) { return; }
  RUN.state = js.state || {}; RUN.files = js.files || {}; RUN.metrics = js.metrics;
  RUN.sets = js.sets || []; RUN.summary = js.summary || RUN.summary;
  const st = RUN.state;
  const busy = !!st.running;
  document.querySelectorAll('.rp-stage').forEach(b => {
    b.classList.toggle('active', b.dataset.stage === st.running);
    b.disabled = busy;
  });
  $$('#rp-pipelines button').forEach(b => { b.disabled = busy; });
  const badge = $('#runner-badge');
  badge.classList.toggle('hidden', !RUN.available);
  badge.textContent = busy ? `⏳ ${st.title || st.running}` : 'пульт';
  if (busy) flashStatus(`выполняется: ${st.title || st.running}`, 'busy');
  else if (st.returncode === 0) flashStatus('готово', 'ok');
  else if (st.returncode === -15 || st.returncode === -9) flashStatus('остановлено', 'ok');
  else if (st.returncode) flashStatus(`ошибка · код ${st.returncode}`, 'fail');
  else flashStatus('готов', 'ok');
  updateJobButtons();
  if (busy && !RUN.poll && !RUN.starting) startLogPoll(st.job);
  // После перезагрузки страницы тоже подхватываем уже идущий расчёт.
  if (busy && st.kind === 'infer' && st.set_id) RUN.currentSet = st.set_id;
}

function flashStatus(text, kind) {
  const el = $('#rp-status');
  if (el) {
    el.textContent = text;
    el.className = 'rp-status' + (kind === 'busy' ? ' busy' : kind === 'fail' ? ' fail' : '');
  }
  const test = $('#test-status');
  if (test) { test.textContent = text; test.className = 'inline-status ' + (kind || 'ok'); }
}

async function runStart(payload) {
  if (!RUN.available) { setTab('run'); notify('Для запуска задач нужен сервер.', 'fail'); return; }
  if (RUN.starting || RUN.state.running) { notify('Дождитесь окончания текущей задачи.', 'busy'); return; }
  setTab('run');
  RUN.starting = true; updateJobButtons();
  try {
    const js = await fetchJSON('api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    flashStatus('выполняется', 'busy');
    startLogPoll(js.job);
    await refreshRunState();
  } catch (e) {
    flashStatus(String(e.message || e), 'fail'); notify(String(e.message || e), 'fail');
  } finally { RUN.starting = false; updateJobButtons(); }
}

async function runSelectedSet() {
  if (!RUN.available) { notify('Откройте дашборд через сервер: кнопке нужен доступ к весам.', 'fail'); return; }
  if (RUN.starting || RUN.state.running) { notify('Уже выполняется задача. Дождитесь её окончания или нажмите «стоп».', 'busy'); return; }
  await ensureCurrentSet();
  const sid = RUN.currentSet;
  if (!sid) { notify('Сначала загрузите комплект изображений.', 'fail'); return; }
  RUN.starting = true; updateJobButtons();
  notify(`Запускаю сеть для ${setLabel(sid)}…`, 'busy');
  try {
    const js = await fetchJSON('api/infer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ set: sid }),
    });
    RUN.state = { ...RUN.state, running: 'infer_set.py', kind: 'infer', set_id: sid, job: js.job };
    flashStatus('сеть считает · результат обновится автоматически', 'busy');
    startLogPoll(js.job);
  } catch (e) {
    flashStatus(String(e.message || e), 'fail'); notify(String(e.message || e), 'fail');
  } finally { RUN.starting = false; updateJobButtons(); }
}

function stopLogPoll() {
  if (RUN.poll) clearTimeout(RUN.poll);
  RUN.poll = null;
  RUN.pollSeq = (RUN.pollSeq || 0) + 1;
}

function startLogPoll(job) {
  if (location.protocol === 'file:') return;
  stopLogPoll();
  RUN.since = 0; RUN.job = job || null;
  $('#rp-log').textContent = '';
  const seq = RUN.pollSeq;
  const step = async () => {
    if (seq !== RUN.pollSeq) return;
    let js;
    try { js = await fetchJSON(`api/log?since=${RUN.since}`, { cache: 'no-store' }); }
    catch (e) { if (seq === RUN.pollSeq) RUN.poll = setTimeout(step, 1200); return; }
    if (seq !== RUN.pollSeq) return;
    if (RUN.job && js.job !== RUN.job) { RUN.poll = setTimeout(step, 300); return; }
    RUN.job = js.job;
    if (js.lines && js.lines.length) {
      const pre = $('#rp-log');
      pre.insertAdjacentHTML('beforeend', escapeHtml(js.lines.join('')));
      if (pre.textContent.length > 60000) pre.textContent = pre.textContent.slice(-40000);
      RUN.since = js.total;
      pre.scrollTop = pre.scrollHeight;
      const last = js.lines.filter(l => l.includes('физика:')).pop();
      if (last && js.kind === 'infer') notify(last.trim(), 'busy');
    }
    if (js.running) {
      RUN.state = { ...RUN.state, ...js };
      updateJobButtons();
      RUN.poll = setTimeout(step, 700);
    } else {
      RUN.poll = null;
      await refreshRunState();
      const rc = RUN.state.returncode;
      $('#rp-log').insertAdjacentHTML('beforeend', '\n— завершено —\n');
      if (rc === 0) {
        notify('Готово. Обновляю маску, дифракцию и метрики…', 'busy');
        await afterJob(js);
        notify(`Готово · ${setLabel(js.set_id || RUN.currentSet)}. Маска и дифракция обновлены.`, 'ok');
      } else if (rc === -15 || rc === -9) {
        notify('Расчёт остановлен. Прежние результаты сохранены.', 'ok');
        await refreshBucket();
      } else {
        const detail = (js.lines || []).slice(-4).join(' ').trim();
        notify(`Расчёт завершился с ошибкой (код ${rc}). ${detail} Полный лог — на вкладке «Запуск».`, 'fail');
      }
    }
  };
  RUN.poll = setTimeout(step, 150);
}

async function afterJob(js = {}) {
  await refreshBucket();
  if (js.kind === 'infer' && js.set_id) {
    RUN.currentSet = js.set_id;
    RUN.defaultDirty = js.set_id === 'default';
    await loadSetIntoPhysics(js.set_id, true);
    RUN.act = null; RUN.actSet = null;
    if (S.tab === 'model') ensureAct();
  } else if (!RUN.viewId) {
    await reloadBundle();
  }
  renderSetSelect(); renderTestTab();
}

async function refreshBucket() {
  const st = await fetchJSON('api/state', { cache: 'no-store' });
  RUN.sets = st.sets || []; RUN.summary = st.summary || RUN.summary;
  const before = JSON.stringify(RUN.files || {});
  RUN.files = st.files || {}; RUN.state = st.state || {};
  if (before !== JSON.stringify(RUN.files)) {
    RUN.graph = null; RUN.act = null; RUN.actSet = null;
    try { RUN.history = await fetchJSON('api/history', { cache: 'no-store' }); } catch (e) { /* необязательно */ }
    if (!RUN.viewId) await reloadBundle();
    if (S.tab === 'model') { ensureGraph(); ensureAct(); }
  }
  renderSetSelect(); renderTestTab(); renderCurves(); updateJobButtons();
}

async function reloadBundle() {
  const r = await fetch('data_bundle.js', { cache: 'no-store' });
  if (!r.ok) throw new Error('не удалось прочитать data_bundle.js');
  const code = await r.text();
  const el = document.createElement('script');
  el.textContent = code;
  document.head.appendChild(el);
  el.remove();
  await loadBundle(); rememberDemo();
  RUN.viewId = null;
  renderActiveSet();
}

/* --------------------------------------------------------- загрузка картинок */
async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  if (!RUN.available) { hint('Для загрузки нужен сервер', true); notify('Загрузка доступна через сервер.', 'fail'); return; }
  const files = [...fileList];
  if (files.length > 4) { hint('Не больше четырёх файлов', true); notify('Выберите от одного до четырёх файлов.', 'fail'); return; }
  const fd = new FormData();
  files.forEach((f, i) => fd.append(`file${i}`, f, f.name));
  hint('загрузка…'); $('#pick-files').disabled = true;
  try {
    const js = await fetchJSON('api/sets', { method: 'POST', body: fd });
    RUN.currentSet = js.id;
    await refreshBucket();
    const loaded = await loadSetIntoPhysics(js.id);
    if (!loaded) return;
    hint(`комплект ${js.id} готов · ${js.info.sources.length} файл(ов) · жмите «прогнать сеть»`);
    notify('Картинка загружена. Нажмите «прогнать сеть», чтобы рассчитать её дифракцию.', 'ok');
    setTab('test');
  } catch (e) { hint(String(e.message || e), true); notify(String(e.message || e), 'fail'); }
  finally { $('#pick-files').disabled = false; $('#file-input').value = ''; }
}

async function ensureCurrentSet() {
  if (!RUN.currentSet && RUN.available) {
    if (!RUN.sets.length) await refreshBucket();
    RUN.currentSet = (RUN.sets.find(s => s.id === 'default') || {}).id || null;
  }
}

async function selectSet(id) {
  RUN.currentSet = id || 'default';
  renderSetSelect();
  const ok = await loadSetIntoPhysics(RUN.currentSet);
  if (ok && S.tab === 'test') renderTestTab();
}

function applySetView(view) {
  if (!DEMO) rememberDemo();
  const buf = {};
  // Распаковка до замены D: при ошибке текущий экран остаётся целым.
  return Promise.all(Object.entries(view.b64).map(async ([key, value]) => {
    buf[key] = await b64ToBufGz(value);
  })).then(() => {
    if (view.id !== RUN.currentSet) return;
    const u8 = k => new Uint8Array(buf[k] || new ArrayBuffer(0));
    const f32 = k => new Float32Array(buf[k] || new ArrayBuffer(0));
    const stack = u8('zstack_nn');
    const px = N * N;
    for (let k = 1; k < Math.floor(stack.length / px); k++)
      for (let i = 0; i < px; i++) stack[k * px + i] = (stack[k * px + i] + stack[(k - 1) * px + i]) & 255;
    D.targets = u8('targets');
    D.phase = { nn: f32('phase_nn'), gs: new Float32Array(0) };
    D.doe = { nn: u8('doe_nn'), gs: new Uint8Array(0) };
    D.recon = { nn: f32('recon_nn'), gs: new Float32Array(0) };
    D.ssimMap = { nn: f32('ssim_map_nn'), gs: new Float32Array(0) };
    D.zstack = { nn: stack, gs: new Uint8Array(0) };
    D.zprobe = { nn: f32('zprobe_nn'), gs: new Float32Array(0) };
    D.bitdepth = { nn: u8('bitdepth_nn'), gs: new Uint8Array(0) };
    D.zgrid = f32('zgrid');
    D.sourceSteps = u8('source_steps');
    D.physSteps = new Uint8Array(0);
    D.manifest = { ...DEMO.manifest, metrics: view.metrics,
      file_metrics: { nn: view.file_metrics.nn }, phys_steps: [], phys_total: view.metrics.steps || 0,
      source_steps: { names: view.source_names || [] }, variants: ['nn'] };
    D.trainSteps = DEMO.trainSteps; D.trainLog = DEMO.trainLog;
    D.trainMasks = DEMO.trainMasks; D.trainRecon = DEMO.trainRecon;
    RUN.viewId = view.id; RUN.viewSignature = view.signature;
    S.variant = 'nn'; S.z = 0; S.playing = false;
    $('#prop-play').textContent = '▶';
    renderActiveSet();
  });
}

function renderActiveSet() {
  CACHE.frames = {}; CACHE.small = {};
  $$('#variant-switch button').forEach(b => {
    b.classList.toggle('active', b.dataset.variant === S.variant);
    b.disabled = b.dataset.variant === 'gs' && !!RUN.viewId;
    b.title = b.disabled ? 'GS рассчитан только для сдаваемого набора' : '';
  });
  renderHeader(); renderSamples(); renderDoe(); renderPropStatic(); setZ(S.z);
  renderPlanes(); renderBitbar(); renderProfiles(); drawSsim(); renderMetrics();
  renderTrain(); renderIterGrid(); renderCurves(); renderSetSelect(); renderTestTab(); drawFlow();
  const names = (D.manifest.source_steps || {}).names || [];
  $('#flow-cap').textContent = names.join(' → ');
  updateJobButtons();
}

async function loadSetIntoPhysics(setId, force = false) {
  const sid = setId || 'default';
  const seq = ++RUN.loadSeq;
  if (sid === 'default' && DEMO && !force && !RUN.defaultDirty) {
    Object.assign(D, DEMO); RUN.viewId = null; S.variant = 'nn';
    renderActiveSet(); return true;
  }
  if (!RUN.available) { notify('Выбор комплектов доступен через сервер.', 'fail'); return false; }
  notify(`Загружаю комплект ${sid}…`, 'busy');
  try {
    const view = await fetchJSON(`api/set-view?set=${encodeURIComponent(sid)}`, { cache: 'no-store' });
    if (seq !== RUN.loadSeq || sid !== RUN.currentSet) return false;
    await applySetView(view);
    if (seq !== RUN.loadSeq || sid !== RUN.currentSet) return false;
    RUN.act = null; RUN.actSet = null;
    if (S.tab === 'model') ensureAct();
    notify(view.note, 'ok'); hint(view.note);
    return true;
  } catch (e) {
    if (seq === RUN.loadSeq) { hint(String(e.message || e), true); notify(String(e.message || e), 'fail'); }
    return false;
  }
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
  sel.innerHTML = sets.length ? sets.map(s =>
    `<option value="${escapeHtml(s.id)}">${escapeHtml(setLabel(s.id))}</option>`).join('') :
    '<option value="default">сдаваемый набор</option>';
  sel.value = RUN.currentSet || 'default';
  sel.onchange = () => selectSet(sel.value);
  $('#set-delete').disabled = !RUN.available || !RUN.currentSet || RUN.currentSet === 'default' || !!RUN.state.running;
  $('#set-delete').onclick = async () => {
    const sid = RUN.currentSet;
    if (!sid || sid === 'default' || !confirm(`Удалить комплект ${sid}?`)) return;
    try {
      await fetchJSON(`api/sets/delete?id=${encodeURIComponent(sid)}`, { method: 'POST' });
      RUN.currentSet = 'default';
      await refreshBucket(); await loadSetIntoPhysics('default');
      hint('комплект удалён');
    } catch (e) { notify(String(e.message || e), 'fail'); }
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
    renderDoe(); renderPlanes(); renderMetrics(); renderProfiles(); drawSsim(); renderTestTab();
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
  $('#pick-demo').onclick = () => selectSet('default');
  $('#flow-play').onclick = playFlow;
  $('#test-run').onclick = runSelectedSet;
  $('#physics-run').onclick = runSelectedSet;
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
    if (!RUN.available) { flashStatus('нужен сервер', 'fail'); return; }
    flashStatus('останавливаю…', 'busy');
    try {
      const r = await fetch('api/stop', { method: 'POST' });
      const js = await r.json();
      flashStatus(js.stopped ? 'остановлено' : 'нечего останавливать', js.stopped ? 'ok' : 'fail');
      await refreshRunState();
    } catch (e) { flashStatus('нет связи', 'fail'); notify('Нет связи с сервером.', 'fail'); }
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
  const W = cv.width = Math.max(300, cv.clientWidth) * dpr();
  const H = cv.height = 220 * dpr();
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const g = RUN.graph;
  if (!g || !g.layers || !g.layers.length) {
    clearCanvas(cv, RUN.graphError || 'Гистограммы появятся после чтения весов');
    $('#hist-probe').textContent = RUN.graphError || 'Нет данных о весах';
    return;
  }
  const index = clamp(S.histLayer || 0, 0, g.layers.length - 1);
  const l = g.layers[index], hist = l.hist || [];
  const left = 16 * dpr(), right = W - 16 * dpr(), top = 30 * dpr(), bottom = H - 30 * dpr();
  const bw = (right - left) / Math.max(1, hist.length);
  ctx.strokeStyle = '#2a3d59'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(left, bottom); ctx.lineTo(right, bottom); ctx.stroke();
  hist.forEach((v, i) => {
    const h = Math.max(1, v) * (bottom - top);
    ctx.fillStyle = '#7aa2ff';
    ctx.fillRect(left + i * bw + 1, bottom - h, Math.max(1, bw - 2), h);
  });
  ctx.fillStyle = '#dce9fa'; ctx.font = `${11 * dpr()}px ui-monospace, monospace`;
  ctx.fillText(l.name.replace(/^net\./, ''), left, 17 * dpr());
  ctx.fillStyle = '#8ea1c2'; ctx.font = `${10 * dpr()}px ui-monospace, monospace`;
  ctx.fillText(fmt(l.min, 4), left, H - 9 * dpr());
  ctx.textAlign = 'right'; ctx.fillText(fmt(l.max, 4), right, H - 9 * dpr()); ctx.textAlign = 'left';
  $('#hist-probe').innerHTML = `${l.shape.join('×')} · ${l.numel.toLocaleString('ru-RU')} значений · mean <b>${fmt(l.mean, 5)}</b> · std <b>${fmt(l.std, 5)}</b> · ненулевых ${fmt(l.nonzero * 100, 1)}%`;
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
  rememberDemo();
  await probeRunner();
  await ensureCurrentSet();
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
  updateJobButtons();
  if (RUN.state.running) startLogPoll(RUN.state.job);
  requestAnimationFrame(tick);
}

window.HDOE = { S, D, RUN, setTab, renderMetrics, renderCurves, uploadFiles, refreshBucket,
  selectSet, loadSetIntoPhysics, runSelectedSet, ensureGraph, ensureAct, drawHistMap };
document.addEventListener('DOMContentLoaded', main);
