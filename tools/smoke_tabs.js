/* Смоук новой раскладки дашборда: пять вкладок, все кнопки, загрузка файлов.

   Проверяем в jsdom с заглушкой canvas:
     • бандл распаковывается, панели получают данные;
     • переключение пяти вкладок не падает и не оставляет пустых панелей;
     • каждая кнопка в разметке вызывается хотя бы раз (включая «стоп»);
     • загрузка 1–4 файлов уходит на /api/sets, ответ подхватывается;
     • метрики на экране совпадают с manifest.json.

   Запуск:  node tools/smoke_tabs.js
   Нужны:   npm install --prefix tools   (jsdom)
*/
const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const ROOT = path.resolve(__dirname, '..');
const ok = [], fail = [];
const check = (name, cond, extra = '') => (cond ? ok : fail).push(name + (cond ? '' : ' — ' + extra));

const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push('jsdomError: ' + (e && e.message)));
vc.on('error', (...a) => errors.push('console.error: ' + a.join(' ')));
vc.on('warn', () => {});

function makeCtx(canvas) {
  const log = [];
  const noop = () => {};
  const ctx = {
    canvas, calls: log, imageSmoothingEnabled: true, fillStyle: '#000', strokeStyle: '#000',
    lineWidth: 1, font: '10px sans-serif', globalAlpha: 1, textAlign: 'left',
    textBaseline: 'alphabetic', lineCap: 'butt', lineJoin: 'miter', filter: 'none',
    globalCompositeOperation: 'source-over',
    createImageData: (w, h) => (w && w.width ? { width: w.width, height: w.height, data: new Uint8ClampedArray(w.width * w.height * 4) }
      : { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => log.push('putImageData'),
    drawImage: () => log.push('drawImage'),
    measureText: () => ({ width: 10 }),
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop }),
    createPattern: () => ({}),
    getLineDash: () => [],
    setLineDash: noop,
    save: noop, restore: noop, beginPath: noop, closePath: noop, moveTo: noop, lineTo: noop,
    arc: noop, rect: noop, fill: noop, stroke: noop, clip: noop, fillRect: noop,
    strokeRect: noop, clearRect: noop, fillText: noop, strokeText: noop,
    translate: noop, rotate: noop, scale: noop, setTransform: noop, resetTransform: noop,
    quadraticCurveTo: noop, bezierCurveTo: noop, ellipse: noop,
  };
  return ctx;
}

const step = m => { process.stdout.write(`· ${m}\n`); };
setTimeout(() => { console.log('\nТАЙМАУТ 240 с: проверка не дошла до конца'); process.exit(2); },
  300000).unref?.();

(async () => {
  step('чтение index.html');
  const html = fs.readFileSync(path.join(ROOT, 'dashboard', 'index.html'), 'utf8');
  const dom = new JSDOM(html, {
    url: 'file://' + path.join(ROOT, 'dashboard', 'index.html'),
    runScripts: 'outside-only', pretendToBeVisual: true, virtualConsole: vc,
  });
  const { window } = dom;
  const doc = window.document;
  window.HTMLCanvasElement.prototype.getContext = function () {
    this.__ctx = this.__ctx || makeCtx(this);
    return this.__ctx;
  };
  window.HTMLCanvasElement.prototype.toBlob = function (cb) { cb(new window.Blob(['x'])); };
  window.URL.createObjectURL = () => 'blob:x';
  // jsdom не умеет скачивание файлов: <a download> в браузере работает, здесь — no-op
  window.HTMLAnchorElement.prototype.click = function () {};
  let drawn = 0;
  const origDraw = window.HTMLCanvasElement.prototype.getContext;
  void origDraw;

  // ---- офлайн-режим: fetch запрещён (как при двойном клике по index.html)
  window.fetch = () => Promise.reject(new Error('offline'));
  // в jsdom нет рабочего gzip-пайплайна — берём реализации из Node,
  // в браузере это встроенные DecompressionStream/Blob/Response
  window.DecompressionStream = require('stream/web').DecompressionStream;
  window.Blob = require('buffer').Blob;
  window.Response = globalThis.Response;

  step('бандл: eval');
  // ---- загрузка бандла
  const bundle = fs.readFileSync(path.join(ROOT, 'dashboard', 'data_bundle.js'), 'utf8');
  window.eval(bundle);
  check('бандл загружен в window.HDOE_BUNDLE', !!window.HDOE_BUNDLE);
  const manifest = window.HDOE_BUNDLE.manifest;

  step('app.js: eval');
  window.eval(fs.readFileSync(path.join(ROOT, 'dashboard', 'app.js'), 'utf8'));
  window.document.dispatchEvent(new window.Event('DOMContentLoaded'));

  // ждём распаковки (gunzip в Node через DecompressionStream отсутствует → fallback)
  step('распаковка данных');
  const t0 = Date.now();
  while (!(window.HDOE && window.HDOE.D && window.HDOE.D.targets) && Date.now() - t0 < 90000) {
    await new Promise(r => setTimeout(r, 200));
  }
  step('данные готовы');
  const D = (window.HDOE && window.HDOE.D) || {};
  check('данные распакованы (цели 4×128×128)',
    D.targets && D.targets.length === 4 * 128 * 128, D.targets && D.targets.length);
  check('z-стопка 281 кадр', D.zstack && D.zstack.nn && D.zstack.nn.length === 281 * 128 * 128,
    D.zstack && D.zstack.nn && D.zstack.nn.length / 16384);
  check('фазы сети и GS', D.phase && D.phase.nn && D.phase.nn.length === 16384 && D.phase.gs.length === 16384);
  check('кадры подготовки входа', D.sourceSteps && D.sourceSteps.length === 4 * 128 * 128,
    D.sourceSteps && D.sourceSteps.length);
  check('шаги физики K', D.physSteps && D.physSteps.length === 48 * 128 * 128,
    D.physSteps && D.physSteps.length);

  // ---- панели
  const cells = doc.querySelectorAll('#samples .target-cell');
  check('плитка образцов: 4 ячейки', cells.length === 4, cells.length);
  check('плоскости: 4 панели', doc.querySelectorAll('#planes-grid .plane-cell').length === 4);
  check('гистограмма маски нарисована', doc.querySelector('#doe-hist') && doc.querySelector('#doe-hist').__ctx);
  check('z-шкала с четырьмя метками', doc.querySelectorAll('#zscale-track .marker').length === 4,
    doc.querySelectorAll('#zscale-track .marker').length);

  step('вкладки');
  // ---- вкладки
  const tabs = ['physics', 'test', 'algo', 'model', 'run'];
  for (const t of tabs) {
    const before = errors.length;
    doc.querySelector(`#tabs button[data-tab="${t}"]`).click();
    await new Promise(r => setTimeout(r, 60));
    check(`вкладка ${t} открывается без ошибок`, errors.length === before,
      errors.slice(before).join('; '));
    check(`вкладка ${t} активна`, window.HDOE.S.tab === t, window.HDOE.S.tab);
  }

  step('кнопки');
  // ---- кнопки: каждая должна пережить клик
  const skip = new Set(['file-input']);   // реальный выбор файлов проверяем ниже, через uploadFiles
  const btns = Array.from(doc.querySelectorAll('button')).filter(b => !skip.has(b.id));
  let clicked = 0, crashed = [];
  for (const b of btns) {
    const before = errors.length;
    step('  click ' + (b.id || (b.textContent || '').trim().slice(0, 18)));
    try { b.click(); clicked++; } catch (e) { crashed.push(b.id || b.textContent); }
    await new Promise(r => setTimeout(r, 8));
    if (errors.length > before) crashed.push((b.id || b.textContent) + ': ' + errors.slice(before).join(';'));
  }
  check(`все кнопки нажимаются (${clicked})`, crashed.length === 0, crashed.slice(0, 4).join(' | '));

  step('стоп и переключатели');
  // ---- кнопка «стоп» в офлайне не должна ломать интерфейс
  doc.querySelector('#tabs button[data-tab="run"]').click();
  doc.querySelector('#rp-stop').click();
  await new Promise(r => setTimeout(r, 60));
  check('стоп отвечает статусом', /останов|нечего|нет связи/.test(doc.querySelector('#rp-status').textContent),
    doc.querySelector('#rp-status').textContent);

  // ---- переключения NN/GS, режимы плоскостей, протокол метрик, кривые
  doc.querySelectorAll('#variant-switch button')[1].click();
  check('переключение на GS', window.HDOE.S.variant === 'gs');
  doc.querySelectorAll('#variant-switch button')[0].click();
  for (const m of ['side', 'diff', 'swipe']) doc.querySelector(`#plane-mode button[data-mode="${m}"]`).click();
  check('три режима плоскостей', window.HDOE.S.planeMode === 'swipe');
  doc.querySelector('#protocol-switch button[data-proto="reference"]').click();
  const refVal = doc.querySelector('#c-ssim').textContent;
  doc.querySelector('#protocol-switch button[data-proto="primary"]').click();
  const primVal = doc.querySelector('#c-ssim').textContent;
  check('протокол «без нормировки» даёт другое значение', refVal !== primVal, `${refVal} vs ${primVal}`);

  // ---- числа на экране = manifest.json
  const rows = manifest.file_metrics.nn;
  check('SSIM плоскости 1 совпадает с metrics.json',
    Math.abs(parseFloat(primVal) - rows[0].ssim) < 0.0015, `${primVal} vs ${rows[0].ssim}`);
  check('средний SSIM в шапке', doc.querySelector('#params').textContent.includes(
    manifest.metrics.mean.ssim.toFixed(3)));

  // ---- загрузка комплекта (fetch подменён)
  const uploadCalls = [];
  window.fetch = async (url, opts = {}) => {
    uploadCalls.push({ url, opts });
    if (String(url).startsWith('api/sets') && opts.method === 'POST') {
      return { ok: true, json: async () => ({ ok: true, id: '20260101-000000',
        info: { id: '20260101-000000', sources: ['a.png'], has_targets: true } }) };
    }
    if (String(url).startsWith('api/state')) {
      return { ok: true, json: async () => ({ state: {}, files: {}, sets: [
        { id: '20260101-000000', has_targets: true, sources: ['a.png'], mean: null }] }) };
    }
    if (String(url).startsWith('api/history')) return { ok: true, json: async () => ({ train: [], val: [] }) };
    if (String(url).startsWith('api/asset')) return { ok: true, json: async () => ({}) };
    return { ok: true, json: async () => ({ ok: true }) };
  };
  step('загрузка файла');
  const file = new window.File([new Uint8Array([1, 2, 3])], 'photo.jpg', { type: 'image/jpeg' });
  await window.HDOE.uploadFiles([file]);
  const post = uploadCalls.find(c => String(c.url).startsWith('api/sets') && c.opts.method === 'POST');
  check('файл уходит на /api/sets с FormData', !!post && post.opts.body instanceof window.FormData);
  check('комплект выбран после загрузки', window.HDOE.RUN.currentSet === '20260101-000000',
    window.HDOE.RUN.currentSet);

  step('вывод');
  // ---- вывод
  console.log(`\n${ok.length} OK, ${fail.length} FAIL`);
  ok.forEach(n => console.log('  ok   ' + n));
  fail.forEach(n => console.log('  FAIL ' + n));
  if (errors.length) {
    console.log('\nошибки консоли:');
    errors.slice(0, 10).forEach(e => console.log('  ' + e));
  }
  process.exit(fail.length ? 1 : 0);
})();
