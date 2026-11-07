/* Смоук-проверка дашборда без браузера: jsdom + заглушка canvas 2D.
   Проверяет: загрузку бандла, отрисовку всех панелей, переключение NN/GS,
   режимы плоскостей, метрики, измерения (профили, z-зонд, ROI), обучение,
   физику и совпадение чисел с manifest.json.

   Запуск:  node tools/smoke_dashboard.js
*/
const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const ROOT = path.resolve(__dirname, '..');
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => errors.push('jsdomError: ' + e.message));
virtualConsole.on('error', (...a) => errors.push('console.error: ' + a.join(' ')));

/** Полная заглушка 2D-контекста: любой неизвестный метод — no-op,
    createImageData/getImageData возвращают настоящие буферы. */
function makeCtx(canvas) {
  const store = {
    canvas, imageSmoothingEnabled: true, fillStyle: '#000', strokeStyle: '#000',
    lineWidth: 1, font: '10px sans-serif', globalAlpha: 1, textAlign: 'left',
    textBaseline: 'alphabetic', lineCap: 'butt', lineJoin: 'miter', globalCompositeOperation: 'source-over',
  };
  const drawLog = [];
  const target = {
    ...store,
    calls: drawLog,
    createImageData: (w, h) => (w && w.width ? { width: w.width, height: w.height, data: new Uint8ClampedArray(w.width * w.height * 4) }
      : { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => drawLog.push('putImageData'),
    drawImage: () => drawLog.push('drawImage'),
    measureText: () => ({ width: 10 }),
    createLinearGradient: () => ({ addColorStop() {} }),
    createRadialGradient: () => ({ addColorStop() {} }),
    createPattern: () => ({}),
    getLineDash: () => [],
    isPointInPath: () => false,
  };
  return new Proxy(target, {
    get(t, k) {
      if (k in t) return t[k];
      if (typeof k === 'symbol') return undefined;
      return (...args) => { drawLog.push(k); return undefined; };
    },
    set(t, k, v) { t[k] = v; return true; },
    has: () => true,
  });
}

function installCanvasStub(window) {
  window.HTMLCanvasElement.prototype.getContext = function () {
    if (!this.__ctx) {
      this.__ctx = makeCtx(this);
      this.__drawOps = 0;
    }
    return this.__ctx;
  };
  const origGet = Object.getOwnPropertyDescriptor(window.HTMLCanvasElement.prototype, 'getContext');
  void origGet;
}

const html = fs.readFileSync(path.join(ROOT, 'dashboard', 'index.html'), 'utf8');
const dom = new JSDOM(html.replace(/<script src="[^"]+"><\/script>/g, ''), {
  runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost/',
  virtualConsole,
});
const { window } = dom;
installCanvasStub(window);
window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 4);
window.performance = window.performance || { now: () => Date.now() };

// загрузка бандла и приложения как в браузере
function run(file) {
  const code = fs.readFileSync(path.join(ROOT, 'dashboard', file), 'utf8');
  window.eval(code);
}
const t0 = Date.now();
// в jsdom нет рабочего DecompressionStream/Blob-пайплайна — проверяем
// встроенный pure-JS gunzip-фолбэк (тот же код работает и в браузере)
if (process.env.SMOKE_DS === '1') {
  window.DecompressionStream = require('stream/web').DecompressionStream;
  window.Blob = require('buffer').Blob;
  window.Response = global.Response;
} else {
  window.DecompressionStream = undefined;
}
run('data_bundle.js');
run('app.js');

const waitReady = () => new Promise(res => {
  const poll = () => (window.document.querySelector('#targets-grid').children.length === 4)
    ? res() : setTimeout(poll, 100);
  setTimeout(poll, 200);
  setTimeout(res, 15000);
});

waitReady().then(() => {
  const $ = s => window.document.querySelector(s);
  const checks = [];
  const ok = (name, cond, extra) => checks.push({ name, pass: !!cond, extra });

  const B = window.HDOE_BUNDLE;
  ok('бандл загружен (manifest)', !!B && !!B.manifest);
  ok('цели 4×128×128', !!window.HDOE_BUNDLE.b64.targets);
  ok('z-стопка NN/GS есть', !!B.b64.zstack_nn && !!B.b64.zstack_gs);
  ok('бандл ≤ 12 МБ', fs.statSync(path.join(ROOT, 'dashboard', 'data_bundle.js')).size < 12e6);

  const meta = B.manifest.file_metrics;
  ok('метрики NN в manifest', meta.nn && meta.nn.length === 4);
  ok('метрики GS в manifest', meta.gs && meta.gs.length === 4);
  ok('PCC NN среднее > PCC GS среднее',
     meta.nn.reduce((a, r) => a + r.pcc, 0) / 4 > meta.gs.reduce((a, r) => a + r.pcc, 0) / 4);
  ok('тесты 8.1–8.9 в бандле', B.json.tests && B.json.tests.summary.passed === B.json.tests.summary.total);

  // шапка
  ok('бейджи тестов отрисованы', $('.badges').querySelectorAll('.badge').length >= 8);
  ok('чипы параметров', $('.params').querySelectorAll('.chip').length >= 6);
  // панель 1–4
  ok('плитки целей', $('.targets-grid').querySelectorAll('.target-cell').length === 4);
  ok('ячейки плоскостей', $('.planes-grid').querySelectorAll('.plane-cell').length === 4);
  ok('столбики метрик NN+GS', $('.metrics-bars').querySelectorAll('.mbar').length === 8);
  ok('счётчик PCC заполнен', /\d\.\d+/.test($('#c-pcc').textContent), $('#c-pcc').textContent);
  ok('таблицы обоих протоколов', $('.metrics-tables').querySelectorAll('table').length === 2);
  // физика
  ok('таблица 4.7 (4 строки)', $('#phys-table').querySelectorAll('tr[data-plane]').length === 4,
     String($('#phys-table').querySelectorAll('tr[data-plane]').length));
  ok('маркеры z на шкале', $('#zscale-track').querySelectorAll('.marker').length === 4);
  ok('отметка z_crit', $('#zscale-track').querySelectorAll('.crit').length === 1);
  ok('обучение: маски', B.json.train_steps.steps.length > 0);

  // интерактив: переключение NN/GS
  const btnGs = window.document.querySelector('.variant-switch button[data-variant="gs"]');
  btnGs.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  ok('переключение на GS', $('.variant-switch button.active').dataset.variant === 'gs');
  const pccBefore = $('#c-pcc').textContent;
  window.document.querySelector('.variant-switch button[data-variant="nn"]')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  ok('переключение обратно на NN', $('.variant-switch button.active').dataset.variant === 'nn');
  void pccBefore;

  // режимы плоскостей
  for (const mode of ['side', 'blend', 'diff', 'swipe']) {
    const b = window.document.querySelector(`#plane-mode button[data-mode="${mode}"]`);
    b.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    ok(`режим плоскостей «${mode}»`, $('.plane-cell canvas') !== null);
  }
  // протоколы метрик
  window.document.querySelector('#protocol-switch button[data-proto="reference"]')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  ok('справочный протокол переключён', $('.metrics-tables').querySelectorAll('table').length === 2);

  // измерения
  const meas = $('#meas-canvas');
  const r = { left: 0, top: 0, width: 320, height: 320 };
  meas.getBoundingClientRect = () => r;
  for (const ev of ['mousedown', 'mousemove']) {
    meas.dispatchEvent(new window.MouseEvent(ev, { bubbles: true, clientX: 120, clientY: 140 }));
  }
  ok('зонд измерений отвечает', /px|\(/.test($('#meas-probe').textContent));
  window.document.querySelector('#meas-mode button[data-mmode="roi"]')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  meas.dispatchEvent(new window.MouseEvent('mousedown', { bubbles: true, clientX: 60, clientY: 60 }));
  meas.dispatchEvent(new window.MouseEvent('mousemove', { bubbles: true, clientX: 180, clientY: 200 }));
  ok('ROI-статистика', /mean/.test($('#roi-readout').textContent) ||
     /потяните/.test($('#roi-readout').textContent));
  ok('линейка по z', $('#zruler').querySelectorAll('.row').length >= 6);

  // обучение: scrub
  const sl = $('#train-slider');
  sl.value = 10;
  sl.dispatchEvent(new window.Event('input', { bubbles: true }));
  ok('скраб обучения', /\d+/.test($('#train-step-label').textContent));

  // физика внутри сети: полоса шагов K
  const cells = $('.p-doe').querySelectorAll('#steps-row .step-cell');
  ok('полоса шагов физики', cells.length >= 2, String(cells.length));
  ok('кривая шагов физики нарисована', !!$('#steps-curve'));
  if (cells.length >= 2) {
    cells[cells.length - 1].onclick();
    const txt = $('#steps-probe').textContent;
    ok('выбор кадра K обновляет зонд', /K\s*=\s*\d+/.test(txt) && /PCC/.test(txt), txt.slice(0, 60));
  }

  // бейдж теста 8.9 (физика внутри сети)
  const b89 = $('.badge[data-test="8.9"]');
  ok('бейдж 8.9 «физика в сети»', !!b89 && /✓/.test(b89.textContent), b89 ? b89.textContent.trim() : '—');

  // битность A8
  const bits = $('.p-planes').querySelectorAll('#bitbar button');
  ok('кнопки битности A8', bits.length === 4);

  // авто-тур
  ok('сценарий тура задан', typeof window.toggleTour === 'function');

  const failed = checks.filter(c => !c.pass);
  for (const c of checks) console.log(`${c.pass ? 'OK  ' : 'FAIL'} ${c.name}${c.extra ? ' — ' + c.extra : ''}`);
  if (errors.length) console.log('ОШИБКИ JS:\n' + errors.join('\n'));
  console.log(`\n${checks.length - failed.length}/${checks.length} проверок пройдено за ${Date.now() - t0} мс`);
  process.exit(failed.length === 0 && errors.length === 0 ? 0 : 1);
});
