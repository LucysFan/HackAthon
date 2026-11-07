/* Проверка пульта запуска в браузерном окружении (jsdom) против живого
   сервера scripts/serve_dashboard.py: кнопка, панель, лог, статусы.

   Запуск (сначала поднять пульт):
     python scripts/serve_dashboard.py --no-browser &
     node tools/smoke_runner_ui.js [http://127.0.0.1:8765/]
*/
const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const ROOT = path.resolve(__dirname, '..');
const BASE = (process.argv[2] || 'http://127.0.0.1:8765/').replace(/\/?$/, '/');
const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push('jsdomError: ' + e.message));
vc.on('error', (...a) => errors.push('console.error: ' + a.join(' ')));

/** Заглушка 2D-контекста: рисуем «в никуда», но методы не падают. */
function makeCtx() {
  const t = {
    canvas: null, calls: [],
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => {}, drawImage: () => {}, measureText: () => ({ width: 10 }),
    createLinearGradient: () => ({ addColorStop() {} }),
    createRadialGradient: () => ({ addColorStop() {} }),
    createPattern: () => ({}), getLineDash: () => [], isPointInPath: () => false,
  };
  return new Proxy(t, {
    get(o, k) {
      if (k in o) return o[k];
      if (typeof k === 'symbol') return undefined;
      return () => {};
    },
    set(o, k, v) { o[k] = v; return true; },
    has: () => true,
  });
}

const html = fs.readFileSync(path.join(ROOT, 'dashboard', 'index.html'), 'utf8');
const dom = new JSDOM(html.replace(/<script src="[^"]+"><\/script>/g, ''), {
  runScripts: 'outside-only', pretendToBeVisual: true, url: BASE, virtualConsole: vc,
});
const { window } = dom;
window.HTMLCanvasElement.prototype.getContext = function () {
  if (!this.__ctx) this.__ctx = makeCtx();
  return this.__ctx;
};
window.requestAnimationFrame = cb => setTimeout(() => cb(Date.now()), 4);
window.fetch = (u, o) => fetch(new URL(u, BASE), o);
window.DecompressionStream = undefined;   // проверяем pure-JS gunzip-фолбэк

const t0 = Date.now();
const run = f => window.eval(fs.readFileSync(path.join(ROOT, 'dashboard', f), 'utf8'));
run('data_bundle.js');
run('app.js');

const wait = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const $ = s => window.document.querySelector(s);
  const checks = [];
  const ok = (n, c, e) => checks.push({ n, pass: !!c, e });

  // в jsdom распаковка бандла медленнее, чем в браузере — ждём появления пульта
  for (let i = 0; i < 300; i++) {
    if (!$('#run-btn').classList.contains('hidden')) break;
    await wait(100);
  }

  ok('кнопка запуска видна', !$('#run-btn').classList.contains('hidden'));
  const stageCount = $('#rp-stages').querySelectorAll('.rp-stage').length;
  ok('панель заполнена этапами', stageCount >= 10, stageCount + ' шт');
  ok('сценарии есть', $('#rp-pipelines').querySelectorAll('button').length >= 3);
  const marks = [...$('#rp-stages').querySelectorAll('small')].length;
  ok('посчитанные файлы подписаны', marks >= 4, marks + ' меток');

  $('#run-btn').click();
  await wait(300);
  ok('панель открывается по кнопке', !$('#runpanel').classList.contains('hidden'));

  const probe = $('.rp-stage[data-stage="probe"]');
  ok('служебный этап в списке', !!probe);
  if (probe) {
    probe.click();
    await wait(5000);
    const logTxt = $('#rp-log').textContent;
    ok('лог стримится в панель', /python 3\.11/.test(logTxt) && /torch 2\.15/.test(logTxt),
       logTxt.slice(0, 90).replace(/\n/g, ' '));
    ok('статус обновился', /готово|выполняется/.test($('#rp-status').textContent),
       $('#rp-status').textContent);
  }
  ok('нет ошибок JS', errors.length === 0, errors.slice(0, 2).join(' | '));

  const failed = checks.filter(c => !c.pass);
  for (const c of checks) console.log(`${c.pass ? 'OK  ' : 'FAIL'} ${c.n}${c.e ? ' — ' + c.e : ''}`);
  console.log(`\n${checks.length - failed.length}/${checks.length} проверок пульта (UI) за ${Date.now() - t0} мс`);
  process.exit(failed.length ? 1 : 0);
})();
