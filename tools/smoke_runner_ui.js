/* Проверка кнопок дашборда в браузерном окружении (jsdom) против ЖИВОГО пульта.

   Запуск (сначала поднять сервер):
     python scripts/serve_dashboard.py --port 8773 --no-browser &
     node tools/smoke_runner_ui.js http://127.0.0.1:8773/

   Что проверяется — то, на что жалуются пользователи: кнопки должны
   срабатывать, «стоп» обязан останавливать длинную задачу, загрузка файла
   должна доходить до сервера и возвращаться комплектом, вкладки — не падать.
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
window.HTMLAnchorElement.prototype.click = function () {};
window.requestAnimationFrame = cb => setTimeout(() => cb(Date.now()), 8);
/* jsdom-овский FormData несовместим с undici: загрузку файлов пересобираем
   вручную (multipart), всё остальное проксируем как обычный fetch. */
async function nodeFetch(u, o = {}) {
  const url = new URL(u, BASE);
  if (o.body instanceof window.FormData) {
    const fd = new FormData();
    for (const [k, v] of o.body.entries()) {
      const buf = Buffer.from(await v.arrayBuffer());
      fd.append(k, new Blob([buf], { type: v.type }), v.name);
    }
    return fetch(url, { method: o.method || 'POST', body: fd });
  }
  const opts = { ...o };
  delete opts.body;
  if (o.body) opts.body = o.body;
  return fetch(url, opts);
}
window.fetch = nodeFetch;
window.DecompressionStream = require('stream/web').DecompressionStream;
window.Blob = require('buffer').Blob;
window.Response = globalThis.Response;

const t0 = Date.now();
const run = f => window.eval(fs.readFileSync(path.join(ROOT, 'dashboard', f), 'utf8'));
run('data_bundle.js');
run('app.js');

const wait = ms => new Promise(r => setTimeout(r, ms));
const $ = s => window.document.querySelector(s);

const step = m => process.stdout.write(`· ${m}\n`);
setTimeout(() => { console.log('\nТАЙМАУТ 300 с'); process.exit(2); }, 300000).unref?.();

(async () => {
  step('старт');
  const checks = [];
  const ok = (n, c, e) => checks.push({ n, pass: !!c, e });

  // ждём, пока распакуется бандл и пульт отзовётся
  for (let i = 0; i < 300; i++) {
    if (window.HDOE && window.HDOE.RUN && window.HDOE.RUN.available) break;
    await wait(100);
  }
  step('пульт');
  ok('пульт определён (офлайн-проверка не мешает)', window.HDOE.RUN.available === true);
  ok('пять вкладок в шапке', window.document.querySelectorAll('#tabs button').length === 5,
    window.document.querySelectorAll('#tabs button').length);
  for (let i = 0; i < 100 && $('#runner-badge').classList.contains('hidden'); i++) await wait(100);
  ok('бейдж пульта виден', !$('#runner-badge').classList.contains('hidden'));

  step('вкладка запуск');
  // ---------------------------------------------------------- вкладка «Запуск»
  window.document.querySelector('#tabs button[data-tab="run"]').click();
  await wait(200);
  const stageCount = $('#rp-stages').querySelectorAll('.rp-stage').length;
  ok('этапы в панели запуска', stageCount >= 10, stageCount + ' шт');
  ok('сценарии есть', $('#rp-pipelines').querySelectorAll('button').length >= 3);
  const marks = $('#rp-stages').querySelectorAll('small').length;
  ok('посчитанные файлы подписаны', marks >= 3, marks + ' меток');

  step('этап probe');
  // -------------------------------------------------- дешёвый этап: probe
  $('.rp-stage[data-stage="probe"]').click();
  await wait(5000);
  const logTxt = $('#rp-log').textContent;
  ok('лог стримится в панель', /torch\s+\d/.test(logTxt), logTxt.slice(0, 70).replace(/\n/g, ' '));
  ok('статус «готово»', /готово/.test($('#rp-status').textContent), $('#rp-status').textContent);

  step('стоп');
  // --------------------------------------- длинный этап и кнопка «стоп»
  $('#rp-log').textContent = '';
  $('.rp-stage[data-stage="tests"]').click();
  await wait(2500);
  const stopBtn = $('#rp-stop');
  ok('кнопка «стоп» активна во время задачи', stopBtn.disabled === false);
  const busyStatus = $('#rp-status').textContent;
  ok('статус «выполняется» во время задачи', /выполняется/.test(busyStatus), busyStatus);
  const tStop = Date.now();
  stopBtn.click();
  let stopped = false;
  for (let i = 0; i < 40; i++) {
    await wait(150);
    if (/остановлено/.test($('#rp-status').textContent)) { stopped = true; break; }
  }
  const dt = Date.now() - tStop;
  ok('«стоп» нажимается и срабатывает', stopped && dt < 3000, `${dt} мс, статус «${$('#rp-status').textContent}»`);
  await wait(600);
  ok('после «стоп» кнопка снова неактивна', $('#rp-stop').disabled === true);
  ok('в логе отметка об остановке', /остановлено/.test($('#rp-log').textContent));

  // повторный запуск после остановки — пульт должен принимать задачи
  $('.rp-stage[data-stage="probe"]').click();
  await wait(4000);
  ok('после «стоп» задачи снова запускаются', /готово/.test($('#rp-status').textContent),
    $('#rp-status').textContent);

  step('загрузка');
  // ------------------------------------------------------ загрузка комплекта
  const png = fs.readFileSync(path.join(ROOT, 'data', 'source', 'object.png'));
  const file = new window.File([png], 'photo.png', { type: 'image/png' });
  await window.HDOE.uploadFiles([file]);
  await wait(1500);
  const hintTxt = $('#load-hint').textContent;
  ok('загрузка принята сервером', /комплект .* готов/.test(hintTxt), hintTxt);
  ok('комплект выбран', !!window.HDOE.RUN.currentSet, window.HDOE.RUN.currentSet);
  ok('комплект появился в списке', $('#set-select').options.length >= 2);
  ok('вкладка переключилась на «Тест»', window.HDOE.S.tab === 'test', window.HDOE.S.tab);

  step('вкладка тест');
  // ---------------------------------------------------------- вкладка «Тест»
  const thumbs = $('#test-thumbs').querySelectorAll('.thumb').length;
  ok('плитки целей комплекта', thumbs === 4, thumbs);
  ok('кнопка «прогнать сеть» активна', !!$('#test-run'));
  $('#test-run').click();
  let inferDone = false;
  for (let i = 0; i < 400; i++) {                     // до 4 минут: K = 2048 шагов физики
    await wait(200);
    const info = await (await window.fetch(`api/sets/${window.HDOE.RUN.currentSet}`)).json();
    // DOE.png пишется раньше metrics.json — ждём именно метрики
    if (info.has_doe && info.has_metrics) { inferDone = true; break; }
  }
  ok('инференс комплекта дошёл до DOE.png', inferDone);
  const info2 = await (await window.fetch(`api/sets/${window.HDOE.RUN.currentSet}`)).json();
  ok('метрики комплекта посчитаны', !!(info2.mean && info2.mean.ssim > 0.2),
    info2.mean ? `SSIM ${info2.mean.ssim.toFixed(3)}` : '—');
  await wait(800);
  ok('DOE показан на вкладке «Тест»', !!$('#test-doe').querySelector('img,canvas'));
  ok('счётчик SSIM заполнен', /0\.\d+/.test($('#c-ssim').textContent), $('#c-ssim').textContent);
  window.HDOE.refreshBucket();
  await wait(1500);

  step('остальные вкладки');
  // -------------------------------------------------- остальные вкладки
  for (const t of ['physics', 'algo', 'model']) {
    const before = errors.length;
    window.document.querySelector(`#tabs button[data-tab="${t}"]`).click();
    await wait(400);
    ok(`вкладка ${t} без ошибок`, errors.length === before, errors.slice(before).join('; '));
  }
  $('#act-play').click();
  await wait(800);
  ok('анимация активаций идёт', window.HDOE.S.actPlaying === true);
  // первый расчёт активаций на CPU идёт несколько секунд — ждём, а не гадаем
  for (let i = 0; i < 200; i++) {
    if (window.HDOE.RUN.act && window.HDOE.RUN.act.stages.length) break;
    await wait(200);
  }
  ok('активации с сервера пришли', !!(window.HDOE.RUN.act && window.HDOE.RUN.act.stages.length),
    window.HDOE.RUN.act && window.HDOE.RUN.act.stages.length);
  for (let i = 0; i < 100; i++) {
    if (window.HDOE.RUN.graph && window.HDOE.RUN.graph.layers.length) break;
    await wait(200);
  }
  ok('граф весов с сервера пришёл', !!(window.HDOE.RUN.graph && window.HDOE.RUN.graph.layers.length),
    window.HDOE.RUN.graph && window.HDOE.RUN.graph.layers.length);

  // уборка: удаляем созданный комплект
  if (window.HDOE.RUN.currentSet) {
    await window.fetch(`api/sets/delete?id=${window.HDOE.RUN.currentSet}`, { method: 'POST' });
  }
  ok('нет ошибок JS', errors.length === 0, errors.slice(0, 3).join(' | '));

  const failed = checks.filter(c => !c.pass);
  for (const c of checks) console.log(`${c.pass ? 'OK  ' : 'FAIL'} ${c.n}${c.e ? ' — ' + c.e : ''}`);
  console.log(`\n${checks.length - failed.length}/${checks.length} проверок кнопок за ${Date.now() - t0} мс`);
  process.exit(failed.length ? 1 : 0);
})();
