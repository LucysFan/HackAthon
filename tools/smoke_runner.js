/* Проверка пульта против живого сервера scripts/serve_dashboard.py.

   Запуск:
     python scripts/serve_dashboard.py --port 8765 --no-browser &
     node tools/smoke_runner.js http://127.0.0.1:8765

   Что проверяется:
     • /api/state: этапы, сценарии, комплекты, интерпретатор, метрики;
     • отдача дашборда и бандла;
     • отказ на неизвестный этап (никаких команд из браузера не принимается);
     • запуск самого дешёвого этапа, живой лог, остановка «стоп» в пределах
       нескольких секунд (в том числе с дочерним процессом);
     • загрузка комплекта из 1 и из 4 файлов, приведение к 128×128/8 бит/L;
     • инференс комплекта прямым проходом и появление DOE.png + metrics.json;
     • /api/model/graph и /api/model/act отдают данные для вкладок.
*/
const http = require('http');
const fs = require('fs');
const path = require('path');

const BASE = (process.argv[2] || 'http://127.0.0.1:8765').replace(/\/$/, '');
const ROOT = path.resolve(__dirname, '..');
const checks = [];
const ok = (name, cond, extra) => checks.push({ name, pass: !!cond, extra });
const t0 = Date.now();
const sleep = ms => new Promise(r => setTimeout(r, ms));

function req(method, p, body, ctype) {
  return new Promise((resolve, reject) => {
    const u = new URL(BASE + p);
    const headers = {};
    let payload = null;
    if (body instanceof Buffer) {
      payload = body;
      headers['Content-Type'] = ctype || 'application/octet-stream';
      headers['Content-Length'] = body.length;
    } else if (body) {
      payload = Buffer.from(body);
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = payload.length;
    }
    const r = http.request({ hostname: u.hostname, port: u.port, path: u.pathname + u.search,
      method, headers }, res => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks),
        text: Buffer.concat(chunks).toString('latin1') }));
    });
    r.on('error', reject);
    if (payload) r.write(payload);
    r.end();
  });
}

const json = async (m, p, b, ctype) => {
  const r = await req(m, p, b, ctype);
  try { return JSON.parse(r.body.toString('utf8')); } catch (e) { return {}; }
};

/** Собрать multipart/form-data из файлов (как браузер). */
function multipart(files) {
  const boundary = '----hdoe' + Date.now();
  const parts = [];
  files.forEach((f, i) => {
    parts.push(Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="file${i}"; filename="${f.name}"\r\n` +
      `Content-Type: ${f.type || 'application/octet-stream'}\r\n\r\n`));
    parts.push(f.data);
    parts.push(Buffer.from('\r\n'));
  });
  parts.push(Buffer.from(`--${boundary}--\r\n`));
  return { body: Buffer.concat(parts), ctype: `multipart/form-data; boundary=${boundary}` };
}

(async () => {
  try {
    // ------------------------------------------------------------ состояние
    const st = await json('GET', '/api/state');
    ok('/api/state отвечает', !!st.stages);
    ok('этапы перечислены', Array.isArray(st.stages) && st.stages.length >= 8, String((st.stages || []).length));
    ok('сценарии перечислены', Array.isArray(st.pipelines) && st.pipelines.length >= 3);
    ok('интерпретатор проекта', /\.venv[\\/]|python/.test(st.python || ''), st.python);
    ok('метрики из metrics.json', st.metrics && st.metrics.pearson > 0.8,
      st.metrics ? `Pearson ${st.metrics.pearson.toFixed(4)}` : '—');
    ok('комплекты в состоянии', Array.isArray(st.sets), String((st.sets || []).length));

    // ------------------------------------------------------------ статика
    const idx = await req('GET', '/');
    ok('index.html отдаётся', idx.status === 200 && /data-tab="run"/.test(idx.text));
    const app = await req('GET', '/app.js');
    ok('app.js отдаётся', app.status === 200 && app.text.length > 20000,
      `${(app.text.length / 1000).toFixed(0)} КБ`);
    const bundle = await req('GET', '/data_bundle.js');
    ok('data_bundle.js отдаётся', bundle.status === 200 && bundle.body.length > 1e6,
      `${(bundle.body.length / 1e6).toFixed(1)} МБ`);
    ok('статика закрыта от выхода из каталога', (await req('GET', '/../scripts/serve_dashboard.py')).status !== 200);

    // ------------------------------------------------------- белый список
    const bad = await json('POST', '/api/run', JSON.stringify({ stages: ['rm -rf /'] }));
    ok('неизвестный этап отклонён', bad.ok === false, bad.message);
    const badPl = await json('POST', '/api/run', JSON.stringify({ pipeline: 'x' }));
    ok('неизвестный сценарий отклонён', badPl.ok === false, badPl.message);
    ok('очередь свободна', !(await json('GET', '/api/state')).state.running);

    // ------------------------------------------------------------ модель
    const g = await json('GET', '/api/model/graph');
    ok('граф весов: слои', Array.isArray(g.layers) && g.layers.length > 10,
      `${(g.layers || []).length} слоёв, ${((g.params || 0) / 1e6).toFixed(2)} М`);
    ok('граф весов: гистограммы', !!(g.layers || [])[0] && (g.layers[0].hist || []).length > 10);
    ok('граф весов: рёбра', Array.isArray(g.edges) && g.edges.length > 5);
    const act = await json('GET', '/api/model/act');
    ok('активации по стадиям', Array.isArray(act.stages) && act.stages.length >= 3,
      `${(act.stages || []).length} стадий`);
    ok('активации 24×24', !!(act.stages || [])[0] && act.stages[0].data.length === 576);

    const hist = await json('GET', '/api/history');
    ok('история обучения', Array.isArray(hist.train) && hist.train.length > 3,
      `${(hist.train || []).length} точек`);
    ok('тестовая точка в истории', !!(hist.test && hist.test.ssim > 0));

    // -------------------------------------------------- запуск и «стоп»
    const run = await json('POST', '/api/run', JSON.stringify({ stages: ['probe'] }));
    ok('этап поставлен в очередь', run.ok === true, run.message);
    await sleep(600);
    const live = await json('GET', '/api/log?since=0');
    ok('лог стримится', Array.isArray(live.lines) && live.lines.length > 0,
      `${live.total} строк`);
    // ждём завершения probe
    for (let i = 0; i < 40 && (await json('GET', '/api/state')).state.running; i++) await sleep(250);
    ok('этап завершился', !(await json('GET', '/api/state')).state.running);
    const done = await json('GET', '/api/log?since=0');
    ok('в логе версия torch', /torch\s+\d/.test(done.lines.join('')),
      done.lines.join(' ').slice(0, 80));

    // «стоп» должен обрывать долгий этап: запускаем tests и сразу останавливаем
    const long = await json('POST', '/api/run', JSON.stringify({ stages: ['tests'] }));
    ok('долгий этап запущен', long.ok === true, long.message);
    await sleep(1200);
    const tStop = Date.now();
    const stop = await json('POST', '/api/stop');
    ok('/api/stop отвечает', stop.ok === true);
    let stopped = false;
    for (let i = 0; i < 30; i++) {
      if (!(await json('GET', '/api/state')).state.running) { stopped = true; break; }
      await sleep(200);
    }
    const stopMs = Date.now() - tStop;
    ok('стоп срабатывает быстрее 3 с', stopped && stopMs < 3000, `${stopMs} мс`);
    const stAfter = (await json('GET', '/api/state')).state;
    ok('состояние сброшено после стопа', !stAfter.running && stAfter.returncode !== null,
      JSON.stringify({ running: stAfter.running, rc: stAfter.returncode }));
    ok('остановка отражена в логе', /остановлено/.test((await json('GET', '/api/log?since=0')).lines.join('')));
    const again = await json('POST', '/api/stop');
    ok('повторный стоп безопасен', again.ok === true);

    // ------------------------------------------------------- комплекты
    const png1 = path.join(ROOT, 'data', 'source', 'object.png');
    const one = multipart([{ name: 'one.png', data: fs.readFileSync(png1), type: 'image/png' }]);
    const up1 = await json('POST', '/api/sets', one.body, one.ctype);
    ok('комплект из одного файла принят', up1.ok === true, up1.message);
    ok('один файл размножен на 4 плоскости',
      up1.info && up1.info.sources.length === 1 && up1.info.has_targets);
    const sid1 = up1.id;

    const names = ['object.png', 'object.png', 'object.png', 'object.png'];
    const four = multipart(names.map((n, i) => ({
      name: `p${i}.png`, data: fs.readFileSync(path.join(ROOT, 'data', 'source', n)), type: 'image/png' })));
    const up4 = await json('POST', '/api/sets', four.body, four.ctype);
    ok('комплект из четырёх файлов принят', up4.ok === true, up4.message);
    const sid4 = up4.id;

    // цели — 128×128, 8 бит, L
    const t1 = await req('GET', `/api/asset?set=${sid1}&name=target_1.png`);
    ok('цель отдаётся как PNG', t1.status === 200 && t1.body.slice(1, 4).toString() === 'PNG');
    const w = t1.body.readUInt32BE(16), h = t1.body.readUInt32BE(20);
    const bitDepth = t1.body[24], colorType = t1.body[25];
    ok('цель 128×128, 8 бит, grayscale', w === 128 && h === 128 && bitDepth === 8 && colorType === 0,
      `${w}×${h} ${bitDepth} бит тип ${colorType}`);
    ok('посторонний файл не отдаётся', (await req('GET', `/api/asset?set=${sid1}&name=../meta.json`)).status === 403);
    ok('несуществующий комплект', (await req('GET', '/api/asset?set=20200101-000000&name=DOE.png')).status === 404);

    // ---------------------------------------------------------- инференс
    const inf = await json('POST', '/api/infer', JSON.stringify({ set: sid4 }));
    ok('инференс запущен', inf.ok === true, inf.message);
    let ready = false;
    // один комплект считается ~1,5–2 мин на CPU (K = 2048 шагов физики),
    // поэтому ждём до 5 минут — это не зависание, а честный счёт
    for (let i = 0; i < 300; i++) {
      const s = await json('GET', '/api/state');
      if (!s.state.running) { ready = true; break; }
      await sleep(1000);
    }
    ok('инференс завершился', ready);
    const info = await json('GET', `/api/sets/${sid4}`);
    ok('DOE.png посчитан', info.has_doe === true);
    ok('метрики комплекта записаны', info.mean && info.mean.ssim > 0.2,
      info.mean ? `SSIM ${info.mean.ssim.toFixed(3)}` : '—');
    const doe = await req('GET', `/api/asset?set=${sid4}&name=DOE.png`);
    ok('DOE 128×128, 8 бит, L', doe.status === 200 && doe.body.readUInt32BE(16) === 128 &&
      doe.body[24] === 8 && doe.body[25] === 0,
      `${doe.body.readUInt32BE(16)}×${doe.body.readUInt32BE(20)} ${doe.body[24]} бит тип ${doe.body[25]}`);
    const rec = await req('GET', `/api/asset?set=${sid4}&name=recon_2.png`);
    ok('восстановление отдаётся', rec.status === 200 && rec.body.readUInt32BE(16) === 128);
    const met = await json('GET', `/api/asset?set=${sid4}&name=metrics.json`);
    ok('metrics.json содержит 4 плоскости', Array.isArray(met.planes) && met.planes.length === 4);

    // уборка: удаляем созданные комплекты
    await json('POST', `/api/sets/delete?id=${sid1}`);
    await json('POST', `/api/sets/delete?id=${sid4}`);
    ok('комплекты удалены', (await json('GET', '/api/sets')).sets.every(s => s.id !== sid1 && s.id !== sid4));
  } catch (e) {
    ok('пульт недоступен', false, String(e && e.message ? e.message : e));
  }
  const failed = checks.filter(c => !c.pass);
  for (const c of checks) console.log(`${c.pass ? 'OK  ' : 'FAIL'} ${c.name}${c.extra ? ' — ' + c.extra : ''}`);
  console.log(`\n${checks.length - failed.length}/${checks.length} проверок пульта за ${((Date.now() - t0) / 1000).toFixed(1)} с`);
  process.exit(failed.length ? 1 : 0);
})();
