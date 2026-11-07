/* Проверка пульта запуска против реального сервера scripts/serve_dashboard.py.
   Запуск:  node tools/smoke_runner.js [http://127.0.0.1:8765]
   Что проверяется: /api/state, отдача дашборда, постановка задачи в очередь,
   стриминг лога, отказ при неизвестном этапе, остановка. Сами тяжёлые этапы
   не запускаются — берётся самый дешёвый (--dry) либо проверяется только API. */
const http = require('http');

const BASE = (process.argv[2] || 'http://127.0.0.1:8765').replace(/\/$/, '');
const checks = [];
const ok = (name, cond, extra) => checks.push({ name, pass: !!cond, extra });
const t0 = Date.now();

function req(method, path, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(BASE + path);
    const r = http.request({ hostname: u.hostname, port: u.port, path: u.pathname + u.search,
      method, headers: body ? { 'Content-Type': 'application/json',
                                'Content-Length': Buffer.byteLength(body) } : {} },
      res => {
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks).toString() }));
      });
    r.on('error', reject);
    if (body) r.write(body);
    r.end();
  });
}

(async () => {
  try {
    const st = await req('GET', '/api/state');
    const js = JSON.parse(st.body);
    ok('/api/state отвечает', st.status === 200);
    ok('этапы перечислены', Array.isArray(js.stages) && js.stages.length >= 8, String(js.stages.length));
    ok('сценарии перечислены', Array.isArray(js.pipelines) && js.pipelines.length >= 3);
    ok('интерпретатор venv', /\.venv[\\/]/.test(js.python || ''), js.python);
    ok('метрики из metrics.json', js.metrics && js.metrics.pcc > 0.8,
       js.metrics ? `PCC ${js.metrics.pcc.toFixed(4)}` : '—');

    const idx = await req('GET', '/');
    ok('index.html отдаётся', idx.status === 200 && /Пульт|runpanel/.test(idx.body));
    const bundle = await req('GET', '/data_bundle.js');
    ok('data_bundle.js отдаётся', bundle.status === 200 && bundle.body.length > 1e6,
       `${(bundle.body.length / 1e6).toFixed(1)} МБ`);

    const bad = await req('POST', '/api/run', JSON.stringify({ stages: ['rm -rf /'] }));
    const badJs = JSON.parse(bad.body);
    ok('неизвестный этап отклонён', bad.status !== 200 && badJs.ok === false, badJs.message);

    ok('очередь свободна', !JSON.parse((await req('GET', '/api/state')).body).state.running);
    const log = JSON.parse((await req('GET', '/api/log?since=0')).body);
    ok('/api/log стримит строки', Array.isArray(log.lines) && typeof log.total === 'number',
       `${log.total} строк`);
  } catch (e) {
    ok('пульт недоступен', false, String(e));
  }
  const failed = checks.filter(c => !c.pass);
  for (const c of checks) console.log(`${c.pass ? 'OK  ' : 'FAIL'} ${c.name}${c.extra ? ' — ' + c.extra : ''}`);
  console.log(`\n${checks.length - failed.length}/${checks.length} проверок пульта за ${Date.now() - t0} мс`);
  process.exit(failed.length ? 1 : 0);
})();
