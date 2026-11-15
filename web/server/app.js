/**
 * 星图服务：静态托管 + 数据 API（零依赖，Node 内置 http）。
 * 端口从环境变量 PORT 读取（平台不注入，必须由启动参数给定）。
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = Number(process.env.PORT || 8791);
const ROOT = path.resolve(__dirname, '..');          // stellar-quant/web
const DATA = path.resolve(ROOT, '..', 'reports', 'data');
const MAX_AGE = { '.js': 0, '.html': 0, '.json': 0 }; // 数据频繁更新，禁用强缓存

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
  '.zip': 'application/zip', '.gz': 'application/gzip', '.md': 'text/markdown; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8', '.txt': 'text/plain; charset=utf-8',
  '.parquet': 'application/octet-stream', '.npy': 'application/octet-stream',
};

const PKG_DIR = path.resolve(ROOT, '..', 'reports', 'package');   // 打包产物目录

/** 只允许读取白名单目录下的文件，杜绝路径穿越 */
function safeJoin(base, rel) {
  const p = path.resolve(base, rel);
  if (!p.startsWith(base + path.sep) && p !== base) return null;
  return p;
}

function send(res, code, body, type) {
  res.writeHead(code, {
    'Content-Type': type || 'text/plain; charset=utf-8',
    'Cache-Control': 'no-cache',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
  });
  res.end(body);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  let p = decodeURIComponent(url.pathname);

  if (p === '/healthz') return send(res, 200, JSON.stringify({ ok: true, ts: Date.now() }), MIME['.json']);

  // 投研工作站桥接：只读转发其公开健康检查与统计（不代理任何写操作）
  if (p === '/api/insight/stats') {
    const { execFileSync } = require('child_process');
    try {
      const out = execFileSync('python3', ['-c',
        'import sys,pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"src"));' +
        'from stellarq.bridge.insight import InsightHub; import json;' +
        'print(json.dumps(InsightHub().stats(), ensure_ascii=False))'],
        { timeout: 8000 });
      return send(res, 200, out.toString(), MIME['.json']);
    } catch (e) { return send(res, 503, JSON.stringify({ ok: false, error: String(e).slice(0,200) }), MIME['.json']); }
  }

  // 数据 API：/api/<name>.json 映射到 reports/data/<name>.json（白名单文件名）
  if (p.startsWith('/api/')) {
    const name = p.slice(5);
    if (!/^[A-Za-z0-9_.-]+\.json$/.test(name)) return send(res, 400, 'bad name');
    const f = safeJoin(DATA, name);
    if (!f || !fs.existsSync(f)) return send(res, 404, 'not found');
    return send(res, 200, fs.readFileSync(f), MIME['.json']);
  }

  // 成果包下载：/download/<file>.zip
  if (p.startsWith('/download/')) {
    const name = p.slice('/download/'.length);
    if (!/^[A-Za-z0-9_.\-]+\.(zip|tar\.gz)$/.test(name)) return send(res, 400, 'bad name');
    const f = safeJoin(PKG_DIR, name);
    if (!f || !fs.existsSync(f)) return send(res, 404, 'package not found');
    const st = fs.statSync(f);
    const type = name.endsWith('.zip') ? MIME['.zip'] : MIME['.gz'];
    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      let start = m && m[1] ? parseInt(m[1], 10) : 0;
      let end = m && m[2] ? parseInt(m[2], 10) : st.size - 1;
      if (start >= st.size) {
        res.writeHead(416, { 'Content-Range': `bytes */${st.size}` });
        return res.end();
      }
      end = Math.min(end, st.size - 1);
      res.writeHead(206, {
        'Content-Type': type, 'Content-Length': end - start + 1,
        'Content-Range': `bytes ${start}-${end}/${st.size}`,
        'Accept-Ranges': 'bytes', 'Content-Disposition': `attachment; filename="${name}"`,
      });
      return fs.createReadStream(f, { start, end }).pipe(res);
    }
    res.writeHead(200, {
      'Content-Type': type, 'Content-Length': st.size, 'Accept-Ranges': 'bytes',
      'Content-Disposition': `attachment; filename="${name}"`,
      'Cache-Control': 'no-cache',
    });
    return fs.createReadStream(f).pipe(res);
  }

  // 根路径先进开机动画，动画结束自动跳 atlas.html（带 ?to= 可直达其他页）
  if (p === '/' || p === '') p = '/boot.html';
  const file = safeJoin(ROOT, p.replace(/^\/+/, ''));
  if (!file || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return send(res, 404, 'not found');
  const ext = path.extname(file).toLowerCase();
  send(res, 200, fs.readFileSync(file), MIME[ext] || 'application/octet-stream');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`starmap service listening on ${PORT}`);
  console.log(`  page: /  -> boot.html
  page: /atlas.html -> atlas.html`);
  console.log(`  alt : /starmap_v2.html (卡片式版)`);
  console.log(`  api : /api/forecast.json`);
  console.log(`  dl  : /download/<name>.zip`);
});
