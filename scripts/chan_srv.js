// 常驻批量计算。只回传结构元素，不回传原始 bars（bars 由 Python 侧自己持有）
// —— 之前回传 bars 导致单行 JSON 超过管道缓冲，读出来是被截断的。
const readline = require('readline');
const C = require(process.env.CHANLUN_LIB || './chanlun/lib/chan.js');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (!line.trim()) return;
  let job;
  try { job = JSON.parse(line); } catch (e) {
    process.stdout.write(JSON.stringify({ id: -1, error: 'bad json' }) + '\n');
    return;
  }
  const out = { id: job.id, results: [] };
  for (const bars of (job.sets || [])) {
    try {
      const r = C.analyze(bars, { period: 'day' });
      out.results.push({
        ok: true,
        nbars: bars.length,
        bis: r.bis || [], segments: r.segments || [], pivots: r.pivots || [],
        signals: r.signals || [], beichi: r.beichi || [], macd: r.macd || {},
      });
    } catch (e) {
      out.results.push({ ok: false, error: String(e.message) });
    }
  }
  process.stdout.write(JSON.stringify(out) + '\n');
});
