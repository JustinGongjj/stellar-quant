"""把 forecast.json 转成前端可直接 <script> 引入的 forecast_data.js（避免 file:// 的 CORS 限制）。"""
import json, pathlib
src=pathlib.Path('reports/data/forecast.json')
dst=pathlib.Path('web/forecast_data.js')
data=json.loads(src.read_text(encoding='utf-8'))
dst.write_text('window.FORECAST = ' + json.dumps(data, ensure_ascii=False, indent=1) + ';\n', encoding='utf-8')
print('wrote', dst, dst.stat().st_size, 'bytes')
