"""生成研报口径的图表与指标表数据（无 matplotlib，纯 SVG 输出，中文可读）。

产出：
  reports/figs/*.svg       —— 净值曲线、超额曲线、IC 序列、分子图
  reports/data/report_metrics.json —— 报告指标表（对照国金月报表格字段）
"""
import sys, json, resource; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
OUT = REPORTS
C='CACHE/'
df=pd.read_csv(OUT / 'data/clock_model_periods.csv', parse_dates=['date'])
# ---- 指标表
def stats(net, bench=None):
    ov=net.rolling(5).mean().dropna()
    r={'n':int(len(ov)),'ann':float(ov.mean()*10.08),'vol':float(ov.std(ddof=1)*np.sqrt(10.08)),
       'sharpe':float(ov.mean()/ov.std(ddof=1)*np.sqrt(10.08))}
    nav=(1+ov).cumprod(); r['max_dd']=float((nav/nav.cummax()-1).min())
    if bench is not None:
        ob=bench.rolling(5).mean().dropna(); ex=ov-ob
        r['bench_ann']=float(ob.mean()*10.08); r['excess_ann']=float(ex.mean()*10.08)
        r['te']=float(ex.std(ddof=1)*np.sqrt(10.08)); r['ir']=float(ex.mean()/ex.std(ddof=1)*np.sqrt(10.08))
        e=(1+ex).cumprod(); r['excess_max_dd']=float((e/e.cummax()-1).min())
        r['win_vs_bench']=float((ex>0).mean())
    return r
m={'strategy':stats(df['net'], df['bench']), 'benchmark':stats(df['bench'])}
json.dump(m, open(OUT / 'data/report_metrics.json','w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k:{kk:round(vv,4) if isinstance(vv,float) else vv for kk,vv in v.items()} for k,v in m.items()}, ensure_ascii=False, indent=1))

# ---- SVG 工具
W,H=980,380; PAD_L,PAD_R,PAD_T,PAD_B=70,30,30,50
def sx(i,n): return PAD_L+(W-PAD_L-PAD_R)*i/max(n-1,1)
def svg_line(chart_title, series, ylabel, path, yfmt='{:.2f}', colors=None, step=5):
    ys=[v for _,vals,_ in series for v in vals if np.isfinite(v)]
    if not ys: return
    lo,hi=min(ys),max(ys); lo=lo-(hi-lo)*0.08; hi=hi+(hi-lo)*0.08
    n=max(len(v) for _,v in [(a,b) for a,b,_ in series])
    def sy(v): return PAD_T+(H-PAD_T-PAD_B)*(1-(v-lo)/(hi-lo+1e-12))
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Noto Sans CJK SC, system-ui, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           f'<text x="{PAD_L}" y="20" font-size="15" font-weight="600" fill="#1a1a1a">{chart_title}</text>']
    for k in range(6):
        v=lo+(hi-lo)*k/5; y=sy(v)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="#666">{yfmt.format(v)}</text>')
    for j,(name,vals,color) in enumerate(series):
        pts=' '.join(f'{sx(i,len(vals)):.1f},{sy(v):.1f}' for i,v in enumerate(vals) if np.isfinite(v))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        parts.append(f'<rect x="{PAD_L+j*170}" y="{H-24}" width="14" height="3" fill="{color}"/>')
        parts.append(f'<text x="{PAD_L+j*170+20}" y="{H-19}" font-size="11.5" fill="#333">{name}</text>')
    parts.append(f'<text x="{PAD_L}" y="{H-38}" font-size="11" fill="#888">{ylabel}</text>')
    parts.append('</svg>')
    open(path,'w',encoding='utf-8').write('\n'.join(parts))
    print('wrote', path)

import os; os.makedirs(OUT / 'figs', exist_ok=True)
from stellarq.paths import CACHE, REPORTS, ROOT  # noqa: E402
net=(1+df['net']).cumprod(); bench=(1+df['bench']).cumprod()
ren=(1+df['net'].rolling(5).mean().fillna(0)).cumprod(); rben=(1+df['bench'].rolling(5).mean().fillna(0)).cumprod()
ex=(1+(df['net']-df['bench']).rolling(5).mean().fillna(0)).cumprod()
svg_line('图1  策略净值 vs 等权基准（5 日重叠修正）',[('策略净值',ren.tolist(),'#c0392b'),('基准净值',rben.tolist(),'#2c3e50')],'净值（起点=1）',OUT / 'figs/fig1_nav.svg')
svg_line('图2  超额净值（策略/基准）',[('超额净值',ex.tolist(),'#16a085')],'超额净值（起点=1）',OUT / 'figs/fig2_excess.svg')
dd_ser=(ren/ren.cummax()-1).tolist()
svg_line('图3  策略回撤',[('回撤',dd_ser,'#8e44ad')],'回撤',OUT / 'figs/fig3_dd.svg')

# ---- 图4：分相位超额（柱状）
g=df.groupby('phase').agg(n=('net','size'),port=('net','mean'),bench=('bench','mean'))
g['excess']=g['port']-g['bench']
ph=g.reindex([p for p in ['复苏','过热','滞胀','衰退','无法判定'] if p in g.index]).dropna()
BW,BH=980,360; base=BH-110; top=70; span=base-top
mx=max(abs(ph['excess'].max()),abs(ph['excess'].min()),1e-4)*1.3
bars=[]
for i,(name,row) in enumerate(ph.iterrows()):
    x=110+i*160; h=abs(row['excess'])/mx*span
    y=base-h if row['excess']>=0 else base
    color='#c0392b' if row['excess']>=0 else '#16a085'
    bars.append(f'<rect x="{x}" y="{y:.1f}" width="76" height="{max(h,1):.1f}" fill="{color}" opacity="0.88"/>')
    if abs(row['excess'])*100 < 0.25:      # 近似为 0 的柱子：标签放到柱子上方固定位置，避免与横轴重叠
        ly = base - 14
    else:
        ly = (y - 10) if row['excess'] >= 0 else (y + h + 18)
    bars.append(f'<text x="{x+38}" y="{ly:.1f}" font-size="12" text-anchor="middle" fill="#222" font-weight="600">{row["excess"]*100:+.2f}%</text>')
    bars.append(f'<text x="{x+38}" y="{base+24}" font-size="13" text-anchor="middle" fill="#222">{name}</text>')
    bars.append(f'<text x="{x+38}" y="{base+41}" font-size="10.5" text-anchor="middle" fill="#999">n={int(row["n"])}</text>')
svg4=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{BW}" height="{BH}" font-family="Noto Sans CJK SC, system-ui, sans-serif">
<rect width="{BW}" height="{BH}" fill="#fff"/>
<text x="110" y="28" font-size="15" font-weight="600" fill="#1a1a1a">图4  各美林时钟相位下的策略超额（期均，5 日持有）</text>
<text x="110" y="48" font-size="11.5" fill="#777">横轴为时钟相位，纵轴为「策略收益 − 等权基准收益」</text>
<line x1="80" y1="{base}" x2="{BW-40}" y2="{base}" stroke="#bbb"/>
{''.join(bars)}
<text x="110" y="{BH-16}" font-size="11" fill="#888">红=跑赢基准，绿=跑输；n 为落在该相位的期数</text>
</svg>'''
open(OUT / 'figs/fig4_phase.svg','w',encoding='utf-8').write(svg4)
print('wrote fig4')
