"""用缠论算法做真正无前视的 IC 检验。

关键修正
--------
上一版 IC 0.2556 是前视偏差造成的假象：
  面板截面停在 2026-09-18，而缠论 API 返回的是**今天(11-15)**的结构，
  等于用未来 6 周的结构去预测 9 月的收益。

正确做法
--------
直接调用缠论工作站的 analyze() 纯函数，传入**截至截面日**的 K 线，
在本地重算结构。这样结构里不含任何未来信息。
"""
import sys, json, subprocess, tempfile, os
__import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

OUT = DATA
C='CACHE/'; R=C+'wide2/'
op=np.load(R+'wide_open.npy',mmap_mode='r')
cl=np.load(R+'wide_close.npy',mmap_mode='r')
hi=np.load(R+'wide_high.npy',mmap_mode='r')
lo=np.load(R+'wide_low.npy',mmap_mode='r')
dates=np.load(R+'wide_dates.npy',allow_pickle=True)
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
valid=np.load(R+'wide_valid.npy')
nd=len(cl)

# 在几个历史截面上检验
CUTOFFS=[nd-70, nd-130, nd-190, nd-250]
H=5
print('截面设置:', [(t, str(pd.Timestamp(dates[t]).date())) for t in CUTOFFS])

# 把 K 线导出成 JSON 给 node 用
def slice_bars(t, n=300):
    i0=max(0,t-n+1)
    out={}
    for j,s in enumerate(syms):
        if not valid[t,j]: continue
        c=np.asarray(cl[i0:t+1,j],dtype='float64')
        if not np.isfinite(c).all() or (c<=0).any(): continue
        out[s]={
          't':[int(pd.Timestamp(d).value//10**6) for d in dates[i0:t+1]],
          'o':np.asarray(op[i0:t+1,j],dtype='float64').tolist(),
          'h':np.asarray(hi[i0:t+1,j],dtype='float64').tolist(),
          'l':np.asarray(lo[i0:t+1,j],dtype='float64').tolist(),
          'c':c.tolist(),
        }
    return out

NODE_SCRIPT = r'''
const {analyze} = require(process.env.CHANLUN_LIB || './chanlun/lib/chan.js');
const fs=require('fs');
const inp=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const out={};
for (const [code,d] of Object.entries(inp)){
  try{
    const n=d.t.length;
    const bars=[];
    for(let i=0;i<n;i++) bars.push({t:d.t[i],o:d.o[i],h:d.h[i],l:d.l[i],c:d.c[i],v:0});
    const r=analyze(bars,{period:'day',limit:300});
    if(!r||!r.ok) continue;
    const st=r.stats||{}, tr=r.trend||{}, piv=st.activePivot||{}, sig=r.signals||[];
    let lastSig=0, sigAge=null;
    if(sig.length){
      const s=sig[sig.length-1], nm=String(s.name||''), tp=String(s.type||'');
      if(nm.includes('买')||tp.endsWith('B')) lastSig=1;
      else if(nm.includes('卖')||tp.endsWith('S')) lastSig=-1;
      if(typeof s.barIdx==='number') sigAge=(r.bars.length-1-s.barIdx);
    }
    let posPct=null;
    if(st.price&&piv.zg&&piv.zd){ const mid=(piv.zg+piv.zd)/2, h=Math.max(piv.zg-piv.zd,1e-9); posPct=(st.price-mid)/h; }
    const TM={'上涨趋势':1,'下跌趋势':-1,'盘整':0};
    const PM={above:1,inside:0,in:0,below:-1};
    out[code]={ch_trend:TM[tr.type]??0, ch_pos:PM[st.position]??0, ch_pivots:st.pivots??0,
               ch_pos_pct:posPct, ch_last_signal:lastSig, ch_sig_age:sigAge};
  }catch(e){}
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
console.log('node done:', Object.keys(out).length);
'''

res={}
allrows=[]
for t in CUTOFFS:
    bars=slice_bars(t)
    with tempfile.TemporaryDirectory() as td:
        ip=os.path.join(td,'in.json'); opth=os.path.join(td,'out.json'); ns=os.path.join(td,'run.js')
        json.dump(bars, open(ip,'w'))
        open(ns,'w').write(NODE_SCRIPT)
        r=subprocess.run(['node',ns,ip,opth],capture_output=True,text=True,timeout=900)
        if r.returncode!=0:
            print(f't={t} node 失败: {r.stderr[:200]}'); continue
        feats=json.load(open(opth))
    # 未来 H 日收益
    fwd=np.asarray(op[t+1+H],dtype='float64')/np.asarray(op[t+1],dtype='float64')-1
    dl=pd.Series(np.where(valid[t],fwd,np.nan),index=syms)
    df=pd.DataFrame(feats).T.join(dl.rename('y'),how='inner').dropna(subset=['y'])
    df=df[pd.to_numeric(df['ch_pos_pct'],errors='coerce').notna() | True]
    print(f'  t={t} ({pd.Timestamp(dates[t]).date()}) 特征 {len(df)} 只')
    allrows.append((t,df))

# 汇总各截面的 IC
print('\n=== 各截面 RankIC（无前视）')
feats=['ch_trend','ch_pos','ch_pivots','ch_pos_pct','ch_last_signal','ch_sig_age']
for f in feats:
    ics=[]
    for t,df in allrows:
        if f not in df.columns: continue
        j=df[[f,'y']].dropna()
        j[f]=pd.to_numeric(j[f],errors='coerce'); j=j.dropna()
        if len(j)<500: continue
        ic=float(np.corrcoef(j[f].rank(), j['y'].rank())[0,1])
        ics.append(ic)
    if ics:
        res[f]={'n_cutoffs':len(ics),'ic_mean':round(float(np.mean(ics)),4),
                'ic_each':[round(x,4) for x in ics]}
        print(f'  {f:16s} IC均值={np.mean(ics):+.4f}  各截面={[round(x,4) for x in ics]}')

res['caveat']='用缠论 analyze() 在本地按截面日重算，结构不含未来信息；仍是截面检验，非完整回测'
json.dump(res, open(OUT / 'chanlun_ic_nolookahead.json','w'), ensure_ascii=False, indent=1)
print('\n', res['caveat'])
