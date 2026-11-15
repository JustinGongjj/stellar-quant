"""稳健性检验（独立脚本，分批算、逐批释放，避免管道撑爆）。"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
__import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.chanlun_state import state_series
from stellarq.paths import ROOT  # noqa: E402

CACHE=Path.home()/'.cache'/'stellarq'/'wide2'
def load(f): return np.load(CACHE/f'wide_{f}.npy',allow_pickle=(f in ('dates','syms')))

H=20
def main(n_sym=300, step=10, warm=400):
    dates=load('dates'); close=load('close')
    o,h,lo,vol=load('open'),load('high'),load('low'),load('volume')
    n_date,_=close.shape
    rng=np.random.default_rng(11)
    liq=np.nanmean(close[-250:]*vol[-250:],axis=0)
    cand=np.where(np.isfinite(liq)&(liq>0))[0]
    cand=cand[np.argsort(-liq[cand])][:n_sym*3]
    picks=np.sort(rng.permutation(cand)[:n_sym])
    print(f'{len(picks)} 只',flush=True)

    # 分批：每批 4 只，算完立刻投影成稀疏记录，不保留原始结构
    recs=[]  # (date, sym, bi_area, fwd20)
    p=subprocess.Popen(['node',str(Path(__file__).with_name('chan_srv.js'))],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,bufsize=1<<22,text=True)
    batch=[]
    def flush(batch):
        if not batch: return
        for j,bars in batch: p.stdin.write(json.dumps({'id':j,'sets':[bars]})+'\n')
        p.stdin.flush()
        for j,bars in batch:
            line=p.stdout.readline()
            try: rr=(json.loads(line)['results'] or [{}])[0] if line.strip() else {'ok':False}
            except: rr={'ok':False}
            if not rr.get('ok'): continue
            rr['bars']=bars
            st=state_series(rr)
            if st.empty: continue
            sd=st['date'].to_numpy()
            for t in range(warm,n_date-H-1,step):
                d=pd.Timestamp(dates[t])
                # 找到该日期在状态表里的行
                idx=np.searchsorted(sd,np.datetime64(d))
                if idx>=len(sd) or pd.Timestamp(sd[idx])!=d: continue
                x=st.iloc[idx]['cs_bi_area']
                c0,c1=close[t,j],close[t+H,j]
                if not np.isfinite(x) or not np.isfinite(c0) or not np.isfinite(c1) or c0<=0: continue
                recs.append((d,j,float(x),float(c1/c0-1)))
    try:
        for k,j in enumerate(picks):
            j=int(j); bars=[]
            for i in range(n_date):
                c_=close[i,j]
                if not np.isfinite(c_): continue
                bars.append({'t':int(pd.Timestamp(dates[i]).timestamp()*1000),
                    'o':float(o[i,j]) if np.isfinite(o[i,j]) else float(c_),
                    'h':float(h[i,j]) if np.isfinite(h[i,j]) else float(c_),
                    'l':float(lo[i,j]) if np.isfinite(lo[i,j]) else float(c_),
                    'c':float(c_),'v':float(vol[i,j]) if np.isfinite(vol[i,j]) else 0.0})
            if len(bars)<warm: continue
            batch.append((j,bars))
            if len(batch)>=4: flush(batch); batch=[]
            if k%40==0: print(f'  {k}/{len(picks)} 累计 {len(recs)} 行',flush=True)
        flush(batch)
    finally:
        try: p.stdin.close(); p.wait(timeout=10)
        except: p.kill()

    df=pd.DataFrame(recs,columns=['date','sym','x','y'])
    print(f'\n样本 {len(df)} 行 / {df.date.nunique()} 截面',flush=True)
    def ic_by(d):
        return d.groupby('date').apply(lambda g: g['x'].corr(g['y'],method='spearman') if len(g)>=30 else np.nan).dropna()
    base=ic_by(df)
    print(f"基准 IC={base.mean():+.4f} t={base.mean()/(base.std(ddof=1)/np.sqrt(len(base))):+.2f} n={len(base)}",flush=True)
    print('\n分年：',flush=True)
    for y,g in df.groupby(df.date.dt.year):
        s=ic_by(g)
        if len(s)>=3: print(f'  {y} IC={s.mean():+.4f} n={len(s)}',flush=True)
    d2=df.copy(); d2['x']=d2.groupby('date')['x'].transform(lambda s:s.clip(s.quantile(.01),s.quantile(.99)))
    s2=ic_by(d2)
    print(f"\nwinsorize IC={s2.mean():+.4f} t={s2.mean()/(s2.std(ddof=1)/np.sqrt(len(s2))):+.2f}",flush=True)
    df['q']=df.groupby('date')['x'].transform(lambda s: pd.qcut(s.rank(method='first'),5,labels=False) if s.notna().sum()>=50 else np.nan)
    g=df.dropna(subset=['q']).groupby(['date','q'])['y'].mean().unstack()
    if g.shape[1]==5:
        mn=g.mean()
        print('分5组: '+'  '.join(f'Q{i+1}={mn[i]*100:+.2f}%' for i in range(5)),flush=True)
        print(f'Q5-Q1 {((mn[4]-mn[0])*100):+.3f}%/20日  单调性={pd.Series(mn.values).corr(pd.Series(range(5)),method="spearman"):+.2f}',flush=True)

main()
