"""打板 / 龙头 打法：用日线数据能做与不能做的边界，以及能做的部分结论。
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402

数据能支持的：
  涨停判定（收盘价触及涨跌停价）、炸板（最高价触及但收盘未封住）、
  连板高度、首板/二板/三板、次日开盘溢价、T 收盘买入 T+1 开盘卖出的收益。
数据不能支持的（必须说明）：
  是否真的能买到（封单量/排队位置）、盘中分时是否开板、龙虎榜席位、
  集合竞价的量价。这些需要 tick / Level-2 / 分时数据，本仓没有。

因此本脚本给的是**上界估计**：假设涨停价一定能买到。
"""
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'limitup.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'limitup.log','w').close()
try:
    import numpy as np, pandas as pd
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,h,l,c,amt = M('open'),M('high'),M('low'),M('close'),M('amount')
    valid=np.load(R+'wide_valid.npy'); dates=np.load(R+'wide_dates.npy',allow_pickle=True); syms=np.load(R+'wide_syms.npy',allow_pickle=True)
    nd,ns=c.shape
    # 涨停幅度：创业板/科创板 20%，其余 10%（ST 已在建宽表时剔除）
    board=np.array([20.0 if str(s).startswith(('688','689','300','301')) else 10.0 for s in syms])
    lim=board[None,:]/100.0
    prev=np.full((nd,ns),np.nan,dtype=np.float64); prev[1:]=np.asarray(c[:-1],dtype=np.float64)
    close=np.asarray(c,dtype=np.float64); high=np.asarray(h,dtype=np.float64)
    ret=close/prev-1.0
    limit_px=np.round(prev*(1+lim)+1e-9, 2)
    zt = (close>=limit_px-1e-9) & valid & np.isfinite(prev)
    zb = (high>=limit_px-1e-9) & (~zt) & valid & np.isfinite(prev)   # 炸板
    w(f'样本 {nd} 交易日 x {ns} 股；涨停 {int(zt.sum())} 次日，炸板 {int(zb.sum())} 次日')
    # 连板高度
    cons=np.zeros((nd,ns),dtype=np.int16)
    for t in range(1,nd):
        cons[t]=np.where(zt[t], cons[t-1]+1, 0)
    # 次日开盘溢价（T 收盘买入 -> T+1 开盘卖出）
    nxt_open=np.full((nd,ns),np.nan); nxt_open[:-1]=np.asarray(o[1:],dtype=np.float64)
    nxt_ret=nxt_open/close-1.0
    # T+1 开盘买 -> T+1 收盘卖
    nxt_close=np.full((nd,ns),np.nan); nxt_close[:-1]=close[1:]
    d1_ret=nxt_close/nxt_open-1.0
    out={}
    # 1) 全部涨停日：次日开盘溢价
    for k in (1,2,3):
        m=zt&(cons==k)&np.isfinite(nxt_ret)
        if m.sum()<100: continue
        r=nxt_ret[m]
        out[f'连板{k}_次日开盘溢价'] = {'n':int(m.sum()),'mean':float(np.mean(r)),'median':float(np.median(r)),
                                  'win':float((r>0).mean()),'p10':float(np.percentile(r,10)),'p90':float(np.percentile(r,90))}
    # 2) 首板(T) 之后的 T+1 表现
    m=zt&(cons==1)&np.isfinite(d1_ret)
    if m.sum()>100:
        r=d1_ret[m]; out['首板_T+1开盘买收盘卖']={'n':int(m.sum()),'mean':float(np.mean(r)),'win':float((r>0).mean())}
    # 3) 炸板日之后
    m=zb&np.isfinite(nxt_ret)
    if m.sum()>100:
        r=nxt_ret[m]; out['炸板_次日开盘溢价']={'n':int(m.sum()),'mean':float(np.mean(r)),'win':float((r>0).mean())}
    # 4) 连板高度分布（按年）
    yr=pd.to_datetime(dates).year
    rows=[]
    for y in np.unique(yr):
        sel=(yr==y)
        d=zt[sel]
        rows.append({'year':int(y),'涨停次日':int(d.sum()),
                     '日均涨停数':float(d.sum()/max(sel.sum(),1)),
                     '3板以上':int((cons[sel]>=3).sum()),
                     '最高连板':int(cons[sel].max()) if d.any() else 0})
    # 5) 涨停日成交额特征（封板强弱的代理）
    amt_t=np.asarray(amt,dtype=np.float64)
    amt_prev=np.full((nd,ns),np.nan); amt_prev[1:]=amt_t[:-1]
    ratio=np.where(zt&np.isfinite(amt_prev)&(amt_prev>0), amt_t/np.where(amt_prev>0,amt_prev,np.nan), np.nan)
    m=np.isfinite(ratio)&np.isfinite(nxt_ret)
    if m.sum()>100:
        q=pd.qcut(pd.Series(ratio[m]),5,labels=False,duplicates='drop')
        grp=pd.DataFrame({'q':q,'nxt':nxt_ret[m]}).groupby('q')['nxt'].agg(['count','mean'])
        out['涨停日放量倍数_分组次日溢价']=grp.round(4).to_dict('index')
    res={'summary':out,'by_year':rows}
    json.dump(res, open(OUT / 'limitup.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    w('DONE '+str(round(time.time()-t0)))
except Exception:
    w(traceback.format_exc())
