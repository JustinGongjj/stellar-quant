"""打板的「成交概率」修正：把上界收益换算成给定时段内的真实期望。
from stellarq.paths import DATA, ROOT, WIDE  # noqa: E402

关键事实：涨停日 T 买入，若 T 日封住（不回封不住），你只能排队；能成交的前提是
有人在涨停价卖给你。可以事后观测的代理是：
  「T 日收盘封板 + T+1 开盘价 == 涨停价」→ 当天根本没成交机会（一字板/开盘即封）
  「T 日收盘封板 + T+1 开盘价 < 涨停价」→ 开盘可成交
本脚本用「T+1 开盘是否仍等于 T 日涨停价」估计「买不进」的比例，再用剩余部分计算净期望。
"""
import sys, time, json, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'limitup_fill.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'limitup_fill.log','w').close()
try:
    import numpy as np, pandas as pd
    t0=time.time()
    R='WIDE/'
    M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
    o,h,c = M('open'),M('high'),M('close')
    valid=np.load(R+'wide_valid.npy')
    nd,ns=c.shape
    lim=np.array([0.20 if str(s).startswith(('688','689','300','301')) else 0.10 for s in np.load(R+'wide_syms.npy',allow_pickle=True)],dtype=np.float64)
    cons=np.zeros(ns,dtype=np.int16)
    res={}
    for k in (1,2,3):
        buy_ok=[]; buy_fail=[]
        for t in range(1,nd-2):
            prev=np.asarray(c[t-1],dtype=np.float64); cur=np.asarray(c[t],dtype=np.float64)
            nxt=np.asarray(o[t+1],dtype=np.float64)
            ok=valid[t]&valid[t+1]&np.isfinite(prev)&(prev>0)
            lpx=np.round(prev*(1+lim)+1e-9,2)
            zt=ok&(cur>=lpx-1e-9)
            cons=np.where(zt,cons+1,0)
            m=zt&(cons==k)&np.isfinite(nxt)&(nxt>0)
            if not m.any(): continue
            # T+1 开盘即涨停 → 视为买不进
            one_word=m&(nxt>=lpx-1e-9)
            fillable=m&(nxt<lpx-1e-9)
            fillable &= np.isfinite(np.asarray(c[t+1],dtype=np.float64))
            if fillable.any():
                buy_ok.append(np.asarray(c[t+1],dtype=np.float64)[fillable]/nxt[fillable]-1.0)
            if one_word.any(): buy_fail.append(one_word[one_word].sum())
        if not buy_ok: continue
        v=np.concatenate(buy_ok); fail=int(np.sum(buy_fail)) if buy_fail else 0
        n_total=len(v)+fail
        res[f'{k}板']={'可成交样本':int(len(v)),'一字板(买不进)':fail,'可成交比例':float(len(v)/max(n_total,1)),
                       'T+1开盘买收盘卖_均值':float(v.mean()),'胜率':float((v>0).mean()),
                       '含买不进的期望(按0计)':float(v.sum()/max(n_total,1))}
        w(f'{k}板 done {round(time.time()-t0)}s')
    json.dump(res, open(OUT / 'limitup_fill.json','w'), ensure_ascii=False, indent=1, default=float)
    w(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    w('DONE')
except Exception:
    w(traceback.format_exc())
