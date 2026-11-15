"""五碗面个股分析框架：基本面 / 政策面 / 技术面 / 情绪面 / 资金面。

每面给出可计算的分项得分（0-100）与依据，最后合成总分。
**不使用时钟象限做选股**——时钟只作为背景说明。

五个面各自的可计算指标：
  基本面：估值分位（PB 反向）、ROE、营收增速、利润率 —— 本仓只有 BP，其余缺
  政策面：是否属于受政策支持的产业环节（来自供应链图与瓶颈分析）
  技术面：均线位置、动量、相对强弱
  情绪面：换手率、波动率、量能异动（我们 ICIR 最强的因子簇）
  资金面：成交额趋势、大单占比 —— 本仓只有成交额
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402
OUT = DATA
C='CACHE/'; R=C+'wide2/'
M=lambda n: np.load(R+f'wide_{n}.npy', mmap_mode='r')
close=M('close'); turn=M('turnover'); amt=M('amount'); cap=M('float_mktcap')
valid=np.load(R+'wide_valid.npy')
syms=np.array([str(s) for s in np.load(R+'wide_syms.npy',allow_pickle=True)])
dates=np.load(R+'wide_dates.npy',allow_pickle=True)

# 取最新可得截面
t=len(close)-1
while t>0 and valid[t].sum()<500: t-=1
print(f'截面日期: {pd.Timestamp(dates[t]).date()}, 有效股票 {int(valid[t].sum())}')

def col(a,i0,i1): return np.asarray(a[i0:i1],dtype='float64')
c=col(close,t-60,t+1); a=col(amt,t-60,t+1); tu=col(turn,t-60,t+1); cp=col(cap,t-60,t+1)
cur=c[-1]; m=valid[t]&np.isfinite(cur)

def pct(x):
    """转成 0-100 的横截面分位。"""
    out=np.full(len(x),np.nan)
    ok=np.isfinite(x)&m
    if ok.sum()<50: return out
    r=pd.Series(x[ok]).rank(pct=True).to_numpy()*100
    out[ok]=r
    return out

# 技术面：MA20 偏离 + 20 日动量
ma20=np.nanmean(c[-20:],axis=0)
dev20=(cur-ma20)/ma20
mom20=cur/c[-21]-1
tech=0.5*pct(dev20)+0.5*pct(mom20)

# 情绪面：换手（反向，低换手好）+ 波动率（反向）
ret=np.full_like(c,np.nan); ret[1:]=c[1:]/c[:-1]-1
vol20=np.nanstd(ret[-20:],axis=0)
turn5=np.nanmean(tu[-5:],axis=0)
sent=0.5*pct(-turn5)+0.5*pct(-vol20)

# 资金面：成交额趋势（放量正向）—— 但我们知道放量后偏弱，故反向
amt5=np.nanmean(a[-5:],axis=0); amt20=np.nanmean(a[-20:],axis=0)
amt_ratio=np.log(amt5/amt20+1e-12)
fund=pct(-amt_ratio)   # 反向：近期放量的扣分

# 基本面：只有 BP 可得（1/PB），小市值也算基本面代理
bp=1.0/np.maximum(cp[-1],1e-9)
fundm=pct(bp)

# 政策面：缺数据，用「是否属于瓶颈环节受益方」代理 —— 本步先置空并标注
policy=np.full(len(cur),np.nan)

faces=pd.DataFrame({
 'code':syms,'tech':tech,'sent':sent,'fund':fund,'fundm':fundm,
 'turn5':turn5,'dev20':dev20,'mom20':mom20,'vol20':vol20,'amt_ratio':amt_ratio,'bp':bp,
})[m].dropna(subset=['tech','sent','fund','fundm'])

# 合成：等权（政策面缺失，先说明）
faces['score']=faces[['tech','sent','fund','fundm']].mean(axis=1)
faces=faces.sort_values('score',ascending=False)
print(f'覆盖 {len(faces)} 只')
print(faces[['code','score','tech','sent','fund','fundm']].head(10).round(1).to_string(index=False))
faces.to_parquet(OUT / 'five_faces_latest.parquet', index=False)

cov={'section':'基本面','available':['BP(账面市值比)'],'missing':['ROE','营收增速','利润率','现金流'],
     'note':'本仓基本面仅覆盖 59 个季度的 BP；ROE/营收等需接财报库'}
info={'as_of':str(pd.Timestamp(dates[t]).date()),'n_stocks':int(len(faces)),
      'weights':'技术25% 情绪25% 资金25% 基本面25%（政策面缺失，未计入）',
      'coverage':[cov,{'section':'政策面','available':[],'missing':['政策文件映射','补贴目录'],
                       'note':'需人工维护政策-产业映射表'}],
      'top10':faces.head(10)[['code','score']].to_dict('records')}
json.dump(info, open(OUT / 'five_faces_info.json','w'), ensure_ascii=False, indent=1)
print('\n已保存 five_faces_latest.parquet / five_faces_info.json')
