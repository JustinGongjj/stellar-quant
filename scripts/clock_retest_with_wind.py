"""用新接入的 Wind 长序列重测时钟框架。

与上次（失败版）的区别
----------------------
上次只有 2012 年起、且 CPI/PPI/GDP 来自旧面板；这次新增：
  * 工业产能利用率（2013Q1 起）—— 产能周期，比 GDP 更贴近"过热/过剩"
  * 产成品存货 + 利润总额 —— 库存周期，这是中国短周期的主导变量
  * 全部走发布滞后，避免前视

三个框架对比：
  A 传统美林（GDP 方向 × CPI 方向）
  B 货币×信用（M2 方向 × PPI 方向）—— 上次 40.1%
  C 库存×产能（存货方向 × 产能利用率方向）—— 本次新增
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd
from stellarq.macro.wind_panel import month_end_panel, load_long

OUT = DATA
eq=pd.read_parquet('/tmp/eq_monthly.parquet')['eq_ret']
# 面板是月末日期(1-31)，收益是按月区间(1-01)标记 —— 必须统一到「月」这一层再对齐
if not isinstance(eq.index, pd.PeriodIndex):
    eq.index = pd.PeriodIndex(eq.index, freq='M')
eq.index = eq.index.asfreq('M')

m=month_end_panel()
# month_end_panel 已返回 PeriodIndex
w=load_long()
print(f'面板 {m.index.min()} ~ {m.index.max()}，{len(m)} 月')
print('非空率:', {c: f'{m[c].notna().mean()*100:.0f}%' for c in ['cpi_yoy','ppi_yoy','gdp_q_yoy','capacity_util','inventory','profit_cum']})

def sgn(s): return pd.Series(np.sign(s.to_numpy(dtype='float64')), index=s.index)

df=m.copy()
# 存货同比（存货是累计水平，需要同比）
inv_yoy = df['inventory'] / df['inventory'].shift(12) - 1
profit_yoy = df['profit_cum'] / df['profit_cum'].shift(12) - 1
df['inv_yoy']=inv_yoy
df['profit_yoy']=profit_yoy

def quadrant(a, b, names):
    out=[]
    for A,B in zip(a,b):
        if not (np.isfinite(A) and np.isfinite(B)): out.append(None); continue
        if A>0 and B>0: out.append(names[0])
        elif A>0 and B<=0: out.append(names[1])
        elif A<=0 and B>0: out.append(names[2])
        else: out.append(names[3])
    return pd.Series(out, index=a.index)

lb=3
df['gdp_d']=sgn(df['gdp_q_yoy'].diff(lb)); df['cpi_d']=sgn(df['cpi_yoy'].diff(lb))
df['ppi_d']=sgn(df['ppi_yoy'].diff(lb))
df['inv_d']=sgn(df['inv_yoy'].diff(lb)); df['cap_d']=sgn(df['capacity_util'].diff(lb))

df['A_trad']=quadrant(df['gdp_d'],df['cpi_d'],['过热','衰退','滞胀','复苏'])
df['C_invcap']=quadrant(df['inv_d'],df['cap_d'],['主动补库','被动补库','主动去库','被动去库'])

# 用外部 M2（上次已取）做 B
import requests
from stellarq.paths import DATA, ROOT  # noqa: E402
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
try:
    p={'reportName':'RPT_ECONOMY_CURRENCY_SUPPLY','columns':'ALL','pageNumber':1,'pageSize':500,
       'sortColumns':'REPORT_DATE','sortTypes':'1'}
    r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=p,headers=H,timeout=25)
    mm=pd.DataFrame(r.json()['result']['data'])[['REPORT_DATE','CURRENCY_SAME']]
    mm.columns=['date','m2']; mm['date']=pd.to_datetime(mm['date']).dt.to_period('M').dt.to_timestamp()
    m2s=mm.set_index('date')['m2'].astype(float).sort_index()
    # M2 约次月中旬发布
    m2s.index = (pd.PeriodIndex(m2s.index, freq='M') + 1)   # 次月才可见
    df['m2']=m2s.reindex(df.index,method='ffill')
    df['m2_d']=sgn(df['m2'].diff(lb))
    df['B_money']=quadrant(df['m2_d'],df['ppi_d'],['宽货币宽信用','宽货币紧信用','紧货币宽信用','紧货币紧信用'])
    print('M2 已接入')
except Exception as e:
    print('M2 不可用:', str(e)[:60])

# 股票偏好映射（每个框架各自的四象限）
PREF={
 'A_trad': {'过热':1,'复苏':1,'滞胀':-1,'衰退':-1},
 'B_money':{'宽货币宽信用':1,'宽货币紧信用':0,'紧货币宽信用':1,'紧货币紧信用':-1},
 'C_invcap':{'主动补库':1,'被动补库':-1,'主动去库':-1,'被动去库':1},
}

print('\n=== 各框架股票方向正确率（下月验证）')
res={}
for fw,pref in PREF.items():
    if fw not in df.columns: continue
    d2=df[[fw]].copy()
    # 用下个月的股票收益验证本月判断
    d2['eq']=eq.reindex(d2.index+1).to_numpy()
    d2=d2.dropna()
    if len(d2)<40: 
        print(f'  {fw}: 样本不足 {len(d2)}'); continue
    pr=d2[fw].map(pref)
    ok=((pr==1)&(d2['eq']>0))|((pr==-1)&(d2['eq']<0))
    mk=pr.notna()
    n=int(mk.sum()); h=int(ok[mk].sum())
    # 相位分布
    dist=d2[fw].value_counts().to_dict()
    res[fw]={'n':n,'hit':h,'rate':round(h/n,3) if n else None,'dist':dist,
             'switch':int((df[fw].dropna()!=df[fw].dropna().shift(1)).sum())}
    print(f'  {fw:10s} {h}/{n} = {h/n*100:.1f}%  切换{res[fw]["switch"]}次')
    print(f'            分布: {dist}')

# 基准
eq_next=eq.reindex(df.index+1)
base_pos=float((eq_next.dropna()>0).mean())
res['baseline_always_long']={'rate':round(base_pos,3),'note':'总是做多的胜率'}
print(f'\n  基准（总是做多）: {base_pos*100:.1f}%')

json.dump(res, open(OUT / 'clock_retest_wind.json','w'), ensure_ascii=False, indent=1)
print('\n已保存 clock_retest_wind.json')
