"""时钟框架的敏感性检验：换资产映射、换判定参数，结论还稳吗？

这一步是必须的——只有通过三条闸门的规则才算数：
  1. 换相近的映射仍成立
  2. 换判定参数仍成立
  3. 不是被单一年份驱动
"""
import pandas as pd, numpy as np, json, itertools
D = DATA
eq=pd.read_parquet('/tmp/eq_monthly.parquet')['eq_ret']
eq.index=eq.index.to_timestamp()

def sgn(s): return pd.Series(np.sign(s.to_numpy(dtype='float64')), index=s.index)

def build(m2, cpi, ppi, gdp, lookback=3):
    df=pd.DataFrame({'m2':m2,'cpi':cpi,'ppi':ppi,'gdp':gdp}).dropna()
    df['m2d']=sgn(df['m2'].diff(lookback)); df['ppid']=sgn(df['ppi'].diff(lookback))
    df['gdpd']=sgn(df['gdp'].diff(lookback)); df['cpid']=sgn(df['cpi'].diff(lookback))
    def ph(a,b):
        A,B=a,b
        if A>0 and B>0: return 0
        if A>0 and B<=0: return 1
        if A<=0 and B>0: return 2
        return 3
    df['new']=df.apply(lambda r: ph(r['m2d'],r['ppid']),axis=1)
    df['old']=df.apply(lambda r: ph(r['gdpd'],r['cpid']),axis=1)
    return df

# 重新载入原始序列
import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import requests
from stellarq.macro.clock import macro_panel
from stellarq.paths import DATA, ROOT  # noqa: E402
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
p={'reportName':'RPT_ECONOMY_CURRENCY_SUPPLY','columns':'ALL','pageNumber':1,'pageSize':500,
   'sortColumns':'REPORT_DATE','sortTypes':'1'}
r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=p,headers=H,timeout=30)
m2=pd.DataFrame(r.json()['result']['data'])[['REPORT_DATE','CURRENCY_SAME']]
m2.columns=['date','v']; m2['date']=pd.to_datetime(m2['date']).dt.to_period('M').dt.to_timestamp()
m2=m2.set_index('date')['v'].astype(float).sort_index()
mp=macro_panel('2012-01-01'); mp.index=pd.to_datetime(mp.index)
idx=m2.loc['2012-01-01':].index
cpi=mp['cpi_yoy'].reindex(idx,method='ffill'); ppi=mp['ppi_yoy'].reindex(idx,method='ffill')
gdp=mp['gdp_yoy'].reindex(idx,method='ffill')

# 多套资产映射（同一相位，股票做多还是回避）
MAPS={
 '标准（滞胀回避）':   {'new':[1,0,1,-1],'old':[1,1,-1,-1]},
 '全部做多':           {'new':[1,1,1,1],'old':[1,1,1,1]},
 '反向（滞胀做多）':   {'new':[-1,0,-1,1],'old':[-1,-1,1,1]},
 '仅货币维度':         {'new':[1,1,-1,-1],'old':[1,1,-1,-1]},
}

def acc(df,col,pref,lb):
    d2=df.dropna(subset=[col]).copy()
    d2['eq']=eq.reindex(d2.index).shift(-1)
    d2=d2.dropna(subset=['eq'])
    pr=d2[col].map(lambda i: pref[i] if i is not None else np.nan)
    ok=((pr==1)&(d2['eq']>0))|((pr==-1)&(d2['eq']<0))
    m=pr.notna()
    return (int(ok[m].sum()), int(m.sum()), float(ok[m].mean()) if m.sum() else np.nan)

print('=== 敏感性：判定窗口 × 资产映射\n')
rows=[]
for lb in (3,6,12):
    df=build(m2,cpi,ppi,gdp,lookback=lb)
    for mname,pref in MAPS.items():
        hn,tn,rn=acc(df,'new',pref['new'],lb)
        ho,to,ro=acc(df,'old',pref['old'],lb)
        rows.append({'lookback':lb,'map':mname,'new_rate':round(rn,3),'old_rate':round(ro,3),
                     'n':tn,'new_hit':hn,'old_hit':ho})
        print(f'  窗口{lb:2d}月 {mname:14s} 货币信用 {rn*100:5.1f}% ({hn}/{tn})  传统 {ro*100:5.1f}% ({ho}/{to})')
    print()

json.dump(rows, open(D / 'clock_sensitivity.json','w'), ensure_ascii=False, indent=1)
# 结论判定
new_wins=sum(1 for r in rows if r['new_rate']>r['old_rate'])
print(f'货币×信用优于传统的次数: {new_wins}/{len(rows)}')
print('结论:', '不稳定，不能宣称改进有效' if new_wins<len(rows)*0.75 else '相对稳健')
