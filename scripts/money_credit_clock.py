"""美林时钟中国化 v2：「货币 × 信用」框架 + 与旧框架的实证对比。

背景
----
传统美林时钟（经济×通胀）在中国 2011 年后正确率仅约 28%（华泰李超团队回溯），
业内称「转成了电风扇」。三大失效原因：
  1. 政策与信用脉冲主导短周期，而 GDP/CPI 是滞后确认；
  2. 地产与地方财政是独立变量，不由通胀刻画；
  3. 外部规则成本（关税/管制）不由通胀刻画。

改良思路（公开方法论）
----------------------
用「货币 × 信用」两维替代「经济 × 通胀」：
  货币（M2 同比，或其变化方向）× 信用（社融/信贷脉冲）→ 四象限配资产。
  公开测算：货币+信用框架准确率约 73%（vs 传统 40%）。

本仓的数据现实
--------------
* M2 同比：可用（东财 RPT_ECONOMY_CURRENCY_SUPPLY）
* 社融存量/信贷脉冲：**接口不可用**（已试 6 个 reportName）
* 因此用「M2 同比的方向」代理货币，「PPI 同比的方向」代理信用/需求，
  并明确标注这是**降级方案**，不能宣称达到文献中的 73%。

产出：与旧框架（GDP×CPI）在同一区间上的资产择时正确率对比。
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, requests
OUT = DATA
H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}

def fetch_m2():
    p={'reportName':'RPT_ECONOMY_CURRENCY_SUPPLY','columns':'ALL',
       'pageNumber':1,'pageSize':500,'sortColumns':'REPORT_DATE','sortTypes':'1'}
    r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=p,headers=H,timeout=30)
    d=r.json()['result']['data']
    df=pd.DataFrame(d)[['REPORT_DATE','CURRENCY_SAME']]
    df.columns=['date','m2_yoy']
    df['date']=pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    df['m2_yoy']=pd.to_numeric(df['m2_yoy'],errors='coerce')
    return df.dropna().set_index('date').sort_index()

m2=fetch_m2()
print(f'M2: {len(m2)} 个月, {m2.index.min().date()} ~ {m2.index.max().date()}')

# 旧框架：GDP × CPI
from stellarq.macro.clock import macro_panel
from stellarq.paths import DATA, ROOT  # noqa: E402
mp=macro_panel('2012-01-01')
mp=mp.set_index(pd.to_datetime(mp.index))

# 对齐到月频
df=pd.DataFrame(index=m2.index)
df['m2']=m2['m2_yoy']
df['cpi']=mp['cpi_yoy'].reindex(df.index,method='ffill')
df['ppi']=mp['ppi_yoy'].reindex(df.index,method='ffill')
df['gdp']=mp['gdp_yoy'].reindex(df.index,method='ffill')
df=df.dropna()
print(f'可用区间: {df.index.min().date()} ~ {df.index.max().date()} 共 {len(df)} 个月')

# 新框架：货币方向（M2 3期差分）× 需求方向（PPI 3期差分）
# pandas 3 下 np.sign(Series) 会报 unorderable types，必须在 ndarray 上算
def sgn(s): return pd.Series(np.sign(s.to_numpy(dtype='float64')), index=s.index)
df['m2_dir']=sgn(df['m2'].diff(3))
df['ppi_dir']=sgn(df['ppi'].diff(3))
# 旧框架：增长方向（GDP 3期差分）× 通胀方向（CPI 3期差分）
df['gdp_dir']=sgn(df['gdp'].diff(3))
df['cpi_dir']=sgn(df['cpi'].diff(3))

def phase(row, a, b, names):
    A, B = row[a], row[b]
    if A>0 and B>0: return names[0]
    if A>0 and B<=0: return names[1]
    if A<=0 and B>0: return names[2]
    return names[3]

df['new']=df.apply(lambda r: phase(r,'m2_dir','ppi_dir',['宽货币宽信用','宽货币紧信用','紧货币宽信用','紧货币紧信用']),axis=1)
df['old']=df.apply(lambda r: phase(r,'gdp_dir','cpi_dir',['过热','衰退','滞胀','复苏']),axis=1)

# 正确率：用「下一季度该象限推荐资产」的实际表现是否为正来判定
# 北大类资产代理：股票=沪深300、债券=中债总财富指数（用 ETF 代理不可得，改用简化口径）
# 这里只做「相位稳定度」与「相位分布」的对比，并如实说明没有资产收益无法算正确率
res={'period':[str(df.index.min().date()),str(df.index.max().date())],'n_months':int(len(df))}
for k,label in [('new','货币×信用（本仓降级版）'),('old','经济×通胀（传统美林）')]:
    vc=df[k].value_counts()
    res[k]={'label':label,'distribution':{k2:int(v2) for k2,v2 in vc.items()},
            'switch_count':int((df[k]!=df[k].shift(1)).sum()),
            'avg_duration':round(float(len(df)/max((df[k]!=df[k].shift(1)).sum(),1)),2)}
    print(f'\n{label}')
    for ph,c in vc.items():
        print(f'   {ph:12s} {c:3d} 月  ({c/len(df)*100:.1f}%)')
    print(f'   相位切换次数 {res[k]["switch_count"]}，平均持续 {res[k]["avg_duration"]} 个月')

res['note']=('本仓缺社融/信贷脉冲数据（东财 6 个接口均不可用），'
             '故用 M2 方向代理货币、PPI 方向代理信用，是**降级方案**；'
             '文献中货币×信用框架准确率 73% 依赖真实社融数据，本仓不能宣称达到该水平。')
json.dump(res, open(OUT / 'clock_china_v2.json','w'), ensure_ascii=False, indent=1)
print('\n' + res['note'])
print('saved clock_china_v2.json')

# ============ 补：用真实资产收益算正确率 ============
# 资产代理：股票=全市场等权月收益；债券/商品无免费月度数据，只评股票方向
eq=pd.read_parquet('/tmp/eq_monthly.parquet')['eq_ret']
eq.index=eq.index.to_timestamp()

# 每个相位对应的股票推荐方向
STOCK_PREF={
 '宽货币宽信用': 1, '宽货币紧信用': 0, '紧货币宽信用': 1, '紧货币紧信用': -1,
 '复苏': 1, '过热': 1, '滞胀': -1, '衰退': -1,
}
df2=df.copy()
df2.index=pd.to_datetime(df2.index)
# 用「下一个月」的股票收益验证本月的判断（避免前视）
df2['eq_next']=eq.reindex(df2.index).shift(-1)
df2=df2.dropna(subset=['eq_next'])

def hit_rate(col):
    pref=df2[col].map(STOCK_PREF)
    valid=pref.notna()
    # 推荐做多 → 下月股票为正；推荐回避 → 下月股票为负
    correct=((pref==1)&(df2['eq_next']>0)) | ((pref==-1)&(df2['eq_next']<0))
    n=int(valid.sum()); c=int(correct[valid].sum())
    return {'n':n,'hit':c,'rate':round(c/n,3) if n else None}

new_r=hit_rate('new'); old_r=hit_rate('old')
res['stock_direction_accuracy']={
  'new':new_r,'old':old_r,
  'asset':'全市场等权月收益（本仓面板 5225 只）',
  'method':'本月相位 → 下月股票方向；推荐做多则要求下月为正，推荐回避则要求为负',
}
print('\n=== 股票方向正确率（下月验证）')
print(f"  货币×信用（降级版）: {new_r['hit']}/{new_r['n']} = {new_r['rate']*100:.1f}%")
print(f"  经济×通胀（传统）  : {old_r['hit']}/{old_r['n']} = {old_r['rate']*100:.1f}%")
json.dump(res, open(OUT / 'clock_china_v2.json','w'), ensure_ascii=False, indent=1)
print('已更新 clock_china_v2.json')
