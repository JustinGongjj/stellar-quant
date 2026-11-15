"""抓全市场季度财务快照（含公告日 NOTICE_DATE，用于无前视的特征对齐）。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

字段：营收、归母净利、同比增速、ROE、毛利率、EPS、BPS、每股经营现金流、净利环比、
       公告日、报告期。按报告期逐季抓，每季约 5000 条。
"""
import sys, time, json, os, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
CACHE='CACHE/fund'
os.makedirs(CACHE, exist_ok=True)
def w(m):
    with open(OUT / 'build_fund.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'build_fund.log','w').close()
try:
    import requests, pandas as pd
    H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
    URL='https://datacenter-web.eastmoney.com/api/data/v1/get'
    COLS=('SECURITY_CODE,SECURITY_NAME_ABBR,REPORTDATE,NOTICE_DATE,TOTAL_OPERATE_INCOME,PARENT_NETPROFIT,'
          'YSTZ,SJLTZ,WEIGHTAVG_ROE,XSMLL,BASIC_EPS,BPS,MGJYXJJE,SJLHZ,TRADE_MARKET')
    quarters=pd.period_range('2012Q1','2026Q3',freq='Q')
    for q in quarters:
        path=f'CACHE/{q}.parquet'
        if os.path.exists(path): continue
        rd=str(q.end_time.date())
        rows=[]; page=1
        while True:
            p={'reportName':'RPT_LICO_FN_CPD','columns':COLS,'pageNumber':page,'pageSize':500,
               'sortColumns':'SECURITY_CODE','sortTypes':'1','filter':f"(REPORTDATE='{rd}')"}
            for attempt in range(3):
                try:
                    j=requests.get(URL,params=p,headers=H,timeout=30).json(); break
                except Exception:
                    time.sleep(1.5); j={}
            res=(j.get('result') or {})
            data=res.get('data') or []
            rows.extend(data)
            if page>=int(res.get('pages') or 1) or not data: break
            page+=1
        df=pd.DataFrame(rows)
        if len(df): pd.DataFrame(df).to_parquet(path,index=False)
        else: pd.DataFrame(columns=COLS.split(',')).to_parquet(path,index=False)
        w(f'{q} rows {len(df)}')
    w('DONE')
except Exception:
    w(traceback.format_exc())
