"""行业分类抓取：用东财财务接口的 BOARD_NAME（申万二级），逐季抓一次全市场。
from stellarq.paths import CACHE, DATA, ROOT  # noqa: E402

注意：行业分类会变，所以按报告期抓取，构建「行业 × 时间」的时点分类。
"""
import sys, glob, json, time, os, traceback; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
CACHE='CACHE/industry'
os.makedirs(CACHE, exist_ok=True)
def w(m): print(m, flush=True)
try:
    import requests, pandas as pd
    H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
    URL='https://datacenter-web.eastmoney.com/api/data/v1/get'
    quarters=pd.period_range('2012Q1','2026Q3',freq='Q')
    for q in quarters:
        path=f'CACHE/{q}.parquet'
        if os.path.exists(path): continue
        rd=str(q.end_time.date()); rows=[]; page=1
        while True:
            p={'reportName':'RPT_LICO_FN_CPD','columns':'SECURITY_CODE,BOARD_NAME,REPORTDATE',
               'pageNumber':page,'pageSize':500,'sortColumns':'SECURITY_CODE','sortTypes':'1',
               'filter':f"(REPORTDATE='{rd}')"}
            j={}
            for a in range(3):
                try:
                    j=requests.get(URL,params=p,headers=H,timeout=30).json(); break
                except Exception: time.sleep(1.5)
            res=(j.get('result') or {}); data=res.get('data') or []
            rows.extend(data)
            if page>=int(res.get('pages') or 1) or not data: break
            page+=1
        df=pd.DataFrame(rows)
        if len(df):
            df['symbol']=df['SECURITY_CODE'].astype(str).str.zfill(6)
            df[['symbol','BOARD_NAME','REPORTDATE']].drop_duplicates('symbol').to_parquet(path,index=False)
        w(f'{q} {len(df)}')
    w('DONE')
except Exception:
    w(traceback.format_exc())
