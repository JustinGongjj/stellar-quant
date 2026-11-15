"""基本面快照（业绩预告/快报 + 定期报告的关键字段）。
from stellarq.paths import DATA, ROOT  # noqa: E402

数据源：巨潮公告只给标题，不给数值。数值优先用东财数据中心接口
（``RPT_LICO_FN_CPD``，含公告日期），东财若被限流则退化为「只用公告事件」。
本脚本先测东财是否可用，可用则抓全市场季度财务。
"""
import sys, time, json, traceback, os; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
OUT = DATA
def w(m):
    with open(OUT / 'build_fund.log','a') as f: f.write(str(m)+'\n')
open(OUT / 'build_fund.log','w').close()
try:
    import requests, pandas as pd, numpy as np
    H={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/'}
    url='https://datacenter-web.eastmoney.com/api/data/v1/get'
    p={'reportName':'RPT_LICO_FN_CPD','columns':'ALL','pageNumber':1,'pageSize':5,
       'sortColumns':'UPDATE_DATE','sortTypes':'-1','filter':'(REPORT_DATE=\'2024-12-31\')'}
    r=requests.get(url,params=p,headers=H,timeout=25)
    j=r.json()
    if not j.get('result'):
        w('东财数据中心不可用: '+str(j)[:200])
        sys.exit(0)
    rows=j['result']['data']
    w('字段: '+', '.join(rows[0].keys()))
    w('样本: '+json.dumps({k:rows[0][k] for k in list(rows[0])[:12]}, ensure_ascii=False, default=str))
except SystemExit:
    pass
except Exception:
    w(traceback.format_exc())
