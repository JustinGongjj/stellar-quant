import requests
H={'User-Agent':'Mozilla/5.0'}
# FRED 中国相关序列可用性
tests={'CHNGDPNQDSMEI':'中国GDP','CPALTT01CNM659N':'CPI同比','IRLTLT01CNM156N':'10年国债','CHNB6BLTT02STSAQ':'?','MKTGDPCNA646NWDB':'GDP现价'}
for sid,name in tests.items():
    try:
        r=requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv',params={'id':sid,'cosd':'2012-01-01'},headers=H,timeout=25)
        lines=[l for l in r.text.strip().split('\n')[1:] if not l.endswith(',')]
        print(sid,name,r.status_code,len(lines), lines[-1] if lines else '')
    except Exception as e: print(sid,'ERR',str(e)[:60])
