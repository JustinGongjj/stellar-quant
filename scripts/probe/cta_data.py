import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
from stellarq.data.tencent import Tencent
t=Tencent()
# 腾讯的期货/商品代码（新浪风格）：AU0 黄金、RB0 螺纹、C0 玉米、IF0 股指
for code in ['AU0','RB0','C0','AG0','CU0','M0','TA0','SC0','I0','JM0']:
    try:
        d=t.daily(code)
        print(code, len(d), d['date'].min() if len(d) else '', d['date'].max() if len(d) else '')
    except Exception as e:
        print(code,'ERR',str(e)[:60])
