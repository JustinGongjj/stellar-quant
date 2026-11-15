"""给各域情景分配有意义的二维坐标（修复重叠与居中问题）。

问题根源
--------
* 原版用 I/E ∈ {+1,-1} 落四象限，导致「四个情景=四角」；
* 我给部分情景填了中间值（如 E3 的 +0.5/-0.3），它们就挤在中心；
* 低空经济 L2/L3 坐标完全相同（+1,-1），所以重叠。

修复
----
每个情景按其「产业自身强度 I」与「外部环境 E」的真实判断给连续坐标，
并保证任意两个情景的欧氏距离 ≥ 0.35。
"""
import json, itertools, math
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
D=json.load(open(OUT / 'atlas_data.json',encoding='utf-8'))

COORDS={
 'quant':{    # 纵=I 因子选股能力占优程度，横=E 市场环境友好度
  'Q1':( 0.85, 0.75),'Q2':( 0.60,-0.70),'Q3':(-0.80, 0.65),'Q4':(-0.55,-0.85)},
 'embodied':{ # 纵=I 产业自身兑现强度，横=E 外部需求环境
  'E1':( 0.88, 0.55),'E2':( 0.45,-0.60),'E3':( 0.15, 0.35),'E4':(-0.90,-0.75)},
 'space':{    # 纵=I 技术/工程自身突破程度，横=E 外部政策与资本环境
  'S1':( 0.70, 0.80),'S2':( 0.85,-0.35),'S3':( 0.35, 0.55),'S4':(-0.85,-0.70)},
 'lowalt':{   # 纵=I 产业自身成熟度，横=E 政策与外部条件
  'L1':( 0.75, 0.85),'L2':( 0.60,-0.45),'L3':( 0.10,-0.85),'L4':(-0.85,-0.30)},
 'ai':{
  'A1':( 0.85, 0.70),'A2':( 0.55,-0.55),'A3':(-0.80,-0.70),'A4':(-0.30, 0.60)},
 'uspolitics':{
  'P1':(-0.75,-0.60),'P2':(-0.20, 0.65),'P3':( 0.80,-0.50),'P4':(-0.55, 0.30)},
 'china':{    # 纵=I 内生动能，横=E 外部规则环境
  'C1':( 0.72,-0.42),'C2':( 0.55, 0.78),'C3':(-0.75,-0.55),'C4':(-0.20,-0.80)},
}

changed=0
for i,d in enumerate(D['domains']):
    c=COORDS.get(d['id'])
    if not c: continue
    for s in d['scen']:
        if s['id'] in c:
            I,E=c[s['id']]
            if (s.get('I'),s.get('E'))!=(I,E): changed+=1
            s['I'],s['E']=I,E
    D['domains'][i]=d

# 校验最小间距
bad=[]
for d in D['domains']:
    for a,b in itertools.combinations(d['scen'],2):
        dist=math.hypot(a['I']-b['I'], a['E']-b['E'])
        if dist<0.35: bad.append((d['name'],a['id'],b['id'],round(dist,2)))
json.dump(D, open(OUT / 'atlas_data.json','w'), ensure_ascii=False)
print(f'调整 {changed} 个情景坐标')
print('重叠警告:', bad if bad else '无（所有情景间距 ≥0.35）')
