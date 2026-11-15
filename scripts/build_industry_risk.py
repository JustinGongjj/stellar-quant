"""把产业冲击转成个股层面的「产业风险分」，供量化模型做过滤。

风险分构造（越高越危险）：
  对每只股票，找出它所在的供应链节点
  - 若它是某个「短缺累积」场景的上游供给方 → 风险 = 该场景对其的影响值
  - 若它是下游受冲击方 → 有缓冲（可切换供应商），风险打 5 折
  - 若它在正向场景（需求拉动）中受益 → 风险取负值（机会）
"""
import json
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
G=json.load(open(OUT / 'supply_graph.json',encoding='utf-8'))
F=json.load(open(OUT / 'impact_forecast_v3.json',encoding='utf-8'))
CODES=json.load(open(OUT / 'stock_codes.json',encoding='utf-8'))

# 泛称节点 → 代表个股（把冲击传到具体标的）
GENERIC={
 'AI服务器厂商':['工业富联','立讯精密','沪电股份','深南电路','寒武纪','海光'],
 '人形机器人厂商':['三花智控','拓普集团','均胜电子','绿的谐波','双环传动','汇川技术'],
 '数据中心':['英维克','麦格米特','欧陆通'],
 '北美云厂商':['中际旭创','新易盛','天孚通信'],
 '微软云':['中际旭创','新易盛'],
 '国内云厂商':['寒武纪','海光'],
 '商业火箭':['中天火箭','博云新材','邵阳液压','中兵红箭','中国卫星','上海沪工'],
 '商业火箭公司':['中天火箭','博云新材','邵阳液压'],
 '商业卫星':['中国卫星','上海沪工','长江通信'],
 '低轨卫星':['中国卫星','上海沪工','长江通信','复旦微','紫光国微'],
}

# 公司 → 它在图中的位置
custs={}; sups={}
for e in G['edges']:
    custs.setdefault(e['from'],[]).append(e['to'])
    sups.setdefault(e['to'],[]).append(e['from'])

risk={}
detail={}
for sc_name, sc in F['scenarios'].items():
    sign = sc['sign']            # -1 负面 / +1 正面
    mode = sc['mode']
    for node, val in sc['impact'].items():
        targets=[]
        if node in CODES: targets=[node]
        elif node in GENERIC: targets=GENERIC[node]
        if not targets: continue
        for tgt in targets:
          code=CODES.get(tgt)
          if not code: continue
          # 泛称节点代表的是"下游客户群"，有替代缓冲 → 打折
          v=val
          if node in GENERIC: v*=0.5
          elif sign<0 and mode=='shortage' and node in sups: v*=0.5
          risk[code]=risk.get(code,0.0)+v
          detail.setdefault(code,[]).append({'scenario':sc_name,'node':node,'adj':round(v,3)})

# 归一化到 [-1,1]
if risk:
    mx=max(abs(v) for v in risk.values()) or 1
    risk={k:round(v/mx,3) for k,v in risk.items()}

out={'method':'负面场景上游=全值，下游受冲击=5折（有替代缓冲）；正向场景取负',
     'n_stocks':len(risk),'risk':dict(sorted(risk.items(), key=lambda x:-x[1])),
     'detail':detail,'name_of':{v:k for k,v in CODES.items()}}
json.dump(out, open(OUT / 'industry_risk.json','w'), ensure_ascii=False, indent=1)
print(f'覆盖 {len(risk)} 只股票')
print('\n风险最高（应回避）:')
for k,v in list(out['risk'].items())[:8]:
    print(f'  {out["name_of"].get(k,"?"):10s} {k}  {v:+.2f}')
print('\n受益（风险为负 = 机会）:')
for k,v in list(out['risk'].items())[-6:]:
    print(f'  {out["name_of"].get(k,"?"):10s} {k}  {v:+.2f}')
