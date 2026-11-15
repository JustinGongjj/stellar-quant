"""冲击传导 v3：引入份额权重。

与 v2 的区别
------------
v2 只判断「有无关系」，每跳的传导强度与是谁供给无关。
v3 用 edge.share 加权：
  * 短缺累积：congestion 再乘「该供应商在客户采购中的份额」
    —— 一家占 70% 的供应商断供，比占 5% 的严重得多
  * 多头供应缓冲：客户若有多个供应商，单一断供的影响按份额稀释
"""
import json, collections
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
G=json.load(open(OUT / 'supply_graph.json',encoding='utf-8'))
B=json.load(open(OUT / 'industry_deepdive.json',encoding='utf-8'))
E=G['edges']

BOTTLENECK_NODES = {
 'DSP 芯片':['博通','Marvell'],'磷化铟衬底':['云南锗业','光智科技','日本厂商'],
 'Lumentum':['Lumentum'],'织布机（丰田）':['丰田织机'],'电子布/覆铜板':['生益科技'],
 '3D 钻孔设备':['沪电股份','深南电路'],'HBM':['三星','SK海力士','美光'],
 '产能（内存厂仅六七座在建）':['三星','SK海力士','美光'],'中国厂商技术':['长鑫存储','长江存储'],
 '燃气轮机涡轮叶片':['三菱重工'],'垂直供电专利（Vicor）':['Vicor'],
 '氟化液（PFAS 限制）':['3M','氟化液市场'],'硅油导热系数':['润禾材料'],
 '灵巧手小型减速电机':['绿的谐波','双环传动'],'传感器':['汇川技术'],
 '零件非标':['人形机器人厂商'],'可复用回收':['蓝箭航天','星河动力','星际荣耀'],
 '电磁弹射工程化':['星河动力','蓝箭航天'],'发射工位':['商业火箭','商业火箭公司'],
 '先进制程':['台积电','中芯国际','华虹'],'ASIC 替代':['谷歌','AMD','寒武纪','海光'],
 '深海耐压与密封':['中科深海所'],'场景非标':['潜行创新'],
}
SEV={}
for ind in B['industries'].values():
    for b in ind['bottlenecks']: SEV[b['node']]=b['severity']
NODE_SEV={}
for bn,nodes in BOTTLENECK_NODES.items():
    for n in nodes: NODE_SEV[n]=max(NODE_SEV.get(n,0), SEV.get(bn,0))

sups, custs = collections.defaultdict(list), collections.defaultdict(list)
SHARE={}
for e in E:
    sups[e['to']].append(e['from']); custs[e['from']].append(e['to'])
    SHARE[(e['from'],e['to'])] = e.get('share')

CONG={5:0.80,4:0.55,3:0.35,2:0.20,1:0.10}
DEFAULT_SHARE=0.5   # 未核定份额时按均分假设

def propagate(seeds, strength, mode, max_hop=4, passthrough=0.5):
    adj = custs if mode in ('shortage','cost') else sups
    imp = {s: strength for s in seeds}
    frontier = list(seeds)
    for _ in range(max_hop):
        nxt=[]
        for n in frontier:
            base = imp[n]
            for m in adj.get(n,[]):
                sh = SHARE.get((n,m))
                if sh is None: sh = DEFAULT_SHARE
                if mode=='shortage':
                    sev = NODE_SEV.get(n,0)
                    cong = CONG.get(sev,0.0) if sev>=4 else 0.0
                    # 份额放大：占 70% 的供给方断供，累积效应按份额计
                    v = min(1.0, base + cong*sh*(1-base)*0.9)
                elif mode=='cost':
                    v = base*passthrough
                else:
                    v = base*sh if sh else base*passthrough
                if v > imp.get(m,0):
                    imp[m]=v; nxt.append(m)
        frontier=list(dict.fromkeys(nxt))
        if not frontier: break
    return imp

SCEN=json.load(open(OUT / 'impact_forecast_v2.json',encoding='utf-8'))['scenarios']
out={'model':'v3: 短缺累积×份额 | 成本/需求衰减(份额加权)','share_default':DEFAULT_SHARE,'scenarios':{}}

print('=== 冲击传导 v3（份额加权）\n')
for name,cfg in SCEN.items():
    imp=propagate(cfg['seed'],cfg['strength'],cfg['mode'])
    sign=cfg['sign']
    net={k:round(v*sign,3) for k,v in imp.items()}
    net=dict(sorted(net.items(), key=lambda x:-abs(x[1])))
    out['scenarios'][name]={'logic':cfg['logic'],'mode':cfg['mode'],'mode_cn':cfg['mode_cn'],
        'seed':cfg['seed'],'strength':cfg['strength'],'sign':sign,cfg.get('polarity') and 'polarity' or 'polarity':cfg.get('polarity','neg' if sign<0 else 'pos'),
        'impact':dict(list(net.items())[:10])}
    print(f'--- {name} [{cfg["mode_cn"]}]')
    for k,v in list(net.items())[:5]:
        print(f'    {"↓" if v<0 else "↑"} {k:14s} {v:+.2f}')
    print()

json.dump(out, open(OUT / 'impact_forecast_v3.json','w'), ensure_ascii=False, indent=1)
print('saved impact_forecast_v3.json')
