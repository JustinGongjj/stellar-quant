"""逐产业深度分析 v2。

v1 的失败：用「上游数 × 下游数」算瓶颈指数，但图谱太稀疏（多数节点上游1、下游1），
所有节点指数都是 4.00，完全无法排序。

v2 改为「事实优先」：
  瓶颈由**已知的产业事实**判定（唯一供应商、认证周期、产能约束），
  图谱只用于展示传导路径，不用于计算瓶颈强度。
  每条瓶颈都带依据（来自知识库的具体表述）。
"""
import json, collections
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
G=json.load(open(OUT / 'supply_graph.json',encoding='utf-8'))
E=G['edges']

# 已知瓶颈（人工判定 + 依据）
BOTTLENECKS={
 '光模块/光芯片':[
   {'node':'Lumentum','type':'唯一/寡头','fact':'磷化铟光芯片订单排到 2029 年，国内旭创/新易盛均向它采购','severity':5},
   {'node':'磷化铟衬底','type':'产能硬约束','fact':'国内云南锗业 15 万片/年 + 光智科技 12 万片/年，其余在日本，扩产周期长','severity':5},
   {'node':'DSP 芯片','type':'技术代差','fact':'来自博通/Marvell，依赖 5nm 光刻机，国内无法自主生产','severity':5},
   {'node':'织布机（丰田）','type':'设备垄断','fact':'全球仅两三家，丰田年卖几千台；国产织机可用一年、日本可用两年','severity':4},
 ],
 'PCB/覆铜板/电子布':[
   {'node':'电子布/覆铜板','type':'涨价周期','fact':'一年加价七八次，现 7.6 元/米，铜箔与环氧树脂同步跟涨','severity':3},
   {'node':'3D 钻孔设备','type':'资本门槛','fact':'一台几百万，一个厂要几千台','severity':3},
 ],
 '电源与电力':[
   {'node':'燃气轮机涡轮叶片','type':'技术+产能','fact':'三菱重工排单已到 2029 年，大尺寸叶片不断裂是门槛','severity':5},
   {'node':'垂直供电专利（Vicor）','type':'专利壁垒','fact':'US10,903,734 B1，2016 申请，产品要出货才发起 337 调查','severity':3},
 ],
 '液冷散热':[
   {'node':'氟化液（PFAS 限制）','type':'合规约束','fact':'散热强不腐蚀硅脂，但贵且受 PFAS 环保限制','severity':3},
   {'node':'硅油导热系数','type':'物理上限','fact':'导热比氟化液低，极限散热有天花板，展台 GPU 仍需加散热器','severity':3},
 ],
 '存储':[
   {'node':'HBM','type':'寡头+长协','fact':'三星/海力士/美光三家；30% 客户需交几十亿保证金；每季涨价 20%','severity':5},
   {'node':'产能（内存厂仅六七座在建）','type':'产能缺口','fact':'逻辑芯片产能上来后，内存厂仅六七座在建且多为旧厂改造/转产 HBM','severity':5},
   {'node':'中国厂商技术','type':'代差','fact':'国内长鑫/长江存储只能做 2–3 层，美光/海力士/三星可做多层高速','severity':4},
 ],
 '具身智能':[
   {'node':'灵巧手小型减速电机','type':'技术卡点','fact':'机器人零部件一万多种，关键在灵巧手的小型减速电机','severity':4},
   {'node':'传感器','type':'价值迁移','fact':'电机利润有限，传感器才是更赚钱的环节','severity':3},
   {'node':'零件非标','type':'供应链复杂度','fact':'零件一万多种，远多于手机（一千多），很多汽车零件无法直接沿用','severity':3},
 ],
 '商业航天':[
   {'node':'可复用回收','type':'技术门槛','fact':'朱雀三号 2026-08 已陆地回收，全球第二个掌握大运力可回收','severity':4},
   {'node':'发射工位','type':'基础设施','fact':'国内可复用火箭首飞窗口临近，发射工位紧张','severity':3},
   {'node':'电磁弹射工程化','type':'未验证','fact':'资阳试验中心 2025-04 首试 1.4m 模型，谷神星二号目标 2028 首飞，运力 3.5 吨','severity':4},
 ],
 '算力芯片':[
   {'node':'先进制程','type':'出口管制','fact':'先进制程与 HBM 管制持续，国产训练侧差距未明显缩小','severity':5},
   {'node':'ASIC 替代','type':'格局变化','fact':'2027 年定制芯片出货量预计首次超过 GPU（谷歌TPU 5GW、AMD 6GW、OpenAI 10GW）','severity':4},
 ],
 '水下机器人':[
   {'node':'深海耐压与密封','type':'工程门槛','fact':'FMTD 传感器 4812m 下 50MPa 结构完整，是长期验证的结果','severity':3},
   {'node':'场景非标','type':'无规模效应','fact':'没有统一场景、没有通用身体、没有可迁移数据，更像一家家工程公司','severity':4},
 ],
}

# 传导路径（用图谱实测）
sups=collections.defaultdict(list); custs=collections.defaultdict(list)
for e in E:
    sups[e['to']].append(e['from']); custs[e['from']].append(e['to'])

def chain(start, depth=0, path=None):
    path=path or [start]
    if depth>=5 or start not in custs: return path
    best=path
    for n in custs[start]:
        p=chain(n, depth+1, path+[n])
        if len(p)>len(best): best=p
    return best

out={'model':'瓶颈由产业事实判定，图谱仅用于展示传导路径',
     'n_edges':len(E),'n_nodes':len(G['degree']),'industries':{}}
print(f"图谱 {len(E)} 条关系 / {len(G['degree'])} 个主体\n")
for seg, bl in BOTTLENECKS.items():
    out['industries'][seg]={'bottlenecks':bl,
        'max_severity':max(b['severity'] for b in bl),
        'n_bottlenecks':len(bl)}
    print(f'=== {seg}  瓶颈 {len(bl)} 个')
    for b in sorted(bl, key=lambda x:-x['severity']):
        stars='★'*b['severity']
        print(f'   {stars:5s} [{b["type"]}] {b["node"]}')
        print(f'          {b["fact"]}')

print('\n=== 图谱实测最长传导链')
for src in ['云南锗业','丰田织机','三菱重工','博通','台积电']:
    if src in custs:
        p=chain(src)
        if len(p)>=3: print(f'   {" → ".join(p)}')

json.dump(out, open(OUT / 'industry_deepdive.json','w'), ensure_ascii=False, indent=1)
print('\nsaved industry_deepdive.json')
