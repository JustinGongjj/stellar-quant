"""冲击传导 v2：区分「短缺累积」与「成本传导衰减」两种机制。

v1 的缺陷
---------
v1 一律用 `0.55^跳` 衰减，即「离事件越远，影响越小」。
这在成本传导（涨价沿链被消化）时对，但在**短缺**时完全相反：
上游断供 → 下游缺料 → 越往下越是「有钱也买不到」，影响是**累积放大**的。

v2 的两条机制
-------------
1. **短缺累积（shortage）**：产能/供给受限类事件，影响沿下游 **放大**。
   模型：impact_{k+1} = min(1, impact_k × (1 + congestion))
   其中 congestion 由该环节的「瓶颈严重度」决定（5★=0.8, 4★=0.55, 3★=0.35）。
   上限封顶 1.0，避免无限增长。

2. **成本传导（cost）**：涨价类事件，影响沿下游 **衰减**。
   模型：impact_{k+1} = impact_k × passthrough
   passthrough 是下游能转嫁的比例（默认 0.5，即消化一半）。

3. **需求拉动（demand）**：下游放量，影响沿上游 **衰减**（上游按份额分享收益）。
   模型：impact_{k+1} = impact_k × share（默认 0.5）。
"""
import json, collections, math
from stellarq.paths import DATA  # noqa: E402
OUT = 'DATA/'
G = json.load(open(OUT / 'supply_graph.json', encoding='utf-8'))
B = json.load(open(OUT / 'industry_deepdive.json', encoding='utf-8'))
E = G['edges']

# 建立「环节 → 瓶颈严重度」查找表
SEV = {}
for seg, ind in B['industries'].items():
    for b in ind['bottlenecks']:
        SEV[b['node']] = b['severity']

sups, custs = collections.defaultdict(list), collections.defaultdict(list)
for e in E:
    sups[e['to']].append(e['from'])
    custs[e['from']].append(e['to'])

CONGESTION = {5: 0.80, 4: 0.55, 3: 0.35, 2: 0.20, 1: 0.10}

# 瓶颈名（概念） → 图谱节点（公司）的精映射。
# 上一版用子串匹配，只有 Lumentum / Vicor 能对上，导致累积几乎不触发。
BOTTLENECK_NODES = {
 'DSP 芯片':                   ['博通','Marvell'],
 '磷化铟衬底':                 ['云南锗业','光智科技','日本厂商'],
 'Lumentum':                   ['Lumentum'],
 '织布机（丰田）':              ['丰田织机'],
 '电子布/覆铜板':              ['生益科技'],
 '3D 钻孔设备':                ['沪电股份','深南电路'],
 'HBM':                        ['三星','SK海力士','美光'],
 '产能（内存厂仅六七座在建）':   ['三星','SK海力士','美光'],
 '中国厂商技术':                ['长鑫存储','长江存储'],
 '燃气轮机涡轮叶片':            ['三菱重工'],
 '垂直供电专利（Vicor）':       ['Vicor'],
 '氟化液（PFAS 限制）':         ['3M','氟化液市场'],
 '硅油导热系数':                ['润禾材料'],
 '灵巧手小型减速电机':          ['绿的谐波','双环传动'],
 '传感器':                     ['汇川技术'],
 '零件非标':                   ['人形机器人厂商'],
 '可复用回收':                  ['蓝箭航天','星河动力','星际荣耀'],
 '电磁弹射工程化':              ['星河动力','蓝箭航天'],
 '发射工位':                   ['商业火箭','商业火箭公司'],
 '先进制程':                   ['台积电','中芯国际','华虹'],
 'ASIC 替代':                  ['谷歌','AMD','寒武纪','海光'],
 '深海耐压与密封':              ['中科深海所'],
 '场景非标':                   ['潜行创新'],
}

# 反向：节点 → 严重度（取最高）
NODE_SEV = {}
for bn, nodes in BOTTLENECK_NODES.items():
    sev = SEV.get(bn, 0)
    for n in nodes:
        NODE_SEV[n] = max(NODE_SEV.get(n, 0), sev)


def severity_of(name):
    """查询某主体作为「供给方」时的瓶颈严重度。"""
    return NODE_SEV.get(name, 0)

def propagate(seeds, strength, mode, max_hop=4,
              passthrough=0.5, share=0.5):
    """mode: 'shortage'（累积放大）| 'cost'（衰减）| 'demand'（向上游衰减）"""
    adj = custs if mode in ('shortage', 'cost') else sups
    imp, frontier = {}, list(seeds)
    for s in frontier:
        imp[s] = strength
    for hop in range(max_hop):
        nxt = []
        for n in frontier:
            base = imp[n]
            for m in adj.get(n, []):
                if mode == 'shortage':
                    # 只有「来源是卡点」才累积；普通环节不加成
                    sev = severity_of(n)
                    cong = CONGESTION.get(sev, 0.0) if sev >= 4 else 0.0
                    # 累积按「每跳增量递减」而非几何放大，避免 3 跳即饱和
                    v = min(1.0, base + cong * (1 - base) * 0.75)
                elif mode == 'cost':
                    v = base * passthrough
                else:
                    v = base * share
                if v > imp.get(m, 0):
                    imp[m] = v
                    nxt.append(m)
        frontier = list(dict.fromkeys(nxt))
        if not frontier:
            break
    return imp

SCEN = {
 'S1 光模块禁运(3.2T)': {
   'seed': ['博通','Marvell','Lumentum'], 'strength': 0.45, 'mode': 'shortage', 'sign': -1,
   'logic': 'DSP 与电转光芯片断供，下游光模块缺料；越往下越缺（短缺累积）'},
 'S2 HBM 长协锁死': {
   'seed': ['三星','SK海力士','美光'], 'strength': 0.70, 'mode': 'shortage', 'sign': -1,
   'logic': 'HBM 产能锁死，AI 服务器缺料；数据中心抢走 70% 内存'},
 'S3 台积电涨价(2027)': {
   'seed': ['台积电'], 'strength': 0.55, 'mode': 'cost', 'sign': -1,
   'logic': '晶圆涨价 3–6%，下游消化一部分、转嫁一部分'},
 'S7 电子布/覆铜板涨价': {
   'seed': ['丰田织机','生益科技'], 'strength': 0.65, 'mode': 'cost', 'sign': -1,
   'logic': '电子布一年加价七八次，沿 PCB 链向下传导并被逐级消化'},
 'S9 数据中心电力瓶颈': {
   'seed': ['三菱重工'], 'strength': 0.70, 'mode': 'shortage', 'sign': -1,
   'logic': '涡轮叶片排单到 2029，数据中心并网受限'},
 'S4 Optimus 放量': {
   'seed': ['特斯拉'], 'strength': 0.75, 'mode': 'demand', 'sign': 1,
   'logic': '下游放量，上游零部件按份额分享'},
 'S5 电磁弹射首飞': {
   'seed': ['蓝箭航天'], 'strength': 0.60, 'mode': 'demand', 'sign': 1,
   'logic': '火箭放量拉动上游材料与器件'},
 'S8 PCB 产能扩张': {
   'seed': ['沪电股份','深南电路','长电科技'], 'strength': 0.55, 'mode': 'demand', 'sign': 1,
   'logic': '扩产拉动上游覆铜板、电子布、设备'},
 'S6 存储涨价链': {
   'seed': ['美光','三星'], 'strength': 0.50, 'mode': 'demand', 'sign': 1,
   'logic': '存储厂盈利改善，资本开支向上游传导'},
}

out = {'model': 'shortage=累积放大(1+congestion) | cost/demand=衰减(passthrough/share)',
       'congestion_by_severity': CONGESTION, 'scenarios': {}}

print('=== 冲击传导 v2（区分累积与衰减）\n')
for name, cfg in SCEN.items():
    imp = propagate(cfg['seed'], cfg['strength'], cfg['mode'])
    net = {k: round(v * cfg['sign'], 3) for k, v in imp.items()}
    net = dict(sorted(net.items(), key=lambda x: -abs(x[1])))
    mode_cn = {'shortage':'短缺累积','cost':'成本衰减','demand':'需求拉动'}[cfg['mode']]
    out['scenarios'][name] = {'logic': cfg['logic'], 'mode': cfg['mode'],
                              'mode_cn': mode_cn, 'seed': cfg['seed'],
                              'strength': cfg['strength'], 'sign': cfg['sign'],
                              'polarity': 'neg' if cfg['sign'] < 0 else 'pos',
                              'impact': dict(list(net.items())[:10])}
    print(f'--- {name}  [{mode_cn}]')
    for k, v in list(net.items())[:6]:
        bar = '█' * int(abs(v)*20)
        print(f'    {"↓" if v<0 else "↑"} {k:14s} {v:+.2f} {bar}')
    print()

# 对比：同一事件在累积 vs 衰减下的差异
print('=== 关键对比：S1 光模块禁运在两种机制下的差异')
a = propagate(SCEN['S1 光模块禁运(3.2T)']['seed'], 0.85, 'shortage')
b = propagate(SCEN['S1 光模块禁运(3.2T)']['seed'], 0.85, 'cost')
for k in ['博通','中际旭创','新易盛','微软']:
    if k in a or k in b:
        print(f'    {k:10s} 累积 {a.get(k,0):.2f} | 衰减 {b.get(k,0):.2f}')

json.dump(out, open(OUT / 'impact_forecast_v2.json','w'), ensure_ascii=False, indent=1)
print('\nsaved impact_forecast_v2.json')
