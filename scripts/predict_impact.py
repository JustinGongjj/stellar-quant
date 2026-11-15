"""预测框架 v2：沿供应链传导事件冲击。

v1 的符号错误
-------------
v1 把所有场景都当「正向」扩散，导致「光模块禁运」让博通/Marvell 涨 0.85，
而「Optimus 放量」让特斯拉跌 0.75——符号与方向都反了。

v2 的两条规则
-------------
* **供给冲击**（上游被卡/涨价/断供）：沿 `customer` 方向**向下游**传导，**全链为负**。
  上游自己失市场、下游自己缺料，两边都受损。
* **需求冲击**（下游放量/政策利好）：沿 `supplier` 方向**向上游**传导，**全链为正**。
  下游多卖、上游多供，两边都受益。

衰减：每跳 ×0.55，最多 3 跳。
依赖度：in-degree 归一化，用于给「谁被卡影响更大」排序。
"""
import json, collections
from stellarq.paths import DATA  # noqa: E402
OUT = 'DATA/'
G = json.load(open(OUT / 'supply_graph.json', encoding='utf-8'))
E, deg = G['edges'], G['degree']

maxin = max(d['in'] for d in deg.values()) or 1
dep = {k: v['in'] / maxin for k, v in deg.items()}

sups, custs = collections.defaultdict(list), collections.defaultdict(list)
for e in E:
    sups[e['to']].append(e['from'])   # 客户 -> 供应商
    custs[e['from']].append(e['to'])  # 供应商 -> 客户


def propagate(seeds, strength, direction, decay=0.55, max_hop=3):
    """direction='down'（供给冲击，走向客户）/ 'up'（需求冲击，走向供应商）。"""
    adj = custs if direction == 'down' else sups
    imp, frontier = {}, {n: strength for n in seeds}
    for _ in range(max_hop):
        nxt = {}
        for n, v in frontier.items():
            imp[n] = max(imp.get(n, 0), v)
            for m in adj.get(n, []):
                nxt[m] = max(nxt.get(m, 0), v * decay)
        frontier = nxt
        if not frontier:
            break
    return imp


# polarity: neg = 负面冲击, pos = 正面机会
SCEN = {
 'S1 光模块禁运(3.2T)': {
   'seed': ['博通', 'Marvell', 'Lumentum'], 'strength': 0.85, 'direction': 'down', 'polarity': 'neg',
   'logic': '美国卡 DSP 与电转光芯片（5nm），不禁低技术环节；微软求白宫放开 3.2T 进口；65% 规则是缓冲'},
 'S2 HBM 长协锁死': {
   'seed': ['三星', 'SK海力士', '美光'], 'strength': 0.70, 'direction': 'down', 'polarity': 'neg',
   'logic': '30% 客户需交几十亿保证金；数据中心抢走 70% 内存；每季涨价 20%；中小客户被挤出原厂直采'},
 'S3 台积电涨价(2027)': {
   'seed': ['台积电'], 'strength': 0.55, 'direction': 'down', 'polarity': 'neg',
   'logic': '2027 晶圆涨价 3–6%，先进封装订单到 2027；毛利 70%+ 具备定价权'},
 'S4 Optimus 放量': {
   'seed': ['特斯拉'], 'strength': 0.75, 'direction': 'up', 'polarity': 'pos',
   'logic': '周产数百台→月产过千，2027 底对公众销售，长期年产 1000 万台目标；零部件一万多种'},
 'S5 电磁弹射首飞': {
   'seed': ['蓝箭航天'], 'strength': 0.60, 'direction': 'up', 'polarity': 'pos',
   'logic': '谷神星二号目标 2028 首飞，运力 3.5 吨；星河动力超导磁悬浮验证平台'},
 'S7 电子布/覆铜板涨价': {
   'seed': ['丰田织机', '生益科技'], 'strength': 0.65, 'direction': 'down', 'polarity': 'neg',
   'logic': '电子布一年加价七八次至 7.6 元/米；织布机全球仅两三家（丰田），国产可用一年、日本两年；铜箔与环氧树脂同步跟涨'},
 'S8 PCB 产能扩张': {
   'seed': ['沪电股份', '深南电路', '长电科技'], 'strength': 0.55, 'direction': 'up', 'polarity': 'pos',
   'logic': '长电 Q3 净利 7 亿翻倍、集资 65 亿扩产；PCB 龙头四家、光模块龙头三家，门槛高；3D 钻孔设备一台几百万、一厂要几千台'},
 'S9 数据中心电力瓶颈': {
   'seed': ['三菱重工'], 'strength': 0.70, 'direction': 'down', 'polarity': 'neg',
   'logic': '涡轮叶片门槛高、排单已到 2029 年；AI 数据中心抢居民用电与土地、电费涨 40%、审批暂缓'},
 'S6 铝电解/存储涨价链': {
   'seed': ['美光', '三星'], 'strength': 0.5, 'direction': 'up', 'polarity': 'pos',
   'logic': '闪迪毛利 85%、净利 80 亿美元；存储厂大额回购计划带来边际买盘'},
}

out = {'model': 'impact = strength × 0.55^hop；供给冲击向下游、需求冲击向上游',
       'decay': 0.55,
       'dependence_top': {k: round(v, 3) for k, v in
                          sorted(dep.items(), key=lambda x: -x[1])[:12]},
       'scenarios': {}}

print('=== 被依赖度 Top8（谁被卡，影响面最大）')
for k, v in sorted(dep.items(), key=lambda x: -x[1])[:8]:
    print(f'  {k:14s} {v:.3f}  (被 {deg[k]["in"]} 家依赖)')

for name, cfg in SCEN.items():
    chain = propagate(cfg['seed'], cfg['strength'], cfg['direction'])
    sign = -1 if cfg['polarity'] == 'neg' else 1
    net = {k: round(v * sign, 3) for k, v in chain.items()}
    net = dict(sorted(net.items(), key=lambda x: -abs(x[1])))
    out['scenarios'][name] = {'logic': cfg['logic'], 'seed': cfg['seed'],
                              'direction': cfg['direction'], 'polarity': cfg['polarity'],
                              'strength': cfg['strength'], 'impact': dict(list(net.items())[:10])}
    print(f'\n=== {name}  [{"负面" if sign<0 else "正面"} / 向{"下游" if cfg["direction"]=="down" else "上游"}]')
    print(f'    依据：{cfg["logic"]}')
    for k, v in list(net.items())[:5]:
        print(f'      {"↓" if v < 0 else "↑"} {k:14s} {v:+.3f}')

json.dump(out, open(OUT / 'impact_forecast.json', 'w'), ensure_ascii=False, indent=1)
print('\nsaved impact_forecast.json')
