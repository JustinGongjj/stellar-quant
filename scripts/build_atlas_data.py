"""把量化结果与产业研究写入「未来星图」的 DATA 结构（沿用原版 schema）。

原版每个域 = 因子星座（nodes+edges）+ 两主轴 + 四情景（scen）+ 驱动路径（paths）。
本脚本产出：
  * 量化域   —— 因子簇作为星座，真实 ICIR 作为载荷，真实相位作为情景
  * 产业域   —— AI / 具身智能 / 商业航天，依据用户提供的产业判断与公开事实
"""
import json, glob
import numpy as np, pandas as pd
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
BASE='/tmp/orig/data.json'
D=json.load(open(BASE, encoding='utf-8'))
CATS=D['cats']

# ============ 1) 量化域：用真实因子簇重写 macro ============
fc=json.load(open(OUT / 'factor_corr.json'))
mine=pd.read_csv(OUT / 'factor_mine_h5_partial.csv')
icir={r['factor']:float(r['icir']) for _,r in mine.iterrows()}
tv={r['factor']:float(r['t_nw']) for _,r in mine.iterrows()}
icm={r['factor']:float(r['ic_mean']) for _,r in mine.iterrows()}
LAW={'动量与位置':'动量效应 / 趋势跟踪','量能异动':'注意力假说 / 过度交易','量能波动':'拥挤度 / 不确定性溢价',
     '规模与活跃度':'小盘效应 / 流动性溢价','波动与换手':'低波动异象 / 换手率效应','形态与离散度':'微观结构 / 价量同步'}
CATMAP={'动量与位置':'economy','量能异动':'industry','量能波动':'structure','规模与活跃度':'structure',
        '波动与换手':'human','形态与离散度':'human'}
# 各因子簇的具体定义（回答「这个因子到底是什么」）
DEFS={
 '动量与位置': '【动量与位置】MA10/20/60 = 收盘价相对 N 日均线的偏离；MIN5/20 = 现价相对 N 日最低价的涨幅；'
               'RET20 = 过去 20 日累计涨幅。含义：买「已经在涨、且站在均线上方」的股票。',
 '量能异动':   '【量能异动】AMT_RATIO60 = 当日成交额 − 过去 60 日成交额均值（放量程度）；VSUMD60 = 60 日内'
               '(上涨日量 − 下跌日量)/(两者之和)，正值=放量上涨、负值=放量下跌；VMA60 = 成交量相对 60 日均量的偏离。'
               '含义：放量往往是情绪高点，后续偏弱——这是「量价背离」类因子。',
 '量能波动':   '【量能波动】VSTD20 = 过去 20 日「成交量的对数」的标准差，衡量成交量的不稳定程度。'
               'ICIR −0.669 是全部因子里最强的，含义：成交量忽大忽小的股票，后续显著跑输，是最强的排除型因子。',
 '规模与活跃度': '【规模与活跃度】LN_CAP = 流通市值的对数（小盘效应）；AMT_MA5 = 过去 5 日平均成交额的对数。'
               '含义：小盘 + 高活跃的股票，短期占优但长期偏弱。',
 '波动与换手': '【波动与换手】STD20 = 过去 20 日日收益标准差；TURN5 = 过去 5 日平均换手率。'
               '含义：高波动高换手的股票后续偏弱，是「低波动异象」的直接体现。',
 '形态与离散度': '【形态与离散度】KMID = (收盘−开盘)/开盘（当日实体方向）；CLOSE_POS = (收盘−最低)/(最高−最低)，'
               '收盘在当日区间的位置；CORR20 = 日收益与成交量变化的相关性。含义：刻画日内与量价的微观结构。',
}

# 横轴载荷：因子对「外部环境」的敏感度（-2..+2）
# 正 = 环境转好时更强；负 = 环境转差时反而更稳
LE_SENS={
 '动量与位置':  1.30,   # 强依赖趋势环境，风格反转时最先失效
 '量能异动':   -0.60,   # 放量异动多为情绪驱动，环境差时更明显
 '量能波动':   -1.40,   # 波动类偏防御，环境差时反而有效
 '规模与活跃度': -0.20,
 '波动与换手': -0.90,
 '形态与离散度': 0.40,
}

def lab_of(fs):
    s=set(fs)
    if s & {'MA10','MA20','MA60','MIN5','MIN20','RET20'}: return '动量与位置'
    if s & {'AMT_RATIO60','AMT_RATIO20','VSUMD60','VMA60'}: return '量能异动'
    if s & {'VSTD10','VSTD20','VSTD60'}: return '量能波动'
    if s & {'AMT_MA5','LN_CAP'}: return '规模与活跃度'
    if s & {'STD20','TURN5'}: return '波动与换手'
    return '形态与离散度'

nodes=[]; node_cluster={}
for ci,g in enumerate(fc['clusters_hier']):
    members=g['factors']; lab=lab_of(members)
    vals=[icir.get(m) for m in members if icir.get(m) is not None]
    mean_icir=float(np.mean(vals)) if vals else 0.0
    nid=f'q{ci:02d}'
    node_cluster[nid]=lab
    nodes.append({
      'id':nid,'cat':CATMAP.get(lab,'structure'),'name':f'{lab}（{len(members)} 因子）',
      'dir':'+' if mean_icir>0 else '−','hz':'全样本 2013–2026',
      'cert':int(min(95, 60+abs(mean_icir)*40)),'w':int(min(95, 45+len(members)*8)),
      # li = 纵轴载荷 = 因子的选股能力（ICIR 强度）
      'li':round(float(np.clip(mean_icir*4,-2,2)),2),
      # le = 横轴载荷 = 对外部环境（市场风格/政策/流动性）的敏感度
      # 依据：动量类依赖趋势环境 → 高敏感；波动/量能类偏防御 → 低敏感
      'le':LE_SENS.get(lab,0.0),
      'desc':('簇均 ICIR %+.3f（%d 个因子）。' % (mean_icir, len(members))) +
             ('正向簇：强者延续，可作为选股主线。' if mean_icir>0 else '避免型簇：指标越极端后续越弱，用于排除而非挑选。') +
             '　' + DEFS.get(lab,''),
      'sig':[m for m in members[:4]],
      'law':LAW.get(lab,'—'),
      'members':[{'f':m,'icir':icir.get(m),'ic':icm.get(m),'t':tv.get(m)} for m in members],
    })
# 边：簇间相关性（取成员相关性绝对值均值，>0.3 才连）
C=np.array(fc['corr']); F=fc['factors']
idx={f:i for i,f in enumerate(F)}
edges=[]
for a in range(len(nodes)):
    for b in range(a+1,len(nodes)):
        ma=fc['clusters_hier'][a]['factors']; mb=fc['clusters_hier'][b]['factors']
        vals=[abs(C[idx[x],idx[y]]) for x in ma for y in mb if x in idx and y in idx]
        if vals and np.mean(vals)>0.3:
            edges.append([nodes[a]['id'],nodes[b]['id'],round(float(np.mean(vals)),2)])

cl=pd.read_csv(OUT / 'clock_history.csv')
ph=cl['phase'].value_counts().to_dict(); tot=sum(v for k,v in ph.items() if k!='无法判定')
# 情景 = 真实美林时钟四相位；prior = 该相位的历史出现占比（真实统计）
PH_PROB={k:v/tot for k,v in ph.items() if k!='无法判定'}
SCEN=[('Q1','过热 · 动量占优','增长↑通胀↑，动量簇主导','+1','+1','#ff8663','过热'),
      ('Q2','复苏 · 反转占优','增长↑通胀↓，超跌与反转有效','+1','-1','#e9c37a','复苏'),
      ('Q3','滞胀 · 全线失效','增长↓通胀↑，量价因子普遍失效','-1','+1','#4fe3b6','滞胀'),
      ('Q4','衰退 · 政策托底','增长↓通胀↓，低波动与大盘占优','-1','-1','#69b0ff','衰退')]
scen=[]
for sid,nm,sub,I,E,glow,phname in SCEN:
    scen.append({'id':sid,'n':nm,'sub':sub,'I':int(I),'E':int(E),'c':'blue',
      'prior':round(float(PH_PROB.get(phname,0.05))-0.25,4),   # 相对均值的对数先验
      'glow':glow,'hist_share':round(float(PH_PROB.get(phname,0.0)),4),
      'tagline':f'{sub} · 历史占比 {PH_PROB.get(phname,0)*100:.1f}%',
      'path':['量价因子有效性随市场风格切换而轮动','ICIR 绝对值越高，该簇对组合贡献越大','避免型因子的作用是降低回撤而非增厚收益'],
      'challenge':['样本外仅 2.1 年，不足以判断长期有效性','冲击成本放大 4 倍则超额归零','面板缺 123 只退市股，实际业绩需打折'],
      'chance':['动量簇 + 量能波动簇的组合是当前最稳结构','CTA 与股票低相关（0.161）可改善风险收益'],
      'signal':['各簇 ICIR 的月度变化','CTA 与股票策略的滚动相关性']})
D['domains'][0]={
  'id':'quant','name':'量化因子轮动','short':'因子轮动','sub':'57 个量价因子的结构与有效性',
  'intro':'【这是什么】用 57 个量价因子构建的选股模型的「内在结构图」，不是宏观预测。'
            '纵轴 I = 该因子簇的选股能力（簇均 ICIR×4，正=能选出跑赢的股票，负=用来排除）；'
            '横轴 E = 该因子簇对外部市场环境（趋势/流动性/风格）的敏感度（正=环境好时更强，负=环境差时反而抗跌）。'
            '四个情景 = 因子有效性轮动的四种状态，概率由真实美林时钟相位占比给出。'
            '【不是什么】它不预测经济、不预测指数，只回答「当前市场状态下，哪类因子更管用」。',
  'temp':1.5,'axes':{'x':'外部环境 E','xp':'开放 +','xn':'割裂 −','y':'内部动能 I','yp':'强 +','yn':'弱 −'},
  'nodes':nodes,'edges':edges,'scen':scen,'paths':[],
}

# ============ 2) AI 域 ============
D['domains'].append({
 'id':'ai','name':'AI 未来','short':'AI','sub':'算力 · 模型 · 应用',
 'intro':'从算力供给、模型能力、应用落地三条线拆解 AI 的未来格局。因子来自公开产业事实与产业链跟踪指标。',
 'temp':1.6,'axes':{'x':'外部环境 E','xp':'开放 +','xn':'割裂 −','y':'内部动能 I','yp':'强 +','yn':'弱 −'},
 'nodes':[
  {'id':'a1','cat':'industry','name':'算力供给与电力约束','dir':'−','hz':'1–3 年','cert':70,'w':92,'li':1.8,'le':0.2,
   'desc':'GPU/HBM 供给与数据中心电力并网是当前最硬的物理约束。算力从训练转向推理，单位算力成本下降决定应用能否规模化。',
   'sig':['GPU 交付量','数据中心并网容量','推理/训练算力占比'],'law':'瓶颈迁移 / 规模经济'},
  {'id':'a2','cat':'industry','name':'模型能力曲线','dir':'+','hz':'1–3 年','cert':55,'w':85,'li':1.6,'le':0.3,
   'desc':'预训练 scaling law 边际递减，能力提升转向后训练（RL/推理时计算）与数据质量。',
   'sig':['基准分数斜率','推理成本/百万 token','上下文与工具调用能力'],'law':'Scaling Law / 边际递减'},
  {'id':'a3','cat':'industry','name':'应用落地与付费转化','dir':'+','hz':'2–4 年','cert':40,'w':70,'li':1.4,'le':0.6,
   'desc':'企业级付费从「试点」走向「规模化」是收入兑现的关键，但目前 ROI 证据仍分散。',
   'sig':['ARR 增速','付费席位渗透率','单客户 ARPU'],'law':'S 曲线 / 技术采纳'},
  {'id':'a4','cat':'geopol','name':'出口管制与国产替代','dir':'−','hz':'2–5 年','cert':72,'w':80,'li':1.2,'le':-1.4,
   'desc':'先进制程与 HBM 管制持续，国产芯片在推理侧替代加速，训练侧差距仍在。',
   'sig':['国产 AI 芯片市占率','先进制程产能','EDA 许可状态'],'law':'技术民族主义 / 李斯特'},
  {'id':'a5','cat':'institution','name':'监管与版权框架','dir':'−','hz':'2–5 年','cert':45,'w':60,'li':0.4,'le':-0.8,
   'desc':'训练数据版权、生成内容标识与合规成本，决定应用能否快速铺开。',
   'sig':['AI 法案进度','版权诉讼结果'],'law':'规则即成本'},
  {'id':'a6','cat':'human','name':'叙事与资本开支拥挤度','dir':'−','hz':'0–2 年','cert':42,'w':78,'li':0.9,'le':0.2,
   'desc':'算力资本开支与估值互为因果，叙事拥挤时边际买盘先耗尽。',
   'sig':['hyperscaler capex 指引','AI 相关估值分位'],'law':'反身性 / 叙事经济学'},
 ],
 'edges':[['a1','a2',0.7],['a1','a4',-0.6],['a2','a3',0.65],['a3','a5',-0.4],['a1','a6',0.55],['a2','a6',0.6]],
 'scen':[
  {'id':'A1','n':'算力军备竞赛延续','sub':'capex 继续超预期','I':1,'E':1,'c':'blue','prior':0.3,'glow':'#ff8663','tagline':'算力链最确定，应用仍待验证',
   'path':['云厂 capex 指引连续上修','推理需求爆发抵消训练放缓','电力与散热成为新瓶颈'],'challenge':['折旧年限与自由现金流矛盾','电力接入物理约束'],'chance':['电力、散热、光模块、HBM'],'signal':['季度 capex 指引','数据中心并网容量']},
  {'id':'A2','n':'应用兑现 · 价值迁移','sub':'从卖算力到卖结果','I':1,'E':-1,'c':'blue','prior':0.15,'glow':'#e9c37a','tagline':'应用层首次跑赢算力层',
   'path':['模型成本下降到可规模化','垂直场景 ROI 可验证','入口从模型转向工作流'],'challenge':['同质化与价格战','数据闭环壁垒不足'],'chance':['垂直 Agent、行业软件重构'],'signal':['ARR 增速','单客户 ROI 披露']},
  {'id':'A3','n':'算力过剩 · 去泡沫','sub':'capex 见顶回落','I':-1,'E':-1,'c':'blue','prior':0.1,'glow':'#4fe3b6','tagline':'估值先于盈利修正',
   'path':['某云厂首次下修 capex','GPU 二手价格崩落','高估值板块连锁下修'],'challenge':['被动资金放大回撤','算力资产减值'],'chance':['现金充裕者逆周期买入','推理侧成本红利'],'signal':['capex 指引下修','GPU 租赁价格']},
  {'id':'A4','n':'国产替代 · 双轨并行','sub':'管制下自成体系','I':-1,'E':1,'c':'blue','prior':0.05,'glow':'#69b0ff','tagline':'国产芯片推理侧放量',
   'path':['先进制程受限，推理侧国产化提速','国产软件栈与生态补齐','双轨下标准分裂'],'challenge':['训练侧差距未缩小','生态兼容成本'],'chance':['国产算力、EDA、IP'],'signal':['国产芯片市占率','软件栈适配度']},
 ],
 'paths':[],
})

# ============ 3) 具身智能域（沿用原版，保留）============
# 原版已有 embodied，不覆盖

# ============ 4) 商业航天域（按用户产业判断）============
D['domains'].append({
 'id':'space','name':'商业航天','short':'商业航天','sub':'电磁弹射 · 可复用 · 星座',
 'intro':'两条主线：电磁弹射（去掉一级助推器，把重量换成载荷）与可重复回收（降成本）。',
 'temp':1.7,'axes':{'x':'外部环境 E','xp':'开放 +','xn':'割裂 −','y':'内部动能 I','yp':'强 +','yn':'弱 −'},
 'nodes':[
  {'id':'s1','cat':'industry','name':'电磁弹射发射','dir':'+','hz':'3–8 年','cert':40,'w':88,'li':1.9,'le':0.6,
   'desc':'无一级助推器，省下的结构重量直接转为载荷。中国电磁弹射从几百公斤到几十吨均可弹射，靠储能飞轮灵活调节。星河动力超导磁悬浮电磁发射验证平台是全球唯一。',
   'sig':['弹射载荷吨位','储能飞轮能量密度','弹射频次','验证平台进展'],'law':'技术迁移 / 第一性原理'},
  {'id':'s2','cat':'industry','name':'可重复使用回收','dir':'+','hz':'2–5 年','cert':60,'w':85,'li':1.7,'le':0.4,
   'desc':'降本的核心手段。一级复用把发射成本压到化学火箭的几分之一，是星座经济性的前提。',
   'sig':['单枚复用次数','回收成功率','单位公斤入轨成本'],'law':'规模经济 / 学习曲线'},
  {'id':'s3','cat':'industry','name':'低轨星座与频轨资源','dir':'+','hz':'2–6 年','cert':72,'w':80,'li':1.2,'le':0.5,
   'desc':'频轨先占先得，组网速度决定资源归属；卫星批产能力是瓶颈。',
   'sig':['在轨卫星数','批产节拍','频轨申报量'],'law':'先占先得 / 网络效应'},
  {'id':'s4','cat':'geopol','name':'军民融合与政策','dir':'+','hz':'2–5 年','cert':65,'w':70,'li':1.0,'le':1.2,
   'desc':'军事技术民用化是顺理成章的路径：航母电磁弹射歼35、076 两栖舰无人机弹射已成熟，民用发射可承接同源技术。',
   'sig':['民商航天政策','军工技术转化项目'],'law':'军民融合 / 技术扩散'},
  {'id':'s5','cat':'structure','name':'载荷与成本结构','dir':'+','hz':'长周期','cert':55,'w':75,'li':1.4,'le':0.3,
   'desc':'长征五号起飞约 870 吨（≈600 多辆小汽车），长征二号F 约 493 吨、长征七号约 597 吨、长征十一号仅 58 吨。电磁弹射可省去庞大一级，载荷上限有量级提升空间。',
   'sig':['起飞重量','载荷系数','每公斤成本'],'law':'结构效率 / 第一性原理'},
  {'id':'s6','cat':'human','name':'资本与叙事热度','dir':'−','hz':'0–2 年','cert':35,'w':60,'li':0.5,'le':0.6,
   'desc':'商业航天叙事容易被单次成功放大，也容易因一次失败冷却；需以复用次数与订单验证。',
   'sig':['一级市场融资额','发射订单储备'],'law':'叙事经济学 / 反身性'},
 ],
 'edges':[['s1','s5',0.8],['s1','s4',0.6],['s2','s3',0.7],['s2','s5',0.65],['s3','s4',0.5],['s1','s6',0.45],['s2','s6',0.4]],
 'scen':[
  {'id':'S1','n':'电磁弹射突破','sub':'载荷量级跃升','I':1,'E':1,'c':'blue','prior':0.2,'glow':'#ff8663','tagline':'去掉一级，把重量换成载荷',
   'path':['超导磁悬浮弹射平台完成大质量验证','地面弹射 + 二级入轨成为新构型','单位公斤成本进入化学火箭的 1/5'],'challenge':['弹射瞬间过载对载荷的适配','储能与快速充电','空域与安全管制'],'chance':['发射服务、载荷集成、特种材料'],'signal':['弹射载荷吨位','验证平台试验频次']},
  {'id':'S2','n':'可复用降本','sub':'化学火箭路线收敛','I':1,'E':-1,'c':'blue','prior':0.3,'glow':'#e9c37a','tagline':'回收复用成为标配',
   'path':['一级复用次数突破两位数','星座组网成本大幅下降','发射从项目制变为航班化'],'challenge':['回收失败的单次成本','复用检修周期'],'chance':['星座运营、遥感数据'],'signal':['复用次数','入轨成本']},
  {'id':'S3','n':'双路线并行','sub':'弹射做货、火箭做人','I':1,'E':0,'c':'blue','prior':0.35,'glow':'#4fe3b6','tagline':'按载荷特性分工',
   'path':['大质量货物走电磁弹射','载人与高精度载荷走可复用火箭','两者共享测控与总装资源'],'challenge':['两套基础设施投入','标准不统一'],'chance':['混合发射服务体系'],'signal':['分工比例','基础设施投资']},
  {'id':'S4','n':'进度延后','sub':'工程验证周期超预期','I':-1,'E':-1,'c':'blue','prior':0.15,'glow':'#69b0ff','tagline':'叙事冷却，回归工程',
   'path':['弹射试验推迟','资本开支收缩','行业集中到少数玩家'],'challenge':['现金流断裂','人才流失'],'chance':['逆周期整合'],'signal':['试验节点延期','融资额下降']},
 ],
 'paths':[],
})

json.dump(D, open(OUT / 'atlas_data.json','w'), ensure_ascii=False)
print('domains:', [(d['id'], d['name'], len(d['nodes']), len(d['scen'])) for d in D['domains']])
