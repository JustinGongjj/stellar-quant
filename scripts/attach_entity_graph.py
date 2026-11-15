"""把实体图挂到星图域，并新增「美股·政治政策」域。"""
import json
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
D=json.load(open(OUT / 'atlas_data.json',encoding='utf-8'))
G=json.load(open(OUT / 'entity_graph.json',encoding='utf-8'))

SEG2DOMAIN={'光模块/光芯片':'ai','液冷散热':'ai','电源模块':'ai','PCB与封装':'ai','存储/内存':'ai',
            '算力芯片':'ai','具身智能零部件':'embodied','商业航天':'space','水下机器人':'space',
            'AI应用':'ai'}

for i,d in enumerate(D['domains']):
    did=d['id']
    segs=[s for s in G['segments'] if SEG2DOMAIN.get(s['id'])==did]
    if not segs: continue
    detail=[]
    for s in segs:
        cos=[{k:v for k,v in c.items() if k!='evidence'} | {'name':name}
             for name,c in G['companies'].items() if c['segment']==s['id']]
        cos.sort(key=lambda x:-x['mentions'])
        detail.append({'segment':s['id'],'layer':s['layer'],'chokepoint':s['chokepoint'],
                       'n_docs':s['docs'],
                       'companies':[{'name':c['name'],'mentions':c['mentions']} for c in cos]})
    d['supply_chain']=sorted(detail,key=lambda x:-x['n_docs'])
    D['domains'][i]=d

# 新增「美股·政策」域
POLICY_NOTES = [
 {'id':'p_ai_ctrl','cat':'geopol','name':'AI 芯片控制权','dir':'−','hz':'0–2 年','cert':75,'w':92,'li':1.8,'le':0.4,
  'desc':'两党共识点：共和党举国家安全，民主党举金融稳定，共同目标是让英伟达交出芯片控制权。2026-11 黄仁勋辞去全部职务后，两院四委员会五机构同时点名。',
  'sig':['国会听证排期','出口许可变化','分拆/剥离提案'],'law':'技术民族主义 / 政治俘获'},
 {'id':'p_fin_stab','cat':'institution','name':'金融稳定监管（民主党主线）','dir':'−','hz':'1–3 年','cert':60,'w':85,'li':1.0,'le':-0.8,
  'desc':'民主党更倾向走「金融稳定 + 反垄断 + 剥离」路线而非单纯出口管制。对大型科技的分拆压力与 AI 安全监管，是其选民基本盘（科技工会、消费者、盟友体系）的诉求。',
  'sig':['反垄断诉讼','AI 安全立法','平台分拆提案'],'law':'反垄断 / 公共利益监管'},
 {'id':'p_ally','cat':'geopol','name':'盟友协同与规则输出','dir':'+','hz':'1–4 年','cert':55,'w':70,'li':0.9,'le':-1.2,
  'desc':'民主党更可能恢复盟友协同（荷兰、日本设备管制联动）与多边规则输出，对中国半导体设备的间接约束上升，但同时降低单边关税的随机性。',
  'sig':['多边管制协调','盟友出口政策'],'law':'联盟政治 / 规则竞争'},
 {'id':'p_tariff','cat':'geopol','name':'关税与 301 周期性','dir':'~','hz':'0–2 年','cert':70,'w':80,'li':0.3,'le':-1.0,
  'desc':'301 豁免 178 项延至 2027-01-09。关税的随机性高于方向性——两党在对外强硬上无分歧，差异在于执行节奏与可预测性。',
  'sig':['301 豁免到期','豁免清单变化','USTR 公告'],'law':'贸易政治 / 转移支付'},
 {'id':'p_distract','cat':'human','name':'国内困局外溢','dir':'−','hz':'长期','cert':50,'w':65,'li':-0.8,'le':-0.5,
  'desc':'历史模式：总统在国内被困住手脚时，力量往往转向对外（克林顿 1998 年丑闻三天后空袭苏丹/阿富汗，弹劾表决前一日发动沙漠之狐）。观察国会山的同时，要看同一周军舰、关税、制裁去了哪里。',
  'sig':['军事行动频次','制裁名单扩张'],'law':'转移视线 / 政治周期'},
 {'id':'p_capex','cat':'economy','name':'AI capex 与估值反身性','dir':'−','hz':'0–2 年','cert':45,'w':88,'li':1.2,'le':0.3,
  'desc':'2026 hyperscaler capex 约 7,325 亿美元，2027 预期 1.2 万亿。capex 推高供应商收入，收入支撑估值，边际买盘耗尽时反向自我强化。政治干预可能成为触发点。',
  'sig':['季度 capex 指引','AI 板块估值分位','被动资金占比'],'law':'反身性 / 叙事经济学'},
 {'id':'p_chip_cycle','cat':'industry','name':'内存长协化','dir':'+','hz':'1–3 年','cert':55,'w':75,'li':1.3,'le':0.2,
  'desc':'美光转向五年期长协、30% 客户需交几十亿美元保证金，内存周期波动可能弱化；中小客户被挤出原厂直采转向模组厂。',
  'sig':['长协占比','保证金规模','HBM 供需缺口'],'law':'周期弱化 / 寡头定价'},
 {'id':'p_optics','cat':'industry','name':'光模块禁运风险','dir':'−','hz':'0–2 年','cert':60,'w':82,'li':-1.4,'le':0.5,
  'desc':'大摩警告美国可能禁运中国 3.2T 光模块。真正卡的是 DSP 与电转光芯片（依赖 5nm），低技术环节不会禁。65% 规则：零部件价值 65% 由美国制造则不禁运。',
  'sig':['DSP 国产化进度','65% 规则适用性','客户认证'],'law':'技术管制 / 价值链切割'},
]
POLICY_SCEN=[
 {'id':'P1','n':'监管收紧 · 估值修正','sub':'政治干预成为触发点','I':-1,'E':-1,'c':'blue','prior':0.32,'glow':'#ff8663','hist_share':None,
  'tagline':'两党共识压制大科技',
  'path':['听证会转为正式立法压力','分拆/剥离提案写入法案','AI 板块估值随 capex 指引下修而修正'],
  'challenge':['被动资金放大回撤','分拆改变公司结构'],'chance':['防守型资产、能源与公用事业'],
  'signal':['听证排期','反垄断诉讼','capex 指引']},
 {'id':'P2','n':'盟友协同 · 规则输出','sub':'多边管制替代单边关税','I':0.2,'E':1,'c':'blue','prior':0.26,'glow':'#e9c37a','hist_share':None,
  'tagline':'设备管制联动升级',
  'path':['荷日设备管制与美协调','多边出口框架成形','单边关税随机性下降'],
  'challenge':['盟友执行力度不一'],'chance':['国产设备替代加速'],'signal':['多边协调公告']},
 {'id':'P3','n':'capex 延续 · 政治噪声','sub':'基本面压过政治','I':1,'E':-0.5,'c':'blue','prior':0.28,'glow':'#4fe3b6','hist_share':None,
  'tagline':'听证不改变算力需求',
  'path':['capex 指引继续上修','政治事件仅造成短期波动','龙头在波动中继续扩张'],
  'challenge':['政策尾部风险'],'chance':['算力链、光模块、液冷'],'signal':['capex 指引','订单能见度']},
 {'id':'P4','n':'对外转向 · 风险外溢','sub':'国内困局转移视线','I':-0.5,'E':1,'c':'blue','prior':0.14,'glow':'#69b0ff','hist_share':None,
  'tagline':'军舰与关税先动',
  'path':['国内政治僵局加深','对外制裁或军事行动频次上升','外贸与供应链承压'],
  'challenge':['外贸订单消失','供应链重构'],'chance':['军工、能源安全'],'signal':['制裁名单','军事行动']},
]
D['domains']=[x for x in D['domains'] if x['id']!='uspolitics']
D['domains'].append({
 'id':'uspolitics','name':'美股·政策','short':'美股政策','sub':'民主党政策 · 政治周期 · AI 监管',
 'intro':'美股走势很大程度依赖政治。民主党主导下的政策偏好可从其选民基本盘推出：科技工会与消费者 → 反垄断与 AI 安全监管；盟友体系 → 多边管制代替单边关税；金融稳定 → 对大型科技的分拆压力。同时历史上总统在国内受困时往往对外转移（克林顿 1998 年模式），需同步观察军舰、关税、制裁的去向。',
 'temp':1.7,'axes':{'x':'外部环境 E','xp':'规则收紧 −','xn':'规则放松 +','y':'内部动能 I','yp':'强 +','yn':'弱 −'},
 'nodes':POLICY_NOTES,'scen':POLICY_SCEN,'paths':[],
 'edges':[['p_ai_ctrl','p_fin_stab',0.7],['p_ai_ctrl','p_capex',-0.5],['p_tariff','p_ally',0.4],
          ['p_distract','p_tariff',0.45],['p_chip_cycle','p_capex',0.55],['p_optics','p_ai_ctrl',0.6],
          ['p_fin_stab','p_capex',-0.45],['p_ally','p_optics',0.35]],
})
# 给新域（及任何缺 research_base 的域）补研报底座
try:
    R=json.load(open(OUT / 'research_index.json',encoding='utf-8'))
    AD=json.load(open(OUT / 'doc_analysis.json',encoding='utf-8'))
    grade={}
    for doc in AD['docs']: grade.setdefault(doc['title'], doc['grade'])
    KW={'uspolitics':['美联储','美股','纳指','标普','降息','加息','关税','美国','英伟达','估值','衰退']}
    for i,d in enumerate(D['domains']):
        if d.get('research_base'): continue
        kws=KW.get(d['id'])
        if not kws: continue
        hits=R['by_domain'].get(d['id'],[])
        if not hits:
            # 用全局文档按关键词现算
            hits=[]
            for doc in AD['docs']:
                sc=sum(1 for k in kws if k in doc['title'])
                if sc: hits.append({'title':doc['title'],'category':doc['category'],'date':doc['date'],
                                    'chars':doc['chars'],'grade':doc['grade'],'snippet':''})
        clean=[h for h in hits if grade.get(h['title'])=='A']
        gaps=[h for h in hits if grade.get(h['title'])!='A']
        for h in clean+gaps: h['grade']=grade.get(h['title'],'?')
        D['domains'][i]['research_base']={'top':clean[:6],'gaps':gaps[:6],'n_docs':len(hits)}
except Exception as e:
    print('补研报底座失败:', e)

json.dump(D, open(OUT / 'atlas_data.json','w'), ensure_ascii=False)
print('域数:', len(D['domains']))
for d in D['domains']:
    sc=len(d.get('supply_chain',[]))
    if sc: print(f"  {d['name']:10s} 供应链环节 {sc} 个")
