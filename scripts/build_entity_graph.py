"""从研报/笔记中抽取「公司—环节—关系」，建成可复用的实体图。

产出 reports/data/entity_graph.json：
  companies: [{name, 环节, segment, role, evidence:[文档], related:[公司]}]
  segments:  [{id, name, 上游/中游/下游, companies, chokepoint}]
"""
import sys, json, re, collections; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.insight import InsightHub

OUT = DATA
h=InsightHub()
docs=h.research_docs()
# 本地知识笔记也算来源
import pathlib
from stellarq.paths import DATA, KNOWLEDGE, ROOT  # noqa: E402
for f in KNOWLEDGE.glob('*.md'):
    t=f.read_text(encoding='utf-8')
    docs.append({'title':f.stem,'category':'本地笔记','body':t,'chars':len(t),'date':'2026-11-14','file':str(f)})

SEGMENTS = {
 '光模块/光芯片': {'layer':'中游','kw':['光模块','光芯片','EML','硅光','CW光源','DSP','磷化铟','800G','1.6T','3.2T'],
   'companies':['中际旭创','新易盛','天孚通信','Lumentum','博通','Marvell','云南锗业','源杰科技']},
 '液冷散热': {'layer':'中游','kw':['液冷','浸没','氟化液','硅油','CDU','冷板','散热','PUE'],
   'companies':['润禾材料','英维克','飞荣达','超频三','思泉新材','LG']},
 '电源模块': {'layer':'中游','kw':['电源','垂直供电','MLCC','Vicor','337调查','供电'],
   'companies':['麦格米特','蓝思微','华润微','浪潮','工业富联','立讯精密','台达','英飞凌','MPS','伟创力','欧陆通']},
 'PCB与封装': {'layer':'中游','kw':['PCB','基板','载板','封装','CoWoS','先进封装'],
   'companies':['沪电股份','深南电路','鹏鼎控股','长电科技','通富微电','华天科技','台积电','中芯国际','华虹']},
 '存储/内存': {'layer':'中游','kw':['内存','DRAM','HBM','长协','美光','内存池化','NAND'],
   'companies':['美光','三星','SK海力士','长鑫存储','长江存储','兆易创新']},
 '算力芯片': {'layer':'中游','kw':['GPU','算力','英伟达','芯片','制程','HBM'],
   'companies':['英伟达','AMD','华为','寒武纪','海光']},
 '具身智能零部件': {'layer':'中游','kw':['机器人','具身','执行器','减速器','丝杠','灵巧手','Optimus'],
   'companies':['拓普集团','三花智控','均胜电子','绿的谐波','双环传动','汇川技术']},
 '商业航天': {'layer':'中游','kw':['商业航天','火箭','卫星','星座','可回收','电磁弹射','发射'],
   'companies':['蓝箭航天','星河动力','天兵科技','星际荣耀','快舟航天','上海沪工','中国卫星','博云新材','邵阳液压','中天火箭','昊志机电','中兵红箭','长江通信','SpaceX']},
 '水下机器人': {'layer':'中游','kw':['水下机器人','ROV','AUV','潜行','Blueye','深海','水下'],
   'companies':['潜行创新','Blueye','中科深海所']},
 'AI应用': {'layer':'下游','kw':['大模型','Agent','应用','ARR','SaaS'],
   'companies':['Anthropic','OpenAI','微软','谷歌']},
}

graph={'segments':[], 'companies':{}, 'generated':'2026-11-14'}
name_hits=collections.defaultdict(list)
seg_of={}
for seg,cfg in SEGMENTS.items():
    entry={'id':seg,'layer':cfg['layer'],'companies':[],'docs':0,'chokepoint':None}
    for d in docs:
        hay=d['title']+' '+d['body']
        score=sum(hay.count(k) for k in cfg['kw'])
        if score>=3: entry['docs']+=1
        for co in cfg['companies']:
            if co in hay:
                name_hits[co].append({'doc':d['title'],'seg':seg,'score':score})
                seg_of.setdefault(co, collections.Counter())[seg]+=1
    graph['segments'].append(entry)

for co, hits in name_hits.items():
    seg=seg_of[co].most_common(1)[0][0]
    byc=collections.Counter(h['doc'] for h in hits)
    graph['companies'][co]={'segment':seg,'mentions':len(hits),
        'evidence':[{'title':t,'n':n} for t,n in byc.most_common(4)]}

# 卡脖子环节：光模块(DSP/EML)、存储(HBM)、先进制程
for s in graph['segments']:
    if s['id']=='光模块/光芯片': s['chokepoint']='DSP 芯片与电转光芯片（依赖 5nm）'
    if s['id']=='存储/内存': s['chokepoint']='HBM 与 DRAM 长协锁定，中小客户被挤出'
    if s['id']=='算力芯片': s['chokepoint']='先进制程与 HBM 出口管制'
    if s['id']=='液冷散热': s['chokepoint']='氟化液受 PFAS 限制，硅油导热系数低'

json.dump(graph, open(OUT / 'entity_graph.json','w'), ensure_ascii=False, indent=1)
print('环节:', len(graph['segments']), '公司:', len(graph['companies']))
print()
for s in graph['segments']:
    cos=[c for c,v in graph['companies'].items() if v['segment']==s['id']]
    print(f"  {s['id']:12s} {s['layer']} 文档{s['docs']:3d} 公司{len(cos):2d} 卡点:{s['chokepoint'] or '—'}")
print()
print('提及最多:', collections.Counter({k:v['mentions'] for k,v in graph['companies'].items()}).most_common(10))
