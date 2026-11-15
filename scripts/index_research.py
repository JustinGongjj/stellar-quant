"""把投研工作站的研报/研究结论正文建成主题索引，并挂到星图各域。

产出 reports/data/research_index.json：
  * topics —— 按关键词聚合的主题（每个主题：命中文档数、总字数、代表文档）
  * by_domain —— 星图域 → 命中的文档列表（标题/分类/日期/字数/摘要片段）
"""
import sys, json, re, collections; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.insight import InsightHub
from stellarq.paths import DATA, ROOT  # noqa: E402

OUT = DATA
h=InsightHub()
docs=h.research_docs()
print(f'读取 {len(docs)} 份文档，{sum(d["chars"] for d in docs)/1e4:.1f} 万字')

# 关键词 → 星图域
DOMAIN_KW = {
 'ai':        ['人工智能','大模型','AI','算力','芯片','推理','LLM','智能体','GPU','英伟达'],
 'embodied':  ['具身智能','人形机器人','机器人','灵巧手','执行器','VLA','丝杠','减速器'],
 'space':     ['商业航天','航天','火箭','卫星','星座','发射','可回收','电磁弹射','低轨'],
 'lowalt':    ['低空经济','eVTOL','飞行汽车','无人机','适航','低空'],
 'quant':     ['量化','因子','回测','多因子','选股','Alpha','择时'],
 'ouhai':     ['瓯海','温州','鞋','眼镜','锁具','服装'],
 'global':    ['全球','地缘','关税','中美','欧洲','制裁','出口管制','一带一路'],
 'us':        ['美股','美联储','纳斯达克','标普','降息','加息'],
}
TOPIC_KW = {
 '半导体国产化': ['半导体','芯片','国产化','先进制程','光刻','HBM','封测'],
 'AI 与算力':   ['人工智能','大模型','算力','推理','GPU','智算'],
 '具身智能':    ['具身智能','人形机器人','灵巧手','执行器','VLA'],
 '商业航天':    ['商业航天','火箭','卫星','星座','电磁弹射','可回收'],
 '低空经济':    ['低空经济','eVTOL','飞行汽车','无人机','适航'],
 '新能源与电池':['锂电','电池','固态','储能','光伏','新能源'],
 '宏观与策略':  ['宏观','GDP','CPI','社融','货币政策','财政','降息','加息'],
 '外贸与关税':  ['关税','出口','进出口','贸易','CBAM','301'],
 '医药健康':    ['医药','创新药','医疗器械','生物'],
 '投后与风控':  ['投后','预警','尽调','风险','条款'],
}

def score(doc, kws):
    hay = (doc['title']+' '+' '.join(doc['tags'] if isinstance(doc['tags'],list) else [])+' '+doc['body'][:6000])
    return sum(hay.count(k) for k in kws)

by_domain={}
for did,kws in DOMAIN_KW.items():
    hits=[]
    for d in docs:
        sc=score(d,kws)
        if sc>=3:
            hits.append({'title':d['title'],'category':d['category'],'date':d['date'],
                         'chars':d['chars'],'score':sc,'file':d['file'],
                         'snippet':re.sub(r'\s+',' ',d['body'])[:180]})
    hits.sort(key=lambda x:(-x['score'],-x['chars']))
    by_domain[did]=hits[:40]
    print(f'{did:10s} 命中 {len(hits):3d} 份')

topics={}
for name,kws in TOPIC_KW.items():
    hs=[(d,score(d,kws)) for d in docs]
    hs=[(d,s) for d,s in hs if s>=3]
    hs.sort(key=lambda x:-x[1])
    topics[name]={'count':len(hs),'chars':sum(d['chars'] for d,_ in hs),
                  'top':[{'title':d['title'],'date':d['date'],'chars':d['chars']} for d,_ in hs[:5]]}
    print(f'主题 {name:12s} {len(hs):3d} 份 / {sum(d["chars"] for d,_ in hs)/1e4:.1f} 万字')

out={'source':'insight-hub/vault','n_docs':len(docs),
     'total_chars':sum(d['chars'] for d in docs),
     'documents':[{'file':d['file'],'title':d['title'],'category':d['category'],
                   'date':d['date'],'chars':d['chars']} for d in docs],
     'by_domain':by_domain,'topics':topics}
json.dump(out, open(OUT / 'research_index.json','w'), ensure_ascii=False, indent=1)
print('saved research_index.json')
