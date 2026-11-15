"""把投研工作站的知识库与产业图谱挂载到星图域上。

对每个星图域：
  * 找同名/近名的产业图谱（industry_maps）或产业链图（chainmaps）
  * 把上游/中游/下游结构、卡脖子点、技术瓶颈作为 reference 挂到域上
  * 把命中的知识库条目作为 citations 挂上
星图页面会在侧栏显示「知识底座」区块，点击可展开这些内容。
"""
import sys, json; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.insight import InsightHub
from stellarq.paths import DATA, ROOT  # noqa: E402

OUT = DATA
D=json.load(open(OUT / 'atlas_data.json',encoding='utf-8'))
h=InsightHub()
maps={m['track']:m for m in h.industry_maps()}
chains={c['industry']:c for c in h.chainmaps()}
IDX=h.track_index()

def pick(keys):
    for k in keys:
        if k in maps: return maps[k]
        for t,m in maps.items():
            if k in t or t in k: return m
    return None
def pick_chain(keys):
    for k in keys:
        if k in chains: return chains[k]
        for t,c in chains.items():
            if k in t or t in k: return c
    return None

attached={}
for i,d in enumerate(D['domains']):
    keys = IDX.get(d['id'], [d['name']])
    im = pick(keys); cm = pick_chain(keys)
    if not im and not cm:
        D['domains'][i]['knowledge_base']={'linked':False,'reason':'投研工作站暂无该赛道的图谱'}
        continue
    ref={}
    if im:
        p=im['payload']
        ref['track']=im['track']; ref['version']=im['version']; ref['updated']=im['updated']
        ref['positioning']=p.get('positioning','')
        ref['layers']=[{'name':l.get('name'),
                        'nodes':[{'name':n.get('name'),'choke':n.get('choke'),'desc':n.get('desc'),
                                  'players':n.get('players',[])} for n in l.get('nodes',[])]}
                       for l in p.get('layers',[])]
        ref['chokepoints']=p.get('chokepoints',[])
        ref['tech_bottlenecks']=p.get('tech_bottlenecks',[])
        ref['sources']=p.get('sources',[])[:5] if isinstance(p.get('sources'),list) else []
    if cm:
        ref['chainmap']={'industry':cm['industry'],'title':cm['title'],'version':cm['version'],
                         'updated':cm['updated'],'note':cm.get('note','')}
    D['domains'][i]['knowledge_base']={'linked':True,'reference':ref}
    attached[d['name']]=(im['track'] if im else None, cm['title'] if cm else None)

# 全局知识库索引
D['insight_links']={
  'stats':h.stats(),
  'kb_recent':[{'id':e['id'],'title':e['title'][:80],'category':e['category'],
                'created_at':e['created_at']} for e in h.kb_entries(12)],
  'frontier':[{'track':f['track'],'title':f['title'][:90],'published':f['published']}
              for f in h.frontier(10)],
}
json.dump(D, open(OUT / 'atlas_data.json','w'), ensure_ascii=False)
print('已挂载:', attached)
print('未挂载:', [d['name'] for d in D['domains'] if not d.get('knowledge_base',{}).get('linked')])
