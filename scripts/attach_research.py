"""把研报索引挂到星图域：每个因子可追溯支撑它的研报，域级别显示研报底座规模。"""
import json
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
D=json.load(open(OUT / 'atlas_data.json',encoding='utf-8'))
R=json.load(open(OUT / 'research_index.json',encoding='utf-8'))

# 域 → 关键词（用于给每个节点找证据）
NODE_KW={
 'space':{'电磁':'电磁弹射|电磁发射|磁悬浮','复用':'可复用|回收|复用','星座':'星座|卫星|频轨',
          '成本':'成本|\\$/kg|运力','政策':'政策|规划|十四五|十五五','瓶颈':'瓶颈|过载|分离',
          '叙事':'融资|估值|叙事'},
 'ai':{'capex':'资本开支|capex|算力投资','推理':'推理|inference','B端':'企业|ARR|付费',
       'ROI':'ROI|回报|效益','C端':'订阅|用户|C端','制程':'制程|HBM|出口管制|管制',
       '电力':'电力|能耗|散热|数据中心','估值':'估值|泡沫|拥挤'},
 'embodied':{'量产':'量产|产能|出货|Optimus','供应链':'供应链|Tier1|执行器|丝杠',
             '路线':'VLA|WAM|分层|端到端','数据':'数据|仿真|遥操','成本':'成本|BOM|售价',
             '资本':'融资|订单|产能'},
}
docs=R['documents']
bodies={}
for did,hits in R['by_domain'].items():
    D_=D
for i,d in enumerate(D['domains']):
    did=d['id']; hits=R['by_domain'].get(did,[])
    d['research_base']={'n_docs':len(hits),
                        'top':[{'title':h['title'],'category':h['category'],'date':h['date'],
                                'chars':h['chars'],'snippet':h['snippet']} for h in hits[:6]]}
    # 节点级证据：按节点名关键词在命中文档里再筛
    kws=NODE_KW.get(did,{})
    if kws and hits:
        for n in d['nodes']:
            import re
            pat=None
            for k,v in kws.items():
                if k in n['name']: pat=v; break
            if not pat: continue
            rx=re.compile(pat)
            ev=[h for h in hits if rx.search(h['title']+h['snippet'])][:3]
            if ev:
                n['research']=[{'title':e['title'],'date':e['date'],'chars':e['chars']} for e in ev]
    D['domains'][i]=d
json.dump(D, open(OUT / 'atlas_data.json','w'), ensure_ascii=False)
print('已挂研报底座:', [(d['name'], d['research_base']['n_docs']) for d in D['domains'] if d.get('research_base')])
