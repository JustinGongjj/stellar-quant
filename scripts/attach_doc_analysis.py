"""把逐份分析结果挂到星图：每域只引用 A 级（可读）文档，并标注 B/C 级的缺口。"""
import json, collections
from stellarq.paths import DATA  # noqa: E402
OUT = DATA
D=json.load(open(OUT / 'atlas_data.json',encoding='utf-8'))
A=json.load(open(OUT / 'doc_analysis.json',encoding='utf-8'))
idx={d['file']:d for d in A['docs']}

# 质量提示
grade_by_title={}
for d in A['docs']:
    grade_by_title.setdefault(d['title'], d['grade'])

for i,d in enumerate(D['domains']):
    rb=d.get('research_base') or {}
    tops=rb.get('top',[])
    clean=[]; gaps=[]
    for t in tops:
        # 用标题匹配分级
        g=None
        for doc in A['docs']:
            if doc['title']==t['title']:
                g=doc['grade']; break
        t['grade']=g or '?'
        (clean if g=='A' else gaps).append(t)
    # 只用 A 级作为「可用证据」，B/C 进缺口
    rb['top']=clean[:6]
    rb['gaps']=gaps[:6]
    rb['quality']=dict(collections.Counter(t.get('grade','?') for t in rb.get('top',[])+rb.get('gaps',[])))
    d['research_base']=rb
    D['domains'][i]=d

# 全局质量摘要
D['research_quality']={
 'total_docs':A['total'],'by_grade':A['by_grade'],
 'note':'A=可读可作为证据；B=正文被截断（源 .md 只有约1.2万字）；C=PDF中文解析失败，仅剩数字与英文',
 'usage_rule':'星图的因子证据只引用 A 级文档；B/C 级列入「待补材料」'
}
json.dump(D, open(OUT / 'atlas_data.json','w'), ensure_ascii=False)
print('质量摘要:', A['by_grade'])
for d in D['domains']:
    rb=d.get('research_base',{})
    print(f"  {d['name']:8s} 可用 {len(rb.get('top',[]))} 份 / 待补 {len(rb.get('gaps',[]))} 份")
