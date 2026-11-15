"""逐份分析研究文档：质量分级 + 结构化抽取（主题、实体、数字、结论句）。

产出 reports/data/doc_analysis.json，每条文档一条记录。
分级：
  A 可读（中文占比>0.3，或英文正文但结构完整）
  B 截断（含"笔记正文已截断"标记）
  C 解析失败（乱码密度高、几乎无自然语言）
"""
import sys, json, re, hashlib, collections; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.insight import InsightHub
from stellarq.paths import DATA, ROOT  # noqa: E402

OUT = DATA
h=InsightHub(); docs=h.research_docs()
facts=h.notes_index()

ENTITY_PAT = re.compile(r'[A-Z][A-Za-z&]{2,}(?:\s+[A-Z][A-Za-z&]{2,})*')
COMPANY_SUFFIX = ('公司','集团','科技','控股','航天','电子','半导体','能源','医药','汽车','电池','智能','网络','数据')

def cjk_ratio(b):
    if not b: return 0.0
    return len(re.findall(r'[\u4e00-\u9fff]', b))/len(b)

def junk_ratio(b):
    if not b: return 1.0
    # 非字母数字非中文非常见标点的比例
    ok = len(re.findall(r'[\u4e00-\u9fffA-Za-z0-9\s，。、；：（）“”‘’—\-·%\.\,\:\;\(\)]', b))
    return 1-ok/len(b)

def grade(d):
    b=d['body']
    if '笔记正文已截断' in b: return 'B'
    r=cjk_ratio(b); j=junk_ratio(b)
    if j>0.35: return 'C'
    if r>0.30: return 'A'
    if r<0.05:
        # 纯英文：看英文单词密度
        words=len(re.findall(r'\b[A-Za-z]{4,}\b', b))
        return 'A' if words>len(b)/40 else 'C'
    return 'A'

def extract(d):
    b=d['body']
    txt=re.sub(r'\s+',' ',b)
    # 数字与百分比
    nums=re.findall(r'\d+(?:\.\d+)?%', txt)[:20]
    # 金额
    amts=re.findall(r'(?:\$|¥|US\$)\s?\d+(?:\.\d+)?\s?(?:billion|million|bn|mn|亿|万亿|万)?', txt)[:10]
    # 英文公司名候选
    comps=collections.Counter(ENTITY_PAT.findall(txt[:8000]))
    comps=[c for c,_ in comps.most_common(30) if len(c)>3][:12]
    # 中文公司/机构名（粗粒度：结尾词）
    cn=re.findall(r'[\u4e00-\u9fff]{2,8}(?:公司|集团|科技|控股|航天|电子|半导体|能源|医药|汽车|电池)', txt)[:12]
    # 结论句：含判断词的句子
    sents=re.split(r'[。！？\n]', txt)
    concl=[s.strip() for s in sents if re.search(r'(预计|有望|将会|建议|判断|认为|目标|预计到|趋势|风险)', s) and 15<len(s)<120][:6]
    return {'numbers':nums,'amounts':amts,'companies_en':comps,'companies_cn':list(dict.fromkeys(cn)),
            'conclusions':concl}

out=[]
for d in docs:
    g=grade(d)
    rec={'file':d['file'],'title':d['title'],'category':d['category'],'date':d['date'],
         'chars':d['chars'],'grade':g,'cjk_ratio':round(cjk_ratio(d['body']),3),
         'junk_ratio':round(junk_ratio(d['body']),3),
         'truncated':'笔记正文已截断' in d['body'],
         'hash':hashlib.md5(d['body'].strip().encode()).hexdigest()[:12]}
    if g in ('A','B'):
        rec.update(extract(d))
    out.append(rec)

json.dump({'total':len(out),'by_grade':dict(collections.Counter(r['grade'] for r in out)),
           'docs':out}, open(OUT / 'doc_analysis.json','w'), ensure_ascii=False, indent=1)
c=collections.Counter(r['grade'] for r in out)
print('分级:', dict(c), f'共 {len(out)} 份')
for g in 'ABC':
    ex=[r for r in out if r['grade']==g][:4]
    print(f'--- {g} 例:')
    for r in ex: print(f'   {r["chars"]:6d} {r["title"][:44]:46s} cjk={r["cjk_ratio"]:.2f} junk={r["junk_ratio"]:.2f}')
