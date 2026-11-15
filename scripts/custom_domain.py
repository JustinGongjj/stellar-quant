"""自定义推演对象生成器：把「检索 → 分析 → 存储 → 上线」固化成可复用流程。

用法：
  python3 scripts/custom_domain.py --id lowalt --name 低空经济 \\
      --src reports/research/lowalt.json

研究文件（--src）由检索阶段产出，结构见下方 RESEARCH_SCHEMA；
本脚本负责：校验 → 套用标准 schema → 写入 domains/<id>.json（持久存储）
→ 合并进 atlas_data.json → 重建 atlas.html。
"""
import argparse, json, pathlib, re, sys, datetime
from stellarq.paths import ROOT, WEB  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT/'reports/data'
STORE = DATA/'domains'
STORE.mkdir(parents=True, exist_ok=True)

VALID_CAT = {'economy','industry','geopol','institution','human','structure'}
QUADRANTS = {(1,1),(1,-1),(-1,1),(-1,-1)}


def validate(d):
    errs=[]
    if d.get('id') in (None,''): errs.append('缺少 id')
    if not d.get('nodes'): errs.append('nodes 为空')
    if len(d.get('scen',[]))!=4: errs.append('scen 必须恰好 4 条')
    ids={n['id'] for n in d.get('nodes',[])}
    for n in d.get('nodes',[]):
        if n.get('cat') not in VALID_CAT: errs.append(f"节点 {n.get('id')} 的 cat 非法：{n.get('cat')}")
        for k in ('id','name','li','le','desc'):
            if k not in n: errs.append(f"节点 {n.get('id','?')} 缺 {k}")
    for s in d.get('scen',[]):
        if (int(s['I']),int(s['E'])) not in QUADRANTS: errs.append(f"情景 {s['id']} 的 (I,E) 不在四象限")
    for e in d.get('edges',[]):
        if e[0] not in ids or e[1] not in ids: errs.append(f"边 {e[:2]} 端点不存在")
    return errs


def merge_into_atlas(domain):
    ap = DATA/'atlas_data.json'
    D = json.loads(ap.read_text(encoding='utf-8'))
    D['domains'] = [d for d in D['domains'] if d['id'] != domain['id']] + [domain]
    ap.write_text(json.dumps(D, ensure_ascii=False), encoding='utf-8')
    return D


def rebuild_html():
    """把 atlas_data.json 注回原版 FUTURE ATLAS 模板。"""
    tpl = WEB / '_atlas_template.html'
    if not tpl.exists():
        raise SystemExit(f'缺少模板 {__import__("pathlib").Path(__file__).resolve().parents[1] / "web/_atlas_template.html"}')
    s = tpl.read_text(encoding='utf-8')
    D = json.loads((DATA/'atlas_data.json').read_text(encoding='utf-8'))
    i = s.find('const DATA = '); start = s.find('{', i)
    depth=0; instr=False; esc=False
    for p in range(start, len(s)):
        ch = s[p]
        if instr:
            if esc: esc=False
            elif ch=='\\': esc=True
            elif ch=='"': instr=False
            continue
        if ch=='"': instr=True
        elif ch=='{': depth+=1
        elif ch=='}':
            depth-=1
            if depth==0: end=p+1; break
    out = s[:i] + 'const DATA = ' + json.dumps(D, ensure_ascii=False) + ';' + s[end:]
    out = out.replace('势 · 数 ｜ 未来星图 · FUTURE ATLAS', '星衡量化 ｜ 未来星图 · FUTURE ATLAS')
    (ROOT/'web/atlas.html').write_text(out, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--src', required=True, help='检索阶段产出的研究文件')
    a = ap.parse_args()

    research = json.loads(pathlib.Path(a.src).read_text(encoding='utf-8'))
    domain = research['domain']
    domain['id'] = a.id
    domain['name'] = a.name
    domain.setdefault('short', a.name)
    domain.setdefault('temp', 1.6)
    domain.setdefault('axes', {'x':'外部环境 E','xp':'开放 +','xn':'割裂 −',
                               'y':'内部动能 I','yp':'强 +','yn':'弱 −'})
    domain.setdefault('paths', [])
    domain['generated'] = datetime.date.today().isoformat()
    domain['provenance'] = research.get('provenance', '检索分析')

    errs = validate(domain)
    if errs:
        print('校验失败：'); [print('  -', e) for e in errs]; sys.exit(1)

    # 1) 持久存储
    sp = STORE/f"{a.id}.json"
    sp.write_text(json.dumps(domain, ensure_ascii=False, indent=1), encoding='utf-8')
    print('已存储 →', sp)

    # 2) 合并进 atlas_data
    D = merge_into_atlas(domain)
    print('已合并，当前域：', [d['name'] for d in D['domains']])

    # 3) 重建页面
    rebuild_html()
    print('已重建 web/atlas.html')


if __name__ == '__main__':
    main()
