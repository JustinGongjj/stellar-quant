"""从模板 + atlas_data.json 生成 web/atlas.html。

模板（web/_atlas_template.html）保存原版全部逻辑与 UI 补丁；
本脚本只做一件事：把 DATA 常量替换为我们的数据。
"""
import json, pathlib, sys
from stellarq.paths import ROOT  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
TPL = ROOT/'web/_atlas_template.html'
DATA = ROOT/'reports/data/atlas_data.json'
OUT = ROOT/'web/atlas.html'
KEY = 'const DATA = window.DATA = '
KEYS = ('const DATA = window.DATA = ', 'const DATA = ')

def find_data_block(s):
    for key in KEYS:
        i = s.find(key)
        if i < 0:
            continue
        start = s.find('{', i)
        if start < 0:
            continue
        depth = 0; instr = False; esc = False
        for p in range(start, len(s)):
            ch = s[p]
            if instr:
                if esc: esc = False
                elif ch == '\\': esc = True
                elif ch == '"': instr = False
                continue
            if ch == '"': instr = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return key, i, p+1
    return None, None, None

REQUIRED_ASSETS = ['趋势推演工作台.html']   # 星图上的「后台工作台」链接指向它

def check_assets():
    missing = [a for a in REQUIRED_ASSETS if not (ROOT/'web'/a).exists()]
    if missing:
        print('警告：缺少被链接的静态页 ' + '、'.join(missing))
        print('      星图上的对应按钮会 404。从归档补：cp /tmp/orig/<name> web/')

def main():
    tpl = TPL.read_text(encoding='utf-8')
    D = json.loads(DATA.read_text(encoding='utf-8'))
    key, i, end = find_data_block(tpl)
    if i is None:
        sys.exit('模板里找不到 DATA')
    out = tpl[:i] + KEY + json.dumps(D, ensure_ascii=False) + ';' + tpl[end:]
    out = out.replace('势 · 数 ｜ 未来星图 · FUTURE ATLAS', '星衡量化 ｜ 未来星图 · FUTURE ATLAS')
    OUT.write_text(out, encoding='utf-8')
    print(f'已生成 {OUT}  {len(out)/1024:.0f} KB  域数 {len(D["domains"])}')
    check_assets()

if __name__ == '__main__':
    main()
