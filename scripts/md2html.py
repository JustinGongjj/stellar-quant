#!/usr/bin/env python3
"""Markdown → 自包含 HTML（图片内联 base64），用于策略月报交付。

支持：标题、段落（含软折行合并）、无序列表（含续行）、表格、引用块、
水平线、行内粗体/代码、图片（内联 base64）。
"""
import base64
import pathlib
import re
import sys

CSS = """
body{margin:0;background:#f4f5f6;font-family:"Noto Sans CJK SC",system-ui,sans-serif;color:#1c1c1c;line-height:1.78}
.page{max-width:900px;margin:0 auto;background:#fff;padding:52px 60px 72px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
h1{font-size:27px;margin:0 0 18px;padding-bottom:12px;border-bottom:3px solid #c0392b;letter-spacing:.5px}
h2{font-size:19px;margin:36px 0 12px;color:#b03a2e;border-left:4px solid #c0392b;padding-left:11px}
h3{font-size:16px;margin:24px 0 8px;color:#2c3e50}
p,li{font-size:14.2px}
li{margin:5px 0}
ol{padding-left:24px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.2px}
th,td{border:1px solid #e3e6e8;padding:8px 11px;text-align:left}
th{background:#fafbfc;font-weight:600}
tbody tr:nth-child(even){background:#fcfdfd}
figure{margin:24px 0}
figure img{width:100%;border:1px solid #e8eaec;border-radius:4px;display:block}
code{background:#f2f4f5;padding:1px 5px;border-radius:3px;font-size:12.6px}
hr{border:0;border-top:1px solid #e8eaec;margin:28px 0}
strong{color:#a93226}
blockquote{margin:16px 0;padding:10px 14px;background:#fdf6f5;border-left:3px solid #c0392b;font-size:13.4px;color:#4a3b39}
blockquote p{margin:0;font-size:13.4px}
@media print{body{background:#fff}.page{box-shadow:none;padding:0}}
"""

BLOCK_START = re.compile(r"^\s*(#{1,4}\s|\||>\s?|[-*]\s+|\d+[.)]\s+|<figure>)")


def inline(t: str) -> str:
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def normalise(md: str) -> list[str]:
    """把软折行合并进上一行（段落、列表项、引用块同一规则）。"""
    out: list[str] = []
    for raw in md.split("\n"):
        ln = raw.rstrip()
        if not ln.strip():
            out.append("")
            continue
        if BLOCK_START.match(ln) or not out or not out[-1].strip():
            out.append(ln)
        else:
            out[-1] = out[-1].rstrip() + ln.lstrip()
    return out


def convert(md_path: str) -> pathlib.Path:
    p = pathlib.Path(md_path)
    base = p.parent
    md = p.read_text(encoding="utf-8")

    def img(m):
        alt, rel = m.group(1), m.group(2)
        data = base64.b64encode((base / rel).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{alt}"></figure>'

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", img, md)

    out: list[str] = []
    state = {"kind": None}
    buf: list[str] = []
    intable = False

    def flush():
        if not buf:
            state["kind"] = None
            return
        k = state["kind"]
        if k == "quote":
            out.append("<blockquote><p>" + inline(" ".join(buf)) + "</p></blockquote>")
        elif k == "ul":
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ul>")
        elif k == "ol":
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ol>")
        else:
            out.append("<p>" + inline(" ".join(buf)) + "</p>")
        buf.clear()
        state["kind"] = None

    for ln in normalise(md):
        s = ln.strip()
        if intable and not s.startswith("|"):
            out.append("</tbody></table>")
            intable = False
        if not s:
            flush()
            continue
        if s.startswith("<figure>"):
            flush()
            out.append(s)
            continue
        if s.startswith("|"):
            flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells and set("".join(cells)) <= set("-: "):
                continue
            if not intable:
                out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                intable = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            flush()
            lv = len(m.group(1))
            out.append(f"<h{lv}>{inline(m.group(2))}</h{lv}>")
            continue
        if re.match(r"^-{3,}$", s):
            flush()
            out.append("<hr>")
            continue
        if s.startswith(">"):
            if state["kind"] != "quote":
                flush()
                state["kind"] = "quote"
            buf.append(s.lstrip(">").strip())
            continue
        if re.match(r"^[-*]\s+", s):
            if state["kind"] != "ul":
                flush()
                state["kind"] = "ul"
            buf.append(re.sub(r"^[-*]\s+", "", s))
            continue
        if re.match(r"^\d+[.)]\s+", s):
            if state["kind"] != "ol":
                flush()
                state["kind"] = "ol"
            buf.append(re.sub(r"^\d+[.)]\s+", "", s))
            continue
        if state["kind"] != "p":
            flush()
            state["kind"] = "p"
        buf.append(s)
    flush()
    if intable:
        out.append("</tbody></table>")

    doc = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{p.stem}</title><style>{CSS}</style></head><body><div class=\"page\">\n"
        + "\n".join(out)
        + "\n</div></body></html>"
    )
    op = p.with_suffix(".html")
    op.write_text(doc, encoding="utf-8")
    return op


if __name__ == "__main__":
    print("wrote", convert(sys.argv[1]))
