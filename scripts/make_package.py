"""构建可在线下载的成果压缩包。

打包内容（不含 .git、缓存、超大中间文件）：
  代码 src/ scripts/ tests/
  报告 reports/*.md
  数据 reports/data/*.json + *.parquet
  页面 web/（含 atlas.html、工作台、开机动画）
  知识 knowledge/
  文档 README + 索引
"""
import os, shutil, json, pathlib, datetime, zipfile
from stellarq.paths import ROOT  # noqa: E402
ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG=ROOT/'reports'/'package'
STAGE=PKG/'.stage'
if STAGE.exists(): shutil.rmtree(STAGE)
if PKG.exists():
    for f in PKG.glob('*.zip'): f.unlink()
STAGE.mkdir(parents=True, exist_ok=True)
NAME=f'stellar-quant_{datetime.date.today().isoformat()}'

SKIP_DIRS={'.git','__pycache__','node_modules','.pytest_cache','package','.stage'}
SKIP_EXT={'.pyc','.pyo','.npy'}
MAX_MB=8

def copy_tree(src, dst):
    n=0; skipped=[]
    for dp, dns, fns in os.walk(src):
        dns[:]=[d for d in dns if d not in SKIP_DIRS]
        rel=os.path.relpath(dp, src)
        out=dst/rel if rel!='.' else dst
        out.mkdir(parents=True, exist_ok=True)
        for fn in fns:
            sp=pathlib.Path(dp)/fn
            if sp.suffix in SKIP_EXT: continue
            try:
                if sp.stat().st_size > MAX_MB*1024*1024:
                    skipped.append(str(sp.relative_to(src))); continue
            except OSError: continue
            shutil.copy2(sp, out/fn); n+=1
    return n, skipped

dst=STAGE/NAME
total=0; all_skip=[]
for sub in ['src','scripts','tests','reports','web','knowledge','docs']:
    s=ROOT/sub
    if not s.exists(): continue
    if sub=='reports':
        # 只取 md 与 data
        n1,k1=0,[]; 
        t=(dst/'reports'); t.mkdir(parents=True, exist_ok=True)
        for f in (s).glob('*.md'):
            shutil.copy2(f, t/f.name); n1+=1
        n2,k2=copy_tree(s/'data', t/'data')
        total+=n1+n2; all_skip+=k1+k2
    else:
        n,k=copy_tree(s, dst/sub)
        total+=n; all_skip+=k
# GitHub 规范文件
gh=ROOT/'.github'
if gh.exists():
    n,k=copy_tree(gh, dst/'.github'); total+=n; all_skip+=k

# 顶层文件
for f in ['README.md','pyproject.toml','LICENSE','CONTRIBUTING.md','CHANGELOG.md','.gitignore']:
    p=ROOT/f
    if p.exists(): shutil.copy2(p, dst/f); total+=1

# 总索引
idx=[]
for dp,dns,fns in os.walk(dst):
    for fn in fns:
        fp=pathlib.Path(dp)/fn
        idx.append({'path':str(fp.relative_to(dst)),'kb':round(fp.stat().st_size/1024,1)})
idx.sort(key=lambda x:x['path'])
(dst/'索引.json').write_text(json.dumps(idx,ensure_ascii=False,indent=1),encoding='utf-8')

readme=(dst/'开始阅读.md')
readme.write_text(f"""# 星衡量化 · 成果包

导出时间：{datetime.datetime.now():%Y-%m-%d %H:%M}
文件数：{total + 1}

## 先看这三份

1. `reports/项目总报告.md` —— 全部结论、六个被证伪的假设、可信度分级
2. `reports/数据获取报告` 与 `knowledge/macro/` —— 宏观数据来源与口径
3. `reports/库存周期行业信号.md` —— 唯一通过三道闸门的信号

## 目录

| 目录 | 内容 |
|---|---|
| `reports/*.md` | 全部研究报告（30+ 份） |
| `reports/data/*.json` | 每个结论背后的原始数据 |
| `src/stellarq/` | 量化框架源码 |
| `scripts/` | 全部跑批脚本（可复现） |
| `web/` | 星图、后台工作台、开机动画 |
| `knowledge/` | 宏观数据、产业笔记 |

## 复现

```bash
pip install -e .
python3 -m stellarq.metrics            # 口径自检
bash scripts/rebuild_all.sh            # 重建星图页面
```

## 重要声明

- 所有回测数字均含已知偏差说明，**请先读「项目总报告」的可信度分级**
- 面板数据存在幸存者偏差（123 只退市股缺失，超额高估 0.73–1.71pp）
- 产业框架不产生 alpha，仅用于认知与风险预警
- **不构成投资建议**
""",encoding='utf-8')

zp=PKG/f'{NAME}.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for dp,dns,fns in os.walk(STAGE):
        for fn in fns:
            fp=pathlib.Path(dp)/fn
            z.write(fp, fp.relative_to(STAGE))
shutil.rmtree(STAGE)
sz=zp.stat().st_size/1024/1024
print(f'已生成 {zp.name}  {sz:.1f} MB  {total+1} 个文件')
if all_skip: print('跳过超限文件:', all_skip[:5])
print('下载地址: /download/'+zp.name)
