"""全仓统一路径入口。

发布到 GitHub 后，任何人 clone 到任意目录都能跑：
所有路径都相对于仓库根目录，而不是写死的绝对路径。

用法：
    from stellarq.paths import DATA, REPORTS, CACHE, ROOT
    df.to_parquet(DATA / 'foo.parquet')
"""
from __future__ import annotations

import os
import pathlib

# 仓库根 = 本文件往上三级（src/stellarq/paths.py → src/stellarq → src → 仓库根）
ROOT = pathlib.Path(__file__).resolve().parents[2]

# 数据与产出
REPORTS = ROOT / 'reports'
DATA = REPORTS / 'data'
KNOWLEDGE = ROOT / 'knowledge'
WEB = ROOT / 'web'
PACKAGE = REPORTS / 'package'

# 行情缓存：体积大，不进仓库。可用环境变量 STELLARQ_CACHE 覆盖
CACHE = pathlib.Path(os.environ.get('STELLARQ_CACHE', pathlib.Path.home() / '.cache' / 'stellarq'))
WIDE = CACHE / 'wide2'

# 外部服务地址（可被环境变量覆盖，便于换部署）
CHANLUN_BASE = os.environ.get('CHANLUN_BASE', 'http://127.0.0.1:8931')
INSIGHT_DB = pathlib.Path(os.environ.get(
    'INSIGHT_DB', str(pathlib.Path.home() / 'default' / 'insight-hub' / 'vault.db')))


def ensure_dirs() -> None:
    """确保输出目录存在（数据目录可以不存在时先建好）。"""
    for d in (DATA, PACKAGE):
        d.mkdir(parents=True, exist_ok=True)


def require_cache() -> None:
    """需要行情缓存的地方调用它，缺失时给出明确指引而不是抛 KeyError。"""
    if not WIDE.exists():
        raise FileNotFoundError(
            f'行情缓存不存在：{WIDE}\n'
            '请先运行 python3 scripts/fetch_data.py 生成，'
            '或设置环境变量 STELLARQ_CACHE 指向已有缓存目录。')
