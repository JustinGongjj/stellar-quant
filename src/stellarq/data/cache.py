"""本地数据缓存：按「数据集 / 键」落盘的简单 CSV + 元数据 JSON。

刻意保持极小：不引入数据库依赖，任何一步都能用文件系统检查、手工替换。
缓存目录默认 ``~/.cache/stellarq``，可用环境变量 ``STELLARQ_CACHE`` 覆盖。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd


def cache_root() -> Path:
    root = Path(os.environ.get("STELLARQ_CACHE", Path.home() / ".cache" / "stellarq"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def path_for(dataset: str, name: str, ext: str = "csv") -> Path:
    folder = cache_root() / dataset
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{_key(name)}.{ext}"


def get_frame(dataset: str, name: str, max_age_s: float | None = None) -> pd.DataFrame | None:
    """读取缓存 DataFrame；超过 ``max_age_s`` 视为失效。"""
    p = path_for(dataset, name)
    if not p.exists():
        return None
    if max_age_s is not None and time.time() - p.stat().st_mtime > max_age_s:
        return None
    meta = p.with_suffix(".meta.json")
    if meta.exists():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info = {}
        if info.get("name") != name:
            return None
    return pd.read_csv(p)


def put_frame(dataset: str, name: str, frame: pd.DataFrame, extra: dict | None = None) -> Path:
    p = path_for(dataset, name)
    tmp = p.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(p)
    info = {"name": name, "rows": int(len(frame)), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if extra:
        info.update(extra)
    p.with_suffix(".meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Parquet 缓存（面板数据量在 GB 级，CSV 读取会触发内存尖峰）
# ---------------------------------------------------------------------------
def get_parquet(dataset: str, name: str, max_age_s: float | None = None):
    """读取 Parquet 缓存；不存在或过期返回 None。"""
    import pyarrow.parquet as pq

    p = path_for(dataset, name, "parquet")
    if not p.exists():
        return None
    if max_age_s is not None and time.time() - p.stat().st_mtime > max_age_s:
        return None
    return pq.read_table(p).to_pandas()


def put_parquet(dataset: str, name: str, frame) -> Path:
    p = path_for(dataset, name, "parquet")
    tmp = p.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(p)
    p.with_suffix(".meta.json").write_text(
        json.dumps({"name": name, "rows": int(len(frame)),
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False),
        encoding="utf-8")
    return p


def list_cache() -> list[dict]:
    out = []
    for meta in sorted(cache_root().rglob("*.meta.json")):
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        info["dataset"] = meta.parent.name
        info["file"] = str(meta.with_suffix("").with_suffix(".csv"))
        out.append(info)
    return out
