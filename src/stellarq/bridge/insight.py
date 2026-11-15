"""投研工作站（Insight Hub）知识库与产业图谱的只读适配器。

数据源
------
投研工作站把知识库、产业链思维导图、产业图谱都存在同一个 SQLite 里：
  ``insight-hub/data/insight.db``
  * ``kb``            知识库条目（title/category/tags/summary/source）
  * ``chainmaps``     产业链思维导图（industry → data[JSON]）
  * ``industry_maps`` 产业图谱（track → payload[JSON]：上游/中游/下游 · 卡脖子 · 技术瓶颈）
  * ``entities`` / ``entity_links`` 实体与关联
  * ``frontier_news`` 前沿情报

约束
----
* **只读打开**（``mode=ro``），绝不写投研工作站的数据；
* 数据库可能正被工作站写入，用 ``immutable=0`` + 超时避免锁冲突；
* 若库不存在，返回空结果而不是抛错——星图要能在没有投研工作站时独立运行。
"""
from __future__ import annotations

import json
import sqlite3
import pathlib
from dataclasses import dataclass

# v2：投研工作站已升级到 v2.64.0，实际在用的库在下面的路径。
# 原来的 insight-hub/data/insight.db 是旧版遗留，早已停止更新
# （实测旧库 25MB / kb 22 条，当前库 172MB / kb 28 条且持续增长）。
# 这里改成按优先级探测：先用当前版本路径，找不到再回退旧路径。
# 可用环境变量 INSIGHT_DB 直接指定；否则按常见部署位置探测。
import os as _os
_home = pathlib.Path.home()
_DB_CANDIDATES = [
    pathlib.Path(_os.environ['INSIGHT_DB']) if _os.environ.get('INSIGHT_DB') else None,
    _home / 'default/insight-hub-v2.64.0/insight-hub/app/data/insight.db',
    _home / 'default/insight-hub/data/insight.db',
]
_DB_CANDIDATES = [p for p in _DB_CANDIDATES if p is not None]


def _pick_db() -> pathlib.Path:
    """挑一个真实可用的库：存在且 kb 表有数据。"""
    import sqlite3 as _sq
    for p in _DB_CANDIDATES:
        if not p.exists():
            continue
        try:
            c = _sq.connect(f'file:{p}?mode=ro', uri=True, timeout=5)
            n = c.execute('SELECT COUNT(*) FROM kb').fetchone()[0]
            c.close()
            if n > 0:
                return p
        except Exception:
            continue
    return _DB_CANDIDATES[0]


DEFAULT_DB = _pick_db()


@dataclass
class InsightHub:
    db_path: pathlib.Path = DEFAULT_DB
    timeout: float = 5.0

    @property
    def available(self) -> bool:
        return self.db_path.exists()

    def _conn(self):
        if not self.available:
            raise FileNotFoundError(f'投研工作站数据库不存在：{self.db_path}')
        uri = f'file:{self.db_path}?mode=ro'
        c = sqlite3.connect(uri, uri=True, timeout=self.timeout)
        c.row_factory = sqlite3.Row
        return c

    # ---------- 知识库 ----------
    def kb_entries(self, limit: int = 500) -> list[dict]:
        if not self.available:
            return []
        try:
            with self._conn() as c:
                rows = c.execute(
                    'SELECT id,title,category,tags,summary,source,created_at '
                    'FROM kb ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def kb_search(self, keyword: str, limit: int = 40) -> list[dict]:
        """在 title/summary/tags 里做子串匹配（避免依赖 FTS 扩展是否可用）。"""
        if not self.available:
            return []
        kw = f'%{keyword}%'
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT id,title,category,tags,summary,source,created_at FROM kb "
                    "WHERE title LIKE ? OR summary LIKE ? OR tags LIKE ? "
                    "ORDER BY created_at DESC LIMIT ?", (kw, kw, kw, limit)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    # ---------- 产业图谱 ----------
    def industry_maps(self) -> list[dict]:
        if not self.available:
            return []
        try:
            with self._conn() as c:
                rows = c.execute('SELECT track,payload,version,updated FROM industry_maps').fetchall()
            out = []
            for r in rows:
                try:
                    p = json.loads(r['payload'])
                except (TypeError, json.JSONDecodeError):
                    continue
                out.append({'track': r['track'], 'version': r['version'],
                            'updated': r['updated'], 'payload': p})
            return out
        except sqlite3.Error:
            return []

    def industry_map(self, track: str) -> dict | None:
        for m in self.industry_maps():
            if m['track'] == track:
                return m
        return None

    def chainmaps(self) -> list[dict]:
        if not self.available:
            return []
        try:
            with self._conn() as c:
                rows = c.execute('SELECT id,industry,title,data,version,source,note,updated FROM chainmaps').fetchall()
            out = []
            for r in rows:
                try:
                    data = json.loads(r['data'])
                except (TypeError, json.JSONDecodeError):
                    data = None
                out.append({'id': r['id'], 'industry': r['industry'], 'title': r['title'],
                            'version': r['version'], 'source': r['source'],
                            'note': r['note'], 'updated': r['updated'], 'data': data})
            return out
        except sqlite3.Error:
            return []

    def frontier(self, limit: int = 60) -> list[dict]:
        if not self.available:
            return []
        try:
            with self._conn() as c:
                rows = c.execute(
                    'SELECT id,track,title,summary,kind,source,url,published FROM frontier_news '
                    'ORDER BY created DESC LIMIT ?', (limit,)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def entities(self, sector: str | None = None, limit: int = 200) -> list[dict]:
        if not self.available:
            return []
        try:
            with self._conn() as c:
                if sector:
                    rows = c.execute('SELECT id,name,kind,sector,aliases,status FROM entities '
                                     'WHERE sector LIKE ? LIMIT ?', (f'%{sector}%', limit)).fetchall()
                else:
                    rows = c.execute('SELECT id,name,kind,sector,aliases,status FROM entities LIMIT ?',
                                     (limit,)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def stats(self) -> dict:
        if not self.available:
            return {'available': False, 'db': str(self.db_path)}
        out = {'available': True, 'db': str(self.db_path)}
        try:
            with self._conn() as c:
                for t in ('kb', 'chainmaps', 'industry_maps', 'entities', 'entity_links', 'frontier_news', 'notes'):
                    try:
                        out[t] = c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                    except sqlite3.Error:
                        out[t] = None
        except sqlite3.Error as exc:
            out['error'] = str(exc)
        return out


    # ---------- 研报 / 研究结论 正文（vault） ----------
    VAULT = pathlib.Path(_os.environ.get('INSIGHT_VAULT', str(_home / 'default/insight-hub/vault')))
    RESEARCH_CATS = ('研报', '研究结论', '研究输入', '产业链图谱', '产业图谱', '投资知识库', '产业月报', '产业周报')

    def vault_files(self, cats=None) -> list[pathlib.Path]:
        """列出知识库各分类下的 .md 正文文件。"""
        cats = cats or self.RESEARCH_CATS
        out = []
        for c in cats:
            d = self.VAULT / c
            if d.is_dir():
                out.extend(sorted(d.glob('*.md')))
        return out

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict, str]:
        """拆分 YAML frontmatter（只取简单 key: value / 列表），返回 (meta, body)。"""
        if not text.startswith('---'):
            return {}, text
        end = text.find('\n---', 3)
        if end < 0:
            return {}, text
        head = text[3:end].strip()
        body = text[end + 4:].lstrip('\n')
        meta = {}
        for line in head.splitlines():
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip(); v = v.strip()
            if v.startswith('[') and v.endswith(']'):
                v = [x.strip().strip('"') for x in v[1:-1].split(',') if x.strip()]
            else:
                v = v.strip('"').strip("'")
            meta[k] = v
        return meta, body

    def research_docs(self, limit: int | None = None) -> list[dict]:
        """读取全部研究文档正文（含标题、分类、标签、来源、字数）。"""
        idx = {r['file']: r for r in self.notes_index()}
        docs = []
        for f in self.vault_files():
            try:
                text = f.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            meta, body = self._split_frontmatter(text)
            rel = f'{f.parent.name}/{f.name}'
            row = idx.get(rel) or idx.get(f.name) or {}
            docs.append({
                'file': rel,
                'title': meta.get('title') or row.get('title') or f.stem,
                'category': f.parent.name,
                'tags': meta.get('tags') or [],
                'source': meta.get('source') or row.get('source') or '',
                'date': meta.get('date') or row.get('date') or '',
                'chars': len(body),
                'body': body,
            })
        docs.sort(key=lambda d: d['date'], reverse=True)
        return docs[:limit] if limit else docs

    def notes_index(self) -> list[dict]:
        """notes 表的索引（title/category/file/date），不读正文。"""
        if not self.available:
            return []
        try:
            with self._conn() as c:
                rows = c.execute('SELECT id,file,title,category,tags,source,date FROM notes').fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    # ---------- 域名对齐 ----------
    def track_index(self) -> dict[str, list[str]]:
        """把星图的域 id/名称映射到投研工作站里的赛道名，用于自动挂载。"""
        return {
            'ai':              ['人工智能', 'AI', '算力'],
            'embodied':        ['具身智能', '机器人'],
            'space':           ['商业航天', '航天'],
            'lowalt':          ['低空经济'],
            'quant':           [],
        }
