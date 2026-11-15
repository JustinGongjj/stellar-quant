"""巨潮资讯公告适配器（事件因子数据源）。

实测（2026-11-14）
------------------
* 接口：``POST http://www.cninfo.com.cn/new/hisAnnouncement/query``
* 需要先访问首页拿 cookie，并带 ``X-Requested-With: XMLHttpRequest``；
* 按日期区间分页，``seDate='YYYY-MM-DD~YYYY-MM-DD'``，单页最多 30 条，
  每次查询有 ``totalAnnouncement`` 可用于翻页；
* 返回字段含 ``secCode``（代码）、``secName``、``announcementTitle``（标题）、
  ``announcementTime``（毫秒时间戳，为**公告日**）、``adjunctUrl``（PDF 路径）。

为什么用公告日而不是报告期：事件因子的时点必须是「市场第一次知道」的那一天，
用报告期会产生前视偏差。本模块一律保留公告日。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"

# 事件类别：用巨潮的「全文检索关键词」抓，标题再做正则复核
EVENT_KEYWORDS = {
    "forecast": "业绩预告",
    "express": "业绩快报",
    "buyback": "回购",
    "increase": "增持",
    "reduce": "减持",
    "restructure": "重大资产重组",
    "penalty": "处罚",
    "equity_incentive": "股权激励",
    "abnormal": "股票交易异常波动",
}


@dataclass
class Cninfo:
    timeout: float = 30.0
    pause: float = 0.25

    def __post_init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": UA,
            "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
            "X-Requested-With": "XMLHttpRequest",
        })
        try:
            self.session.get("http://www.cninfo.com.cn/new/index", timeout=self.timeout)
        except Exception:  # noqa: BLE001
            pass

    def _query(self, page: int, page_size: int, se_date: str, keyword: str = "",
               column: str = "szse") -> dict:
        payload = {
            "pageNum": page, "pageSize": page_size, "column": column, "tabName": "fulltext",
            "stock": "", "seDate": se_date, "searchkey": keyword,
            "sortName": "", "sortType": "", "isHLtitle": "true",
        }
        for i in range(3):
            try:
                r = self.session.post(QUERY, data=payload, timeout=self.timeout)
                r.raise_for_status()
                time.sleep(self.pause)
                return r.json() or {}
            except Exception:  # noqa: BLE001
                time.sleep(1.2 * (i + 1))
        return {}

    def announcements(self, start: str, end: str, keyword: str = "",
                      max_pages: int = 200, page_size: int = 30) -> pd.DataFrame:
        """抓取 [start, end] 区间内的公告（可按关键词过滤）。"""
        se_date = f"{start}~{end}"
        first = self._query(1, page_size, se_date, keyword)
        total = int(first.get("totalAnnouncement") or 0)
        recs = list(first.get("announcements") or [])
        pages = min(max_pages, (total + page_size - 1) // page_size)
        for p in range(2, pages + 1):
            j = self._query(p, page_size, se_date, keyword)
            recs.extend(j.get("announcements") or [])
        if not recs:
            return pd.DataFrame(columns=["symbol", "name", "ann_date", "title", "url", "keyword"])
        df = pd.DataFrame(recs)
        out = pd.DataFrame({
            "symbol": df["secCode"].astype(str).str.zfill(6),
            "name": df["secName"].astype(str),
            "ann_date": pd.to_datetime(df["announcementTime"], unit="ms").dt.normalize(),
            "title": df["announcementTitle"].astype(str).str.replace(r"<[^>]+>", "", regex=True),
            "url": "http://static.cninfo.com.cn/" + df["adjunctUrl"].astype(str),
        })
        out["keyword"] = keyword
        return out.drop_duplicates(subset=["symbol", "ann_date", "title"]).reset_index(drop=True)
