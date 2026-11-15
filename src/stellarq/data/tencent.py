"""腾讯行情适配器（主数据源）。

为什么改用腾讯
--------------
2026-11-13 实测：东方财富 `push2his` 对高频/批量请求返回 `rc:102`（限流封禁），
且其 `pct_chg` 字段与相邻收盘价反推的涨跌幅**不一致**（实测 2024-02-21 平安银行
两口径差 2.8 个百分点），说明该字段不可作为价格或校验依据。
腾讯接口在本环境稳定、无 key、且同一请求可重复取到一致结果（qfq 两次请求漂移 0）。

接口能力（实测）
----------------
* 日线：``web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=<sym>,day,,<end>,2000,<mode>``
  - 单次上限 2000 根，用 ``end`` 向前翻页（3 次即可覆盖 2012 至今）；
  - ``mode`` 为空 = 不复权，``qfq`` = 前复权（腾讯内部对最近 1919 根保持一致）；
  - 每根 K 线可选第 7 个元素为除权事件字典：
    ``{"nd":"2024","fh_sh":"2.46","djr":"2024-10-09","cqr":"2024-10-10","FHcontent":"10派2.46元"}``
  - **``fh_sh`` 的实测含义**：派息日 = 每 10 股税后派现；送转日 = 每 10 股税前派现。
6 只样本逐日核对：不复权收盘价与腾讯完全一致，可用作唯一价格口径。

字段口径（不复权）
------------------
``[date, open, close, high, low, volume(手)]`` —— 注意第 2 位是**收盘**、第 3 位是**开盘**，
与常见的 OHLC 顺序不同，误用会把所有 K 线因子算反。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import pandas as pd
import requests

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"
FQKL_HOSTS = ("https://ifzq.gtimg.cn/appstock/app/fqkline/get",
              "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get")
MKLINE = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
QUOTE = "https://qt.gtimg.cn/q="


def to_tx_symbol(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("5", "6", "9")):
        return "sh" + s
    if s.startswith(("4", "8")):
        return "bj" + s
    return "sz" + s


@dataclass
class Tencent:
    timeout: float = 30.0
    pause: float = 0.05

    def __post_init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Referer": "https://gu.qq.com/"})

    def _get_json(self, url, params: dict, retries: int = 3) -> dict:
        """url 可以是字符串或候选主机列表；501/429/5xx 视为限流，换主机并退避重试。"""
        hosts = list(url) if isinstance(url, (list, tuple)) else [url]
        last = None
        for attempt in range(retries):
            for host in hosts:
                try:
                    r = self.session.get(host, params=params, timeout=self.timeout)
                    if r.status_code in (429, 501, 502, 503):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    payload = r.json()
                    time.sleep(self.pause)
                    return payload
                except Exception as exc:  # noqa: BLE001
                    last = exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"腾讯接口失败 {hosts[0]}: {last}")

    # ------------------------------------------------------------------ 日线
    def daily(self, symbol: str, mode: str = "", start: str = "2012-01-01",
              pages: int = 3) -> pd.DataFrame:
        """翻页取全量日线（不复权 mode="" / 前复权 mode="qfq"）。

        返回列：date / open / close / high / low / volume / corp_action(dict or None)。
        """
        sym = to_tx_symbol(symbol)
        rows: list[list] = []
        end = ""
        for _ in range(pages):
            j = self._get_json(FQKL_HOSTS, {"param": f"{sym},day,{start},{end},2000,{mode}"})
            data = j.get("data")
            if not isinstance(data, dict) or sym not in data:
                break
            node = data[sym]
            keys = [k for k, v in node.items() if isinstance(v, list) and v and isinstance(v[0], list)]
            if not keys:
                break
            part = node[keys[0]]
            rows = part + rows
            end = str(part[0][0])
            if end <= start:
                break
        if not rows:
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "corp_action"])
        recs = []
        for r in rows:
            if len(r) < 6:
                continue
            recs.append({
                "date": r[0],
                "open": float(r[1]),
                "close": float(r[2]),
                "high": float(r[3]),
                "low": float(r[4]),
                "volume": float(r[5]) if r[5] not in ("", None) else float("nan"),
                "corp_action": r[6] if len(r) > 6 and isinstance(r[6], dict) and r[6] else None,
            })
        df = pd.DataFrame(recs).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = str(symbol).zfill(6)
        return df

    def universe(self) -> pd.DataFrame:
        """股票清单：用腾讯板块成分接口逐板块取（沪深主板/创业板/科创板/北交所）。"""
        out = []
        boards = ["hs_a"]  # 沪深 A 股全量
        for b in boards:
            page = 1
            while True:
                j = self._get_json("https://proxy.finance.qq.com/cgi/cgi-bin/rank/pt/getRank",
                                  {"board_type": b, "sort_type": "price", "direct": "down",
                                   "offset": (page - 1) * 200, "count": 200})
                data = (j.get("data") or {})
                rows = data.get("rank_list") or data.get("list") or []
                if not rows:
                    break
                out.extend(rows)
                if len(rows) < 200:
                    break
                page += 1
                if page > 40:
                    break
        if not out:
            return pd.DataFrame(columns=["symbol", "name"])
        df = pd.DataFrame(out)
        col_code = next((c for c in ("code", "stock_code", "symbol") if c in df.columns), None)
        col_name = next((c for c in ("name", "stock_name") if c in df.columns), None)
        if col_code is None:
            return pd.DataFrame(columns=["symbol", "name"])
        res = pd.DataFrame({"symbol": df[col_code].astype(str).str.zfill(6)})
        res["name"] = df[col_name].astype(str) if col_name else ""
        return res.drop_duplicates("symbol").reset_index(drop=True)

    # ------------------------------------------------------------------ 实时行情
    def quotes(self, symbols: list[str]) -> pd.DataFrame:
        """批量实时行情（含官方昨收 ``prec``、涨跌停价、换手率、流通市值）。"""
        syms = [to_tx_symbol(s) for s in symbols]
        out = []
        for i in range(0, len(syms), 60):
            chunk = syms[i:i + 60]
            r = self.session.get(QUOTE + ",".join(chunk), timeout=self.timeout)
            r.encoding = "gbk"
            for line in r.text.split(";"):
                if "=" not in line or '"' not in line:
                    continue
                key = line.split("=")[0].strip().replace("v_", "")
                body = line.split('"')[1].split("~")
                if len(body) < 50:
                    continue
                def f(idx, default=float("nan")):
                    try:
                        return float(body[idx])
                    except (ValueError, IndexError):
                        return default
                out.append({
                    "symbol": key[2:], "name": body[1], "now": f(3), "prev_close": f(4), "open": f(5),
                    "volume": f(6), "high": f(33), "low": f(34), "amount": f(37) * 1e4,
                    "turnover": f(38), "pe": f(39), "amplitude": f(43), "float_mktcap": f(44) * 1e8,
                    "total_mktcap": f(45) * 1e8, "pb": f(46), "limit_up": f(47), "limit_down": f(48),
                    "time": body[30],
                })
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ 公司行为
    def corporate_actions_from(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """从已取回的日线里提取公司行为（避免为同一只股票重复请求两次）。"""
        rows = []
        for _, r in df.iterrows():
            ca = r.get("corp_action")
            if not isinstance(ca, dict):
                continue
            plan = str(ca.get("FHcontent", ""))
            try:
                per10 = float(ca.get("fh_sh", 0) or 0)
            except (TypeError, ValueError):
                per10 = 0.0
            bonus = 0.0
            for pat in (r"送(\d+(?:\.\d+)?)股", r"转(?:增)?(\d+(?:\.\d+)?)股"):
                m = re.search(pat, plan)
                if m:
                    bonus += float(m.group(1)) / 10.0
            is_after_tax = "扣税后" in plan or "税后" in plan
            rows.append({
                "symbol": str(symbol).zfill(6),
                "ex_date": pd.Timestamp(ca.get("cqr") or r["date"]),
                "cash_per_share": per10 / 10.0,
                "bonus_per_share": bonus,
                "tax": "after" if is_after_tax and bonus == 0 else "before",
                "plan": plan,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["symbol", "ex_date", "cash_per_share", "bonus_per_share", "tax", "plan"])
        return (out.drop_duplicates(subset=["ex_date", "cash_per_share", "bonus_per_share"])
                .sort_values("ex_date").reset_index(drop=True))

    def corporate_actions(self, symbol: str, pages: int = 2) -> pd.DataFrame:
        """从日线附带的除权字典提取公司行为，并解析送转比例。

        返回列：ex_date / cash_per_share / bonus_per_share / tax / plan。
        口径说明：``fh_sh`` 在派息日为**税后**、在送转日（含送转）为**税前**，
        这里统一折算为「交易所除权参考价口径」，额外保留 ``tax`` 列供审计。
        """
        df = self.daily(symbol, mode="", pages=pages)
        rows = []
        for _, r in df.iterrows():
            ca = r["corp_action"]
            if not isinstance(ca, dict):
                continue
            plan = str(ca.get("FHcontent", ""))
            try:
                per10 = float(ca.get("fh_sh", 0) or 0)
            except (TypeError, ValueError):
                per10 = 0.0
            bonus = 0.0
            for pat in (r"送(\d+(?:\.\d+)?)股", r"转(?:增)?(\d+(?:\.\d+)?)股"):
                m = re.search(pat, plan)
                if m:
                    bonus += float(m.group(1)) / 10.0
            is_after_tax = "扣税后" in plan or "税后" in plan
            rows.append({
                "symbol": str(symbol).zfill(6),
                "ex_date": pd.Timestamp(ca.get("cqr") or r["date"]),
                "cash_per_share": per10 / 10.0,
                "bonus_per_share": bonus,
                "tax": "after" if is_after_tax and bonus == 0 else "before",
                "plan": plan,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["symbol", "ex_date", "cash_per_share", "bonus_per_share", "tax", "plan"])
        return out.drop_duplicates(subset=["ex_date", "cash_per_share", "bonus_per_share"]).sort_values("ex_date").reset_index(drop=True)
