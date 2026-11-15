"""东方财富公开行情接口客户端（A 股专用）。

接口与字段口径来自实测（2026-11）。为便于审计，每个解析函数只做
「原始 JSON → 标准 DataFrame」一件事，不在这里做任何清洗决策。

关键口径
--------
* ``klt``：101 日 / 102 周 / 103 月；``fqt``：0 不复权 / 1 前复权 / 2 后复权。
* 前复权价（fqt=1）回看长历史会出现负价（分红累加超过历史价），**不能**用于
  收益率以外的绝对价格展示；收益率本身不受影响。
* 流通股数由 ``成交额 / 成交均价 / 换手率`` 反推：
  ``float_shares_t = amount_t / (avg_px_t * turnover_t)``，量纲转换为「股」。
  这一步是推断而非官方字段，因此在 manifest 中显式标注。
* 股票列表接口单页上限 100 条，必须按页拉全（约 60 页）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
PUSH = "https://push2.eastmoney.com/api/qt"
PUSHHIS = "https://push2his.eastmoney.com/api/qt"
DC = "https://datacenter-web.eastmoney.com/api/data/v1/get"

KLINE_FIELDS = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KLINE_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount",
                 "amplitude", "pct_chg", "change", "turnover"]

A_SHARE_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"


class EastmoneyError(RuntimeError):
    pass


@dataclass
class Client:
    timeout: float = 25.0
    retries: int = 3
    pause: float = 0.12

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"})

    def _get_json(self, url: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                payload = r.json()
                time.sleep(self.pause)
                return payload
            except Exception as exc:  # noqa: BLE001 - 网络层异常统一重试
                last = exc
                time.sleep(0.8 * (attempt + 1))
        raise EastmoneyError(f"请求失败 {url}: {last}")

    # ---------------------------------------------------------------- 股票池
    def stock_list(self) -> pd.DataFrame:
        """全市场 A 股清单（含实时价、市值、市盈率、行业名）。"""
        out: list[dict] = []
        page = 1
        while True:
            j = self._get_json(f"{PUSH}/clist/get", {
                "pn": page, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f12", "fs": A_SHARE_FS,
                "fields": "f12,f14,f3,f6,f9,f20,f21,f100,f13",
            })
            data = j.get("data") or {}
            diff = data.get("diff") or []
            if not diff:
                break
            out.extend(diff)
            if len(diff) < 100 or len(out) >= int(data.get("total") or 0):
                break
            page += 1
        df = pd.DataFrame(out)
        if df.empty:
            raise EastmoneyError("股票清单为空")
        df = df.rename(columns={
            "f12": "symbol", "f14": "name", "f13": "market_id", "f3": "pct_chg_today",
            "f6": "amount_today", "f9": "pe_ttm", "f20": "mktcap_total", "f21": "mktcap_float",
            "f100": "industry",
        })
        keep = ["symbol", "name", "market_id", "industry", "pe_ttm", "mktcap_total", "mktcap_float"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        for col in ("pe_ttm", "mktcap_total", "mktcap_float"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["exchange"] = df["symbol"].map(secid_prefix)
        return df.sort_values("symbol").reset_index(drop=True)

    # ---------------------------------------------------------------- 行情
    def daily_kline(self, symbol: str, start: str = "20050101", end: str = "20500101",
                    adjust: int = 0, kind: int = 101) -> pd.DataFrame:
        """日线（可复权）。返回列见 ``KLINE_COLUMNS``。"""
        secid = to_secid(symbol)
        j = self._get_json(f"{PUSHHIS}/stock/kline/get", {
            "secid": secid, "fields1": "f1,f2,f3,f4,f5,f6", "fields2": KLINE_FIELDS,
            "klt": kind, "fqt": adjust, "beg": start, "end": end,
        })
        data = j.get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        rows = [line.split(",") for line in klines]
        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        for col in KLINE_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["symbol"] = str(symbol).zfill(6)
        df["name"] = data.get("name")
        df["adjust"] = adjust
        return df

    def board_hist(self, board_code: str, start: str = "20050101", end: str = "20500101",
                   kind: int = 101, adjust: int = 1) -> pd.DataFrame:
        """行业板块历史行情（secid 前缀 90）。"""
        j = self._get_json(f"{PUSHHIS}/stock/kline/get", {
            "secid": board_code if "." in board_code else f"90.{board_code}",
            "fields1": "f1,f2,f3,f4,f5,f6", "fields2": KLINE_FIELDS,
            "klt": kind, "fqt": adjust, "beg": start, "end": end,
        })
        data = j.get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        df = pd.DataFrame([line.split(",") for line in klines], columns=KLINE_COLUMNS)
        for col in KLINE_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["board"] = data.get("code")
        df["name"] = data.get("name")
        return df

    def board_list(self) -> pd.DataFrame:
        j = self._get_json(f"{PUSH}/clist/get", {
            "pn": 1, "pz": 600, "po": 1, "np": 1, "fltt": 2, "invt": 2, "fid": "f12",
            "fs": "m:90+t:2", "fields": "f12,f14,f3",
        })
        df = pd.DataFrame((j.get("data") or {}).get("diff") or [])
        return df.rename(columns={"f12": "board", "f14": "name", "f3": "pct_chg"})

    def index_daily(self, symbol: str = "000300", kind: int = 101) -> pd.DataFrame:
        """指数日线（300 系列在 1. 前缀；000xxx 上证在 1.，深市 399xxx 在 0.）。"""
        return self.daily_kline(symbol, kind=kind, adjust=0)

    # ---------------------------------------------------------------- 基本面
    def fundamentals(self, symbol: str) -> pd.DataFrame:
        """业绩快报口径的核心财务字段，按报告期倒序。含公告日（时点纪律用）。"""
        j = self._get_json(DC, {
            "reportName": "RPT_LICO_FN_CPD", "columns": "ALL",
            "filter": f'(SECURITY_CODE="{symbol}")', "pageNumber": 1, "pageSize": 60,
            "sortColumns": "REPORTDATE", "sortTypes": -1, "source": "WEB", "client": "WEB",
        })
        rows = ((j.get("result") or {}).get("data")) or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "REPORTDATE": "report_date", "NOTICE_DATE": "notice_date",
            "TOTAL_OPERATE_INCOME": "revenue", "PARENT_NETPROFIT": "net_profit",
            "WEIGHTAVG_ROE": "roe", "YSTZ": "rev_yoy", "SJLTZ": "profit_yoy",
            "BPS": "bps", "BASIC_EPS": "eps", "XSMLL": "gross_margin",
            "BOARD_NAME": "board", "SECURITY_NAME_ABBR": "name",
        })
        df["symbol"] = str(symbol).zfill(6)
        return df


def secid_prefix(symbol: str) -> str:
    s = str(symbol).zfill(6)
    if s.startswith(("60", "68", "90", "50", "51", "58", "11")):
        return "SH"
    if s.startswith(("00", "30", "12", "15", "16", "18")):
        return "SZ"
    return "BJ"


def to_secid(symbol: str) -> str:
    s = str(symbol).zfill(6)
    ex = secid_prefix(s)
    market = 1 if ex == "SH" else 0
    return f"{market}.{s}"


def board_of_symbol(symbol: str) -> str:
    """按代码段划分交易所板块（用于涨跌幅限制与交易约束）。"""
    s = str(symbol).zfill(6)
    if s.startswith(("688", "689")):
        return "STAR"      # 科创：20%
    if s.startswith(("300", "301")):
        return "CHINEXT"   # 创业：20%
    if s.startswith(("8", "4", "92")):
        return "BSE"       # 北交所：30%
    return "MAIN"          # 主板：10%
