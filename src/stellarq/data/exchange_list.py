"""交易所官方股票清单（上交所 + 深交所）。

* 上交所：``query.sse.com.cn/sseQuery/commonQuery.do``，sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L，
  每页最多 100 条，需要 Referer=www.sse.com.cn。
* 深交所：``www.szse.cn/api/report/ShowReport/data``，CATALOGID=1110（A 股列表），每页 20 条。

两者都是交易所自己发布的清单，用来避免依赖第三方股票池（也就避免了第三方把退市股悄悄去掉）。
"""
from __future__ import annotations

import time

import pandas as pd
import requests

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def sse_list(page_size: int = 100, max_pages: int = 40) -> pd.DataFrame:
    out = []
    for page in range(1, max_pages + 1):
        r = requests.get(
            "https://query.sse.com.cn/sseQuery/commonQuery.do",
            params={"sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L", "isPagination": "true",
                    "pageHelp.pageSize": page_size, "pageHelp.pageNo": page,
                    "pageHelp.beginPage": page, "pageHelp.cacheSize": 1, "pageHelp.endPage": page},
            headers={"User-Agent": UA, "Referer": "https://www.sse.com.cn/"}, timeout=25)
        data = ((r.json() or {}).get("pageHelp") or {}).get("data") or []
        if not data:
            break
        out.extend(data)
        if len(data) < page_size:
            break
        time.sleep(0.1)
    if not out:
        return pd.DataFrame(columns=["symbol", "name", "exchange"])
    df = pd.DataFrame(out)
    rank = {
        "LISKIND": None,
        "COMPANY_ABBR": "name",
        "A_STOCK_CODE": "symbol",
        "LIST_BOARD": None,
        "STOCK_TYPE": None,
        "AREA_NAME": None,
    }
    res = pd.DataFrame({
        "symbol": df["A_STOCK_CODE"].astype(str).str.zfill(6),
        "name": df["COMPANY_ABBR"].astype(str),
    })
    res["exchange"] = "SSE"
    return res.drop_duplicates("symbol").reset_index(drop=True)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Connection": "close"})
    return s


def _get(sess: requests.Session, url: str, params: dict, referer: str, retries: int = 4):
    last = None
    for i in range(retries):
        try:
            return sess.get(url, params=params, headers={"Referer": referer}, timeout=25)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"交易所接口失败 {url}: {last}")


def szse_list(page_size: int = 20, max_pages: int = 200) -> pd.DataFrame:
    out = []
    sess = _session()
    for page in range(1, max_pages + 1):
        r = _get(sess, "https://www.szse.cn/api/report/ShowReport/data",
                 {"SHOWTYPE": "JSON", "CATALOGID": "1110", "TABKEY": "tab1", "PAGENO": page},
                 "https://www.szse.cn/")
        data = r.json()
        rows = []
        for block in data if isinstance(data, list) else []:
            rows.extend((block.get("data") or []))
        if not rows:
            break
        out.extend(rows)
        if len(rows) < page_size:
            break
        time.sleep(0.05)
    if not out:
        return pd.DataFrame(columns=["symbol", "name", "exchange"])
    df = pd.DataFrame(out)
    code_col = next((c for c in ("agdm", "证券代码", "code") if c in df.columns), None)
    name_col = next((c for c in ("agjc", "证券简称", "name") if c in df.columns), None)
    if code_col is None:
        return pd.DataFrame(columns=["symbol", "name", "exchange"])
    res = pd.DataFrame({"symbol": df[code_col].astype(str).str.zfill(6),
                        "name": df[name_col].astype(str) if name_col else ""})
    res["exchange"] = "SZSE"
    return res.drop_duplicates("symbol").reset_index(drop=True)


def exchange_universe() -> pd.DataFrame:
    a = sse_list()
    b = szse_list()
    u = pd.concat([a, b], ignore_index=True).drop_duplicates("symbol")
    return u.reset_index(drop=True)
