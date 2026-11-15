"""新浪期货日线适配器（CTA / 趋势策略数据源）。

实测（2026-11-14）
------------------
* 接口：``stock2.finance.sina.com.cn/futures/api/jsonp.php/...InnerFuturesNewService.getDailyKLine?symbol=AU0``
* 返回 JSONP，需剥掉 ``/*<script>...*/ var _X=`` 前缀；
* 覆盖：AU0(黄金) 2008-01 起、RB0(螺纹) 2009-03 起、C0(玉米) 2005-01 起、
  CU0(铜) 2005-01 起、IF0(沪深300股指) 2017-01 起；
* 字段：``d/o/h/l/c/v``（连续合约，已做主力拼接，**不含隔夜跳空处理**）。

注意：连续合约在换月处有跳空，做趋势回测时必须用「价格比例」而非「价差」，
并且换月日的收益要单独识别（本模块给出 ``roll_gap`` 标记供回测剔除）。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import pandas as pd
import requests

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"
URL = ("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
       "var%20_{sym}=/InnerFuturesNewService.getDailyKLine?symbol={sym}")

# 本仓用的商品/股指池（覆盖贵金属、黑色、有色、农产品、能化、股指）
UNIVERSE = {
    "AU0": ("黄金", "贵金属"), "AG0": ("白银", "贵金属"),
    "RB0": ("螺纹钢", "黑色"), "I0": ("铁矿石", "黑色"), "JM0": ("焦煤", "黑色"),
    "CU0": ("铜", "有色"), "AL0": ("铝", "有色"), "ZN0": ("锌", "有色"),
    "C0": ("玉米", "农产品"), "M0": ("豆粕", "农产品"), "Y0": ("豆油", "农产品"),
    "TA0": ("PTA", "能化"), "MA0": ("甲醇", "能化"), "SC0": ("原油", "能化"),
    "IF0": ("沪深300股指", "股指"), "IC0": ("中证500股指", "股指"),
    "IH0": ("上证50股指", "股指"), "IM0": ("中证1000股指", "股指"),
}


@dataclass
class SinaFutures:
    timeout: float = 25.0
    pause: float = 0.25

    def __post_init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Referer": "https://finance.sina.com.cn"})

    def daily(self, symbol: str, retries: int = 3) -> pd.DataFrame:
        last = None
        for i in range(retries):
            try:
                r = self.session.get(URL.format(sym=symbol), timeout=self.timeout)
                r.raise_for_status()
                txt = r.text
                m = re.search(r"=\s*\((\[.*\])\)", txt, re.S)
                if not m:
                    return pd.DataFrame()
                rows = json.loads(m.group(1))
                time.sleep(self.pause)
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows)
                df = df.rename(columns={"d": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                for c in ("open", "high", "low", "close", "volume"):
                    if c in df:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df["date"] = pd.to_datetime(df["date"])
                df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                df["symbol"] = symbol
                # 连续合约换月跳空：单日收益超过 15% 视为换月/异常，标记出来
                ret = df["close"].pct_change()
                df["roll_gap"] = ret.abs() > 0.15
                df["ret"] = ret.where(~df["roll_gap"])
                return df
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1.0 * (i + 1))
        raise RuntimeError(f"新浪期货接口失败 {symbol}: {last}")


def load_universe(symbols=None) -> pd.DataFrame:
    """批量取回并纵向拼接。"""
    cli = SinaFutures()
    syms = list(symbols or UNIVERSE)
    frames = []
    for s in syms:
        try:
            d = cli.daily(s)
            if len(d):
                d["name"], d["sector"] = UNIVERSE.get(s, (s, "其他"))
                frames.append(d)
        except Exception:  # noqa: BLE001
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
