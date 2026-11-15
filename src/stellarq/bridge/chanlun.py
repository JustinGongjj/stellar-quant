"""缠论工作站适配器：把「走势结构」转成可计算的技术面特征。

接口：GET /api/chan?market=A&code=<code>&period=day&limit=<n>

提取的特征（全部来自缠论结构，不是传统技术指标）：
  ch_trend       走势类型编码（下跌趋势 -1 / 盘整 0 / 上涨趋势 +1）
  ch_pos         当前价相对最近中枢的位置（中枢上方 +1 / 内部 0 / 下方 -1）
  ch_pos_pct     (现价 − 中枢中轴)/中枢高度，连续值，越正越强
  ch_pivots      中枢数量（>5 表示长期震荡或趋势延伸）
  ch_last_signal 最近买卖点（1/2/3 类买点为正，卖点为负，无信号 0）
  ch_sig_age     最近信号距今天数（越近越新鲜）
  ch_seg_dir     最后线段方向
  ch_bi_dir      最后一笔方向
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE = 'http://127.0.0.1:8931'

TREND_MAP = {'上涨趋势': 1, '下跌趋势': -1, '盘整': 0}
POS_MAP = {'above': 1, 'inside': 0, 'in': 0, 'below': -1}


@dataclass
class ChanlunStation:
    base: str = DEFAULT_BASE
    timeout: float = 12.0

    def _get(self, path: str, **params) -> dict:
        q = urllib.parse.urlencode(params)
        url = f'{self.base}{path}?{q}'
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            return json.loads(r.read().decode('utf-8'))

    def available(self) -> bool:
        try:
            self._get('/api/kline', market='A', code='000001', period='day', limit=5)
            return True
        except Exception:
            return False

    def analyze(self, code: str, market: str = 'A', period: str = 'day', limit: int = 300) -> dict | None:
        try:
            d = self._get('/api/chan', market=market, code=code, period=period, limit=limit)
            return d if d.get('ok') else None
        except Exception:
            return None

    @staticmethod
    def features(d: dict) -> dict:
        """从 /api/chan 的返回里抽技术面特征。"""
        if not d:
            return {}
        st = d.get('stats') or {}
        tr = d.get('trend') or {}
        piv = st.get('activePivot') or {}
        price = st.get('price')
        f = {
            'ch_trend': TREND_MAP.get(tr.get('type'), 0),
            'ch_pos': POS_MAP.get(st.get('position'), 0),
            'ch_pivots': st.get('pivots', 0),
            'ch_bi': st.get('bis', 0),
            'ch_segments': st.get('segments', 0),
        }
        # 相对中枢的连续位置
        if price and piv.get('zg') and piv.get('zd'):
            mid = (piv['zg'] + piv['zd']) / 2
            h = max(piv['zg'] - piv['zd'], 1e-9)
            f['ch_pos_pct'] = (price - mid) / h
        # 最近买卖点
        sigs = d.get('signals') or []
        bars = d.get('bars') or []
        if sigs:
            s = sigs[-1]
            # 站点的信号编码是 1B/2B/3B（买点）、1S/2S/3S（卖点），
            # 中文名在 name 字段（如「第三类买点」）。两者都要认，否则全部误判为卖点。
            kind = str(s.get('type') or s.get('kind') or '')
            nm = str(s.get('name') or '')
            if '买' in nm or kind.upper().endswith('B'):
                side = 1
            elif '卖' in nm or kind.upper().endswith('S'):
                side = -1
            else:
                side = 0
            f['ch_last_signal'] = side
            f['ch_signal_type'] = kind
            bi = s.get('barIdx', s.get('bar', None))
            f['ch_sig_age'] = (len(bars) - 1 - int(bi)) if (bi is not None and bars) else None
        else:
            f['ch_last_signal'] = 0
            f['ch_signal_type'] = ''
            f['ch_sig_age'] = None
        # 线段/笔方向
        segs = d.get('segments') or []
        bis = d.get('bis') or []
        f['ch_seg_dir'] = segs[-1].get('dir', 0) if segs else 0
        f['ch_bi_dir'] = bis[-1].get('dir', 0) if bis else 0
        return f
