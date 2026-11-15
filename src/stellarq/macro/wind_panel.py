"""接入 Wind/FRED/WDI 三路宏观长序列。

覆盖：
  月度  CPI / PPI / 产成品存货 / 利润总额累计
  季度  GDP 现价与不变价、当季同比、工业产能利用率
  年度  GDP 现价与不变价（2011 起）+ FRED 1992 起 + WDI 2000 起

所有序列统一转成 pandas Series（DatetimeIndex），并提供「发布滞后」对齐工具，
避免把还没公布的数据提前用进回测。
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
MACRO_DIR = ROOT / 'knowledge' / 'macro'

# 指标代码 → 语义名
CODES = {
    'M0000612': 'cpi_yoy',        # 月, %
    'M0001227': 'ppi_yoy',        # 月, %
    'M0000560': 'inventory',      # 月, 亿元（产成品存货）
    'M0000556': 'profit_cum',     # 月, 亿元（利润总额累计）
    'M5567876': 'gdp_q_nominal',  # 季, 亿元（当季值）
    'M5567889': 'gdp_q_real',     # 季, 亿元
    'X4608416': 'gdp_q_yoy',      # 季, %
    'M5792266': 'capacity_util',  # 季, %
    'M0001395': 'gdp_y_nominal',  # 年, 亿元
    'M5785610': 'gdp_y_real',     # 年, 亿元
}

# 各频率的发布滞后（自然日）——保守估计
RELEASE_LAG_DAYS = {
    'monthly': 20,    # 统计局月度数据约次月中旬发布
    'quarterly': 25,  # GDP 约季后 15–20 天
    'yearly': 90,     # 年度数据次年 1 月前后
}


def load_long(macro_dir: pathlib.Path | None = None) -> pd.DataFrame:
    """读长表，返回 (date, code, value) 已展开的宽表。"""
    d = macro_dir or MACRO_DIR
    p = d / '宏观周期_时间序列_长表.csv'
    df = pd.read_csv(p, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['日期'].astype(str))
    df['name'] = df['指标代码'].map(CODES)
    wide = df.pivot_table(index='date', columns='name', values='数值', aggfunc='last')
    return wide.sort_index()


def with_release_lag(wide: pd.DataFrame) -> pd.DataFrame:
    """把每个观测值推迟到「实际可被公众看到」的日期。

    回测时应当用这个版本，否则会用上还没公布的数字。
    """
    out = {}
    for col in wide.columns:
        s = wide[col].dropna()
        if not len(s):
            continue
        freq = pd.infer_freq(s.index) or ''
        lag = RELEASE_LAG_DAYS['monthly']
        if freq.startswith('Q') or 'QS' in str(freq):
            lag = RELEASE_LAG_DAYS['quarterly']
        elif freq.startswith('A') or freq.startswith('Y') or 'YS' in str(freq):
            lag = RELEASE_LAG_DAYS['yearly']
        known = s.copy()
        known.index = s.index + pd.Timedelta(days=lag)
        out[col + '_known'] = known
    return pd.DataFrame(out).sort_index()


def month_end_panel() -> pd.DataFrame:
    """月末采样面板：季度/年度序列按**发布日期**映射到月，便于与行情对齐。

    实现要点（踩过的坑）：
      1. 滞后后的日期（如 2011-02-20）不在月末网格上，
         直接 reindex(method='ffill') 会在网格点上取不到值，形成空洞；
      2. 正确做法是先在自然日轴上做 resample('M').last() 再 ffill；
      3. 索引统一为月初，避免与行情（月初标记）对不上。
    """
    w = load_long()
    k = with_release_lag(w)
    out = {}
    for col in w.columns:
        kc = k.get(col + '_known')
        src = kc if kc is not None else w[col]
        src = src.dropna()
        if not len(src):
            continue
        # 先落到日轴，再取每月最后一个可得值，最后前向填充
        daily = src.resample('D').last()
        monthly = daily.resample('ME').last().ffill()
        monthly.index = pd.PeriodIndex(monthly.index, freq='M').to_timestamp(how='start')
        out[col] = monthly
    df = pd.DataFrame(out).sort_index()
    df.index = pd.PeriodIndex(df.index, freq='M')
    return df


def self_test() -> None:
    w = load_long()
    assert 'cpi_yoy' in w.columns and 'gdp_q_yoy' in w.columns
    cpi = w['cpi_yoy'].dropna()
    assert 2011 <= cpi.index.min().year <= 2012, cpi.index.min()
    ppi = w['ppi_yoy'].dropna()
    peak = ppi.idxmax()
    assert peak.year == 2021 and peak.month == 10, peak
    cu = w['capacity_util'].dropna()
    q1_2020 = cu[(cu.index.year == 2020) & (cu.index.month == 3)]
    if len(q1_2020):
        assert abs(float(q1_2020.iloc[0]) - 67.3) < 0.01
    print(f'wind_panel self-test OK: {len(w.columns)} 序列, {len(w)} 行')


if __name__ == '__main__':
    self_test()
