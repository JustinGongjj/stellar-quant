"""大样本因果探针：结构状态序列的截面预测力（高统计力版）。

方法学说明（为什么这样做）
------------------------
逐截面"截断重算"最严格，但代价是每个截面都要把全历史 K 线重新喂一遍，
10 个截面 × 200 只已经是十几分钟，统计力仍然很弱（10 个 IC 样本）。

本脚本改为：**每只股票只算一次全历史结构，再按 `i1 <= t` 因果投影**，
在每个截面日取该日对应的状态行。好处：
  · 一次计算 → 可免费采样几十上百个截面，统计力提升一个数量级；
  · 投影规则与 `chanlun_state.state_series` 完全一致（只认 i1<=t）。

代价（必须如实记录）：全历史算出的"已确认笔/中枢"与"截断到 t 重算"会有差异
——这是缠论本身的非因果性，不是投影层的问题。`tests/test_chanlun_state.py`
里有一条截断漂移测试专门量化它。**若漂移很大，本脚本结论只能当作上界。**

输出：每个特征 × 期限的
  · IC 均值、t 统计量（Newey-West 未做，按独立截面近似）
  · 符号一致率
  · 分年度稳定性（防止某一年的行情主导结论）
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

__import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.chanlun_state import NUMERIC_STATE, state_series
from stellarq.paths import DATA, ROOT  # noqa: E402

CACHE = Path.home() / ".cache" / "stellarq" / "wide2"
NODE_SRV = str(Path(__file__).with_name("chan_srv.js"))


def load_wide(field):
    return np.load(CACHE / f"wide_{field}.npy", allow_pickle=(field in ("dates", "syms")))


class ChanEngine:
    def __init__(self):
        self.p = subprocess.Popen(
            ["node", NODE_SRV], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            bufsize=1 << 22, text=True,
        )

    def analyze_many(self, jobs, chunk=4):
        """分批发送。

        一次性把 400 只 × 3200 根 K 线灌进管道会让 Node 端缓冲区撑爆（BrokenPipe），
        所以按 chunk 分批写、写一批读一批，内存峰值可控。
        """
        out = []
        for i in range(0, len(jobs), chunk):
            part = jobs[i:i + chunk]
            try:
                for j in part:
                    self.p.stdin.write(json.dumps(j) + "\n")
                self.p.stdin.flush()
            except BrokenPipeError:
                # Node 挂了：重启并标记这批失败
                self._restart()
                for _ in part:
                    out.append({"results": [{"ok": False, "error": "pipe"}]})
                continue
            for _ in part:
                line = self.p.stdout.readline()
                try:
                    out.append(json.loads(line) if line.strip() else {"results": [{"ok": False}]})
                except json.JSONDecodeError:
                    out.append({"results": [{"ok": False, "error": "truncated"}]})
        return out

    def _restart(self):
        try:
            self.p.kill()
        except Exception:
            pass
        self.p = subprocess.Popen(
            ["node", NODE_SRV], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            bufsize=1 << 22, text=True,
        )

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=15)
        except Exception:
            self.p.kill()


def rank_ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 40:
        return np.nan
    return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()))


def main(n_sym=400, horizons=(5, 10, 20), step=20, warm=400):
    dates = load_wide("dates")
    close = load_wide("close")
    o, h, lo = load_wide("open"), load_wide("high"), load_wide("low")
    vol = load_wide("volume")
    n_date, n_cols = close.shape

    rng = np.random.default_rng(11)
    liq = np.nanmean(close[-250:] * vol[-250:], axis=0)
    cand = np.where(np.isfinite(liq) & (liq > 0))[0]
    cand = cand[np.argsort(-liq[cand])][: n_sym * 3]
    picks = np.sort(rng.permutation(cand)[:n_sym])
    print(f"股票 {len(picks)} 只 · 面板 {n_date} 日", flush=True)

    # 一次算全历史
    jobs, meta = [], []
    for k, j in enumerate(picks):
        j = int(j)
        bars = []
        for i in range(n_date):
            c_ = close[i, j]
            if not np.isfinite(c_):
                continue
            bars.append({
                "t": int(pd.Timestamp(dates[i]).timestamp() * 1000),
                "o": float(o[i, j]) if np.isfinite(o[i, j]) else float(c_),
                "h": float(h[i, j]) if np.isfinite(h[i, j]) else float(c_),
                "l": float(lo[i, j]) if np.isfinite(lo[i, j]) else float(c_),
                "c": float(c_),
                "v": float(vol[i, j]) if np.isfinite(vol[i, j]) else 0.0,
            })
        if len(bars) < warm:
            continue
        jobs.append({"id": k, "sets": [bars]})
        meta.append((j, bars))

    print(f"待算 {len(jobs)} 只，开始…", flush=True)
    eng = ChanEngine()
    states = {}
    try:
        res = eng.analyze_many(jobs)
    finally:
        eng.close()
    print(f"结构算完，投影中…", flush=True)

    fail = 0
    for (j, bars), r in zip(meta, res):
        rr = (r.get("results") or [{}])[0]
        if not rr.get("ok"):
            fail += 1
            continue
        # Node 端不再回传 bars（会撑爆管道），这里补回本地持有的一份
        rr["bars"] = bars
        st = state_series(rr)
        if not st.empty:
            states[j] = st

    print(f"可用 {len(states)} 只（失败 {fail}）", flush=True)
    if len(states) < 50:
        print("样本不足，退出")
        return

    # 建立 date -> 行号 映射（按状态表自身的日期轴）
    any_st = next(iter(states.values()))
    st_dates = any_st["date"].to_numpy()
    # 面板日期 -> 状态行号（状态只含有效 bar，日期是子集）
    dpos = {pd.Timestamp(x): i for i, x in enumerate(st_dates)}

    sym_index = {j: i for i, j in enumerate(sorted(states))}
    syms_sorted = sorted(states)

    # 采样截面：从 warm 之后到末尾，每 step 个交易日
    cuts = [t for t in range(warm, n_date - max(horizons) - 1, step) if pd.Timestamp(dates[t]) in dpos]
    print(f"截面 {len(cuts)} 个", flush=True)

    recs = {}
    for t in cuts:
        d_t = pd.Timestamp(dates[t])
        ri = dpos[d_t]
        rows, ys = [], {hh: [] for hh in horizons}
        for j in syms_sorted:
            st = states[j]
            if ri >= len(st):
                continue
            row = st.iloc[ri]
            ok = True
            vals = {}
            for k in NUMERIC_STATE:
                v = row[k]
                vals[k] = v
            # 前向收益
            rets = {}
            for hh in horizons:
                tt = t + hh
                if tt < n_date and np.isfinite(close[t, j]) and np.isfinite(close[tt, j]) and close[t, j] > 0:
                    rets[hh] = close[tt, j] / close[t, j] - 1
                else:
                    ok = False
                    break
            if not ok:
                continue
            rows.append(vals)
            for hh in horizons:
                ys[hh].append(rets[hh])
        if len(rows) < 80:
            continue
        df = pd.DataFrame(rows)
        for hh in horizons:
            y = np.asarray(ys[hh], dtype=float)
            for k in NUMERIC_STATE:
                x = df[k].to_numpy(dtype=float)
                if np.nanstd(x) == 0:
                    continue
                ic = rank_ic(x, y)
                if np.isfinite(ic):
                    recs.setdefault((k, hh), []).append((d_t, ic))

    # 汇总
    print("\n" + "=" * 78)
    print(f"{'特征':<14}{'期限':>5}{'截面数':>6}{'IC均值':>9}{'t值':>7}{'同号':>7}{'正年':>6}{'负年':>6}")
    print("=" * 78)
    out_lines = []
    for k in NUMERIC_STATE:
        for hh in horizons:
            v = recs.get((k, hh), [])
            if len(v) < 10:
                continue
            ics = np.array([x[1] for x in v])
            dts = [x[0] for x in v]
            mu = ics.mean()
            se = ics.std(ddof=1) / np.sqrt(len(ics))
            tstat = mu / se if se > 0 else np.nan
            cons = max((ics > 0).sum(), (ics < 0).sum()) / len(ics)
            # 分年度
            yr = pd.Series(ics, index=pd.DatetimeIndex(dts)).groupby(lambda x: x.year).mean()
            pos_y, neg_y = int((yr > 0).sum()), int((yr < 0).sum())
            line = (f"{k:<14}{hh:>5}{len(ics):>6}{mu:+9.3f}{tstat:>7.2f}"
                    f"{cons*100:>6.0f}%{pos_y:>6}{neg_y:>6}")
            print(line, flush=True)
            out_lines.append(line)

    # 存盘
    outp = Path("DATA/chan_state_ic.txt")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n已存 {outp}")


def robust_check(states, dpos, syms, close, dates, n_date, cuts, H=20):
    """对唯一显著特征 cs_bi_area 做四项稳健性检验。"""
    rows = []
    for t in cuts:
        d_t = pd.Timestamp(dates[t])
        if d_t not in dpos:
            continue
        ri = dpos[d_t]
        for j in syms:
            st = states[j]
            if ri >= len(st):
                continue
            x = st.iloc[ri]["cs_bi_area"]
            tt = t + H
            if not np.isfinite(x) or tt >= n_date:
                continue
            c0, c1 = close[t, j], close[tt, j]
            if not np.isfinite(c0) or not np.isfinite(c1) or c0 <= 0:
                continue
            rows.append({"date": d_t, "x": float(x), "y": float(c1 / c0 - 1)})
    if len(rows) < 5000:
        print("\n稳健性检验：样本不足")
        return
    df = pd.DataFrame(rows)

    def ic_by(g):
        return g.groupby("date").apply(
            lambda d: d["x"].corr(d["y"], method="spearman") if len(d) >= 40 else np.nan
        ).dropna()

    print("\n" + "=" * 78)
    print("cs_bi_area 稳健性检验（T+20）")
    print("=" * 78)
    base = ic_by(df)
    tstat = base.mean() / (base.std(ddof=1) / np.sqrt(len(base)))
    print(f"基准          IC={base.mean():+.4f}  t={tstat:+.2f}  n={len(base)}")

    halves = df.groupby(df["date"].dt.to_period("6M"))
    bad = 0
    tot = 0
    for p, g in halves:
        s = ic_by(g)
        if len(s) >= 5:
            tot += 1
            if s.mean() <= 0:
                bad += 1
    print(f"分半年        正向 {tot - bad}/{tot} 个半年")

    d2 = df.copy()
    d2["x"] = d2.groupby("date")["x"].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
    s2 = ic_by(d2)
    print(f"winsorize     IC={s2.mean():+.4f}  t={s2.mean() / (s2.std(ddof=1) / np.sqrt(len(s2))):+.2f}")

    df["q"] = df.groupby("date")["x"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.notna().sum() >= 50 else np.nan)
    g = df.dropna(subset=["q"]).groupby(["date", "q"])["y"].mean().unstack()
    if g.shape[1] == 5:
        mn = g.mean()
        print("分 5 组：      " + "  ".join(f"Q{i+1}={mn[i]*100:+.2f}%" for i in range(5)))
        spread = (mn[4] - mn[0]) * 100
        mono = pd.Series(mn.values).corr(pd.Series(range(5)), method="spearman")
        print(f"Q5-Q1         {spread:+.3f}% / 20日   单调性={mono:+.2f}")


if __name__ == "__main__":
    main()
