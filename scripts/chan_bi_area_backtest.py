"""阶段 5：cs_bi_area 作为组合微调项的回测对照（含成本）。

要回答的问题
------------
稳健性检验显示 cs_bi_area 分 5 组完美单调（Q5−Q1 = +1.32%/20日）。
但那是**毛收益**。这里回答三个更实际的问题：

  1. 扣掉 A 股真实成本（印花税 0.05% + 佣金万1.5 + 过户费 + 滑点）后还剩多少？
  2. 按"排序微调"用（全样本受益），比按"阈值只买 Q5"（损失 80% 样本）好多少？
  3. 与"随机换手同样多的仓位"相比，超额是否来自真实信息？

三个对照组合（同一批股票、同一批交易日、同一成本模型）
----------------------------------------------------
  A 基准   ：全样本等权持有（不做任何结构微调）
  B 微调   ：按 cs_bi_area 分位对权重做 ±20% 倾斜，再归一到满仓
  C 随机   ：同样倾斜幅度，但分组随机打乱（安慰剂对照）

超额 = B − C 才是"结构信息"的净贡献；B − A 会混入"倾斜本身"的效应。

执行口径
--------
  · T 日算分 → T+1 日收盘按目标权重调仓（与 portfolio/construct 一致）
  · 每 rebal 个交易日调一次
  · 成本按 turnover × 单边费率扣
  · 全部用面板原始 close（不复权），只用于相对比较
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

__import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from stellarq.bridge.chanlun_state import state_series
from stellarq.paths import ROOT  # noqa: E402

CACHE = Path.home() / ".cache" / "stellarq" / "wide2"
NODE_SRV = Path(__file__).with_name("chan_srv.js")

# A 股真实成本（与 portfolio/costs.py 口径一致）
BUY_COST = 0.00015 + 0.00001 + 0.0005      # 佣金 + 过户费 + 滑点
SELL_COST = 0.00015 + 0.00001 + 0.0005 + 0.0005  # 再加印花税
ROUND_TRIP = BUY_COST + SELL_COST


def load(f):
    return np.load(CACHE / f"wide_{f}.npy", allow_pickle=(f in ("dates", "syms")))


def build_bars(close, o, h, lo, vol, j, n_date, dates):
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
    return bars


def main(n_sym=300, rebal=20, warm=400, start_frac=0.55):
    dates = load("dates")
    close, o, h, lo, vol = load("close"), load("open"), load("high"), load("low"), load("volume")
    n_date, _ = close.shape

    rng = np.random.default_rng(11)
    liq = np.nanmean(close[-250:] * vol[-250:], axis=0)
    cand = np.where(np.isfinite(liq) & (liq > 0))[0]
    cand = cand[np.argsort(-liq[cand])][: n_sym * 3]
    picks = np.sort(rng.permutation(cand)[:n_sym])
    print(f"股票 {len(picks)} 只 · 调仓间隔 {rebal} 日", flush=True)

    # 算结构，投影成 date -> {sym: bi_area}
    p = subprocess.Popen(["node", str(NODE_SRV)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, bufsize=1 << 22, text=True)
    feat = {}   # date -> array(bi_area) 对齐 picks
    batch, done = [], 0
    all_rows = {}

    def flush(batch):
        nonlocal done
        if not batch:
            return
        for j, bars in batch:
            p.stdin.write(json.dumps({"id": int(j), "sets": [bars]}) + "\n")
        p.stdin.flush()
        for j, bars in batch:
            line = p.stdout.readline()
            try:
                rr = (json.loads(line)["results"] or [{}])[0] if line.strip() else {"ok": False}
            except Exception:
                rr = {"ok": False}
            if not rr.get("ok"):
                continue
            rr["bars"] = bars
            st = state_series(rr)
            if st.empty:
                continue
            all_rows[int(j)] = st[["date", "cs_bi_area"]].set_index("date")["cs_bi_area"]
            done += 1

    try:
        for k, j in enumerate(picks):
            j = int(j)
            bars = build_bars(close, o, h, lo, vol, j, n_date, dates)
            if len(bars) < warm:
                continue
            batch.append((j, bars))
            if len(batch) >= 4:
                flush(batch)
                batch = []
            if k % 50 == 0:
                print(f"  {k}/{len(picks)} 完成 {done}", flush=True)
        flush(batch)
    finally:
        try:
            p.stdin.close()
            p.wait(timeout=10)
        except Exception:
            p.kill()

    print(f"结构可用 {len(all_rows)} 只", flush=True)
    if len(all_rows) < 50:
        print("样本不足")
        return

    valid_syms = sorted(all_rows)
    # 对齐成矩阵：date × sym
    series = []
    for j in valid_syms:
        s = all_rows[j]
        s.index = pd.to_datetime(s.index)
        series.append(s.rename(j))
    feat_df = pd.concat(series, axis=1).sort_index()

    trade_dates = [pd.Timestamp(d) for d in dates]
    t0 = int(n_date * start_frac)
    # 同时保留「面板行号」和「日期」，后面要用行号取 close 矩阵
    rebal_ts = [(t, trade_dates[t]) for t in range(t0, n_date - rebal - 1, rebal)]
    rebal_ts = [(t, d) for t, d in rebal_ts if d in feat_df.index]
    print(f"调仓日 {len(rebal_ts)} 个（{rebal_ts[0][1].date()} → {rebal_ts[-1][1].date()}）", flush=True)

    sym_idx = {j: i for i, j in enumerate(valid_syms)}
    col_of = {j: feat_df.columns.get_loc(j) for j in valid_syms}

    def weights_for(t_pos, tilt):
        """返回目标权重 dict。tilt=False 时等权；True 时按分位倾斜。"""
        d = trade_dates[t_pos]
        if d not in feat_df.index:
            return None
        row = feat_df.loc[d]
        avail = [j for j in valid_syms if np.isfinite(row.get(j, np.nan))
                 and np.isfinite(close[t_pos, j]) and close[t_pos, j] > 2.0]
        if len(avail) < 40:
            return None
        if not tilt:
            return {j: 1.0 / len(avail) for j in avail}
        x = row[avail].to_numpy(dtype=float)
        q = pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False).to_numpy()
        # ±20% 倾斜：Q1 0.8, Q2 0.9, Q3 1.0, Q4 1.1, Q5 1.2
        mult = np.array([0.8, 0.9, 1.0, 1.1, 1.2])[q]
        w = mult / mult.sum()
        return dict(zip(avail, w))

    # 随机组：一次性生成固定分位标签并**持续持有**，使换手与结构组量级一致。
    # 原实现每次重新随机 → 换手 10%，而结构组仅 3.1%，成本不对等会让结论偏向 B。
    rand_label = {}

    def random_weights(t_pos):
        d = trade_dates[t_pos]
        if d not in feat_df.index:
            return None
        row = feat_df.loc[d]
        avail = [j for j in valid_syms if np.isfinite(row.get(j, np.nan))
                 and np.isfinite(close[t_pos, j]) and close[t_pos, j] > 2.0]
        if len(avail) < 40:
            return None
        for j in avail:
            if j not in rand_label:
                rand_label[j] = int(rng.integers(0, 5))
        mult = np.array([0.8, 0.9, 1.0, 1.1, 1.2])[[rand_label[j] for j in avail]]
        w = mult / mult.sum()
        return dict(zip(avail, w))

    def run(strategy):
        nav = 1.0
        prev_w = {}
        series_out = []
        turnovers = []
        for i, (t_pos, _d) in enumerate(rebal_ts):
            t_next = rebal_ts[i + 1][0] if i + 1 < len(rebal_ts) else n_date - 1
            if t_next <= t_pos:
                break
            w = strategy(t_pos)
            if w is None:
                continue
            # 区间收益（等权按权重加总）
            ret = 0.0
            for j, wj in w.items():
                c0, c1 = close[t_pos, j], close[t_next, j]
                if np.isfinite(c0) and np.isfinite(c1) and c0 > 0:
                    ret += wj * (c1 / c0 - 1)
            # 换手与成本
            names = set(w) | set(prev_w)
            turn = sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in names) / 2.0
            cost = turn * ROUND_TRIP
            net = ret - cost
            nav *= (1 + net)
            series_out.append((trade_dates[t_pos], net, ret, turn))
            turnovers.append(turn)
            prev_w = w
        df = pd.DataFrame(series_out, columns=["date", "net", "gross", "turnover"]).set_index("date")
        return df, nav

    res = {}
    for name, fn in [("A 基准等权", lambda t: weights_for(t, False)),
                     ("B 结构倾斜", lambda t: weights_for(t, True)),
                     ("C 随机倾斜", lambda t: random_weights(t))]:
        df, nav = run(fn)
        res[name] = (df, nav)
        n = len(df)
        ann = nav ** (252 / (n * rebal)) - 1 if n > 0 else np.nan
        shp = df["net"].mean() / df["net"].std() * np.sqrt(252 / rebal) if df["net"].std() > 0 else np.nan
        cum = np.cumprod(1 + df["net"])
        dd = float((1 - cum / np.maximum.accumulate(cum)).max()) if n > 0 else np.nan
        print(f"{name}: 期数 {n}  净值 {nav:.4f}  年化 {ann*100:+.2f}%  "
              f"Sharpe {shp:.2f}  最大回撤 {dd*100:.1f}%  平均换手 {df['turnover'].mean()*100:.1f}%", flush=True)

    # 关键对照
    a = res["A 基准等权"][0]["net"]
    b = res["B 结构倾斜"][0]["net"]
    c = res["C 随机倾斜"][0]["net"]
    idx = a.index.intersection(b.index).intersection(c.index)
    if len(idx) > 5:
        ex_bc = (b.loc[idx] - c.loc[idx]).mean()
        ex_ba = (b.loc[idx] - a.loc[idx]).mean()
        se = (b.loc[idx] - c.loc[idx]).std(ddof=1) / np.sqrt(len(idx))
        print(f"\nB−C（结构净贡献）：{ex_bc*100:+.3f}%/期  t={ex_bc/se if se>0 else np.nan:+.2f}")
        print(f"B−A（含倾斜效应）：{ex_ba*100:+.3f}%/期")
        print(f"→ 年化差异：B−C 约 {ex_bc*(252/rebal)*100:+.2f}%/年")

        # 保存
        outp = Path(__file__).parent.parent / "reports" / "data" / "chan_bi_area_backtest.csv"
        pd.DataFrame({"A_base": a, "B_tilt": b, "C_random": c}).to_csv(outp)
        print(f"\n明细已存 {outp}")


if __name__ == "__main__":
    main()
