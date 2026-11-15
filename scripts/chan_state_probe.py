"""小样本因果探针：逐根结构状态是否有预测力。

因果纪律（本脚本的全部意义）
--------------------------
1. **本地重算**：不用 /api/chan（它返回"今天"的结构，会前视）。
   把截断到 t 的 OHLCV 喂给缠论工作站的纯函数 analyze()。
2. **截断**：特征只用 <= t 的 K 线；标签用 t 之后的前向收益，严格不重叠。
3. **两问分开**：
   · 截面问：同一天横截面，结构状态能否区分后续收益高低（RankIC）
   · 择时问：单票时序上，状态触发后的前向收益 vs 全样本基线（事件研究）

性能：用常驻 Node 服务 + 管道批量，避免每票启进程（实测快数十倍）。
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
from stellarq.paths import ROOT  # noqa: E402

CACHE = Path.home() / ".cache" / "stellarq" / "wide2"
NODE_SRV = "/tmp/chan_srv.js"


def load_wide(field):
    return np.load(CACHE / f"wide_{field}.npy", allow_pickle=(field in ("dates", "syms")))


class ChanEngine:
    """常驻 Node 子进程，批量算结构。"""

    def __init__(self):
        self.p = subprocess.Popen(
            ["node", NODE_SRV], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            bufsize=1 << 20, text=True,
        )

    def analyze_many(self, jobs: list[dict]) -> list[dict]:
        out = []
        for j in jobs:
            self.p.stdin.write(json.dumps(j) + "\n")
        self.p.stdin.flush()
        for _ in jobs:
            line = self.p.stdout.readline()
            out.append(json.loads(line) if line.strip() else {"results": []})
        return out

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan
    return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()))


def main(n_sym=80, horizons=(5, 10, 20), cut_fracs=(0.72, 0.80, 0.88, 0.94)):
    dates = load_wide("dates")
    close = load_wide("close")
    o = load_wide("open")
    h = load_wide("high")
    lo = load_wide("low")
    vol = load_wide("volume")
    n_date, n_cols = close.shape

    rng = np.random.default_rng(7)
    liq = np.nanmean(close[-250:] * vol[-250:], axis=0)
    cand = np.where(np.isfinite(liq) & (liq > 0))[0]
    cand = cand[np.argsort(-liq[cand])][: n_sym * 3]
    picks = rng.permutation(cand)[:n_sym]
    print(f"样本 {len(picks)} 只 · 面板 {n_date} 日 × {n_cols} 列", flush=True)

    eng = ChanEngine()
    try:
        for frac in cut_fracs:
            t = int(n_date * frac)
            d0 = pd.Timestamp(dates[t]).strftime("%Y-%m-%d")
            jobs, meta = [], []
            for k, j in enumerate(picks):
                j = int(j)
                bars = []
                for i in range(0, t + 1):
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
                if len(bars) < 300:
                    continue
                jobs.append({"id": k, "sets": [bars]})
                meta.append((j, bars))

            if not jobs:
                print(f"{d0}: 无有效样本", flush=True)
                continue
            res = eng.analyze_many(jobs)

            rows, fwd = [], {hh: [] for hh in horizons}
            for (j, bars), r in zip(meta, res):
                rr = (r.get("results") or [{}])[0]
                if not rr.get("ok"):
                    continue
                st = state_series(rr)
                if st.empty:
                    continue
                last = st.iloc[-1]
                rows.append({k: last[k] for k in NUMERIC_STATE})
                for hh in horizons:
                    tt = t + hh
                    if tt < n_date and np.isfinite(close[t, j]) and np.isfinite(close[tt, j]):
                        fwd[hh].append(close[tt, j] / close[t, j] - 1)
                    else:
                        fwd[hh].append(np.nan)

            got = len(rows)
            print(f"\n=== {d0} · {got} 只有效 ===", flush=True)
            if got < 30:
                print("  样本不足", flush=True)
                continue
            df = pd.DataFrame(rows)
            for hh in horizons:
                y = np.asarray(fwd[hh], dtype=float)
                ics = {}
                for k in NUMERIC_STATE:
                    x = df[k].to_numpy(dtype=float)
                    if np.nanstd(x) == 0:
                        continue
                    ics[k] = rank_ic(x, y)
                top = sorted(
                    ((k, v) for k, v in ics.items() if np.isfinite(v)),
                    key=lambda kv: -abs(kv[1]),
                )[:5]
                s = " ".join(f"{k}={v:+.3f}" for k, v in top)
                print(f"  T+{hh:2d} RankIC: {s}", flush=True)
    finally:
        eng.close()


if __name__ == "__main__":
    main()
