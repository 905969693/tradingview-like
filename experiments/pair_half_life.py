"""观察列表里的 ETF 两两配对，只估计 2024 年之前的对数比半衰期。

不做背离。外汇、个股和指数不进配对。比例方向不影响半衰期，每对只算一次。
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import signal_presets as presets  # noqa: E402
import tradingview_realization_v2 as tv  # noqa: E402
from experiments.gld_slv_spread import read_cached  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "pair_half_life"
TEST_START = pd.Timestamp("2024-01-01")
MIN_OBS = 504
NOT_ETF = {"1810.HK", "O39.SI", "_KS11"}


def etf_names():
    names = []
    for name in presets.watchlist():
        if name in NOT_ETF or "=" in name or name.endswith("_X"):
            continue
        safe = tv.safe_chart_name(name)
        period = "10y" if (tv.CACHE_DIR / f"{safe}_10y.csv").exists() else "5y"
        if not (tv.CACHE_DIR / f"{safe}_{period}.csv").exists():
            continue
        names.append((name, period))
    return names


def half_life_stats(log_px):
    y = pd.Series(log_px).dropna()
    lag = y.shift(1)
    dy = y.diff()
    valid = pd.concat([dy, lag], axis=1).dropna()
    x = valid.iloc[:, 1].to_numpy()
    target = valid.iloc[:, 0].to_numpy()
    n = len(target)
    design = np.column_stack([np.ones(n), x])
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    sigma2 = float(resid @ resid) / (n - 2)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    slope = float(coef[1])
    half_life = np.inf if slope >= 0 or not np.isfinite(slope) else float(-np.log(2) / slope)
    return half_life, slope, slope / se, n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = {}
    for name, period in etf_names():
        close = read_cached(name, period)["Close"].astype(float)
        close = close[close > 0]
        series[name] = close.loc[close.index < TEST_START]
    rows = []
    for left, right in combinations(sorted(series), 2):
        pair = pd.concat(
            [np.log(series[left]).rename("left"), np.log(series[right]).rename("right")],
            axis=1,
            join="inner",
        ).dropna()
        if len(pair) < MIN_OBS + 2:
            continue
        half_life, slope, tstat, n = half_life_stats(pair["left"] - pair["right"])
        rows.append(
            {
                "甲": left,
                "乙": right,
                "半衰期": half_life,
                "斜率": slope,
                "t值": tstat,
                "样本数": n,
                "开始": pair.index.min().date().isoformat(),
                "结束": pair.index.max().date().isoformat(),
            }
        )
    out = pd.DataFrame(rows).sort_values(["半衰期", "甲", "乙"])
    out.to_csv(OUT_DIR / "half_life.csv", index=False)
    fast = out[out["半衰期"] <= 100]
    print(f"ETF {len(series)} 只，配对 {len(out)} 对，半衰期<=100 的 {len(fast)} 对")
    print(", ".join(sorted(series)))
    shown = fast.copy()
    shown["半衰期"] = shown["半衰期"].map(lambda v: f"{v:.1f}")
    shown["斜率"] = shown["斜率"].map(lambda v: f"{v:.4f}")
    shown["t值"] = shown["t值"].map(lambda v: f"{v:.2f}")
    print(shown.to_string(index=False))


if __name__ == "__main__":
    main()
