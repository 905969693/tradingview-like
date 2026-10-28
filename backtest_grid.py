"""
参数网格回测：PIVOT_ORDER(3~6) × MIN_COUNT(3~6)，共 16 个组合

- 数据全部直接读本地 cache/ 的 CSV，零网络请求（无缓存的标的才回退在线下载）。
- 每个组合内 ENTRY_SHIFT = 该组合的 pivot_order（确认滞后随参数变化）。
- DEDUP_BARS 固定 10（与 v2 默认一致）。
- 排名主指标：多空合并的不重叠累计收益（多空各半仓，跨标的平均）；
  辅指标：平均胜率 / 平均每笔收益 / 盈利标的占比。

输出（backtest/grid_<日期>/）：
- grid_summary.csv        16 个组合的全部汇总指标（按主指标排序）
- grid_heatmap.png        胜率 / 平均每笔收益 / 多空合并累计收益 三张热图
- best_pivot{p}_min{m}/   最优组合下逐标的交易明细 + 净值图
"""
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tradingview_realization_v2 as tv
import backtest_divergence as bt

# ============ 0. 可调参数 ============
TICKERS = tv.TICKERS
PERIOD = tv.PERIOD
DEDUP_BARS = tv.DEDUP_BARS
HORIZONS = tv.STATS_HORIZONS
PIVOT_RANGE = range(3, 7)      # PIVOT_ORDER: 3,4,5,6
MINCOUNT_RANGE = range(3, 7)   # MIN_COUNT:  3,4,5,6
OUT_DIR = tv.SCRIPT_DIR / "backtest" / f"grid_{date.today().isoformat()}"


def load_cached_only(ticker):
    """优先直接读 cache/ CSV（零网络）；无缓存才回退在线下载。"""
    cache_path = tv.CACHE_DIR / f"{tv.safe_chart_name(ticker)}_{PERIOD}.csv"
    if cache_path.exists():
        raw = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return tv.add_indicators(tv._to_ohlcv(raw))
    print(f"  [warn] {ticker} 无本地缓存，回退在线下载")
    return tv.add_indicators(tv.load_ohlcv(ticker, period=PERIOD))


def signals_with(df, pivot_order, min_count, dedup_bars=DEDUP_BARS):
    """与 tv.compute_signals 同逻辑，但参数可传入。"""
    bull_s = pd.Series(0, index=df.index)
    bear_s = pd.Series(0, index=df.index)
    for n in tv.INDICATORS:
        b, r = tv.find_divergences(df, df[n], order=pivot_order)
        bull_s.iloc[b] += 1
        bear_s.iloc[r] += 1
    bull = tv.dedup_signals(bull_s[bull_s >= min_count], df, gap=dedup_bars)
    bear = tv.dedup_signals(bear_s[bear_s >= min_count], df, gap=dedup_bars)
    return bull, bear


def combo_metrics(df, bull, bear, pivot_order):
    """单标的在单个参数组合下的指标。

    返回: 逐笔统计(池化多空×三档持有期, 允许重叠) + 多空半仓合并的不重叠累计收益。
    """
    tl = bt.build_trades(df, bull, side="long", entry_shift=pivot_order)
    ts = bt.build_trades(df, bear, side="short", entry_shift=pivot_order)
    cols = [f"{h}日_收益%" for h in HORIZONS]
    parts = [t_[c] for t_ in (tl, ts) if len(t_) for c in cols if c in t_.columns]
    rets = pd.concat(parts).dropna() if parts else pd.Series(dtype=float)

    comb_rets = []
    for h in HORIZONS:
        eqL = bt.equity_curve(df, bull, "long", h, entry_shift=pivot_order)
        eqS = bt.equity_curve(df, bear, "short", h, entry_shift=pivot_order)
        rL = float(eqL.iloc[-1] - 1) if not eqL.empty else 0.0
        rS = float(eqS.iloc[-1] - 1) if not eqS.empty else 0.0
        comb_rets.append((rL + rS) / 2)  # 多空各半仓合并
    return {
        "n": int(len(rets)),
        "win_rate": float((rets > 0).mean()) if len(rets) else np.nan,
        "avg_ret": float(rets.mean()) if len(rets) else np.nan,
        "med_ret": float(rets.median()) if len(rets) else np.nan,
        "comb_ret": float(np.mean(comb_rets)),
    }


def plot_heatmaps(grid, save_path):
    pivots = sorted(grid.PIVOT_ORDER.unique())
    mins = sorted(grid.MIN_COUNT.unique())

    def mat(col):
        M = np.full((len(pivots), len(mins)), np.nan)
        for _, r in grid.iterrows():
            M[pivots.index(r.PIVOT_ORDER), mins.index(r.MIN_COUNT)] = r[col]
        return M

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), dpi=110)
    specs = [
        ("平均胜率", "平均胜率", lambda v: f"{v:.1%}"),
        ("平均每笔收益%", "平均每笔收益%(池化)", lambda v: f"{v:+.2f}"),
        ("多空合并平均累计收益%", "多空合并累计收益%(不重叠)", lambda v: f"{v:+.1f}"),
    ]
    for ax, (col, title, fmt) in zip(axes, specs):
        M = mat(col)
        im = ax.imshow(M, cmap="RdYlGn", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(mins)), [f"min={m}" for m in mins])
        ax.set_yticks(range(len(pivots)), [f"pivot={p}" for p in pivots])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"PIVOT_ORDER × MIN_COUNT 参数网格（{len(TICKERS)} 标的平均，DEDUP_BARS={DEDUP_BARS}）",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_best_combo(grid, data, out_dir):
    best = grid.iloc[0]
    p, m = int(best.PIVOT_ORDER), int(best.MIN_COUNT)
    detail_dir = Path(out_dir) / f"best_pivot{p}_min{m}"
    detail_dir.mkdir(exist_ok=True)
    for t, df in data.items():
        bull, bear = signals_with(df, p, m)
        bt.build_trades(df, bull, side="long", entry_shift=p).to_csv(
            detail_dir / f"trades_{tv.safe_chart_name(t)}_long.csv", index=False)
        bt.build_trades(df, bear, side="short", entry_shift=p).to_csv(
            detail_dir / f"trades_{tv.safe_chart_name(t)}_short.csv", index=False)
        long_curves = {h: bt.equity_curve(df, bull, "long", h, entry_shift=p) for h in HORIZONS}
        short_curves = {h: bt.equity_curve(df, bear, "short", h, entry_shift=p) for h in HORIZONS}
        bt.plot_equity(df, long_curves, short_curves, t,
                       detail_dir / f"equity_{tv.safe_chart_name(t)}.png")
    return detail_dir, p, m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for t in TICKERS:
        try:
            data[t] = load_cached_only(t)
        except Exception as e:
            print(f"[skip] {t}: {e}")
    print(f"已载入 {len(data)} 个标的（全部来自本地缓存，零网络请求）\n")

    rows = []
    for p in PIVOT_RANGE:
        for m in MINCOUNT_RANGE:
            per_ticker = {}
            for t, df in data.items():
                bull, bear = signals_with(df, p, m)
                per_ticker[t] = combo_metrics(df, bull, bear, p)
            dfm = pd.DataFrame(per_ticker).T
            row = {
                "PIVOT_ORDER": p,
                "MIN_COUNT": m,
                "总交易数": int(dfm["n"].sum()),
                "平均胜率": round(float(dfm["win_rate"].mean()), 4),
                "平均每笔收益%": round(float(dfm["avg_ret"].mean()), 3),
                "中位收益%": round(float(dfm["med_ret"].mean()), 3),
                "多空合并平均累计收益%": round(float(dfm["comb_ret"].mean()) * 100, 2),
                "盈利标的占比": round(float((dfm["comb_ret"] > 0).mean()), 3),
            }
            rows.append(row)
            print(f"pivot={p} min_count={m} | 交易 {row['总交易数']:>4} | "
                  f"胜率 {row['平均胜率']:.1%} | 均笔 {row['平均每笔收益%']:+.3f}% | "
                  f"合并累计 {row['多空合并平均累计收益%']:+.2f}% | "
                  f"盈利标的 {row['盈利标的占比']:.0%}")

    grid = pd.DataFrame(rows).sort_values("多空合并平均累计收益%", ascending=False).reset_index(drop=True)
    grid.to_csv(OUT_DIR / "grid_summary.csv", index=False)
    plot_heatmaps(grid, OUT_DIR / "grid_heatmap.png")

    detail_dir, p, m = save_best_combo(grid, data, OUT_DIR)
    print(f"\n最优组合: PIVOT_ORDER={p}, MIN_COUNT={m}")
    print(f"明细目录: {detail_dir}")
    print(f"汇总表:   {OUT_DIR / 'grid_summary.csv'}")
    print(f"热图:     {OUT_DIR / 'grid_heatmap.png'}")
    return grid


if __name__ == "__main__":
    main()
