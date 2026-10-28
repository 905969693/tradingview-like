"""
背离信号回测（基于 tradingview_realization_v2 模块）

交易规则：
- 底背离共振数 >= MIN_COUNT：pivot 需右侧 PIVOT_ORDER 根K线才确认，因此
  入场 = 信号 bar + ENTRY_SHIFT 那根K线的收盘价（无未来函数的最早可成交点），
  持有 h 根K线后按收盘价平仓（h ∈ HORIZONS）。
- 顶背离同理，方向做空（收益取负）。
- 同一方向在持仓期内再出信号 → 跳过（不加仓）；多/空相互独立。
- 交易成本 COST_BPS 为单边 bps，往返一笔记 2×。
- USE_TREND_FILTER=True 时先对信号做顺势滤波（tv.apply_trend_filter，
  MA_N=200：均线上方只留底背离、下方只留顶背离）。

输出（backtest/<日期>/）：
- trades_{ticker}.csv      每笔信号的交易明细（5/10/20 日三档收益并列）
- equity_{ticker}.png       多/空 × 三档持有期的净值曲线（虚线为买入持有基准）
- backtest_summary.csv      全部标的 × 方向 × 持有期 的汇总指标

运行：python backtest_divergence.py
"""
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tradingview_realization_v2 as tv

# ============ 0. 可调参数 ============
TICKERS = tv.TICKERS                 # 回测标的
PERIOD = tv.PERIOD                   # 数据区间（与缓存文件名一致）
PIVOT_ORDER = tv.PIVOT_ORDER
MIN_COUNT = tv.MIN_COUNT
DEDUP_BARS = tv.DEDUP_BARS
HORIZONS = tv.STATS_HORIZONS         # (5, 10, 20)
ENTRY_SHIFT = PIVOT_ORDER            # pivot 确认滞后：信号后第 order 根收盘才可入场
COST_BPS = 0.0                       # 单边交易成本（bps）
USE_TREND_FILTER = True              # 回测前是否应用顺势滤波
MA_N = 200                           # 顺势滤波均线周期（日线 200 / 周线 40）
OUT_ROOT = tv.SCRIPT_DIR / "backtest"
LONG_LABEL, SHORT_LABEL = "多(底背离)", "空(顶背离)"


# ============ 1. 交易明细 ============
def build_trades(df, sig, side, horizons=HORIZONS,
                 entry_shift=ENTRY_SHIFT, cost_bps=COST_BPS):
    """把信号序列展开成交易明细表；side='long' 正常收益，'short' 收益取反。"""
    closes = df.Close.values
    dates = df.index
    round_cost = 2 * cost_bps / 1e4
    rows = []
    for idx, v in sig.items():
        xi = df.index.get_loc(idx)
        ei = xi + entry_shift
        if ei >= len(df):
            continue  # 信号太新，确认日已超出数据
        rec = {
            "信号日期": dates[xi].date().isoformat(),
            "共振数": int(v),
            "入场日期": dates[ei].date().isoformat(),
            "入场价": round(float(closes[ei]), 4),
        }
        for h in horizons:
            rec[f"{h}日_出场日"] = pd.NaT
            rec[f"{h}日_收益%"] = np.nan
            xj = ei + h
            if xj < len(df):
                r = closes[xj] / closes[ei] - 1
                if side == "short":
                    r = -r
                rec[f"{h}日_出场日"] = dates[xj].date().isoformat()
                rec[f"{h}日_收益%"] = round((r - round_cost) * 100, 3)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(trades, side, ticker, horizons=HORIZONS):
    """逐持有期统计：胜率/均值/中位/盈亏比（基于全部信号，允许重叠）。"""
    label = LONG_LABEL if side == "long" else SHORT_LABEL
    rows = []
    for h in horizons:
        col = f"{h}日_收益%"
        rets = trades[col].dropna() if col in trades.columns else pd.Series(dtype=float)
        n = len(rets)
        wins = rets[rets > 0].sum()
        losses = -rets[rets < 0].sum()
        rows.append({
            "ticker": ticker,
            "方向": label,
            "持有期": f"{h}日",
            "交易数": n,
            "胜率": round(wins_.mean(), 4) if (wins_ := (rets > 0)).size else np.nan,
            "平均收益%": round(rets.mean(), 3) if n else np.nan,
            "中位收益%": round(rets.median(), 3) if n else np.nan,
            "盈亏比": round(wins / losses, 3) if losses > 0 else np.nan,
        })
    return rows


# ============ 2. 净值曲线（同方向不重叠、复利） ============
def equity_curve(df, sig, side, horizon, entry_shift=ENTRY_SHIFT, cost_bps=COST_BPS):
    """同方向持仓期内的新信号跳过；净值在出场日跳变，其余时间持平。"""
    closes = df.Close.values
    cost = 2 * cost_bps / 1e4
    busy_until = -1
    pts = {}
    eq = 1.0
    for idx in sig.index:
        xi = df.index.get_loc(idx)
        ei = xi + entry_shift
        xj = ei + horizon
        if ei >= len(df) or xj >= len(df) or ei < busy_until:
            continue
        r = closes[xj] / closes[ei] - 1
        if side == "short":
            r = -r
        eq *= (1 + r - cost)
        pts[df.index[xj]] = eq
        busy_until = xj + 1
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series(pts).sort_index()
    s = s.reindex(df.index, method="ffill")
    return s


def max_drawdown(s):
    if s.empty:
        return np.nan
    return round(float((s / s.cummax() - 1).min()) * 100, 2)


def plot_equity(df, long_curves, short_curves, ticker, save_path):
    n = len(HORIZONS)
    fig, axes = plt.subplots(
        2, n, figsize=(5.2 * n, 7), dpi=110, sharex=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )
    bh = df.Close / df.Close.iloc[0]
    for row, (curves, label, color) in enumerate(
        [(long_curves, LONG_LABEL, "#1b9e77"), (short_curves, SHORT_LABEL, "#d95f02")]
    ):
        for col, h in enumerate(HORIZONS):
            ax = axes[row][col]
            s = curves.get(h)
            if s is not None and not s.empty:
                ax.plot(s.index, s.values, color=color, lw=1.4, label=f"{h}日持有")
                ax.text(0.02, 0.95, f"累计 {s.iloc[-1] - 1:+.1%} | 最大回撤 {max_drawdown(s)}%",
                        transform=ax.transAxes, fontsize=8, va="top",
                        color=color)
            else:
                ax.text(0.5, 0.5, "无可交易信号", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
            ax.plot(bh.index, bh.values, ls="--", lw=0.8, color="gray", alpha=0.7,
                    label="买入持有")
            ax.axhline(1, color="k", lw=0.6)
            ax.set_title(f"{label} · 持有{h}日", fontsize=10)
            if row == 1:
                ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"{ticker} 背离信号净值（入场=信号+{ENTRY_SHIFT}根收盘价，"
        f"同向不重叠，成本 {COST_BPS}bps/边）",
        fontsize=11,
    )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ============ 3. 单标的 & 批量 ============
def backtest_ticker(ticker, out_dir, use_filter=USE_TREND_FILTER, ma_n=MA_N):
    df = tv.add_indicators(tv.load_ohlcv(ticker, period=PERIOD))
    bull_sig, bear_sig = tv.compute_signals(df)
    if use_filter:
        bull_sig, bear_sig = tv.apply_trend_filter(df, bull_sig, bear_sig,
                                                   ma_n=ma_n, mode="aligned")

    trades_long = build_trades(df, bull_sig, side="long")
    trades_short = build_trades(df, bear_sig, side="short")
    trades_long.to_csv(out_dir / f"trades_{tv.safe_chart_name(ticker)}_long.csv", index=False)
    trades_short.to_csv(out_dir / f"trades_{tv.safe_chart_name(ticker)}_short.csv", index=False)

    # 汇总（交易层统计，允许重叠）
    summary_rows = summarize(trades_long, "long", ticker)
    summary_rows += summarize(trades_short, "short", ticker)

    # 净值曲线（不重叠复利）+ 基准
    bh_ret = (df.Close.iloc[-1] / df.Close.iloc[0] - 1) * 100
    long_curves, short_curves = {}, {}
    for side, sig, curves in (("long", bull_sig, long_curves),
                              ("short", bear_sig, short_curves)):
        for h in HORIZONS:
            curves[h] = equity_curve(df, sig, side, h)

    for row in summary_rows:
        side = "long" if row["方向"] == LONG_LABEL else "short"
        curves = long_curves if side == "long" else short_curves
        h = int(row["持有期"].rstrip("日"))
        s = curves[h]
        row["累计收益%(不重叠)"] = round(float(s.iloc[-1] - 1) * 100, 2) if not s.empty else np.nan
        row["最大回撤%"] = max_drawdown(s)
        row["基准买入持有%"] = round(bh_ret, 2)
        row["样本区间"] = f"{df.index[0].date()} ~ {df.index[-1].date()}"

    eq_path = out_dir / f"equity_{tv.safe_chart_name(ticker)}.png"
    plot_equity(df, long_curves, short_curves, ticker, eq_path)

    return summary_rows, eq_path, len(bull_sig), len(bear_sig)


def run(tickers=TICKERS, out_dir=None, use_filter=USE_TREND_FILTER):
    if out_dir is None:
        suffix = "_filtered" if use_filter else ""
        out_dir = OUT_ROOT / (date.today().isoformat() + suffix)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, errors = [], []
    for ticker in tickers:
        try:
            rows, eq_path, n_bull, n_bear = backtest_ticker(ticker, out_dir, use_filter=use_filter)
            all_rows.extend(rows)
            print(f"[ok] {ticker}  底背离信号 {n_bull} 个 | 顶背离 {n_bear} 个 -> {eq_path}")
            df_show = pd.DataFrame(rows)
            print(df_show.to_string(index=False))
            print()
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            print(f"[fail] {ticker}: {e}")

    summary_path = out_dir / "backtest_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")
    if errors:
        print("Errors:", errors)
    return all_rows


if __name__ == "__main__":
    run()
