"""
对比固定 10 日分数和 3–40 根曲线中位数。

参数全标的统一：
pivot 3、共振 3、去重 10、MA200 顺势。不用各标的已经优化过的预设。
只用缓存里有成交量的标的。
入场是信号后第 pivot 根收盘。无条件对照与 unconditional_returns 相同。

尖峰是该侧超额平均最高的持有根数，并列时取更短的那根。它不参与分数。
运行：python backtest_horizon_curve.py
"""
from datetime import date

import numpy as np
import pandas as pd

import tradingview_realization_v2 as tv

PERIOD = "10y"
PIVOT_ORDER = tv.PIVOT_ORDER
MIN_COUNT = tv.MIN_COUNT
DEDUP_BARS = tv.DEDUP_BARS
MA_N = tv.TREND_MA_N
HORIZON = 10
OUT_DIR = tv.SCRIPT_DIR / "backtest" / f"horizon_curve_{date.today().isoformat()}"


def load_cached(ticker, period=PERIOD):
    path = tv.CACHE_DIR / f"{tv.safe_chart_name(ticker)}_{period}.csv"
    return tv._to_ohlcv(pd.read_csv(path, index_col=0, parse_dates=True))


def cached_tickers(period=PERIOD):
    names = []
    for path in sorted(tv.CACHE_DIR.glob(f"*_{period}.csv")):
        stem = path.name[: -len(f"_{period}.csv")]
        if stem.startswith("dual_tf_backtest"):
            continue
        names.append(stem)
    return names


def side_10d(df, sig, short):
    stats = tv.signal_stats(
        df, sig, horizons=(HORIZON,), short=short, entry_shift=PIVOT_ORDER,
    )
    base = tv.unconditional_stats(df, horizons=(HORIZON,), short=short)
    n = int(stats.iloc[0]["信号数"])
    win = float(stats.iloc[0]["胜率"])
    base_win = float(base.iloc[0]["无条件胜率"])
    return n, tv.edge_score(win, base_win, n)


def main():
    rows = []
    skipped = []
    span = tv.POSTERIOR_SPAN
    for ticker in cached_tickers():
        df = load_cached(ticker)
        vol = df.Volume.to_numpy(dtype=float)
        if not np.isfinite(vol).any() or np.nansum(vol) <= 0:
            skipped.append(ticker)
            continue
        df = tv.add_indicators(df)
        bull, bear = tv.raw_deduped_signals(
            df, pivot_order=PIVOT_ORDER, min_count=MIN_COUNT, dedup_bars=DEDUP_BARS,
        )
        bull, bear = tv.apply_trend_filter(
            df, bull, bear, ma_n=MA_N, mode="aligned", adx_th=None,
        )
        long_n, long_score = side_10d(df, bull, False)
        short_n, short_score = side_10d(df, bear, True)
        score_10 = tv.combine_edge_scores(long_score, long_n, short_score, short_n, False)
        closes = df.Close.to_numpy(dtype=float)
        base_long = tv.unconditional_horizon_baselines(closes, span, short=False)
        base_short = tv.unconditional_horizon_baselines(closes, span, short=True)
        long_curve = tv.hold_edge_curve(
            df, bull, short=False, entry_shift=PIVOT_ORDER, horizons=span,
            baselines=base_long,
        )
        short_curve = tv.hold_edge_curve(
            df, bear, short=True, entry_shift=PIVOT_ORDER, horizons=span,
            baselines=base_short,
        )
        rows.append({
            "标的": ticker,
            "十日分数": score_10,
            "曲线中位数": tv.book_curve_median(long_curve, short_curve, False),
            "底尖峰": tv.curve_peak_hold(long_curve),
            "顶尖峰": tv.curve_peak_hold(short_curve),
            "底笔数": long_n,
            "顶笔数": short_n,
        })
        print(
            f"{ticker}  10日 {_fmt(score_10)}  曲线 {_fmt(rows[-1]['曲线中位数'])}"
            f"  底尖峰 {rows[-1]['底尖峰']}  顶尖峰 {rows[-1]['顶尖峰']}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    if out.empty:
        print("没有可计算的标的")
        return
    out["十日名次"] = out["十日分数"].rank(ascending=False, method="min").astype(int)
    out["曲线名次"] = out["曲线中位数"].rank(ascending=False, method="min").astype(int)
    out["名次差"] = out["十日名次"] - out["曲线名次"]
    out = out.sort_values(["曲线中位数", "标的"], ascending=[False, True])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "scores.csv", index=False)
    meta = pd.DataFrame([{
        "标的数": int(len(out)),
        "跳过无成交量": ",".join(skipped),
        "参数": f"pivot={PIVOT_ORDER} min={MIN_COUNT} dedup={DEDUP_BARS} MA={MA_N} 顺势",
        "持有期": f"{span[0]}–{span[-1]}",
        "分数相关": float(out["十日分数"].corr(out["曲线中位数"], method="spearman")),
    }])
    meta.to_csv(OUT_DIR / "meta.csv", index=False)
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(meta.to_string(index=False))
    print(out.to_string(index=False))
    print(f"写出 {OUT_DIR}")


def _fmt(score):
    if score is None or not np.isfinite(score):
        return "—"
    return f"{float(score) * 100:+.1f}"


if __name__ == "__main__":
    main()
