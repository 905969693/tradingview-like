"""思路 1：GLD/SLV 比例背离的价差试验。

只读上层 tradingview_realization_v2 和 cache，结果写在 experiments/gld_slv/。
参数在看收益之前定死，不搜索。

比例用两条收盘价之比，拐点打在收盘价上。指标只留 RSI、MACD 柱、动量、CCI。
共振主门槛是 2：四个指标里取 3 会比成熟系统的「8 取 3」更严。共振 3 只作一张对照表。
进场 = 信号 bar + pivot_order 的收盘。收益是波动率配平的价差，不是 GLD 本身。
权重在进场日用过去 60 日波动率定死：波动大的那条腿少配。
反转：底背离做多比例，顶背离做空比例。
顺势：沿用均线滤波，均线上方只留底背离，下方只留顶背离。
基准：比例收盘在 MA200 上方就持有配平后的多头价差，下方空仓。
样本外从 2024-01-01 起，按进场日切。半衰期用 2024 之前的对数比例估计，用来拉开持有期。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tradingview_realization_v2 as tv  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "gld_slv"
PERIOD = "10y"
PIVOT = 3
DEDUP = 10
MA_N = 200
VOL_N = 60
PRIMARY_MIN = 2
ROBUST_MIN = 3
TEST_START = pd.Timestamp("2024-01-01")
INDICATORS = ("RSI", "MACD_H", "MOM", "CCI")
FOCUS_SHORT = 20


def read_cached(ticker, period=PERIOD):
    path = tv.CACHE_DIR / f"{tv.safe_chart_name(ticker)}_{period}.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    out = tv._to_ohlcv(frame)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def half_life(log_px):
    """Δy = a + b y_lag。b < 0 时半衰期 = -ln2 / b，单位是交易日。"""
    y = pd.Series(log_px).dropna()
    lag = y.shift(1)
    dy = y.diff()
    valid = pd.concat([dy, lag], axis=1).dropna()
    valid.columns = ["dy", "lag"]
    design = np.column_stack([np.ones(len(valid)), valid["lag"].to_numpy()])
    coef, _, _, _ = np.linalg.lstsq(design, valid["dy"].to_numpy(), rcond=None)
    slope = float(coef[1])
    if slope >= 0 or not np.isfinite(slope):
        return np.inf, slope, len(valid)
    return float(-np.log(2) / slope), slope, len(valid)


def choose_horizons(train_hl):
    grid = [5, 10, 20, 60, 120, 180, 252]
    if np.isfinite(train_hl):
        for mult in (1.0, 2.0):
            horizon = int(np.clip(round(train_hl * mult), 5, 252))
            if all(abs(horizon - item) > 10 for item in grid):
                grid.append(horizon)
    return tuple(sorted(set(grid)))


def focus_horizons(horizons, train_hl):
    if np.isfinite(train_hl):
        matched = min(horizons, key=lambda item: abs(item - train_hl))
    else:
        matched = 120 if 120 in horizons else max(horizons)
    if matched == FOCUS_SHORT:
        other = 120 if 120 in horizons else max(h for h in horizons if h != FOCUS_SHORT)
        return (FOCUS_SHORT, other)
    return (FOCUS_SHORT, matched)


def add_ratio_indicators(df):
    out = df.copy()
    close = out["Close"]
    out["RSI"] = tv.rsi(close)
    out["MACD_H"] = tv.macd(close)[2]
    out["MOM"] = tv.mom(close)
    out["CCI"] = tv.cci(close, close, close)
    return out


def _drop_warmup(sig, ma):
    if len(sig) == 0:
        return sig
    alive = ma.reindex(sig.index).notna()
    return sig[alive.to_numpy()]


def make_signals(df, min_count):
    bull_score = pd.Series(0, index=df.index)
    bear_score = pd.Series(0, index=df.index)
    for name in INDICATORS:
        bull_idx, bear_idx = tv.find_divergences(df, df[name], order=PIVOT)
        bull_score.iloc[bull_idx] += 1
        bear_score.iloc[bear_idx] += 1
    bull = tv.dedup_signals(bull_score[bull_score >= min_count], df, gap=DEDUP)
    bear = tv.dedup_signals(bear_score[bear_score >= min_count], df, gap=DEDUP)
    ma = df["Close"].rolling(MA_N).mean()
    bull = _drop_warmup(bull, ma)
    bear = _drop_warmup(bear, ma)
    aligned_bull, aligned_bear = tv.apply_trend_filter(
        df, bull, bear, ma_n=MA_N, mode="aligned", adx_th=None
    )
    return {
        "reversal": (bull, bear),
        "aligned": (aligned_bull, aligned_bear),
    }


def hold_returns(horizon, weight_g, weight_s, gold, silver):
    n = len(gold)
    out = np.full(n, np.nan)
    if horizon <= 0 or horizon >= n:
        return out
    gold_ret = gold[horizon:] / gold[:-horizon] - 1.0
    silver_ret = silver[horizon:] / silver[:-horizon] - 1.0
    out[:-horizon] = weight_g[:-horizon] * gold_ret - weight_s[:-horizon] * silver_ret
    return out


def describe(values):
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(series))
    if n == 0:
        return {"交易数": 0, "胜率": np.nan, "平均收益%": np.nan, "中位收益%": np.nan}
    return {
        "交易数": n,
        "胜率": round(float((series > 0).mean()), 4),
        "平均收益%": round(float(series.mean()) * 100, 3),
        "中位收益%": round(float(series.median()) * 100, 3),
    }


def slice_mask(dates, name):
    dates = pd.DatetimeIndex(dates)
    if name == "all":
        return np.ones(len(dates), dtype=bool)
    if name == "train":
        return dates < TEST_START
    if name == "test":
        return dates >= TEST_START
    raise KeyError(name)


def signal_rows(label, side, entry_locs, spread_by_h, index, min_count):
    rows = []
    if len(entry_locs) == 0:
        return rows
    entry_locs = np.asarray(entry_locs, dtype=int)
    entry_dates = index[entry_locs]
    for horizon, spread in spread_by_h.items():
        raw = spread[entry_locs]
        if side == "short":
            raw = -raw
        for slice_name in ("all", "train", "test"):
            chosen = raw[slice_mask(entry_dates, slice_name)]
            row = describe(chosen)
            row.update({
                "共振门槛": min_count,
                "书": label,
                "方向": "多比例" if side == "long" else "空比例",
                "持有期": horizon,
                "样本": slice_name,
            })
            rows.append(row)
    return rows


def control_rows(name, entry_locs, spread_by_h, index):
    rows = []
    entry_locs = np.asarray(entry_locs, dtype=int)
    entry_dates = index[entry_locs]
    for horizon, spread in spread_by_h.items():
        raw = spread[entry_locs]
        for slice_name in ("all", "train", "test"):
            chosen = raw[slice_mask(entry_dates, slice_name)]
            row = describe(chosen)
            row.update({
                "共振门槛": 0,
                "书": name,
                "方向": "多比例",
                "持有期": horizon,
                "样本": slice_name,
            })
            rows.append(row)
    return rows


def book_trades(bull, bear, spread_by_h, index):
    """同一本账上多空互斥。同一进场日两边都有信号则跳过。"""
    by_entry = {}
    for side, sig in (("long", bull), ("short", bear)):
        for stamp in sig.index:
            xi = int(index.get_loc(stamp))
            ei = xi + PIVOT
            by_entry.setdefault(ei, []).append((side, stamp, xi))
    taken = {horizon: [] for horizon in spread_by_h}
    for horizon, spread in spread_by_h.items():
        cursor = 0
        for ei in sorted(by_entry):
            events = by_entry[ei]
            sides = {item[0] for item in events}
            if len(sides) != 1 or ei < cursor:
                continue
            ej = ei + horizon
            if ej >= len(index) or not np.isfinite(spread[ei]):
                continue
            side = events[0][0]
            ret = float(spread[ei] if side == "long" else -spread[ei])
            taken[horizon].append({
                "信号日期": pd.Timestamp(events[0][1]).date().isoformat(),
                "进场日期": pd.Timestamp(index[ei]).date().isoformat(),
                "出场日期": pd.Timestamp(index[ej]).date().isoformat(),
                "进场位置": ei,
                "出场位置": ej,
                "方向": "多比例" if side == "long" else "空比例",
                "持有期": horizon,
                "收益": ret,
            })
            cursor = ej + 1
    return taken


def equity_from_exits(index, trades):
    wealth = np.ones(len(index), dtype=float)
    level = 1.0
    for trade in trades:
        level *= 1.0 + trade["收益"]
        wealth[trade["出场位置"]:] = level
    return pd.Series(wealth, index=index)


def max_drawdown(equity):
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def window_of(index, slice_name):
    if slice_name == "train":
        return index[index < TEST_START]
    if slice_name == "test":
        return index[index >= TEST_START]
    return index


def book_summary(name, min_count, horizon, trades, index, slice_name):
    if slice_name == "train":
        keep = [t for t in trades if pd.Timestamp(t["进场日期"]) < TEST_START]
    elif slice_name == "test":
        keep = [t for t in trades if pd.Timestamp(t["进场日期"]) >= TEST_START]
    else:
        keep = list(trades)
    window = window_of(index, slice_name)
    wealth = pd.Series(1.0, index=window)
    level = 1.0
    for trade in keep:
        level *= 1.0 + trade["收益"]
        exit_ts = pd.Timestamp(trade["出场日期"])
        if len(window) == 0:
            continue
        if exit_ts in window:
            loc = int(window.get_loc(exit_ts))
        elif exit_ts > window[-1]:
            # 持有期跨过样本切分时，这笔收益仍算在进场所属样本。
            loc = len(window) - 1
        else:
            continue
        wealth.iloc[loc:] = level
    days = int(sum(t["持有期"] for t in keep))
    n_days = int(len(window))
    terminal = float(wealth.iloc[-1] - 1.0) if len(wealth) else np.nan
    return {
        "共振门槛": min_count,
        "书": name,
        "持有期": horizon,
        "样本": slice_name,
        "成交笔数": len(keep),
        "在场占比": round(days / n_days, 4) if n_days else np.nan,
        "累计收益%": round(terminal * 100, 2) if np.isfinite(terminal) else np.nan,
        "最大回撤%": round(max_drawdown(wealth) * 100, 2) if len(wealth) else np.nan,
        "笔均收益%": round(float(np.mean([t["收益"] for t in keep])) * 100, 3) if keep else np.nan,
    }


def baseline_curve(active, spread_daily, index):
    """收盘决定、下一根才赚到。active 和日收益都已经对齐到『这一根的收益』。"""
    earned = spread_daily.where(active, 0.0).fillna(0.0)
    return (1.0 + earned).cumprod(), earned


def baseline_summary(name, curve, active, live, index, slice_name):
    window = window_of(index, slice_name)
    piece = curve.reindex(window)
    if len(piece):
        piece = piece / piece.iloc[0]
    on = active.reindex(window).fillna(False)
    live_window = live.reindex(window).fillna(False)
    occupancy_base = int(live_window.sum())
    terminal = float(piece.iloc[-1] - 1.0) if len(piece) else np.nan
    return {
        "共振门槛": 0,
        "书": name,
        "持有期": 0,
        "样本": slice_name,
        "成交笔数": np.nan,
        "在场占比": round(float(on.sum()) / occupancy_base, 4) if occupancy_base else np.nan,
        "累计收益%": round(terminal * 100, 2) if np.isfinite(terminal) else np.nan,
        "最大回撤%": round(max_drawdown(piece) * 100, 2) if len(piece) else np.nan,
        "笔均收益%": np.nan,
    }


def plot_ratio(df, ma, raw, aligned, save_path):
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=110)
    ax.plot(df.index, df["Close"], color="#222222", lw=0.9, label="GLD/SLV 收盘比")
    ax.plot(df.index, ma, color="#c4a35a", lw=1.0, ls="--", label=f"MA{MA_N}")
    specs = (
        (raw[0], aligned[0], "^", "#1b9e77", "底背离"),
        (raw[1], aligned[1], "v", "#7570b3", "顶背离"),
    )
    for full, kept, marker, color, label in specs:
        kept_set = set(kept.index)
        rejected = full.index.difference(kept.index)
        if len(kept_set):
            ax.scatter(
                kept.index, df.loc[kept.index, "Close"],
                marker=marker, s=28, color=color, zorder=3, label=f"{label}·顺势保留",
            )
        if len(rejected):
            ax.scatter(
                rejected, df.loc[rejected, "Close"],
                marker=marker, s=28, facecolors="none", edgecolors=color,
                linewidths=0.8, zorder=3, label=f"{label}·滤波剔除",
            )
    ax.set_title(
        f"GLD/SLV 收盘比（pivot {PIVOT}，共振 ≥{PRIMARY_MIN}，"
        f"指标 {'/'.join(INDICATORS)}）"
    )
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_ylabel("GLD 收盘 / SLV 收盘")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_equity(curves, titles, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), dpi=110, sharex=False)
    colors = {
        "均线上方持有": "#4d4d4d",
        "反转": "#1b9e77",
        "顺势": "#7570b3",
    }
    for ax, title in zip(axes.ravel(), titles):
        drawn = curves[title]
        for name, series in drawn.items():
            style = "--" if name == "均线上方持有" else "-"
            ax.plot(series.index, series.values, lw=1.2, ls=style, color=colors[name], label=name)
        ax.axhline(1.0, color="black", lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("价差净值：波动率配平，确认后收盘进场，持仓期内的新信号跳过", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def slice_curve(equity, index, slice_name):
    if slice_name == "train":
        window = index[index < TEST_START]
    elif slice_name == "test":
        window = index[index >= TEST_START]
    else:
        window = index
    piece = equity.reindex(window)
    if len(piece):
        piece = piece / piece.iloc[0]
    return piece


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_df = read_cached("GLD")
    silver_df = read_cached("SLV")
    shared = gold_df.index.intersection(silver_df.index).sort_values()
    gold_px = gold_df.loc[shared, "Close"].to_numpy(dtype=float)
    silver_px = silver_df.loc[shared, "Close"].to_numpy(dtype=float)
    ratio_px = gold_px / silver_px
    ratio = pd.DataFrame(
        {"Open": ratio_px, "High": ratio_px, "Low": ratio_px, "Close": ratio_px},
        index=shared,
    )
    ratio = add_ratio_indicators(ratio)
    log_px = np.log(ratio["Close"])
    train_log = log_px[log_px.index < TEST_START]
    test_log = log_px[log_px.index >= TEST_START]
    hl_train, slope_train, n_train = half_life(train_log)
    hl_test, slope_test, n_test = half_life(test_log)
    hl_all, slope_all, n_all = half_life(log_px)
    horizons = choose_horizons(hl_train)
    focuses = focus_horizons(horizons, hl_train)

    gold_ret = pd.Series(gold_px, index=shared).pct_change()
    silver_ret = pd.Series(silver_px, index=shared).pct_change()
    vol_g = gold_ret.rolling(VOL_N).std()
    vol_s = silver_ret.rolling(VOL_N).std()
    weight_g = (vol_s / (vol_g + vol_s)).to_numpy(dtype=float)
    weight_s = (vol_g / (vol_g + vol_s)).to_numpy(dtype=float)
    equal = np.full(len(shared), 0.5)

    spread_by_h = {h: hold_returns(h, weight_g, weight_s, gold_px, silver_px) for h in horizons}
    equal_20 = hold_returns(FOCUS_SHORT, equal, equal, gold_px, silver_px)
    ma = ratio["Close"].rolling(MA_N).mean()
    above = (ratio["Close"] > ma) & ma.notna() & np.isfinite(weight_g)
    below = (ratio["Close"] < ma) & ma.notna() & np.isfinite(weight_g)
    any_day = ma.notna() & np.isfinite(weight_g)

    signal_stats = []
    books = []
    trade_frames = []
    primary = make_signals(ratio, PRIMARY_MIN)
    robust = make_signals(ratio, ROBUST_MIN)
    packs = ((PRIMARY_MIN, primary), (ROBUST_MIN, robust))
    for min_count, pack in packs:
        for book_name, (bull, bear) in pack.items():
            bull_entries = [int(shared.get_loc(stamp)) + PIVOT for stamp in bull.index]
            bear_entries = [int(shared.get_loc(stamp)) + PIVOT for stamp in bear.index]
            bull_entries = [i for i in bull_entries if i < len(shared)]
            bear_entries = [i for i in bear_entries if i < len(shared)]
            signal_stats += signal_rows(
                book_name, "long", bull_entries, spread_by_h, shared, min_count
            )
            signal_stats += signal_rows(
                book_name, "short", bear_entries, spread_by_h, shared, min_count
            )
            if min_count != PRIMARY_MIN:
                continue
            taken = book_trades(bull, bear, spread_by_h, shared)
            for horizon, trades in taken.items():
                for slice_name in ("all", "train", "test"):
                    books.append(book_summary(
                        book_name, min_count, horizon, trades, shared, slice_name
                    ))
                frame = pd.DataFrame(trades)
                if len(frame):
                    frame.insert(0, "书", book_name)
                    trade_frames.append(frame)

    signal_stats += control_rows(
        "任意一日做多", np.flatnonzero(any_day.to_numpy()), spread_by_h, shared
    )
    signal_stats += control_rows(
        "均线上方做多", np.flatnonzero(above.to_numpy()), spread_by_h, shared
    )
    signal_stats += control_rows(
        "均线下方做多", np.flatnonzero(below.to_numpy()), spread_by_h, shared
    )

    # 基准：t 日收盘已知均线和权重，赚 t→t+1 的价差。持有期列 0 表示连续持有，不是 N 日回合。
    daily = pd.Series(weight_g, index=shared).shift(1) * gold_ret - pd.Series(weight_s, index=shared).shift(1) * silver_ret
    active = above.shift(1).fillna(False) & daily.notna()
    base_equity, _earned = baseline_curve(active, daily, shared)
    live = ma.notna().shift(1).fillna(False)
    for slice_name in ("all", "train", "test"):
        books.append(baseline_summary("均线上方持有", base_equity, active, live, shared, slice_name))

    # 等金额对照只看主门槛、全样本、20 日，用来确认银的高波动有没有主导结果。
    equal_entries = [
        int(shared.get_loc(stamp)) + PIVOT for stamp in primary["reversal"][0].index
    ]
    equal_entries = [i for i in equal_entries if i < len(shared)]
    equal_vals = equal_20[equal_entries]
    equal_stats = describe(equal_vals)
    vol_stats = describe(spread_by_h[FOCUS_SHORT][equal_entries])

    stats_df = pd.DataFrame(signal_stats)
    books_df = pd.DataFrame(books)
    order_cols = ["共振门槛", "书", "方向", "持有期", "样本", "交易数", "胜率", "平均收益%", "中位收益%"]
    stats_df = stats_df[order_cols]
    stats_df.to_csv(OUT_DIR / "signal_stats.csv", index=False)
    books_df.to_csv(OUT_DIR / "books.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT_DIR / "trades.csv", index=False)

    hl_df = pd.DataFrame([
        {"样本": "train_<2024", "半衰期_交易日": hl_train, "斜率b": slope_train, "观测数": n_train},
        {"样本": "test_>=2024", "半衰期_交易日": hl_test, "斜率b": slope_test, "观测数": n_test},
        {"样本": "all", "半衰期_交易日": hl_all, "斜率b": slope_all, "观测数": n_all},
    ])
    hl_df.to_csv(OUT_DIR / "half_life.csv", index=False)

    vol_ratio = (vol_s / vol_g).replace([np.inf, -np.inf], np.nan).dropna()
    meta = {
        "区间": f"{shared[0].date()} ~ {shared[-1].date()}",
        "GLD根数": int(len(gold_df)),
        "SLV根数": int(len(silver_df)),
        "交集根数": int(len(shared)),
        "银波动/金波动_中位数": round(float(vol_ratio.median()), 3),
        "训练半衰期_交易日": None if not np.isfinite(hl_train) else round(hl_train, 1),
        "持有期": list(horizons),
        "图上持有期": list(focuses),
        "20日底背离_等金额平均收益%": equal_stats["平均收益%"],
        "20日底背离_配平平均收益%": vol_stats["平均收益%"],
        "成本": 0,
        "样本外起点": TEST_START.date().isoformat(),
    }
    pd.Series(meta).to_csv(OUT_DIR / "meta.csv", header=["值"])

    plot_ratio(
        ratio, ma, primary["reversal"], primary["aligned"], OUT_DIR / "ratio.png"
    )

    curves = {}
    titles = []
    for slice_name, slice_label in (("all", "全样本"), ("test", "2024年起")):
        for horizon in focuses:
            title = f"{slice_label} · 背离持有 {horizon} 日"
            titles.append(title)
            panel = {}
            panel["均线上方持有"] = slice_curve(base_equity, shared, slice_name)
            for book_name, label in (("reversal", "反转"), ("aligned", "顺势")):
                trades = book_trades(
                    *primary[book_name], {horizon: spread_by_h[horizon]}, shared
                )[horizon]
                if slice_name == "train":
                    trades = [t for t in trades if pd.Timestamp(t["进场日期"]) < TEST_START]
                elif slice_name == "test":
                    trades = [t for t in trades if pd.Timestamp(t["进场日期"]) >= TEST_START]
                panel[label] = slice_curve(equity_from_exits(shared, trades), shared, slice_name)
            curves[title] = panel
    plot_equity(curves, titles, OUT_DIR / "equity.png")

    show = stats_df[
        (stats_df["共振门槛"].isin([0, PRIMARY_MIN]))
        & (stats_df["持有期"].isin(focuses))
        & (stats_df["样本"].isin(["all", "test"]))
    ]
    print(hl_df.to_string(index=False))
    print("horizons", horizons, "focus", focuses)
    print(pd.Series(meta).to_string())
    print(show.to_string(index=False))
    print(books_df[books_df["持有期"].isin([0, *focuses])].to_string(index=False))


if __name__ == "__main__":
    main()
