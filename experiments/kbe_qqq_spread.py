"""思路 1 用在 KBE/QQQ。规则与 experiments/gld_slv_spread.py 相同，不改那次结果。

比例 = KBE 收盘 / QQQ 收盘，拐点打在收盘价上。指标只留 RSI、MACD 柱、动量、CCI。
共振主门槛 2，共振 3 只作对照。进场 = 信号后第 3 根收盘。
底背离做多比例（多 KBE、空 QQQ），顶背离做空比例。权重用过去 60 日波动率，波动大的腿少配。
顺势沿用均线滤波。基准是比例在 MA200 上方就持有配平多头价差。
样本外从 2024-01-01 起。半衰期用 2024 之前的对数比例估计。成本 0。
同时记下配平价差里有多少来自净多 KBE，多少来自 KBE 相对 QQQ。
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gld_slv_spread as gs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "kbe_qqq"
NUMERATOR = "KBE"
DENOMINATOR = "QQQ"


def plot_ratio(df, ma, raw, aligned, save_path):
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=110)
    ax.plot(df.index, df["Close"], color="#222222", lw=0.9, label="KBE/QQQ 收盘比")
    ax.plot(df.index, ma, color="#c4a35a", lw=1.0, ls="--", label=f"MA{gs.MA_N}")
    specs = (
        (raw[0], aligned[0], "^", "#1b9e77", "底背离"),
        (raw[1], aligned[1], "v", "#7570b3", "顶背离"),
    )
    for full, kept, marker, color, label in specs:
        rejected = full.index.difference(kept.index)
        if len(kept):
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
        f"KBE/QQQ 收盘比（pivot {gs.PIVOT}，共振 ≥{gs.PRIMARY_MIN}，"
        f"指标 {'/'.join(gs.INDICATORS)}）"
    )
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_ylabel("KBE 收盘 / QQQ 收盘")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def compound_from(ret, start):
    equity = (1.0 + ret.fillna(0.0)).cumprod()
    base = equity.shift(1).loc[start]
    return equity.loc[start:] / base


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_df = gs.read_cached(NUMERATOR)
    den_df = gs.read_cached(DENOMINATOR)
    shared = num_df.index.intersection(den_df.index).sort_values()
    num_px = num_df.loc[shared, "Close"].to_numpy(dtype=float)
    den_px = den_df.loc[shared, "Close"].to_numpy(dtype=float)
    ratio_px = num_px / den_px
    ratio = pd.DataFrame(
        {"Open": ratio_px, "High": ratio_px, "Low": ratio_px, "Close": ratio_px},
        index=shared,
    )
    ratio = gs.add_ratio_indicators(ratio)
    log_px = np.log(ratio["Close"])
    hl_train, slope_train, n_train = gs.half_life(log_px[log_px.index < gs.TEST_START])
    hl_test, slope_test, n_test = gs.half_life(log_px[log_px.index >= gs.TEST_START])
    hl_all, slope_all, n_all = gs.half_life(log_px)
    horizons = gs.choose_horizons(hl_train)
    focuses = gs.focus_horizons(horizons, hl_train)

    num_ret = pd.Series(num_px, index=shared).pct_change()
    den_ret = pd.Series(den_px, index=shared).pct_change()
    vol_n = num_ret.rolling(gs.VOL_N).std()
    vol_d = den_ret.rolling(gs.VOL_N).std()
    weight_n = (vol_d / (vol_n + vol_d)).to_numpy(dtype=float)
    weight_d = (vol_n / (vol_n + vol_d)).to_numpy(dtype=float)
    spread_by_h = {
        h: gs.hold_returns(h, weight_n, weight_d, num_px, den_px) for h in horizons
    }
    ma = ratio["Close"].rolling(gs.MA_N).mean()
    above = (ratio["Close"] > ma) & ma.notna() & np.isfinite(weight_n)
    below = (ratio["Close"] < ma) & ma.notna() & np.isfinite(weight_n)
    any_day = ma.notna() & np.isfinite(weight_n)

    signal_stats = []
    books = []
    trade_frames = []
    primary = gs.make_signals(ratio, gs.PRIMARY_MIN)
    robust = gs.make_signals(ratio, gs.ROBUST_MIN)
    for min_count, pack in ((gs.PRIMARY_MIN, primary), (gs.ROBUST_MIN, robust)):
        for book_name, (bull, bear) in pack.items():
            bull_entries = [int(shared.get_loc(stamp)) + gs.PIVOT for stamp in bull.index]
            bear_entries = [int(shared.get_loc(stamp)) + gs.PIVOT for stamp in bear.index]
            bull_entries = [i for i in bull_entries if i < len(shared)]
            bear_entries = [i for i in bear_entries if i < len(shared)]
            signal_stats += gs.signal_rows(
                book_name, "long", bull_entries, spread_by_h, shared, min_count
            )
            signal_stats += gs.signal_rows(
                book_name, "short", bear_entries, spread_by_h, shared, min_count
            )
            if min_count != gs.PRIMARY_MIN:
                continue
            taken = gs.book_trades(bull, bear, spread_by_h, shared)
            for horizon, trades in taken.items():
                for slice_name in ("all", "train", "test"):
                    books.append(gs.book_summary(
                        book_name, min_count, horizon, trades, shared, slice_name
                    ))
                frame = pd.DataFrame(trades)
                if len(frame):
                    frame.insert(0, "书", book_name)
                    trade_frames.append(frame)

    signal_stats += gs.control_rows(
        "任意一日做多", np.flatnonzero(any_day.to_numpy()), spread_by_h, shared
    )
    signal_stats += gs.control_rows(
        "均线上方做多", np.flatnonzero(above.to_numpy()), spread_by_h, shared
    )
    signal_stats += gs.control_rows(
        "均线下方做多", np.flatnonzero(below.to_numpy()), spread_by_h, shared
    )

    weight_n_s = pd.Series(weight_n, index=shared)
    weight_d_s = pd.Series(weight_d, index=shared)
    daily = weight_n_s.shift(1) * num_ret - weight_d_s.shift(1) * den_ret
    active = above.shift(1).fillna(False) & daily.notna()
    base_equity, earned = gs.baseline_curve(active, daily, shared)
    live = ma.notna().shift(1).fillna(False)
    for slice_name in ("all", "train", "test"):
        books.append(gs.baseline_summary(
            "均线上方持有", base_equity, active, live, shared, slice_name
        ))

    net = (weight_n_s - weight_d_s).shift(1)
    bias = (net * (num_ret + den_ret) / 2).where(active, 0.0).fillna(0.0)
    relative = (0.5 * (num_ret - den_ret)).where(active, 0.0).fillna(0.0)
    start = live[live].index[0]
    kbe_timed = compound_from(num_ret.where(active, 0.0).fillna(0.0), start)
    kbe_always = compound_from(num_ret.where(live, 0.0).fillna(0.0), start)
    qqq_always = compound_from(den_ret.where(live, 0.0).fillna(0.0), start)

    def end_stats(curve):
        dd = float((curve / curve.cummax() - 1.0).min())
        return {
            "期末净值": round(float(curve.iloc[-1]), 4),
            "累计收益%": round(float(curve.iloc[-1] - 1.0) * 100, 2),
            "最大回撤%": round(dd * 100, 2),
        }

    attribution = pd.DataFrame([
        {
            "样本": "all",
            "配平加总%": round(float(earned.sum()) * 100, 2),
            "净多KBE加总%": round(float(bias.sum()) * 100, 2),
            "相对涨跌加总%": round(float(relative.sum()) * 100, 2),
            "配平复利%": round(float((1.0 + earned).prod() - 1.0) * 100, 2),
            "持仓日多头权重": round(float(weight_n_s.shift(1)[active].mean()), 4),
            "持仓日空头权重": round(float(weight_d_s.shift(1)[active].mean()), 4),
        },
        {
            "样本": "test",
            "配平加总%": round(float(earned[earned.index >= gs.TEST_START].sum()) * 100, 2),
            "净多KBE加总%": round(float(bias[bias.index >= gs.TEST_START].sum()) * 100, 2),
            "相对涨跌加总%": round(float(relative[relative.index >= gs.TEST_START].sum()) * 100, 2),
            "配平复利%": round(float((1.0 + earned[earned.index >= gs.TEST_START]).prod() - 1.0) * 100, 2),
            "持仓日多头权重": round(float(weight_n_s.shift(1)[active & (active.index >= gs.TEST_START)].mean()), 4),
            "持仓日空头权重": round(float(weight_d_s.shift(1)[active & (active.index >= gs.TEST_START)].mean()), 4),
        },
    ])
    legs = pd.DataFrame([
        {"规则": "均线上方只做多KBE", **end_stats(kbe_timed), "在场占比": round(float(active.loc[start:].mean()), 4)},
        {"规则": "一直持有KBE", **end_stats(kbe_always), "在场占比": 1.0},
        {"规则": "一直持有QQQ", **end_stats(qqq_always), "在场占比": 1.0},
    ])
    attribution.to_csv(OUT_DIR / "attribution.csv", index=False)
    legs.to_csv(OUT_DIR / "legs.csv", index=False)

    stats_df = pd.DataFrame(signal_stats)[
        ["共振门槛", "书", "方向", "持有期", "样本", "交易数", "胜率", "平均收益%", "中位收益%"]
    ]
    books_df = pd.DataFrame(books)
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
    vol_ratio = (vol_d / vol_n).replace([np.inf, -np.inf], np.nan).dropna()
    meta = {
        "区间": f"{shared[0].date()} ~ {shared[-1].date()}",
        "KBE根数": int(len(num_df)),
        "QQQ根数": int(len(den_df)),
        "交集根数": int(len(shared)),
        "QQQ波动/KBE波动_中位数": round(float(vol_ratio.median()), 3),
        "训练半衰期_交易日": None if not np.isfinite(hl_train) else round(hl_train, 1),
        "持有期": list(horizons),
        "图上持有期": list(focuses),
        "策略起点": start.date().isoformat(),
        "成本": 0,
        "样本外起点": gs.TEST_START.date().isoformat(),
    }
    pd.Series(meta).to_csv(OUT_DIR / "meta.csv", header=["值"])
    plot_ratio(ratio, ma, primary["reversal"], primary["aligned"], OUT_DIR / "ratio.png")

    curves = {}
    titles = []
    for slice_name, slice_label in (("all", "全样本"), ("test", "2024年起")):
        for horizon in focuses:
            title = f"{slice_label} · 背离持有 {horizon} 日"
            titles.append(title)
            panel = {"均线上方持有": gs.slice_curve(base_equity, shared, slice_name)}
            for book_name, label in (("reversal", "反转"), ("aligned", "顺势")):
                trades = gs.book_trades(
                    *primary[book_name], {horizon: spread_by_h[horizon]}, shared
                )[horizon]
                if slice_name == "test":
                    trades = [t for t in trades if pd.Timestamp(t["进场日期"]) >= gs.TEST_START]
                panel[label] = gs.slice_curve(gs.equity_from_exits(shared, trades), shared, slice_name)
            curves[title] = panel
    gs.plot_equity(curves, titles, OUT_DIR / "equity.png")

    show = stats_df[
        (stats_df["共振门槛"].isin([0, gs.PRIMARY_MIN]))
        & (stats_df["持有期"].isin(focuses))
        & (stats_df["样本"].isin(["all", "test"]))
    ]
    print(hl_df.to_string(index=False))
    print("horizons", horizons, "focus", focuses)
    print(pd.Series(meta).to_string())
    print(show.to_string(index=False))
    print(books_df[books_df["持有期"].isin([0, *focuses])].to_string(index=False))
    print(attribution.to_string(index=False))
    print(legs.to_string(index=False))


if __name__ == "__main__":
    main()
