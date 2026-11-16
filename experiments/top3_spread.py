"""思路 1 用在半衰期最短的三对：IWM/SLV、GLD/XLP、DBA/KBE。

规则与 experiments/gld_slv_spread.py 相同。比例 = 表上的前者收盘 / 后者收盘。
底背离做多比例，顶背离做空比例。成本 0。样本外从 2024-01-01 起。
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

OUT_DIR = Path(__file__).resolve().parent / "top3_spread"
PAIRS = (("IWM", "SLV"), ("GLD", "XLP"), ("DBA", "KBE"))


def plot_ratio(df, ma, raw, aligned, num, den, save_path):
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=110)
    ax.plot(df.index, df["Close"], color="#222222", lw=0.9, label=f"{num}/{den} 收盘比")
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
        f"{num}/{den} 收盘比（pivot {gs.PIVOT}，共振 ≥{gs.PRIMARY_MIN}，"
        f"指标 {'/'.join(gs.INDICATORS)}）"
    )
    ax.legend(fontsize=8, ncol=3, loc="best")
    ax.set_ylabel(f"{num} 收盘 / {den} 收盘")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_equity(curves, titles, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), dpi=110, sharex=False)
    colors = {
        "均线上方持有": "#4d4d4d",
        "均线下方持有": "#d95f02",
        "反转": "#1b9e77",
        "顺势": "#7570b3",
    }
    styles = {
        "均线上方持有": "--",
        "均线下方持有": ":",
        "反转": "-",
        "顺势": "-",
    }
    for ax, title in zip(axes.ravel(), titles):
        drawn = curves[title]
        for name, series in drawn.items():
            ax.plot(
                series.index, series.values, lw=1.2, ls=styles[name],
                color=colors[name], label=name,
            )
        ax.axhline(1.0, color="black", lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("价差净值：波动率配平，确认后收盘进场，持仓期内的新信号跳过", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def compound_from(ret, start):
    equity = (1.0 + ret.fillna(0.0)).cumprod()
    base = equity.shift(1).loc[start]
    return equity.loc[start:] / base


def run_pair(num, den):
    out_dir = OUT_DIR / f"{num}_{den}"
    out_dir.mkdir(parents=True, exist_ok=True)
    num_df = gs.read_cached(num)
    den_df = gs.read_cached(den)
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
    below_active = below.shift(1).fillna(False) & daily.notna()
    base_equity, earned = gs.baseline_curve(active, daily, shared)
    below_equity, below_earned = gs.baseline_curve(below_active, daily, shared)
    live = ma.notna().shift(1).fillna(False)
    for slice_name in ("all", "train", "test"):
        books.append(gs.baseline_summary(
            "均线上方持有", base_equity, active, live, shared, slice_name
        ))
        books.append(gs.baseline_summary(
            "均线下方持有", below_equity, below_active, live, shared, slice_name
        ))

    net = (weight_n_s - weight_d_s).shift(1)

    def side_parts(flag):
        bias_s = (net * (num_ret + den_ret) / 2).where(flag, 0.0).fillna(0.0)
        relative_s = (0.5 * (num_ret - den_ret)).where(flag, 0.0).fillna(0.0)
        earned_s = daily.where(flag, 0.0).fillna(0.0)
        return earned_s, bias_s, relative_s

    _, bias, relative = side_parts(active)
    below_bias_src, below_rel_src = side_parts(below_active)[1:]
    start = live[live].index[0]
    num_timed = compound_from(num_ret.where(active, 0.0).fillna(0.0), start)
    num_always = compound_from(num_ret.where(live, 0.0).fillna(0.0), start)
    den_always = compound_from(den_ret.where(live, 0.0).fillna(0.0), start)

    def end_stats(curve):
        dd = float((curve / curve.cummax() - 1.0).min())
        return {
            "期末净值": round(float(curve.iloc[-1]), 4),
            "累计收益%": round(float(curve.iloc[-1] - 1.0) * 100, 2),
            "最大回撤%": round(dd * 100, 2),
        }

    def slice_attr(earned_s, bias_s, relative_s, flag, mask, label, rule):
        return {
            "规则": rule,
            "样本": label,
            "配平加总%": round(float(earned_s[mask].sum()) * 100, 2),
            "净敞口加总%": round(float(bias_s[mask].sum()) * 100, 2),
            "相对涨跌加总%": round(float(relative_s[mask].sum()) * 100, 2),
            "配平复利%": round(float((1.0 + earned_s[mask]).prod() - 1.0) * 100, 2),
            "持仓日多分子权重": round(float(weight_n_s.shift(1)[flag & mask].mean()), 4),
            "持仓日空分母权重": round(float(weight_d_s.shift(1)[flag & mask].mean()), 4),
        }

    full_mask = pd.Series(True, index=earned.index)
    test_mask = earned.index >= gs.TEST_START
    attribution = pd.DataFrame([
        slice_attr(earned, bias, relative, active, full_mask, "all", "均线上方持有"),
        slice_attr(earned, bias, relative, active, test_mask, "test", "均线上方持有"),
        slice_attr(below_earned, below_bias_src, below_rel_src, below_active, full_mask, "all", "均线下方持有"),
        slice_attr(below_earned, below_bias_src, below_rel_src, below_active, test_mask, "test", "均线下方持有"),
    ])
    legs = pd.DataFrame([
        {"规则": f"均线上方只做多{num}", **end_stats(num_timed), "在场占比": round(float(active.loc[start:].mean()), 4)},
        {"规则": f"一直持有{num}", **end_stats(num_always), "在场占比": 1.0},
        {"规则": f"一直持有{den}", **end_stats(den_always), "在场占比": 1.0},
    ])
    attribution.to_csv(out_dir / "attribution.csv", index=False)
    legs.to_csv(out_dir / "legs.csv", index=False)

    stats_df = pd.DataFrame(signal_stats)[
        ["共振门槛", "书", "方向", "持有期", "样本", "交易数", "胜率", "平均收益%", "中位收益%"]
    ]
    books_df = pd.DataFrame(books)
    stats_df.to_csv(out_dir / "signal_stats.csv", index=False)
    books_df.to_csv(out_dir / "books.csv", index=False)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if len(trades):
        trades.to_csv(out_dir / "trades.csv", index=False)
    hl_df = pd.DataFrame([
        {"样本": "train_<2024", "半衰期_交易日": hl_train, "斜率b": slope_train, "观测数": n_train},
        {"样本": "test_>=2024", "半衰期_交易日": hl_test, "斜率b": slope_test, "观测数": n_test},
        {"样本": "all", "半衰期_交易日": hl_all, "斜率b": slope_all, "观测数": n_all},
    ])
    hl_df.to_csv(out_dir / "half_life.csv", index=False)
    vol_ratio = (vol_d / vol_n).replace([np.inf, -np.inf], np.nan).dropna()
    meta = {
        "区间": f"{shared[0].date()} ~ {shared[-1].date()}",
        f"{num}根数": int(len(num_df)),
        f"{den}根数": int(len(den_df)),
        "交集根数": int(len(shared)),
        f"{den}波动/{num}波动_中位数": round(float(vol_ratio.median()), 3),
        "训练半衰期_交易日": None if not np.isfinite(hl_train) else round(hl_train, 1),
        "持有期": list(horizons),
        "图上持有期": list(focuses),
        "策略起点": start.date().isoformat(),
        "成本": 0,
        "样本外起点": gs.TEST_START.date().isoformat(),
    }
    pd.Series(meta).to_csv(out_dir / "meta.csv", header=["值"])
    plot_ratio(ratio, ma, primary["reversal"], primary["aligned"], num, den, out_dir / "ratio.png")

    curves = {}
    titles = []
    for slice_name, slice_label in (("all", "全样本"), ("test", "2024年起")):
        for horizon in focuses:
            title = f"{slice_label} · 背离持有 {horizon} 日"
            titles.append(title)
            panel = {
                "均线上方持有": gs.slice_curve(base_equity, shared, slice_name),
                "均线下方持有": gs.slice_curve(below_equity, shared, slice_name),
            }
            for book_name, label in (("reversal", "反转"), ("aligned", "顺势")):
                taken = gs.book_trades(
                    *primary[book_name], {horizon: spread_by_h[horizon]}, shared
                )[horizon]
                if slice_name == "test":
                    taken = [t for t in taken if pd.Timestamp(t["进场日期"]) >= gs.TEST_START]
                panel[label] = gs.slice_curve(gs.equity_from_exits(shared, taken), shared, slice_name)
            curves[title] = panel
    plot_equity(curves, titles, out_dir / "equity.png")
    return meta, hl_df, stats_df, books_df, attribution, legs, trades, focuses


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for num, den in PAIRS:
        meta, hl_df, stats_df, books_df, attribution, legs, trades, focuses = run_pair(num, den)
        print(f"\n===== {num}/{den} =====")
        print(hl_df.to_string(index=False))
        print(pd.Series(meta).to_string())
        show = stats_df[
            (stats_df["共振门槛"].isin([0, gs.PRIMARY_MIN]))
            & (stats_df["持有期"].isin(focuses))
            & (stats_df["样本"].isin(["all", "test"]))
        ]
        print(show.to_string(index=False))
        print(books_df[books_df["持有期"].isin([0, *focuses])].to_string(index=False))
        print(attribution.to_string(index=False))
        print(legs.to_string(index=False))
        if len(trades):
            rev = trades[trades["书"] == "reversal"]
            for horizon in focuses:
                part = rev[rev["持有期"] == horizon]
                for slice_name, mask in (
                    ("all", pd.Series(True, index=part.index)),
                    ("test", pd.to_datetime(part["进场日期"]) >= gs.TEST_START),
                ):
                    cut = part[mask.to_numpy()]
                    for side, group in cut.groupby("方向"):
                        print(
                            f"不重叠 {slice_name} {horizon}日 {side}: "
                            f"n={len(group)} 均值%={group['收益'].mean() * 100:.3f} "
                            f"加总%={group['收益'].sum() * 100:.3f}"
                        )


if __name__ == "__main__":
    main()
