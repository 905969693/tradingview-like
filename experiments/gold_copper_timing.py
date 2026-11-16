"""思路 2：金/铜比当作增长与避险状态，看它能不能帮 ETF 择时。

只读上层模块和 cache 里已有的 ETF。铜用 CPER，下载结果只放在本目录，不写上层 cache。
比例不对背离。参数事先定死。

金/铜 = GLD 收盘 / CPER 收盘。
收盘高于 MA200：金强于铜，记为避险（增长偏弱）。
收盘低于 MA200：铜强于金，记为增长。
增长状态的下一根才持有 EEM、IWM、SOXX。避险状态的下一根才持有 GLD、TLT。
另一状态空仓，不反向做空。
对照是同一窗口一直持有该 ETF。
另外用默认背离（pivot 3、共振 3、八个指标、去重 10）看闸门能不能提高底背离之后 20 日的收益。
进场仍是信号之后第 3 根收盘。样本外从 2024-01-01 起。成本 0。
一条比例不能把通胀和增长拆成两个独立因子。这里只测上面这一个开关。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tradingview_realization_v2 as tv  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "gold_copper"
COPPER_PATH = OUT_DIR / "CPER.csv"
MA_N = 200
PIVOT = 3
MIN_COUNT = 3
DEDUP = 10
HOLD = 20
TEST_START = pd.Timestamp("2024-01-01")
GROWTH = ("EEM", "IWM", "SOXX")
DEFENSIVE = ("GLD", "TLT")
ROLE = {name: "增长" for name in GROWTH}
ROLE.update({name: "避险" for name in DEFENSIVE})


def read_cached(ticker, period="10y"):
    path = tv.CACHE_DIR / f"{tv.safe_chart_name(ticker)}_{period}.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    out = tv._to_ohlcv(frame)
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_copper():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if COPPER_PATH.exists():
        frame = pd.read_csv(COPPER_PATH, index_col=0, parse_dates=True)
        return tv._to_ohlcv(frame)
    raw = yf.download("CPER", period="10y", interval="1d", auto_adjust=True, progress=False)
    out = tv._to_ohlcv(raw)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise RuntimeError("CPER 没有下载到数据")
    out.to_csv(COPPER_PATH)
    return out


def max_drawdown(equity):
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def performance(equity, daily, active):
    terminal = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    on = active.reindex(equity.index).fillna(False)
    taken = daily.reindex(equity.index).where(on)
    return {
        "累计收益%": round(terminal * 100, 2) if np.isfinite(terminal) else np.nan,
        "最大回撤%": round(max_drawdown(equity) * 100, 2) if len(equity) else np.nan,
        "在场占比": round(float(on.mean()), 4) if len(on) else np.nan,
        "在场日均bps": round(float(taken.mean()) * 10000, 2) if taken.notna().any() else np.nan,
    }


def forward_mean(close, entry_locs, hold):
    px = close.to_numpy(dtype=float)
    vals = []
    for ei in entry_locs:
        ej = ei + hold
        if ej >= len(px) or px[ei] == 0:
            continue
        vals.append(px[ej] / px[ei] - 1.0)
    return vals


def describe(vals):
    series = pd.Series(vals, dtype=float).dropna()
    n = int(len(series))
    if n == 0:
        return {"笔数": 0, "胜率": np.nan, "平均收益%": np.nan, "中位收益%": np.nan}
    return {
        "笔数": n,
        "胜率": round(float((series > 0).mean()), 4),
        "平均收益%": round(float(series.mean()) * 100, 3),
        "中位收益%": round(float(series.median()) * 100, 3),
    }


def slice_index(index, name):
    if name == "test":
        return index[index >= TEST_START]
    return index


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copper = load_copper()
    gold = read_cached("GLD")
    prices = {name: read_cached(name) for name in (*GROWTH, *DEFENSIVE)}
    shared = gold.index.intersection(copper.index)
    for frame in prices.values():
        shared = shared.intersection(frame.index)
    shared = shared.sort_values()

    ratio_px = gold.loc[shared, "Close"] / copper.loc[shared, "Close"]
    ma = ratio_px.rolling(MA_N).mean()
    gold_leads = (ratio_px > ma) & ma.notna()
    copper_leads = (ratio_px < ma) & ma.notna()
    agree = {name: copper_leads if name in GROWTH else gold_leads for name in prices}

    curves = {}
    rows = []
    signal_rows = []
    for name, frame in prices.items():
        close = frame.loc[shared, "Close"]
        ret = close.pct_change()
        state = agree[name]
        active = state.shift(1).fillna(False) & ret.notna()
        live = ma.notna().shift(1).fillna(False) & ret.notna()
        timed_ret = ret.where(active, 0.0).fillna(0.0)
        hold_ret = ret.where(live, 0.0).fillna(0.0)
        timed = (1.0 + timed_ret).cumprod()
        held = (1.0 + hold_ret).cumprod()
        start = live[live].index[0]
        base_timed = timed.shift(1).loc[start]
        base_held = held.shift(1).loc[start]
        curves[name] = {
            "择时": timed.loc[start:] / base_timed,
            "一直持有": held.loc[start:] / base_held,
            "在场": active.loc[start:],
        }
        for slice_name in ("all", "test"):
            window = slice_index(curves[name]["择时"].index, slice_name)
            timed_w = curves[name]["择时"].reindex(window)
            held_w = curves[name]["一直持有"].reindex(window)
            timed_w = timed_w / timed_w.iloc[0]
            held_w = held_w / held_w.iloc[0]
            on = curves[name]["在场"].reindex(window).fillna(False)
            off = live.reindex(window).fillna(False) & ~on
            day_ret = ret.reindex(window)
            for label, equity, mask in (
                ("状态同意才持有", timed_w, on),
                ("一直持有", held_w, pd.Series(True, index=window)),
            ):
                stat = performance(equity, day_ret, mask)
                stat.update({"标的": name, "角色": ROLE[name], "规则": label, "样本": slice_name})
                rows.append(stat)
            agree_daily = ret.reindex(window).where(on)
            flat_daily = ret.reindex(window).where(off)
            signal_rows.append({
                "标的": name,
                "角色": ROLE[name],
                "样本": slice_name,
                "种类": "同意日涨跌",
                "笔数": int(agree_daily.notna().sum()),
                "胜率": round(float((agree_daily.dropna() > 0).mean()), 4) if agree_daily.notna().any() else np.nan,
                "平均收益%": round(float(agree_daily.mean()) * 100, 4) if agree_daily.notna().any() else np.nan,
                "中位收益%": round(float(agree_daily.median()) * 100, 4) if agree_daily.notna().any() else np.nan,
            })
            signal_rows.append({
                "标的": name,
                "角色": ROLE[name],
                "样本": slice_name,
                "种类": "空仓日涨跌",
                "笔数": int(flat_daily.notna().sum()),
                "胜率": round(float((flat_daily.dropna() > 0).mean()), 4) if flat_daily.notna().any() else np.nan,
                "平均收益%": round(float(flat_daily.mean()) * 100, 4) if flat_daily.notna().any() else np.nan,
                "中位收益%": round(float(flat_daily.median()) * 100, 4) if flat_daily.notna().any() else np.nan,
            })

        bull, _bear = tv.raw_deduped_signals(
            frame.loc[shared], pivot_order=PIVOT, min_count=MIN_COUNT, dedup_bars=DEDUP
        )
        entries = []
        flags = []
        for stamp in bull.index:
            xi = int(shared.get_loc(stamp))
            ei = xi + PIVOT
            if ei >= len(shared):
                continue
            entries.append(ei)
            flags.append(bool(state.iloc[ei]))
        entries = np.asarray(entries, dtype=int)
        flags = np.asarray(flags, dtype=bool)
        for slice_name in ("all", "test"):
            if slice_name == "test":
                keep = np.array([shared[i] >= TEST_START for i in entries], dtype=bool)
            else:
                keep = np.ones(len(entries), dtype=bool)
            groups = {
                "底背离全部": entries[keep],
                "底背离且状态同意": entries[keep][flags[keep]] if keep.any() else [],
                "底背离且状态相反": entries[keep][~flags[keep]] if keep.any() else [],
            }
            for kind, locs in groups.items():
                stat = describe(forward_mean(close, locs, HOLD))
                stat.update({"标的": name, "角色": ROLE[name], "样本": slice_name, "种类": kind})
                signal_rows.append(stat)

    book = pd.DataFrame(rows)
    signals = pd.DataFrame(signal_rows)
    book.to_csv(OUT_DIR / "timing.csv", index=False)
    signals.to_csv(OUT_DIR / "signals.csv", index=False)

    ratio_frame = pd.DataFrame({"金铜比": ratio_px, "MA200": ma})
    ratio_frame.to_csv(OUT_DIR / "ratio.csv")
    meta = pd.Series({
        "区间": f"{shared[0].date()} ~ {shared[-1].date()}",
        "交集根数": int(len(shared)),
        "铜": "CPER",
        "均线": MA_N,
        "避险占比": round(float(gold_leads.mean()), 4),
        "增长占比": round(float(copper_leads.mean()), 4),
        "成本": 0,
        "样本外起点": TEST_START.date().isoformat(),
    })
    meta.to_csv(OUT_DIR / "meta.csv", header=["值"])

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=110)
    ax.plot(ratio_px.index, ratio_px.values, color="#222222", lw=0.9, label="GLD/CPER")
    ax.plot(ma.index, ma.values, color="#c4a35a", lw=1.0, ls="--", label="MA200")
    ax.set_title("金/铜收盘比")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ratio.png", bbox_inches="tight")
    plt.close(fig)

    names = list(prices)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.3 * len(names)), dpi=110, sharex=True)
    for ax, name in zip(axes, names):
        series = curves[name]
        ax.plot(series["一直持有"].index, series["一直持有"].values, color="#4d4d4d", lw=1.0, ls="--", label="一直持有")
        ax.plot(series["择时"].index, series["择时"].values, color="#1b9e77", lw=1.2, label="状态同意才持有")
        ax.axhline(1.0, color="black", lw=0.5)
        end_timed = series["择时"].iloc[-1]
        end_held = series["一直持有"].iloc[-1]
        ax.set_ylabel(name)
        ax.set_title(
            f"{name} · {ROLE[name]}  择时 {end_timed:.2f} 倍 / 一直持有 {end_held:.2f} 倍",
            fontsize=10, loc="left",
        )
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("金/铜 MA200 开关。增长持有 EEM、IWM、SOXX；避险持有 GLD、TLT。成本 0。", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity.png", bbox_inches="tight")
    plt.close(fig)

    show_book = book[book["样本"].isin(["all", "test"])][
        ["标的", "角色", "规则", "样本", "累计收益%", "最大回撤%", "在场占比", "在场日均bps"]
    ]
    show_sig = signals[signals["种类"].isin(["同意日涨跌", "空仓日涨跌", "底背离全部", "底背离且状态同意", "底背离且状态相反"])]
    print(meta.to_string())
    print(show_book.to_string(index=False))
    print(show_sig.to_string(index=False))


if __name__ == "__main__":
    main()
