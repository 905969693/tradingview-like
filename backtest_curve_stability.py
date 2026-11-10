"""
用 Dashboard 当前预设，看收缩超额曲线在四段里的符号是否一致。

参数与打开 Dashboard 时相同：内置 ASSET_PRESETS 打底，cache/fixed_presets.json
同名覆盖。没有预设的标的用侧边栏默认（pivot 3、共振 3、去重 10、MA200 顺势、
多空都算）。只做多的预设，曲线中位数只取底背离，笔数也只数这一侧。
入场滞后等于该标的自己的 pivot。

入场归到入场日所在窗口。出场可以落在窗口之外。
每一段的对照是同一日期掩码下、全部可入场 K 线的无条件胜率。
2025 年起的入场不参与。均线预热用更早的 K 线，只影响信号生成。

四段：2017–2018、2019–2020、2021–2022、2023–2024。
另外在 2017–2024 的入场上单独去掉 2020、单独去掉 2022。

尖峰不参与同号判断。同号要求四段中位数都有定义，且每段计入分数的笔数至少 4。
运行：python backtest_curve_stability.py
"""
import json
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tradingview_realization_v2 as tv

PERIOD = "10y"
PIVOT_ORDER = tv.PIVOT_ORDER
MIN_COUNT = tv.MIN_COUNT
DEDUP_BARS = tv.DEDUP_BARS
MA_N = tv.TREND_MA_N
HORIZONS = tv.POSTERIOR_SPAN
MIN_N = 4
SAMPLE_START = pd.Timestamp("2017-01-01")
SAMPLE_END = pd.Timestamp("2024-12-31")
WINDOWS = (
    ("2017_2018", pd.Timestamp("2017-01-01"), pd.Timestamp("2018-12-31")),
    ("2019_2020", pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31")),
    ("2021_2022", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
    ("2023_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
)
YEARS = tuple(range(2017, 2025))
STRICT = ("KRE", "KBE", "IAT", "REM")
CONTROL = ("SPY", "SOXX", "GSG")
WINDOW_NAMES = tuple(name for name, _start, _end in WINDOWS)
OUT_DIR = tv.SCRIPT_DIR / "backtest" / f"curve_stability_preset_{date.today().isoformat()}"
EDGE = HORIZONS[0], HORIZONS[-1]
# 与 dashboard.ASSET_PRESETS 相同。fixed_presets.json 覆盖同名。
_USDCHF_PRESET = {
    "pivot_order": 4,
    "min_count": 4,
    "dedup_bars": 10,
    "trend_on": False,
    "aligned_keep": True,
    "adx_th": None,
    "ma_n": 200,
    "summary": "滤波关、pivot 4、共振 4、去重 10",
}
BUILTIN_PRESETS = {
    "KBE": {
        "pivot_order": 6,
        "min_count": 4,
        "dedup_bars": 10,
        "trend_on": True,
        "aligned_keep": False,
        "adx_th": None,
        "ma_n": 100,
        "summary": "逆势、ADX 关、MA100、pivot 6、共振 4、去重 10",
    },
    "KRE": {
        "pivot_order": 6,
        "min_count": 3,
        "dedup_bars": 10,
        "trend_on": True,
        "aligned_keep": True,
        "adx_th": 25,
        "ma_n": 200,
        "summary": "顺势、ADX>25、MA200、pivot 6、共振 3、去重 10",
    },
    "IWM": {
        "pivot_order": 6,
        "min_count": 3,
        "dedup_bars": 10,
        "trend_on": True,
        "aligned_keep": True,
        "adx_th": 25,
        "ma_n": 200,
        "summary": "顺势、ADX>25、MA200、pivot 6、共振 3、去重 10",
    },
    "DVY": {
        "pivot_order": 6,
        "min_count": 5,
        "dedup_bars": 10,
        "trend_on": False,
        "aligned_keep": True,
        "adx_th": None,
        "ma_n": 200,
        "summary": "滤波关、pivot 6、共振 5、去重 10",
    },
    "USDCHF=X": _USDCHF_PRESET,
    "USDCHF_X": _USDCHF_PRESET,
}
SIDEBAR_DEFAULT = {
    "pivot_order": PIVOT_ORDER,
    "min_count": MIN_COUNT,
    "dedup_bars": DEDUP_BARS,
    "trend_on": True,
    "aligned_keep": True,
    "mode": "aligned",
    "adx_th": None,
    "ma_n": MA_N,
    "long_only": False,
    "summary": f"侧边栏默认、顺势、ADX 关、MA{MA_N}、pivot {PIVOT_ORDER}、共振 {MIN_COUNT}、去重 {DEDUP_BARS}",
    "source": "侧边栏默认",
}


def _normalize_preset(spec, source):
    adx = spec.get("adx_th")
    if adx in ("", None):
        adx = None
    else:
        adx = int(adx)
    trend_on = bool(spec["trend_on"])
    aligned_keep = bool(spec["aligned_keep"])
    if not trend_on:
        mode = "off"
    elif not aligned_keep:
        mode = "counter"
    else:
        mode = "aligned"
    return {
        "pivot_order": int(spec["pivot_order"]),
        "min_count": int(spec["min_count"]),
        "dedup_bars": int(spec["dedup_bars"]),
        "trend_on": trend_on,
        "aligned_keep": aligned_keep,
        "mode": mode,
        "adx_th": adx if trend_on else None,
        "ma_n": int(spec["ma_n"]),
        "long_only": bool(spec.get("long_only", False)),
        "summary": str(spec.get("summary") or ""),
        "source": source,
    }


def load_dashboard_presets():
    """与 Dashboard 相同：内置打底，fixed_presets.json 覆盖同名。"""
    merged = {
        key: _normalize_preset(spec, "内置")
        for key, spec in BUILTIN_PRESETS.items()
    }
    path = tv.CACHE_DIR / "fixed_presets.json"
    if not path.exists():
        return merged
    raw = json.loads(path.read_text())
    for key, spec in raw.items():
        if not isinstance(key, str) or not isinstance(spec, dict):
            continue
        merged[key] = _normalize_preset(spec, str(spec.get("source") or "fixed_presets"))
    return merged


def resolve_preset(ticker, merged):
    if ticker in merged:
        return merged[ticker]
    safe = tv.safe_chart_name(ticker)
    if safe in merged:
        return merged[safe]
    for key, spec in merged.items():
        if tv.safe_chart_name(key) == safe:
            return spec
    return None


def params_for(ticker, merged):
    spec = resolve_preset(ticker, merged)
    if spec is None:
        return dict(SIDEBAR_DEFAULT)
    return spec


def load_cached(ticker, period=PERIOD):
    path = tv.CACHE_DIR / f"{ticker}_{period}.csv"
    df = tv._to_ohlcv(pd.read_csv(path, index_col=0, parse_dates=True))
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def cached_tickers(period=PERIOD):
    names = []
    suffix = f"_{period}.csv"
    for path in sorted(tv.CACHE_DIR.glob(f"*{suffix}")):
        stem = path.name[: -len(suffix)]
        if stem.startswith("dual_tf_backtest"):
            continue
        names.append(stem)
    return names


def _dates_of(index, entry_idx):
    entry_idx = np.asarray(entry_idx, dtype=int)
    ok = (entry_idx >= 0) & (entry_idx < len(index))
    entry_idx = entry_idx[ok]
    if entry_idx.size == 0:
        return entry_idx, np.array([], dtype="datetime64[ns]")
    ts = pd.DatetimeIndex(index[entry_idx]).tz_localize(None)
    return entry_idx, ts


def mask_entries(index, entry_idx, start, end, drop_years=()):
    entry_idx, ts = _dates_of(index, entry_idx)
    if entry_idx.size == 0:
        return entry_idx
    keep = (ts >= start) & (ts <= end)
    if drop_years:
        keep = keep & ~np.isin(ts.year, list(drop_years))
    return entry_idx[np.asarray(keep)]


def _returns(closes, entry_idx, horizon, short):
    rets = tv._signal_horizon_returns(closes, entry_idx, horizon, short)
    if rets.size == 0:
        return rets
    return rets[np.isfinite(rets)]


def side_curve(closes, sig_entries, base_entries, short):
    rows = []
    for horizon in HORIZONS:
        sig_rets = _returns(closes, sig_entries, horizon, short)
        base_rets = _returns(closes, base_entries, horizon, short)
        base_win = float((base_rets > 0).mean()) if base_rets.size else np.nan
        base_mean = float(base_rets.mean()) if base_rets.size else np.nan
        rows.append(tv._curve_row(horizon, sig_rets, base_win, base_mean))
    return pd.DataFrame(rows)


def combined_scores(long_curve, short_curve):
    left = long_curve.set_index("持有根数")
    right = short_curve.set_index("持有根数")
    rows = []
    for horizon in HORIZONS:
        long_n = int(left.at[horizon, "信号数"]) if horizon in left.index else 0
        short_n = int(right.at[horizon, "信号数"]) if horizon in right.index else 0
        long_score = float(left.at[horizon, "分数"]) if horizon in left.index else np.nan
        short_score = float(right.at[horizon, "分数"]) if horizon in right.index else np.nan
        score = tv.combine_edge_scores(long_score, long_n, short_score, short_n, False)
        rows.append({
            "持有根数": horizon,
            "分数": score,
            "多笔数": long_n,
            "空笔数": short_n,
        })
    return pd.DataFrame(rows)


def _pos_share(combined):
    scores = combined["分数"].to_numpy(dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.nan
    return float((finite > 0).mean())


def _sign(value):
    if value is None or not np.isfinite(value):
        return "无"
    if value > 0:
        return "正"
    if value < 0:
        return "负"
    return "零"


def _boundary(peak):
    if peak is None or not np.isfinite(peak):
        return False
    return int(peak) in EDGE


def score_frame(long_curve, short_curve, long_only):
    """只做多时，分数和笔数都只留底背离，与 book_curve_median(long_only=True) 一致。"""
    if not long_only:
        return combined_scores(long_curve, short_curve)
    frame = long_curve.rename(columns={"信号数": "多笔数"}).copy()
    frame["空笔数"] = 0
    return frame[["持有根数", "分数", "多笔数", "空笔数"]]


def evaluate_mask(
    closes, index, bull_e, bear_e, base_all, start, end, drop_years=(), long_only=False,
):
    bull = mask_entries(index, bull_e, start, end, drop_years)
    bear = mask_entries(index, bear_e, start, end, drop_years)
    base = mask_entries(index, base_all, start, end, drop_years)
    long_curve = side_curve(closes, bull, base, False)
    short_curve = side_curve(closes, bear, base, True)
    scored = score_frame(long_curve, short_curve, long_only)
    median = tv.book_curve_median(long_curve, short_curve, long_only)
    finite = scored["分数"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    long_n = int(long_curve.iloc[0]["信号数"]) if len(long_curve) else 0
    short_n = int(short_curve.iloc[0]["信号数"]) if len(short_curve) else 0
    return {
        "median": median,
        "n": long_n if long_only else long_n + short_n,
        "long_n": long_n,
        "short_n": short_n,
        "pos_share": _pos_share(scored),
        "score_span": float(finite.max() - finite.min()) if finite.size else np.nan,
        "peak_long": tv.curve_peak_hold(long_curve),
        "peak_short": None if long_only else tv.curve_peak_hold(short_curve),
        "combined": scored,
        "long_curve": long_curve,
        "short_curve": short_curve,
    }


def signals_for(df, params):
    bull, bear = tv.raw_deduped_signals(
        df,
        pivot_order=params["pivot_order"],
        min_count=params["min_count"],
        dedup_bars=params["dedup_bars"],
    )
    bull, bear = tv.apply_trend_filter(
        df, bull, bear,
        ma_n=params["ma_n"], mode=params["mode"], adx_th=params["adx_th"],
    )
    shift = params["pivot_order"]
    bull_e = tv._signal_entry_positions(df, bull, shift)
    bear_e = tv._signal_entry_positions(df, bear, shift)
    return bull_e, bear_e


def curve_rows(ticker, slice_name, result):
    combined = result["combined"]
    rows = []
    for _, row in combined.iterrows():
        rows.append({
            "标的": ticker,
            "切片": slice_name,
            "持有根数": int(row["持有根数"]),
            "分数": float(row["分数"]) if np.isfinite(row["分数"]) else np.nan,
            "多笔数": int(row["多笔数"]),
            "空笔数": int(row["空笔数"]),
        })
    return rows


def summary_row(ticker, slice_name, result):
    return {
        "标的": ticker,
        "切片": slice_name,
        "中位数": result["median"],
        "符号": _sign(result["median"]),
        "笔数": result["n"],
        "多笔数": result["long_n"],
        "空笔数": result["short_n"],
        "正分占比": result["pos_share"],
        "分数极差": result["score_span"],
        "底尖峰": result["peak_long"] if result["peak_long"] is not None else np.nan,
        "顶尖峰": result["peak_short"] if result["peak_short"] is not None else np.nan,
        "底贴边": _boundary(result["peak_long"]),
        "顶贴边": _boundary(result["peak_short"]),
    }


def agreement_status(parts):
    """parts: list of (median, n) for the four windows, in order."""
    medians = [median for median, _n in parts]
    ns = [n for _median, n in parts]
    if any((not np.isfinite(median)) or n < MIN_N for median, n in parts):
        return "样本不足"
    signs = [_sign(median) for median in medians]
    if all(sign == "正" for sign in signs):
        return "四段同正"
    if all(sign == "负" for sign in signs):
        return "四段同负"
    return "四段变号"


def _points(value):
    if value is None or not np.isfinite(value):
        return np.nan
    return float(value) * 100.0


def plot_heatmap(summary, path):
    window = summary[summary["切片"].isin(WINDOW_NAMES)].copy()
    tickers = _heatmap_order(window)
    grid = (
        window.pivot(index="标的", columns="切片", values="中位数")
        .reindex(index=tickers, columns=list(WINDOW_NAMES))
    )
    counts = (
        window.pivot(index="标的", columns="切片", values="笔数")
        .reindex(index=tickers, columns=list(WINDOW_NAMES))
    )
    values = grid.to_numpy(dtype=float) * 100.0
    finite = values[np.isfinite(values)]
    limit = 8.0 if finite.size == 0 else float(np.nanpercentile(np.abs(finite), 90))
    limit = max(limit, 2.0)
    fig_h = max(6.0, 0.38 * len(tickers) + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(WINDOW_NAMES)))
    ax.set_xticklabels(["2017–2018", "2019–2020", "2021–2022", "2023–2024"])
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            n = counts.to_numpy()[i, j]
            if not np.isfinite(value):
                label = "—"
            else:
                mark = "" if np.isfinite(n) and n >= MIN_N else "*"
                label = f"{value:+.1f}{mark}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="#1a1a1a")
    ax.set_title("四段曲线中位数（收缩超额胜率，单位：百分点）")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="百分点")
    fig.text(
        0.01, 0.01,
        "各标的用 Dashboard 当前预设（文件覆盖内置）。没有预设的用侧边栏默认。"
        f"格内为中位数×100，* 表示计入分数的笔数不足 {MIN_N}。只做多只数底背离。对照是同段无条件入场。",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _heatmap_order(window):
    ranks = []
    for ticker, part in window.groupby("标的"):
        medians = [
            float(part.loc[part["切片"] == name, "中位数"].iloc[0])
            if (part["切片"] == name).any() else np.nan
            for name in WINDOW_NAMES
        ]
        ns = [
            int(part.loc[part["切片"] == name, "笔数"].iloc[0])
            if (part["切片"] == name).any() else 0
            for name in WINDOW_NAMES
        ]
        status = agreement_status(list(zip(medians, ns)))
        status_rank = {"四段同正": 0, "四段同负": 1, "四段变号": 2, "样本不足": 3}[status]
        focus_rank = 0 if ticker in STRICT else (1 if ticker in CONTROL else 2)
        finite = [value for value in medians if np.isfinite(value)]
        level = float(np.median(finite)) if finite else -999.0
        ranks.append((focus_rank, status_rank, -level, ticker))
    ranks.sort()
    return [ticker for _a, _b, _c, ticker in ranks]


def plot_curves(curves, tickers, path):
    show = [ticker for ticker in tickers if ticker in set(curves["标的"])]
    if not show:
        return
    ncols = 2
    nrows = int(np.ceil(len(show) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.1 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {
        "2017_2018": "#4C78A8",
        "2019_2020": "#F58518",
        "2021_2022": "#E45756",
        "2023_2024": "#54A24B",
    }
    labels = {
        "2017_2018": "2017–2018",
        "2019_2020": "2019–2020",
        "2021_2022": "2021–2022",
        "2023_2024": "2023–2024",
    }
    for ax, ticker in zip(axes, show):
        part = curves[curves["标的"] == ticker]
        for name in WINDOW_NAMES:
            rows = part[part["切片"] == name].sort_values("持有根数")
            if rows.empty:
                continue
            y = rows["分数"].to_numpy(dtype=float) * 100.0
            ax.plot(rows["持有根数"], y, color=colors[name], label=labels[name], lw=1.4)
        ax.axhline(0, color="#666666", lw=0.6)
        ax.set_title(ticker)
        ax.set_ylabel("收缩超额（百分点）")
        ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    for ax in axes[len(show):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="best")
    for ax in axes[max(0, len(show) - ncols): len(show)]:
        ax.set_xlabel("持有根数")
    fig.suptitle("四段收缩超额曲线（Dashboard 当前预设，纵轴为相对同段无条件胜率）", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_drop_year(summary, tickers, path):
    slices = ("2017_2024", "drop_2020", "drop_2022")
    labels = ("2017–2024", "去掉 2020 入场", "去掉 2022 入场")
    show = [ticker for ticker in tickers if (summary["标的"] == ticker).any()]
    if not show:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 0.85 * len(show) + 2), 4.6))
    x = np.arange(len(show))
    width = 0.25
    colors = ("#4C78A8", "#F58518", "#E45756")
    for offset, (slice_name, label, color) in enumerate(zip(slices, labels, colors)):
        vals = []
        for ticker in show:
            rows = summary[(summary["标的"] == ticker) & (summary["切片"] == slice_name)]
            vals.append(_points(float(rows["中位数"].iloc[0])) if len(rows) else np.nan)
        ax.bar(x + (offset - 1) * width, vals, width=width, label=label, color=color)
    ax.axhline(0, color="#666666", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(show, rotation=30, ha="right")
    ax.set_ylabel("曲线中位数（百分点）")
    ax.set_title("去掉 2020 或 2022 的入场之后，2017–2024 曲线中位数")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _focus_tickers(agreement):
    same = agreement[agreement["结论"].isin(["四段同正", "四段同负", "四段变号"])]["标的"].tolist()
    ordered = []
    for ticker in list(STRICT) + list(CONTROL) + same:
        if ticker not in ordered:
            ordered.append(ticker)
    return ordered


def main():
    summary_rows = []
    curve_store = []
    agree_rows = []
    skipped = []
    loaded = []
    presets = load_dashboard_presets()
    for ticker in cached_tickers():
        df = load_cached(ticker)
        volume = df.Volume.to_numpy(dtype=float)
        if not np.isfinite(volume).any() or np.nansum(volume) <= 0:
            skipped.append(ticker)
            continue
        params = params_for(ticker, presets)
        df = tv.add_indicators(df)
        bull_e, bear_e = signals_for(df, params)
        closes = df.Close.to_numpy(dtype=float)
        base_all = np.arange(len(df), dtype=int)
        index = df.index
        loaded.append(ticker)
        long_only = bool(params["long_only"])
        slices = [(name, start, end, ()) for name, start, end in WINDOWS]
        slices.append(("2017_2024", SAMPLE_START, SAMPLE_END, ()))
        slices.append(("drop_2020", SAMPLE_START, SAMPLE_END, (2020,)))
        slices.append(("drop_2022", SAMPLE_START, SAMPLE_END, (2022,)))
        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01")
            end = pd.Timestamp(f"{year}-12-31")
            slices.append((f"y{year}", start, end, ()))
        window_parts = []
        for name, start, end, drop_years in slices:
            result = evaluate_mask(
                closes, index, bull_e, bear_e, base_all, start, end, drop_years,
                long_only=long_only,
            )
            summary_rows.append(summary_row(ticker, name, result))
            curve_store.extend(curve_rows(ticker, name, result))
            if name in WINDOW_NAMES:
                window_parts.append((result["median"], result["n"]))
        medians = [median for median, _n in window_parts]
        ns = [n for _median, n in window_parts]
        agree_rows.append({
            "标的": ticker,
            "结论": agreement_status(window_parts),
            "参数": params["summary"],
            "来源": params["source"],
            "只做多": long_only,
            "2017_2018": medians[0],
            "2019_2020": medians[1],
            "2021_2022": medians[2],
            "2023_2024": medians[3],
            "笔数_2017_2018": ns[0],
            "笔数_2019_2020": ns[1],
            "笔数_2021_2022": ns[2],
            "笔数_2023_2024": ns[3],
            "符号": "".join(_sign(median) for median in medians),
        })
        print(
            f"{ticker}  {agreement_status(window_parts)}  {params['summary']}  "
            + " ".join(f"{_sign(median)}{ns[i]}" for i, median in enumerate(medians)),
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_store)
    agreement = pd.DataFrame(agree_rows)
    if agreement.empty:
        print("没有可计算的标的")
        return
    status_rank = {"四段同正": 0, "四段同负": 1, "四段变号": 2, "样本不足": 3}
    agreement["_rank"] = agreement["结论"].map(status_rank)
    agreement = agreement.sort_values(["_rank", "标的"]).drop(columns="_rank")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "slices.csv", index=False)
    curves.to_csv(OUT_DIR / "curves.csv", index=False)
    agreement.to_csv(OUT_DIR / "agreement.csv", index=False)

    focus = [ticker for ticker in _focus_tickers(agreement) if ticker in loaded]
    plot_heatmap(summary, OUT_DIR / "windows_heatmap.png")
    plot_curves(curves, focus, OUT_DIR / "window_curves.png")
    plot_drop_year(summary, focus, OUT_DIR / "drop_year.png")

    peaks = summary[summary["切片"].isin(WINDOW_NAMES)]
    boundary_n = int(peaks["底贴边"].sum() + peaks["顶贴边"].sum())
    peak_n = int(peaks["底尖峰"].notna().sum() + peaks["顶尖峰"].notna().sum())
    counts = agreement["结论"].value_counts()
    meta = pd.DataFrame([{
        "标的数": int(len(agreement)),
        "跳过无成交量": ",".join(skipped),
        "参数": "Dashboard 当前预设；无预设则侧边栏默认。只做多只数底背离",
        "持有期": f"{HORIZONS[0]}–{HORIZONS[-1]}",
        "同号门槛": f"四段都有定义且每段笔数>={MIN_N}",
        "四段同正": int(counts.get("四段同正", 0)),
        "四段同负": int(counts.get("四段同负", 0)),
        "四段变号": int(counts.get("四段变号", 0)),
        "样本不足": int(counts.get("样本不足", 0)),
        "尖峰贴边占比": (boundary_n / peak_n) if peak_n else np.nan,
    }])
    meta.to_csv(OUT_DIR / "meta.csv", index=False)
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda value: f"{value:.4f}")
    print(meta.to_string(index=False))
    print(agreement.to_string(index=False))
    print(f"写出 {OUT_DIR}")


if __name__ == "__main__":
    main()
