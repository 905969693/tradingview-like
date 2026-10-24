import os
import re
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from scipy.signal import argrelextrema

# 中文显示：选用本机第一个可用的 CJK 字体（macOS 常见 Heiti SC / Songti SC）
_CJK = ("PingFang SC", "Heiti SC", "Songti SC", "Hiragino Sans GB", "Arial Unicode MS", "SimHei", "Microsoft YaHei")
_available = {f.name for f in font_manager.fontManager.ttflist}
for _name in _CJK:
    if _name in _available:
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# ============ 0. 可调参数 ============
TICKERS = [
    "SPY",
    "USDJPY=X",
    "KBE",
    "USO",
    "TLT",
    "GLD",
    "EEM",
    "SOXX",
    "1810.HK"
]
PERIOD = "5y"
PIVOT_ORDER = 6
MIN_COUNT = 3
DEDUP_BARS = 10
LOOKBACK = 160
SIGNAL_RECENT_DAYS = 7
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"          # 新增：数据缓存目录
VP_WINDOW = 60                             # 新增：滚动 Volume Profile 窗口（交易日）
VP_BINS = 40                               # 新增：剖面分档数
BB_PERIOD = 20
BB_STD = 2.0
STATS_HORIZONS = (5, 10, 20)               # 新增：信号后验持有期（交易日）

INDICATORS = ["RSI", "MACD_H", "STOCH_K", "MOM", "CMF", "MFI", "OBV", "CCI"]


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macd(c, f=12, s=26, g=9):
    dif = ema(c, f) - ema(c, s)
    dea = dif.ewm(span=g, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def stoch(h, l, c, n=14, d=3):
    k = 100 * (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min())
    return k, k.rolling(d).mean()


def mom(c, n=10):
    return c.diff(n)


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    m = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - m) / (0.015 * md)


def cmf(h, l, c, v, n=20):
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    return (mfm * v).rolling(n).sum() / v.rolling(n).sum()


def mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3
    rmf = tp * v
    d = tp.diff()
    pos = rmf.where(d > 0, 0).rolling(n).sum()
    neg = rmf.where(d < 0, 0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg)


def obv(c, v):
    return (np.sign(c.diff()).fillna(0) * v).cumsum()


def vwma(c, v, n=20):
    return (c * v).rolling(n).sum() / v.rolling(n).sum()


def bollinger_bands(close, n=BB_PERIOD, k=BB_STD):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def bollinger_position(close, mid, upper, lower):
    width = (upper - lower).replace(0, np.nan)
    return (close - mid) / width


# ============ 数据（新增：本地缓存 + 增量更新） ============
_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")


def _flatten_yf_columns(df):
    if df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df
    df = df.copy()
    level0 = df.columns.get_level_values(0).astype(str).str.lower()
    if level0.isin(("open", "high", "low", "close", "volume", "adj close")).any():
        df.columns = level0
    else:
        df.columns = df.columns.get_level_values(-1).astype(str)
    return df


def _to_ohlcv(df):
    """Normalize yfinance / cache frames to standard OHLCV columns."""
    if df.empty:
        return df
    df = _flatten_yf_columns(df)
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "")
        if key in ("open", "high", "low", "close", "volume"):
            rename[c] = key.capitalize() if key != "close" else "Close"
    df = df.rename(columns=rename)
    missing = [c for c in _OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns {missing}: got {list(df.columns)}")
    out = df.loc[:, _OHLCV_COLS].copy()
    out.index = pd.to_datetime(out.index)
    return out.dropna()

def resample_ohlcv(df, rule="W-FRI"):
    """日线 OHLCV 聚合为周线（默认周五收盘为一周结束），供周线模式使用"""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg).dropna()
    return out

def load_ohlcv(ticker, period=PERIOD, use_cache=True):
    """先读本地缓存；有缓存则只增量拉取新增交易日，合并去重后写回。"""
    cache_path = CACHE_DIR / f"{safe_chart_name(ticker)}_{period}.csv"
    if use_cache and cache_path.exists():
        cached = _to_ohlcv(pd.read_csv(cache_path, index_col=0, parse_dates=True))
        last = pd.Timestamp(cached.index[-1]).normalize()
        inc_start = last + pd.Timedelta(days=1)
        today = pd.Timestamp.today().normalize()
        if inc_start <= today:
            try:
                new_raw = yf.download(
                    ticker,
                    start=inc_start.date().isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                new = _to_ohlcv(new_raw) if new_raw is not None and not new_raw.empty else pd.DataFrame()
                if not new.empty:
                    df = pd.concat([cached, new])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                else:
                    df = cached
            except Exception:
                df = cached
        else:
            df = cached
    else:
        raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        df = _to_ohlcv(raw)

    if df.empty:
        raise ValueError(f"no data for {ticker}")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
    return df


def add_indicators(df):
    df = df.copy()
    df["RSI"] = rsi(df.Close)
    df["MACD_H"] = macd(df.Close)[2]
    df["STOCH_K"] = stoch(df.High, df.Low, df.Close)[0]
    df["MOM"] = mom(df.Close)
    df["CCI"] = cci(df.High, df.Low, df.Close)
    df["CMF"] = cmf(df.High, df.Low, df.Close, df.Volume)
    df["MFI"] = mfi(df.High, df.Low, df.Close, df.Volume)
    df["OBV"] = obv(df.Close, df.Volume)
    df["VWMA"] = vwma(df.Close, df.Volume)
    return df


def find_divergences(df, ind, order=PIVOT_ORDER):
    lows = argrelextrema(df.Low.values, np.less, order=order)[0]
    highs = argrelextrema(df.High.values, np.greater, order=order)[0]
    bull, bear = [], []
    for pts, is_low, out in ((lows, True, bull), (highs, False, bear)):
        for a, b in zip(pts[:-1], pts[1:]):
            if is_low:
                if df.Low.iat[a] > df.Low.iat[b] and ind.iat[a] < ind.iat[b]:
                    out.append(b)
            else:
                if df.High.iat[a] < df.High.iat[b] and ind.iat[a] > ind.iat[b]:
                    out.append(b)
    return bull, bear


def dedup_signals(signals, df, gap=DEDUP_BARS):
    if len(signals) == 0:
        return signals
    keep = []
    group = [signals.index[0]]
    for idx in signals.index[1:]:
        if df.index.get_loc(idx) - df.index.get_loc(group[-1]) <= gap:
            group.append(idx)
        else:
            keep.append(max(group, key=lambda i: signals[i]))
            group = [idx]
    keep.append(max(group, key=lambda i: signals[i]))
    return signals.loc[keep]


def compute_signals(df):
    bull_s = pd.Series(0, index=df.index)
    bear_s = pd.Series(0, index=df.index)
    for n in INDICATORS:
        b, r = find_divergences(df, df[n])
        bull_s.iloc[b] += 1
        bear_s.iloc[r] += 1
    bull_sig = dedup_signals(bull_s[bull_s >= MIN_COUNT], df)
    bear_sig = dedup_signals(bear_s[bear_s >= MIN_COUNT], df)
    return bull_sig, bear_sig


# ============ 信号历史统计（新增） ============
def signal_stats(df, sig, horizons=STATS_HORIZONS):
    """信号触发后持有 h 根K线的收益统计：胜率/均值/中位数/极值。"""
    rows = []
    closes = df.Close.values
    for h in horizons:
        rets = []
        for i, v in sig.items():
            xi = df.index.get_loc(i)
            if xi + h < len(df):
                rets.append(closes[xi + h] / closes[xi] - 1)
        rets = np.array(rets) if rets else np.array([np.nan])
        rows.append({
            "持有期": f"{h}根K线",
            "信号数": int(np.isfinite(rets).sum()),
            "胜率": float((rets > 0).mean()),
            "平均收益": float(np.nanmean(rets)),
            "中位收益": float(np.nanmedian(rets)),
            "最好": float(np.nanmax(rets)),
            "最差": float(np.nanmin(rets)),
        })
    return pd.DataFrame(rows)


def fib_levels(df, lookback=LOOKBACK):
    seg = df if lookback is None else df.iloc[-lookback:]
    hi, lo = seg.High.max(), seg.Low.min()
    return {r: hi - (hi - lo) * r for r in [0.236, 0.382, 0.5, 0.618]}


# ============ 滚动 Volume Profile（新增，无未来数据） ============
def _profile_core(tp_vals, vol_vals, n_bins, va_pct):
    """给定一段典型价/成交量，返回 (档位中心, 各档量, POC, VAH, VAL)。"""
    lo_b, hi_b = tp_vals.min(), tp_vals.max()
    if hi_b <= lo_b or vol_vals.sum() <= 0:
        return None
    bins = np.linspace(lo_b, hi_b, n_bins + 1)
    bi = np.clip(np.digitize(tp_vals, bins) - 1, 0, n_bins - 1)
    vol = np.zeros(n_bins)
    np.add.at(vol, bi, vol_vals)
    c = (bins[:-1] + bins[1:]) / 2
    poc_i = int(np.argmax(vol))
    lo_i = hi_i = poc_i
    acc = vol[poc_i]
    total = vol.sum()
    while acc < va_pct * total and (lo_i > 0 or hi_i < n_bins - 1):
        dl = vol[lo_i - 1] if lo_i > 0 else -1
        dr = vol[hi_i + 1] if hi_i < n_bins - 1 else -1
        if dr >= dl:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    return c, vol, c[poc_i], c[hi_i], c[lo_i]


def rolling_volume_profile(df, window=VP_WINDOW, n_bins=VP_BINS, va_pct=0.70):
    """第 t 行的 POC/VAH/VAL 只用到 t-window+1..t 的数据，无前视偏差，可作回测特征。"""
    tp = ((df.High + df.Low + df.Close) / 3).values
    vol = df.Volume.values
    n = len(df)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    for t in range(window - 1, n):
        r = _profile_core(tp[t - window + 1:t + 1], vol[t - window + 1:t + 1], n_bins, va_pct)
        if r is not None:
            _, _, poc[t], vah[t], val[t] = r
    return pd.DataFrame({"POC": poc, "VAH": vah, "VAL": val}, index=df.index)


def volume_profile(df, n_bins=VP_BINS, va_pct=0.70):
    """整段剖面（右侧面板用）。取最后一个滚动窗口，避免全样本前视。"""
    tp = ((df.High + df.Low + df.Close) / 3).values
    return _profile_core(tp[-VP_WINDOW:], df.Volume.values[-VP_WINDOW:], n_bins, va_pct)


def safe_chart_name(ticker):
    return re.sub(r'[^\w.\-]+', "_", ticker)


def _set_date_xticks(ax, index, max_ticks=12):
    """Integer bar index on x for plotting; show calendar dates on the axis."""
    n = len(index)
    if n == 0:
        return
    step = max(1, n // max_ticks)
    pos = list(range(0, n, step))
    if pos[-1] != n - 1:
        pos.append(n - 1)
    labels = [pd.Timestamp(index[i]).strftime("%Y-%m-%d") for i in pos]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


def signals_in_window(bull_sig, bear_sig, df, ticker, days=SIGNAL_RECENT_DAYS):
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    rows = []
    for idx, v in bull_sig.items():
        ts = pd.Timestamp(idx).normalize()
        if ts >= cutoff:
            rows.append(
                {
                    "ticker": ticker,
                    "signal": "底背离",
                    "signal_en": "bullish_divergence",
                    "date": ts.date().isoformat(),
                    "indicator_count": int(v),
                    "close": round(float(df.loc[idx, "Close"]), 4),
                }
            )
    for idx, v in bear_sig.items():
        ts = pd.Timestamp(idx).normalize()
        if ts >= cutoff:
            rows.append(
                {
                    "ticker": ticker,
                    "signal": "顶背离",
                    "signal_en": "bearish_divergence",
                    "date": ts.date().isoformat(),
                    "indicator_count": int(v),
                    "close": round(float(df.loc[idx, "Close"]), 4),
                }
            )
    return rows


def plot_chart(df, ticker, bull_sig, bear_sig, fib, save_path, rvp=None):
    x = np.arange(len(df))
    bb_mid, bb_upper, bb_lower = bollinger_bands(df.Close)
    bb_pos = bollinger_position(df.Close, bb_mid, bb_upper, bb_lower)

    fig, (ax, ax_bb) = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        dpi=110,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06},
    )
    up = df.Close >= df.Open
    ax.vlines(x[~up], df.Low[~up], df.High[~up], color="#ef5350", lw=0.8)
    ax.vlines(x[up], df.Low[up], df.High[up], color="#26a69a", lw=0.8)
    for i in x:
        c = "#26a69a" if up.iat[i] else "#ef5350"
        ax.add_patch(
            Rectangle(
                (i - 0.35, min(df.Open.iat[i], df.Close.iat[i])),
                0.7,
                abs(df.Close.iat[i] - df.Open.iat[i]) + 1e-9,
                facecolor=c,
                edgecolor=c,
            )
        )
    ax.plot(x, df.Close.rolling(10).mean(), color="white", lw=1, label="MA10")
    ax.plot(x, df.Close.rolling(20).mean(), color="#42a5f5", lw=1, label="MA20")
    ax.plot(x, df.VWMA, color="#ff9800", lw=1, label="VWMA20")
    ax.fill_between(x, bb_lower, bb_upper, color="#78909c", alpha=0.12, label=f"布林带({BB_PERIOD},{BB_STD}σ)")
    ax.plot(x, bb_mid, color="#78909c", lw=0.9, ls="--", alpha=0.85)
    ax.plot(x, bb_upper, color="#78909c", lw=0.65, alpha=0.75)
    ax.plot(x, bb_lower, color="#78909c", lw=0.65, alpha=0.75)
    # 新增：滚动 POC / VA 带
    if rvp is not None:
        ax.plot(x, rvp.POC, color="#ff9800", lw=0.8, alpha=0.5, label=f"滚动POC({VP_WINDOW}d)")
        ax.fill_between(x, rvp.VAL, rvp.VAH, color="#26a69a", alpha=0.08,
                        label=f"滚动VA({VP_WINDOW}d)")
    for r, p in fib.items():
        ax.axhline(p, ls="--", lw=0.7, color="#ff9800", alpha=0.7)
        ax.text(1, p, f" {r} ({p:,.0f})", fontsize=8, color="#ff9800", va="bottom")
    for i, v in bull_sig.items():
        xi = df.index.get_loc(i)
        ax.annotate(
            f"底背离x{int(v)}",
            (xi, df.Low.iat[xi]),
            xytext=(xi, df.Low.iat[xi] * 0.985),
            fontsize=8,
            ha="center",
            bbox=dict(boxstyle="round", fc="gold"),
        )
    for i, v in bear_sig.items():
        xi = df.index.get_loc(i)
        ax.annotate(
            f"顶背离x{int(v)}",
            (xi, df.High.iat[xi]),
            xytext=(xi, df.High.iat[xi] * 1.015),
            fontsize=8,
            ha="center",
            bbox=dict(boxstyle="round", fc="#ce93d8"),
        )
    centers, vol, poc, vah, val = volume_profile(df)
    xr = len(df) * 1.02
    vmax = vol.max()
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    scale = vol / vmax * len(df) * 0.13
    in_va = (centers >= val) & (centers <= vah)
    ax.barh(centers[in_va], scale[in_va], left=xr, color="#26a69a", alpha=0.75)
    ax.barh(centers[~in_va], scale[~in_va], left=xr, color="#26a69a", alpha=0.25)
    ax.axhline(poc, color="#ff9800", lw=1.2)
    ax.text(xr, poc, f" POC {poc:,.0f}", fontsize=9, color="#ff9800", va="bottom")
    ax.axhline(vah, color="#26a69a", lw=0.9, ls=":")
    ax.text(xr, vah, f" VAH {vah:,.0f}", fontsize=8, color="#26a69a", va="bottom")
    ax.axhline(val, color="#ef5350", lw=0.9, ls=":")
    ax.text(xr, val, f" VAL {val:,.0f}", fontsize=8, color="#ef5350", va="bottom")
    ax.set_xlim(0, len(df) * 1.18)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{ticker} 背离+斐波那契+布林带+滚动Volume Profile")

    ax_bb.plot(x, bb_pos, color="#42a5f5", lw=1, label="(Close-mid)/(upper-lower)")
    ax_bb.axhline(0, color="#ef5350", lw=0.7, alpha=0.85)
    ax_bb.axhline(0.5, color="#78909c", lw=0.7, ls=":", alpha=0.85)
    ax_bb.axhline(1, color="#26a69a", lw=0.7, alpha=0.85)
    ax_bb.set_xlim(0, len(df) * 1.18)
    ax_bb.set_ylabel("(C-mid)/(U-L)", fontsize=8)
    _set_date_xticks(ax_bb, df.index)
    ax_bb.set_xlabel("Date")
    ax_bb.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ============ 新增：信号后验收益分布图 ============
def plot_signal_returns(df, ticker, bull_sig, bear_sig, save_path,
                        horizons=STATS_HORIZONS):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=110)
    closes = df.Close.values
    # 三个持有期用不同色块，避免仅靠线型区分
    H_COLORS = ["#1b9e77", "#7570b3", "#d95f02"]  # 绿/紫/橙，色盲友好
    for ax_, (sig, title, base_color) in zip(
        axes, [(bull_sig, "底背离后验收益", "#26a69a"),
               (bear_sig, "顶背离后验收益", "#ef5350")]
    ):
        for h, hcol in zip(horizons, H_COLORS):
            rets = [
                closes[df.index.get_loc(i) + h] / closes[df.index.get_loc(i)] - 1
                for i in sig.index if df.index.get_loc(i) + h < len(df)
            ]
            if rets:
                ax_.hist(rets, bins=min(10, len(rets)), alpha=0.55, label=f"{h}日",
                         color=hcol, edgecolor="black")
        ax_.axvline(0, color="k", lw=0.8)
        ax_.set_title(title)
        ax_.set_xlabel("后验收益率")
        ax_.legend()
    fig.suptitle(f"{ticker} 信号后验收益分布")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def process_ticker(ticker, out_dir):
    df = add_indicators(load_ohlcv(ticker))
    bull_sig, bear_sig = compute_signals(df)
    fib = fib_levels(df)
    rvp = rolling_volume_profile(df)                       # 新增
    chart_path = out_dir / f"{safe_chart_name(ticker)}.png"
    plot_chart(df, ticker, bull_sig, bear_sig, fib, chart_path, rvp=rvp)

    # 新增：信号历史统计 + 后验分布图
    stats_bull = signal_stats(df, bull_sig)
    stats_bear = signal_stats(df, bear_sig)
    stats_path = out_dir / f"stats_{safe_chart_name(ticker)}.csv"
    pd.concat([stats_bull.assign(方向="底背离", ticker=ticker),
               stats_bear.assign(方向="顶背离", ticker=ticker)]).to_csv(stats_path, index=False)
    ret_chart_path = out_dir / f"returns_{safe_chart_name(ticker)}.png"
    plot_signal_returns(df, ticker, bull_sig, bear_sig, ret_chart_path)

    recent = signals_in_window(bull_sig, bear_sig, df, ticker)
    return {
        "ticker": ticker,
        "chart_path": str(chart_path),
        "returns_chart_path": str(ret_chart_path),
        "stats_path": str(stats_path),
        "stats_bull": stats_bull,                          # 新增
        "stats_bear": stats_bear,                          # 新增
        "bull_all": [(str(i.date()), int(v)) for i, v in bull_sig.items()],
        "bear_all": [(str(i.date()), int(v)) for i, v in bear_sig.items()],
        "fib": {f"fib {r}": round(p, 2) for r, p in fib.items()},
        "recent": recent,
        "error": None,
    }


def run_batch(tickers, out_dir=None):
    if out_dir is None:
        out_dir = SCRIPT_DIR / date.today().isoformat()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    recent_rows = []
    errors = []

    for ticker in tickers:
        try:
            r = process_ticker(ticker, out_dir)
            results.append(r)
            recent_rows.extend(r["recent"])
            print(f"[ok] {ticker} -> {r['chart_path']}")
            print(f"     底背离: {r['bull_all']}")
            print(f"     顶背离: {r['bear_all']}")
            # 新增：打印信号后验统计
            print(f"     后验统计(底背离):\n{r['stats_bull'].to_string(index=False)}")
            print(f"     后验统计(顶背离):\n{r['stats_bear'].to_string(index=False)}")
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            print(f"[fail] {ticker}: {e}")

    alerts_path = out_dir / "signals_recent.csv"
    if recent_rows:
        alerts_df = pd.DataFrame(recent_rows).sort_values(["date", "ticker"], ascending=[False, True])
    else:
        alerts_df = pd.DataFrame(
            columns=["ticker", "signal", "signal_en", "date", "indicator_count", "close"]
        )
    alerts_df.to_csv(alerts_path, index=False)

    # 新增：汇总所有标的的信号统计表
    stats_rows = []
    for r in results:
        for _, row in r["stats_bull"].assign(ticker=r["ticker"], 方向="底背离").iterrows():
            stats_rows.append(row)
        for _, row in r["stats_bear"].assign(ticker=r["ticker"], 方向="顶背离").iterrows():
            stats_rows.append(row)
    all_stats_path = out_dir / "signal_stats_all.csv"
    pd.DataFrame(stats_rows).to_csv(all_stats_path, index=False)

    summary_rows = []
    recent_by_ticker = {t: [r for r in recent_rows if r["ticker"] == t] for t in tickers}
    for r in results:
        rec = recent_by_ticker.get(r["ticker"], [])
        summary_rows.append(
            {
                "ticker": r["ticker"],
                "alert_past_7d": bool(rec),
                "signals_past_7d": len(rec),
                "chart": r["chart_path"],
            }
        )
    for err in errors:
        summary_rows.append(
            {"ticker": err["ticker"], "alert_past_7d": False, "signals_past_7d": 0, "chart": "", "error": err["error"]}
        )
    summary_path = out_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print()
    print(f"Charts folder: {out_dir}")
    print(f"Recent signals (past {SIGNAL_RECENT_DAYS} days): {alerts_path}")
    if not alerts_df.empty:
        print("\n=== Alerts (past 7 days) ===")
        print(alerts_df.to_string(index=False))
    else:
        print(f"\nNo divergence signals in the past {SIGNAL_RECENT_DAYS} days for this ticker list.")
    print(f"Signal stats: {all_stats_path}")
    print(f"Summary: {summary_path}")
    return out_dir, alerts_df, results


if __name__ == "__main__":
    run_batch(TICKERS)
