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
    "EEM"
]
PERIOD = "2y"
PIVOT_ORDER = 6
MIN_COUNT = 3
DEDUP_BARS = 10
LOOKBACK = 160
SIGNAL_RECENT_DAYS = 7
SCRIPT_DIR = Path(__file__).resolve().parent

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


def load_ohlcv(ticker, period=PERIOD):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna()
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


def fib_levels(df, lookback=LOOKBACK):
    seg = df if lookback is None else df.iloc[-lookback:]
    hi, lo = seg.High.max(), seg.Low.min()
    return {r: hi - (hi - lo) * r for r in [0.236, 0.382, 0.5, 0.618]}


def trendline(df, kind="low", n_pts=3, order=8):
    col = df.Low if kind == "low" else df.High
    pts = argrelextrema(col.values, np.less if kind == "low" else np.greater, order=order)[0]
    if len(pts) < 2:
        return None
    pts = pts[-n_pts:]
    return np.polyfit(pts, col.values[pts], 1)


def volume_profile(df, n_bins=40, va_pct=0.70):
    tp = (df.High + df.Low + df.Close) / 3
    bins = np.linspace(df.Low.min(), df.High.max(), n_bins + 1)
    idx = np.clip(np.digitize(tp, bins) - 1, 0, n_bins - 1)
    vol = np.zeros(n_bins)
    for i, v in zip(idx, df.Volume):
        vol[i] += v
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


def safe_chart_name(ticker):
    return re.sub(r'[^\w.\-]+', "_", ticker)


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


def plot_chart(df, ticker, bull_sig, bear_sig, fib, save_path):
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
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
    for r, p in fib.items():
        ax.axhline(p, ls="--", lw=0.7, color="#ff9800", alpha=0.7)
        ax.text(1, p, f" {r} ({p:,.0f})", fontsize=8, color="#ff9800", va="bottom")
    for kind, color in (("low", "#ffeb3b"), ("high", "#ab47bc")):
        tl = trendline(df, kind)
        if tl is not None:
            k, b = tl
            ax.plot(x, k * x + b, color=color, lw=1.2, ls="-.", alpha=0.8)
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
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{ticker} 背离+斐波那契+趋势线+Volume Profile")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def process_ticker(ticker, out_dir):
    df = add_indicators(load_ohlcv(ticker))
    bull_sig, bear_sig = compute_signals(df)
    fib = fib_levels(df)
    chart_path = out_dir / f"{safe_chart_name(ticker)}.png"
    plot_chart(df, ticker, bull_sig, bear_sig, fib, chart_path)
    recent = signals_in_window(bull_sig, bear_sig, df, ticker)
    return {
        "ticker": ticker,
        "chart_path": str(chart_path),
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
    print(f"Summary: {summary_path}")
    return out_dir, alerts_df, results


if __name__ == "__main__":
    run_batch(TICKERS)
