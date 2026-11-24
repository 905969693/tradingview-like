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
PIVOT_ORDER = 3
MIN_COUNT = 3
DEDUP_BARS = 10
SIGNAL_RECENT_DAYS = 30
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"          # 新增：数据缓存目录
BB_PERIOD = 20
BB_STD = 2.0
STATS_HORIZONS = (5, 10, 20)               # 汇总表与回测用的持有期
POSTERIOR_SPAN = tuple(range(3, 41))       # 分布图：3 到 40 根（约两个月交易日）
USE_TREND_FILTER = True                    # 新增：批处理信号是否先过顺势滤波
TREND_MA_N = 200                           # 新增：顺势滤波均线周期（日线 200 / 周线 40）
# Dashboard @st.cache_data 用此版本号失效旧缓存（改 apply_trend_filter 时递增）
TREND_FILTER_LOGIC_REV = 3
# 改去重或末端 pivot 规则时递增，避免 Streamlit 沿用旧信号缓存
DEDUP_LOGIC_REV = 3

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

def resample_ohlcv(df, rule="W-FRI", drop_partial=True):
    """日线 OHLCV 聚合为周线（默认周五收盘为一周结束）。

    最后一周若还没到标签日（例如周中），默认去掉。未走完的周 K 不能用来确认 pivot。
    """
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg).dropna()
    if (
        drop_partial
        and len(out)
        and len(df)
        and pd.Timestamp(out.index[-1]).normalize()
        > pd.Timestamp(df.index[-1]).normalize()
    ):
        out = out.iloc[:-1]
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


def _swing_points(values, comparator, order):
    """左右各有 order 根真实 K 线时才保留摆动点。

    argrelextrema 在序列两端会用最后一根重复填充，末端会提前出现 pivot。
    """
    idx = argrelextrema(np.asarray(values, dtype=float), comparator, order=order)[0]
    if len(idx) == 0 or order <= 0:
        return idx
    n = len(values)
    return idx[(idx >= order) & (idx <= n - 1 - order)]


def td_sequential_counts(close, lookback=4, show_from=5):
    """神奇九转计数。

    上涨：收盘高于 lookback 根前的收盘。下跌：收盘低于 lookback 根前的收盘。
    已经走完的历史段只标注完整的 1–9。贴着最后一根、尚未中断的计数，
    连到 show_from 根才标注 1..n；中途断在 show_from 之前则不标注。
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    up = np.zeros(n, dtype=int)
    down = np.zeros(n, dtype=int)
    up_mask = np.zeros(n, dtype=bool)
    down_mask = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        if c[i] > c[i - lookback]:
            up_mask[i] = True
        elif c[i] < c[i - lookback]:
            down_mask[i] = True

    def _paint(mask, out):
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            start = i
            while i < n and mask[i]:
                i += 1
            length = i - start
            pos = start
            left = length
            while left >= 9:
                for k in range(9):
                    out[pos + k] = k + 1
                pos += 9
                left -= 9
            if left >= show_from and i == n:
                for k in range(left):
                    out[pos + k] = k + 1

    _paint(up_mask, up)
    _paint(down_mask, down)
    index = close.index if hasattr(close, "index") else None
    return pd.Series(up, index=index), pd.Series(down, index=index)


def td_completed_nines(close, lookback=4):
    """已经走完的九转。返回 (下跌九转, 上涨九转)。

    索引是计数正好为 9 的那根 K 线。贴在样本末端、还没数到 9 的计数不会出现。
    入场在下一根收盘，由调用方把 entry_shift 设为 1。
    """
    up, down = td_sequential_counts(close, lookback=lookback)
    return down[down == 9], up[up == 9]


def td_current_counts(close, lookback=4):
    """最后一根 K 线上，上涨和下跌九转的当前计数（1–9，没有则为 0）。

    与 td_sequential_counts 同一套比较：收盘高于 lookback 根前计入上涨，
    低于则计入下跌。连续段每满 9 根重新起计；当前计数是贴着最后一根、
    尚未中断的这一段已经走过的根数（正好走完 9 则为 9）。不受 show_from
    影响，图上未标注的 1–4 也会返回。
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    if n <= lookback:
        return 0, 0

    def _trailing(higher):
        length = 0
        for i in range(n - 1, lookback - 1, -1):
            if higher and c[i] > c[i - lookback]:
                length += 1
            elif (not higher) and c[i] < c[i - lookback]:
                length += 1
            else:
                break
        if length == 0:
            return 0
        remainder = length % 9
        return 9 if remainder == 0 else int(remainder)

    return _trailing(True), _trailing(False)


def market_session_close(ticker=None):
    """该标的常规收盘所用的时区和整点小时。无法识别时按美股 16:00。"""
    name = str(ticker or "").upper().replace("-", "_")
    if name.endswith(".HK"):
        return "Asia/Hong_Kong", 16
    if name.endswith(".SI"):
        return "Asia/Singapore", 17
    if name.endswith(".SS") or name.endswith(".SZ"):
        return "Asia/Shanghai", 15
    if name.endswith(".KS") or name.endswith(".KQ") or name.endswith("KS11"):
        return "Asia/Seoul", 16
    if name.endswith(".T"):
        return "Asia/Tokyo", 15
    if name.endswith("=X") or name.endswith("_X"):
        return "America/New_York", 17
    return "America/New_York", 16


def drop_incomplete_daily_bar(df, now=None, ticker=None):
    """若最后一根日 K 的日期仍处于该标的未收盘的常规交易日，则去掉。

    按标的交易所时区判断，不把已收盘的港股、新交所日 K 当成美股盘中半根。
    Yahoo 增量更新在盘中可能写入当天半根，九转提醒只用已经走完的日 K。
    """
    if df is None or len(df) == 0:
        return df
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("Asia/Hong_Kong")
        else:
            now = now.tz_convert("UTC")
    last = pd.Timestamp(df.index[-1]).tz_localize(None).normalize()
    zone, close_hour = market_session_close(ticker)
    local = now.tz_convert(zone)
    if last != pd.Timestamp(local.date()):
        return df
    session_close = local.normalize() + pd.Timedelta(hours=close_hour)
    if local < session_close:
        return df.iloc[:-1]
    return df


def bars_for_complete_session(daily_df, timeframe="日线", now=None, ticker=None):
    """日线去掉尚未收盘的当天；周线再用完整周五周 K（去掉未走完的一周）。"""
    daily_df = drop_incomplete_daily_bar(daily_df, now=now, ticker=ticker)
    if timeframe == "周线":
        return resample_ohlcv(daily_df, drop_partial=True)
    return daily_df


def find_divergences(df, ind, order=PIVOT_ORDER):
    lows = _swing_points(df.Low.values, np.less, order)
    highs = _swing_points(df.High.values, np.greater, order)
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


def _signal_positions(df, sig):
    """信号索引在 df 里的位置。缺失的时间戳丢掉。"""
    if sig is None or len(sig) == 0:
        return np.array([], dtype=int)
    pos = df.index.get_indexer(sig.index)
    return pos[pos >= 0]


def confirmed_nine_overlap_pairs(df, div_sig, nine_sig, pivot_order):
    """九转第 7、8、9 根和背离点左右各 pivot_order 根的每一对重合。

    返回列 pivot、nine、entry。entry 是较晚的一根：九转第 9 根，或背离确认的最后一根。
    pivot_order 用调用方传入的数。
    """
    empty = pd.DataFrame(columns=["pivot", "nine", "entry"])
    nine_at = _signal_positions(df, nine_sig)
    div_at = _signal_positions(df, div_sig)
    if nine_at.size == 0 or div_at.size == 0:
        return empty
    order = int(pivot_order)
    n = len(df)
    rows = []
    for today in nine_at:
        today = int(today)
        if today < 2:
            continue
        nine_bars = (today - 2, today - 1, today)
        for pivot in div_at:
            pivot = int(pivot)
            left = pivot - order
            right = pivot + order
            if right >= n or not any(left <= bar <= right for bar in nine_bars):
                continue
            rows.append({
                "pivot": df.index[pivot],
                "nine": df.index[today],
                "entry": df.index[max(today, right)],
            })
    if not rows:
        return empty
    return pd.DataFrame(rows)


def confirmed_nine_overlap(df, div_sig, nine_sig, pivot_order):
    """重合的入场 bar。同一根入场只保留一次，调用方把 entry_shift 设为 0。"""
    pairs = confirmed_nine_overlap_pairs(df, div_sig, nine_sig, pivot_order)
    if pairs.empty:
        return pd.Series(dtype=float)
    entry_idx = pd.DatetimeIndex(pairs["entry"]).unique().sort_values()
    return pd.Series(1, index=entry_idx)


def _naive_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def nine_overlap_note(pairs, pivot_ts, pivot_order):
    """这条背离若参与了九转配合，返回说明；否则返回空字符串。"""
    if pairs is None or len(pairs) == 0:
        return ""
    day = _naive_timestamp(pivot_ts).normalize()
    pivots = pd.DatetimeIndex(pd.to_datetime(pairs["pivot"]).map(_naive_timestamp)).normalize()
    hit = pairs.loc[pivots == day]
    if hit.empty:
        return ""
    entry = _naive_timestamp(hit["entry"].iloc[-1])
    return (
        f"九转配合已确认背离：计数 7、8、9 与该背离点左右各 {int(pivot_order)} 根重合。"
        f"入场 {entry.date().isoformat()}，取较晚的一根收盘"
        "（九转第 9 根，或背离确认的最后一根）。"
        "这是对这条已确认背离的加强。"
    )


def dedup_signals(signals, df, gap=DEDUP_BARS):
    """保留先出现的信号，其后 gap 根 K 线内的信号丢弃。

    决定只依赖已经出现的信号：不回头比较整簇的共振数，因此不会用到更晚的信号。
    """
    if len(signals) == 0:
        return signals
    keep = []
    last_loc = None
    for idx in signals.index:
        loc = int(df.index.get_loc(idx))
        if last_loc is not None and loc - last_loc <= gap:
            continue
        keep.append(idx)
        last_loc = loc
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


def raw_deduped_signals(df, pivot_order=PIVOT_ORDER, min_count=MIN_COUNT,
                        dedup_bars=DEDUP_BARS):
    """去重后的顶/底背离序列（与 Dashboard 扫描参数一致）。"""
    d = add_indicators(df)
    bull_s = pd.Series(0, index=d.index)
    bear_s = pd.Series(0, index=d.index)
    for n in INDICATORS:
        b, r = find_divergences(d, d[n], order=pivot_order)
        bull_s.iloc[b] += 1
        bear_s.iloc[r] += 1
    bull = dedup_signals(bull_s[bull_s >= min_count], d, gap=dedup_bars)
    bear = dedup_signals(bear_s[bear_s >= min_count], d, gap=dedup_bars)
    return bull, bear


def trading_days_between(daily_index, ts_a, ts_b):
    """两时间点在日线序列上的 bar 序号差（绝对值）。"""
    idx = pd.DatetimeIndex(daily_index)

    def _loc(ts):
        t = pd.Timestamp(ts).normalize()
        if t in idx:
            return idx.get_loc(t)
        i = int(idx.searchsorted(t, side="left"))
        return min(max(i, 0), len(idx) - 1)

    return abs(_loc(ts_a) - _loc(ts_b))


def match_daily_weekly_signals(daily_sig, weekly_sig, daily_index,
                               tolerance=5, recent_days=7):
    """
    在 recent_days 窗口内的日线信号中，查找与任一周线信号
    相距 <= tolerance 交易日的配对。返回最佳（最新日线）配对或 None。
    """
    if len(daily_sig) == 0 or len(weekly_sig) == 0:
        return None
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=recent_days)
    best = None
    for d_idx, d_val in daily_sig.items():
        d_ts = pd.Timestamp(d_idx).normalize()
        if d_ts < cutoff:
            continue
        for w_idx, w_val in weekly_sig.items():
            gap = trading_days_between(daily_index, d_idx, w_idx)
            if gap <= tolerance:
                row = {
                    "daily_date": d_ts.date().isoformat(),
                    "weekly_date": pd.Timestamp(w_idx).date().isoformat(),
                    "gap_days": gap,
                    "daily_count": int(d_val),
                    "weekly_count": int(w_val),
                }
                if best is None or d_ts > pd.Timestamp(best["daily_date"]):
                    best = row
    return best


DUAL_TF_TOLERANCE_DAYS = 5


def dual_timeframe_divergence_matches(
    daily_df,
    pivot_order=PIVOT_ORDER,
    min_count=MIN_COUNT,
    dedup_bars=DEDUP_BARS,
    tolerance=DUAL_TF_TOLERANCE_DAYS,
    recent_days=7,
):
    """
    日线 + 周线均出现同向背离，且两周期信号日期在 daily_df 上相差 <= tolerance 根交易日。
    recent_days 默认 7，与 SIGNAL_RECENT_DAYS（近一个月日线告警窗口）相互独立。
    返回 (bull_match, bear_match)，各为 dict 或 None（取窗口内最新日线配对）。
    """
    weekly_df = resample_ohlcv(daily_df)
    d_bull, d_bear = raw_deduped_signals(
        daily_df, pivot_order=pivot_order, min_count=min_count, dedup_bars=dedup_bars
    )
    w_bull, w_bear = raw_deduped_signals(
        weekly_df, pivot_order=pivot_order, min_count=min_count, dedup_bars=dedup_bars
    )
    idx = daily_df.index
    bull = match_daily_weekly_signals(
        d_bull, w_bull, idx, tolerance=tolerance, recent_days=recent_days
    )
    bear = match_daily_weekly_signals(
        d_bear, w_bear, idx, tolerance=tolerance, recent_days=recent_days
    )
    return bull, bear


def _index_loc(index, ts):
    idx = pd.DatetimeIndex(index)
    t = pd.Timestamp(ts)
    if t not in idx:
        t = t.normalize()
    loc = idx.get_loc(t)
    return int(loc.start) if isinstance(loc, slice) else int(loc)


def _last_daily_loc_on_or_before(daily_index, ts):
    """周线 bar 收盘日映射到不晚于该日的最后一根日线。"""
    idx = pd.DatetimeIndex(daily_index)
    i = int(idx.searchsorted(pd.Timestamp(ts).normalize(), side="right")) - 1
    if i < 0:
        return None
    return i


def enumerate_dual_tf_resonance_events(
    daily_df,
    pivot_order=PIVOT_ORDER,
    min_count=MIN_COUNT,
    dedup_bars=DEDUP_BARS,
    tolerance=DUAL_TF_TOLERANCE_DAYS,
):
    """
    全样本日+周同向背离共振（pivot 日期相距 <= tolerance 交易日）。

    日线 pivot 在其后 pivot_order 根日线收盘才确认；
    周线 pivot 在其后 pivot_order 根周线收盘才确认。
    开仓 = 这两个确认日中更晚的那根日线收盘。
    同向事件若开仓日相距 <= dedup_bars，只保留更早的那次。
    """
    weekly_df = resample_ohlcv(daily_df)
    d_bull, d_bear = raw_deduped_signals(
        daily_df, pivot_order=pivot_order, min_count=min_count, dedup_bars=dedup_bars
    )
    w_bull, w_bear = raw_deduped_signals(
        weekly_df, pivot_order=pivot_order, min_count=min_count, dedup_bars=dedup_bars
    )
    idx = daily_df.index
    raw = []

    def _collect(direction, short, d_sig, w_sig):
        for d_idx, d_val in d_sig.items():
            for w_idx, w_val in w_sig.items():
                if trading_days_between(idx, d_idx, w_idx) > tolerance:
                    continue
                d_conf = _index_loc(idx, d_idx) + pivot_order
                w_conf = _index_loc(weekly_df.index, w_idx) + pivot_order
                if d_conf >= len(idx) or w_conf >= len(weekly_df):
                    continue
                w_daily = _last_daily_loc_on_or_before(idx, weekly_df.index[w_conf])
                if w_daily is None:
                    continue
                entry_i = max(d_conf, w_daily)
                raw.append(
                    {
                        "direction": direction,
                        "short": short,
                        "daily_date": pd.Timestamp(d_idx).date().isoformat(),
                        "weekly_date": pd.Timestamp(w_idx).date().isoformat(),
                        "daily_confirm": pd.Timestamp(idx[d_conf]).date().isoformat(),
                        "weekly_confirm": pd.Timestamp(idx[w_daily]).date().isoformat(),
                        "gap_days": trading_days_between(idx, d_idx, w_idx),
                        "daily_count": int(d_val),
                        "weekly_count": int(w_val),
                        "entry_i": entry_i,
                    }
                )

    _collect("底背离", False, d_bull, w_bull)
    _collect("顶背离", True, d_bear, w_bear)
    if not raw:
        return pd.DataFrame()

    raw.sort(key=lambda r: (r["direction"], r["entry_i"], r["daily_date"]))
    kept = []
    for row in raw:
        if not kept:
            kept.append(row)
            continue
        prev = kept[-1]
        if (
            row["direction"] == prev["direction"]
            and row["entry_i"] - prev["entry_i"] <= dedup_bars
        ):
            continue
        kept.append(row)

    return pd.DataFrame(kept)


DUAL_TF_BACKTEST_HORIZONS = (5, 10, 20, 30, 60)


def dual_tf_resonance_event_returns(
    daily_df,
    events,
    pivot_order=PIVOT_ORDER,
    horizons=DUAL_TF_BACKTEST_HORIZONS,
):
    """开仓为日线确认日与周线确认日中更晚的一根日线收盘，再按交易日持有期算收益。"""
    if events is None or len(events) == 0:
        return pd.DataFrame()
    closes = daily_df.Close.values
    idx = daily_df.index
    rows = []
    for _, ev in events.iterrows():
        entry_i = int(ev["entry_i"])
        if entry_i >= len(idx):
            continue
        short = bool(ev["short"])
        entry_date = pd.Timestamp(idx[entry_i]).date().isoformat()
        if ev["weekly_confirm"] > ev["daily_confirm"]:
            entry_by = "周线"
        elif ev["daily_confirm"] > ev["weekly_confirm"]:
            entry_by = "日线"
        else:
            entry_by = "同日"
        base = {
            "direction": ev["direction"],
            "entry_by": entry_by,
            "daily_date": ev["daily_date"],
            "weekly_date": ev["weekly_date"],
            "daily_confirm": ev["daily_confirm"],
            "weekly_confirm": ev["weekly_confirm"],
            "entry_date": entry_date,
            "gap_days": ev["gap_days"],
            "daily_count": ev["daily_count"],
            "weekly_count": ev["weekly_count"],
        }
        for h in horizons:
            exit_i = entry_i + h
            if exit_i < len(idx):
                r = closes[exit_i] / closes[entry_i] - 1
                if short:
                    r = -r
                base[f"ret_{h}d"] = float(r)
            else:
                base[f"ret_{h}d"] = np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def dual_tf_resonance_backtest_summary(trades, horizons=DUAL_TF_BACKTEST_HORIZONS):
    """按方向汇总各持有期收益统计。"""
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    group_keys = ["direction"]
    if "entry_by" in trades.columns:
        group_keys.append("entry_by")
    parts = []
    for keys, sub in trades.groupby(group_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        direction = keys[0]
        entry_by = keys[1] if len(keys) > 1 else ""
        for h in horizons:
            col = f"ret_{h}d"
            finite = sub[col].dropna()
            rets = finite.values if len(finite) else np.array([np.nan])
            row = {
                "方向": direction,
                "开仓依据": entry_by,
                "持有期": f"{h}交易日",
                "样本数": int(np.isfinite(rets).sum()),
                "胜率": float((rets > 0).mean()) if len(finite) else np.nan,
                "平均收益": float(np.nanmean(rets)),
                "中位收益": float(np.nanmedian(rets)),
                "最好": float(np.nanmax(rets)),
                "最差": float(np.nanmin(rets)),
            }
            if len(finite) and "ticker" in sub.columns:
                row["最好样本"] = (
                    f"{sub.loc[finite.idxmax(), 'ticker']} {sub.loc[finite.idxmax(), 'entry_date']}"
                )
                row["最差样本"] = (
                    f"{sub.loc[finite.idxmin(), 'ticker']} {sub.loc[finite.idxmin(), 'entry_date']}"
                )
            parts.append(row)
    return pd.DataFrame(parts)


# ============ 信号历史统计（新增） ============
def signal_returns(
    df,
    sig,
    horizon,
    entry_shift=PIVOT_ORDER,
    short=False,
    daily_df=None,
    horizon_count_daily=False,
):
    """
    pivot 确认后入场，持有 horizon 根 K 线后的收益率序列。

    df：与信号同周期（日线或周线）。horizon_count_daily=True 时 horizon 为「交易日根数」，
    入场仍按 df 上 pivot 确认（周线信号则 entry_shift 为周线根数），
    出场价取 daily_df 上 entry 所在日起再数 horizon 根日线。
    """
    rets = []
    if horizon_count_daily and daily_df is not None and len(daily_df):
        closes_d = daily_df.Close.values
        idx_d = daily_df.index
        for i in sig.index:
            if i not in df.index:
                continue
            entry_i = df.index.get_loc(i) + entry_shift
            if entry_i >= len(df):
                continue
            entry_ts = pd.Timestamp(df.index[entry_i])
            entry_d = int(idx_d.searchsorted(entry_ts, side="left"))
            if entry_d >= len(idx_d):
                continue
            exit_d = entry_d + horizon
            if exit_d < len(idx_d):
                r = closes_d[exit_d] / closes_d[entry_d] - 1
                if short:
                    r = -r
                rets.append(r)
        return np.asarray(rets, dtype=float)

    closes = df.Close.values
    for i in sig.index:
        entry_i = df.index.get_loc(i) + entry_shift
        exit_i = entry_i + horizon
        if exit_i < len(df):
            r = closes[exit_i] / closes[entry_i] - 1
            if short:
                r = -r
            rets.append(r)
    return np.asarray(rets, dtype=float)


def hold_period_label(h, horizon_count_daily=False, signal_timeframe="daily"):
    if horizon_count_daily:
        return f"{h}交易日"
    if signal_timeframe == "weekly":
        return f"{h}周"
    return f"{h}根K线"


def collect_posterior_returns(
    df,
    sig,
    horizons=STATS_HORIZONS,
    entry_shift=PIVOT_ORDER,
    short=False,
    daily_df=None,
    horizon_count_daily=False,
    signal_timeframe="daily",
):
    """Strip 图用：[(持有期标签, 收益率小数), ...]"""
    rows = []
    for h in horizons:
        rets = signal_returns(
            df, sig, h, entry_shift=entry_shift, short=short,
            daily_df=daily_df, horizon_count_daily=horizon_count_daily,
        )
        lbl = hold_period_label(h, horizon_count_daily, signal_timeframe)
        for r in rets:
            rows.append((lbl, float(r)))
    return rows


def _posterior_strip_limits(all_pct):
    """对称 Y 轴范围（百分比），与参考图一致。"""
    if not all_pct:
        return (-5.0, 5.0)
    m = max(max(abs(v) for v in all_pct), 0.5)
    pad = m * 0.08
    return (-m - pad, m + pad)


def signal_stats(
    df,
    sig,
    horizons=STATS_HORIZONS,
    entry_shift=PIVOT_ORDER,
    short=False,
    daily_df=None,
    horizon_count_daily=False,
    signal_timeframe="daily",
):
    """pivot 确认后入场的收益统计：胜率/均值/中位数/极值。"""
    rows = []
    for h in horizons:
        rets = signal_returns(
            df, sig, h, entry_shift=entry_shift, short=short,
            daily_df=daily_df, horizon_count_daily=horizon_count_daily,
        )
        rets = rets if rets.size else np.array([np.nan])
        rows.append({
            "持有期": hold_period_label(h, horizon_count_daily, signal_timeframe),
            "信号数": int(np.isfinite(rets).sum()),
            "胜率": float((rets > 0).mean()),
            "平均收益": float(np.nanmean(rets)),
            "中位收益": float(np.nanmedian(rets)),
            "最好": float(np.nanmax(rets)),
            "最差": float(np.nanmin(rets)),
        })
    return pd.DataFrame(rows)


def conjunction_compare_table(
    df,
    joint_sig,
    nine_sig,
    div_sig,
    short=False,
    joint_shift=0,
    nine_shift=1,
    div_shift=PIVOT_ORDER,
    horizons=STATS_HORIZONS,
    daily_df=None,
    horizon_count_daily=False,
    signal_timeframe="daily",
):
    """共振的信号数、胜率、中位收益，并排单独九转和单独已确认背离的中位收益。"""
    kw = dict(
        horizons=horizons,
        short=short,
        daily_df=daily_df,
        horizon_count_daily=horizon_count_daily,
        signal_timeframe=signal_timeframe,
    )
    nine = signal_stats(df, nine_sig, entry_shift=nine_shift, **kw)
    div = signal_stats(df, div_sig, entry_shift=div_shift, **kw)
    nine_med = nine["中位收益"].where(nine["信号数"] > 0)
    div_med = div["中位收益"].where(div["信号数"] > 0)
    if joint_sig is None or len(joint_sig) == 0:
        return pd.DataFrame({
            "持有期": nine["持有期"],
            "共振信号数": 0,
            "共振胜率": np.nan,
            "共振中位收益": np.nan,
            "九转中位收益": nine_med,
            "背离中位收益": div_med,
        })
    joint = signal_stats(df, joint_sig, entry_shift=joint_shift, **kw)
    counted = joint["信号数"] > 0
    return pd.DataFrame({
        "持有期": joint["持有期"],
        "共振信号数": joint["信号数"],
        "共振胜率": joint["胜率"].where(counted),
        "共振中位收益": joint["中位收益"].where(counted),
        "九转中位收益": nine_med,
        "背离中位收益": div_med,
    })


def unconditional_returns(
    df,
    horizon,
    short=False,
    daily_df=None,
    horizon_count_daily=False,
):
    """同一区间里每个还能持满的 bar 都做一笔。

    入场价是该 bar 收盘。持有根数、做空时把收益取反，都和 signal_returns 一致。
    随机抽与信号数相同的时点，期望胜率和平均收益等于这里的全体结果。
    """
    rets = []
    if horizon_count_daily and daily_df is not None and len(daily_df):
        closes_d = daily_df.Close.values
        idx_d = daily_df.index
        for entry_ts in df.index:
            entry_d = int(idx_d.searchsorted(pd.Timestamp(entry_ts), side="left"))
            exit_d = entry_d + horizon
            if entry_d >= len(idx_d) or exit_d >= len(idx_d):
                continue
            entry_px = closes_d[entry_d]
            exit_px = closes_d[exit_d]
            if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px == 0:
                continue
            r = exit_px / entry_px - 1
            if short:
                r = -r
            rets.append(r)
        return np.asarray(rets, dtype=float)

    closes = df.Close.values
    last_entry = len(closes) - horizon
    for entry_i in range(max(last_entry, 0)):
        entry_px = closes[entry_i]
        exit_px = closes[entry_i + horizon]
        if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px == 0:
            continue
        r = exit_px / entry_px - 1
        if short:
            r = -r
        rets.append(r)
    return np.asarray(rets, dtype=float)


def unconditional_stats(
    df,
    horizons=STATS_HORIZONS,
    short=False,
    daily_df=None,
    horizon_count_daily=False,
    signal_timeframe="daily",
):
    """无条件基准：胜率、平均收益，以及参与计算的交易日个数。收益 > 0 为胜。"""
    rows = []
    for h in horizons:
        rets = unconditional_returns(
            df, h, short=short, daily_df=daily_df,
            horizon_count_daily=horizon_count_daily,
        )
        rows.append({
            "持有期": hold_period_label(h, horizon_count_daily, signal_timeframe),
            "对照样本": int(rets.size),
            "无条件胜率": float((rets > 0).mean()) if rets.size else np.nan,
            "无条件平均": float(rets.mean()) if rets.size else np.nan,
        })
    return pd.DataFrame(rows)


# 收缩超额胜率。样本少于大约 10 笔时只计入一部分。10 是笔数，不是持有天数。
EDGE_PRIOR_N = 10
EDGE_Z90 = 1.64
# 曲线中位数换算成百分点后，相差不足 1 个点视为并列。
METHOD_CURVE_TIE = 0.01


def edge_score(win_rate, base_rate, n, prior_n=EDGE_PRIOR_N):
    """收缩超额胜率：(信号胜率 − 无条件胜率) × n / (n + prior_n)。"""
    if n <= 0 or not np.isfinite(win_rate) or not np.isfinite(base_rate):
        return np.nan
    return float((win_rate - base_rate) * n / (n + prior_n))


def edge_lower_bound(win_rate, base_rate, n, z=EDGE_Z90):
    """超额胜率的 90% 置信下界。胜率正好是 0 或 1 时，开方项为 0。"""
    if n <= 0 or not np.isfinite(win_rate) or not np.isfinite(base_rate):
        return np.nan
    gap = float(win_rate - base_rate)
    se = float(np.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / n))
    return gap - z * se


def combine_edge_scores(long_score, long_n, short_score, short_n, long_only=False):
    """只做多时用底背离。多空都做时，两侧收缩分数按笔数加权。"""
    if long_only:
        return float(long_score) if long_n > 0 else np.nan
    parts = []
    if long_n > 0 and np.isfinite(long_score):
        parts.append((float(long_score), int(long_n)))
    if short_n > 0 and np.isfinite(short_score):
        parts.append((float(short_score), int(short_n)))
    if not parts:
        return np.nan
    weight = sum(n for _, n in parts)
    return sum(score * n for score, n in parts) / weight


def unconditional_horizon_baselines(closes, horizons, short=False):
    """每个持有期的无条件胜率和平均收益，口径与 unconditional_returns 相同。"""
    closes = np.asarray(closes, dtype=float)
    wins, means = {}, {}
    n_bars = len(closes)
    for horizon in horizons:
        horizon = int(horizon)
        last = n_bars - horizon
        if last <= 0:
            wins[horizon] = np.nan
            means[horizon] = np.nan
            continue
        entry = closes[:last]
        exit_px = closes[horizon : horizon + last]
        ok = np.isfinite(entry) & np.isfinite(exit_px) & (entry != 0)
        if not ok.any():
            wins[horizon] = np.nan
            means[horizon] = np.nan
            continue
        rets = exit_px[ok] / entry[ok] - 1.0
        if short:
            rets = -rets
        wins[horizon] = float((rets > 0).mean())
        means[horizon] = float(rets.mean())
    return wins, means


def _signal_entry_positions(df, sig, entry_shift):
    if sig is None or len(sig) == 0:
        return np.array([], dtype=int)
    pos = df.index.get_indexer(sig.index)
    keep = pos >= 0
    return pos[keep] + int(entry_shift)


def _signal_horizon_returns(closes, entry_idx, horizon, short):
    """与 signal_returns 的同周期路径一致：能持满的信号都计入，不过滤价格。"""
    closes = np.asarray(closes, dtype=float)
    entry_idx = np.asarray(entry_idx, dtype=int)
    horizon = int(horizon)
    if entry_idx.size == 0:
        return np.array([], dtype=float)
    exit_i = entry_idx + horizon
    valid = (entry_idx >= 0) & (exit_i < len(closes))
    if not valid.any():
        return np.array([], dtype=float)
    rets = closes[exit_i[valid]] / closes[entry_idx[valid]] - 1.0
    if short:
        rets = -rets
    return np.asarray(rets, dtype=float)


def _curve_row(horizon, rets, base_win, base_mean):
    rets = np.asarray(rets, dtype=float)
    n = int(np.isfinite(rets).sum()) if rets.size else 0
    win = float((rets > 0).mean()) if rets.size else np.nan
    mean_sig = float(np.mean(rets)) if rets.size and np.isfinite(rets).all() else (
        float(np.nanmean(rets)) if n else np.nan
    )
    excess = (
        mean_sig - base_mean
        if np.isfinite(mean_sig) and np.isfinite(base_mean)
        else np.nan
    )
    return {
        "持有根数": int(horizon),
        "信号数": n,
        "胜率": win,
        "无条件胜率": float(base_win) if np.isfinite(base_win) else np.nan,
        "分数": edge_score(win, base_win, n),
        "超额平均": excess,
    }


def hold_edge_curve(
    df,
    sig,
    horizons=POSTERIOR_SPAN,
    entry_shift=PIVOT_ORDER,
    short=False,
    daily_df=None,
    horizon_count_daily=False,
    baselines=None,
):
    """每个持有期一行：收缩超额胜率，以及信号平均减去无条件平均。

    baselines 是 unconditional_horizon_baselines 的返回值，用来避免重复计算对照。
    按日线计数的周线持有期仍走 signal_returns，和分布图同一口径。
    """
    horizons = tuple(int(h) for h in horizons)
    rows = []
    if horizon_count_daily and daily_df is not None and len(daily_df):
        for horizon in horizons:
            rets = signal_returns(
                df, sig, horizon, entry_shift=entry_shift, short=short,
                daily_df=daily_df, horizon_count_daily=True,
            )
            base = unconditional_returns(
                df, horizon, short=short, daily_df=daily_df, horizon_count_daily=True,
            )
            base_win = float((base > 0).mean()) if base.size else np.nan
            base_mean = float(base.mean()) if base.size else np.nan
            rows.append(_curve_row(horizon, rets, base_win, base_mean))
        return pd.DataFrame(rows)

    closes = df.Close.to_numpy(dtype=float)
    if baselines is None:
        baselines = unconditional_horizon_baselines(closes, horizons, short=short)
    base_win, base_mean = baselines
    entries = _signal_entry_positions(df, sig, entry_shift)
    for horizon in horizons:
        rets = _signal_horizon_returns(closes, entries, horizon, short)
        rows.append(_curve_row(
            horizon, rets, base_win.get(horizon, np.nan), base_mean.get(horizon, np.nan),
        ))
    return pd.DataFrame(rows)


def curve_median_edge(curve):
    """有定义的收缩超额胜率的中位数。某一根没有样本时不计入。"""
    if curve is None or curve.empty:
        return np.nan
    edges = curve["分数"].to_numpy(dtype=float)
    finite = edges[np.isfinite(edges)]
    if finite.size == 0:
        return np.nan
    return float(np.median(finite))


def _median_finite(values):
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.median(finite))


def method_side_metrics(
    df,
    sig,
    short=False,
    entry_shift=0,
    horizons=POSTERIOR_SPAN,
    daily_df=None,
    horizon_count_daily=False,
):
    """一条信号在一个方向上的挑选用数。

    曲线中位数是 3–40 根收缩超额胜率的中位数。
    下界中位数是同一段里每一根超额胜率的 90% 下界再取中位数。
    笔数是这段里最短持有期还能出场的信号数，默认就是持有 3 根。
    """
    curve = hold_edge_curve(
        df, sig, horizons=horizons, entry_shift=entry_shift, short=short,
        daily_df=daily_df, horizon_count_daily=horizon_count_daily,
    )
    empty = {"n": 0, "curve": np.nan, "lower": np.nan}
    if curve is None or curve.empty:
        return empty
    lowers = [
        edge_lower_bound(win, base, int(n) if np.isfinite(n) else 0)
        for win, base, n in zip(
            curve["胜率"].to_numpy(dtype=float),
            curve["无条件胜率"].to_numpy(dtype=float),
            curve["信号数"].to_numpy(dtype=float),
        )
    ]
    first = int(tuple(horizons)[0])
    counted = curve.loc[curve["持有根数"] == first, "信号数"]
    n_out = int(counted.iloc[0]) if len(counted) and np.isfinite(counted.iloc[0]) else 0
    return {
        "n": n_out,
        "curve": curve_median_edge(curve),
        "lower": _median_finite(lowers),
    }


def _method_clears(metrics):
    lower = metrics.get("lower", np.nan) if metrics else np.nan
    return bool(np.isfinite(lower) and float(lower) > 0)


def _pick_primary(div_metrics, nine_metrics):
    """下界中位数大于 0 才有资格。都过线时比曲线中位数，相差不足 1 个点时比笔数。"""
    div_ok = _method_clears(div_metrics)
    nine_ok = _method_clears(nine_metrics)
    if not div_ok and not nine_ok:
        return "none"
    if div_ok and not nine_ok:
        return "divergence"
    if nine_ok and not div_ok:
        return "nine"
    div_curve = float(div_metrics["curve"])
    nine_curve = float(nine_metrics["curve"])
    div_finite = np.isfinite(div_curve)
    nine_finite = np.isfinite(nine_curve)
    if div_finite and nine_finite and abs(div_curve - nine_curve) < METHOD_CURVE_TIE:
        div_n = int(div_metrics["n"])
        nine_n = int(nine_metrics["n"])
        if div_n != nine_n:
            return "divergence" if div_n > nine_n else "nine"
        return "divergence" if div_curve >= nine_curve else "nine"
    if div_finite and (not nine_finite or div_curve > nine_curve):
        return "divergence"
    if nine_finite and (not div_finite or nine_curve > div_curve):
        return "nine"
    div_n = int(div_metrics["n"])
    nine_n = int(nine_metrics["n"])
    return "divergence" if div_n >= nine_n else "nine"


def choose_methods(
    div_long, div_short,
    nine_long, nine_short,
    overlap_long, overlap_short,
    long_only=False,
):
    """每个方向在背离和九转里选主方法。共振只标记能否加仓，不参与竞选。

    只做多时做空仍保留算出来的数，adopted 为 False，调用方把判定写成不采用。
    """
    def one_side(div_metrics, nine_metrics, overlap_metrics, adopted):
        return {
            "primary": _pick_primary(div_metrics, nine_metrics),
            "adopted": bool(adopted),
            "addon": _method_clears(overlap_metrics),
            "divergence": div_metrics,
            "nine": nine_metrics,
            "overlap": overlap_metrics,
        }

    return {
        "long": one_side(div_long, nine_long, overlap_long, True),
        "short": one_side(div_short, nine_short, overlap_short, not long_only),
    }


def evaluate_method_choice(
    df, div_long, div_short, pivot_order, *,
    long_only=False, daily_df=None, horizon_count_daily=False,
    horizons=POSTERIOR_SPAN,
):
    """用当前保留的背离、不过滤的九转，以及二者的已确认重合，选出主方法和加仓。"""
    down, up = td_completed_nines(df.Close)
    overlap_long = confirmed_nine_overlap(df, div_long, down, pivot_order)
    overlap_short = confirmed_nine_overlap(df, div_short, up, pivot_order)
    kw = dict(
        daily_df=daily_df,
        horizon_count_daily=horizon_count_daily,
        horizons=horizons,
    )
    return choose_methods(
        method_side_metrics(df, div_long, short=False, entry_shift=pivot_order, **kw),
        method_side_metrics(df, div_short, short=True, entry_shift=pivot_order, **kw),
        method_side_metrics(df, down, short=False, entry_shift=1, **kw),
        method_side_metrics(df, up, short=True, entry_shift=1, **kw),
        method_side_metrics(df, overlap_long, short=False, entry_shift=0, **kw),
        method_side_metrics(df, overlap_short, short=True, entry_shift=0, **kw),
        long_only=long_only,
    )


def method_primary_label(side):
    """做多/做空主方法的短名。未采用的一侧写不采用。"""
    if not side or not side.get("adopted"):
        return "不采用"
    return {"divergence": "背离", "nine": "九转"}.get(side.get("primary"), "没有可用优势")


def method_addon_label(side):
    """共振是否加仓。未采用的一侧写不采用。"""
    if not side or not side.get("adopted"):
        return "不采用"
    return "加仓" if side.get("addon") else "不加仓"


def method_choice_summary(choice):
    """一句话：两侧主方法，以及共振是加仓还是不加仓。"""
    return (
        f"做多主方法：{method_primary_label(choice['long'])}。"
        f"做空主方法：{method_primary_label(choice['short'])}。"
        f"共振：做多{method_addon_label(choice['long'])}，做空{method_addon_label(choice['short'])}。"
    )


def method_verdict(side, method):
    """表里的判定。method 是 divergence、nine 或 overlap。"""
    if not side or not side.get("adopted"):
        return "不采用"
    if method == "overlap":
        return "加仓" if side.get("addon") else "不加仓"
    if side.get("primary") == method:
        return "主方法"
    metrics = side.get(method) or {}
    if _method_clears(metrics):
        return "过线"
    return "未过下界"


def method_signal_note(choice, side_name, kind):
    """一条信号上的说明。kind 是 divergence、nine 或 overlap。"""
    side = choice.get(side_name) or {}
    if not side.get("adopted"):
        return "该标的只做多，这一侧不采用。"
    if kind == "nine":
        if side.get("primary") == "nine":
            return "这是该方向的主方法。入场是下一根收盘。"
        return ""
    if kind == "overlap":
        if side.get("addon"):
            return "下界中位数为正，按加仓看，不替换主方法。"
        return "下界中位数不大于 0，只标记，不加仓。"
    if side.get("primary") == "divergence":
        return "这条是该方向的主方法。"
    if side.get("primary") == "nine":
        return "这条背离不是该方向的主方法，主方法是九转。"
    return "该方向没有通过下界的主方法。"


def peak_hold(horizons, values):
    """超额平均最高的持有根数。并列时取更短的那根。"""
    best_h = None
    best_v = None
    for horizon, value in zip(horizons, values):
        if value is None or not np.isfinite(value):
            continue
        value = float(value)
        if best_v is None or value > best_v:
            best_h = int(horizon)
            best_v = value
    return best_h


def curve_peak_hold(curve):
    if curve is None or curve.empty:
        return None
    return peak_hold(curve["持有根数"].tolist(), curve["超额平均"].tolist())


def book_curve_median(long_curve, short_curve, long_only=False):
    """每一根先按笔数合成两侧，再取这些数的中位数。只做多时只用底背离。"""
    if long_only or short_curve is None or short_curve.empty:
        return curve_median_edge(long_curve)
    if long_curve is None or long_curve.empty:
        return curve_median_edge(short_curve)
    left = long_curve.set_index("持有根数")
    right = short_curve.set_index("持有根数")
    scores = []
    for horizon in list(dict.fromkeys([*left.index, *right.index])):
        if horizon in left.index:
            long_n = int(left.at[horizon, "信号数"])
            long_score = float(left.at[horizon, "分数"])
        else:
            long_n, long_score = 0, np.nan
        if horizon in right.index:
            short_n = int(right.at[horizon, "信号数"])
            short_score = float(right.at[horizon, "分数"])
        else:
            short_n, short_score = 0, np.nan
        score = combine_edge_scores(long_score, long_n, short_score, short_n, False)
        if np.isfinite(score):
            scores.append(float(score))
    if not scores:
        return np.nan
    return float(np.median(np.asarray(scores, dtype=float)))


# ============ 趋势滤波（新增） ============
def _adx_parts(df, n=14):
    high, low, close = df.High, df.Low, df.Close
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.clip(lower=0).where(up_move > down_move, 0.0)
    minus_dm = down_move.clip(lower=0).where(down_move > up_move, 0.0)
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / n, adjust=False).mean()
    return pdi, mdi, adx_line


def adx(df, n=14):
    """平均趋向指标：衡量趋势强度（非方向）。>25 趋势强，<20 震荡。"""
    return _adx_parts(df, n)[2]


def trend_regime_per_bar(df, ma_n, adx_th=None, index=None):
    """
    主图背景分区，与 apply_trend_filter 一致。
    开 ADX 时深绿/深红 还需 +DI/-DI 方向一致（避免急跌仍深绿）。
    """
    idx = df.index if index is None else index
    ma = df.Close.rolling(ma_n).mean().reindex(idx)
    close = df.Close.reindex(idx)
    valid = ma.notna()
    above = (close > ma) & valid
    states = pd.Series("warmup", index=idx, dtype=object)
    if adx_th is None:
        states[valid & above] = "bull_light"
        states[valid & ~above] = "bear_light"
        return states
    pdi, mdi, adx_line = _adx_parts(df)
    pdi = pdi.reindex(idx)
    mdi = mdi.reindex(idx)
    adx_line = adx_line.reindex(idx)
    strong = (adx_line > adx_th) & valid
    bull_ok = above & strong & (pdi > mdi)
    bear_ok = (~above) & strong & (mdi > pdi)
    states[valid & ~strong] = "adx_weak"
    states[valid & strong & ~(bull_ok | bear_ok)] = "adx_weak"
    states[bull_ok] = "bull_strong"
    states[bear_ok] = "bear_strong"
    return states


def trend_filter_masks(df, ma_n=200, adx_th=None):
    """
    返回趋势滤波各分区布尔序列（与 apply_trend_filter / 主图背景一致）。

    开 ADX 时震荡段 chop（灰底）：顶、底背离全部保留。
    """
    ma = df.Close.rolling(ma_n).mean()
    above = (df.Close > ma).fillna(False)
    valid = ma.notna()
    aligned_bull = pd.Series(False, index=df.index)
    aligned_bear = pd.Series(False, index=df.index)
    chop = pd.Series(False, index=df.index)

    if adx_th is None:
        aligned_bull = above & valid
        aligned_bear = (~above) & valid
    else:
        pdi, mdi, adx_line = _adx_parts(df)
        strong = (adx_line > adx_th).fillna(False)
        aligned_bull = above & strong & (pdi > mdi).fillna(False)
        aligned_bear = (~above) & strong & (mdi > pdi).fillna(False)
        chop = valid & ~(aligned_bull | aligned_bear)

    bull_ok = (~valid) | aligned_bull | chop
    bear_ok = (~valid) | aligned_bear | chop
    return {
        "aligned_bull": aligned_bull,
        "aligned_bear": aligned_bear,
        "chop": chop,
        "bull_ok": bull_ok,
        "bear_ok": bear_ok,
        "valid": valid,
    }


def apply_trend_filter(df, bull_sig, bear_sig, ma_n=200,
                       mode="aligned", adx_th=None):
    """
    mode="aligned"（顺势）:
        深绿 / 均线上方 → 只保留底背离
        深红 / 均线下方 → 只保留顶背离
    mode="counter"（逆势）:
        深绿 / 均线上方 → 只保留顶背离
        深红 / 均线下方 → 只保留底背离
    mode="off": 不过滤，返回原信号。

    adx_th（可选，默认 None 不用）:
        趋势段：ADX > adx_th 且 +DI/-DI 与 MA 同向，才按上面的方向取舍。
        震荡段（灰底）：顶、底背离全部保留，与 mode 无关。

    均线预热段（前 ma_n 根，MA 为 NaN）不做趋势判断：两个方向的信号都保留。
    """
    if mode == "off":
        return bull_sig, bear_sig
    masks = trend_filter_masks(df, ma_n, adx_th=adx_th)
    if mode == "counter":
        bull_zone, bear_zone = masks["aligned_bear"], masks["aligned_bull"]
    else:
        bull_zone, bear_zone = masks["aligned_bull"], masks["aligned_bear"]
    bull_ok = (~masks["valid"]) | bull_zone | masks["chop"]
    bear_ok = (~masks["valid"]) | bear_zone | masks["chop"]

    def _mask(sig, cond):
        if len(sig) == 0:
            return sig
        m = cond.reindex(sig.index).fillna(False).astype(bool)
        return sig[m.values]

    return _mask(bull_sig, bull_ok), _mask(bear_sig, bear_ok)


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


def plot_chart(df, ticker, bull_sig, bear_sig, save_path,
               ma_n=TREND_MA_N, use_filter=USE_TREND_FILTER,
               bull_all=None, bear_all=None):
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
    # 标注全部背离：顺势保留的实心底色，被滤波剔除的逆势信号用白底描边+"·逆"
    def _annotate(sig_all, kept, kind):
        dropped = set(sig_all.index) - set(kept.index)
        for i, v in sig_all.items():
            xi = df.index.get_loc(i)
            if kind == "low":
                y, dy, fc, ec = df.Low.iat[xi], 0.985, "gold", "#b8860b"
                name = "底背离"
            else:
                y, dy, fc, ec = df.High.iat[xi], 1.015, "#ce93d8", "#6a1b9a"
                name = "顶背离"
            counter = i in dropped
            ax.annotate(
                f"{name}x{int(v)}{'·逆' if counter else ''}",
                (xi, y),
                xytext=(xi, y * dy),
                fontsize=8,
                ha="center",
                alpha=0.75 if counter else 1.0,
                bbox=dict(boxstyle="round",
                          fc="white" if counter else fc,
                          ec=ec, lw=0.8),
            )

    _annotate(bull_all if bull_all is not None else bull_sig, bull_sig, "low")
    _annotate(bear_all if bear_all is not None else bear_sig, bear_sig, "high")
    # 趋势背景着色（与 dashboard 一致）：均线上方绿底=留底背离，下方红底=留顶背离
    if use_filter:
        ma = df.Close.rolling(ma_n).mean()
        above = ((df.Close > ma) & ma.notna()).values
        ax.plot(x, ma, color="#ffee58", lw=1, ls=":", label=f"MA{ma_n}(趋势分界)")
        y0, y1 = ax.get_ylim()
        ax.fill_between(x, y0, y1, where=above, color="#26a69a", alpha=0.07,
                        zorder=0, interpolate=True)
        ax.fill_between(x, y0, y1, where=~above, color="#ef5350", alpha=0.07,
                        zorder=0, interpolate=True)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=8)
    title = f"{ticker} 背离+布林带"
    if use_filter:
        title += f"（趋势背景: 绿=MA{ma_n}上方留底背离/红=下方留顶背离）"
    ax.set_title(title, fontsize=10)

    ax_bb.plot(x, bb_pos, color="#42a5f5", lw=1, label="(Close-mid)/(upper-lower)")
    ax_bb.axhline(0, color="#ef5350", lw=0.7, alpha=0.85)
    ax_bb.axhline(0.5, color="#78909c", lw=0.7, ls=":", alpha=0.85)
    ax_bb.axhline(1, color="#26a69a", lw=0.7, alpha=0.85)
    ax_bb.set_xlim(-0.5, len(df) - 0.5)
    ax_bb.set_ylabel("(C-mid)/(U-L)", fontsize=8)
    _set_date_xticks(ax_bb, df.index)
    ax_bb.set_xlabel("Date")
    ax_bb.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ============ 信号后验收益分布（strip / jitter 散点） ============
def _draw_posterior_strip_ax(ax, points, horizon_labels, color, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    by_h = {h: [] for h in horizon_labels}
    for h, r in points:
        if h in by_h:
            by_h[h].append(r * 100.0)
    all_pct = [v for vals in by_h.values() for v in vals]
    ylo, yhi = _posterior_strip_limits(all_pct)
    for i, h in enumerate(horizon_labels):
        ys = by_h[h]
        if not ys:
            continue
        jit = rng.uniform(-0.22, 0.22, len(ys))
        ax.scatter(i + jit, ys, s=32, c=color, alpha=0.82, edgecolors="none", zorder=3)
    ax.axhline(0, color="#9e9e9e", linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(horizon_labels)), horizon_labels)
    ax.set_xlim(-0.55, len(horizon_labels) - 0.45)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def plot_signal_returns(df, ticker, bull_sig, bear_sig, save_path,
                        horizons=STATS_HORIZONS, entry_shift=PIVOT_ORDER,
                        bar_unit="根"):
    sf = "weekly" if bar_unit == "周" else "daily"
    kw = dict(horizons=horizons, entry_shift=entry_shift, signal_timeframe=sf)
    bull_pts = collect_posterior_returns(df, bull_sig, short=False, **kw)
    bear_pts = collect_posterior_returns(df, bear_sig, short=True, **kw)
    horizon_labels = [hold_period_label(h, False, sf) for h in horizons]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=110)
    fig.patch.set_facecolor("white")
    panels = [
        (axes[0], bull_pts, "底背离后验", "#42a5f5"),
        (axes[1], bear_pts, "顶背离后验", "#ef5350"),
    ]
    for ax_, pts, title, color in panels:
        _draw_posterior_strip_ax(
            ax_, pts, horizon_labels, color,
            rng=np.random.default_rng(hash(title) % 2**32),
        )
        ax_.set_title(title, fontsize=11, pad=8)
        for spine in ax_.spines.values():
            spine.set_color("#e0e0e0")
    fig.suptitle(
        f"{ticker} · 每个点代表一次信号在对应持有期的收益",
        fontsize=10, color="#616161", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _horizon_axis_title(horizon_count_daily, signal_timeframe):
    if horizon_count_daily:
        return "持有交易日"
    if signal_timeframe == "weekly":
        return "持有周数"
    return "持有 K 线根数"


def _returns_by_horizon(
    df, sig, horizons, entry_shift, short, daily_df, horizon_count_daily,
):
    """每个持有期一列收益（小数）。信号为空时各列为空数组。"""
    out = {}
    for h in horizons:
        rets = signal_returns(
            df, sig, h, entry_shift=entry_shift, short=short,
            daily_df=daily_df, horizon_count_daily=horizon_count_daily,
        )
        out[int(h)] = np.asarray(rets, dtype=float)
    return out


def posterior_returns_plotly(
    df,
    bull_sig,
    bear_sig,
    entry_shift=PIVOT_ORDER,
    horizons=STATS_HORIZONS,
    daily_df=None,
    horizon_count_daily=False,
    signal_timeframe="daily",
    long_name="底背离",
    short_name="顶背离",
):
    """Dashboard 用的后验收益图。

    持有期不超过 5 档时是分组散点。更长的序列（如 3 到 40）画成两行：
    上行是原始收益，下行每笔减去同一持有期、同一方向的无条件平均。
    long_name / short_name 只改标题，默认仍是顶底背离。
    """
    horizons = tuple(int(h) for h in horizons)
    if len(horizons) <= 5:
        return _posterior_strip_plotly(
            df, bull_sig, bear_sig, entry_shift, horizons,
            daily_df, horizon_count_daily, signal_timeframe,
            long_name, short_name,
        )
    return _posterior_span_plotly(
        df, bull_sig, bear_sig, entry_shift, horizons,
        daily_df, horizon_count_daily, signal_timeframe,
        long_name, short_name,
    )


def _posterior_strip_plotly(
    df, bull_sig, bear_sig, entry_shift, horizons,
    daily_df, horizon_count_daily, signal_timeframe,
    long_name="底背离", short_name="顶背离",
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    kw = dict(
        horizons=horizons,
        entry_shift=entry_shift,
        daily_df=daily_df,
        horizon_count_daily=horizon_count_daily,
        signal_timeframe=signal_timeframe,
    )
    bull_pts = collect_posterior_returns(df, bull_sig, short=False, **kw)
    bear_pts = collect_posterior_returns(df, bear_sig, short=True, **kw)
    horizon_labels = [
        hold_period_label(h, horizon_count_daily, signal_timeframe) for h in horizons
    ]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{long_name}后验", f"{short_name}后验"),
        horizontal_spacing=0.08,
    )
    panels = [
        (1, bull_pts, "#42a5f5", "底"),
        (2, bear_pts, "#ef5350", "顶"),
    ]
    for col, pts, color, seed_tag in panels:
        by_h = {h: [] for h in horizon_labels}
        for h, r in pts:
            if h in by_h:
                by_h[h].append(r * 100.0)
        all_pct = [v for vals in by_h.values() for v in vals]
        ylo, yhi = _posterior_strip_limits(all_pct)
        rng = np.random.default_rng(hash(seed_tag) % 2**32)
        for i, h in enumerate(horizon_labels):
            ys = by_h[h]
            if not ys:
                continue
            jit = rng.uniform(-0.22, 0.22, len(ys))
            fig.add_trace(
                go.Scatter(
                    x=[i + j for j in jit],
                    y=ys,
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.82),
                    showlegend=False,
                    hovertemplate="持有 %{customdata} · 收益 %{y:.2f}%<extra></extra>",
                    customdata=[h] * len(ys),
                ),
                row=1, col=col,
            )
        fig.add_hline(y=0, line_color="#9e9e9e", line_width=1, row=1, col=col)
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(horizon_labels))),
            ticktext=horizon_labels,
            range=[-0.55, len(horizon_labels) - 0.45],
            showgrid=False,
            row=1, col=col,
        )
        fig.update_yaxes(
            range=[ylo, yhi],
            ticksuffix="%",
            gridcolor="#e8e8e8",
            zeroline=False,
            row=1, col=col,
        )
    fig.update_layout(
        height=400,
        margin=dict(t=70, b=40, l=40, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text="每个点代表一次信号在对应持有期的收益",
            x=0.5,
            xanchor="center",
            font=dict(size=12, color="#757575"),
        ),
    )
    return fig


def _posterior_span_plotly(
    df, bull_sig, bear_sig, entry_shift, horizons,
    daily_df, horizon_count_daily, signal_timeframe,
    long_name="底背离", short_name="顶背离",
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    axis_title = _horizon_axis_title(horizon_count_daily, signal_timeframe)
    if horizon_count_daily:
        unit = "个交易日"
    elif signal_timeframe == "weekly":
        unit = "周"
    else:
        unit = "根"
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"{long_name}后验（未扣无条件平均）",
            f"{short_name}后验（未扣无条件平均）",
            f"{long_name}后验（已扣无条件平均）",
            f"{short_name}后验（已扣无条件平均）",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    panels = [
        (1, bull_sig, False, "#42a5f5", "rgba(66,165,245,0.18)", "底"),
        (2, bear_sig, True, "#ef5350", "rgba(239,83,80,0.18)", "顶"),
    ]
    tickvals = [h for h in horizons if h == horizons[0] or h % 5 == 0]
    for col, sig, short, color, band, seed_tag in panels:
        by_h = _returns_by_horizon(
            df, sig, horizons, entry_shift, short, daily_df, horizon_count_daily,
        )
        raw_med, raw_p25, raw_p75 = [], [], []
        ex_mean, ex_p25, ex_p75 = [], [], []
        raw_x, raw_y, raw_h = [], [], []
        ex_x, ex_y, ex_h = [], [], []
        raw_pct, ex_pct = [], []
        base_med = []
        rng = np.random.default_rng(hash(seed_tag) % 2**32)
        for h in horizons:
            rets = by_h[h]
            pct = rets * 100.0
            base = unconditional_returns(
                df, h, short=short, daily_df=daily_df,
                horizon_count_daily=horizon_count_daily,
            )
            base_mean = float(np.mean(base) * 100.0) if base.size else np.nan
            base_med.append(float(np.median(base) * 100.0) if base.size else np.nan)
            excess = pct - base_mean if pct.size and np.isfinite(base_mean) else pct
            raw_pct.extend(pct.tolist())
            ex_pct.extend(excess.tolist())
            if pct.size:
                jit = rng.uniform(-0.28, 0.28, pct.size)
                raw_x.extend((h + jit).tolist())
                raw_y.extend(pct.tolist())
                raw_h.extend([h] * int(pct.size))
                ex_x.extend((h + jit).tolist())
                ex_y.extend(excess.tolist())
                ex_h.extend([h] * int(pct.size))
                raw_med.append(float(np.median(pct)))
                raw_p25.append(float(np.percentile(pct, 25)))
                raw_p75.append(float(np.percentile(pct, 75)))
                ex_mean.append(float(np.mean(excess)))
                ex_p25.append(float(np.percentile(excess, 25)))
                ex_p75.append(float(np.percentile(excess, 75)))
            else:
                raw_med.append(np.nan)
                raw_p25.append(np.nan)
                raw_p75.append(np.nan)
                ex_mean.append(np.nan)
                ex_p25.append(np.nan)
                ex_p75.append(np.nan)
        hs = list(horizons)
        show_legend = col == 1
        layers = (
            (1, raw_x, raw_y, raw_h, raw_p25, raw_p75, raw_med, base_med, raw_pct, "收益", False),
            (2, ex_x, ex_y, ex_h, ex_p25, ex_p75, ex_mean, None, ex_pct, "超额", True),
        )
        for row, xs, ys, point_h, p25, p75, line_y, dash_y, cloud, value_name, is_excess in layers:
            fig.add_trace(
                go.Scatter(
                    x=hs + hs[::-1],
                    y=p75 + p25[::-1],
                    fill="toself",
                    fillcolor=band,
                    line=dict(color="rgba(0,0,0,0)"),
                    name="25%–75%",
                    legendgroup="band",
                    showlegend=show_legend and row == 1,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=5, color=color, opacity=0.45),
                    name="单笔收益",
                    legendgroup="pts",
                    showlegend=show_legend and row == 1,
                    hovertemplate=(
                        f"持有 %{{customdata}} {unit} · {value_name} %{{y:.2f}}%<extra></extra>"
                    ),
                    customdata=point_h,
                ),
                row=row, col=col,
            )
            line_name = "超额平均" if is_excess else "信号中位数"
            line_group = "alpha" if is_excess else "med"
            fig.add_trace(
                go.Scatter(
                    x=hs, y=line_y, mode="lines",
                    line=dict(color=color, width=2),
                    name=line_name,
                    legendgroup=line_group,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"持有 %{{x}} {unit} · {line_name} %{{y:.2f}}%<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
            if dash_y is not None:
                fig.add_trace(
                    go.Scatter(
                        x=hs, y=dash_y, mode="lines",
                        line=dict(color="#78909c", width=1.4, dash="dash"),
                        name="无条件中位数",
                        legendgroup="base",
                        showlegend=show_legend,
                        hovertemplate=(
                            f"持有 %{{x}} {unit} · 无条件中位 %{{y:.2f}}%<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )
            ylo, yhi = _posterior_strip_limits(cloud)
            fig.add_hline(y=0, line_color="#9e9e9e", line_width=1, row=row, col=col)
            fig.update_xaxes(
                title_text=axis_title if row == 2 else "",
                tickmode="array",
                tickvals=tickvals,
                range=[horizons[0] - 0.8, horizons[-1] + 0.8],
                showgrid=False,
                row=row, col=col,
            )
            fig.update_yaxes(
                range=[ylo, yhi],
                ticksuffix="%",
                gridcolor="#e8e8e8",
                zeroline=False,
                row=row, col=col,
            )
        peak = peak_hold(hs, ex_mean)
        if peak is not None:
            for mark_row in (1, 2):
                fig.add_vline(
                    x=peak, line_width=1.2, line_dash="dot", line_color=color,
                    row=mark_row, col=col,
                )
            fig.add_annotation(
                x=peak, y=yhi, text=f"超额平均最高 · {peak} {unit}",
                showarrow=False, yanchor="top",
                font=dict(size=11, color=color),
                bgcolor="rgba(255,255,255,0.85)",
                row=2, col=col,
            )
    fig.update_layout(
        height=920,
        margin=dict(t=48, b=64, l=48, r=16),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
    )
    for ann in fig.layout.annotations:
        text = ann.text or ""
        ann.font.size = 11 if text.startswith("超额平均最高") else 13
    return fig


def process_ticker(ticker, out_dir, use_filter=USE_TREND_FILTER, ma_n=TREND_MA_N):
    df = add_indicators(load_ohlcv(ticker))
    bull_all, bear_all = compute_signals(df)               # 原始信号（不过滤）
    bull_sig, bear_sig = bull_all, bear_all
    if use_filter:
        bull_sig, bear_sig = apply_trend_filter(df, bull_all, bear_all,
                                                ma_n=ma_n, mode="aligned")
    chart_path = out_dir / f"{safe_chart_name(ticker)}.png"
    plot_chart(df, ticker, bull_sig, bear_sig, chart_path,
               bull_all=bull_all, bear_all=bear_all)

    # 新增：信号历史统计 + 后验分布图
    stats_bull = signal_stats(df, bull_sig)
    stats_bear = signal_stats(df, bear_sig, short=True)
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
                "alert_past_30d": bool(rec),
                "signals_past_30d": len(rec),
                "chart": r["chart_path"],
            }
        )
    for err in errors:
        summary_rows.append(
            {"ticker": err["ticker"], "alert_past_30d": False, "signals_past_30d": 0, "chart": "", "error": err["error"]}
        )
    summary_path = out_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print()
    print(f"Charts folder: {out_dir}")
    print(f"Recent signals (past {SIGNAL_RECENT_DAYS} days): {alerts_path}")
    if not alerts_df.empty:
        print(f"\n=== Alerts (past {SIGNAL_RECENT_DAYS} days) ===")
        print(alerts_df.to_string(index=False))
    else:
        print(f"\nNo divergence signals in the past {SIGNAL_RECENT_DAYS} days for this ticker list.")
    print(f"Signal stats: {all_stats_path}")
    print(f"Summary: {summary_path}")
    return out_dir, alerts_df, results


if __name__ == "__main__":
    run_batch(TICKERS)
