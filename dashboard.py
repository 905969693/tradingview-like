"""
交互式复盘 Dashboard —— 复用 tradingview_realization_v2.py 的全部函数

启动方式（在本文件所在目录）：
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    streamlit run dashboard.py

左侧选标的/调参数，右侧联动：主图(K线+均线+布林+斐波那契+滚动VP+背离标注)
+ 信号后验统计表 + 后验收益分布 + 近7日信号。
"""
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(layout="wide", page_title="背离复盘 Dashboard")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

try:
    import tradingview_realization_v2 as tv
except ModuleNotFoundError:
    import tradingview_realization as tv

# ---------------- 数据加载（Streamlit 缓存 + 你的增量缓存双保险） ----------------
@st.cache_data(ttl=1800, show_spinner="下载/读取数据...")
def load_data(ticker, period, use_cache, timeframe):
    daily = tv.load_ohlcv(ticker, period=period, use_cache=use_cache)
    if timeframe == "周线":
        return tv.resample_ohlcv(daily)
    return daily

@st.cache_data(show_spinner=False)
def compute_all(df, pivot_order, min_count, dedup_bars, vp_window):
    """信号 + 统计 + 滚动VP；参数变化时自动重算"""
    dfi = tv.add_indicators(df)
    bull_s = pd.Series(0, index=dfi.index)
    bear_s = pd.Series(0, index=dfi.index)
    for n in tv.INDICATORS:
        b, r = tv.find_divergences(dfi, dfi[n], order=pivot_order)
        bull_s.iloc[b] += 1
        bear_s.iloc[r] += 1
    bull_sig = tv.dedup_signals(bull_s[bull_s >= min_count], dfi, gap=dedup_bars)
    bear_sig = tv.dedup_signals(bear_s[bear_s >= min_count], dfi, gap=dedup_bars)
    rvp = tv.rolling_volume_profile(dfi, window=vp_window)
    return dfi, bull_sig, bear_sig, rvp

# ---------------- 侧边栏 ----------------
st.sidebar.title("⚙️ 参数")

# 标的列表：配置文件里的 + 缓存目录里出现过的
cache_tickers = []
if tv.CACHE_DIR.exists():
    cache_tickers = sorted({p.name.rsplit("_", 1)[0]
                            for p in tv.CACHE_DIR.glob("*.csv")})
tickers = list(dict.fromkeys(list(tv.TICKERS) + cache_tickers))
ticker = st.sidebar.selectbox(
    "标的（下拉选择或直接输入）",
    options=tickers,
    index=0,
    accept_new_options=True,
    help="例：AAPL、0700.HK、USDJPY=X、000001.SS、BTC-USD",
)
period = st.sidebar.selectbox("数据区间", ["2y", "5y", "10y", "max"],
                              index=list(["2y", "5y", "10y", "max"]).index(tv.PERIOD)
                              if tv.PERIOD in ["2y", "5y", "10y", "max"] else 1)
timeframe = st.sidebar.selectbox("K线周期", ["日线", "周线"], index=0)
is_weekly = timeframe == "周线"
force_refresh = st.sidebar.checkbox("强制重新下载（绕过缓存）", value=False)
if is_weekly:
    show_bars = st.sidebar.slider("图表显示最近 N 根K线", 26, 260, 104, 13)
else:
    show_bars = st.sidebar.slider("图表显示最近 N 根K线", 100, 800, 400, 50)
bar_unit = "周" if is_weekly else "根"
recent_window_days = 56 if is_weekly else tv.SIGNAL_RECENT_DAYS
recent_window_label = "8 周" if is_weekly else f"{tv.SIGNAL_RECENT_DAYS} 日"

st.sidebar.divider()
pivot_order = st.sidebar.slider("pivot 灵敏度 PIVOT_ORDER", 3, 12, tv.PIVOT_ORDER)
min_count = st.sidebar.slider("共振指标数 MIN_COUNT", 2, 8, tv.MIN_COUNT)
dedup_bars = st.sidebar.slider("去重窗口 DEDUP_BARS", 5, 30, tv.DEDUP_BARS)
vp_window = st.sidebar.slider("滚动VP窗口 VP_WINDOW", 30, 150, tv.VP_WINDOW, 10)
lookback = st.sidebar.number_input("斐波那契波段 LOOKBACK（0=整段）",
                                   min_value=0, max_value=1500,
                                   value=0 if tv.LOOKBACK is None else tv.LOOKBACK)

# ---------------- 计算 ----------------
try:
    df_raw = load_data(ticker, period, use_cache=not force_refresh, timeframe=timeframe)
except Exception as exc:
    st.error(f"无法加载 {ticker} 数据：{exc}")
    st.stop()

if df_raw is None or df_raw.empty:
    st.warning(f"{ticker} 无可用{timeframe}数据，请换标的、换周期或勾选「强制重新下载」。")
    st.stop()

df, bull_sig, bear_sig, rvp = compute_all(
    df_raw, pivot_order, min_count, dedup_bars, vp_window)
fib = tv.fib_levels(df, lookback=None if lookback == 0 else int(lookback))

# 显示窗口裁剪（信号/统计仍基于全量数据计算）
disp = df.iloc[-show_bars:]
rvp_disp = rvp.iloc[-show_bars:]

# 右侧 Volume Profile：取显示窗口最后 vp_window 根
vp_result = tv.volume_profile(df)
centers, vp_vol, poc, vah, val = vp_result

# ---------------- 顶部指标条 ----------------
last = df.Close.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最新收盘", f"{last:,.2f}")
c2.metric("POC", f"{poc:,.2f}", f"{last-poc:+,.2f}")
c3.metric("VAH", f"{vah:,.2f}", f"{last-vah:+,.2f}")
c4.metric("VAL", f"{val:,.2f}", f"{last-val:+,.2f}")
n_recent = len(
    tv.signals_in_window(
        bull_sig, bear_sig, df, ticker, days=recent_window_days
    )
)
c5.metric(f"近{recent_window_label}信号", n_recent)

# ---------------- 侧边栏：标的列表近期信号扫描 ----------------
@st.cache_data(ttl=900, show_spinner=False)
def scan_recent_signals(tickers_tuple, period, days, pivot_order, min_count, dedup_bars):
    """扫描标的列表里近期有背离信号的 ticker。使用当前侧边栏参数。"""
    rows = []
    for t in tickers_tuple:
        try:
            d = tv.load_ohlcv(t, period=period)
            d = tv.add_indicators(d)
            bull_s = pd.Series(0, index=d.index)
            bear_s = pd.Series(0, index=d.index)
            for n in tv.INDICATORS:
                b, r = tv.find_divergences(d, d[n], order=pivot_order)
                bull_s.iloc[b] += 1
                bear_s.iloc[r] += 1
            bull = tv.dedup_signals(bull_s[bull_s >= min_count], d, gap=dedup_bars)
            bear = tv.dedup_signals(bear_s[bear_s >= min_count], d, gap=dedup_bars)
            sigs = tv.signals_in_window(bull, bear, d, t, days=days)
            if sigs:
                latest = sorted(sigs, key=lambda x: x["date"], reverse=True)[0]
                rows.append({
                    "ticker": t,
                    "最新信号": latest["signal"],
                    "日期": latest["date"],
                    "共振数": latest["indicator_count"],
                    "信号数(窗口内)": len(sigs),
                })
        except Exception:
            continue  # 某个标的拉取失败不阻塞整体扫描
    return pd.DataFrame(rows)

# 放侧边栏：展开面板，不占用主界面
with st.sidebar.expander(f"🔔 近{recent_window_label}有信号的标的", expanded=True):
    scan = scan_recent_signals(
        tuple(tickers), period, recent_window_days,
        pivot_order, min_count, dedup_bars,
    )
    if scan.empty:
        st.success(f"近 {recent_window_label} 全部标的无背离信号")
    else:
        # 底背离绿、顶背离红，便于一眼区分
        def _color(sig):
            return "color: #26a69a" if sig == "底背离" else "color: #ef5350"
        st.dataframe(
            scan.sort_values("日期", ascending=False).style.map(
                lambda s: _color(s), subset=["最新信号"]),
            width="stretch", hide_index=True,
        )

# ---------------- 主图 ----------------
fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                    column_widths=[0.85, 0.15], horizontal_spacing=0.01)
fig.add_trace(go.Candlestick(x=disp.index, open=disp.Open, high=disp.High,
    low=disp.Low, close=disp.Close, name=ticker,
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp.Close.rolling(10).mean(),
    name="MA10", line=dict(color="gray", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp.Close.rolling(20).mean(),
    name="MA20", line=dict(color="#42a5f5", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp.VWMA,
    name="VWMA20", line=dict(color="#ff9800", width=1)), row=1, col=1)
# 布林带
bb_mid, bb_up, bb_lo = tv.bollinger_bands(disp.Close)
fig.add_trace(go.Scatter(x=list(disp.index)+list(disp.index)[::-1],
    y=list(bb_up)+list(bb_lo)[::-1], fill="toself",
    fillcolor="rgba(120,144,156,0.12)", line=dict(color="rgba(0,0,0,0)"),
    name=f"布林({tv.BB_PERIOD},{tv.BB_STD}σ)"), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=rvp_disp.POC, name=f"滚动POC({vp_window}{bar_unit})",
    line=dict(color="#ff9800", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=list(disp.index)+list(disp.index)[::-1],
    y=list(rvp_disp.VAH)+list(rvp_disp.VAL)[::-1], fill="toself",
    fillcolor="rgba(38,166,154,0.10)", line=dict(color="rgba(0,0,0,0)"),
    name="滚动VA"), row=1, col=1)
for r, p in fib.items():
    fig.add_hline(y=p, line_dash="dash", line_color="#ff9800", line_width=1,
                  annotation_text=f"{r}", annotation_font_size=10, row=1, col=1)
# 背离标注（仅显示窗口内的）
b_in = bull_sig[bull_sig.index >= disp.index[0]]
r_in = bear_sig[bear_sig.index >= disp.index[0]]
if len(b_in):
    fig.add_trace(go.Scatter(x=b_in.index, y=df.Low.loc[b_in.index]*0.985,
        mode="markers+text", text=[f"底x{int(v)}" for v in b_in],
        textposition="bottom center",
        marker=dict(symbol="triangle-up", size=12, color="gold"),
        name="底背离"), row=1, col=1)
if len(r_in):
    fig.add_trace(go.Scatter(x=r_in.index, y=df.High.loc[r_in.index]*1.015,
        mode="markers+text", text=[f"顶x{int(v)}" for v in r_in],
        textposition="top center",
        marker=dict(symbol="triangle-down", size=12, color="#ce93d8"),
        name="顶背离"), row=1, col=1)
# 右侧 Volume Profile
fig.add_trace(go.Bar(x=vp_vol, y=centers, orientation="h", showlegend=False,
    marker_color="rgba(38,166,154,0.6)"), row=1, col=2)
for p_, lbl, col in [(poc, "POC", "#ff9800"), (vah, "VAH", "#26a69a"), (val, "VAL", "#ef5350")]:
    fig.add_hline(y=p_, line_color=col, row=1, col=2)
    fig.add_annotation(x=vp_vol.max()*0.98, y=p_, text=lbl, showarrow=False,
                       font=dict(color=col, size=10), xanchor="right", row=1, col=2)

fig.update_layout(
    title=f"{ticker} [{timeframe}] 背离 + 斐波那契 + 布林带 + 滚动 Volume Profile",
    xaxis_rangeslider_visible=False,
    height=720,
    legend=dict(orientation="h", y=1.04),
)
if not is_weekly:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
st.plotly_chart(fig, width="stretch", alt=f"{ticker} 价格与 Volume Profile")

# ---------------- 统计区 ----------------
left, right = st.columns([1, 1.4])
with left:
    st.subheader("信号后验统计（基于全量数据）")
    stats_bull = tv.signal_stats(df, bull_sig)
    stats_bear = tv.signal_stats(df, bear_sig)
    st.markdown("**底背离（期望上涨）**")
    st.dataframe(stats_bull.set_index("持有期"), width="stretch", alt="底背离后验统计")
    st.markdown("**顶背离（期望下跌）**")
    st.dataframe(stats_bear.set_index("持有期"), width="stretch", alt="顶背离后验统计")
    st.subheader(f"近 {recent_window_label} 信号")
    recent = pd.DataFrame(
        tv.signals_in_window(
            bull_sig, bear_sig, df, ticker, days=recent_window_days
        )
    )
    if recent.empty:
        st.info(f"近 {recent_window_label} 无信号")
    else:
        st.dataframe(recent, width="stretch", alt=f"近{recent_window_label}背离信号")

with right:
    st.subheader("后验收益分布")
    closes = df.Close.values
    H_COLORS = ["#1b9e77", "#7570b3", "#d95f02"]
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("底背离后验", "顶背离后验"))
    for col, (sig, title) in enumerate([(bull_sig, "底背离"), (bear_sig, "顶背离")], start=1):
        for h, hcol in zip(tv.STATS_HORIZONS, H_COLORS):
            rets = [closes[df.index.get_loc(i)+h]/closes[df.index.get_loc(i)]-1
                    for i in sig.index if df.index.get_loc(i)+h < len(df)]
            if rets:
                fig2.add_trace(go.Histogram(x=rets, name=f"{h}{bar_unit}", marker_color=hcol,
                    opacity=0.55, nbinsx=10), row=1, col=col)
        fig2.add_vline(x=0, line_color="black", line_width=1, row=1, col=col)
    fig2.update_layout(barmode="overlay", height=380, showlegend=True,
                       legend=dict(orientation="h", y=1.15),
                       margin=dict(t=40))
    st.plotly_chart(fig2, width="stretch", alt="背离信号后验收益分布")

    st.subheader("斐波那契关键位")
    fib_df = pd.DataFrame({"比例": list(fib.keys()), "价位": list(fib.values())})
    fib_df["距现价%"] = (fib_df["价位"] / last - 1) * 100
    fib_display = fib_df.copy()
    fib_display["价位"] = fib_display["价位"].map(lambda x: f"{x:,.2f}")
    fib_display["距现价%"] = fib_display["距现价%"].map(lambda x: f"{x:+.2f}%")
    st.dataframe(fib_display, width="stretch", alt="斐波那契关键位")

st.caption(
    f"周期: {timeframe} | 数据: {df.index[0].date()} -> {df.index[-1].date()} | "
    f"底背离信号 {len(bull_sig)} 个 | 顶背离信号 {len(bear_sig)} 个 | "
    f"模块: {tv.__name__}.py"
)
