"""
交互式复盘 Dashboard —— 复用 tradingview_realization_v2.py 的全部函数

启动方式（在本文件所在目录）：
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    streamlit run dashboard.py

左侧选标的/调参数，右侧联动：主图(K线+均线+布林+背离标注)
+ 十字光标/区间测量(A→B) + 信号后验统计表 + 后验收益分布 + 近30日信号。
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

import backtest_asset_opt as asset_opt
import decision_prompt as dp
import signal_presets as _presets

_reload_presets = _presets.reload_presets
_remember_fixed_preset = _presets.remember_fixed_preset
_preset_for = _presets.preset_for
_listed_preset_tickers = _presets.listed_preset_tickers
_preset_cache_key = _presets.preset_cache_key


def _presets_match(current, candidate):
    if not current or not candidate:
        return False
    fields = (
        "pivot_order",
        "min_count",
        "dedup_bars",
        "trend_on",
        "aligned_keep",
        "adx_th",
        "ma_n",
    )
    if any(current.get(field) != candidate.get(field) for field in fields):
        return False
    return bool(current.get("long_only", False)) == bool(candidate.get("long_only", False))


_reload_presets()

_EDGE_WINDOW_LABEL = {"2y": "2 年", "5y": "5 年", "10y": "10 年", "max": "全部历史"}


def _edge_origin(source):
    if not source or not str(source).startswith("edge_"):
        return ""
    window = str(source).removeprefix("edge_")
    return _EDGE_WINDOW_LABEL.get(window, window)


def _ohlcv_token(ticker, period):
    path = tv.CACHE_DIR / f"{tv.safe_chart_name(ticker)}_{period}.csv"
    if not path.exists():
        return "missing"
    last = ""
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = line
    return last.split(",", 1)[0]


def _cache_as_of(ticker, period):
    """本地缓存最后一根 K 线的日期。没有这份缓存时返回 None。"""
    token = _ohlcv_token(ticker, period)
    if not token or token == "missing":
        return None
    try:
        ts = pd.Timestamp(token)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _refresh_ohlcv_incremental(names, period, parent):
    """沿用本地缓存，只下载还没有的交易日并合并写回。"""
    ordered = []
    seen = set()
    for name in names:
        safe = tv.safe_chart_name(name)
        if not name or safe in seen:
            continue
        seen.add(safe)
        ordered.append(name)
    errors = []
    total = len(ordered)
    progress = parent.progress(0.0, text="准备更新")
    for index, name in enumerate(ordered, start=1):
        progress.progress(
            (index - 1) / total if total else 1.0,
            text=f"正在更新 {name}（{index}/{total}）",
        )
        try:
            tv.load_ohlcv(name, period=period, use_cache=True)
        except Exception as exc:
            errors.append(f"{name}（{exc}）")
    progress.progress(1.0, text="已合并到本地缓存")
    return errors


_BATCH_OPT_SKIP = frozenset({"KBE", "KRE"})


def _points_label(score):
    if score is None or not np.isfinite(score):
        return "—"
    points = float(score) * 100
    if abs(points) < 0.05:
        return "0.0"
    return f"{points:+.1f}"


def _peak_pair_label(bull_peak, bear_peak):
    left = f"底 {bull_peak}" if bull_peak else "底 —"
    right = f"顶 {bear_peak}" if bear_peak else "顶 —"
    return f"{left} / {right}"


@st.cache_data(ttl="6h", max_entries=64, show_spinner=False)
def _cached_optimize(ticker, period, data_token):
    """在这段日线上最大化 3 到 40 根收缩超额胜率的中位数。"""
    del data_token
    df = tv.add_indicators(tv.load_ohlcv(ticker, period=period))
    best = asset_opt.search_best_edge(df)
    if best is None:
        return {
            "method": "edge",
            "status": "no_edge",
            "period": period,
            "start": df.index[0].date().isoformat(),
            "end": df.index[-1].date().isoformat(),
            "n_bars": int(len(df)),
        }
    preset = asset_opt.preset_from_row(best, long_only=best["long_only"])
    preset["source"] = f"edge_{period}"
    return {
        "method": "edge",
        "status": "selected",
        "period": period,
        "preset": preset,
        "score": best["score"],
        "baseline": best["baseline"],
        "n": best["n"],
        "long_n": best["long_n"],
        "short_n": best["short_n"],
        "long_score": best["long_score"],
        "short_score": best["short_score"],
        "bull_peak": best.get("bull_peak"),
        "bear_peak": best.get("bear_peak"),
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "n_bars": int(len(df)),
    }


def _optimize_listed_tickers(names, period):
    """除 KBE、KRE 外，把每个标的的固定参数改成当前区间分数最高的一组。"""
    targets = [
        name for name in names
        if name not in _BATCH_OPT_SKIP and tv.safe_chart_name(name) not in _BATCH_OPT_SKIP
    ]
    rows = []
    if not targets:
        return rows
    progress = st.sidebar.progress(0.0, text="准备优化")
    for index, name in enumerate(targets, start=1):
        progress.progress(
            (index - 1) / len(targets),
            text=f"正在优化 {name}（{index}/{len(targets)}）",
        )
        try:
            tv.load_ohlcv(name, period=period)
            result = _cached_optimize(name, period, _ohlcv_token(name, period))
        except Exception as exc:
            rows.append({"标的": name, "分数": "—", "参数": f"失败：{exc}"})
            continue
        if result.get("status") != "selected":
            rows.append({"标的": name, "分数": "—", "参数": "无法优化"})
            continue
        _remember_fixed_preset(name, result["preset"])
        rows.append({
            "标的": name,
            "分数": _points_label(result["score"]),
            "尖峰": _peak_pair_label(result.get("bull_peak"), result.get("bear_peak")),
            "参数": result["preset"]["summary"],
        })
    progress.progress(1.0, text=f"已优化 {len(targets)} 个标的")
    return rows


def _show_optimize_result(ticker, result):
    if result.get("method") != "edge":
        return
    if result["status"] != "selected":
        st.warning(
            f"日线 {result['start']} → {result['end']}（{result['n_bars']} 根）。"
            "144 组里没有一组能算出曲线中位数。",
            title="无法优化",
        )
        return

    preset = result["preset"]
    st.markdown(f"**{preset['summary']}**")
    adopted = "只做多，分数只按底背离" if preset.get("long_only") else "每一根先按笔数合成两侧"
    bull_peak = result.get("bull_peak")
    bear_peak = result.get("bear_peak")
    st.caption(
        f"日线 {result['start']} → {result['end']}（{result['n_bars']} 根）。"
        f"采用：{adopted}。"
        "分数是 3–40 根收缩超额胜率的中位数。"
        "尖峰是该侧超额平均最高的持有根数，只显示，不参与比较。"
        "笔数是持有 3 根还能出场的信号数。"
        "只做多时，图上和后验表仍会列出顶背离和做空收益。"
    )
    st.dataframe(
        pd.DataFrame([
            {
                "项目": "底背离",
                "笔数": result["long_n"],
                "分数": _fmt_edge_points(result["long_score"]) or "—",
                "尖峰": f"{bull_peak} 根" if bull_peak else "—",
            },
            {
                "项目": "顶背离",
                "笔数": result["short_n"],
                "分数": _fmt_edge_points(result["short_score"]) or "—",
                "尖峰": f"{bear_peak} 根" if bear_peak else "—",
            },
            {
                "项目": "采用",
                "笔数": result["n"],
                "分数": _fmt_edge_points(result["score"]) or "—",
                "尖峰": "—",
            },
            {
                "项目": "默认参数",
                "笔数": "—",
                "分数": _fmt_edge_points(result["baseline"]) or "—",
                "尖峰": "—",
            },
        ]),
        hide_index=True,
        alt="底背离、顶背离、采用分数、默认参数分数和尖峰根数",
    )
    current = _preset_for(ticker)
    if _presets_match(current, preset):
        st.success("侧边栏已经锁定这组参数。", title="已是固定参数")
        return
    if current:
        st.caption(f"选入后会替换当前固定参数：{current['summary']}")
    if st.button(
        "选入固定参数",
        type="primary",
        icon=":material/lock:",
        key=f"pin_preset_{ticker}",
    ):
        _remember_fixed_preset(ticker, preset)
        st.rerun(scope="app")


@st.fragment
def _param_optimize_panel(ticker, period):
    window = _EDGE_WINDOW_LABEL.get(period, period)
    with st.container(border=True):
        st.subheader("参数优化")
        st.caption(
            f"在近 {window} 的日线上搜索 144 组，最大化 3–40 根收缩超额胜率的中位数。"
            "每一组都比较多空一起和只做多，取得分更高的。"
            "尖峰根数只显示，不参与比较，也不搜索持仓天数。"
            "不看四段验证，也不看 2025 至今。"
            "图表切到周线时，这次搜索仍用日线。"
        )
        if st.button(
            "一键优化",
            icon=":material/tune:",
            key=f"optimize_{ticker}",
        ):
            try:
                with st.spinner(f"正在用近 {window} 的日线搜索 144 组参数…"):
                    tv.load_ohlcv(ticker, period=period)
                    result = _cached_optimize(ticker, period, _ohlcv_token(ticker, period))
            except Exception as exc:
                st.error(f"优化失败：{exc}")
            else:
                stored = dict(st.session_state.get("opt_results") or {})
                stored[ticker] = result
                st.session_state.opt_results = stored
        result = (st.session_state.get("opt_results") or {}).get(ticker)
        if result and result.get("period") == period:
            _show_optimize_result(ticker, result)

# ---------------- 数据加载（Streamlit 缓存 + 你的增量缓存双保险） ----------------
@st.cache_data(ttl=1800, show_spinner="下载/读取数据...")
def load_data(ticker, period, use_cache, timeframe):
    daily = tv.load_ohlcv(ticker, period=period, use_cache=use_cache)
    if timeframe == "周线":
        return tv.resample_ohlcv(daily)
    return daily

_TREND_BG_FILL = {
    "warmup": "rgba(189,189,189,0.14)",
    "bull_light": "rgba(38,166,154,0.10)",
    "bear_light": "rgba(239,83,80,0.10)",
    "bull_strong": "rgba(22,128,96,0.32)",
    "bear_strong": "rgba(198,40,40,0.32)",
    "adx_weak": "rgba(158,158,158,0.12)",
}


def _trend_background_legend(adx_th, ma_n, counter=False):
    """主图竖条背景色与信号保留规则（与 trend_regime_per_bar 一致）。"""
    rows = [
        {
            "背景色": "灰色（浅）",
            "判定条件": f"前 {ma_n} 根 K 线 MA 尚未有效（预热）",
            "保留规则": "底/顶背离都保留（可显示为实心）",
        },
    ]
    if adx_th is None:
        rows.extend([
            {
                "背景色": "浅绿",
                "判定条件": f"收盘 > MA{ma_n}（黄虚线）",
                "保留规则": "只保留顶背离" if counter else "只保留底背离",
            },
            {
                "背景色": "浅红",
                "判定条件": f"收盘 < MA{ma_n}（黄虚线）",
                "保留规则": "只保留底背离" if counter else "只保留顶背离",
            },
        ])
    else:
        rows.extend([
            {
                "背景色": "深绿",
                "判定条件": (
                    f"收盘 > MA{ma_n}，且 ADX > {adx_th}，且 +DI > -DI"
                ),
                "保留规则": "只保留顶背离" if counter else "只保留底背离",
            },
            {
                "背景色": "深红",
                "判定条件": (
                    f"收盘 < MA{ma_n}，且 ADX > {adx_th}，且 -DI > +DI"
                ),
                "保留规则": "只保留底背离" if counter else "只保留顶背离",
            },
            {
                "背景色": "灰色（浅）",
                "判定条件": (
                    f"ADX ≤ {adx_th}，或 MA 方向与 +DI/-DI 不一致"
                    "（例如在 MA 上但 -DI 更强）"
                ),
                "保留规则": "震荡段：顶、底背离全部保留（标 ·震）",
            },
        ])
    rows.append({
        "背景色": "—",
        "判定条件": "MA10 / MA20 蓝线 / VWMA 等不参与背景着色",
        "保留规则": "仅以侧边栏「趋势滤波 MA 周期」为准",
    })
    return pd.DataFrame(rows)


def _signal_filter_matrix(adx_th, counter=False):
    """按背景分区说明顶/底背离保留还是过滤。"""
    if adx_th is None:
        if counter:
            return pd.DataFrame([
                {"背景": "浅绿（MA 上）", "底背离": "过滤 ·滤", "顶背离": "保留（逆势）"},
                {"背景": "浅红（MA 下）", "底背离": "保留（逆势）", "顶背离": "过滤 ·滤"},
            ])
        return pd.DataFrame([
            {"背景": "浅绿（MA 上）", "底背离": "保留", "顶背离": "过滤 ·滤"},
            {"背景": "浅红（MA 下）", "底背离": "过滤 ·滤", "顶背离": "保留"},
        ])
    if counter:
        return pd.DataFrame([
            {"背景": "深绿（趋势多）", "底背离": "过滤 ·滤", "顶背离": "保留（逆势）"},
            {"背景": "深红（趋势空）", "底背离": "保留（逆势）", "顶背离": "过滤 ·滤"},
            {"背景": "灰色（震荡）", "底背离": "全部保留 ·震", "顶背离": "全部保留 ·震"},
        ])
    return pd.DataFrame([
        {"背景": "深绿（趋势多）", "底背离": "保留（顺势）", "顶背离": "过滤 ·滤"},
        {"背景": "深红（趋势空）", "底背离": "过滤 ·滤", "顶背离": "保留（顺势）"},
        {"背景": "灰色（震荡）", "底背离": "全部保留 ·震", "顶背离": "全部保留 ·震"},
    ])


def _add_trend_vrects(fig, states, row=1, col=1):
    seg_start, seg_key = None, None
    for ts, key in states.items():
        if seg_key is None:
            seg_start, seg_key = ts, key
        elif key != seg_key:
            fig.add_vrect(
                x0=seg_start, x1=ts,
                fillcolor=_TREND_BG_FILL[seg_key],
                line_width=0, layer="below", row=row, col=col,
            )
            seg_start, seg_key = ts, key
    if seg_key is not None:
        fig.add_vrect(
            x0=seg_start, x1=states.index[-1],
            fillcolor=_TREND_BG_FILL[seg_key],
            line_width=0, layer="below", row=row, col=col,
        )


def _apply_price_chart_crosshair(fig, row=None, col=None):
    """主图十字光标：spike 线 + 轴上日期/价格标签（Plotly 内置，无需 JS）。

    子图（make_subplots）传 row/col。普通 Figure 不传，否则 Plotly 会要求 subplot 网格。
    """
    spike = dict(
        showspikes=True,
        spikemode="across+toaxis",
        spikesnap="cursor",
        spikecolor="rgba(160,160,160,0.9)",
        spikethickness=1,
        spikedash="dot",
    )
    axis_kw = {}
    if row is not None and col is not None:
        axis_kw = dict(row=row, col=col)
    fig.update_xaxes(**spike, **axis_kw)
    fig.update_yaxes(**spike, **axis_kw)


def _nearest_index_ts(index, x_val):
    ts = pd.Timestamp(x_val)
    pos = index.get_indexer([ts], method="nearest")[0]
    if pos < 0:
        return None
    return index[pos]


def _measure_stats(t_a, p_a, t_b, p_b, bar_index):
    t_lo, t_hi = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
    p_lo = p_a if t_a <= t_b else p_b
    p_hi = p_b if t_a <= t_b else p_a
    pct = (p_hi / p_lo - 1.0) * 100.0 if p_lo else 0.0
    cal_days = abs((pd.Timestamp(t_b) - pd.Timestamp(t_a)).days)
    try:
        i_a = bar_index.get_loc(t_a)
        i_b = bar_index.get_loc(t_b)
        if isinstance(i_a, slice):
            i_a = i_a.start
        if isinstance(i_b, slice):
            i_b = i_b.start
        trade_bars = abs(int(i_b) - int(i_a))
    except KeyError:
        trade_bars = None
    return {
        "pct": pct,
        "cal_days": cal_days,
        "trade_bars": trade_bars,
        "t_lo": t_lo,
        "t_hi": t_hi,
        "p_lo": p_lo,
        "p_hi": p_hi,
    }


def _add_measure_overlay(fig, t_a, p_a, t_b, p_b, stats, row=1, col=1):
    fig.add_shape(
        type="line",
        x0=t_a,
        y0=p_a,
        x1=t_b,
        y1=p_b,
        line=dict(color="#ce93d8", width=2, dash="dash"),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=[t_a, t_b],
            y=[p_a, p_b],
            mode="markers+text",
            marker=dict(size=9, color="#e040fb", symbol="circle"),
            text=["A", "B"],
            textposition=["top center", "bottom center"],
            textfont=dict(size=11, color="#e040fb"),
            name="区间测量",
            hovertemplate=(
                "A %{customdata[0]}<br>收盘 %{customdata[1]:,.2f}<extra></extra>"
            ),
            customdata=[
                [t_a.strftime("%Y-%m-%d"), p_a],
                [t_b.strftime("%Y-%m-%d"), p_b],
            ],
        ),
        row=row,
        col=col,
    )
    tb = stats["trade_bars"]
    tb_txt = f"{tb} 根K" if tb is not None else "— 根K"
    label = (
        f"Δ {stats['pct']:+.2f}% · {stats['cal_days']} 自然日 · {tb_txt}"
    )
    mid_t = t_a + (t_b - t_a) / 2
    mid_p = (p_a + p_b) / 2
    fig.add_annotation(
        x=mid_t,
        y=mid_p,
        text=label,
        showarrow=False,
        bgcolor="rgba(30,30,30,0.82)",
        bordercolor="#ce93d8",
        borderwidth=1,
        font=dict(color="white", size=11),
        row=row,
        col=col,
    )


def _attach_unconditional(stats, base):
    """把同一持有期的无条件胜率和平均收益接到信号表上。"""
    if stats is None or stats.empty:
        return stats
    out = stats.merge(
        base[["持有期", "对照样本", "无条件胜率", "无条件平均"]],
        on="持有期",
        how="left",
    )
    order = [
        "持有期", "信号数", "胜率", "无条件胜率",
        "平均收益", "无条件平均", "中位收益", "最好", "最差", "对照样本",
    ]
    out = out[order]
    out["对照样本"] = out["对照样本"].astype("Int64")
    return _attach_edge_columns(out)


def _attach_edge_columns(stats):
    """持有 10 日那一行写入收缩分数和 90% 下界。5 日、20 日、按周计数的 10 周留空。"""
    if stats is None or stats.empty or "胜率" not in stats.columns:
        return stats
    out = stats.copy()
    scores = []
    lowers = []
    for _, row in out.iterrows():
        label = str(row["持有期"])
        n = int(row["信号数"]) if pd.notna(row["信号数"]) else 0
        if label not in ("10根K线", "10交易日"):
            scores.append(np.nan)
            lowers.append(np.nan)
            continue
        scores.append(tv.edge_score(row["胜率"], row["无条件胜率"], n))
        lowers.append(tv.edge_lower_bound(row["胜率"], row["无条件胜率"], n))
    out["分数"] = scores
    out["90%下界"] = lowers
    order = [
        "持有期", "信号数", "胜率", "无条件胜率", "分数", "90%下界",
        "平均收益", "无条件平均", "中位收益", "最好", "最差", "对照样本",
    ]
    return out[order]


def _format_stats_table(stats_df):
    """后验统计表：胜率与收益列以百分比展示（底层仍为小数）。"""
    out = stats_df.set_index("持有期").copy()
    for col in ("胜率", "无条件胜率"):
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x) * 100:.1f}%"
            )
    for col in ("平均收益", "无条件平均", "中位收益", "最好", "最差"):
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x) * 100:+.2f}%"
            )
    for col in ("分数", "90%下界"):
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x) * 100:+.1f}"
            )
    return out


def _format_conjunction_table(table):
    """共振对照表：胜率与中位收益以百分比展示。"""
    out = table.set_index("持有期").copy()
    if "共振胜率" in out.columns:
        out["共振胜率"] = out["共振胜率"].map(
            lambda x: "" if pd.isna(x) else f"{float(x) * 100:.1f}%"
        )
    for col in ("共振中位收益", "九转中位收益", "背离中位收益"):
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x) * 100:+.2f}%"
            )
    if "共振信号数" in out.columns:
        out["共振信号数"] = out["共振信号数"].astype("Int64")
    return out


@st.cache_data(show_spinner=False)
def compute_all(
    df, pivot_order, min_count, dedup_bars, ma_n, adx_th,
    filter_mode="aligned",
    _filter_rev=tv.TREND_FILTER_LOGIC_REV,
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """信号 + 趋势滤波；参数变化时自动重算。"""
    del _filter_rev, _dedup_rev
    dfi = tv.add_indicators(df)
    bull_s = pd.Series(0, index=dfi.index)
    bear_s = pd.Series(0, index=dfi.index)
    for n in tv.INDICATORS:
        b, r = tv.find_divergences(dfi, dfi[n], order=pivot_order)
        bull_s.iloc[b] += 1
        bear_s.iloc[r] += 1
    bull_sig = tv.dedup_signals(bull_s[bull_s >= min_count], dfi, gap=dedup_bars)
    bear_sig = tv.dedup_signals(bear_s[bear_s >= min_count], dfi, gap=dedup_bars)
    bull_f, bear_f = tv.apply_trend_filter(
        dfi, bull_sig, bear_sig, ma_n=ma_n, mode=filter_mode, adx_th=adx_th,
    )
    return dfi, bull_sig, bear_sig, bull_f, bear_f

# ---------------- 侧边栏 ----------------
_data_status_slot = st.sidebar.container()
st.sidebar.title("⚙️ 参数")

# 标的列表：配置里的 + 缓存里出现过的。删除会记在 hidden_tickers.txt，并清掉该标的缓存。
_HIDDEN_TICKERS_PATH = tv.CACHE_DIR / "hidden_tickers.txt"
_EXTRA_TICKERS_PATH = tv.CACHE_DIR / "extra_tickers.txt"


def _load_hidden_tickers():
    if not _HIDDEN_TICKERS_PATH.exists():
        return set()
    return {ln.strip() for ln in _HIDDEN_TICKERS_PATH.read_text().splitlines() if ln.strip()}


def _save_hidden_tickers(names):
    tv.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(sorted(names))
    _HIDDEN_TICKERS_PATH.write_text(text + ("\n" if text else ""))


def _load_extra_tickers():
    if not _EXTRA_TICKERS_PATH.exists():
        return []
    return [ln.strip() for ln in _EXTRA_TICKERS_PATH.read_text().splitlines() if ln.strip()]


def _save_extra_tickers(names):
    tv.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(dict.fromkeys(names))
    _EXTRA_TICKERS_PATH.write_text(text + ("\n" if text else ""))


def _cache_ticker_names():
    if not tv.CACHE_DIR.exists():
        return []
    names = []
    for path in tv.CACHE_DIR.glob("*.csv"):
        stem = path.name.rsplit("_", 1)[0]
        if stem.startswith("dual_tf_backtest"):
            continue
        names.append(stem)
    return names


def _remove_ticker(name, options):
    hidden = set(st.session_state.hidden_tickers)
    hidden.add(name)
    st.session_state.hidden_tickers = hidden
    _save_hidden_tickers(hidden)
    extras = [item for item in st.session_state.extra_tickers if item != name]
    st.session_state.extra_tickers = extras
    _save_extra_tickers(extras)
    prefix = tv.safe_chart_name(name)
    if tv.CACHE_DIR.exists():
        for path in tv.CACHE_DIR.glob(f"{prefix}_*.csv"):
            path.unlink(missing_ok=True)
    load_data.clear()
    if st.session_state.get("ticker_choice") == name:
        rest = [item for item in options if item != name]
        st.session_state.ticker_choice = rest[0] if rest else ""


if "hidden_tickers" not in st.session_state:
    st.session_state.hidden_tickers = _load_hidden_tickers()
if "extra_tickers" not in st.session_state:
    st.session_state.extra_tickers = _load_extra_tickers()

_choice = (st.session_state.get("ticker_choice") or "").strip()
_preset_names = _listed_preset_tickers()
_preset_aliases = {tv.safe_chart_name(name) for name in _preset_names}
for _name in _preset_names:
    if _choice == tv.safe_chart_name(_name) and _choice != _name:
        _choice = _name
        st.session_state.ticker_choice = _name
        break
tickers = []
_seen = set()
for item in (
    _preset_names
    + list(tv.TICKERS)
    + _cache_ticker_names()
    + list(st.session_state.extra_tickers)
    + ([_choice] if _choice else [])
):
    if not item or item in st.session_state.hidden_tickers or item in _seen:
        continue
    if item in _preset_aliases and item not in _preset_names:
        continue
    _seen.add(item)
    tickers.append(item)
if not _choice or _choice in st.session_state.hidden_tickers:
    st.session_state.ticker_choice = next(
        (item for item in tickers if not _preset_for(item)),
        tickers[0] if tickers else "",
    )

ticker_menu_slot = st.sidebar.container()

ticker = st.session_state.ticker_choice
if not ticker:
    st.sidebar.warning("列表是空的，请先添加一个标的。")
    st.stop()
period = st.sidebar.selectbox("数据区间", ["2y", "5y", "10y", "max"],
                              index=list(["2y", "5y", "10y", "max"]).index(tv.PERIOD)
                              if tv.PERIOD in ["2y", "5y", "10y", "max"] else 1)
_as_of = _cache_as_of(ticker, period)
with _data_status_slot:
    if _as_of:
        st.badge(
            f"数据更新至 {_as_of}",
            icon=":material/calendar_today:",
            color="gray",
            width="stretch",
            help=f"{ticker} 在「{period}」区间里，本地缓存的最后一根 K 线。",
        )
    else:
        st.badge(
            "数据尚未下载",
            icon=":material/calendar_today:",
            color="orange",
            width="stretch",
            help=f"{ticker} 在「{period}」区间还没有本地缓存。",
        )
    if st.button(
        "更新至最新",
        key="update_to_latest",
        icon=":material/refresh:",
        width="stretch",
        help="列表里的标的都会补到最新。已有的 K 线会保留，只下载还没有的交易日。",
    ):
        _refresh_errors = _refresh_ohlcv_incremental(tickers, period, _data_status_slot)
        load_data.clear()
        st.session_state.price_cache_nonce = st.session_state.get("price_cache_nonce", 0) + 1
        st.session_state.price_refresh_errors = _refresh_errors
        st.rerun()
    _refresh_errors = st.session_state.pop("price_refresh_errors", None)
    if _refresh_errors:
        st.caption("部分标的没有更新：" + "；".join(_refresh_errors))
    st.divider()
if st.sidebar.button(
    "一键优化全部标的",
    icon=":material/tune:",
    width="stretch",
    help="除 KBE、KRE 外，用当前数据区间的日线，按曲线中位数写入固定参数。",
):
    st.session_state.batch_opt_rows = _optimize_listed_tickers(tickers, period)
    st.session_state.batch_opt_period = period
    st.rerun(scope="app")
st.sidebar.caption("除 KBE、KRE 外都会写入固定参数。这两只保持原来的参数。")
_batch_rows = st.session_state.get("batch_opt_rows")
if _batch_rows and st.session_state.get("batch_opt_period") == period:
    _batch_window = _EDGE_WINDOW_LABEL.get(period, period)
    with st.sidebar.expander(f"全部优化结果（近 {_batch_window}）", expanded=False):
        st.caption("KBE、KRE 没有改。")
        st.dataframe(
            pd.DataFrame(_batch_rows),
            hide_index=True,
            width="stretch",
            alt="除 KBE 和 KRE 外各标的的优化分数和参数",
        )
timeframe = st.sidebar.selectbox("K线周期", ["日线", "周线"], index=0)
is_weekly = timeframe == "周线"
force_refresh = st.sidebar.checkbox("强制重新下载（绕过缓存）", value=False)
bar_unit = "周" if is_weekly else "根"
if is_weekly:
    hold_horizon_mode = st.sidebar.selectbox(
        "后验持有期计数",
        ["按周线 K 线根数", "按日线 K 线根数"],
        help="背离信号仍按周线计算。分布图横轴是 3 到 40。"
        "选「日线」时按交易日计数，入场仍为周线 pivot 确认后的收盘对应交易日。",
    )
    hold_horizon_daily = hold_horizon_mode == "按日线 K 线根数"
else:
    hold_horizon_daily = False
recent_window_days = 56 if is_weekly else tv.SIGNAL_RECENT_DAYS
recent_window_label = "8 周" if is_weekly else f"{tv.SIGNAL_RECENT_DAYS} 日"

st.sidebar.divider()
preset = _preset_for(ticker)
if preset:
    origin = _edge_origin(preset.get("source"))
    if origin:
        origin = f"近 {origin}、曲线中位数选出"
    else:
        origin = "日线十年样本选出"
    st.sidebar.caption(f"{ticker} 使用固定参数（{origin}）")
    st.sidebar.markdown(preset["summary"])
    if preset.get("long_only"):
        st.sidebar.caption("分数只算底背离。图上和后验表仍列出顶背离和做空收益。")
    pivot_order = preset["pivot_order"]
    min_count = preset["min_count"]
    dedup_bars = preset["dedup_bars"]
    trend_on = preset["trend_on"]
    aligned_keep = preset["aligned_keep"]
    adx_th = preset["adx_th"]
    ma_n = preset["ma_n"]
else:
    pivot_order = st.sidebar.slider("pivot 灵敏度 PIVOT_ORDER", 3, 12, tv.PIVOT_ORDER)
    min_count = st.sidebar.slider("共振指标数 MIN_COUNT", 2, 8, tv.MIN_COUNT)
    dedup_bars = st.sidebar.slider("去重窗口 DEDUP_BARS", 5, 30, tv.DEDUP_BARS)
if preset is None:
    st.sidebar.divider()
    trend_on = st.sidebar.toggle("趋势滤波（只留顺势信号）", value=True,
                               help="打开后按下方规则过滤信号。关闭则图上保留全部背离。")
    aligned_keep = st.sidebar.toggle(
        "强趋势区保留顺势信号",
        value=True,
        disabled=not trend_on,
        help="打开：深绿只留底背离，深红只留顶背离。"
        "关闭：深绿只留顶背离，深红只留底背离。灰色震荡区两种都全部保留。"
        "未开 ADX 时，关闭此开关则均线上方只留顶背离、下方只留底背离。",
    )
    adx_sel = st.sidebar.selectbox(
        "ADX 趋势强度过滤",
        options=["关闭", ">20（弱趋势起）", ">25（标准）", ">30（强趋势）"],
        index=0,
        help="开启后，只在 ADX 超过阈值（趋势足够强）时才保留顺势信号；关闭则只看价格 vs 均线",
        disabled=not trend_on,
    )
    ADX_MAP = {"关闭": None, ">20（弱趋势起）": 20, ">25（标准）": 25, ">30（强趋势）": 30}
    adx_th = ADX_MAP[adx_sel]
    _ma_default = 40 if is_weekly else 200
    ma_n = st.sidebar.slider(
        "趋势滤波 MA 周期",
        min_value=10,
        max_value=120 if is_weekly else 250,
        value=_ma_default,
        step=1,
        help="黄虚线为趋势分界；背景着色与顺势滤波都看这一根均线。"
        " 图中 MA20 蓝线仅作短周期参考，不参与背景判定。",
        disabled=not trend_on,
    )

_SHARED_EDGE_DEFAULT = {
    "pivot_order": tv.PIVOT_ORDER,
    "min_count": tv.MIN_COUNT,
    "dedup_bars": tv.DEDUP_BARS,
    "trend_on": True,
    "aligned_keep": True,
    "adx_th": None,
    "ma_n": 200,
}
if preset is None and not is_weekly:
    st.session_state.shared_edge_params = {
        "pivot_order": pivot_order,
        "min_count": min_count,
        "dedup_bars": dedup_bars,
        "trend_on": trend_on,
        "aligned_keep": aligned_keep,
        "adx_th": adx_th,
        "ma_n": ma_n,
    }
_shared_edge = st.session_state.get("shared_edge_params", _SHARED_EDGE_DEFAULT)


def _edge_params_for(name, shared):
    """固定参数用自己的一套。其余用最近一次日线侧边栏。分数按日线曲线中位数。"""
    spec = _preset_for(name)
    if spec is None:
        trend = bool(shared["trend_on"])
        if not trend:
            mode = "off"
        elif not shared["aligned_keep"]:
            mode = "counter"
        else:
            mode = "aligned"
        return {
            "pivot_order": int(shared["pivot_order"]),
            "min_count": int(shared["min_count"]),
            "dedup_bars": int(shared["dedup_bars"]),
            "mode": mode,
            "adx_th": shared["adx_th"] if trend else None,
            "ma_n": int(shared["ma_n"]),
            "long_only": False,
        }
    if not spec["trend_on"]:
        mode = "off"
    elif not spec["aligned_keep"]:
        mode = "counter"
    else:
        mode = "aligned"
    return {
        "pivot_order": int(spec["pivot_order"]),
        "min_count": int(spec["min_count"]),
        "dedup_bars": int(spec["dedup_bars"]),
        "mode": mode,
        "adx_th": spec["adx_th"],
        "ma_n": int(spec["ma_n"]),
        "long_only": bool(spec.get("long_only", False)),
    }


def _signals_for_edge(df, params):
    dfi = tv.add_indicators(df)
    bull_s = pd.Series(0, index=dfi.index)
    bear_s = pd.Series(0, index=dfi.index)
    for name in tv.INDICATORS:
        bull_idx, bear_idx = tv.find_divergences(dfi, dfi[name], order=params["pivot_order"])
        bull_s.iloc[bull_idx] += 1
        bear_s.iloc[bear_idx] += 1
    bull = tv.dedup_signals(bull_s[bull_s >= params["min_count"]], dfi, gap=params["dedup_bars"])
    bear = tv.dedup_signals(bear_s[bear_s >= params["min_count"]], dfi, gap=params["dedup_bars"])
    bull, bear = tv.apply_trend_filter(
        dfi, bull, bear, ma_n=params["ma_n"], mode=params["mode"], adx_th=params["adx_th"],
    )
    return bull, bear


@st.cache_data(ttl="15m", max_entries=8, show_spinner="按曲线中位数给标的打分")
def score_ticker_list(
    tickers_tuple, period, shared_key, preset_rev, data_rev,
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """每个标的在自己的参数下，日线 3–40 根收缩超额胜率的中位数，以及两侧尖峰。"""
    del preset_rev, data_rev, _dedup_rev
    shared = dict(shared_key)
    scores = {}
    span = tv.POSTERIOR_SPAN
    for name in tickers_tuple:
        try:
            params = _edge_params_for(name, shared)
            data = tv.load_ohlcv(name, period=period)
            bull, bear = _signals_for_edge(data, params)
            shift = params["pivot_order"]
            long_curve = tv.hold_edge_curve(
                data, bull, short=False, entry_shift=shift, horizons=span,
            )
            short_curve = tv.hold_edge_curve(
                data, bear, short=True, entry_shift=shift, horizons=span,
            )
            scores[name] = {
                "score": tv.book_curve_median(
                    long_curve, short_curve, long_only=params["long_only"],
                ),
                "bull_peak": tv.curve_peak_hold(long_curve),
                "bear_peak": tv.curve_peak_hold(short_curve),
            }
        except Exception:
            scores[name] = {"score": np.nan, "bull_peak": None, "bear_peak": None}
    return scores


def _fmt_edge_points(score):
    if score is None or not np.isfinite(score):
        return ""
    points = float(score) * 100
    if abs(points) < 0.05:
        return "0.0"
    return f"{points:+.1f}"


def _ticker_score_parts(scores, item):
    row = scores.get(item) or {}
    if not isinstance(row, dict):
        return np.nan, None, None
    return row.get("score", np.nan), row.get("bull_peak"), row.get("bear_peak")


def _render_ticker_menu(slot, names, scores):
    ordered = sorted(
        names,
        key=lambda item: (
            not np.isfinite(_ticker_score_parts(scores, item)[0]),
            -(
                _ticker_score_parts(scores, item)[0]
                if np.isfinite(_ticker_score_parts(scores, item)[0])
                else 0.0
            ),
            item,
        ),
    )
    current = st.session_state.ticker_choice
    current_score = _fmt_edge_points(_ticker_score_parts(scores, current)[0])
    label = current or "添加标的"
    if current and _preset_for(current):
        label = f"{label} · 已固定"
    if current_score:
        label = f"{label}  {current_score}"
    with slot:
        st.caption("标的")
        with st.popover(
            label,
            width="stretch",
            help="按曲线中位数从高到低。点某一行切换标的。右侧删除会从列表移除，并删除该标的的本地缓存。",
        ):
            st.caption(
                "分数是 3–40 根收缩超额胜率的中位数，单位是百分点。"
                "尖峰根数写在各标的的按钮说明里，不参与排序。"
                "趋势滤波打开时用滤波后的信号。多空都做时每一根先按笔数加权。只做多只算底背离。"
            )
            new_name = st.text_input(
                "添加标的",
                placeholder="AAPL、0700.HK、USDJPY=X",
                key="new_ticker_input",
            )
            if st.button("添加", key="add_ticker_btn", width="stretch"):
                name = new_name.strip()
                if name:
                    hidden = set(st.session_state.hidden_tickers)
                    hidden.discard(name)
                    st.session_state.hidden_tickers = hidden
                    _save_hidden_tickers(hidden)
                    extras = list(st.session_state.extra_tickers)
                    if name not in extras:
                        extras.append(name)
                    st.session_state.extra_tickers = extras
                    _save_extra_tickers(extras)
                    st.session_state.ticker_choice = name
                    st.rerun()

            def _ticker_row(item):
                spec = _preset_for(item)
                score, bull_peak, bear_peak = _ticker_score_parts(scores, item)
                points = _fmt_edge_points(score)
                title = item
                if spec:
                    title = f"{title} · 已固定"
                if points:
                    title = f"{title}  {points}"
                peak_bits = []
                if bull_peak:
                    peak_bits.append(f"底背离尖峰 {bull_peak} 根")
                if bear_peak:
                    peak_bits.append(f"顶背离尖峰 {bear_peak} 根")
                help_bits = ["曲线中位数是 3–40 根收缩超额胜率的中位数（百分点）"]
                if peak_bits:
                    help_bits.append("，".join(peak_bits))
                help_bits.append("尖峰不参与排序")
                help_text = "。".join(help_bits) + "。"
                if spec:
                    help_text = f"{spec['summary']}。{help_text}"
                pick, remove = st.columns([4, 1], vertical_alignment="center")
                if pick.button(
                    title,
                    key=f"pick_ticker_{item}",
                    width="stretch",
                    type="primary" if item == current else "secondary",
                    help=help_text,
                ):
                    st.session_state.ticker_choice = item
                    st.rerun()
                if remove.button(
                    "删除",
                    key=f"del_ticker_{item}",
                    help=f"从列表删除 {item}",
                ):
                    _remove_ticker(item, names)
                    st.rerun()

            for item in ordered:
                _ticker_row(item)


_edge_scores = score_ticker_list(
    tuple(tickers),
    period,
    tuple(sorted(
        (key, _shared_edge.get(key, _SHARED_EDGE_DEFAULT[key]))
        for key in _SHARED_EDGE_DEFAULT
    )),
    _preset_cache_key(),
    st.session_state.get("price_cache_nonce", 0),
)
_render_ticker_menu(ticker_menu_slot, tickers, _edge_scores)

# ---------------- 计算 ----------------
try:
    df_raw = load_data(ticker, period, use_cache=not force_refresh, timeframe=timeframe)
except Exception as exc:
    st.error(f"无法加载 {ticker} 数据：{exc}")
    st.stop()

if df_raw is None or df_raw.empty:
    st.warning(f"{ticker} 无可用{timeframe}数据，请换标的、换周期或勾选「强制重新下载」。")
    st.stop()

filter_mode = "counter" if trend_on and not aligned_keep else "aligned"
df, bull_sig, bear_sig, bull_f, bear_f = compute_all(
    df_raw, pivot_order, min_count, dedup_bars, ma_n, adx_th,
    filter_mode,
)
df_daily_stats = None
if is_weekly and hold_horizon_daily:
    df_daily_stats = load_data(
        ticker, period, use_cache=not force_refresh, timeframe="日线",
    )
signal_tf = "weekly" if is_weekly else "daily"
_stats_kw = dict(
    entry_shift=pivot_order,
    daily_df=df_daily_stats,
    horizon_count_daily=hold_horizon_daily and is_weekly,
    signal_timeframe=signal_tf,
)
_span = f"{tv.POSTERIOR_SPAN[0]}–{tv.POSTERIOR_SPAN[-1]}"
if is_weekly and hold_horizon_daily:
    hold_caption = (
        f"汇总表为 {', '.join(str(h) for h in tv.STATS_HORIZONS)} 个交易日；"
        f"分布图为持有 {_span} 个交易日（出场按日线收盘）"
    )
elif is_weekly:
    hold_caption = (
        f"汇总表为 {', '.join(str(h) for h in tv.STATS_HORIZONS)} 周；"
        f"分布图为持有 {_span} 周"
    )
else:
    hold_caption = (
        f"汇总表为 {', '.join(str(h) for h in tv.STATS_HORIZONS)} 根日线；"
        f"分布图为持有 {_span} 根日线（40 根约两个月）"
    )
bull_plot, bear_plot = (bull_f, bear_f) if trend_on else (bull_sig, bear_sig)

# 主图横轴跟「数据区间」走。2y/5y/10y 是最近对应年数；max 用已加载数据的全长。
# 信号和后验统计仍用整段已加载数据，不跟着这根轴裁。
_PERIOD_YEARS = {"2y": 2, "5y": 5, "10y": 10}
chart_end = df.index[-1]
if period == "max":
    chart_start = df.index[0]
else:
    chart_start = chart_end - pd.DateOffset(years=_PERIOD_YEARS.get(period, 5))
disp = df.loc[df.index >= chart_start]

# ---------------- 顶部指标条 ----------------
last = df.Close.iloc[-1]
c1, c2 = st.columns(2)
c1.metric("最新收盘", f"{last:,.2f}")
n_recent = len(
    tv.signals_in_window(
        bull_plot, bear_plot, df, ticker, days=recent_window_days
    )
)
c2.metric(f"近{recent_window_label}信号", n_recent)

_param_optimize_panel(ticker, period)

# ---------------- 侧边栏：标的列表近期信号扫描 ----------------
@st.cache_data(ttl=900, show_spinner=False)
def _params_for(ticker, pivot_order, min_count, dedup_bars, preset_rev):
    """有固定参数的标的用自己的一套，其余用当前侧边栏。"""
    del preset_rev
    preset = _preset_for(ticker)
    if preset is None:
        return pivot_order, min_count, dedup_bars, False, "aligned", None, 200, False
    mode = "counter" if preset["trend_on"] and not preset["aligned_keep"] else "aligned"
    if not preset["trend_on"]:
        mode = "off"
    return (
        preset["pivot_order"],
        preset["min_count"],
        preset["dedup_bars"],
        preset["trend_on"],
        mode,
        preset["adx_th"],
        preset["ma_n"],
        bool(preset.get("long_only", False)),
    )


def scan_recent_signals(
    tickers_tuple, period, days, pivot_order, min_count, dedup_bars,
    preset_rev,
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """扫描标的列表里近期有背离信号的 ticker。固定参数的标的用自己的一套。"""
    del _dedup_rev
    rows = []
    for t in tickers_tuple:
        try:
            po, mc, gap, use_filter, mode, adx_th, ma_n, _long_only = _params_for(
                t, pivot_order, min_count, dedup_bars, preset_rev,
            )
            d = tv.load_ohlcv(t, period=period)
            d = tv.add_indicators(d)
            bull_s = pd.Series(0, index=d.index)
            bear_s = pd.Series(0, index=d.index)
            for n in tv.INDICATORS:
                b, r = tv.find_divergences(d, d[n], order=po)
                bull_s.iloc[b] += 1
                bear_s.iloc[r] += 1
            bull = tv.dedup_signals(bull_s[bull_s >= mc], d, gap=gap)
            bear = tv.dedup_signals(bear_s[bear_s >= mc], d, gap=gap)
            if use_filter:
                bull, bear = tv.apply_trend_filter(
                    d, bull, bear, ma_n=ma_n, mode=mode, adx_th=adx_th,
                )
            sigs = tv.signals_in_window(bull, bear, d, t, days=days)
            if sigs:
                down_n, up_n = tv.td_completed_nines(d.Close)
                long_pairs = tv.confirmed_nine_overlap_pairs(d, bull, down_n, po)
                short_pairs = tv.confirmed_nine_overlap_pairs(d, bear, up_n, po)
                latest = sorted(sigs, key=lambda x: x["date"], reverse=True)[0]
                latest_pairs = long_pairs if latest["signal"] == "底背离" else short_pairs
                latest_note = tv.nine_overlap_note(latest_pairs, latest["date"], po)
                any_note = False
                for one in sigs:
                    one_pairs = long_pairs if one["signal"] == "底背离" else short_pairs
                    if tv.nine_overlap_note(one_pairs, one["date"], po):
                        any_note = True
                        break
                if latest_note:
                    nine_col = "这一条是"
                elif any_note:
                    nine_col = "窗口内有"
                else:
                    nine_col = "—"
                rows.append({
                    "ticker": t,
                    "最新信号": latest["signal"],
                    "日期": latest["date"],
                    "共振数": latest["indicator_count"],
                    "信号数(窗口内)": len(sigs),
                    "九转配合": nine_col,
                })
        except Exception:
            continue  # 某个标的拉取失败不阻塞整体扫描
    return pd.DataFrame(rows)

# 放侧边栏：展开面板，不占用主界面
with st.sidebar.expander(f"🔔 近{recent_window_label}有信号的标的", expanded=True):
    scan = scan_recent_signals(
        tuple(tickers), period, recent_window_days,
        pivot_order, min_count, dedup_bars,
        _preset_cache_key(),
    )
    if scan.empty:
        st.success(f"近 {recent_window_label} 全部标的无背离信号")
    else:
        st.caption("「九转配合」为「这一条是」表示最新信号同时满足九转第 7、8、9 根与背离点重合；「窗口内有」表示近窗口里另有一条满足。")
        # 底背离绿、顶背离红，便于一眼区分
        def _color(sig):
            return "color: #26a69a" if sig == "底背离" else "color: #ef5350"
        st.dataframe(
            scan.sort_values("日期", ascending=False).style.map(
                lambda s: _color(s), subset=["最新信号"]),
            width="stretch", hide_index=True,
        )


@st.cache_data(ttl="15m", max_entries=8, show_spinner=False)
def method_choice_rows(
    tickers_tuple, period, pivot_order, min_count, dedup_bars, preset_rev, data_rev,
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """每个标的的主方法。信号和近窗扫描同一套：固定参数，开了趋势滤波就用滤波后的背离。"""
    del data_rev, _dedup_rev
    rows = []
    for name in tickers_tuple:
        try:
            po, mc, gap, use_filter, mode, adx_i, ma_i, long_only = _params_for(
                name, pivot_order, min_count, dedup_bars, preset_rev,
            )
            data = tv.add_indicators(tv.load_ohlcv(name, period=period))
            bull_s = pd.Series(0, index=data.index)
            bear_s = pd.Series(0, index=data.index)
            for ind in tv.INDICATORS:
                b_idx, r_idx = tv.find_divergences(data, data[ind], order=po)
                bull_s.iloc[b_idx] += 1
                bear_s.iloc[r_idx] += 1
            bull = tv.dedup_signals(bull_s[bull_s >= mc], data, gap=gap)
            bear = tv.dedup_signals(bear_s[bear_s >= mc], data, gap=gap)
            if use_filter:
                bull, bear = tv.apply_trend_filter(
                    data, bull, bear, ma_n=ma_i, mode=mode, adx_th=adx_i,
                )
            choice = tv.evaluate_method_choice(
                data, bull, bear, po, long_only=bool(long_only),
            )
            addon_bits = []
            if choice["long"]["adopted"] and choice["long"]["addon"]:
                addon_bits.append("做多")
            if choice["short"]["adopted"] and choice["short"]["addon"]:
                addon_bits.append("做空")
            rows.append({
                "标的": name,
                "做多主方法": tv.method_primary_label(choice["long"]),
                "做空主方法": tv.method_primary_label(choice["short"]),
                "共振加仓": "、".join(addon_bits) if addon_bits else "—",
            })
        except Exception:
            rows.append({
                "标的": name,
                "做多主方法": "失败",
                "做空主方法": "失败",
                "共振加仓": "—",
            })
    return pd.DataFrame(rows)


_method_list = st.sidebar.expander("各标的主方法", on_change="rerun")
if _method_list.open:
    with _method_list:
        st.caption(
            "与上面的近窗扫描同一套日线和固定参数。"
            "主方法只在背离和九转之间产生，持有 3–40 根的下界中位数大于 0 才过线。"
            "共振加仓不替换主方法。这是全样本比较，换一段行情可能换方法。"
        )
        _method_list_rows = method_choice_rows(
            tuple(tickers),
            period,
            pivot_order,
            min_count,
            dedup_bars,
            _preset_cache_key(),
            st.session_state.get("price_cache_nonce", 0),
        )
        st.dataframe(
            _method_list_rows,
            width="stretch",
            hide_index=True,
            alt="各标的做多、做空主方法，以及共振是否加仓",
        )

dual_scan_days = max(recent_window_days, 42)


@st.cache_data(ttl=900, show_spinner=False)
def scan_dual_timeframe_divergence(
    tickers_tuple,
    period,
    recent_days,
    pivot_order,
    min_count,
    dedup_bars,
    tolerance,
    preset_rev,
    data_rev,
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """日 K 与周 K 同向背离，且信号日期相差 <= tolerance 个交易日。"""
    del data_rev, _dedup_rev
    rows = []
    for t in tickers_tuple:
        try:
            po, mc, gap, _use_filter, _mode, _adx, _ma, _long_only = _params_for(
                t, pivot_order, min_count, dedup_bars, preset_rev,
            )
            d = tv.load_ohlcv(t, period=period)
            bull_m, bear_m = tv.dual_timeframe_divergence_matches(
                d,
                pivot_order=po,
                min_count=mc,
                dedup_bars=gap,
                tolerance=tolerance,
                recent_days=recent_days,
            )
            if bull_m:
                rows.append(
                    {
                        "ticker": t,
                        "方向": "底背离",
                        "日线日期": bull_m["daily_date"],
                        "周线日期": bull_m["weekly_date"],
                        "相差(交易日)": bull_m["gap_days"],
                        "日共振": bull_m["daily_count"],
                        "周共振": bull_m["weekly_count"],
                    }
                )
            if bear_m:
                rows.append(
                    {
                        "ticker": t,
                        "方向": "顶背离",
                        "日线日期": bear_m["daily_date"],
                        "周线日期": bear_m["weekly_date"],
                        "相差(交易日)": bear_m["gap_days"],
                        "日共振": bear_m["daily_count"],
                        "周共振": bear_m["weekly_count"],
                    }
                )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("日线日期", ascending=False)


with st.sidebar.expander(
    f"📊 日+周同向背离（≤{tv.DUAL_TF_TOLERANCE_DAYS} 交易日）",
    expanded=False,
):
    st.caption(
        f"在近 {dual_scan_days} 个自然日内有日线信号，且与任一周线同向背离 "
        f"相距 ≤ {tv.DUAL_TF_TOLERANCE_DAYS} 根交易日（参数与上方 pivot/共振/去重一致）。"
    )
    dual = scan_dual_timeframe_divergence(
        tuple(tickers),
        period,
        dual_scan_days,
        pivot_order,
        min_count,
        dedup_bars,
        tv.DUAL_TF_TOLERANCE_DAYS,
        _preset_cache_key(),
        st.session_state.get("price_cache_nonce", 0),
    )
    if dual.empty:
        st.info("当前列表无日周双周期共振背离")
    else:

        def _dir_color(sig):
            return "color: #26a69a" if sig == "底背离" else "color: #ef5350"

        st.dataframe(
            dual.style.map(lambda s: _dir_color(s), subset=["方向"]),
            width="stretch",
            hide_index=True,
        )

# ---------------- 主图 ----------------
fig = make_subplots(rows=1, cols=1)
fig.add_trace(
    go.Candlestick(
        x=disp.index,
        open=disp.Open,
        high=disp.High,
        low=disp.Low,
        close=disp.Close,
        name=ticker,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        hovertemplate=(
            "日期 %{x|%Y-%m-%d}<br>"
            "开 %{open:,.2f} 高 %{high:,.2f}<br>"
            "低 %{low:,.2f} 收 %{close:,.2f}<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)
_td_up, _td_dn = tv.td_sequential_counts(df.Close)
_td_up = _td_up.reindex(disp.index)
_td_dn = _td_dn.reindex(disp.index)
_td_up_hit = _td_up[_td_up > 0]
_td_dn_hit = _td_dn[_td_dn > 0]
if len(_td_up_hit):
    fig.add_trace(go.Scatter(
        x=_td_up_hit.index, y=df.High.reindex(_td_up_hit.index) * 1.012,
        mode="text", text=[str(int(v)) for v in _td_up_hit],
        textposition="top center", textfont=dict(size=12, color="#c62828"),
        name="上涨九转", hovertemplate="上涨九转 %{text}<extra></extra>",
    ), row=1, col=1)
if len(_td_dn_hit):
    fig.add_trace(go.Scatter(
        x=_td_dn_hit.index, y=df.Low.reindex(_td_dn_hit.index) * 0.988,
        mode="text", text=[str(int(v)) for v in _td_dn_hit],
        textposition="bottom center", textfont=dict(size=12, color="#2e7d32"),
        name="下跌九转", hovertemplate="下跌九转 %{text}<extra></extra>",
    ), row=1, col=1)
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
# 趋势背景：与顺势滤波一致；开 ADX 时深绿/深红 = MA 方向且 ADX 过关
if trend_on:
    ma_line = df.Close.rolling(ma_n).mean().loc[disp.index]
    fig.add_trace(go.Scatter(x=disp.index, y=ma_line,
        name=f"MA{ma_n}（趋势分界·背景按此线）",
        line=dict(color="#ffee58", width=2.5, dash="dot")), row=1, col=1)
    _add_trend_vrects(
        fig, tv.trend_regime_per_bar(df, ma_n, adx_th=adx_th, index=disp.index),
    )
# 背离标注：显示窗口内全部信号；顺势/震荡逆势=实心，剔除=空心「·逆」
ref_bull = bull_f if trend_on else bull_sig
ref_bear = bear_f if trend_on else bear_sig
b_all = bull_sig[bull_sig.index >= chart_start]
r_all = bear_sig[bear_sig.index >= chart_start]
_trend_masks = tv.trend_filter_masks(df, ma_n, adx_th) if trend_on else None


_counter = trend_on and filter_mode == "counter"
if _trend_masks is not None and _counter:
    _bull_zone, _bear_zone = _trend_masks["aligned_bear"], _trend_masks["aligned_bull"]
elif _trend_masks is not None:
    _bull_zone, _bear_zone = _trend_masks["aligned_bull"], _trend_masks["aligned_bear"]
else:
    _bull_zone = _bear_zone = None
_kept_name = "逆势" if _counter else ("顺势" if trend_on else "未滤波")


def _split_sig(sig, ref, zone_keep):
    kept = sig[sig.index.isin(ref.index)]
    drop = sig[~sig.index.isin(ref.index)]
    if zone_keep is None or adx_th is None or len(kept) == 0:
        return kept, pd.Series(dtype=float), drop
    al = zone_keep.reindex(kept.index).fillna(False)
    ct = _trend_masks["chop"].reindex(kept.index).fillna(False)
    warm = ~(al | ct)
    return kept[al | warm], kept[ct], drop


b_al, b_ct, b_drop = _split_sig(b_all, ref_bull, _bull_zone)
r_al, r_ct, r_drop = _split_sig(r_all, ref_bear, _bear_zone)
if len(b_al):
    fig.add_trace(go.Scatter(x=b_al.index, y=df.Low.loc[b_al.index] * 0.985,
        mode="markers+text", text=[f"底x{int(v)}" for v in b_al],
        textposition="bottom center",
        marker=dict(symbol="triangle-up", size=12, color="gold"),
        name=f"底背离({_kept_name})"), row=1, col=1)
if len(b_ct):
    fig.add_trace(go.Scatter(x=b_ct.index, y=df.Low.loc[b_ct.index] * 0.985,
        mode="markers+text", text=[f"底x{int(v)}·震" for v in b_ct],
        textposition="bottom center",
        marker=dict(symbol="triangle-up", size=12, color="#26c6da"),
        name="底背离(震荡·全保留)"), row=1, col=1)
if len(b_drop):
    fig.add_trace(go.Scatter(x=b_drop.index, y=df.Low.loc[b_drop.index] * 0.985,
        mode="markers+text", text=[f"底x{int(v)}·滤" for v in b_drop],
        textposition="bottom center",
        marker=dict(symbol="triangle-up-open", size=12, color="#b8860b"),
        name="底背离(已过滤)"), row=1, col=1)
if len(r_al):
    fig.add_trace(go.Scatter(x=r_al.index, y=df.High.loc[r_al.index] * 1.015,
        mode="markers+text", text=[f"顶x{int(v)}" for v in r_al],
        textposition="top center",
        marker=dict(symbol="triangle-down", size=12, color="#ce93d8"),
        name=f"顶背离({_kept_name})"), row=1, col=1)
if len(r_ct):
    fig.add_trace(go.Scatter(x=r_ct.index, y=df.High.loc[r_ct.index] * 1.015,
        mode="markers+text", text=[f"顶x{int(v)}·震" for v in r_ct],
        textposition="top center",
        marker=dict(symbol="triangle-down", size=12, color="#ff7043"),
        name="顶背离(震荡·全保留)"), row=1, col=1)
if len(r_drop):
    fig.add_trace(go.Scatter(x=r_drop.index, y=df.High.loc[r_drop.index] * 1.015,
        mode="markers+text", text=[f"顶x{int(v)}·滤" for v in r_drop],
        textposition="top center",
        marker=dict(symbol="triangle-down-open", size=12, color="#6a1b9a"),
        name="顶背离(已过滤)"), row=1, col=1)

_td_completed_dn, _td_completed_up = tv.td_completed_nines(df.Close)
_overlap_long_pairs = tv.confirmed_nine_overlap_pairs(
    df, bull_sig, _td_completed_dn, pivot_order,
)
_overlap_short_pairs = tv.confirmed_nine_overlap_pairs(
    df, bear_sig, _td_completed_up, pivot_order,
)
_overlap_long = tv.confirmed_nine_overlap(df, bull_sig, _td_completed_dn, pivot_order)
_overlap_short = tv.confirmed_nine_overlap(df, bear_sig, _td_completed_up, pivot_order)


def _chart_overlap(sig):
    if sig is None or len(sig) == 0:
        return sig
    return sig[sig.index >= chart_start]


def _mark_nine_overlap(sig, bullish):
    visible = _chart_overlap(sig)
    if visible is None or len(visible) == 0:
        return
    label = "九转∩底" if bullish else "九转∩顶"
    color = "#00E5FF" if bullish else "#FFD600"
    price = df.Low if bullish else df.High
    scale = 0.955 if bullish else 1.045
    ay = 48 if bullish else -48
    y = price.reindex(visible.index) * scale
    fig.add_trace(go.Scatter(
        x=visible.index,
        y=y,
        mode="markers",
        marker=dict(
            symbol="star",
            size=30,
            color=color,
            line=dict(color="#000000", width=2),
        ),
        name=f"九转配合{'底' if bullish else '顶'}背离",
        hovertemplate=(
            "九转配合已确认背离<br>%{x|%Y-%m-%d} 入场"
            "<extra></extra>"
        ),
    ), row=1, col=1)
    for ts, yy in y.items():
        fig.add_annotation(
            x=ts,
            y=float(yy),
            text=label,
            showarrow=False,
            font=dict(size=15, color="#000000"),
            bgcolor=color,
            bordercolor="#000000",
            borderwidth=2,
            borderpad=3,
            yshift=ay,
        )


_mark_nine_overlap(_overlap_long, True)
_mark_nine_overlap(_overlap_short, False)

fig.update_layout(
    title=f"{ticker} [{timeframe}] 背离 + 布林带",
    xaxis_rangeslider_visible=False,
    height=720,
    legend=dict(orientation="h", y=1.04),
    hovermode="x unified",
    dragmode="zoom",
)
_price_axis = dict(range=[chart_start, chart_end], row=1, col=1)
if not is_weekly:
    _price_axis["rangebreaks"] = [dict(bounds=["sat", "mon"])]
fig.update_xaxes(**_price_axis)
_apply_price_chart_crosshair(fig, row=1, col=1)

_chart_safe = tv.safe_chart_name(ticker)
_main_chart_key = f"main_price_chart_{_chart_safe}"
_m1, _m2, _m3 = st.columns([1.1, 1, 4])
_measure_on = _m1.toggle(
    "区间测量 (A→B)",
    key=f"measure_toggle_{_chart_safe}",
    help="用下方日期选 A/B，不占用图表点击；主图可正常框选缩放与滚轮缩放。",
)
_disp_d_min = disp.index.min().date()
_disp_d_max = disp.index.max().date()
_measure_a_key = f"measure_a_{_chart_safe}"
_measure_b_key = f"measure_b_{_chart_safe}"
if _m2.button("重置 A/B", key=f"measure_clear_{_chart_safe}"):
    st.session_state[_measure_a_key] = _disp_d_min
    st.session_state[_measure_b_key] = _disp_d_max
    st.rerun()

_ta = _tb = _pa = _pb = None
_mstats = None
if _measure_on:
    _m3.caption(
        "在下方选 **起点 A / 终点 B**（自动对齐最近一根 K 线）；"
        "图表仍可用拖拽框选、滚轮缩放。涨跌幅按 **收盘价** 计算。"
    )
    _mda, _mdb = st.columns(2)
    with _mda:
        st.date_input(
            "起点 A",
            value=_disp_d_min,
            min_value=_disp_d_min,
            max_value=_disp_d_max,
            key=_measure_a_key,
            help="非交易日会自动对齐到最近的 K 线。",
        )
    with _mdb:
        st.date_input(
            "终点 B",
            value=_disp_d_max,
            min_value=_disp_d_min,
            max_value=_disp_d_max,
            key=_measure_b_key,
        )
    _ta = _nearest_index_ts(disp.index, st.session_state[_measure_a_key])
    _tb = _nearest_index_ts(disp.index, st.session_state[_measure_b_key])
    if _ta is not None and _tb is not None:
        _pa = float(disp.loc[_ta, "Close"])
        _pb = float(disp.loc[_tb, "Close"])
        _mstats = _measure_stats(_ta, _pa, _tb, _pb, disp.index)
        _add_measure_overlay(fig, _ta, _pa, _tb, _pb, _mstats, row=1, col=1)

_plot_cfg = {
    "scrollZoom": True,
    "displaylogo": False,
}
st.plotly_chart(
    fig,
    width="stretch",
    alt=f"{ticker} 价格与背离",
    key=_main_chart_key,
    on_select="ignore",
    config=_plot_cfg,
)
_n_overlap_vis = len(_chart_overlap(_overlap_long)) + len(_chart_overlap(_overlap_short))
st.caption(
    "大号星标是九转配合已确认背离的入场，和普通三角不是同一种标记。"
    f"九转用计数 7、8、9，背离点左右各 {pivot_order} 根，入场取较晚的一根收盘。"
    f" 这张图上有 {_n_overlap_vis} 个。"
)

if _measure_on and _mstats is not None:
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("A → B 涨跌幅", f"{_mstats['pct']:+.2f}%")
    _c2.metric("自然日跨度", f"{_mstats['cal_days']} 天")
    _tbars = _mstats["trade_bars"]
    _c3.metric("K 线根数", f"{_tbars}" if _tbars is not None else "—")
    _c4.metric(
        "对齐后区间",
        f"{_ta.strftime('%Y-%m-%d')} → {_tb.strftime('%Y-%m-%d')}",
    )
    if (
        st.session_state[_measure_a_key] != _ta.date()
        or st.session_state[_measure_b_key] != _tb.date()
    ):
        st.caption(
            f"日期已对齐到 K 线：A={_ta.strftime('%Y-%m-%d')}，"
            f"B={_tb.strftime('%Y-%m-%d')}。"
        )

if trend_on:
    _rule_name = "逆势保留" if _counter else "顺势保留"
    with st.expander(f"主图背景色说明（MA{ma_n} 黄虚线 · {_rule_name}）", expanded=True):
        st.dataframe(
            _trend_background_legend(adx_th, ma_n, counter=_counter),
            width="stretch",
            hide_index=True,
            alt="主图背景色与信号保留规则",
        )
        st.markdown("**顶/底背离在各背景下的处理**")
        st.dataframe(
            _signal_filter_matrix(adx_th, counter=_counter),
            width="stretch",
            hide_index=True,
            alt="顶底背离过滤规则矩阵",
        )
        if _counter and adx_th is not None:
            _edge_note = "深绿里的底背离、深红里的顶背离标 ·滤。"
        elif adx_th is not None:
            _edge_note = "深绿里的顶背离、深红里的底背离标 ·滤。"
        else:
            _edge_note = "被过滤的方向标 ·滤。"
        st.caption(
            "「·滤」= 该环境下规则不保留（不是信号失效）；"
            "「·震」= 震荡段保留的背离。"
            f" {_edge_note}"
            " 判定按信号当日收盘 vs MA，不是整段 K 线视觉位置。"
        )

# ---------------- 统计区 ----------------
st.subheader("信号后验统计（基于全量数据）")
st.caption(
    f"入场 = 信号 bar 后第 {pivot_order} 根{bar_unit}线收盘价（pivot 确认）；{hold_caption}"
    " 无条件胜率、无条件平均是对照：当前数据区间里，每个还能持满的交易日都按同一方向做一笔。"
    "底背离表按买入，顶背离表按做空。随机抽与信号数相同的时点，期望胜率和平均收益就等于这两列。"
    "对照样本是这些交易日的个数。用来看信号有没有超出这段行情本身的涨跌。"
    " 分数和 90%下界只填持有 10 日那一行，单位是百分点，用来和旧口径对照。"
    "标的列表和一键优化用分布图说明里的曲线中位数。"
    "分数 = (信号胜率 − 无条件胜率) × n / (n + 10)。"
    "90%下界 = (信号胜率 − 无条件胜率) − 1.64 × √[信号胜率 × (1 − 信号胜率) / n]。"
    "胜率正好是 0% 或 100% 时，开方项为 0。"
)
long_only = bool(preset and preset.get("long_only"))
_base_kw = dict(
    daily_df=df_daily_stats,
    horizon_count_daily=hold_horizon_daily and is_weekly,
    signal_timeframe=signal_tf,
)
base_long = tv.unconditional_stats(df, short=False, **_base_kw)
base_short = tv.unconditional_stats(df, short=True, **_base_kw)
stats_bull = _attach_unconditional(
    tv.signal_stats(df, bull_sig, short=False, **_stats_kw), base_long,
)
stats_bull_f = _attach_unconditional(
    tv.signal_stats(df, bull_f, short=False, **_stats_kw), base_long,
)
stats_bear = _attach_unconditional(
    tv.signal_stats(df, bear_sig, short=True, **_stats_kw), base_short,
)
stats_bear_f = _attach_unconditional(
    tv.signal_stats(df, bear_f, short=True, **_stats_kw), base_short,
)
if _counter:
    filt_desc = f"逆势保留：MA{ma_n} 上方只留顶背离、下方只留底背离"
else:
    filt_desc = f"顺势滤波：MA{ma_n} 上方只留底背离、下方只留顶背离"
if adx_th is not None:
    filt_desc += f"；ADX>{adx_th} 的强趋势区才按方向取舍，灰色震荡全部保留"
if long_only:
    filt_desc += "。分数只算底背离；下表仍列出顶背离的做空收益"


def _render_posterior_panel(
    bull_s, bear_s, stats_b, stats_be, filt_note=None, panel_key="raw",
):
    if filt_note:
        st.caption(filt_note)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**底背离（期望上涨）**")
        st.dataframe(
            _format_stats_table(stats_b), width="stretch",
            alt="底背离后验统计", key=f"{panel_key}_bull_stats",
        )
    with col_r:
        st.markdown("**顶背离（期望下跌）**")
        st.dataframe(
            _format_stats_table(stats_be), width="stretch",
            alt="顶背离后验统计", key=f"{panel_key}_bear_stats",
        )
    st.subheader("后验收益分布")
    span_unit = (
        "个交易日" if _stats_kw["horizon_count_daily"] else ("周" if is_weekly else "根")
    )
    bull_curve = tv.hold_edge_curve(
        df, bull_s, short=False, horizons=tv.POSTERIOR_SPAN,
        entry_shift=_stats_kw["entry_shift"],
        daily_df=_stats_kw["daily_df"],
        horizon_count_daily=_stats_kw["horizon_count_daily"],
    )
    bear_curve = tv.hold_edge_curve(
        df, bear_s, short=True, horizons=tv.POSTERIOR_SPAN,
        entry_shift=_stats_kw["entry_shift"],
        daily_df=_stats_kw["daily_df"],
        horizon_count_daily=_stats_kw["horizon_count_daily"],
    )
    curve_score = tv.book_curve_median(bull_curve, bear_curve, long_only=long_only)
    bull_peak = tv.curve_peak_hold(bull_curve)
    bear_peak = tv.curve_peak_hold(bear_curve)
    bull_peak_text = f"{bull_peak} {span_unit}" if bull_peak else "—"
    bear_peak_text = f"{bear_peak} {span_unit}" if bear_peak else "—"
    st.caption(
        "上排未扣无条件平均：实线是信号中位数，灰虚线是无条件中位数。"
        "下排已扣无条件平均：每笔减去同一持有期、同一方向的无条件平均，实线是超额平均，0 轴表示没有 alpha。"
        f" 曲线中位数 {_fmt_edge_points(curve_score) or '—'}："
        f"持有 {tv.POSTERIOR_SPAN[0]}–{tv.POSTERIOR_SPAN[-1]} {span_unit}的收缩超额胜率取中位数。"
        f"底背离尖峰 {bull_peak_text}，顶背离尖峰 {bear_peak_text}。"
        "竖线标在该侧超额平均最高的持有期上。这是样本内尖峰，不用于排名。"
    )
    fig2 = tv.posterior_returns_plotly(
        df, bull_s, bear_s, horizons=tv.POSTERIOR_SPAN, **_stats_kw,
    )
    st.plotly_chart(
        fig2, width="stretch", alt="背离信号后验收益分布",
        key=f"{panel_key}_posterior",
    )


_filt_tab = "趋势滤波后（逆势）" if _counter else "趋势滤波后（顺势）"
tab_raw, tab_filt = st.tabs(["原始信号", _filt_tab])
with tab_raw:
    _render_posterior_panel(
        bull_sig, bear_sig, stats_bull, stats_bear, panel_key="raw",
    )
with tab_filt:
    _render_posterior_panel(
        bull_f, bear_f, stats_bull_f, stats_bear_f, filt_note=filt_desc,
        panel_key="filtered",
    )

st.subheader("九转后验")
if is_weekly:
    _td_when = "第九根周 K 成立后，下一根周 K 的收盘"
else:
    _td_when = "第九根日 K 成立后，下一根日 K 的收盘"
st.caption(
    f"下跌九转在{_td_when}买入；上涨九转在同一根收盘做空。"
    "只计已经走完的完整 9，贴在最后一根、还没数到 9 的不计。"
    f"表的持有期、胜率和无条件对照与上面的背离后验相同。{hold_caption}"
)
_td_long, _td_short = tv.td_completed_nines(df.Close)
_td_kw = dict(_stats_kw)
_td_kw["entry_shift"] = 1
_td_long_stats = _attach_unconditional(
    tv.signal_stats(df, _td_long, short=False, **_td_kw), base_long,
)
_td_short_stats = _attach_unconditional(
    tv.signal_stats(df, _td_short, short=True, **_td_kw), base_short,
)
_td_col_l, _td_col_r = st.columns(2)
with _td_col_l:
    st.markdown("**下跌九转（买入）**")
    st.dataframe(
        _format_stats_table(_td_long_stats), width="stretch",
        alt="下跌九转后验统计", key="td_long_stats",
    )
with _td_col_r:
    st.markdown("**上涨九转（做空）**")
    st.dataframe(
        _format_stats_table(_td_short_stats), width="stretch",
        alt="上涨九转后验统计", key="td_short_stats",
    )
_td_long_curve = tv.hold_edge_curve(
    df, _td_long, short=False, horizons=tv.POSTERIOR_SPAN,
    entry_shift=1,
    daily_df=_stats_kw["daily_df"],
    horizon_count_daily=_stats_kw["horizon_count_daily"],
)
_td_short_curve = tv.hold_edge_curve(
    df, _td_short, short=True, horizons=tv.POSTERIOR_SPAN,
    entry_shift=1,
    daily_df=_stats_kw["daily_df"],
    horizon_count_daily=_stats_kw["horizon_count_daily"],
)
_td_long_peak = tv.curve_peak_hold(_td_long_curve)
_td_short_peak = tv.curve_peak_hold(_td_short_curve)
_td_unit = (
    "个交易日" if _stats_kw["horizon_count_daily"] else ("周" if is_weekly else "根")
)
_td_long_peak_text = f"{_td_long_peak} {_td_unit}" if _td_long_peak else "—"
_td_short_peak_text = f"{_td_short_peak} {_td_unit}" if _td_short_peak else "—"
st.caption(
    f"下跌九转 {len(_td_long)} 次，上涨九转 {len(_td_short)} 次。"
    f"下跌九转尖峰 {_td_long_peak_text}，上涨九转尖峰 {_td_short_peak_text}。"
    "上排未扣无条件平均：实线是信号中位数，灰虚线是无条件中位数。"
    "下排已扣无条件平均：每笔减去同一持有期、同一方向的无条件平均，0 轴表示没有超出这段行情本身。"
)
fig_td = tv.posterior_returns_plotly(
    df, _td_long, _td_short,
    horizons=tv.POSTERIOR_SPAN,
    long_name="下跌九转",
    short_name="上涨九转",
    **_td_kw,
)
st.plotly_chart(
    fig_td, width="stretch", alt="九转信号后验收益分布",
    key="td_posterior",
)

_conj_kw = dict(
    daily_df=_stats_kw["daily_df"],
    horizon_count_daily=_stats_kw["horizon_count_daily"],
    signal_timeframe=_stats_kw["signal_timeframe"],
)


def _render_conjunction(title, caption, long_sig, short_sig, long_name, short_name,
                        entry_shift, panel_key):
    st.subheader(title)
    st.caption(caption)
    long_table = tv.conjunction_compare_table(
        df, long_sig, _td_long, bull_sig, short=False,
        joint_shift=entry_shift, nine_shift=1, div_shift=pivot_order, **_conj_kw,
    )
    short_table = tv.conjunction_compare_table(
        df, short_sig, _td_short, bear_sig, short=True,
        joint_shift=entry_shift, nine_shift=1, div_shift=pivot_order, **_conj_kw,
    )
    st.caption(
        f"{long_name} {len(long_sig)} 次，{short_name} {len(short_sig)} 次。"
        "表里的共振信号数只计还能持满的笔。九转中位收益、背离中位收益是单独信号，不是交集。"
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**{long_name}（买入）**")
        st.dataframe(
            _format_conjunction_table(long_table), width="stretch",
            alt=f"{long_name}胜率与中位收益", key=f"{panel_key}_long_table",
        )
    with col_r:
        st.markdown(f"**{short_name}（做空）**")
        st.dataframe(
            _format_conjunction_table(short_table), width="stretch",
            alt=f"{short_name}胜率与中位收益", key=f"{panel_key}_short_table",
        )
    fig_joint = tv.posterior_returns_plotly(
        df, long_sig, short_sig,
        horizons=tv.POSTERIOR_SPAN,
        long_name=long_name,
        short_name=short_name,
        entry_shift=entry_shift,
        **_conj_kw,
    )
    st.plotly_chart(
        fig_joint, width="stretch", alt=f"{title}后验收益分布",
        key=f"{panel_key}_posterior",
    )


_bar_name = "周" if is_weekly else "日"
_render_conjunction(
    "九转配合已确认背离",
    "配合测试，不替换上面的背离信号。"
    f"九转用计数 7、8、9 这三根{_bar_name} K，"
    f"背离点左右各 {pivot_order} 根{_bar_name} K。"
    "两边有重合就记一次。"
    f"入场是较晚的一根收盘：九转第 9 根，或背离确认的最后一根"
    f"（摆动点后再数 {pivot_order} 根）。"
    f" 持有期与上面相同。{hold_caption}",
    _overlap_long, _overlap_short,
    "下跌九转∩底背离", "上涨九转∩顶背离",
    entry_shift=0, panel_key="overlap",
)

_method_choice = tv.evaluate_method_choice(
    df, bull_plot, bear_plot, pivot_order,
    long_only=long_only,
    daily_df=_stats_kw["daily_df"],
    horizon_count_daily=_stats_kw["horizon_count_daily"],
)
if _stats_kw["horizon_count_daily"]:
    _method_unit = "个交易日"
elif is_weekly:
    _method_unit = "周"
else:
    _method_unit = "根"


def _method_side_frame(side_key):
    side = _method_choice[side_key]
    rows = []
    for key, label in (
        ("divergence", "背离"),
        ("nine", "九转"),
        ("overlap", "九转配合已确认背离"),
    ):
        metrics = side[key]
        rows.append({
            "方法": label,
            "笔数": int(metrics["n"]),
            "曲线中位数": _fmt_edge_points(metrics["curve"]) or "—",
            "下界中位数": _fmt_edge_points(metrics["lower"]) or "—",
            "判定": tv.method_verdict(side, key),
        })
    return pd.DataFrame(rows)


st.subheader("方法选择")
st.markdown(f"**{tv.method_choice_summary(_method_choice)}**")
st.caption(
    f"曲线中位数和下界中位数都来自持有 {_span} {_method_unit}，不是某一天。"
    "曲线中位数是收缩超额胜率的中位数，单位是百分点。"
    "下界中位数是每一根超额胜率的 90% 下界再取中位数，大于 0 才过线。"
    "背离用当前图上保留的信号，开了趋势滤波就是滤波后的。九转不过滤。"
    "两侧都过线时，曲线中位数更高的一侧是主方法；相差不足 1 个点时，取笔数更多的一侧。"
    "过线表示下界中位数大于 0，但不是这一侧的主方法。"
    "共振不参与主方法竞选。只做多时，做空判定写成不采用。"
    "这是全样本内的比较，换一段行情主方法可能换。"
)
_method_col_l, _method_col_r = st.columns(2)
with _method_col_l:
    st.markdown("**买入**")
    st.dataframe(
        _method_side_frame("long"),
        width="stretch",
        hide_index=True,
        alt="买入方向的背离、九转和共振谁是主方法",
    )
with _method_col_r:
    st.markdown("**做空**")
    st.dataframe(
        _method_side_frame("short"),
        width="stretch",
        hide_index=True,
        alt="做空方向的背离、九转和共振谁是主方法",
    )

recent = pd.DataFrame(
    tv.signals_in_window(
        bull_plot, bear_plot, df, ticker, days=recent_window_days
    )
)


def _px(value):
    number = float(value)
    if abs(number) >= 100:
        return f"{number:,.2f}"
    return f"{number:,.4f}"


def _at(series, bar):
    value = series.loc[bar]
    if isinstance(value, pd.Series):
        return value.iloc[-1]
    return value


def _last_date_text():
    last_ts = pd.Timestamp(df.index[-1])
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert("UTC").tz_localize(None)
    return last_ts.date().isoformat()


def _ma_note_at(bar, when):
    if bar is None:
        return f"{when}未能对齐到 K 线，MA{ma_n} 位置未计算"
    ma_val = _at(df.Close.rolling(ma_n).mean(), bar)
    close_at = _at(df.Close, bar)
    if pd.isna(ma_val) or float(ma_val) == 0:
        return f"{when} MA{ma_n} 尚未形成"
    gap = abs(float(close_at) / float(ma_val) - 1) * 100
    side = "上方" if float(close_at) >= float(ma_val) else "下方"
    return f"{when}收盘在 MA{ma_n} {side} {gap:.2f}%（均线 {_px(ma_val)}）"


def _adx_note_at(bar, when):
    if not trend_on or adx_th is None:
        return "ADX 过滤未启用"
    if bar is None:
        return f"ADX 过滤已开启（阈值 {int(adx_th)}），{when}未能对齐"
    adx_val = _at(tv.adx(df), bar)
    if pd.isna(adx_val):
        return f"ADX 过滤已开启（阈值 {int(adx_th)}），{when}数值未算出"
    return f"{when} ADX(14) 为 {float(adx_val):.1f}，阈值 {int(adx_th)}"


def _posterior_text(signal_name):
    bullish = signal_name == "底背离"
    if trend_on:
        stats = stats_bull_f if bullish else stats_bear_f
    else:
        stats = stats_bull if bullish else stats_bear
    return dp.format_posterior(stats)


def _prompt_for_signal_row(row):
    """用当前页已经算好的行情和后验，填这条信号的研究 prompt。"""
    bar = dp.match_bar(df.index, row["date"])
    on_chop = False
    if trend_on and _trend_masks is not None and bar is not None:
        on_chop = bool(_at(_trend_masks["chop"], bar))
    if not trend_on:
        filter_status = "滤波关闭，图上保留全部背离"
    elif on_chop:
        filter_status = "震荡段保留（灰色震荡区，顶背离和底背离都保留）"
    elif _counter:
        filter_status = "逆势保留"
    else:
        filter_status = "顺势保留"
    nine_note = _nine_note_for_signal(row["signal"], row["date"])
    return dp.build_decision_prompt(
        ticker=ticker,
        timeframe=timeframe,
        recent_window_label=recent_window_label,
        signal=row["signal"],
        signal_date=row["date"],
        signal_close=row["close"],
        indicator_count=row["indicator_count"],
        last_date=_last_date_text(),
        last_close=float(last),
        filter_status=filter_status,
        ma_note=_ma_note_at(bar, "信号日"),
        adx_note=_adx_note_at(bar, "信号日"),
        posterior=_posterior_text(row["signal"]),
        nine_note=nine_note,
        method_note=_method_note_for_signal(row["signal"], not nine_note.startswith("无")),
    )


def _prompt_without_signal():
    """近窗口没有顶/底背离时，事实段照实写无信号。"""
    if not trend_on:
        filter_status = (
            f"滤波关闭。近 {recent_window_label} 没有底背离，也没有顶背离。"
        )
    elif _counter:
        filter_status = (
            f"当前规则是逆势保留。近 {recent_window_label} 在该规则下没有保留的信号。"
        )
    else:
        filter_status = (
            f"当前规则是顺势保留。近 {recent_window_label} 在该规则下没有保留的信号。"
        )
    bar = df.index[-1]
    posterior = (
        "底背离（历史）：\n"
        + _posterior_text("底背离")
        + "\n顶背离（历史）：\n"
        + _posterior_text("顶背离")
    )
    return dp.build_decision_prompt(
        ticker=ticker,
        timeframe=timeframe,
        recent_window_label=recent_window_label,
        last_date=_last_date_text(),
        last_close=float(last),
        filter_status=filter_status,
        ma_note=_ma_note_at(bar, "最新"),
        adx_note=_adx_note_at(bar, "最新"),
        posterior=posterior,
        nine_note=_nine_note_in_window(),
        method_note=tv.method_choice_summary(_method_choice),
    )


def _method_note_for_signal(signal_name, has_overlap):
    """这条背离是不是主方法；若同时是九转配合，接着写加仓还是只标记。"""
    side = "long" if signal_name == "底背离" else "short"
    text = tv.method_signal_note(_method_choice, side, "divergence")
    if has_overlap:
        text += tv.method_signal_note(_method_choice, side, "overlap")
    return text


def _nine_note_for_signal(signal_name, signal_date):
    pairs = _overlap_long_pairs if signal_name == "底背离" else _overlap_short_pairs
    note = tv.nine_overlap_note(pairs, signal_date, pivot_order)
    if note:
        return note
    return "无。这一条背离没有和九转第 7、8、9 根重合。"


def _nine_note_in_window():
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=recent_window_days)
    bits = []
    for pairs, side in ((_overlap_long_pairs, "买入"), (_overlap_short_pairs, "做空")):
        if pairs is None or len(pairs) == 0:
            continue
        entries = pd.DatetimeIndex(
            pd.to_datetime(pairs["entry"]).map(tv._naive_timestamp)
        ).normalize()
        hit = pairs.loc[entries >= cutoff]
        for _, one in hit.iterrows():
            entry = tv._naive_timestamp(one["entry"])
            bits.append(f"{entry.date().isoformat()} {side}")
    if not bits:
        return "无。近窗口没有九转第 7、8、9 根与已确认背离点重合。"
    listed = "、".join(dict.fromkeys(bits))
    return (
        "有。入场落在近窗口："
        f"{listed}。入场取较晚的一根收盘（九转第 9 根，或背离确认的最后一根）。"
        "这是对已确认背离的加强，不是另一条独立信号。"
    )


st.subheader(f"近 {recent_window_label} 信号")
if recent.empty:
    st.info(f"近 {recent_window_label} 无信号")
    _window_note = _nine_note_in_window()
    if not _window_note.startswith("无"):
        st.warning(_window_note)
else:
    recent = recent.copy()
    recent["配合"] = [
        tv.nine_overlap_note(
            _overlap_long_pairs if row.signal == "底背离" else _overlap_short_pairs,
            row.date,
            pivot_order,
        ) or "—"
        for row in recent.itertuples(index=False)
    ]
    recent["方法"] = [
        _method_note_for_signal(row.signal, row.配合 != "—")
        for row in recent.itertuples(index=False)
    ]
    st.caption("「方法」列写这条背离是不是该方向的主方法。若这一条同时是九转配合，后面接着写加仓还是只标记。")
    if recent["配合"].ne("—").any():
        st.warning("下面有信号同时是九转配合已确认背离。配合列写了入场日。大号星标在主图的入场那根上。")
    st.dataframe(recent, width="stretch", alt=f"近{recent_window_label}背离信号及主方法判定")

with st.container(border=True):
    st.subheader("决策 prompt")
    st.caption("生成一段可复制的研究 prompt，贴到 ChatGPT 或 Claude。本页不联网，也不调用模型。")
    choice_key = f"decision_signal_{ticker}"
    if recent.empty:
        st.caption(f"近 {recent_window_label} 无信号。生成时会在 prompt 里照实写明。")
        clicked = st.button(
            "生成决策 prompt",
            icon=":material/edit_note:",
            type="primary",
            key=f"gen_decision_prompt_{ticker}",
        )
        signal_id = "none"
        prompt_text = _prompt_without_signal
    else:
        ordered = recent.sort_values(["date", "signal"], ascending=[False, True])
        labels = []
        label_to_row = {}
        seen = {}
        for _, row in ordered.iterrows():
            label = f"{row['date']} · {row['signal']} · 共振 {int(row['indicator_count'])}"
            if tv.nine_overlap_note(
                _overlap_long_pairs if row["signal"] == "底背离" else _overlap_short_pairs,
                row["date"],
                pivot_order,
            ):
                label += " · 九转配合"
            if label in seen:
                seen[label] += 1
                label = f"{label} · 收盘 {_px(row['close'])}"
            else:
                seen[label] = 1
            labels.append(label)
            label_to_row[label] = row
        if len(labels) == 1:
            label = labels[0]
            st.caption(f"当前信号：{label}")
            clicked = st.button(
                "生成决策 prompt",
                icon=":material/edit_note:",
                type="primary",
                key=f"gen_decision_prompt_{ticker}",
            )
        else:
            if st.session_state.get(choice_key) not in labels:
                st.session_state[choice_key] = labels[0]
            pick, action = st.columns([3, 1], vertical_alignment="bottom")
            with pick:
                label = st.selectbox("用哪条信号", labels, key=choice_key)
            with action:
                clicked = st.button(
                    "生成决策 prompt",
                    icon=":material/edit_note:",
                    type="primary",
                    width="stretch",
                    key=f"gen_decision_prompt_{ticker}",
                )
        chosen = label_to_row[label]
        signal_id = (
            f"{chosen['date']}|{chosen['signal']}|"
            f"{int(chosen['indicator_count'])}|{chosen['close']}"
        )
        prompt_text = lambda chosen=chosen: _prompt_for_signal_row(chosen)
    if clicked:
        st.session_state["decision_prompt"] = {
            "ticker": ticker,
            "signal_id": signal_id,
            "text": prompt_text(),
        }
    saved = st.session_state.get("decision_prompt")
    if (
        isinstance(saved, dict)
        and saved.get("ticker") == ticker
        and saved.get("signal_id") == signal_id
    ):
        st.code(saved["text"], language=None, wrap_lines=True)

trend_note = " | 背景色说明见主图下方展开项" if trend_on else ""
st.caption(
    f"周期: {timeframe} | 数据: {df.index[0].date()} -> {df.index[-1].date()} | "
    f"底背离 {len(bull_sig)} 个（滤波后 {len(bull_f)}）| "
    f"顶背离 {len(bear_sig)} 个（滤波后 {len(bear_f)}）| "
    f"模块: {tv.__name__}.py" + trend_note
)
