"""
交互式复盘 Dashboard —— 复用 tradingview_realization_v2.py 的全部函数

启动方式（在本文件所在目录）：
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    streamlit run dashboard.py

左侧选标的/调参数，右侧联动：主图(K线+均线+布林+斐波那契+滚动VP+背离标注)
+ 信号后验统计表 + 后验收益分布 + 近7日信号。
"""
import json
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

# 选中这些标的时主图直接用这套，不跟侧边栏走。
# KBE、KRE、IWM、DVY：日线十年、四段中位数入选。KBE 的共振数是 4，3–6 的搜索里它赢过共振 3。
# USDCHF：外汇十年、四段累计收益入选。滤波关闭，均线和 ADX 不参与。
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
ASSET_PRESETS = {
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
# 一键优化选出更高的 10 日超额胜率后写入这里，覆盖同名的内置预设。
_USER_PRESETS_PATH = tv.CACHE_DIR / "fixed_presets.json"
_PRESET_FIELDS = (
    "pivot_order",
    "min_count",
    "dedup_bars",
    "trend_on",
    "aligned_keep",
    "adx_th",
    "ma_n",
    "summary",
)
_MERGED_PRESETS = dict(ASSET_PRESETS)


def _normalize_preset(spec):
    adx = spec.get("adx_th")
    if adx in ("", None):
        adx = None
    else:
        adx = int(adx)
    return {
        "pivot_order": int(spec["pivot_order"]),
        "min_count": int(spec["min_count"]),
        "dedup_bars": int(spec["dedup_bars"]),
        "trend_on": bool(spec["trend_on"]),
        "aligned_keep": bool(spec["aligned_keep"]),
        "adx_th": adx,
        "ma_n": int(spec["ma_n"]),
        "long_only": bool(spec.get("long_only", False)),
        "summary": str(spec.get("summary") or ""),
        "source": str(spec.get("source") or ""),
    }


def _load_user_presets():
    if not _USER_PRESETS_PATH.exists():
        return {}
    try:
        raw = json.loads(_USER_PRESETS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    loaded = {}
    for ticker, spec in raw.items():
        if not isinstance(ticker, str) or not isinstance(spec, dict):
            continue
        if any(field not in spec for field in _PRESET_FIELDS if field != "summary"):
            continue
        try:
            loaded[ticker] = _normalize_preset(spec)
        except (TypeError, ValueError):
            continue
    return loaded


def _reload_presets():
    global _MERGED_PRESETS
    merged = dict(ASSET_PRESETS)
    merged.update(_load_user_presets())
    _MERGED_PRESETS = merged


def _remember_fixed_preset(ticker, spec):
    """同一标的的 Yahoo 代码和缓存文件名写同一套，避免列表里两套参数。"""
    user = _load_user_presets()
    safe = tv.safe_chart_name(ticker)
    keys = {ticker}
    for key in list(_MERGED_PRESETS) + list(user):
        if key and tv.safe_chart_name(key) == safe:
            keys.add(key)
    stored = _normalize_preset(spec)
    for key in keys:
        user[key] = stored
    _USER_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USER_PRESETS_PATH.write_text(
        json.dumps(user, ensure_ascii=False, indent=2) + "\n"
    )
    _reload_presets()


def _preset_for(name):
    """Yahoo 代码和缓存文件名都对得上同一套固定参数。"""
    if not name:
        return None
    if name in _MERGED_PRESETS:
        return _MERGED_PRESETS[name]
    safe = tv.safe_chart_name(name)
    if safe in _MERGED_PRESETS:
        return _MERGED_PRESETS[safe]
    for key, spec in _MERGED_PRESETS.items():
        if tv.safe_chart_name(key) == safe:
            return spec
    return None


def _listed_preset_tickers():
    """每个固定参数只列一次，优先用 Yahoo 代码，不重复缓存文件名。"""
    names = []
    seen_safe = set()
    for key in _MERGED_PRESETS:
        safe = tv.safe_chart_name(key)
        if safe in seen_safe:
            continue
        seen_safe.add(safe)
        names.append(key)
    return names


def _preset_cache_key():
    return tuple(
        (
            ticker,
            spec["pivot_order"],
            spec["min_count"],
            spec["dedup_bars"],
            spec["trend_on"],
            spec["aligned_keep"],
            spec["adx_th"],
            spec["ma_n"],
            bool(spec.get("long_only", False)),
        )
        for ticker, spec in sorted(_MERGED_PRESETS.items())
    )


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


@st.cache_data(ttl="6h", max_entries=32, show_spinner=False)
def _cached_optimize(ticker, period, data_token):
    """在这段日线上最大化持有 10 日的收缩超额胜率。"""
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
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "n_bars": int(len(df)),
    }


def _show_optimize_result(ticker, result):
    if result.get("method") != "edge":
        return
    if result["status"] != "selected":
        st.warning(
            f"日线 {result['start']} → {result['end']}（{result['n_bars']} 根）。"
            "144 组里没有一组能算出持有 10 日的分数。",
            title="无法优化",
        )
        return

    preset = result["preset"]
    st.markdown(f"**{preset['summary']}**")
    adopted = "只做多，分数只按底背离" if preset.get("long_only") else "多空按笔数加权"
    st.caption(
        f"日线 {result['start']} → {result['end']}（{result['n_bars']} 根）。"
        f"采用：{adopted}。"
        "只做多时，图上和后验表仍会列出顶背离和做空收益。"
    )
    st.dataframe(
        pd.DataFrame([
            {
                "项目": "底背离",
                "笔数": result["long_n"],
                "分数": _fmt_edge_points(result["long_score"]) or "—",
            },
            {
                "项目": "顶背离",
                "笔数": result["short_n"],
                "分数": _fmt_edge_points(result["short_score"]) or "—",
            },
            {
                "项目": "采用",
                "笔数": result["n"],
                "分数": _fmt_edge_points(result["score"]) or "—",
            },
            {
                "项目": "默认参数",
                "笔数": "—",
                "分数": _fmt_edge_points(result["baseline"]) or "—",
            },
        ]),
        hide_index=True,
        alt="底背离、顶背离、采用分数和默认参数分数",
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
            f"在近 {window} 的日线上搜索 144 组，最大化持有 10 日的收缩超额胜率。"
            "每一组都比较多空一起和只做多，取得分更高的。"
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


@st.cache_data(show_spinner=False)
def compute_all(
    df, pivot_order, min_count, dedup_bars, vp_window, ma_n, adx_th,
    filter_mode="aligned",
    _filter_rev=tv.TREND_FILTER_LOGIC_REV,
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """信号 + 统计 + 滚动VP + 趋势滤波信号；参数变化时自动重算"""
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
    rvp = tv.rolling_volume_profile(dfi, window=vp_window)
    # 趋势滤波：顺势模式（价格在均线上方只留底背离，下方只留顶背离；
    # adx_th 启用后仅在趋势足够强时保留顺势信号）
    bull_f, bear_f = tv.apply_trend_filter(
        dfi, bull_sig, bear_sig, ma_n=ma_n, mode=filter_mode, adx_th=adx_th,
    )
    return dfi, bull_sig, bear_sig, rvp, bull_f, bear_f

# ---------------- 侧边栏 ----------------
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
timeframe = st.sidebar.selectbox("K线周期", ["日线", "周线"], index=0)
is_weekly = timeframe == "周线"
force_refresh = st.sidebar.checkbox("强制重新下载（绕过缓存）", value=False)
if is_weekly:
    show_bars = st.sidebar.slider("图表显示最近 N 根K线", 26, 260, 104, 13)
else:
    show_bars = st.sidebar.slider("图表显示最近 N 根K线", 100, 800, 400, 50)
bar_unit = "周" if is_weekly else "根"
if is_weekly:
    hold_horizon_mode = st.sidebar.selectbox(
        "后验持有期计数",
        ["按周线 K 线根数", "按日线 K 线根数"],
        help="背离信号仍按周线计算；选「日线」时 5/10/20 表示持有 5/10/20 个交易日，"
        "入场仍为周线 pivot 确认后的收盘对应交易日。",
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
        origin = f"近 {origin}、持有 10 日超额胜率选出"
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
vp_window = st.sidebar.slider("滚动VP窗口 VP_WINDOW", 30, 150, tv.VP_WINDOW, 10)
lookback = st.sidebar.number_input("斐波那契波段 LOOKBACK（0=整段）",
                                   min_value=0, max_value=1500,
                                   value=0 if tv.LOOKBACK is None else tv.LOOKBACK)
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
    """固定参数用自己的一套。其余用最近一次日线侧边栏。分数始终按日线持有 10 日。"""
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
    if params["long_only"]:
        bear = bear.iloc[0:0]
    return bull, bear


def _side_edge_10d(df, sig, short, entry_shift):
    stats = tv.signal_stats(
        df, sig, horizons=(10,), short=short, entry_shift=entry_shift,
    )
    base = tv.unconditional_stats(df, horizons=(10,), short=short)
    n = int(stats.iloc[0]["信号数"])
    win = float(stats.iloc[0]["胜率"])
    base_win = float(base.iloc[0]["无条件胜率"])
    return n, tv.edge_score(win, base_win, n)


@st.cache_data(ttl="15m", max_entries=8, show_spinner="按持有 10 日的超额胜率给标的打分")
def score_ticker_list(
    tickers_tuple, period, shared_key, preset_rev,
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """每个标的在自己的参数下，日线持有 10 日的收缩超额胜率。"""
    del preset_rev, _dedup_rev
    shared = dict(shared_key)
    scores = {}
    for name in tickers_tuple:
        try:
            params = _edge_params_for(name, shared)
            data = tv.load_ohlcv(name, period=period)
            bull, bear = _signals_for_edge(data, params)
            shift = params["pivot_order"]
            long_n, long_score = _side_edge_10d(data, bull, False, shift)
            if params["long_only"]:
                short_n, short_score = 0, np.nan
            else:
                short_n, short_score = _side_edge_10d(data, bear, True, shift)
            scores[name] = tv.combine_edge_scores(
                long_score, long_n, short_score, short_n, params["long_only"],
            )
        except Exception:
            scores[name] = np.nan
    return scores


def _fmt_edge_points(score):
    if score is None or not np.isfinite(score):
        return ""
    points = float(score) * 100
    if abs(points) < 0.05:
        return "0.0"
    return f"{points:+.1f}"


def _render_ticker_menu(slot, names, scores):
    ordered = sorted(
        names,
        key=lambda item: (
            not np.isfinite(scores.get(item, np.nan)),
            -(scores[item] if np.isfinite(scores.get(item, np.nan)) else 0.0),
            item,
        ),
    )
    current = st.session_state.ticker_choice
    current_score = _fmt_edge_points(scores.get(current, np.nan))
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
            help="按日线持有 10 日的分数从高到低。点某一行切换标的。右侧删除会从列表移除，并删除该标的的本地缓存。",
        ):
            st.caption(
                "分数是持有 10 日的超额胜率，按笔数收缩后的百分点。"
                "趋势滤波打开时用滤波后的信号。多空都做时两侧按笔数加权。只做多只算底背离。"
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
                points = _fmt_edge_points(scores.get(item, np.nan))
                title = item
                if spec:
                    title = f"{title} · 已固定"
                if points:
                    title = f"{title}  {points}"
                help_text = "日线持有 10 日的收缩超额胜率（百分点）。"
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
df, bull_sig, bear_sig, rvp, bull_f, bear_f = compute_all(
    df_raw, pivot_order, min_count, dedup_bars, vp_window, ma_n, adx_th,
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
if is_weekly and hold_horizon_daily:
    hold_caption = (
        f"持有 {', '.join(str(h) for h in tv.STATS_HORIZONS)} 个交易日（出场按日线收盘）"
    )
elif is_weekly:
    hold_caption = (
        f"持有 {', '.join(str(h) for h in tv.STATS_HORIZONS)} 周（周线 K 线根数）"
    )
else:
    hold_caption = (
        f"持有 {', '.join(str(h) for h in tv.STATS_HORIZONS)} 根日线 K 线"
    )
bull_plot, bear_plot = (bull_f, bear_f) if trend_on else (bull_sig, bear_sig)
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
        bull_plot, bear_plot, df, ticker, days=recent_window_days
    )
)
c5.metric(f"近{recent_window_label}信号", n_recent)

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
        _preset_cache_key(),
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
    _dedup_rev=tv.DEDUP_LOGIC_REV,
):
    """日 K 与周 K 同向背离，且信号日期相差 <= tolerance 个交易日。"""
    del _dedup_rev
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
fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                    column_widths=[0.85, 0.15], horizontal_spacing=0.01)
fig.add_trace(go.Candlestick(x=disp.index, open=disp.Open, high=disp.High,
    low=disp.Low, close=disp.Close, name=ticker,
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
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
fig.add_trace(go.Scatter(x=disp.index, y=rvp_disp.POC, name=f"滚动POC({vp_window}{bar_unit})",
    line=dict(color="#ff9800", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=list(disp.index)+list(disp.index)[::-1],
    y=list(rvp_disp.VAH)+list(rvp_disp.VAL)[::-1], fill="toself",
    fillcolor="rgba(38,166,154,0.10)", line=dict(color="rgba(0,0,0,0)"),
    name="滚动VA"), row=1, col=1)
for r, p in fib.items():
    fig.add_hline(y=p, line_dash="dash", line_color="#ff9800", line_width=1,
                  annotation_text=f"{r}", annotation_font_size=10, row=1, col=1)
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
b_all = bull_sig[bull_sig.index >= disp.index[0]]
r_all = bear_sig[bear_sig.index >= disp.index[0]]
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

# ADX 强度条：只在启用 ADX 过滤时显示，让"趋势是否够强"可见
if trend_on and adx_th is not None:
    adx_s = tv.adx(df).loc[disp.index]
    fig_adx = go.Figure()
    fig_adx.add_trace(go.Scatter(x=disp.index, y=adx_s, name="ADX(14)",
        line=dict(color="#ab47bc", width=1.2),
        fill="tozeroy", fillcolor="rgba(171,71,188,0.10)"))
    fig_adx.add_hline(y=adx_th, line_dash="dash", line_color="#ef5350", line_width=1)
    fig_adx.update_layout(
        height=150, margin=dict(l=0, r=0, t=25, b=0), showlegend=False,
        title=dict(text=f"ADX 趋势强度 — 虚线=阈值 {adx_th}，虚线上方=趋势够强（保留信号）",
                   font=dict(size=11)))
    if not is_weekly:
        fig_adx.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig_adx, width="stretch", alt="ADX 趋势强度")

# ---------------- 统计区 ----------------
st.subheader("信号后验统计（基于全量数据）")
st.caption(
    f"入场 = 信号 bar 后第 {pivot_order} 根{bar_unit}线收盘价（pivot 确认）；{hold_caption}"
    " 无条件胜率、无条件平均是对照：当前数据区间里，每个还能持满的交易日都按同一方向做一笔。"
    "底背离表按买入，顶背离表按做空。随机抽与信号数相同的时点，期望胜率和平均收益就等于这两列。"
    "对照样本是这些交易日的个数。用来看信号有没有超出这段行情本身的涨跌。"
    " 分数和 90%下界只填持有 10 日那一行，单位是百分点。"
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
    col_l, col_r = st.columns([1, 1.4])
    with col_l:
        if filt_note:
            st.caption(filt_note)
        st.markdown("**底背离（期望上涨）**")
        st.dataframe(
            _format_stats_table(stats_b), width="stretch",
            alt="底背离后验统计", key=f"{panel_key}_bull_stats",
        )
        st.markdown("**顶背离（期望下跌）**")
        st.dataframe(
            _format_stats_table(stats_be), width="stretch",
            alt="顶背离后验统计", key=f"{panel_key}_bear_stats",
        )
    with col_r:
        st.subheader("后验收益分布")
        fig2 = tv.posterior_returns_plotly(df, bull_s, bear_s, **_stats_kw)
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

left, right = st.columns([1, 1.4])
with left:
    st.subheader(f"近 {recent_window_label} 信号")
    recent = pd.DataFrame(
        tv.signals_in_window(
            bull_plot, bear_plot, df, ticker, days=recent_window_days
        )
    )
    if recent.empty:
        st.info(f"近 {recent_window_label} 无信号")
    else:
        st.dataframe(recent, width="stretch", alt=f"近{recent_window_label}背离信号")

with right:
    st.subheader("斐波那契关键位")
    fib_df = pd.DataFrame({"比例": list(fib.keys()), "价位": list(fib.values())})
    fib_df["距现价%"] = (fib_df["价位"] / last - 1) * 100
    fib_display = fib_df.copy()
    fib_display["价位"] = fib_display["价位"].map(lambda x: f"{x:,.2f}")
    fib_display["距现价%"] = fib_display["距现价%"].map(lambda x: f"{x:+.2f}%")
    st.dataframe(fib_display, width="stretch", alt="斐波那契关键位")

trend_note = " | 背景色说明见主图下方展开项" if trend_on else ""
st.caption(
    f"周期: {timeframe} | 数据: {df.index[0].date()} -> {df.index[-1].date()} | "
    f"底背离 {len(bull_sig)} 个（滤波后 {len(bull_f)}）| "
    f"顶背离 {len(bear_sig)} 个（滤波后 {len(bear_f)}）| "
    f"模块: {tv.__name__}.py" + trend_note
)
