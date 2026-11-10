"""标的固定参数与观察列表。Dashboard 和每日更新共用，避免两套规则。"""
import json

import tradingview_realization_v2 as tv

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

USER_PRESETS_PATH = tv.CACHE_DIR / "fixed_presets.json"
HIDDEN_TICKERS_PATH = tv.CACHE_DIR / "hidden_tickers.txt"
EXTRA_TICKERS_PATH = tv.CACHE_DIR / "extra_tickers.txt"
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


def normalize_preset(spec):
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


def load_user_presets():
    if not USER_PRESETS_PATH.exists():
        return {}
    try:
        raw = json.loads(USER_PRESETS_PATH.read_text())
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
            loaded[ticker] = normalize_preset(spec)
        except (TypeError, ValueError):
            continue
    return loaded


def reload_presets():
    global _MERGED_PRESETS
    merged = dict(ASSET_PRESETS)
    merged.update(load_user_presets())
    _MERGED_PRESETS = merged


def remember_fixed_preset(ticker, spec):
    """同一标的的 Yahoo 代码和缓存文件名写同一套，避免列表里两套参数。"""
    user = load_user_presets()
    safe = tv.safe_chart_name(ticker)
    keys = {ticker}
    for key in list(_MERGED_PRESETS) + list(user):
        if key and tv.safe_chart_name(key) == safe:
            keys.add(key)
    stored = normalize_preset(spec)
    for key in keys:
        user[key] = stored
    USER_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_PRESETS_PATH.write_text(
        json.dumps(user, ensure_ascii=False, indent=2) + "\n"
    )
    reload_presets()


def preset_for(name):
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


def listed_preset_tickers():
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


def preset_cache_key():
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


def _lines(path):
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def cache_ticker_names():
    if not tv.CACHE_DIR.exists():
        return []
    names = []
    for path in tv.CACHE_DIR.glob("*.csv"):
        stem = path.name.rsplit("_", 1)[0]
        if stem.startswith("dual_tf_backtest"):
            continue
        names.append(stem)
    return names


def watchlist():
    """和 Dashboard 侧边栏同一份列表：预设、默认标的、缓存、手动添加，去掉隐藏的。"""
    reload_presets()
    hidden = set(_lines(HIDDEN_TICKERS_PATH))
    preset_names = listed_preset_tickers()
    preset_aliases = {tv.safe_chart_name(name) for name in preset_names}
    names = []
    seen = set()
    for item in preset_names + list(tv.TICKERS) + cache_ticker_names() + _lines(EXTRA_TICKERS_PATH):
        if not item or item in hidden or item in seen:
            continue
        if item in preset_aliases and item not in preset_names:
            continue
        seen.add(item)
        names.append(item)
    return names


reload_presets()
