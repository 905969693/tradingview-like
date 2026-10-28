"""
逐资产参数搜索：每个标的单独选参，2025 年起只做成绩单。

搜索（144 组）：
- PIVOT_ORDER ∈ {3,4,5,6}，入场滞后锁死为同一个值
- MIN_COUNT ∈ {3,4,5,6}
- 趋势模式 off / aligned / counter
- 模式不是 off 时：ADX ∈ {关, 25} × MA ∈ {100, 200}
- DEDUP_BARS 固定 10；成本与 backtest_divergence.py 一致（默认 0）

信号在全历史上因果计算（pivot 右侧确认后才入场）。一笔成交属于它的入场日
所在窗口；同向不重叠沿整段行情生效，因此前一窗口尚未平仓的单会挡住后一窗口。

数据为 10 年日线。标的：DBA、IWM、LEMB、DVY、ASHR。

窗口（按入场日）：
- 验证：2017–2018、2019–2020、2021–2022、2023–2024
- 成绩单：2025-01-01 ~ 缓存末日（不参与选参）

入选看每笔交易的收益率中位数。多空成交放在一起，偶数笔时中位数是中间两笔的平均值。
复利合并收益 comb_* 仍写入，只供对照，不参与入选。
胜率 win_* = 该窗口里收益 >= 0 的成交占比，收益为 0 算胜。没有成交时留空。胜率不参与入选。

- 四段验证的不重叠成交都 ≥ 4，且四段中位数都 > 0
- 在合格格子里取四段中位数的最小值最大者
- 平手时取合格邻格更多者（只改 pivot±1 或 min_count±1，其余参数相同；
  对角格子不算邻格）
- 仍平手时依次比较四段中位数之和、更小的 pivot、更小的 min_count、
  模式 off < aligned < counter、ADX 关优先、更短的 MA，只为结果可重复
- 没有合格格子则 status=no_edge，不挑“最不差”的参数

对照参数：pivot=3, min_count=3, aligned, MA=200, ADX 关。

median_* / comb_* 都是收益比例，0.01 表示 +1%。

输出（backtest/asset_opt_10y_median_try_<日期>/）：
- grid.csv       每个标的 × 144 组的交易数、10 日中位数（入选用）、胜率和 10 日复利（对照）
- selected.csv   每个标的一行：入选参数、四段中位数与胜率、2025 起 5/10/20 日中位数与胜率、
                 以及相对对照参数的差值

运行：python backtest_asset_opt.py
"""
from datetime import date

import numpy as np
import pandas as pd

import backtest_divergence as bt
import tradingview_realization_v2 as tv

# ============ 0. 搜索空间与窗口 ============
PERIOD = "10y"
TICKERS = ["DBA", "IWM", "LEMB", "DVY", "ASHR"]
PIVOTS = (3, 4, 5, 6)
MIN_COUNTS = (3, 4, 5, 6)
# 邻格按 2–6 的整数步长定义；不在本次网格里的格子不会入选。
NEIGHBOR_MIN_COUNTS = (2, 3, 4, 5, 6)
MODES = ("off", "aligned", "counter")
ADX_CHOICES = (None, 25)
MA_CHOICES = (100, 200)
DEDUP_BARS = tv.DEDUP_BARS
PRIMARY_H = 10
REPORT_HORIZONS = (5, 20)
MIN_TRADES = 4
VAL_WINDOWS = (
    ("2017_2018", pd.Timestamp("2017-01-01"), pd.Timestamp("2018-12-31")),
    ("2019_2020", pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31")),
    ("2021_2022", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
    ("2023_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
)
TEST_START = pd.Timestamp("2025-01-01")
WINDOWS = VAL_WINDOWS + (("2025", TEST_START, None),)
DEFAULT = {
    "mode": "aligned",
    "adx_th": None,
    "ma_n": 200,
    "PIVOT_ORDER": 3,
    "MIN_COUNT": 3,
}
MODE_ORDER = {"off": 0, "aligned": 1, "counter": 2}
OUT_DIR = tv.SCRIPT_DIR / "backtest" / f"asset_opt_10y_median_try_{date.today().isoformat()}"


def iter_params():
    """off 不带 ADX / MA，避免同一组信号重复计数。"""
    for pivot in PIVOTS:
        for min_count in MIN_COUNTS:
            yield ("off", None, None, pivot, min_count)
            for mode in ("aligned", "counter"):
                for adx_th in ADX_CHOICES:
                    for ma_n in MA_CHOICES:
                        yield (mode, adx_th, ma_n, pivot, min_count)


def param_key(row):
    return (
        row["mode"],
        row["adx_th"],
        row["ma_n"],
        row["PIVOT_ORDER"],
        row["MIN_COUNT"],
    )


def neighbor_keys(row):
    """pivot±1 或 min_count±1，其余参数不变。对角格子不算。"""
    mode, adx_th, ma_n, pivot, min_count = param_key(row)
    keys = []
    for step in (-1, 1):
        if pivot + step in PIVOTS:
            keys.append((mode, adx_th, ma_n, pivot + step, min_count))
        if min_count + step in NEIGHBOR_MIN_COUNTS:
            keys.append((mode, adx_th, ma_n, pivot, min_count + step))
    return keys


def divergence_counts(df, pivot_order):
    """每个指标的背离次数。只随 pivot 变化，供后面的共振阈值复用。"""
    bull_s = pd.Series(0, index=df.index, dtype=int)
    bear_s = pd.Series(0, index=df.index, dtype=int)
    for name in tv.INDICATORS:
        bull_idx, bear_idx = tv.find_divergences(df, df[name], order=pivot_order)
        bull_s.iloc[bull_idx] += 1
        bear_s.iloc[bear_idx] += 1
    return bull_s, bear_s


def signals_from_counts(df, bull_s, bear_s, min_count):
    bull = tv.dedup_signals(bull_s[bull_s >= min_count], df, gap=DEDUP_BARS)
    bear = tv.dedup_signals(bear_s[bear_s >= min_count], df, gap=DEDUP_BARS)
    return bull, bear


def apply_mode(df, bull, bear, mode, ma_n, adx_th):
    if mode == "off":
        return bull, bear
    return tv.apply_trend_filter(
        df, bull, bear, ma_n=ma_n, mode=mode, adx_th=adx_th,
    )


def executed_returns(df, sig, side, horizon, entry_shift, cost_bps=bt.COST_BPS):
    """与 equity_curve 相同的同向不重叠规则，并记下入场日以便划入窗口。"""
    if len(sig) == 0:
        return []
    closes = df.Close.values
    cost = 2 * cost_bps / 1e4
    busy_until = -1
    trades = []
    for idx in sig.index:
        xi = df.index.get_loc(idx)
        ei = xi + entry_shift
        xj = ei + horizon
        if ei >= len(df) or xj >= len(df) or ei < busy_until:
            continue
        ret = closes[xj] / closes[ei] - 1
        if side == "short":
            ret = -ret
        trades.append((pd.Timestamp(df.index[ei]), ret - cost))
        busy_until = xj + 1
    return trades


def _compound(trades):
    equity = 1.0
    for _, ret in trades:
        equity *= 1 + ret
    return equity - 1


def _assert_matches_equity_curve(df, sig, side, horizon, entry_shift):
    """确认拆出来的成交与 equity_curve 的最终净值一致。"""
    trades = executed_returns(df, sig, side, horizon, entry_shift)
    got = _compound(trades)
    curve = bt.equity_curve(
        df, sig, side, horizon, entry_shift=entry_shift, cost_bps=bt.COST_BPS,
    )
    expect = 0.0 if curve.empty else float(curve.iloc[-1] - 1)
    if abs(got - expect) > 1e-9:
        raise AssertionError(
            f"equity_curve mismatch side={side} h={horizon}: {got} vs {expect}"
        )


def _window_compound(trades, start, end):
    equity = 1.0
    n = 0
    for ts, ret in trades:
        if ts < start or (end is not None and ts > end):
            continue
        equity *= 1 + ret
        n += 1
    return n, equity - 1.0


def window_pair(long_trades, short_trades, start, end):
    """多空各半仓。某一边没有成交时，该边收益记 0，与 equity_curve 空曲线一致。"""
    n_long, ret_long = _window_compound(long_trades, start, end)
    n_short, ret_short = _window_compound(short_trades, start, end)
    return n_long + n_short, (ret_long + ret_short) / 2.0


def _window_returns(trades, start, end):
    return [
        ret for ts, ret in trades
        if not (ts < start or (end is not None and ts > end))
    ]


def window_stats(long_trades, short_trades, start, end, book="both"):
    """笔数、累计收益、单笔收益中位数、胜率。收益 >= 0 为胜。

    book=both 时累计收益是多空各半仓的平均。book=long 时只数底背离做多，
    累计收益是多头自己的复利，不把空头记成 0 再除以 2。
    """
    if book == "long":
        rets = _window_returns(long_trades, start, end)
        if not rets:
            return 0, 0.0, 0.0, np.nan
        equity = 1.0
        for ret in rets:
            equity *= 1 + ret
        win_rate = float(np.mean([ret >= 0 for ret in rets]))
        return len(rets), equity - 1.0, float(np.median(rets)), win_rate
    n_trades, comb = window_pair(long_trades, short_trades, start, end)
    rets = _window_returns(long_trades, start, end) + _window_returns(short_trades, start, end)
    if not rets:
        return n_trades, comb, 0.0, np.nan
    median = float(np.median(rets))
    win_rate = float(np.mean([ret >= 0 for ret in rets]))
    return n_trades, comb, median, win_rate


def evaluate_ticker(df, book="both"):
    counts = {pivot: divergence_counts(df, pivot) for pivot in PIVOTS}
    check_bull, check_bear = signals_from_counts(df, *counts[PIVOTS[0]], MIN_COUNTS[0])
    for side, sig in (("long", check_bull), ("short", check_bear)):
        _assert_matches_equity_curve(df, sig, side, PRIMARY_H, PIVOTS[0])

    cached_signals = {}
    for pivot in PIVOTS:
        bull_s, bear_s = counts[pivot]
        for min_count in MIN_COUNTS:
            cached_signals[(pivot, min_count)] = signals_from_counts(
                df, bull_s, bear_s, min_count,
            )

    rows = []
    for mode, adx_th, ma_n, pivot, min_count in iter_params():
        bull, bear = cached_signals[(pivot, min_count)]
        bull, bear = apply_mode(df, bull, bear, mode, ma_n, adx_th)
        by_horizon = {}
        for horizon in tv.STATS_HORIZONS:
            long_trades = executed_returns(df, bull, "long", horizon, pivot)
            short_trades = executed_returns(df, bear, "short", horizon, pivot)
            by_horizon[horizon] = {
                name: window_stats(long_trades, short_trades, start, end, book=book)
                for name, start, end in WINDOWS
            }
        primary = by_horizon[PRIMARY_H]
        row = {
            "mode": mode,
            "adx_th": adx_th,
            "ma_n": ma_n,
            "PIVOT_ORDER": pivot,
            "MIN_COUNT": min_count,
        }
        for name, _start, _end in WINDOWS:
            n_trades, comb, median, win_rate = primary[name]
            row[f"n_{name}"] = n_trades
            row[f"comb_{name}"] = comb
            row[f"median_{name}"] = median
            row[f"win_{name}"] = win_rate
        for horizon in REPORT_HORIZONS:
            _n, comb, median, win_rate = by_horizon[horizon]["2025"]
            row[f"comb_2025_h{horizon}"] = comb
            row[f"median_2025_h{horizon}"] = median
            row[f"win_2025_h{horizon}"] = win_rate
        rows.append(row)
    return rows


def _val_medians(row):
    return [row[f"median_{name}"] for name, _start, _end in VAL_WINDOWS]


def _val_combs(row):
    return [row[f"comb_{name}"] for name, _start, _end in VAL_WINDOWS]


def _val_counts(row):
    return [row[f"n_{name}"] for name, _start, _end in VAL_WINDOWS]


def annotate(rows):
    for row in rows:
        counts = _val_counts(row)
        medians = _val_medians(row)
        row["qualified"] = int(
            all(n >= MIN_TRADES for n in counts) and all(median > 0 for median in medians)
        )
        row["score"] = min(medians)
    qualified_keys = {param_key(row) for row in rows if row["qualified"]}
    for row in rows:
        row["n_neighbors"] = sum(
            key in qualified_keys for key in neighbor_keys(row)
        )


def _rank_key(row):
    adx_rank = 0 if row["adx_th"] is None else int(row["adx_th"])
    ma_rank = -1 if row["ma_n"] is None else int(row["ma_n"])
    return (
        -row["score"],
        -row["n_neighbors"],
        -sum(_val_medians(row)),
        row["PIVOT_ORDER"],
        row["MIN_COUNT"],
        MODE_ORDER[row["mode"]],
        adx_rank,
        ma_rank,
    )


def pick(rows):
    """返回合格格子中的平台中心；没有合格格子时返回 None。"""
    qualified = [row for row in rows if row["qualified"]]
    if not qualified:
        return None
    return min(qualified, key=_rank_key)


def _blank_window_fields():
    fields = {}
    for name, _start, _end in VAL_WINDOWS:
        fields[f"n_{name}"] = ""
        fields[f"median_{name}"] = ""
        fields[f"win_{name}"] = ""
        fields[f"comb_{name}"] = ""
    return fields


def _blank_params():
    return {
        "mode": "",
        "adx_th": "",
        "ma_n": "",
        "PIVOT_ORDER": "",
        "MIN_COUNT": "",
        **_blank_window_fields(),
        "score": "",
        "n_neighbors": "",
        "n_2025": "",
        "median_2025": "",
        "win_2025": "",
        "comb_2025": "",
        "median_2025_h5": "",
        "win_2025_h5": "",
        "median_2025_h20": "",
        "win_2025_h20": "",
        "comb_2025_h5": "",
        "comb_2025_h20": "",
        "delta_score": "",
        "delta_median_2025": "",
    }


def _public_adx(mode, adx_th):
    if mode == "off":
        return ""
    if adx_th is None or (isinstance(adx_th, float) and np.isnan(adx_th)):
        return "off"
    return int(adx_th)


def _public_ma(mode, ma_n):
    if mode == "off" or ma_n is None or (isinstance(ma_n, float) and np.isnan(ma_n)):
        return ""
    return int(ma_n)


def _r6(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return round(float(value), 6)


def _window_metrics(row, prefix=""):
    fields = {}
    for name, _start, _end in VAL_WINDOWS:
        fields[f"{prefix}n_{name}"] = row[f"n_{name}"]
        fields[f"{prefix}median_{name}"] = _r6(row[f"median_{name}"])
        fields[f"{prefix}win_{name}"] = _r6(row[f"win_{name}"])
        fields[f"{prefix}comb_{name}"] = _r6(row[f"comb_{name}"])
    return fields


def grid_record(ticker, row):
    return {
        "ticker": ticker,
        "mode": row["mode"],
        "adx_th": _public_adx(row["mode"], row["adx_th"]),
        "ma_n": _public_ma(row["mode"], row["ma_n"]),
        "PIVOT_ORDER": row["PIVOT_ORDER"],
        "MIN_COUNT": row["MIN_COUNT"],
        **_window_metrics(row),
        "n_2025": row["n_2025"],
        "median_2025": _r6(row["median_2025"]),
        "win_2025": _r6(row["win_2025"]),
        "comb_2025": _r6(row["comb_2025"]),
        "qualified": row["qualified"],
        "score": _r6(row["score"]),
        "n_neighbors": row["n_neighbors"],
    }


def _default_fields(default_row):
    return {
        "default_mode": default_row["mode"],
        "default_adx_th": _public_adx(default_row["mode"], default_row["adx_th"]),
        "default_ma_n": _public_ma(default_row["mode"], default_row["ma_n"]),
        "default_PIVOT_ORDER": default_row["PIVOT_ORDER"],
        "default_MIN_COUNT": default_row["MIN_COUNT"],
        **_window_metrics(default_row, prefix="default_"),
        "default_qualified": default_row["qualified"],
        "default_n_2025": default_row["n_2025"],
        "default_median_2025": _r6(default_row["median_2025"]),
        "default_win_2025": _r6(default_row["win_2025"]),
        "default_comb_2025": _r6(default_row["comb_2025"]),
        "default_median_2025_h5": _r6(default_row["median_2025_h5"]),
        "default_win_2025_h5": _r6(default_row["win_2025_h5"]),
        "default_median_2025_h20": _r6(default_row["median_2025_h20"]),
        "default_win_2025_h20": _r6(default_row["win_2025_h20"]),
    }


def selected_record(ticker, rows, chosen):
    default_row = next(row for row in rows if param_key(row) == param_key(DEFAULT))
    record = {"ticker": ticker, "status": "no_edge" if chosen is None else "selected"}
    if chosen is None:
        record.update(_blank_params())
    else:
        default_score = min(_val_medians(default_row))
        record.update({
            "mode": chosen["mode"],
            "adx_th": _public_adx(chosen["mode"], chosen["adx_th"]),
            "ma_n": _public_ma(chosen["mode"], chosen["ma_n"]),
            "PIVOT_ORDER": chosen["PIVOT_ORDER"],
            "MIN_COUNT": chosen["MIN_COUNT"],
            **_window_metrics(chosen),
            "score": _r6(chosen["score"]),
            "n_neighbors": chosen["n_neighbors"],
            "n_2025": chosen["n_2025"],
            "median_2025": _r6(chosen["median_2025"]),
            "win_2025": _r6(chosen["win_2025"]),
            "comb_2025": _r6(chosen["comb_2025"]),
            "median_2025_h5": _r6(chosen["median_2025_h5"]),
            "win_2025_h5": _r6(chosen["win_2025_h5"]),
            "median_2025_h20": _r6(chosen["median_2025_h20"]),
            "win_2025_h20": _r6(chosen["win_2025_h20"]),
            "delta_score": _r6(chosen["score"] - default_score),
            "delta_median_2025": _r6(chosen["median_2025"] - default_row["median_2025"]),
        })
    record.update(_default_fields(default_row))
    return record


def _self_check():
    _check_forward_gate()
    params = list(iter_params())
    n_filter = 1 + len(("aligned", "counter")) * len(ADX_CHOICES) * len(MA_CHOICES)
    expected = len(PIVOTS) * len(MIN_COUNTS) * n_filter
    if len(params) != expected:
        raise AssertionError(f"expected {expected} combos, got {len(params)}")
    if len(params) != len(set(params)):
        raise AssertionError("duplicate parameter combo")
    default_key = param_key(DEFAULT)
    if default_key not in {(m, a, ma, p, c) for m, a, ma, p, c in params}:
        raise AssertionError("default combo missing from grid")

    def row(pivot, min_count, comb_a, comb_b, n_a=4, n_b=4, mode="aligned",
            median_a=None, median_b=None):
        """前三段用 comb_a，最后一段用 comb_b。中位数默认与累计相同。"""
        if median_a is None:
            median_a = comb_a
        if median_b is None:
            median_b = comb_b
        record = {
            "mode": mode,
            "adx_th": None,
            "ma_n": 200,
            "PIVOT_ORDER": pivot,
            "MIN_COUNT": min_count,
        }
        names = [name for name, _start, _end in VAL_WINDOWS]
        for name in names[:-1]:
            record[f"n_{name}"] = n_a
            record[f"comb_{name}"] = comb_a
            record[f"median_{name}"] = median_a
        record[f"n_{names[-1]}"] = n_b
        record[f"comb_{names[-1]}"] = comb_b
        record[f"median_{names[-1]}"] = median_b
        return record

    huge = [
        (pd.Timestamp("2017-03-01"), 0.01),
        (pd.Timestamp("2017-06-01"), 0.02),
        (pd.Timestamp("2017-09-01"), 0.90),
    ]
    _n, comb, median, win_rate = window_stats(huge, [], *VAL_WINDOWS[0][1:])
    if abs(median - 0.02) > 1e-12:
        raise AssertionError(f"median should ignore the outlier, got {median}")
    if comb <= median:
        raise AssertionError(f"compound {comb} should exceed median {median}")
    if abs(win_rate - 1.0) > 1e-12:
        raise AssertionError(f"all positive trades should win, got {win_rate}")
    flat = [
        (pd.Timestamp("2017-03-01"), 0.01),
        (pd.Timestamp("2017-06-01"), 0.0),
        (pd.Timestamp("2017-09-01"), -0.02),
    ]
    _n, _comb, _median, win_rate = window_stats(flat, [], *VAL_WINDOWS[0][1:])
    if abs(win_rate - 2 / 3) > 1e-12:
        raise AssertionError(f"a zero return counts as a win, got {win_rate}")

    higher_score = [
        row(4, 4, 0.20, 0.20),
        row(5, 4, 0.05, 0.05),
        row(6, 4, 0.05, 0.04),
    ]
    annotate(higher_score)
    if param_key(pick(higher_score)) != param_key(row(4, 4, 0, 0)):
        raise AssertionError("higher min across folds should beat a denser but weaker plateau")

    plateau = [
        row(4, 4, 0.05, 0.05),
        row(6, 4, 0.05, 0.05),
        row(5, 4, 0.01, 0.01),
        row(4, 5, 0.01, 0.01),
        row(4, 3, 0.01, 0.01),
    ]
    annotate(plateau)
    chosen = pick(plateau)
    if (chosen["PIVOT_ORDER"], chosen["MIN_COUNT"]) != (4, 4):
        raise AssertionError("tie on score should prefer the cell with more qualified neighbors")
    if chosen["n_neighbors"] != 3:
        raise AssertionError(f"expected 3 neighbors, got {chosen['n_neighbors']}")

    diagonal = [row(4, 4, 0.02, 0.02), row(5, 5, 0.01, 0.01)]
    annotate(diagonal)
    if any(item["n_neighbors"] != 0 for item in diagonal):
        raise AssertionError("diagonal cells must not count as neighbors")

    def rejected(sample, message):
        annotate(sample)
        if pick(sample) is not None:
            raise AssertionError(message)

    rejected([row(4, 4, 0.10, -0.01)], "a negative window must disqualify the cell")
    rejected([row(4, 4, 0.10, 0.0)], "a zero window is not strictly positive")
    rejected(
        [row(4, 4, 0.05, 0.04, median_a=-0.02, median_b=-0.01)],
        "a negative median must disqualify the cell even if cumulative return is positive",
    )
    kept = [row(4, 4, -0.02, -0.01, median_a=0.05, median_b=0.04)]
    annotate(kept)
    if pick(kept) is None:
        raise AssertionError("a positive median qualifies even if cumulative return is negative")
    rejected([row(4, 4, 0.10, 0.10, n_a=3)], "fewer than 4 trades must disqualify the cell")
    rejected(
        [row(4, 4, -0.02, -0.01, mode="off")],
        "no qualified cell should return None",
    )


FORWARD_WIN_MIN = 0.5
FORWARD_COMB_MIN = 0.0


def _finite(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def passes_forward_gate(row):
    """2025 至今：胜率 >= 50%，且多空各半仓的累计收益 >= 0。没有成交则不过。"""
    win = _finite(row.get("win_2025"))
    comb = _finite(row.get("comb_2025"))
    if win is None or comb is None:
        return False
    return win >= FORWARD_WIN_MIN and comb >= FORWARD_COMB_MIN


def preset_summary(mode, adx_th, ma_n, pivot, min_count, dedup_bars=DEDUP_BARS, long_only=False):
    if mode == "off":
        head = "滤波关"
    else:
        side = "顺势" if mode == "aligned" else "逆势"
        adx = "ADX 关" if adx_th is None else f"ADX>{int(adx_th)}"
        head = f"{side}、{adx}、MA{int(ma_n)}"
    text = f"{head}、pivot {int(pivot)}、共振 {int(min_count)}、去重 {int(dedup_bars)}"
    if long_only:
        text += "、只做多"
    return text


def preset_from_row(row, long_only=False):
    """把入选格子写成 Dashboard 固定参数。滤波关闭时均线不参与，仍记下 200 与现有预设一致。"""
    mode = row["mode"]
    trend_on = mode != "off"
    adx_th = row["adx_th"]
    ma_n = row["ma_n"]
    if not trend_on or adx_th is None:
        adx_th = None
    else:
        adx_th = int(adx_th)
    if not trend_on or ma_n is None:
        ma_n = 200
    else:
        ma_n = int(ma_n)
    pivot = int(row["PIVOT_ORDER"])
    min_count = int(row["MIN_COUNT"])
    return {
        "pivot_order": pivot,
        "min_count": min_count,
        "dedup_bars": int(DEDUP_BARS),
        "trend_on": trend_on,
        "aligned_keep": mode != "counter",
        "adx_th": adx_th,
        "ma_n": ma_n,
        "long_only": bool(long_only),
        "summary": preset_summary(
            mode, adx_th, ma_n, pivot, min_count, long_only=long_only,
        ),
    }


def _signal_win_rate(df, sig, pivot, short, horizon=PRIMARY_H):
    """与 signal_stats 相同：持有 horizon 日，收益 > 0 为胜。返回 (信号数, 胜率)。"""
    closes = df.Close.to_numpy(dtype=float)
    if len(sig) == 0:
        return 0, np.nan
    pos = df.index.get_indexer(sig.index)
    entry = pos + int(pivot)
    exit_i = entry + int(horizon)
    valid = (pos >= 0) & (exit_i < len(closes))
    rets = closes[exit_i[valid]] / closes[entry[valid]] - 1.0
    if short:
        rets = -rets
    if rets.size == 0:
        return 0, np.nan
    return int(np.isfinite(rets).sum()), float((rets > 0).mean())


def search_best_edge(df):
    """在 144 组里最大化持有 10 日的收缩超额胜率。

    每一组都比较「多空一起」和「只做多」，取得分更高的那一边。
    不要求四段验证合格。没有可用分数时返回 None。
    """
    if "RSI" not in df.columns:
        df = tv.add_indicators(df)
    base_long = float(tv.unconditional_stats(df, horizons=(PRIMARY_H,), short=False).iloc[0]["无条件胜率"])
    base_short = float(tv.unconditional_stats(df, horizons=(PRIMARY_H,), short=True).iloc[0]["无条件胜率"])
    counts = {pivot: divergence_counts(df, pivot) for pivot in PIVOTS}
    cached_signals = {}
    for pivot in PIVOTS:
        bull_s, bear_s = counts[pivot]
        for min_count in MIN_COUNTS:
            cached_signals[(pivot, min_count)] = signals_from_counts(
                df, bull_s, bear_s, min_count,
            )

    best = None
    best_key = None
    baseline = np.nan
    for mode, adx_th, ma_n, pivot, min_count in iter_params():
        bull, bear = cached_signals[(pivot, min_count)]
        bull, bear = apply_mode(df, bull, bear, mode, ma_n, adx_th)
        long_n, long_win = _signal_win_rate(df, bull, pivot, short=False)
        short_n, short_win = _signal_win_rate(df, bear, pivot, short=True)
        long_score = tv.edge_score(long_win, base_long, long_n)
        short_score = tv.edge_score(short_win, base_short, short_n)
        both_score = tv.combine_edge_scores(
            long_score, long_n, short_score, short_n, False,
        )
        if (
            mode == DEFAULT["mode"]
            and adx_th == DEFAULT["adx_th"]
            and ma_n == DEFAULT["ma_n"]
            and pivot == DEFAULT["PIVOT_ORDER"]
            and min_count == DEFAULT["MIN_COUNT"]
        ):
            baseline = both_score
        candidates = (
            (False, both_score, long_n + short_n),
            (True, tv.combine_edge_scores(long_score, long_n, short_score, short_n, True), long_n),
        )
        for long_only, score, n_used in candidates:
            if not np.isfinite(score):
                continue
            rank = (
                -float(score),
                -int(n_used),
                1 if long_only else 0,
                int(pivot),
                int(min_count),
                MODE_ORDER[mode],
                0 if adx_th is None else int(adx_th),
                10**9 if ma_n is None else int(ma_n),
            )
            if best_key is None or rank < best_key:
                best_key = rank
                best = {
                    "mode": mode,
                    "adx_th": adx_th,
                    "ma_n": ma_n,
                    "PIVOT_ORDER": pivot,
                    "MIN_COUNT": min_count,
                    "long_only": long_only,
                    "score": float(score),
                    "baseline": float(baseline) if np.isfinite(baseline) else np.nan,
                    "n": int(n_used),
                    "long_n": int(long_n),
                    "short_n": int(short_n),
                    "long_score": float(long_score) if np.isfinite(long_score) else np.nan,
                    "short_score": float(short_score) if np.isfinite(short_score) else np.nan,
                }
    if best is not None:
        best["baseline"] = float(baseline) if np.isfinite(baseline) else np.nan
    return best


def _chosen_fields(chosen, book):
    return {
        "book": book,
        "passes_forward": passes_forward_gate(chosen),
        "preset": preset_from_row(chosen, long_only=(book == "long")),
        "score": float(chosen["score"]),
        "n_neighbors": int(chosen["n_neighbors"]),
        "windows": [
            {
                "name": name,
                "n": int(chosen[f"n_{name}"]),
                "median": float(chosen[f"median_{name}"]),
                "win": _finite(chosen[f"win_{name}"]),
                "comb": float(chosen[f"comb_{name}"]),
            }
            for name, _start, _end in VAL_WINDOWS
        ],
        "forward": {
            "n": int(chosen["n_2025"]),
            "median": _finite(chosen["median_2025"]),
            "win": _finite(chosen["win_2025"]),
            "comb": _finite(chosen["comb_2025"]),
        },
    }


def optimize_ticker(ticker):
    """单标的搜索。多空一起没有合格格子时，再只按底背离做多选一次。

    两次都没有合格格子则 status=no_edge，不挑最不差的参数。
    """
    df = tv.add_indicators(tv.load_ohlcv(ticker, period=PERIOD))
    rows = evaluate_ticker(df, book="both")
    annotate(rows)
    chosen = pick(rows)
    book = "both"
    if chosen is None:
        rows = evaluate_ticker(df, book="long")
        annotate(rows)
        chosen = pick(rows)
        book = "long"
    result = {
        "ticker": ticker,
        "status": "no_edge" if chosen is None else "selected",
        "book": book if chosen is not None else "none",
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "n_bars": int(len(df)),
        "n_combos": int(len(rows)),
        "n_qualified": int(sum(row["qualified"] for row in rows)),
    }
    if chosen is None:
        return result
    result.update(_chosen_fields(chosen, book))
    return result


def _check_forward_gate():
    if not passes_forward_gate({"win_2025": 0.5, "comb_2025": 0.0}):
        raise AssertionError("win 50% and flat cumulative return should pass")
    if passes_forward_gate({"win_2025": 0.499, "comb_2025": 0.1}):
        raise AssertionError("win below 50% must fail")
    if passes_forward_gate({"win_2025": 1.0, "comb_2025": -0.0001}):
        raise AssertionError("negative cumulative return must fail")
    if passes_forward_gate({"win_2025": np.nan, "comb_2025": 0.0}):
        raise AssertionError("no forward trades must fail")
    one_long = [(pd.Timestamp("2017-03-01"), 0.10)]
    _n, long_comb, _median, _win = window_stats(
        one_long, [], *VAL_WINDOWS[0][1:], book="long",
    )
    _n, both_comb, _median, _win = window_stats(one_long, [], *VAL_WINDOWS[0][1:])
    if abs(long_comb - 0.10) > 1e-12:
        raise AssertionError(f"long-only compound should stay 10%, got {long_comb}")
    if abs(both_comb - 0.05) > 1e-12:
        raise AssertionError(f"empty short side should halve the both-book, got {both_comb}")
    long_preset = preset_from_row({
        "mode": "off",
        "adx_th": None,
        "ma_n": None,
        "PIVOT_ORDER": 4,
        "MIN_COUNT": 4,
    }, long_only=True)
    if not long_preset["long_only"] or not long_preset["summary"].endswith("、只做多"):
        raise AssertionError(long_preset["summary"])
    counter = preset_from_row({
        "mode": "counter",
        "adx_th": None,
        "ma_n": 100,
        "PIVOT_ORDER": 6,
        "MIN_COUNT": 4,
    })
    if counter["summary"] != "逆势、ADX 关、MA100、pivot 6、共振 4、去重 10":
        raise AssertionError(counter["summary"])
    if counter["aligned_keep"] or not counter["trend_on"] or counter["adx_th"] is not None:
        raise AssertionError("counter preset mapping")
    aligned = preset_from_row({
        "mode": "aligned",
        "adx_th": 25,
        "ma_n": 200,
        "PIVOT_ORDER": 6,
        "MIN_COUNT": 3,
    })
    if aligned["summary"] != "顺势、ADX>25、MA200、pivot 6、共振 3、去重 10":
        raise AssertionError(aligned["summary"])
    off = preset_from_row({
        "mode": "off",
        "adx_th": None,
        "ma_n": None,
        "PIVOT_ORDER": 4,
        "MIN_COUNT": 4,
    })
    if off["trend_on"] or off["adx_th"] is not None or off["ma_n"] != 200:
        raise AssertionError(f"off preset mapping: {off}")
    if off["summary"] != "滤波关、pivot 4、共振 4、去重 10":
        raise AssertionError(off["summary"])


def _fmt_choice(record):
    if record["status"] != "selected":
        return "no_edge"
    ma = record["ma_n"] if record["ma_n"] != "" else "-"
    adx = record["adx_th"] if record["adx_th"] != "" else "-"
    return (
        f"{record['mode']} adx={adx} ma={ma} "
        f"pivot={int(record['PIVOT_ORDER'])} min={int(record['MIN_COUNT'])}"
    )


def load_10y(ticker):
    df = tv.add_indicators(tv.load_ohlcv(ticker, period=PERIOD))
    print(f"  {ticker}: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} bars)")
    return df


def _fmt_folds(chosen):
    return " ".join(
        f"{name} 中位{chosen[f'median_{name}']:+.2%} 胜率{chosen[f'win_{name}']:.0%} 累计{chosen[f'comb_{name}']:+.2%}"
        for name, _start, _end in VAL_WINDOWS
    )


def main():
    _self_check()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("标的:", ", ".join(TICKERS))
    n_combos = len(list(iter_params()))

    grid_rows = []
    selected_rows = []
    loaded = 0
    for ticker in TICKERS:
        try:
            df = load_10y(ticker)
        except Exception as exc:
            print(f"[skip] {ticker}: {exc}")
            continue
        loaded += 1
        rows = evaluate_ticker(df)
        annotate(rows)
        chosen = pick(rows)
        grid_rows.extend(grid_record(ticker, row) for row in rows)
        selected = selected_record(ticker, rows, chosen)
        selected_rows.append(selected)
        n_ok = sum(row["qualified"] for row in rows)
        if chosen is None:
            print(f"{ticker}: no_edge（合格 {n_ok}/{n_combos}）")
        else:
            print(
                f"{ticker}: {_fmt_choice(selected)} | {_fmt_folds(chosen)} "
                f"score {chosen['score']:+.2%} | "
                f"2025中位 {chosen['median_2025']:+.2%} 胜率{chosen['win_2025']:.0%} "
                f"2025累计 {chosen['comb_2025']:+.2%} | 合格 {n_ok}/{n_combos}"
            )

    grid = pd.DataFrame(grid_rows)
    selected_df = pd.DataFrame(selected_rows)
    grid_path = OUT_DIR / "grid.csv"
    selected_path = OUT_DIR / "selected.csv"
    grid.to_csv(grid_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    n_selected = int((selected_df["status"] == "selected").sum()) if len(selected_df) else 0
    print(f"\n已载入 {loaded} 个标的，入选 {n_selected}，no_edge {loaded - n_selected}")
    print(f"网格: {grid_path}")
    print(f"入选: {selected_path}")
    return selected_df


if __name__ == "__main__":
    main()
