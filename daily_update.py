"""每个交易日早晨增量更新行情，并把新出现的信号写成待发送文本。

规则与 Dashboard 侧边栏「近 N 日有信号的标的」相同：有固定参数的标的用自己的一套，
其余用默认 pivot / 共振 / 去重，且不做趋势滤波。背离和九转配合都会写。
某个方向的主方法是九转时，走完的九转也写，入场是下一根收盘。
第一次运行只建立基线，不报历史信号。
之后只有没见过的、且落在近 SIGNAL_RECENT_DAYS 日里的信号才写入
cache/pending_grok_alert.txt。Grok Bot 读到正文后发到对话里，再把文件改成 NONE。

macOS 计划任务每天 07:00 和 18:10（本机时区）执行本脚本。
07:00 赶上美股收盘，18:10 赶上港股收盘。
"""
import argparse
import json
from datetime import datetime

import pandas as pd

import signal_presets as presets
import tradingview_realization_v2 as tv

STATE_PATH = tv.CACHE_DIR / "signal_state.json"
ALERT_PATH = tv.CACHE_DIR / "pending_grok_alert.txt"
NONE_MARK = "NONE"


def cache_periods(ticker):
    safe = tv.safe_chart_name(ticker)
    prefix = f"{safe}_"
    periods = []
    if tv.CACHE_DIR.exists():
        for path in sorted(tv.CACHE_DIR.glob(f"{safe}_*.csv")):
            period = path.name[len(prefix):-4]
            if period:
                periods.append(period)
    if tv.PERIOD not in periods:
        periods.append(tv.PERIOD)
    return periods


def refresh_caches(ticker):
    last = {}
    errors = []
    for period in cache_periods(ticker):
        try:
            df = tv.load_ohlcv(ticker, period=period)
            last[period] = pd.Timestamp(df.index[-1]).date().isoformat()
        except Exception as exc:
            errors.append(f"{period}: {exc}")
    return last, errors


def scan_params(ticker):
    """无固定参数时与侧边栏扫描一致：用默认 pivot，不套趋势滤波。"""
    spec = presets.preset_for(ticker)
    if spec is None:
        return {
            "pivot_order": tv.PIVOT_ORDER,
            "min_count": tv.MIN_COUNT,
            "dedup_bars": tv.DEDUP_BARS,
            "use_filter": False,
            "mode": "aligned",
            "adx_th": None,
            "ma_n": tv.TREND_MA_N,
            "long_only": False,
            "rule": "默认参数，未做趋势滤波",
        }
    if not spec["trend_on"]:
        mode = "off"
    elif spec["aligned_keep"]:
        mode = "aligned"
    else:
        mode = "counter"
    return {
        "pivot_order": spec["pivot_order"],
        "min_count": spec["min_count"],
        "dedup_bars": spec["dedup_bars"],
        "use_filter": bool(spec["trend_on"]),
        "mode": mode,
        "adx_th": spec["adx_th"],
        "ma_n": spec["ma_n"],
        "long_only": bool(spec.get("long_only", False)),
        "rule": spec.get("summary") or "",
    }


def recent_signals(ticker, days=tv.SIGNAL_RECENT_DAYS):
    params = scan_params(ticker)
    df = tv.add_indicators(tv.load_ohlcv(ticker, period=tv.PERIOD))
    bull_s = pd.Series(0, index=df.index)
    bear_s = pd.Series(0, index=df.index)
    for name in tv.INDICATORS:
        bull_idx, bear_idx = tv.find_divergences(df, df[name], order=params["pivot_order"])
        bull_s.iloc[bull_idx] += 1
        bear_s.iloc[bear_idx] += 1
    bull = tv.dedup_signals(bull_s[bull_s >= params["min_count"]], df, gap=params["dedup_bars"])
    bear = tv.dedup_signals(bear_s[bear_s >= params["min_count"]], df, gap=params["dedup_bars"])
    if params["use_filter"]:
        bull, bear = tv.apply_trend_filter(
            df, bull, bear, ma_n=params["ma_n"], mode=params["mode"], adx_th=params["adx_th"],
        )
    rows = tv.signals_in_window(bull, bear, df, ticker, days=days)
    order = params["pivot_order"]
    down, up = tv.td_completed_nines(df.Close)
    long_pairs = tv.confirmed_nine_overlap_pairs(df, bull, down, order)
    short_pairs = tv.confirmed_nine_overlap_pairs(df, bear, up, order)
    choice = tv.evaluate_method_choice(
        df, bull, bear, order, long_only=bool(params["long_only"]),
    )
    listed = set()
    for row in rows:
        pairs = long_pairs if row["signal"] == "底背离" else short_pairs
        note = tv.nine_overlap_note(pairs, row["date"], order)
        side = "long" if row["signal"] == "底背离" else "short"
        method_note = tv.method_signal_note(choice, side, "divergence")
        if note:
            method_note += tv.method_signal_note(choice, side, "overlap")
        row["rule"] = params["rule"]
        row["pivot_order"] = order
        row["nine_overlap"] = bool(note)
        row["nine_note"] = note
        row["method_note"] = method_note
        listed.add((row["signal"], row["date"]))
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    for pairs, base_signal, signal, signal_en in (
        (long_pairs, "底背离", "九转配合底背离", "nine_bullish_overlap"),
        (short_pairs, "顶背离", "九转配合顶背离", "nine_bearish_overlap"),
    ):
        if pairs is None or len(pairs) == 0:
            continue
        for _, one in pairs.iterrows():
            pivot = tv._naive_timestamp(one["pivot"]).normalize()
            entry = tv._naive_timestamp(one["entry"]).normalize()
            if entry < cutoff or (base_signal, pivot.date().isoformat()) in listed:
                continue
            entry_iso = entry.date().isoformat()
            pivot_iso = pivot.date().isoformat()
            rows.append({
                "ticker": ticker,
                "signal": signal,
                "signal_en": signal_en,
                "date": entry_iso,
                "indicator_count": 0,
                "close": round(float(df.loc[one["entry"], "Close"]), 4),
                "rule": params["rule"],
                "pivot_order": order,
                "nine_overlap": True,
                "nine_note": (
                    f"九转配合已确认背离：计数 7、8、9 与该背离点左右各 {order} 根重合。"
                    f"背离点在 {pivot_iso}，入场 {entry_iso}，取较晚的一根收盘"
                    "（九转第 9 根，或背离确认的最后一根）。"
                    "这是对这条已确认背离的加强。"
                ),
                "method_note": _overlap_method_note(
                    choice, "long" if base_signal == "底背离" else "short",
                ),
            })
    for sig, side, signal, signal_en in (
        (down, "long", "下跌九转", "nine_bullish"),
        (up, "short", "上涨九转", "nine_bearish"),
    ):
        side_choice = choice[side]
        if not side_choice["adopted"] or side_choice["primary"] != "nine":
            continue
        if sig is None or len(sig) == 0:
            continue
        for idx in sig.index:
            ts = tv._naive_timestamp(idx).normalize()
            if ts < cutoff:
                continue
            close = df.loc[idx, "Close"]
            if isinstance(close, pd.Series):
                close = close.iloc[-1]
            rows.append({
                "ticker": ticker,
                "signal": signal,
                "signal_en": signal_en,
                "date": ts.date().isoformat(),
                "indicator_count": 0,
                "close": round(float(close), 4),
                "rule": params["rule"],
                "pivot_order": order,
                "nine_overlap": False,
                "nine_note": "",
                "method_note": tv.method_signal_note(choice, side, "nine"),
            })
    return rows


def _overlap_method_note(choice, side):
    """单独成行的共振：写加仓还是只标记。该方向没有主方法时先说明。"""
    note = tv.method_signal_note(choice, side, "overlap")
    picked = choice.get(side) or {}
    if picked.get("adopted") and picked.get("primary") == "none":
        return "该方向没有通过下界的主方法。" + note
    return note


def signal_key(row):
    return f"{row['ticker']}|{row['signal_en']}|{row['date']}"


def apply_diff(previous, found):
    """found 只含本次成功扫描的标的。失败的标的沿用上次记录，避免下次误报成新信号。"""
    next_state = {
        key: row for key, row in previous.items() if row.get("ticker") not in found
    }
    new_rows = []
    for rows in found.values():
        for row in rows:
            key = signal_key(row)
            if key not in previous:
                new_rows.append(row)
            next_state[key] = row
    new_rows.sort(key=lambda row: (row["date"], row["ticker"], row["signal"]))
    return new_rows, next_state


def _append_method(lines, row):
    if row.get("method_note"):
        lines.append(f"  {row['method_note']}")


def format_message(rows):
    lines = [f"信号更新：新增 {len(rows)} 条"]
    for row in rows:
        en = str(row.get("signal_en", ""))
        if en in ("nine_bullish", "nine_bearish"):
            lines.append(
                f"{row['date']}  {row['ticker']}  {row['signal']}  第9根收盘 {row['close']}"
            )
            _append_method(lines, row)
            continue
        if en.startswith("nine_"):
            lines.append(
                f"{row['date']}  {row['ticker']}  {row['signal']}  入场收盘 {row['close']}"
            )
            if row.get("nine_note"):
                lines.append(f"  {row['nine_note']}")
            _append_method(lines, row)
            continue
        rule = f"（{row['rule']}）" if row.get("rule") else ""
        lines.append(
            f"{row['date']}  {row['ticker']}  {row['signal']} ×{row['indicator_count']}"
            f"  收盘 {row['close']}{rule}"
        )
        if row.get("nine_note"):
            lines.append(f"  {row['nine_note']}")
        _append_method(lines, row)
    lines.append(
        "规则与 Dashboard 侧边栏一致。九转配合是对已确认背离的加强，不是临时背离。"
        "下界中位数大于 0 时按加仓，否则只标记，不替换主方法。"
        "主方法是九转时才会单独写出九转，入场是下一根收盘。"
        "这是复盘提醒，不是买卖建议。"
    )
    return "\n".join(lines) + "\n"


def load_state():
    if not STATE_PATH.exists():
        return None
    try:
        raw = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    signals = raw.get("signals") if isinstance(raw, dict) else None
    if not isinstance(signals, dict):
        return None
    return signals


def save_state(signals):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "signals": signals,
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def pending_text():
    if not ALERT_PATH.exists():
        return ""
    text = ALERT_PATH.read_text().strip()
    if not text or text == NONE_MARK:
        return ""
    return text


def queue_alert(message):
    previous = pending_text()
    body = f"{previous}\n\n{message}" if previous else message
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_PATH.write_text(body)


def ack_alert():
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_PATH.write_text(NONE_MARK + "\n")


def run_update():
    tickers = presets.watchlist()
    print(f"watchlist {len(tickers)}: {', '.join(tickers)}")
    found = {}
    failures = []
    for ticker in tickers:
        last, errors = refresh_caches(ticker)
        if errors and tv.PERIOD not in last:
            failures.append(f"{ticker}: {'; '.join(errors)}")
            print(f"[fail] {ticker}: {'; '.join(errors)}")
            continue
        try:
            rows = recent_signals(ticker)
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            print(f"[fail] {ticker}: {exc}")
            continue
        found[ticker] = rows
        as_of = last.get(tv.PERIOD, "?")
        print(f"[ok] {ticker} as_of={as_of} recent={len(rows)}")

    previous = load_state()
    new_rows, nxt = apply_diff(previous or {}, found)
    save_state(nxt)
    if previous is None:
        print(f"STATUS baseline signals_in_window={sum(len(v) for v in found.values())}")
        return
    if not new_rows:
        print("STATUS no_new")
        if failures:
            print("failures: " + " | ".join(failures))
        return
    message = format_message(new_rows)
    queue_alert(message)
    print("STATUS new")
    print(message, end="")
    if failures:
        print("failures: " + " | ".join(failures))


def self_test():
    previous = {
        "SPY|bullish_divergence|2026-11-01": {
            "ticker": "SPY",
            "signal_en": "bullish_divergence",
            "signal": "底背离",
            "date": "2026-11-01",
        }
    }
    found = {
        "SPY": [
            previous["SPY|bullish_divergence|2026-11-01"],
            {
                "ticker": "SPY",
                "signal_en": "bearish_divergence",
                "signal": "顶背离",
                "date": "2026-11-07",
                "indicator_count": 4,
                "close": 700.0,
                "rule": "测试",
            },
        ]
    }
    new_rows, nxt = apply_diff(previous, found)
    assert [row["date"] for row in new_rows] == ["2026-11-07"]
    assert "SPY|bearish_divergence|2026-11-07" in nxt
    kept_prev = dict(previous)
    kept_prev["TLT|bullish_divergence|2026-11-02"] = {
        "ticker": "TLT",
        "signal_en": "bullish_divergence",
        "signal": "底背离",
        "date": "2026-11-02",
    }
    _new_rows, nxt = apply_diff(kept_prev, found)
    assert "TLT|bullish_divergence|2026-11-02" in nxt
    empty, nxt = apply_diff(kept_prev, {"SPY": []})
    assert empty == []
    assert not any(key.startswith("SPY|") for key in nxt)
    text = format_message([
        {
            "date": "2026-11-07",
            "ticker": "SPY",
            "signal": "底背离",
            "signal_en": "bullish_divergence",
            "indicator_count": 4,
            "close": 700.0,
            "rule": "测试",
            "nine_note": "九转配合已确认背离：计数 7、8、9 与该背离点左右各 3 根重合。",
        },
        {
            "date": "2026-11-10",
            "ticker": "TLT",
            "signal": "九转配合顶背离",
            "signal_en": "nine_bearish_overlap",
            "close": 90.0,
            "nine_note": "九转配合已确认背离：背离点在 2026-11-01，入场 2026-11-10。",
        },
    ])
    assert "九转配合已确认背离" in text
    assert "不是临时背离" in text
    assert "九转配合顶背离" in text
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser(description="增量更新行情，并排队新的背离信号")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ack", action="store_true", help="把待发送文件标成 NONE")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.ack:
        ack_alert()
        print("STATUS acked")
        return
    run_update()


if __name__ == "__main__":
    main()
