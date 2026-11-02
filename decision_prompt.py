"""把一条顶/底背离填成可复制的研究 prompt。不访问网络，也不调用模型。"""
import math

import pandas as pd

_KNOWN_ETFS = {
    "SPY": "美股宽基 ETF",
    "IWM": "美股小盘 ETF",
    "KBE": "美股银行 ETF",
    "KRE": "美股区域银行 ETF",
    "SOXX": "半导体 ETF",
    "EEM": "新兴市场股票 ETF",
    "DVY": "美股红利 ETF",
    "GLD": "黄金 ETF",
    "USO": "原油 ETF",
    "TLT": "美国长期国债 ETF",
}

_UNCLASSIFIED = "未内置分类：请先判断是个股、ETF、商品、债券还是外汇，再按对应口径分析"


def asset_hint(ticker):
    """按代码后缀和已知 ETF 给出资产提示。其余交给外部模型自行判断。"""
    raw = str(ticker).strip()
    upper = raw.upper()
    if upper.endswith("=X") or upper.endswith("_X"):
        return "外汇"
    if upper.endswith(".HK"):
        return "港股"
    if upper.endswith(".SS") or upper.endswith(".SZ"):
        return "A股"
    base = upper.split(".")[0]
    return _KNOWN_ETFS.get(base, _UNCLASSIFIED)


def match_bar(index, date_iso):
    """返回日历日等于 date_iso 的索引标签；对不上则返回 None。"""
    day = pd.Timestamp(date_iso).normalize()
    raw = pd.DatetimeIndex(pd.to_datetime(index))
    if raw.tz is not None:
        raw = raw.tz_convert("UTC").tz_localize(None)
    hits = [index[i] for i, ts in enumerate(raw.normalize()) if ts == day]
    if not hits:
        return None
    return hits[-1]


def format_posterior(stats):
    """后验表压成几行文字。胜率和收益仍是小数，这里换成百分数。"""
    if stats is None or len(stats) == 0:
        return "这段数据没有该方向的后验统计。"
    lines = []
    for _, row in stats.iterrows():
        n = int(row["信号数"]) if pd.notna(row["信号数"]) else 0
        lines.append(
            f"- {row['持有期']}：信号数 {n}，胜率 {_pct(row['胜率'])}，"
            f"无条件胜率 {_pct(row['无条件胜率'])}，平均收益 {_signed_pct(row['平均收益'])}"
        )
    return "\n".join(lines)


_TECHNICAL_SECTION = """## 技术面点评

先只根据上面「已给定的事实」做技术面点评。不要重算指标，不要改写背离是否成立，也不要补充事实里没有的价位或指标值。某一项若写的是「无」，点评里就说明没有这项，不要补一条信号。

逐项写它对当前决策是支持、削弱，还是信息不足，并给一句理由：

- 信号日和收盘
- 共振数
- 滤波状态
- 相对均线的位置
- 斐波那契
- POC、VAH、VAL
- 该方向的 5/10/20 日后验
"""

_FRAMEWORK = """资产提示若是「未内置分类」，先判断该代码是个股、ETF、商品、债券还是外汇，再选口径。

## 分析口径

- 个股：盈利与预期差、估值是否极端、资产负债表、近期指引
- ETF / 指数：底层驱动（利率、盈利、商品供需）、成分集中度、资金申赎
- 外汇：利差、央行路径、风险偏好
- 商品：库存、实际利率、供需事件
- 长债：期限溢价、央行、通胀数据

## 必须覆盖的四块

技术面点评写完后再写这四块。每一块先给一个标签，再写证据。查不到的数字写「未查到」。不要编造数字、日期或事件。
"""


def build_decision_prompt(
    *,
    ticker,
    timeframe,
    recent_window_label,
    signal=None,
    signal_date=None,
    signal_close=None,
    indicator_count=None,
    last_date,
    last_close,
    filter_status,
    ma_note,
    adx_note,
    fib_lines,
    poc,
    vah,
    val,
    posterior,
):
    """用当前页已经算好的事实填一段给外部模型的研究 prompt。

    signal 为空表示近窗口没有顶/底背离，事实段会照实写「无」。
    """
    fib_block = "\n".join(f"- {line}" for line in fib_lines) if fib_lines else "- 无"
    vp = _vp_note(last_close, poc, vah, val)
    if signal:
        bias = "偏多，期望价格上涨" if signal == "底背离" else "偏空，期望价格下跌"
        change = _change_since(signal_close, last_close)
        intro = f"""你是研究助理。下面这条技术面信号已经由本地多指标背离框架给出，视为既定事实。

请先根据已给出的事实做技术面点评，再用公开信息评估基本面、短期催化剂、资金流、上行风险和下行风险是否支持按这个信号做决定。时间尺度对齐持有 5、10、20 个交易日（大约未来 4 周）。不要写年度展望。

这是研究备忘，不是下单指令。结论里不要写具体仓位、买卖价或止损价。
"""
        facts = f"""## 已给定的事实

- 标的：{ticker}
- 资产提示：{asset_hint(ticker)}
- 周期：{timeframe}
- 近期窗口：近 {recent_window_label}
- 信号：{signal}（{bias}）
- 信号日和收盘：{signal_date}，收盘 {_price(signal_close)}
- 共振数：{int(indicator_count)}（8 个指标里同时背离的个数）
- 最新交易日：{last_date}，收盘 {_price(last_close)}，自信号日以来 {change}
- 滤波状态：{filter_status}
- 相对均线的位置：{ma_note}
- ADX：{adx_note}
- 斐波那契（相对最新收盘）：
{fib_block}
- POC、VAH、VAL：{vp}
- 该方向的 5/10/20 日后验（入场是信号确认后的收盘；用来看这段历史里信号有没有超出行情本身，不是对这笔交易的预测）：
{posterior}
"""
        tasks = f"""{_FRAMEWORK}
1. 基本面：和「{bias}」同向、反向，还是关系不大。标签用：支持信号、矛盾、或无关或未知。
2. 短期催化剂：未来约 4 周已经排期的事件（财报、央行会议、CPI 等）。没有已排期事件就写未查到。
3. 资金流：只用这个资产能查到的口径（ETF 申赎、成交拥挤、持仓、北向等）。口径对不上就写未查到，不要套用别的资产的资金流。
4. 上行风险点与下行风险点：各写 2 到 4 个可以被证伪的点，并写清触发条件。没有可靠的幅度就不要写百分比。

## 结尾格式

- 一句话结论：支持执行、观望、或与信号相反。并写信心：高、中、或低。结论要同时用到技术面点评和上面四块。
- 失效条件：若按信号行动，出现什么事实就否定这笔判断；若观望，要等到哪个事实。
- 信息缺口：还缺哪些关键事实。
"""
    else:
        intro = f"""你是研究助理。近 {recent_window_label}，{ticker} 没有底背离，也没有顶背离。不要编造一条信号，也不要把历史后验说成当前这笔交易。

请先根据已给出的事实做技术面点评，再用公开信息评估基本面、短期催化剂、资金流、上行风险和下行风险是否足以在没有信号时单独行动。时间尺度对齐未来约 4 周（大约 5、10、20 个交易日）。不要写年度展望。

这是研究备忘，不是下单指令。结论里不要写具体仓位、买卖价或止损价。
"""
        facts = f"""## 已给定的事实

- 标的：{ticker}
- 资产提示：{asset_hint(ticker)}
- 周期：{timeframe}
- 近期窗口：近 {recent_window_label}
- 信号：无。近 {recent_window_label} 无底背离、无顶背离
- 信号日和收盘：无
- 共振数：无
- 最新交易日：{last_date}，收盘 {_price(last_close)}
- 滤波状态：{filter_status}
- 相对均线的位置：{ma_note}
- ADX：{adx_note}
- 斐波那契（相对最新收盘）：
{fib_block}
- POC、VAH、VAL：{vp}
- 该方向的 5/10/20 日后验：无当前方向。下面是同一套参数的全样本历史，不是当前这一笔。
{posterior}
"""
        tasks = f"""{_FRAMEWORK}
没有当前信号，所以四块的标签改为：偏多、偏空、或不足以单独行动。

1. 基本面：现在的公开基本面更偏多、偏空，还是不足以在没有技术信号时单独行动。
2. 短期催化剂：未来约 4 周已经排期的事件（财报、央行会议、CPI 等）。没有已排期事件就写未查到。
3. 资金流：只用这个资产能查到的口径（ETF 申赎、成交拥挤、持仓、北向等）。口径对不上就写未查到，不要套用别的资产的资金流。
4. 上行风险点与下行风险点：各写 2 到 4 个可以被证伪的点，并写清触发条件。没有可靠的幅度就不要写百分比。

## 结尾格式

- 一句话结论：没有信号所以观望，或非技术面理由足够强所以单独偏多，或单独偏空。并写信心：高、中、或低。
- 失效条件：这个结论要被什么新事实推翻；若观望，要等到哪个事实。
- 信息缺口：还缺哪些关键事实。
"""
    return f"{intro}\n{facts}\n{_TECHNICAL_SECTION}\n{tasks}"


def _price(value):
    number = float(value)
    if abs(number) >= 100:
        return f"{number:,.2f}"
    return f"{number:,.4f}"


def _pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "未计算"
    return f"{float(value) * 100:.1f}%"


def _signed_pct(value):
    if value is None or pd.isna(value):
        return "未计算"
    return f"{float(value) * 100:+.2f}%"


def _change_since(signal_close, last_close):
    start = float(signal_close)
    if start == 0 or not math.isfinite(start) or not math.isfinite(float(last_close)):
        return "未计算"
    return _signed_pct(float(last_close) / start - 1)


def _vp_note(last_close, poc, vah, val):
    last = float(last_close)
    if last > float(vah):
        zone = "价值区上方"
    elif last < float(val):
        zone = "价值区下方"
    else:
        zone = "价值区内"
    side = "上方" if last >= float(poc) else "下方"
    return (
        f"POC {_price(poc)}，VAH {_price(vah)}，VAL {_price(val)}。"
        f"现价在{zone}，位于 POC {side}"
    )
