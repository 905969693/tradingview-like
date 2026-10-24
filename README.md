# 多指标背离分析系统（TradingView 复刻 + 扩展）

复刻并扩展了那张 TradingView 创业板指的图表思路：8 个技术指标的多指标共振背离检测 + 斐波那契回撤 + 趋势线 + 布林带 + 滚动 Volume Profile，并在此基础上增加了**信号后验统计**、**数据缓存**和**交互式复盘 Dashboard**。

## 文件结构

```
.
├── tradingview_realization_v2.py   # 核心模块：指标、背离检测、统计、绘图、批处理
├── dashboard.py                    # Streamlit 交互式复盘界面（需与核心模块同目录）
└── cache/                          # 行情缓存（自动生成），增量更新避免重复下载
```

## 快速开始

```bash
pip install yfinance pandas numpy scipy matplotlib plotly streamlit

# 批处理模式：跑一次，输出图表 + 统计 + 信号告警 CSV
python tradingview_realization_v2.py

# Dashboard 模式：浏览器交互式复盘
streamlit run dashboard.py
```

### 本机运行方式（macOS）

项目目录 `~/文稿/tradingview`，使用独立虚拟环境 `.venv`（依赖已通过 `requirements.txt` 安装）：

```bash
cd "/Users/yanyunfeng/文稿/tradingview"
source .venv/bin/activate
streamlit run dashboard.py
```

注意：修改了 `tradingview_realization_v2.py` 后需重启 Streamlit（对 `sys.path` 导入的模块热重载不生效）。

## 可调参数详解

所有参数集中在 `tradingview_realization_v2.py` 第 0 节，Dashboard 侧边栏可实时调节并联动重算。

### 背离信号参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `PIVOT_ORDER` | 6 | **pivot 灵敏度**。`argrelextrema` 判定局部高低点时的左右确认窗口（单位：K线根数）。调小（如 3）→ 小级别回调也算 pivot，信号变多变噪；调大（如 10）→ 只认显著转折点，信号少但可靠。**这是控制信号数量的首要旋钮** |
| `MIN_COUNT` | 3 | **共振指标数**。8 个指标（RSI/MACD柱/随机/动量/CCI/CMF/MFI/OBV）中至少几个同时出现背离才打标签。调大 → 信号极少但质量高（×6~8 通常对应重要顶底）；调小 → 信号多但假信号比例上升 |
| `DEDUP_BARS` | 10 | **去重窗口**。同方向信号在 N 根K线内只保留计数最高的一个。单边市里同一个顶部会产生一簇相邻信号，这个参数决定合并的松紧 |
| `STATS_HORIZONS` | (5, 10, 20) | 信号后验统计的持有期（交易日）。看信号后 5/10/20 天的收益分布，判断信号的有效时间窗口 |

**背离判定逻辑**：对每对相邻 pivot 比较——底背离 = 价格 lower low 但指标 higher low；顶背离相反。单指标背离假信号很多，多指标共振（动量类 + 摆动类 + 成交量类同时衰竭）可靠性才够，这也是复刻原图"×N"标签的设计。

⚠️ **确认延迟**：pivot 需要右侧 `PIVOT_ORDER` 根K线才能确认，所以信号实际是滞后的。用于回测时信号应 shift `PIVOT_ORDER` 根，否则结果偏乐观。

### 结构分析参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `LOOKBACK` | 160 | **斐波那契取哪段波段**。对最近 N 根K线的高低点计算 0.236/0.382/0.5/0.618 回撤位。波段选择决定一切：选最近一波推动浪最合理；设 `None` 用整段数据适合看长期结构 |
| `VP_WINDOW` | 60 | **滚动 Volume Profile 窗口**。第 t 天的 POC/VAH/VAL 只用过去 60 天数据计算，**无未来函数**，可直接当回测特征。调小 → 更贴近近期筹码；调大 → 更平滑稳定 |
| `VP_BINS` | 40 | Volume Profile 的价格分档数。调大 → 档位更细、渲染更精细；调小 → 更概括 |
| `BB_PERIOD` / `BB_STD` | 20 / 2.0 | 布林带周期/倍数。与背离配合使用：价格触及布林下轨 + 底背离共振，是经典的高胜率组合 |

### 运行参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `TICKERS` | SPY 等 | 批处理标的列表。支持美股、外汇（USDJPY=X）、港股（1810.HK）、A 股指数（000001.SS，Yahoo 数据质量一般建议换 akshare） |
| `PERIOD` | 5y | 数据区间。**信号统计建议至少 5 年**，2 年数据去重后信号太少，统计结论不可靠；有缓存后增量更新很快 |
| `SIGNAL_RECENT_DAYS` | 7 | "近 N 日信号"告警的截断窗口，输出到 `signals_recent.csv` |

## 输出文件

批处理每次运行生成一个以日期命名的目录：

- `{ticker}.png` — 主图：K线 + MA/VWMA + 布林带 + 斐波那契 + 滚动 POC/VA + 背离标注 + 右侧 Volume Profile
- `returns_{ticker}.png` — 底/顶背离后验收益分布直方图（5/10/20 日三色）
- `stats_{ticker}.csv` / `signal_stats_all.csv` — 单标的/跨标的信号后验统计表
- `signals_recent.csv` — 近 7 日全标的信号告警
- `run_summary.csv` — 运行摘要

## Dashboard 使用

`streamlit run dashboard.py` 后浏览器打开 `http://localhost:8501`：

- 左侧选标的、调参数，**全部图表联动重算**
- 顶部指标条：现价 vs POC/VAH/VAL 的位置
- 主图可框选缩放、悬停看数值；周末自动断开
- 统计表基于全量数据计算，不受"显示最近 N 根"的窗口裁剪影响

## 解读指南

- **背离信号**提示动量衰竭，适合减仓/收紧止损，**不要单独用来反手**；强趋势里会连续失效
- **VA 带**内运行 = 筹码平衡区；价格贴着带上/下沿运行并带动整条带迁移 = 趋势延续信号
- **POC（控制点）** = 市场成本重心，价格在 POC 上方说明多数人浮盈，反之套牢
- 后验统计表里**底背离胜率长期低于 50% 的标的**（如部分商品），信号应反着用或过滤掉

## 免责声明

本工具仅用于研究和复盘，不构成投资建议。背离等技术指标存在固有滞后性和假信号，历史统计不代表未来表现。
