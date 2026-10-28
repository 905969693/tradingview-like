"""
样本外验证：训练段 (2021-10 ~ 2024-12) 选参数，测试段 (2025-01 起) 验证

方法要点：
- 训练/测试的信号分别在各自时间段内独立重算（pivot 只用段内数据，无边界前视；
  段边缘 pivot 缺少对侧 context，属正常截断）。
- 指标口径与 backtest_grid.py 完全一致：池化逐笔统计（5/10/20 日三档合并）
  + 多空半仓不重叠累计收益（三档均值）。
- 输出：16 组合 train/test 对照表 + 柱状对比图 + train 排名 vs test 表现的散点。

运行：python backtest_oos.py
"""
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tradingview_realization_v2 as tv
import backtest_divergence as bt
from backtest_grid import signals_with, load_cached_only

# ============ 0. 可调参数 ============
TICKERS = tv.TICKERS
PERIOD = tv.PERIOD
PIVOT_RANGE = range(3, 7)
MINCOUNT_RANGE = range(3, 7)
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
OUT_DIR = tv.SCRIPT_DIR / "backtest" / f"oos_{date.today().isoformat()}"


def seg_metrics(df_seg, p, m, use_filter=False, ma_n=200):
    """某时间段内单个参数组合的指标（与 grid 的 combo_metrics 同口径）。

    use_filter=True 时应用顺势滤波（tv.apply_trend_filter，段内计算 MA）。
    """
    bull, bear = signals_with(df_seg, p, m)
    if use_filter:
        bull, bear = tv.apply_trend_filter(df_seg, bull, bear, ma_n=ma_n, mode="aligned")
    tl = bt.build_trades(df_seg, bull, side="long", entry_shift=p)
    ts = bt.build_trades(df_seg, bear, side="short", entry_shift=p)
    cols = [f"{h}日_收益%" for h in tv.STATS_HORIZONS]
    parts = [t_[c] for t_ in (tl, ts) if len(t_) for c in cols if c in t_.columns]
    rets = pd.concat(parts).dropna() if parts else pd.Series(dtype=float)

    comb = []
    for h in tv.STATS_HORIZONS:
        eqL = bt.equity_curve(df_seg, bull, "long", h, entry_shift=p)
        eqS = bt.equity_curve(df_seg, bear, "short", h, entry_shift=p)
        rL = float(eqL.iloc[-1] - 1) if not eqL.empty else 0.0
        rS = float(eqS.iloc[-1] - 1) if not eqS.empty else 0.0
        comb.append((rL + rS) / 2)
    return {
        "n": int(len(rets)),
        "win_rate": float((rets > 0).mean()) if len(rets) else np.nan,
        "avg_ret": float(rets.mean()) if len(rets) else np.nan,
        "comb_ret": float(np.mean(comb)),
    }


def plot_train_test(rows, save_path):
    df = pd.DataFrame(rows)
    df["combo"] = df.apply(lambda r: f"p{r.PIVOT_ORDER}m{r.MIN_COUNT}", axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=110)
    for row, filt in enumerate(("off", "on")):
        sub = df[df["滤波"] == filt].sort_values("train_comb", ascending=False).reset_index(drop=True)
        x = np.arange(len(sub))
        w = 0.4
        axes[row][0].bar(x - w / 2, sub.train_comb * 100, w, label="train (21-10~24-12)", color="#42a5f5")
        axes[row][0].bar(x + w / 2, sub.test_comb * 100, w, label="test (25-01~至今)", color="#ef5350")
        axes[row][0].axhline(0, color="k", lw=0.8)
        axes[row][0].set_xticks(x, sub.combo, rotation=45, ha="right", fontsize=8)
        axes[row][0].set_ylabel("多空合并累计收益%")
        axes[row][0].set_title(f"趋势滤波 {filt}：train vs test（按 train 排序）")
        axes[row][0].legend(fontsize=8)

        axes[row][1].scatter(sub.train_comb * 100, sub.test_comb * 100, c="#7e57c2", s=40)
        for _, r in sub.iterrows():
            axes[row][1].annotate(r.combo,
                                  (r.train_comb * 100, r.test_comb * 100),
                                  fontsize=7, alpha=0.8)
        lim = max(abs(sub.train_comb.max()), abs(sub.test_comb.max()),
                  abs(sub.train_comb.min()), abs(sub.test_comb.min())) * 100 * 1.15
        axes[row][1].plot([-lim, lim], [-lim, lim], ls="--", color="gray", lw=0.8)
        axes[row][1].axhline(0, color="k", lw=0.6)
        axes[row][1].axvline(0, color="k", lw=0.6)
        axes[row][1].set_xlabel("train 累计收益%")
        axes[row][1].set_ylabel("test 累计收益%")
        corr = sub[["train_comb", "test_comb"]].corr().iloc[0, 1]
        axes[row][1].set_title(f"train-test 一致性 滤波{filt}（r = {corr:.2f}）")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for t in TICKERS:
        try:
            data[t] = load_cached_only(t)
        except Exception as e:
            print(f"[skip] {t}: {e}")
    print(f"已载入 {len(data)} 个标的（全部来自本地缓存，零网络请求）")
    print(f"训练段: ~{TRAIN_END.date()} | 测试段: {TEST_START.date()}~\n")

    rows = []
    for filt in (False, True):
        for p in PIVOT_RANGE:
            for m in MINCOUNT_RANGE:
                train_m, test_m = [], []
                for t, df in data.items():
                    tr = df[df.index <= TRAIN_END]
                    te = df[df.index >= TEST_START]
                    if len(tr) < 100 or len(te) < 60:
                        continue
                    train_m.append(seg_metrics(tr, p, m, use_filter=filt))
                    test_m.append(seg_metrics(te, p, m, use_filter=filt))
                trd, ted = pd.DataFrame(train_m), pd.DataFrame(test_m)
                rows.append({
                    "滤波": "on" if filt else "off",
                    "PIVOT_ORDER": p, "MIN_COUNT": m,
                    "train_交易数": int(trd["n"].sum()),
                    "train_胜率": round(float(trd["win_rate"].mean()), 4),
                    "train_均笔%": round(float(trd["avg_ret"].mean()), 3),
                    "train_comb": float(trd["comb_ret"].mean()),
                    "test_交易数": int(ted["n"].sum()),
                    "test_胜率": round(float(ted["win_rate"].mean()), 4),
                    "test_均笔%": round(float(ted["avg_ret"].mean()), 3),
                    "test_comb": float(ted["comb_ret"].mean()),
                })
                r = rows[-1]
                print(f"滤波={'on ' if filt else 'off'} pivot={p} min={m} | train: "
                      f"交易{r['train_交易数']:>4} 胜率{r['train_胜率']:.1%} comb {r['train_comb']:+.2%} | "
                      f"test: 交易{r['test_交易数']:>4} 胜率{r['test_胜率']:.1%} comb {r['test_comb']:+.2%}")

    df = pd.DataFrame(rows)
    df["train_comb%"] = (df.train_comb * 100).round(2)
    df["test_comb%"] = (df.test_comb * 100).round(2)
    out = df.sort_values(["滤波", "train_comb"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(OUT_DIR / "oos_summary.csv", index=False)
    plot_train_test(rows, OUT_DIR / "oos_train_test.png")

    for filt in ("off", "on"):
        sub = df[df["滤波"] == filt].sort_values("train_comb", ascending=False)
        best_train = sub.iloc[0]
        print(f"\n滤波={filt} train 最优: pivot={int(best_train.PIVOT_ORDER)} "
              f"min={int(best_train.MIN_COUNT)} (train comb {best_train.train_comb:+.2%}) "
              f"-> test comb {best_train.test_comb:+.2%}")
        corr = sub[["train_comb", "test_comb"]].corr().iloc[0, 1]
        print(f"滤波={filt} 16 组合 train/test 相关性 r = {corr:.2f}")
    print(f"输出: {OUT_DIR / 'oos_summary.csv'} | {OUT_DIR / 'oos_train_test.png'}")


if __name__ == "__main__":
    main()
