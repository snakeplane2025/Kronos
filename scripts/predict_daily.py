#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国债期货日线预测脚本：T / TF / TL / TS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

定时任务：每日 16:00 运行（收盘后）
功能：用 Kronos-base 模型从日K线角度预测四个国债期货品种的未来5个交易日走势

输出：~/quant/Kronos/reports/futures_prediction_YYYYMMDD.md

用法：
  # 手动运行
  cd ~/quant/Kronos && source venv/bin/activate
  python3 scripts/predict_daily.py

  # 指定预测步数
  python3 scripts/predict_daily.py --pred_len 10
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# ─── 路径 ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.path.dirname(SCRIPT_DIR)  # ~/quant/Kronos
REPORT_DIR = os.path.join(KRONOS_DIR, "reports")

sys.path.insert(0, KRONOS_DIR)
sys.path.insert(0, "/home/llzjj/quant/macro_monitor")

from model import KronosTokenizer, Kronos, KronosPredictor
from kronos_dm_bridge import (
    KronosDM, _get_nearby_futures_code, load_continuous_futures,
)

os.makedirs(REPORT_DIR, exist_ok=True)

# ─── 配置 ───────────────────────────────────────────────
FUTURES_SYMBOLS = {
    "TS": "2年期国债期货",
    "TF": "5年期国债期货",
    "T":  "10年期国债期货",
    "TL": "30年期国债期货",
}

LOOKBACK = 500       # 输入的历史K线根数（连续合约数据充足）
PRED_LEN = 5         # 预测步数（交易日）
KLINE_TYPE = 6       # 日线
SAMPLE_COUNT = 5     # 采样次数（取平均）
TEMPERATURE = 1.0
TOP_P = 0.9


def load_model() -> KronosPredictor:
    """加载 Kronos-base 模型（每个品种独立加载）"""
    tokenizer = KronosTokenizer.from_pretrained(
        os.path.join(KRONOS_DIR, "models/Kronos-Tokenizer-base")
    )
    model = Kronos.from_pretrained(
        os.path.join(KRONOS_DIR, "models/Kronos-base")
    )
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def predict_futures(symbol: str) -> dict:
    """预测单个品种（独立加载模型 + 独立取数 + 独立预测）

    每个品种独立调用：
      1. load_continuous_futures() → 长历史连续合约数据
      2. 新建 KronosPredictor（每次独立）
      3. predict_from_df() → 独立的推理管线

    Returns:
        dict: {symbol, name, code, last_close, predictions, trend, pct_change}
    """
    name = FUTURES_SYMBOLS[symbol]
    code = _get_nearby_futures_code(symbol)

    # 1. 加载连续合约数据
    df = load_continuous_futures(symbol)
    if df.empty or len(df) < LOOKBACK + PRED_LEN:
        return {
            "symbol": symbol, "name": name, "code": code,
            "error": f"数据不足: 需要{LOOKBACK + PRED_LEN}行, 实际{len(df)}行",
            "data_bars": len(df),
        }

    # 当前收盘价
    last_close = float(df["close"].iloc[-1])

    # 2. 独立加载预测器（确保每个品种独立推理）
    kp = load_model()
    dm = KronosDM(kp)

    # 3. 预测
    pred_df = dm.predict_from_df(
        df, lookback=LOOKBACK, pred_len=PRED_LEN,
        T=TEMPERATURE, top_p=TOP_P, sample_count=SAMPLE_COUNT, verbose=False
    )

    if pred_df.empty:
        return {
            "symbol": symbol, "name": name, "code": code,
            "error": "预测结果为空",
            "last_close": last_close,
        }

    # 提取预测结果
    pred_close = pred_df["close"].values
    pred_last = float(pred_close[-1])
    pct_change = (pred_last - last_close) / last_close * 100

    # 方向
    if pct_change > 0.3:
        trend = "📈 看涨"
    elif pct_change < -0.3:
        trend = "📉 看跌"
    else:
        trend = "➡️ 震荡"

    return {
        "symbol": symbol,
        "name": name,
        "code": code,
        "last_close": last_close,
        "pred_last": pred_last,
        "pct_change": pct_change,
        "trend": trend,
        "pred_close": pred_close,
        "pred_open": pred_df["open"].values,
        "pred_high": pred_df["high"].values,
        "pred_low": pred_df["low"].values,
        "pred_dates": [str(d)[:10] for d in pred_df.index],
        "data_bars": len(df),
    }


def generate_report(all_results: list, run_time: str) -> str:
    """生成 Markdown 报告"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    lines = []
    lines.append(f"# 📊 国债期货Kronos预测报告")
    lines.append(f"")
    lines.append(f"**日期**: {date_str} | **生成时间**: {run_time}")
    lines.append(f"**模型**: Kronos-base (102.3M) | **数据**: DM Quant API")
    lines.append(f"**参数**: lookback={LOOKBACK}, pred_len={PRED_LEN}, samples={SAMPLE_COUNT}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # ═══ 综合汇总 ═══
    lines.append(f"## 📌 综合汇总")
    lines.append(f"")
    lines.append(f"| 品种 | 代码 | 当前收盘 | 预测收盘 | 变动% | 趋势 |")
    lines.append(f"|------|------|---------|---------|------|------|")

    valid = [r for r in all_results if "error" not in r]
    for r in valid:
        lines.append(
            f"| {r['name']} | {r['code']} | "
            f"{r['last_close']:.4f} | {r['pred_last']:.4f} | "
            f"{r['pct_change']:+.2f}% | {r['trend']} |"
        )

    errors = [r for r in all_results if "error" in r]
    for r in errors:
        lines.append(f"| {r['name']} | {r['code']} | — | — | — | ❌ {r['error']} |")

    lines.append(f"")

    # ═══ 各品种详情 ═══
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 🔍 各品种预测详情")
    lines.append(f"")

    for r in valid:
        lines.append(f"### {r['name']} ({r['code']})")
        lines.append(f"")
        lines.append(f"- **数据**: {r['data_bars']} 条日K线")
        lines.append(f"- **当前收盘**: {r['last_close']:.4f}")
        lines.append(f"- **预测{PRED_LEN}日后收盘**: {r['pred_last']:.4f}")
        lines.append(f"- **预期变动**: {r['pct_change']:+.2f}%")
        lines.append(f"- **趋势判断**: {r['trend']}")
        lines.append(f"")

        # 逐日明细表
        lines.append(f"| 交易日 | 预测开盘 | 预测最高 | 预测最低 | 预测收盘 |")
        lines.append(f"|--------|---------|---------|---------|---------|")
        for i in range(PRED_LEN):
            lines.append(
                f"| Day {i+1} ({r['pred_dates'][i]}) | "
                f"{r['pred_open'][i]:.4f} | {r['pred_high'][i]:.4f} | "
                f"{r['pred_low'][i]:.4f} | {r['pred_close'][i]:.4f} |"
            )
        lines.append(f"")

    for r in errors:
        lines.append(f"### ❌ {r['name']} ({r['code']})")
        lines.append(f"")
        lines.append(f"- **错误**: {r['error']}")
        lines.append(f"")

    # ═══ 策略建议 ═══
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 🎯 策略参考")
    lines.append(f"")

    # 综合方向判断
    directions = [r.get("pct_change", 0) for r in valid]
    avg_direction = sum(directions) / len(directions) if directions else 0

    if avg_direction > 0.5:
        lines.append(f"📈 **整体看多偏强**（平均变动 {avg_direction:+.2f}%）")
    elif avg_direction < -0.5:
        lines.append(f"📉 **整体看空偏弱**（平均变动 {avg_direction:+.2f}%）")
    else:
        lines.append(f"➡️ **整体偏震荡**（平均变动 {avg_direction:+.2f}%）")
    lines.append(f"")

    # 品种间相对强弱
    if len(valid) >= 2:
        sorted_r = sorted(valid, key=lambda x: x["pct_change"], reverse=True)
        strongest = sorted_r[0]
        weakest = sorted_r[-1]
        lines.append(f"- 🟢 **最强**: {strongest['name']} ({strongest['pct_change']:+.2f}%)")
        lines.append(f"- 🔴 **最弱**: {weakest['name']} ({weakest['pct_change']:+.2f}%)")
        lines.append(f"- 📏 **期限结构**: {'陡峭化' if sorted_r[0]['symbol'] in ['TS','TF'] else '平坦化'}倾向")
        lines.append(f"")

    lines.append(f"> ⚠️ 免责: Kronos 预测基于历史统计模式，不构成投资建议。")
    lines.append(f"> 预测仅供参考，实际交易需结合基本面分析与风控。")
    lines.append(f">")
    lines.append(f"> *由 Hermes Agent + Kronos-base + DM Quant API 自动生成*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="国债期货日线Kronos预测")
    parser.add_argument("--pred_len", type=int, default=5, help="预测步数")
    parser.add_argument("--output", help="输出文件路径（默认自动生成）")
    parser.add_argument("--no-save", action="store_true", help="仅打印，不保存文件")
    args = parser.parse_args()

    pred_len = args.pred_len

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y%m%d")

    print(f"🚀 Kronos 国债期货日线预测 | {run_time}")
    print(f"   模型: Kronos-base | 每个品种独立加载 | lookback={LOOKBACK} pred_len={pred_len}")
    print("=" * 60)

    # 逐个预测（每个品种独立加载模型，独立推理）
    results = []
    for symbol in ["TS", "TF", "T", "TL"]:
        name = FUTURES_SYMBOLS[symbol]
        print(f"\n🔮 [{symbol}] 预测 {name}...")
        result = predict_futures(symbol)
        if "error" in result:
            print(f"   ❌ {result['error']}")
        else:
            print(f"   ✅ 当前: {result['last_close']:.4f} → 预测: {result['pred_last']:.4f} "
                  f"({result['pct_change']:+.2f}%) {result['trend']}")
        results.append(result)

    # 生成报告
    report = generate_report(results, run_time)
    print("\n" + "=" * 60)
    print(report)

    # 保存
    if not args.no_save:
        output_path = args.output or os.path.join(REPORT_DIR, f"futures_prediction_{date_str}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 报告已保存: {output_path}")
    else:
        print("\n⚠️ --no-save，未保存文件")

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
