#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国债期货 Kronos 生产级预测脚本（冻结模型版本）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每个品种使用独立的最优模型（原始 vs 微调），由冻结配置决定。

冻结配置: kronos_production_config.json
品种映射:
  TS(2Y) → 原始模型 (Original Sharpe 44.55 > FT 25.92)
  TF(5Y) → 原始模型 (Original Sharpe 54.51 > FT 4.78)
  T(10Y) → 微调模型 ✅ (FT Sharpe 40.12 > Original 23.63, 7ep早停)
  TL(30Y) → 原始模型 (数据不足，原始更稳定)

用法:
  # 默认（所有品种）
  python3 scripts/predict_daily_production.py

  # 指定品种
  python3 scripts/predict_daily_production.py --symbols T,TL

  # 不保存
  python3 scripts/predict_daily_production.py --no-save
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, argparse, json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.path.dirname(SCRIPT_DIR)
REPORT_DIR = os.path.join(KRONOS_DIR, "reports")

sys.path.insert(0, KRONOS_DIR)
sys.path.insert(0, "/home/llzjj/quant/macro_monitor")

from model import KronosTokenizer, Kronos, KronosPredictor
from kronos_dm_bridge import _get_nearby_futures_code, load_continuous_futures

# ─── 加载冻结配置 ──────────────────────────────────────
CONFIG_PATH = os.path.join(KRONOS_DIR, "kronos_production_config.json")
with open(CONFIG_PATH, encoding="utf-8") as f:
    PRODUCTION_CFG = json.load(f)

FROZEN_VERSION = PRODUCTION_CFG["config_version"]
VARIETIES = PRODUCTION_CFG["varieties"]
INF_PARAMS = PRODUCTION_CFG["inference_params"]

os.makedirs(REPORT_DIR, exist_ok=True)


def load_model_for_variety(symbol: str):
    """按冻结配置加载品种的最优模型"""
    cfg = VARIETIES.get(symbol)
    if not cfg:
        raise ValueError(f"未知品种: {symbol}")

    model_type = cfg["selected_model"]
    paths = cfg["model_paths"]
    tokenizer_path = os.path.join(KRONOS_DIR, paths["tokenizer"])
    basemodel_path = os.path.join(KRONOS_DIR, paths["basemodel"])

    print(f"  📦 加载 {model_type} 模型: {symbol}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(basemodel_path)
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


def predict_variety(symbol: str) -> dict:
    """预测单个品种"""
    cfg = VARIETIES[symbol]
    name = cfg["full_name"]
    code = _get_nearby_futures_code(symbol)

    LOOKBACK = INF_PARAMS["lookback"]
    PRED_LEN = INF_PARAMS["pred_len"]
    SAMPLE_COUNT = INF_PARAMS["sample_count"]
    TEMPERATURE = INF_PARAMS["temperature"]
    TOP_P = INF_PARAMS["top_p"]

    # 1. 加载数据
    from kronos_dm_bridge import KronosDM
    df = load_continuous_futures(symbol)
    if df.empty or len(df) < LOOKBACK + PRED_LEN:
        return {"symbol": symbol, "name": name, "code": code,
                "error": f"数据不足: 需要{LOOKBACK + PRED_LEN}行, 实际{len(df)}行",
                "data_bars": len(df)}

    last_close = float(df["close"].iloc[-1])

    # 2. 加载品种对应的最优模型
    kp = load_model_for_variety(symbol)
    dm = KronosDM(kp)

    # 3. 预测
    pred_df = dm.predict_from_df(
        df, lookback=LOOKBACK, pred_len=PRED_LEN,
        T=TEMPERATURE, top_p=TOP_P, sample_count=SAMPLE_COUNT, verbose=False
    )

    if pred_df.empty:
        return {"symbol": symbol, "name": name, "code": code,
                "error": "预测为空", "last_close": last_close}

    pred_close = pred_df["close"].values
    pred_last = float(pred_close[-1])
    pct_change = (pred_last - last_close) / last_close * 100

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
        "model_type": cfg["selected_model"],
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


def generate_report(results: list, run_time: str) -> str:
    """生成 Markdown 报告"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    cfg = PRODUCTION_CFG

    lines = [f"# 📊 国债期货Kronos预测报告（生产版）"]
    lines.append("")
    lines.append(f"**日期**: {date_str} | **生成时间**: {run_time}")
    lines.append(f"**配置**: {cfg['config_version']} (冻结于 {cfg['frozen_at']})")
    lines.append(f"**参数**: lookback={INF_PARAMS['lookback']}, pred_len={INF_PARAMS['pred_len']}, samples={INF_PARAMS['sample_count']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 汇总表
    lines.append("## 📌 综合汇总")
    lines.append("")
    lines.append("| 品种 | 代码 | 模型 | 当前收盘 | 预测收盘 | 变动% | 趋势 |")
    lines.append("|------|------|------|---------|---------|------|------|")

    valid = [r for r in results if "error" not in r]
    for r in valid:
        model_tag = "🎯 FT" if r['model_type'] == "fine-tuned" else "⚪ ORIG"
        lines.append(
            f"| {r['name']} | {r['code']} | {model_tag} | "
            f"{r['last_close']:.4f} | {r['pred_last']:.4f} | "
            f"{r['pct_change']:+.2f}% | {r['trend']} |")

    errors = [r for r in results if "error" in r]
    for r in errors:
        lines.append(f"| {r['name']} | {r['code']} | ❌ | — | — | — | {r['error']} |")
    lines.append("")

    # 各品种详情
    lines.append("---")
    lines.append("")
    lines.append("## 🔍 各品种预测详情")
    lines.append("")

    for r in valid:
        model_label = "🎯 微调模型 (fine-tuned)" if r['model_type'] == "fine-tuned" else "⚪ 原始模型 (original)"
        lines.append(f"### {r['name']} ({r['code']}) — {model_label}")
        lines.append("")
        lines.append(f"- **数据**: {r['data_bars']} 条日K线")
        lines.append(f"- **当前收盘**: {r['last_close']:.4f}")
        lines.append(f"- **预测{INF_PARAMS['pred_len']}日后收盘**: {r['pred_last']:.4f}")
        lines.append(f"- **预期变动**: {r['pct_change']:+.2f}%")
        lines.append(f"- **趋势判断**: {r['trend']}")
        lines.append("")

        lines.append(f"| 交易日 | 预测开盘 | 预测最高 | 预测最低 | 预测收盘 |")
        lines.append(f"|--------|---------|---------|---------|---------|")
        for i in range(INF_PARAMS["pred_len"]):
            lines.append(
                f"| Day {i+1} ({r['pred_dates'][i]}) | "
                f"{r['pred_open'][i]:.4f} | {r['pred_high'][i]:.4f} | "
                f"{r['pred_low'][i]:.4f} | {r['pred_close'][i]:.4f} |")
        lines.append("")

    for r in errors:
        lines.append(f"### ❌ {r['name']} ({r['code']})")
        lines.append("")
        lines.append(f"- **错误**: {r['error']}")
        lines.append("")

    # 模型选择依据
    lines.append("---")
    lines.append("")
    lines.append("## 📋 模型选择依据")
    lines.append("")
    lines.append("| 品种 | 选择 | 原始 Sharpe | FT Sharpe | 依据 |")
    lines.append("|------|------|:----------:|:---------:|------|")
    for sym, v in VARIETIES.items():
        mt = "🎯 FT" if v['selected_model'] == "fine-tuned" else "⚪ Original"
        os = f"{v['backtest_metrics']['original_sharpe']:.2f}" if v['backtest_metrics'].get('original_sharpe') else "—"
        fs = f"{v['backtest_metrics']['ft_sharpe']:.2f}" if v['backtest_metrics'].get('ft_sharpe') else "—"
        lines.append(f"| {v['full_name']} | {mt} | {os} | {fs} | {v['rationale']} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> ⚠️ 免责: Kronos 预测基于历史统计模式，不构成投资建议。")
    lines.append(">")
    lines.append(f"> *由 Hermes Agent + {cfg['config_version']} + DM Quant API 自动生成*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Kronos 生产级预测（冻结模型）")
    parser.add_argument("--symbols", default="TS,TF,T,TL",
                        help="品种列表，逗号分隔 (默认: TS,TF,T,TL)")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--no-save", action="store_true", help="仅打印")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip() in VARIETIES]
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y%m%d")

    print(f"🚀 Kronos 生产级日线预测 | {run_time}")
    print(f"   配置: {FROZEN_VERSION} (冻结于 {PRODUCTION_CFG['frozen_at']})")
    print(f"   品种: {', '.join(symbols)}")
    print(f"   模型选择: 每品种独立最优模型")
    print("=" * 60)

    # 逐个预测
    results = []
    for sym in symbols:
        name = VARIETIES[sym]["full_name"]
        model_type = VARIETIES[sym]["selected_model"]
        print(f"\n🔮 [{sym}] {name} ({'🎯 FT' if model_type=='fine-tuned' else '⚪ ORIG'})...")
        result = predict_variety(sym)
        if "error" in result:
            print(f"   ❌ {result['error']}")
        else:
            print(f"   ✅ {result['last_close']:.4f} → {result['pred_last']:.4f} "
                  f"({result['pct_change']:+.2f}%) {result['trend']}")
        results.append(result)

    # CSV追加输出(四品种日更, 供signal_snapshot.py读取) — 2026-09-20新增
    if not args.no_save:
        try:
            import pandas as pd
            csv_path = os.path.join(REPORT_DIR, "kronos_predictions_backfill.csv")
            rows = []
            for r in results:
                if "error" in r:
                    continue
                pred_date = r["pred_dates"][-1] if r.get("pred_dates") else datetime.now().strftime("%Y-%m-%d")
                rows.append({
                    "date": pred_date,
                    "symbol": r["symbol"],
                    "full_name": r["name"],
                    "model_type": r["model_type"],
                    "model_tag": "FT" if r["model_type"] == "fine-tuned" else "ORIG",
                    "last_close": r["last_close"],
                    "pred_close": r["pred_last"],
                    "pct_change": round(r["pct_change"], 4),
                    "trend": r["trend"],
                    "data_bars": r["data_bars"],
                })
            if rows:
                new_df = pd.DataFrame(rows)
                if os.path.exists(csv_path):
                    old_df = pd.read_csv(csv_path)
                    merged = pd.concat([old_df, new_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=["date", "symbol"], keep="last")
                else:
                    merged = new_df
                merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"\n📄 CSV已更新: {csv_path} (+{len(rows)}行, 总{len(merged)}行)")
        except Exception as e:
            print(f"\n⚠️ CSV输出失败: {type(e).__name__}: {e}")

    # 报告
    report = generate_report(results, run_time)
    print("\n" + "=" * 60)
    print(report)

    if not args.no_save:
        output_path = args.output or os.path.join(REPORT_DIR, f"kronos_prediction_{date_str}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 报告已保存: {output_path}")
    else:
        print("\n⚠️ --no-save，未保存")

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
