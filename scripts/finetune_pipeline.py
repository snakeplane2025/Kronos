#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kronos 微调全流程 — 数据准备 + 配置生成 + 训练 + 回测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为 T/TF/TL/TS 四个国债期货品种分别：
1. 从连续合约CSV提取训练数据（按时间分割 train/val/test）
2. 生成各品种的 YAML 配置文件
3. 调用 finetune_csv 框架进行微调
4. 回测验证效果，输出最优解报告

用法:
  cd ~/quant/Kronos && source venv/bin/activate

  # 1. 数据准备
  python3 scripts/finetune_pipeline.py prepare

  # 2. 微调（所有品种）
  python3 scripts/finetune_pipeline.py train --epochs 5

  # 3. 回测
  python3 scripts/finetune_pipeline.py backtest

  # 4. 完整流程（prepare → train → backtest → report）
  python3 scripts/finetune_pipeline.py all --epochs 5
"""

import sys
import os
import argparse
import yaml
import json
from datetime import datetime, timedelta

# ─── 路径 ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.path.dirname(SCRIPT_DIR)  # ~/quant/Kronos
FINETUNE_CSV_DIR = os.path.join(KRONOS_DIR, "finetune_csv")
MODELS_DIR = os.path.join(KRONOS_DIR, "models")
OUTPUT_DIR = os.path.join(KRONOS_DIR, "finetuned_models")
REPORT_DIR = os.path.join(KRONOS_DIR, "reports")
CSV_DATA_DIR = "/home/llzjj/quant/backtraderUS/data/tickers"

sys.path.insert(0, KRONOS_DIR)
sys.path.insert(0, FINETUNE_CSV_DIR)

FUTURES_SYMBOLS = {
    "TS": "2年期国债期货",
    "TF": "5年期国债期货",
    "T": "10年期国债期货",
    "TL": "30年期国债期货",
}

# ─── 1. 数据准备 ────────────────────────────────────────

def prepare_data():
    """从连续合约CSV创建各品种训练数据"""
    os.makedirs(os.path.join(KRONOS_DIR, "finetune_data"), exist_ok=True)
    import pandas as pd

    for sym in ["TS", "TF", "T", "TL"]:
        csv_path = os.path.join(CSV_DATA_DIR, f"his_data_{sym}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ 跳过 {sym}: 文件不存在 {csv_path}")
            continue

        print(f"\n📦 准备 {sym} ({FUTURES_SYMBOLS[sym]})...")
        df = pd.read_csv(csv_path)

        # 压缩主力连续（每天多合约取成交量最大）
        df["timestamp"] = pd.to_datetime(df["datetime"])
        df["date_only"] = df["timestamp"].dt.date
        df = df.loc[df.groupby("date_only")["volume"].idxmax().dropna()].drop(columns=["date_only"])

        # 重命名列: 标准 CSV 格式 (finetune_csv 要求列名为 timestamps)
        df_out = df.rename(columns={"amt": "amount", "timestamp": "timestamps"})
        df_out = df_out[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
        df_out = df_out.sort_values("timestamps").reset_index(drop=True)

        # 按时间分割: 80% train, 10% val, 10% test
        n = len(df_out)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        df_train = df_out.iloc[:train_end]
        df_val = df_out.iloc[train_end:val_end]
        df_test = df_out.iloc[val_end:]

        # 保存
        base_path = os.path.join(KRONOS_DIR, "finetune_data", sym)
        os.makedirs(base_path, exist_ok=True)

        df_out.to_csv(os.path.join(base_path, f"{sym}_all.csv"), index=False)
        df_train.to_csv(os.path.join(base_path, f"{sym}_train.csv"), index=False)
        df_val.to_csv(os.path.join(base_path, f"{sym}_val.csv"), index=False)
        df_test.to_csv(os.path.join(base_path, f"{sym}_test.csv"), index=False)

        print(f"  ✅ {sym}: {len(df_out)} 行 (train={len(df_train)}, val={len(df_val)}, test={len(df_test)})")
        print(f"     时间范围: {str(df_out['timestamps'].iloc[0])[:10]} ~ {str(df_out['timestamps'].iloc[-1])[:10]}")

    print("\n✅ 数据准备完成!")


# ─── 2. 配置文件生成 ────────────────────────────────────

def create_config(symbol: str, epochs: int = 5) -> str:
    """创建单个品种的微调配置"""
    config = {
        "data": {
            "data_path": os.path.join(KRONOS_DIR, f"finetune_data/{symbol}/{symbol}_all.csv"),
            "lookback_window": 128,
            "predict_window": 10,
            "max_context": 512,
            "clip": 5.0,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
        },
        "training": {
            "tokenizer_epochs": max(1, epochs),
            "basemodel_epochs": max(1, epochs),
            "batch_size": 16,
            "log_interval": 20,
            "num_workers": 2,
            "seed": 42,
            "tokenizer_learning_rate": 2e-4,
            "predictor_learning_rate": 1e-5,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 1,
            "early_stop_patience": 5,
            "early_stop_delta": 0.001,
        },
        "model_paths": {
            "pretrained_tokenizer": os.path.join(MODELS_DIR, "Kronos-Tokenizer-base"),
            "pretrained_predictor": os.path.join(MODELS_DIR, "Kronos-base"),
            "exp_name": f"futures_{symbol}",
            "base_path": OUTPUT_DIR,
            "base_save_path": "",
            "finetuned_tokenizer": "",
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": f"kronos_finetune_{symbol}",
            "description": f"Kronos fine-tune on {symbol} ({FUTURES_SYMBOLS[symbol]}) daily futures data",
            "use_comet": False,
            "train_tokenizer": True,
            "train_basemodel": True,
            "skip_existing": False,
        },
        "device": {
            "use_cuda": False,
            "device_id": 0,
        },
    }

    config_dir = os.path.join(KRONOS_DIR, "finetune_configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"config_{symbol}.yaml")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

    print(f"  📝 配置已生成: {config_path}")
    return config_path


def create_all_configs(epochs: int = 5):
    """生成所有4品种的配置"""
    print("\n📋 生成各品种微调配置...")
    configs = {}
    for sym in ["TS", "TF", "T", "TL"]:
        configs[sym] = create_config(sym, epochs)
    return configs


# ─── 3. 微调执行 ────────────────────────────────────────

def run_finetune(symbol: str, epochs: int = 5, skip_tokenizer: bool = False,
                 skip_basemodel: bool = False, skip_existing: bool = False):
    """运行单个品种的微调"""
    config_path = os.path.join(KRONOS_DIR, "finetune_configs", f"config_{symbol}.yaml")
    if not os.path.exists(config_path):
        create_config(symbol, epochs)

    # 更新配置中的epochs
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["training"]["tokenizer_epochs"] = epochs
    config["training"]["basemodel_epochs"] = epochs
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

    print(f"\n🚀 [{symbol}] 开始微调 (epochs={epochs})...")

    # 构造命令行
    cmd = f"cd {FINETUNE_CSV_DIR} && python3 train_sequential.py --config {config_path}"
    if skip_tokenizer:
        cmd += " --skip-tokenizer"
    if skip_basemodel:
        cmd += " --skip-basemodel"
    if skip_existing:
        cmd += " --skip-existing"

    print(f"   $ {cmd}")
    ret = os.system(cmd)
    if ret == 0:
        print(f"\n✅ [{symbol}] 微调完成!")
        return True
    else:
        print(f"\n❌ [{symbol}] 微调失败 (exit={ret})")
        return False


def run_all_finetune(epochs: int = 5, skip_existing: bool = False):
    """对4个品种依次微调"""
    results = {}
    for sym in ["TS", "TF", "T", "TL"]:
        results[sym] = run_finetune(sym, epochs, skip_existing=skip_existing)
    return results


# ─── 4. 回测 ────────────────────────────────────────────

def run_backtest(symbol: str, model_path: str = None, tokenizer_path: str = None,
                 lookback: int = None, pred_len: int = 10, sample_count: int = 5,
                 use_original: bool = False):
    """对指定品种的模型进行回测

    回测逻辑:
      - 用测试集（最后10%数据）做滚动预测
      - 每次用 lookback 条历史预测 pred_len 条未来
      - 对比预测 vs 实际，计算准确率、方向胜率、误差等

    use_original=True: 强制使用原始预训练模型（对比基线）
    lookback=None: 从训练配置中读取，否则用默认128(T/TF/TS)或48(TL)
    """
    import sys
    sys.path.insert(0, KRONOS_DIR)
    sys.path.insert(0, "/home/llzjj/quant/macro_monitor")

    import pandas as pd
    import numpy as np
    from model import KronosTokenizer, Kronos, KronosPredictor

    # 从训练配置读取lookback
    if lookback is None:
        config_path = os.path.join(KRONOS_DIR, "finetune_configs", f"config_{symbol}.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            lookback = cfg.get("data", {}).get("lookback_window", 128)
        else:
            lookback = 48 if symbol == "TL" else 128

    # 路径解析
    ft_model_path = os.path.join(OUTPUT_DIR, f"futures_{symbol}", "basemodel", "best_model")
    ft_tk_path = os.path.join(OUTPUT_DIR, f"futures_{symbol}", "tokenizer", "best_model")

    if use_original:
        model_path = os.path.join(MODELS_DIR, "Kronos-base")
        tokenizer_path = os.path.join(MODELS_DIR, "Kronos-Tokenizer-base")
    else:
        model_path = model_path or ft_model_path
        tokenizer_path = tokenizer_path or ft_tk_path

    # 判断使用哪种模型
    use_finetuned = not use_original and os.path.exists(model_path) and os.path.exists(tokenizer_path)

    print(f"\n🔬 [{symbol}] 回测开始..." if use_finetuned else f"\n🔬 [{symbol}] 回测 (使用原始模型)...")

    # 加载模型
    if use_finetuned:
        print(f"  📥 加载微调模型: {os.path.basename(model_path)}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        model = Kronos.from_pretrained(model_path)
    else:
        print(f"  📥 加载原始预训练模型: Kronos-base")
        tokenizer = KronosTokenizer.from_pretrained(os.path.join(MODELS_DIR, "Kronos-Tokenizer-base"))
        model = Kronos.from_pretrained(os.path.join(MODELS_DIR, "Kronos-base"))

    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    # 加载测试数据
    test_csv = os.path.join(KRONOS_DIR, "finetune_data", symbol, f"{symbol}_test.csv")
    if not os.path.exists(test_csv):
        # 如果没有分割数据，用全部数据
        test_csv = os.path.join(KRONOS_DIR, "finetune_data", symbol, f"{symbol}_all.csv")

    df = pd.read_csv(test_csv)
    df["timestamp"] = pd.to_datetime(df["timestamps"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    print(f"  📊 测试数据: {len(df)} 行 ({df.index[0].date()} ~ {df.index[-1].date()})")

    if len(df) < lookback + pred_len + 1:
        print(f"  ⚠️ 数据不足，跳过回测")
        return None

    # 滚动回测
    results = []
    errors = []
    directions = []  # 1=方向正确, 0=错误

    total_steps = len(df) - lookback - pred_len
    step = max(1, pred_len)  # 每次滑动pred_len步（避免重叠）

    print(f"  🔄 滚动回测 {total_steps} 组...")

    for i in range(0, total_steps, step):
        # 取输入数据
        x_df = df.iloc[i:i+lookback][["open", "high", "low", "close", "volume", "amount"]]
        actual = df.iloc[i+lookback:i+lookback+pred_len]

        # 跳过数据不足
        if len(x_df) < lookback or len(actual) < pred_len:
            break

        # 时间戳
        x_ts = x_df.index.to_series().reset_index(drop=True)
        y_ts = actual.index.to_series().reset_index(drop=True)

        try:
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, T=1.0, top_p=0.9,
                sample_count=sample_count, verbose=False
            )

            if pred_df.empty:
                continue

            # 评估：只看收盘价方向
            actual_close = actual["close"].values[:len(pred_df)]
            pred_close = pred_df["close"].values

            # 方向准确率 (涨/跌)
            actual_dir = np.sign(np.diff(actual_close, prepend=actual_close[0]))
            pred_dir = np.sign(np.diff(pred_close, prepend=pred_close[0]))
            dir_correct = np.mean(actual_dir == pred_dir)

            # 绝对误差
            mape = np.mean(np.abs((pred_close - actual_close) / (actual_close + 1e-10))) * 100
            rmse = np.sqrt(np.mean((pred_close - actual_close) ** 2))

            # 最终方向
            final_actual_dir = 1 if actual_close[-1] > actual_close[0] else -1
            final_pred_dir = 1 if pred_close[-1] > pred_close[0] else -1
            dir_match = final_actual_dir == final_pred_dir

            results.append({
                "step": i,
                "date": actual.index[0],
                "actual_close_start": actual_close[0],
                "actual_close_end": actual_close[-1],
                "pred_close_end": pred_close[-1],
                "mape": mape,
                "rmse": rmse,
                "dir_accuracy": dir_correct,
                "final_dir_match": dir_match,
                "actual_dir": "涨" if final_actual_dir > 0 else "跌",
                "pred_dir": "涨" if final_pred_dir > 0 else "跌",
            })

            if (i // step) % 10 == 0 and (i // step) < 20:
                print(f"    Step {i}: pred={pred_close[-1]:.2f} actual={actual_close[-1]:.2f} "
                      f"dir={'✅' if dir_match else '❌'} mape={mape:.2f}%")

        except Exception as e:
            print(f"    ⚠️ Step {i}: {e}")
            continue

    if not results:
        print(f"  ⚠️ 无有效回测结果")
        return None

    # 汇总统计
    df_results = pd.DataFrame(results)
    total_trades = len(df_results)
    dir_win_rate = df_results["final_dir_match"].mean() * 100
    avg_mape = df_results["mape"].mean()
    avg_dir_accuracy = df_results["dir_accuracy"].mean() * 100
    avg_rmse = df_results["rmse"].mean()

    # 收益率模拟 (简单做多方向准确的交易)
    returns = []
    for _, r in df_results.iterrows():
        ret = (r["actual_close_end"] - r["actual_close_start"]) / r["actual_close_start"]
        if r["final_dir_match"]:
            returns.append(ret)  # 方向正确则获得收益
        else:
            returns.append(-ret)  # 方向错误则亏损

    total_return = (1 + np.array(returns)).prod() - 1
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 / (pred_len / 252))

    # 结果
    backtest_result = {
        "symbol": symbol,
        "model": "fine-tuned" if use_finetuned else "original",
        "total_trades": total_trades,
        "direction_win_rate": round(dir_win_rate, 2),
        "avg_step_dir_accuracy": round(avg_dir_accuracy, 2),
        "avg_mape": round(avg_mape, 2),
        "avg_rmse": round(avg_rmse, 2),
        "total_return": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "avg_return_per_trade": round(np.mean(returns) * 100, 3),
        "std_return": round(np.std(returns) * 100, 3),
    }

    print(f"\n  📊 回测结果 ({symbol}):")
    print(f"     总交易次数: {backtest_result['total_trades']}")
    print(f"     方向胜率: {backtest_result['direction_win_rate']:.1f}%")
    print(f"     步内方向准确率: {backtest_result['avg_step_dir_accuracy']:.1f}%")
    print(f"     平均MAPE: {backtest_result['avg_mape']:.2f}%")
    print(f"     累计收益率: {backtest_result['total_return']:+.2f}%")
    print(f"     夏普比率: {backtest_result['sharpe_ratio']}")

    return backtest_result


def run_all_backtest():
    """对所有品种进行回测（原始模型 vs 微调模型对比）"""
    print("\n" + "=" * 60)
    print("📊 全品种回测对比")
    print("=" * 60)

    all_results = []
    for sym in ["TS", "TF", "T", "TL"]:
        # 先测原始模型（强制使用原始预训练模型）
        result_orig = run_backtest(sym, use_original=True)
        if result_orig:
            result_orig["model_type"] = "original"
            all_results.append(result_orig)

        # 再测微调模型（如果存在）
        ft_path = os.path.join(OUTPUT_DIR, f"futures_{sym}", "basemodel", "best_model")
        tk_path = os.path.join(OUTPUT_DIR, f"futures_{sym}", "tokenizer", "best_model")
        if os.path.exists(ft_path) and os.path.exists(tk_path):
            result_ft = run_backtest(sym, ft_path, tk_path)
            if result_ft:
                result_ft["model_type"] = "fine-tuned"
                all_results.append(result_ft)

    return all_results


# ─── 5. 报告生成 ────────────────────────────────────────

def generate_report(backtest_results: list = None):
    """生成最终报告PDF/Markdown"""
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = os.path.join(REPORT_DIR, f"finetune_report_{date_str}.md")

    lines = []
    lines.append(f"# Kronos 国债期货微调报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**模型**: Kronos-base → 分别微调 T/TF/TL/TS")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    if backtest_results:
        lines.append(f"## 📊 回测结果汇总")
        lines.append(f"")
        lines.append(f"| 品种 | 模型类型 | 交易次数 | 方向胜率 | 步内准确率 | MAPE | 累计收益率 | 夏普 |")
        lines.append(f"|------|---------|---------|---------|-----------|------|----------|------|")

        for r in backtest_results:
            lines.append(
                f"| {r['symbol']} | {r['model_type']} | {r['total_trades']} | "
                f"{r['direction_win_rate']}% | {r['avg_step_dir_accuracy']}% | "
                f"{r['avg_mape']}% | {r['total_return']:+.2f}% | {r['sharpe_ratio']} |"
            )
        lines.append(f"")

        # 最优解
        best = max(backtest_results, key=lambda x: x.get("sharpe_ratio", -999))
        lines.append(f"### 🏆 最优解")
        lines.append(f"")
        lines.append(f"- **品种**: {best['symbol']} ({FUTURES_SYMBOLS[best['symbol']]})")
        lines.append(f"- **模型**: {best['model_type']}")
        lines.append(f"- **夏普比率**: {best['sharpe_ratio']}")
        lines.append(f"- **方向胜率**: {best['direction_win_rate']}%")
        lines.append(f"- **累计收益率**: {best['total_return']:+.2f}%")
        lines.append(f"")
    else:
        lines.append(f"*无回测数据*")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"## 实验配置")
    lines.append(f"")
    lines.append(f"- **数据来源**: 连续合约CSV (主力连续)")
    lines.append(f"- **分割比例**: 80% train / 10% val / 10% test")
    lines.append(f"- **回测方式**: 滚动预测，每次lookback步预测pred_len步")
    lines.append(f"")

    report_text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n📄 报告已保存: {report_path}")
    return report_path


# ─── 主入口 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kronos 国债期货微调全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("action", choices=["prepare", "config", "train", "backtest",
                                            "report", "all"],
                        help="执行动作")
    parser.add_argument("--symbol", default=None,
                        help="品种代码 (TS/TF/T/TL)，默认全部")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数 (默认5)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的模型")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="跳过Tokenizer训练")
    parser.add_argument("--skip-basemodel", action="store_true",
                        help="跳过基础模型训练")

    args = parser.parse_args()
    symbols = [args.symbol] if args.symbol else ["TS", "TF", "T", "TL"]

    if args.action == "prepare":
        prepare_data()
        create_all_configs(args.epochs)

    elif args.action == "config":
        create_all_configs(args.epochs)

    elif args.action == "train":
        for sym in symbols:
            run_finetune(sym, args.epochs,
                         skip_tokenizer=args.skip_tokenizer,
                         skip_basemodel=args.skip_basemodel,
                         skip_existing=args.skip_existing)

    elif args.action == "backtest":
        results = run_all_backtest()
        generate_report(results)

    elif args.action == "report":
        generate_report()

    elif args.action == "all":
        print("=" * 60)
        print("🏁 Kronos 国债期货微调全流程")
        print("=" * 60)

        # Step 1: 准备数据
        print("\n📦 Step 1/3: 数据准备")
        prepare_data()
        create_all_configs(args.epochs)

        # Step 2: 训练
        print("\n🎯 Step 2/3: 微调训练")
        for sym in symbols:
            run_finetune(sym, args.epochs, skip_existing=args.skip_existing)

        # Step 3: 回测
        print("\n📊 Step 3/3: 回测")
        results = run_all_backtest()

        # 报告
        report_path = generate_report(results)
        print(f"\n✅ 全流程完成! 报告: {report_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
