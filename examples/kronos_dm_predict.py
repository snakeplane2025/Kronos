#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kronos × DM API — 实战预测示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用法:
  cd ~/quant/Kronos
  source venv/bin/activate

  # 预测国债期货 T 主力日线（下一个交易日）
  python3 examples/kronos_dm_predict.py --type futures --symbol T --lookback 200 --pred_len 5

  # 预测沪深300日线
  python3 examples/kronos_dm_predict.py --type index --code 000300.SH --lookback 200 --pred_len 5

  # 显示所有可用选项
  python3 examples/kronos_dm_predict.py --help
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# ─── 路径设置 ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import KronosTokenizer, Kronos, KronosPredictor
from kronos_dm_bridge import (
    KronosDM, _get_nearby_futures_code,
    fetch_futures_bars, fetch_equity_bars,
)


# ─── 模型路径 ────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_kronos(model_name: str = "Kronos-base", device: str = "cpu") -> KronosPredictor:
    """加载 Kronos 模型"""
    tokenizer_name = "Kronos-Tokenizer-base"
    tokenizer_path = os.path.join(MODEL_DIR, tokenizer_name)
    model_path = os.path.join(MODEL_DIR, model_name)

    print(f"🤖 加载 {model_name}...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(model_path)
    model.eval()

    max_context = 512  # base模型
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    print(f"   ✅ {model_name}: {sum(p.numel() for p in model.parameters()):,} params | device={device}")
    return predictor


def main():
    parser = argparse.ArgumentParser(
        description="Kronos × DM API — 金融K线预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 国债期货T主力 日线预测
  python3 examples/kronos_dm_predict.py -t futures -s T -b 200 -p 5

  # 5分钟K线预测沪深300
  python3 examples/kronos_dm_predict.py -t index -c 000300.SH -b 400 -p 24 -k 2

  # 预测10Y国债活跃券
  python3 examples/kronos_dm_predict.py -t bond -c 2400001.IB -b 200 -p 10
        """
    )
    parser.add_argument("-t", "--type", choices=["futures", "index", "bond"],
                        default="futures", help="预测类型")
    parser.add_argument("-s", "--symbol", default="T", help="期货品种 (T/TF/TL), 默认T")
    parser.add_argument("-c", "--code", help="证券/指数/债券代码 (如 000300.SH, 2400001.IB)")
    parser.add_argument("-b", "--lookback", type=int, default=200, help="输入长度 (默认200)")
    parser.add_argument("-p", "--pred_len", type=int, default=5, help="预测步数 (默认5)")
    parser.add_argument("-k", "--kline_type", type=int, default=6,
                        help="K线类型: 1=1min 2=5min 3=15min 4=30min 5=60min 6=日线")
    parser.add_argument("-m", "--model", default="Kronos-base",
                        help="模型名 (Kronos-small / Kronos-base)")
    parser.add_argument("--device", default="cpu", help="设备 (cpu / cuda:0)")
    parser.add_argument("--temp", type=float, default=1.0, help="采样温度")
    parser.add_argument("--samples", type=int, default=5, help="采样次数")
    parser.add_argument("--verbose", action="store_true", default=True, help="显示详细输出")

    args = parser.parse_args()

    # 1. 加载模型
    predictor = load_kronos(args.model, args.device)

    # 2. 创建 DM 桥接
    dm = KronosDM(predictor)

    # 3. 执行预测
    print("\n" + "=" * 60)
    kline_names = {1:"1分钟", 2:"5分钟", 3:"15分钟", 4:"30分钟", 5:"60分钟", 6:"日线", 7:"周线", 8:"月线"}
    print(f"📈 {kline_names.get(args.kline_type, f'K线type{args.kline_type}')} 预测")
    print("=" * 60)

    if args.type == "futures":
        code = args.code or _get_nearby_futures_code(args.symbol)
        print(f"🎯 期货: {args.symbol} ({code})")
        pred_df = dm.predict_futures(
            code=code,
            lookback=args.lookback,
            pred_len=args.pred_len,
            kline_type=args.kline_type,
            T=args.temp,
            sample_count=args.samples,
            verbose=args.verbose,
        )

    elif args.type == "index":
        code = args.code or "000300.SH"
        name = {"000300.SH": "沪深300", "000905.SH": "中证500",
                "000016.SH": "上证50", "000688.SH": "科创50"}.get(code, code)
        print(f"🎯 指数: {name}")
        pred_df = dm.predict_index(
            code=code,
            lookback=args.lookback,
            pred_len=args.pred_len,
            kline_type=args.kline_type,
            T=args.temp,
            sample_count=args.samples,
            verbose=args.verbose,
        )

    elif args.type == "bond":
        code = args.code or "2400001.IB"
        print(f"🎯 债券: {code}")
        pred_df = dm.predict_bond(
            code=code,
            lookback=args.lookback,
            pred_len=args.pred_len,
            kline_type=args.kline_type,
            T=args.temp,
            sample_count=args.samples,
            verbose=args.verbose,
        )

    # 4. 打印摘要
    dm.print_summary(pred_df)

    # 5. 终端表格显示
    if not pred_df.empty:
        n_display = min(args.pred_len, len(pred_df))
        print(f"\n📊 预测明细 (前{n_display}行):")
        display = pred_df[["open", "high", "low", "close"]].copy()
        display = display.round(4)
        print(display.to_string())

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
