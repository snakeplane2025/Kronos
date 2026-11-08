#!/usr/bin/env python3
"""
Kronos 全历史回填预测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
按生产配置（per-variety 最优模型），每20个交易日滑动窗口，
生成全历史预测信号。
输出: CSV（完整回测信号）+ MD报告（与现有格式兼容）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, json, glob, time
from datetime import datetime, timedelta
from pathlib import Path

KRONOS_DIR = "/home/llzjj/quant/Kronos"
sys.path.insert(0, KRONOS_DIR)
sys.path.insert(0, "/home/llzjj/quant/macro_monitor")

import numpy as np
import pandas as pd
from model import KronosTokenizer, Kronos, KronosPredictor
from kronos_dm_bridge import load_continuous_futures, KronosDM

# ─── 加载生产配置 ────────────────────────────────────
with open(os.path.join(KRONOS_DIR, "kronos_production_config.json")) as f:
    CFG = json.load(f)

VARIETIES = CFG["varieties"]
INF_PARAMS = CFG["inference_params"]
LOOKBACK = INF_PARAMS["lookback"]
PRED_LEN = INF_PARAMS["pred_len"]
STRIDE = 20  # 每月频滑动（约20个交易日）

REPORT_DIR = os.path.join(KRONOS_DIR, "reports")
CSV_PATH = os.path.join(REPORT_DIR, "kronos_predictions_backfill.csv")
os.makedirs(REPORT_DIR, exist_ok=True)

def load_model(symbol):
    cfg = VARIETIES[symbol]
    mtype = cfg["selected_model"]
    paths = cfg["model_paths"]
    tok = KronosTokenizer.from_pretrained(os.path.join(KRONOS_DIR, paths["tokenizer"]))
    model = Kronos.from_pretrained(os.path.join(KRONOS_DIR, paths["basemodel"]))
    model.eval()
    print(f"  [{symbol}] loaded {mtype} model")
    return KronosPredictor(model, tok, device="cpu", max_context=512)

def trend_label(pct):
    if pct > 0.3: return "看涨"
    elif pct < -0.3: return "看跌"
    else: return "震荡"

def trend_emoji(pct):
    if pct > 0.3: return "📈"
    elif pct < -0.3: return "📉"
    else: return "➡️"

def run_backfill():
    all_records = []
    
    for symbol in ["T", "TF", "TS", "TL"]:
        cfg = VARIETIES[symbol]
        print(f"\n{'='*50}")
        print(f"  {cfg['full_name']} ({symbol})")
        print(f"{'='*50}")
        
        # 加载数据
        df = load_continuous_futures(symbol)
        if df.empty or len(df) < LOOKBACK + PRED_LEN:
            print(f"  SKIP: 数据不足 ({len(df)} rows, need {LOOKBACK + PRED_LEN})")
            continue
        
        print(f"  Data: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        
        # 计算窗口
        total_windows = (len(df) - LOOKBACK - PRED_LEN) // STRIDE + 1
        print(f"  Windows: {total_windows} (stride={STRIDE})")
        
        # 加载模型
        kp = load_model(symbol)
        dm = KronosDM(kp)
        model_tag = "🎯 FT" if cfg["selected_model"] == "fine-tuned" else "⚪ ORIG"
        
        symbol_records = []
        t0 = time.time()
        
        for w in range(total_windows):
            # 窗口结束位置 = 第w个窗口的最后一天
            end_idx = len(df) - (total_windows - w - 1) * STRIDE
            # 需要至少 lookback + pred_len 行
            start_idx = max(0, end_idx - LOOKBACK - PRED_LEN)
            window = df.iloc[start_idx:end_idx]
            
            if len(window) < LOOKBACK + 1:
                continue
            
            # 预测日期 = 窗口内 lookback 部分的最后一天
            pred_date = window.index[min(LOOKBACK, len(window)) - 1].date()
            
            try:
                pred_df = dm.predict_from_df(
                    window, lookback=LOOKBACK, pred_len=PRED_LEN,
                    T=INF_PARAMS["temperature"], top_p=INF_PARAMS["top_p"],
                    sample_count=INF_PARAMS["sample_count"], verbose=False
                )
                
                if pred_df.empty:
                    continue
                
                last_close = float(window["close"].iloc[-1])
                pred_close = float(pred_df["close"].values[-1])
                pct = (pred_close - last_close) / last_close * 100
                
                record = {
                    "date": str(pred_date),
                    "symbol": symbol,
                    "full_name": cfg["full_name"],
                    "model_type": cfg["selected_model"],
                    "model_tag": model_tag,
                    "last_close": round(last_close, 4),
                    "pred_close": round(pred_close, 4),
                    "pct_change": round(pct, 2),
                    "trend": f"{trend_emoji(pct)} {trend_label(pct)}",
                    "data_bars": len(window),
                }
                symbol_records.append(record)
                all_records.append(record)
                
            except Exception as e:
                print(f"  [W{w+1}/{total_windows}] {pred_date}: ERROR {e}")
                continue
            
            elapsed = time.time() - t0
            avg = elapsed / (w + 1)
            remaining = avg * (total_windows - w - 1)
            if (w + 1) % 10 == 0 or w == 0:
                print(f"  [{w+1}/{total_windows}] {pred_date}: pct={pct:.2f}%, "
                      f"elapsed={elapsed:.0f}s, eta={remaining:.0f}s", flush=True)
        
        print(f"  Done: {len(symbol_records)} predictions in {time.time()-t0:.0f}s")
    
    # ─── 保存 CSV ────────────────────
    if all_records:
        df_out = pd.DataFrame(all_records)
        cols = ["date", "symbol", "full_name", "model_type", "model_tag",
                "last_close", "pred_close", "pct_change", "trend", "data_bars"]
        df_out = df_out[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
        df_out.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\n{'='*50}")
        print(f"CSV saved: {CSV_PATH}")
        print(f"  Total records: {len(df_out)}")
        print(f"  By symbol:")
        for sym, grp in df_out.groupby("symbol"):
            print(f"    {sym}: {len(grp)} preds, {grp['date'].min()} ~ {grp['date'].max()}")
    else:
        print("\nNo predictions generated!")

if __name__ == "__main__":
    run_backfill()
