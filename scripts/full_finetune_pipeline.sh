#!/bin/bash
# Kronos 国债期货微调全流程 — epochs 15
set -e

EPOCHS=15
cd /home/llzjj/quant/Kronos
source venv/bin/activate

echo "========================================"
echo "🚀 Kronos 国债期货微调全流程 (epochs=${EPOCHS})"
echo "DATE: $(date)"
echo "========================================"

# Step 1: 全品种训练 (TS→TF→T→TL)
echo ""
echo "=== [1/2] 全品种 ${EPOCHS}epoch 训练 ==="
echo "Start: $(date)"
python3 scripts/finetune_pipeline.py train --epochs ${EPOCHS}
echo "End: $(date)"

# Step 2: 全品种回测 + 报告
echo ""
echo "=== [2/2] 全品种回测（原始 vs 微调对比）+ 报告 ==="
echo "Start: $(date)"
python3 scripts/finetune_pipeline.py backtest
echo "End: $(date)"

echo ""
echo "========================================"
echo "✅ 全流程完成!"
ls -t /home/llzjj/quant/Kronos/reports/finetune_report_*.md | head -1
echo "========================================"
