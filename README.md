# Kronos + DM Quant API — 本地部署与实战指南

> **版本**: v1.0 | **更新**: 2026-06-16
> **项目**: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) (AAAI 2026)
> **模型**: Kronos-base (102.3M) | **数据**: DM Quant API

---

## 目录

1. [项目概览](#1-项目概览)
2. [环境与安装](#2-环境与安装)
3. [模型下载与加载](#3-模型下载与加载)
4. [DM API 数据取数](#4-dm-api-数据取数)
5. [Kronos × DM 桥梁架构](#5-kronos--dm-桥梁架构)
6. [API 参考](#6-api-参考)
7. [实战案例](#7-实战案例)
8. [坑点与注意事项](#8-坑点与注意事项)
9. [错误排查](#9-错误排查)

---

## 1. 项目概览

### 1.1 Kronos 是什么

Kronos 是**首个开源金融K线基础模型**（AAAI 2026），采用 **decoder-only Transformer** 架构，在 **45+全球交易所** 的 K 线数据上预训练。核心创新是**两阶段框架**：

```
原始OHLCV数据
    │
    ▼
┌─────────────────┐
│  量化 Tokenizer  │  ← 将高维OHLCV量化为层次离散token (s1/s2)
│  BSQuantizer     │     s1 = 粗粒度codebook
│                  │     s2 = 细粒度codebook  
└────────┬────────┘
         │
         ▼ token序列
┌─────────────────┐
│  Autoregressive  │  ← Transformer在token序列上预训练
│  Transformer     │     预测下一个token = 预测未来价格走势
└────────┬────────┘
         │
         ▼
    OHLCV预测输出
```

### 1.2 模型系列

| 模型 | 参数量 | 上下文长度 | 适用场景 |
|------|--------|-----------|---------|
| Kronos-mini | **4.1M** | 2048 | 快速验证、低资源设备 |
| Kronos-small | **24.7M** | 512 | 轻量级预测 |
| **Kronos-base** ⭐ | **102.3M** | 512 | **推荐，精度与速度平衡** |
| Kronos-large | 499.2M | 512 | ❌ 未开源 |

### 1.3 系统架构

```
┌────────────────────────────────────────────────────────┐
│                    用户入口                               │
│  CLI: kronos_dm_predict.py  │  Python API: KronosDM    │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│              kronos_dm_bridge.py (桥梁层)                │
│                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ 期货K线取数  │  │ 指数K线取数  │  │ 债券K线取数  │    │
│  │ futures_bars│  │ equity_bars │  │ bond_bars   │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │               │               │            │
│         └───────────────┼───────────────┘            │
│                         ▼                            │
│              ┌──────────────────┐                    │
│              │  KronosPredictor  │  ← 归一化+推理+反归一化│
│              └────────┬─────────┘                    │
└───────────────────────┼──────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌──────────────────┐     ┌──────────────────────┐
│  Kronos-base     │     │  DM Quant API        │
│  102.3M params   │     │  东吴数聚             │
│  (HuggingFace)   │     │  15+接口             │
└──────────────────┘     └──────────────────────┘
```

---

## 2. 环境与安装

### 2.1 目录结构

```
~/quant/Kronos/
├── venv/                          # 独立 Python 虚拟环境
├── models/
│   ├── Kronos-base/               # 模型权重 (102.3M params)
│   │   ├── model.safetensors      # ~410MB
│   │   ├── config.json
│   │   └── ...
│   └── Kronos-Tokenizer-base/     # 量化 Tokenizer
│       ├── model.safetensors
│       ├── config.json
│       └── ...
├── model/
│   ├── __init__.py                # 包入口
│   ├── kronos.py                  # Kronos/KronosTokenizer/KronosPredictor
│   └── module.py                  # 网络层定义
├── kronos_dm_bridge.py            # ⭐ DM API 桥梁（核心集成代码）
├── examples/
│   ├── kronos_dm_predict.py       # ⭐ CLI 预测工具
│   ├── prediction_example.py      # 官方示例（需数据文件）
│   └── prediction_wo_vol_example.py
├── requirements.txt
└── README.md                      # 本文件
```

### 2.2 安装步骤

```bash
# 1. 克隆仓库（已完成）
cd ~/quant
git clone https://github.com/shiyu-coder/Kronos.git

# 2. 创建虚拟环境（已完成）
cd Kronos
python3 -m venv venv

# 3. 安装依赖
source venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
uv pip install huggingface_hub==0.33.1 pandas==2.2.2 einops==0.8.1

# 4. 安装 DM API 客户端
uv pip install "/home/llzjj/quant/backtraderUS/dm api/dm_quant_api_client-0.2.2-py3-none-any.whl"
```

### 2.3 网络限制（国内环境）

由于国内无法直接访问 `huggingface.co`，下载模型需使用镜像：

```bash
# 方式一：环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 方式二：每命令前缀
HF_ENDPOINT=https://hf-mirror.com python3 your_script.py

# 验证
curl -I https://hf-mirror.com  # 应返回 200
```

---

## 3. 模型下载与加载

### 3.1 下载命令

```python
from huggingface_hub import snapshot_download
import os

# 下载模型（约 410MB）
snapshot_download(
    repo_id="NeoQuasar/Kronos-base",
    local_dir="models/Kronos-base"
)

# 下载 Tokenizer（约 85MB）
snapshot_download(
    repo_id="NeoQuasar/Kronos-Tokenizer-base",
    local_dir="models/Kronos-Tokenizer-base"
)
```

### 3.2 加载模型

```python
import sys
sys.path.insert(0, '.')  # 确保能找到 model 包

from model import KronosTokenizer, Kronos, KronosPredictor

# 加载 Tokenizer
tokenizer = KronosTokenizer.from_pretrained("models/Kronos-Tokenizer-base")

# 加载模型
model = Kronos.from_pretrained("models/Kronos-base")
model.eval()

# 创建 Predictor（自动检测 CPU/CUDA）
predictor = KronosPredictor(
    model, tokenizer,
    device="cpu",          # 或 "cuda:0"
    max_context=512        # base/small 模型上限
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 3.3 模型参数量验证

| 模型 | 加载后确认输出 |
|------|---------------|
| Kronos-small | `24,717,248 params` |
| Kronos-base | `102,310,592 params` |

---

## 4. DM API 数据取数

### 4.1 凭证配置

DM Quant API 凭证存储在 `~/quant/backtraderUS/.env`：

```bash
# .env 格式（必须是 Unix LF！）
INNO_APP_KEY=your_app_key_here
INNO_SM4_KEY=your_sm4_key_here
```

**⚠️ CRLF 陷阱**：Windows 编辑过的 `.env` 会携带 `\r` 导致请求失败，用以下命令修复：
```bash
sed -i 's/\r$//' ~/quant/backtraderUS/.env
```

### 4.2 可用 K 线接口

| 接口 | API 路径 | 支持类型 | 代码格式 |
|------|---------|---------|---------|
| **期货K线** | `/futures/market-data/bars` | T/TF/TL | `"T2606"`（**无**后缀） |
| **权益K线** | `/equity/market-data/bars` | 股票/基金/指数 | `"000300.SH"`（带后缀） |
| **债券K线** | `/bond/market-data/bars` | 国债/信用债 | `"2400001.IB"`（带后缀） |

### 4.3 K 线类型 (kline_type)

| 值 | 类型 | 每天约产生 | 跨度限制 |
|----|------|-----------|---------|
| 1 | 1分钟 | 240条 | ≤ 1天 |
| 2 | 5分钟 | 48条 | ≤ 5天 |
| 3 | 15分钟 | 16条 | ≤ 15天 |
| 4 | 30分钟 | 8条 | ≤ 1月 |
| 5 | 60分钟 | 4条 | ≤ 2月 |
| **6** ⭐ | **日线** | 1条 | ≤ **1年** |
| 7 | 周线 | 0.2条 | ≤ 5年 |
| 8 | 月线 | 0.05条 | ≤ 10年 |

### 4.4 自动主力合约推断

```python
from kronos_dm_bridge import _get_nearby_futures_code

# 当前时间 → 自动推断最近的主力合约
code = _get_nearby_futures_code('T')     # → 'T2609' (2026-09月合约)
code = _get_nearby_futures_code('TF')    # → 'TF2609'
code = _get_nearby_futures_code('TL')    # → 'TL2609'
```

逻辑：中金所国债期货的交割月为 **3月、6月、9月、12月**。函数取"当前月+2个月后的下一个交割月"。

### 4.5 活跃券查询

```python
from kronos_dm_bridge import get_active_bond_codes

# 获取10Y国债活跃券
active_10y = get_active_bond_codes(
    bond_filter_type=1,  # 1=国债, 3=国开
    key_tenor="10Y",
    sequence_type=1      # 1=活跃, 2=次活跃
)
# → ['2400001.IB', ...]
```

---

## 5. Kronos × DM 桥梁架构

### 5.1 数据流

```
DM API 原始响应
    │
    ▼
┌────────────────────────────┐
│  _parse_bars()              │  字段名映射 + 时间解析
│  • trade_date → DatetimeIndex │  + 缺失列填充
│  • open_price → open        │
│  • ...                      │
└──────────┬─────────────────┘
           ▼
DataFrame (index=DatetimeIndex, columns=open/high/low/close/volume/amount)
           │
    lookback 步          pred_len 步
    ├──────┼────────┼────────┤
    │ 输入  │  预测   │        │
    │ x_df  │ y_true  │        │
    └──────┴────────┴────────┘
           │
           ▼
┌────────────────────────────┐
│  KronosPredictor.predict()  │
│  • z-score 归一化           │
│  • tokenizer.encode()       │  量化→token
│  • 自回归推理 (pred_len步)   │  Transformer解码
│  • tokenizer.decode()       │  token→OHLCV
│  • 反归一化                  │
└──────────┬─────────────────┘
           ▼
    预测结果 DataFrame
```

### 5.2 数据格式要求

**输入 DataFrame**（`predict_from_df` 要求）：
- Index: `DatetimeIndex`（时间升序）
- Columns: `['open', 'high', 'low', 'close']`（必需）
- Columns: `['volume', 'amount']`（可选，缺失自动填0）
- 无 NaN 值

**输入长度限制**：
- `lookback` ≤ `max_context`（base/small=512）
- 总数据量需要 ≥ `lookback + pred_len`

### 5.3 预测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `T` (temperature) | 1.0 | 采样温度。越低越确定，越高越多样 |
| `top_p` | 0.9 | Nucleus sampling，保留累积概率 top_p 的 token |
| `sample_count` | 5 | 采样次数，最终取平均。越高越稳定 |
| `pred_len` | — | 预测步数，由用户指定 |

---

## 6. API 参考

### 6.1 独立取数函数

#### `fetch_futures_bars(code, start_date, end_date, kline_type=6, fetcher=None)`
**用途**：拉取国债期货 K 线（T/TF/TL）
**返回**：`pd.DataFrame` (index=DatetimeIndex)
```python
df = fetch_futures_bars("T2609", "2026-01-01", "2026-06-15", kline_type=6)
```

#### `fetch_equity_bars(code, start_date, end_date, kline_type=6, security_category=3, fetcher=None)`
**用途**：拉取指数/股票 K 线
**返回**：`pd.DataFrame`
```python
df = fetch_equity_bars("000300.SH", "2026-01-01", "2026-06-15", kline_type=6)
```

#### `fetch_bond_bars(code, start_date, end_date, kline_type=6, data_source=1, fetcher=None)`
**用途**：拉取债券 K 线
**返回**：`pd.DataFrame`
```python
df = fetch_bond_bars("2400001.IB", "2026-01-01", "2026-06-15", kline_type=6)
```

### 6.2 KronosDM 类

#### 初始化
```python
from model import KronosTokenizer, Kronos, KronosPredictor
from kronos_dm_bridge import KronosDM

tokenizer = KronosTokenizer.from_pretrained("models/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("models/Kronos-base")
model.eval()
kp = KronosPredictor(model, tokenizer, device="cpu")

dm = KronosDM(kp)
```

#### `predict_from_df(df, lookback=200, pred_len=24, T=1.0, top_p=0.9, sample_count=5, verbose=True)`
从已有 DataFrame 预测。
```python
pred = dm.predict_from_df(df, lookback=200, pred_len=5)
```

#### `predict_futures(code, lookback=200, pred_len=24, kline_type=6, ...)`
取期货 K 线 → 预测，全自动。
```python
pred = dm.predict_futures("T2609", lookback=200, pred_len=5)
```

#### `predict_index(code="000300.SH", lookback=200, pred_len=5, kline_type=6, ...)`
取指数 K 线 → 预测。
```python
pred = dm.predict_index("000300.SH", lookback=200, pred_len=5)
```

#### `predict_bond(code, lookback=200, pred_len=24, kline_type=6, ...)`
取债券 K 线 → 预测。
```python
pred = dm.predict_bond("2400001.IB", lookback=200, pred_len=10)
```

#### `compare(pred_df, actual_df)`
比较预测值与实际值。
```python
comparison = dm.compare(pred_df, actual_df)
print(comparison[["pred_close", "actual_close", "error_pct"]])
```

#### `print_summary(pred_df, label="预测结果")`
打印预测摘要。

---

## 7. 实战案例

### 7.1 命令行：预测 T 主力期货

```bash
cd ~/quant/Kronos
source venv/bin/activate

# 预测 T主力 日线，未来5个交易日
python3 examples/kronos_dm_predict.py \
    -t futures -s T \
    -b 200 -p 5 \
    --samples 5
```

输出示例：
```
🤖 加载 Kronos-base...
   ✅ Kronos-base: 102,310,592 params | device=cpu

📈 日线 预测
🎯 期货: T (T2609)
🔍 取数: T2609 (daily) | 2025-11-18 ~ 2026-06-15
      实际数据: 119 行

📊 输入: 100 步 → 预测: 5 步
      时间范围: 2026-01-04 ~ 2026-06-11

📋 预测结果
    最新预测收盘: 109.1562
    趋势: 📈 上涨
```

### 7.2 Python API：全流程

```python
import sys
sys.path.insert(0, '.')

from model import KronosTokenizer, Kronos, KronosPredictor
from kronos_dm_bridge import (
    KronosDM,
    _get_nearby_futures_code,
    fetch_futures_bars,
)

# 1. 加载模型
tok = KronosTokenizer.from_pretrained("models/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("models/Kronos-base")
model.eval()
kp = KronosPredictor(model, tok, device="cpu")
print(f"模型加载完成: {sum(p.numel() for p in model.parameters()):,} params")

# 2. 创建 DM 桥梁
dm = KronosDM(kp)

# 3. 获取数据 + 预测（一站式）
code = _get_nearby_futures_code("T")
pred = dm.predict_futures(code, lookback=200, pred_len=5, verbose=True)

# 4. 查看结果
print(pred[["open", "high", "low", "close"]].round(4))

# 5. 获取实际数据做对比
df = fetch_futures_bars(code, "2026-01-01", "2026-06-15")
print(dm.compare(pred, df).round(4))
```

### 7.3 5分钟K线预测

```python
# 取5分钟K线，前400根输入，预测后24根（=2小时）
pred = dm.predict_futures(
    code="T2609",
    lookback=400,
    pred_len=24,
    kline_type=2,     # 5分钟K线
    sample_count=5,
)
```

### 7.4 批量预测多个品种

```python
for symbol in ["T", "TF", "TL"]:
    code = _get_nearby_futures_code(symbol)
    pred = dm.predict_futures(code, lookback=100, pred_len=5)
    dm.print_summary(pred, label=f"{symbol} ({code})")
```

---

## 8. 坑点与注意事项

### ⚠️ 8.1 上下文长度限制

**Kronos-base 和 Kronos-small 的 `max_context=512`**。输入的历史K线长度（`lookback`）不能超过此值。 predictor 会自动截断，但截断后的历史可能丢失重要信息。

**对策**：
- 日线预测：`lookback` 设为 200-500（≈1-2年日线）
- 分钟线预测：`lookback` 设为 200-400（剩余空间留给 `pred_len`）
- 总长度 = `lookback` + `pred_len`，但模型只看最后的 `max_context` 个 token

### ⚠️ 8.2 DM API 取数限制

| 限制 | 详情 |
|------|------|
| 跨日范围 | 日线 ≤ 1年，分钟线 ≤ 1天~2月 |
| 并发 | 无严格限制，但建议每次调用间隔 0.3s |
| 字段名 | 期货用 `trade_date`，权益用 `data_date`，**不是** `timestamp` |
| 期货代码 | **不带** `.IB`/`.SH` 后缀（区别于债券/权益） |
| 债券代码 | **必须带** `.IB` 后缀 |

### ⚠️ 8.3 HuggingFace 国内下载

```bash
# 必须使用镜像
export HF_ENDPOINT=https://hf-mirror.com
# 然后运行任何 huggingface_hub 操作
```

### ⚠️ 8.4 模型输入的时间维度

Kronos 的 `calc_time_stamps()` 会从时间戳提取 `minute/hour/weekday/day/month` 作为特征。传入的 `x_timestamp` / `y_timestamp` 必须是 **`pd.Series`**（含 datetime 值），**不能是** `DatetimeIndex` 或 `Index`。

`kronos_dm_bridge.py` 已自动处理此转换，直接传 DataFrame 即可。

### ⚠️ 8.5 资金面/货币市场数据不适用

Kronos 训练数据是 **OHLCV K 线**（全球交易所），**不是**货币市场利率或债券收益率曲线。以下数据不可用：
- ❌ DR007 / R007 / Shibor
- ❌ 收益率曲线（国债/国开）
- ❌ 机构买卖行为

以下可用：
- ✅ **国债期货** K 线 (T/TF/TL) — 最直接的应用
- ✅ **股票指数** K 线（沪深300、中证500、上证50等）
- ✅ 个股 K 线
- ✅ 债券 K 线（需有足够流动性）

---

## 9. 错误排查

### 9.1 DM API 连接问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `app_key is required` | 未加载 .env | 检查 `~/quant/backtraderUS/.env` 是否存在 |
| `CRLF detected` | .env 有 Windows 换行 | `sed -i 's/\r$//' .env` |
| `Network is unreachable` | 网络不通 | 检查 VPN/代理 |
| `quota_exhausted` / 429 | API 额度用完 | 减少调用量 |

### 9.2 模型加载问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `LocalEntryNotFoundError` | 模型未下载 | 用 `HF_ENDPOINT=https://hf-mirror.com` 下载 |
| `CUDA out of memory` | GPU 显存不足 | 改用 `device="cpu"` |
| `AttributeError: 'int' object has no attribute 'date'` | 时间戳未正确解析 | 确认数据包含 `trade_date` 字段 |

### 9.3 预测问题

| 现象 | 原因 | 解决 |
|------|------|------|
| 预测值全部相同 | `T`（temperature）过低或 `sample_count`=1 | 设 `T=1.0`, `sample_count≥5` |
| 预测值严重偏离 | 数据包含 NaN | 检查输入数据是否有空值 |
| 推理极慢 | CPU 运行大模型 | 每分钟线推理约 0.5s/token，日线约 0.15s/token |
| 返回空 DataFrame | 数据不足 | 需要 ≥ `lookback + pred_len` 行 |

### 9.4 性能基准

| 场景 | 推理速度 (CPU) | 推理速度 (估计 CUDA) |
|------|---------------|-------------------|
| Kronos-base, lookback=200, pred_len=5 (日线) | ~0.8s | ~0.05s |
| Kronos-base, lookback=400, pred_len=24 (5分钟) | ~4.5s | ~0.3s |
| Kronos-small, lookback=200, pred_len=5 | ~0.3s | ~0.02s |

---

## 附录

### A. DM API 关键端点速查

| 路径 | 用途 | 日期参数名 |
|------|------|-----------|
| `/futures/market-data/bars` | 期货K线 | `start_datetime` / `end_datetime` |
| `/equity/market-data/bars` | 权益K线 | `start_datetime` / `end_datetime` |
| `/bond/market-data/bars` | 债券K线 | `start_datetime` / `end_datetime` |
| `/bond/market-data/rolling-bonds` | 活跃券 | `start_date` / `end_date` |

### B. 文件速查

| 文件 | 关键内容 |
|------|---------|
| `~/quant/Kronos/kronos_dm_bridge.py` | 核心桥梁，~440行 |
| `~/quant/Kronos/examples/kronos_dm_predict.py` | CLI入口，~160行 |
| `~/quant/Kronos/model/kronos.py` | Kronos 模型定义，~662行 |
| `~/quant/backtraderUS/.env` | DM API 凭证 |
| `~/quant/macro_monitor/dm_fetcher.py` | DM API 封装取数层 |

### C. 快速启动

```bash
# 1. 激活环境
cd ~/quant/Kronos && source venv/bin/activate

# 2. 启动预测（T主力，日线，未来5天）
python3 examples/kronos_dm_predict.py -t futures -s T -b 200 -p 5

# 3. Python交互式
python3 -c "
from model import *
from kronos_dm_bridge import *
kp = KronosPredictor(
    Kronos.from_pretrained('models/Kronos-base').eval(),
    KronosTokenizer.from_pretrained('models/Kronos-Tokenizer-base')
)
dm = KronosDM(kp)
pred = dm.predict_futures(_get_nearby_futures_code('T'), 200, 5)
print(pred[['close']])
"
```
