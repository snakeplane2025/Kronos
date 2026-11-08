#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kronos × DM Quant API Bridge
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

将 DM Quant API 的 K 线数据接入 Kronos 预测模型。

核心能力:
  1. fetch_futures_bars   — 国债期货 K 线 (T/TF/TL)
  2. fetch_equity_bars    — 权益/指数 K 线 (个股/ETF/指数)
  3. fetch_bond_bars      — 债券 K 线
  4. KronosDM             — 一站式预测器（取数→转换→预测→可视化）

用法:
  from kronos_dm_bridge import fetch_futures_bars, KronosDM

  # 独立取数
  df = fetch_futures_bars("T2606", "2026-01-01", "2026-06-15", kline_type=6)

  # 一站式预测
  predictor = KronosDM(predictor=...)  # 传入已有的 KronosPredictor
  result = predictor.predict_futures("T2606", lookback=200, pred_len=24)
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# ─── DM API 取数 ─────────────────────────────────────────────
DM_FETCHER_PATH = "/home/llzjj/quant/macro_monitor"
if DM_FETCHER_PATH not in sys.path:
    sys.path.insert(0, DM_FETCHER_PATH)

from dm_fetcher import DMFetcher


# ─── K线类型对照表 ────────────────────────────────────────────
KLINE_TYPE_MAP = {
    "1min": 1, "5min": 2, "15min": 3, "30min": 4,
    "60min": 5, "daily": 6, "weekly": 7, "monthly": 8,
}
KLINE_TYPE_NAMES = {v: k for k, v in KLINE_TYPE_MAP.items()}

# 期货主力合约（按当前时间自动推断）
# 中金所国债期货：T/TF/TL，合约月份为最近的3、6、9、12月
def _get_nearby_futures_code(symbol: str, dt: Optional[datetime] = None) -> str:
    """获取最近的主力合约代码

    Args:
        symbol: 'T' / 'TF' / 'TL'
        dt: 参考日期（默认今天）

    Returns:
        str: 合约代码，如 'T2606'
    """
    if dt is None:
        dt = datetime.now()
    # 中金所国债期货：3月、6月、9月、12月
    contract_months = [3, 6, 9, 12]
    y = dt.year
    m = dt.month
    # 找最近的一个交割月：在当月+2个月之后的第一个交割月
    target = m + 2
    for cm in contract_months:
        if cm >= target:
            cy = y if cm >= m else y + 1
            return f"{symbol}{str(cy)[-2:]}{cm:02d}"
    # 如果过了12月，推到明年3月
    return f"{symbol}{str(y+1)[-2:]}03"

# 字段名映射 ───────────────────────────────────────────────
# DM API 返回字段 → 标准 OHLCV 字段
FIELD_MAP = {
    # 期货K线（trade_date + open_price 等）
    "open_price": "open",
    "high_price": "high",
    "low_price": "low",
    "close_price": "close",
    "settlement_price": "settlement_price",
    "open_interest": "open_interest",
    "trade_date": "timestamp",
    "issue_date": "timestamp",
    # 权益K线
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "business_amount": "amount",
    "business_volume": "volume",
    "trade_volume": "volume",
    "trade_amount": "amount",
    "volume": "volume",
    "amount": "amount",
    "turnover": "amount",
    "business_balance": "amount",
    # 时间（通用）
    "data_date": "timestamp",
    "timestamp": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
}

# ─── 统一的 K线取数函数 ───────────────────────────────────────

def _parse_bars(raw, required_cols: List[str] = None) -> pd.DataFrame:
    """解析 DM API 返回的 K线数据列表为 DataFrame"""
    if required_cols is None:
        required_cols = ["open", "high", "low", "close"]

    items = raw if isinstance(raw, list) else raw.get("list", [])
    if not items:
        return pd.DataFrame()

    rows = []
    for item in items:
        row = {}
        for dm_key, std_key in FIELD_MAP.items():
            if dm_key in item and item[dm_key] is not None:
                row[std_key] = item[dm_key]
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 确保必需列存在（缺失则填充0）
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # 时间解析
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        ts_col = None
        # 尝试找时间列
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                ts_col = c
                break

    if ts_col:
        # 处理不同的时间格式
        if df[ts_col].dtype in (np.int64, np.float64):
            # 毫秒时间戳 (e.g. 1700000000000)
            # ⚠️ DM API 返回的是北京时间(CST)的时间戳，pd.to_datetime(unit='s')按UTC解析
            # 不修复的话会偏移 -8h，导致日期少1天
            if len(df) > 0 and df[ts_col].iloc[0] > 1e10:
                df[ts_col] = (pd.to_datetime(df[ts_col] / 1000, unit="s")
                              + pd.Timedelta(hours=8)).dt.normalize()
            else:
                df[ts_col] = pd.to_datetime(df[ts_col], unit="s")
        else:
            # 字符串日期 (e.g. "2026-01-02")
            df[ts_col] = pd.to_datetime(df[ts_col].astype(str).str[:19], errors="coerce")
        df = df.set_index(ts_col)
        df.index.name = "timestamp"

    # 期货API没返回volume/amount，用0填充
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = 0.0

    # 类型转换
    for col in required_cols + ["volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 按时间升序排列
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    return df


def fetch_futures_bars(
    code: str,
    start_date: str,
    end_date: str,
    kline_type: int = 6,
    fetcher: Optional[DMFetcher] = None,
) -> pd.DataFrame:
    """拉取国债期货 K 线数据

    Args:
        code: 合约代码，如 'T2606'（不带后缀）
        start_date: '2026-01-01'
        end_date: '2026-06-15'
        kline_type: 1=1分钟 2=5分钟 3=15分钟 4=30分钟 5=60分钟 6=日线
        fetcher: 可复用的 DMFetcher 实例

    Returns:
        DataFrame with columns: ['open', 'high', 'low', 'close', 'volume', 'amount']
                               index: DatetimeIndex (timestamp)
    """
    if fetcher is None:
        fetcher = DMFetcher()

    raw = fetcher._post({
        "security_id_list": [code],
        "kline_type": kline_type,
        "start_datetime": start_date,
        "end_datetime": end_date,
    }, "/dm-quant-func-service/api/v1/futures/market-data/bars")

    df = _parse_bars(raw)
    if df.empty:
        print(f"⚠️ 期货 {code} K线数据为空 ({start_date} ~ {end_date}, kline_type={kline_type})")
    return df


def fetch_equity_bars(
    code: str,
    start_date: str,
    end_date: str,
    kline_type: int = 6,
    security_category: int = 3,  # 3=指数
    fetcher: Optional[DMFetcher] = None,
) -> pd.DataFrame:
    """拉取权益/指数 K 线数据

    Args:
        code: 证券代码，如 '000300.SH'（上证50）、'000905.SH'（中证500）
              个股如 '600519.SH'
        start_date: '2026-01-01'
        end_date: '2026-06-15'
        kline_type: 1-8
        security_category: 1=股票 2=基金 3=指数
        fetcher: 可复用的 DMFetcher

    Returns:
        DataFrame with OHLCV columns
    """
    if fetcher is None:
        fetcher = DMFetcher()

    raw = fetcher._post({
        "security_category": security_category,
        "security_id_list": [code],
        "kline_type": kline_type,
        "start_datetime": start_date,
        "end_datetime": end_date,
    }, "/dm-quant-func-service/api/v1/equity/market-data/bars")

    df = _parse_bars(raw)
    if df.empty:
        print(f"⚠️ 权益 {code} K线数据为空")
    return df


def fetch_bond_bars(
    code: str,
    start_date: str,
    end_date: str,
    kline_type: int = 6,
    data_source: int = 1,  # 1=经纪商
    fetcher: Optional[DMFetcher] = None,
) -> pd.DataFrame:
    """拉取债券 K 线数据

    Args:
        code: 债券代码，如 '2400001.IB'
        start_date: 开始日期
        end_date: 结束日期
        kline_type: 1-8
        data_source: 1=经纪商 2=外汇 3=上交所 4=上固收 7=深交所
        fetcher: 可复用的 DMFetcher

    Returns:
        DataFrame with OHLCV columns
    """
    if fetcher is None:
        fetcher = DMFetcher()

    raw = fetcher._post({
        "security_id_list": [code],
        "data_source_list": [data_source],
        "kline_type": kline_type,
        "start_datetime": start_date,
        "end_datetime": end_date,
    }, "/dm-quant-func-service/api/v1/bond/market-data/bars")

    df = _parse_bars(raw)
    if df.empty:
        print(f"⚠️ 债券 {code} K线数据为空")
    return df


# ─── 连续合约数据加载（CSV + DM API 拼接） ────────────────
# 拼接连续CSV（长历史） + DM API最新行情（近期），确保 lookback 可达500

CONTINUOUS_DATA_DIR = "/home/llzjj/quant/backtraderUS/data/tickers"
FUTURES_CSV_FILES = {
    "TS": "his_data_TS.csv",
    "TF": "his_data_TF.csv",
    "T":  "his_data_T.csv",
    "TL": "his_data_TL.csv",
}


def load_continuous_futures(symbol: str, fetcher: Optional[DMFetcher] = None) -> pd.DataFrame:
    """加载连续合约数据（CSV历史 + DM API最新行情拼接）

    1. 从 tickers/his_data_{symbol}.csv 读取连续合约历史
    2. 从 DM API 获取当前主力合约最新数据
    3. 拼接去重，按时间升序排列

    Args:
        symbol: 'T' / 'TF' / 'TL' / 'TS'

    Returns:
        DataFrame with columns: [open, high, low, close, volume, amount]
                                index: DatetimeIndex
    """
    csv_path = os.path.join(CONTINUOUS_DATA_DIR, FUTURES_CSV_FILES.get(symbol, f"his_data_{symbol}.csv"))

    # 1. 读取CSV历史
    if not os.path.exists(csv_path):
        print(f"⚠️ 连续合约文件不存在: {csv_path}")
        # 回退到纯DM API
        code = _get_nearby_futures_code(symbol)
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return fetch_futures_bars(code, "2026-01-01", end, fetcher=fetcher)

    df_csv = pd.read_csv(csv_path)
    # CSV列: datetime,open,high,low,close,volume,amt,oi,symbol
    df_csv["timestamp"] = pd.to_datetime(df_csv["datetime"])
    df_csv = df_csv.rename(columns={
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume", "amt": "amount",
    })
    df_csv = df_csv[["timestamp", "open", "high", "low", "close", "volume", "amount"]]
    # ⚠️ CSV每天有3~4行（不同合约月份），按日期取成交量最大的行=主力连续
    # 先过滤掉成交量缺失的行（某些合约末行可能volume为空）
    df_csv_clean = df_csv.dropna(subset=["volume"])
    if df_csv_clean.empty:
        print(f"⚠️ {symbol} CSV主力连续构建失败（成交量全为空），回退到按close非空过滤")
        df_csv_clean = df_csv.dropna(subset=["close"])
    df_csv_clean["date_only"] = df_csv_clean["timestamp"].dt.date
    
    # 同一天多个合约 → 取成交量最大的（主力）
    df_csv = df_csv_clean.loc[
        df_csv_clean.groupby("date_only")["volume"].idxmax().dropna()
    ].drop(columns=["date_only"])
    df_csv = df_csv.set_index("timestamp")
    df_csv.index.name = "timestamp"
    # 类型转换
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df_csv[c] = pd.to_numeric(df_csv[c], errors="coerce")
    df_csv = df_csv.sort_index()

    # 2. 取DM API最新数据（覆盖CSV可能缺失的近期）
    csv_last_date = df_csv.index[-1]
    now = datetime.now()
    # 16:15 cron 在收盘后运行，取当日数据；09:00 cron 用昨日数据
    # 判断逻辑：当前时间 >= 16:00 则用今天，否则用昨天
    today = now
    if now.hour < 16:
        dm_end = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        dm_end = now.strftime("%Y-%m-%d")
    dm_start = (csv_last_date - timedelta(days=5)).strftime("%Y-%m-%d")

    code = _get_nearby_futures_code(symbol)
    df_dm = fetch_futures_bars(code, dm_start, dm_end, fetcher=fetcher)

    # 3. 拼接去重
    if not df_dm.empty:
        # 只保留与CSV一致的列
        keep_cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in keep_cols:
            if c not in df_dm.columns:
                df_dm[c] = 0.0
        df_dm = df_dm[keep_cols]

        # 只取CSV最后日期之后的数据
        df_dm_new = df_dm[df_dm.index > csv_last_date]
        df_all = pd.concat([df_csv, df_dm_new])
        # 去重（以timestamp为唯一键）
        df_all = df_all[~df_all.index.duplicated(keep="last")]
        df_all = df_all.sort_index()
    else:
        df_all = df_csv

    print(f"   连续合约 {symbol}: CSV {len(df_csv)}行 + DM补充 {len(df_dm) if not df_dm.empty else 0}行"
          f" = 合计 {len(df_all)}行 ({df_all.index[0].date()} ~ {df_all.index[-1].date()})")
    return df_all


# ─── Kronos × DM 一站式预测器 ──────────────────────────────

class KronosDM:
    """Kronos + DM API 一站式预测器

    用法:
        predictor = KronosDM(kronos_predictor=my_predictor)

        # 预测国债期货
        result = predictor.predict_futures("T2606", lookback=200, pred_len=24)

        # 预测指数
        result = predictor.predict_index("000300.SH", lookback=200, pred_len=5)
    """

    def __init__(self, kronos_predictor, fetcher: Optional[DMFetcher] = None):
        """
        Args:
            kronos_predictor: KronosPredictor 实例（已加载模型和tokenizer）
            fetcher: DMFetcher 实例（自动创建）
        """
        self.kp = kronos_predictor
        self.fetcher = fetcher or DMFetcher()

    def _prepare_data(
        self, df: pd.DataFrame, lookback: int, pred_len: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """将原始 DataFrame 切割为 Kronos 输入格式

        Args:
            df: 原始K线DataFrame（index=DatetimeIndex, columns含open/high/low/close）
            lookback: 输入长度（建议 ≤ 512），取最后 lookback 行
            pred_len: 预测步数

        Returns:
            (x_df, x_timestamp, y_timestamp)
            x_timestamp: 最后 lookback 行的时间戳
            y_timestamp: 从最后一个时间点向后推 pred_len 步的未来时间戳
        """
        if df.empty or len(df) < lookback + 1:
            raise ValueError(
                f"数据不足: 需要至少 {lookback + 1} 行, "
                f"实际 {len(df)} 行"
            )

        cols = ["open", "high", "low", "close"]
        # 补齐 volume / amount
        if "volume" not in df.columns:
            df["volume"] = 0.0
        if "amount" not in df.columns:
            df["amount"] = 0.0

        # 取最后 lookback 行作为输入
        x_df = df.iloc[-lookback:][cols + ["volume", "amount"]]
        x_ts = x_df.index.to_series().reset_index(drop=True)

        # 生成未来时间戳
        last_idx = df.index[-1]
        # 判断频率：日线用工作日，其他用检测到的频率
        if lookback >= 2:
            gap = (df.index[-1] - df.index[-2]).total_seconds()
        else:
            gap = 86400

        if 0.9 * 86400 < gap < 1.1 * 86400:
            # 日线 → 用工作日，跳过周末
            y_ts = pd.bdate_range(start=last_idx + pd.Timedelta(days=1),
                                  periods=pred_len)
        else:
            # 分钟线/其他 → 用检测到的间隔
            freq = pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=int(gap)))
            y_ts = pd.date_range(start=last_idx + pd.Timedelta(seconds=int(gap)),
                                 periods=pred_len, freq=freq)
        y_ts = pd.Series(y_ts).reset_index(drop=True)

        return x_df, x_ts, y_ts

    def predict_from_df(
        self,
        df: pd.DataFrame,
        lookback: int = 200,
        pred_len: int = 24,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """从已有 DataFrame 运行 Kronos 预测

        Args:
            df: OHLCV DataFrame (index=DatetimeIndex)
            lookback: 输入长度
            pred_len: 预测步数
            T: 采样温度
            top_p: nucleus sampling
            sample_count: 采样次数（取平均）
            verbose: 显示进度

        Returns:
            pred_df: 预测结果 DataFrame (columns: open/high/low/close/volume/amount)
        """
        x_df, x_ts, y_ts = self._prepare_data(df, lookback, pred_len)

        if verbose:
            print(f"  📊 输入: {lookback} 步 → 预测: {pred_len} 步")
            print(f"      时间范围: {x_ts.iloc[0]} ~ {y_ts.iloc[-1]}")

        pred_df = self.kp.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        return pred_df

    def predict_futures(
        self,
        code: str,
        lookback: int = 200,
        pred_len: int = 24,
        kline_type: int = 6,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """拉取期货K线 → 预测

        Args:
            code: 合约代码，如 'T2606'
            lookback: 输入长度
            pred_len: 预测步数
            kline_type: K线类型 (1-8)
            start_date: 数据起始日期（自动推算）
            end_date: 数据结束日期（默认今天-1）
            T, top_p, sample_count: 预测参数
            verbose: 显示进度

        Returns:
            pred_df: 预测结果
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        # 自动推算起始日期：根据kline_type和需要的行数
        needed_days = self._estimate_days_needed(kline_type, lookback + pred_len + 20)
        if start_date is None:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_dt - timedelta(days=needed_days)).strftime("%Y-%m-%d")

        if verbose:
            print(f"🔍 取数: {code} ({KLINE_TYPE_NAMES.get(kline_type, f'type{kline_type}')})")
            print(f"       {start_date} ~ {end_date}")

        df = fetch_futures_bars(code, start_date, end_date, kline_type, self.fetcher)
        if df.empty:
            print(f"❌ 无法获取 {code} 的K线数据")
            return pd.DataFrame()

        if verbose:
            print(f"      实际数据: {len(df)} 行 ({df.index[0].date()} ~ {df.index[-1].date()})")

        return self.predict_from_df(df, lookback, pred_len, T, top_p, sample_count, verbose)

    def predict_index(
        self,
        code: str = "000300.SH",
        lookback: int = 200,
        pred_len: int = 5,
        kline_type: int = 6,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """指数K线预测

        Args:
            code: 指数代码，默认沪深300 '000300.SH'
                  也可用 '000905.SH'（中证500）、'000016.SH'（上证50）
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        needed_days = self._estimate_days_needed(kline_type, lookback + pred_len + 20)
        if start_date is None:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_dt - timedelta(days=needed_days)).strftime("%Y-%m-%d")

        if verbose:
            name = {"000300.SH": "沪深300", "000905.SH": "中证500",
                    "000016.SH": "上证50", "000688.SH": "科创50"}.get(code, code)
            print(f"🔍 取数: {name} ({code})")
            print(f"       {start_date} ~ {end_date}")

        df = fetch_equity_bars(code, start_date, end_date, kline_type, fetcher=self.fetcher)
        if df.empty:
            print(f"❌ 无法获取 {code} 的K线数据")
            return pd.DataFrame()

        return self.predict_from_df(df, lookback, pred_len, T, top_p, sample_count, verbose)

    def predict_bond(
        self,
        code: str,
        lookback: int = 200,
        pred_len: int = 24,
        kline_type: int = 6,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """债券K线预测

        Args:
            code: 债券代码，如 '2400001.IB'（需查询活跃券列表获取最新代码）
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        needed_days = self._estimate_days_needed(kline_type, lookback + pred_len + 20)
        if start_date is None:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_dt - timedelta(days=needed_days)).strftime("%Y-%m-%d")

        if verbose:
            print(f"🔍 取数: 债券 {code}")
            print(f"       {start_date} ~ {end_date}")

        df = fetch_bond_bars(code, start_date, end_date, kline_type, fetcher=self.fetcher)
        if df.empty:
            print(f"❌ 无法获取 {code} 的K线数据")
            return pd.DataFrame()

        return self.predict_from_df(df, lookback, pred_len, T, top_p, sample_count, verbose)

    @staticmethod
    def _estimate_days_needed(kline_type: int, n_rows: int) -> int:
        """根据K线类型估算需要多少天才能获得 n_rows 条数据"""
        multipliers = {
            1: 1,    # 1min → 每天240条, 1天足够
            2: 2,    # 5min → 每天48条
            3: 3,    # 15min → 每天16条
            4: 5,    # 30min → 每天8条
            5: 10,   # 60min → 每天4条
            6: int(n_rows * 1.7),  # 日线→ 乘以1.7倍（跳过周末）
            7: int(n_rows * 10),   # 周线
            8: int(n_rows * 30),   # 月线
        }
        return max(multipliers.get(kline_type, n_rows), 30)

    @staticmethod
    def compare(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
        """比较预测值与实际值"""
        common_idx = pred_df.index.intersection(actual_df.index)
        if len(common_idx) == 0:
            print("⚠️ 预测与实际无重叠时间")
            return pd.DataFrame()

        comparison = pd.DataFrame({
            "pred_close": pred_df.loc[common_idx, "close"],
            "actual_close": actual_df.loc[common_idx, "close"],
        })
        comparison["error"] = comparison["pred_close"] - comparison["actual_close"]
        comparison["error_pct"] = (comparison["error"] / comparison["actual_close"] * 100)
        comparison["direction"] = np.where(
            comparison["pred_close"] > comparison["actual_close"], "🟢偏多", "🔴偏空"
        )
        return comparison

    @staticmethod
    def print_summary(pred_df: pd.DataFrame, label: str = "预测结果"):
        """打印预测摘要"""
        if pred_df.empty:
            print(f"❌ {label}: 无数据")
            return
        print(f"\n📋 {label}")
        print(f"    时间范围: {pred_df.index[0]} ~ {pred_df.index[-1]}")
        print(f"    预测步数: {len(pred_df)}")
        print(f"    收盘价范围: {pred_df['close'].min():.4f} ~ {pred_df['close'].max():.4f}")
        print(f"    均价: {pred_df['close'].mean():.4f}")
        print(f"    最新预测收盘: {pred_df['close'].iloc[-1]:.4f}")
        print(f"    趋势: {'📈 上涨' if pred_df['close'].iloc[-1] > pred_df['close'].iloc[0] else '📉 下跌'}")


# ─── 活跃券查询（辅助工具） ────────────────────────────────

def get_active_bond_codes(
    bond_filter_type: int = 1,  # 1=国债 3=国开
    key_tenor: str = "10Y",
    sequence_type: int = 1,     # 1=活跃 2=次活跃
    fetcher: Optional[DMFetcher] = None,
) -> List[str]:
    """获取当前活跃券/次活跃券代码

    Args:
        bond_filter_type: 1=国债 3=国开
        key_tenor: '1Y'/'3Y'/'5Y'/'7Y'/'10Y'/'30Y'/'50Y'
        sequence_type: 1=活跃 2=次活跃
        fetcher: DMFetcher

    Returns:
        [security_id, ...]
    """
    if fetcher is None:
        fetcher = DMFetcher()

    raw = fetcher._post({
        "sequence_type": sequence_type,
        "bond_filter_type": bond_filter_type,
        "key_tenor": key_tenor,
        "start_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        "end_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
    }, "/dm-quant-func-service/api/v1/bond/market-data/rolling-bonds")

    items = raw if isinstance(raw, list) else raw.get("list", [])
    codes = []
    for item in items:
        sid = item.get("security_id") or item.get("security_code")
        if sid and sid not in codes:
            codes.append(sid)
    return codes


# ─── 快捷入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    # 测试：获取期货合约代码并验证取数
    print("=" * 60)
    print("Kronos × DM API Bridge — 快速验证")
    print("=" * 60)

    # 测试主力合约推断
    test_dt = datetime.now()
    for sym in ["T", "TF", "TL"]:
        code = _get_nearby_futures_code(sym, test_dt)
        print(f"  {sym} 主力合约: {code}")

    # 测试取数
    fetcher = DMFetcher()
    for sym, code in [("T", _get_nearby_futures_code("T"))]:
        print(f"\n  📡 测试 {sym} ({code}) 日线取数...")
        df = fetch_futures_bars(code, "2026-01-01", "2026-06-15", fetcher=fetcher)
        if not df.empty:
            print(f"     ✅ {len(df)} 行数据: {df.index[0].date()} ~ {df.index[-1].date()}")
            print(f"     最新: O={df['open'].iloc[-1]:.4f} H={df['high'].iloc[-1]:.4f} "
                  f"L={df['low'].iloc[-1]:.4f} C={df['close'].iloc[-1]:.4f}")
        else:
            print(f"     ❌ 取数失败（可能非交易日或合约未上市）")
