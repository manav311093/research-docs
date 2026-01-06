"""
SMA vs price trigger based
"""

import polars as pl
import polars.selectors as cs
import numpy as np
import datetime
import sys
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

from sklearn.model_selection import train_test_split # For general splitting, but we'll do chronological
from sklearn.metrics import precision_recall_curve
from joblib import dump, load

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

pl.Config(tbl_rows=30)

# Tick DF Schema ---
tick_schema_dict = {
    "timestamp": pl.Datetime(time_unit="us"),
    "instrument_token": pl.UInt32,
    "last_price": pl.Float64,
    "last_quantity": pl.UInt32, # Quantity traded in the last tick
    "volume": pl.UInt32, # Cumulative daily volume
    "buy_quantity": pl.UInt32, # Total open buy orders (all levels)
    "sell_quantity": pl.UInt32, # Total open sell orders (all levels)
    "last_trade_time": pl.Datetime(time_unit="us")
}

TRAIN_DAYS = [    
    "2025-05-08", "2025-05-09",
    "2025-05-26", "2025-05-29", "2025-05-23", "2025-05-20", "2025-05-12", "2025-05-14", "2025-05-13", "2025-05-15",
    "2025-05-21", "2025-05-19", "2025-05-22", "2025-05-27", "2025-05-30", "2025-06-02", "2025-06-03", "2025-06-04",
    "2025-06-05", "2025-06-06", "2025-06-09", "2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-16",
    "2025-06-17", "2025-06-19", "2025-06-20", "2025-06-23", "2025-11-04", "2025-11-06", "2025-11-07", "2025-11-10",
    "2025-11-11", "2025-11-12", "2025-11-13", "2025-11-14", "2025-11-17", "2025-11-19", "2025-11-20", "2025-11-21",
    "2025-11-24", "2025-11-25", "2025-11-26", "2025-11-27", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04",
    "2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18", "2025-07-21", "2025-07-22", "2025-07-23",
    "2025-07-24", "2025-07-25", "2025-07-28", "2025-07-29", "2025-07-30", "2025-07-31", "2025-08-04", "2025-08-05",
    "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14", "2025-08-18",
    "2025-08-19", "2025-08-22", "2025-08-25", "2025-08-26", "2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04",
    "2025-09-05", "2025-09-08", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-22", "2025-09-23", "2025-09-24",
    "2025-09-25", "2025-09-26", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-03", "2025-10-06", "2025-10-07",
    "2025-10-08", "2025-10-09", "2025-10-13", "2025-10-14", "2025-10-15", "2025-10-16", "2025-10-17", "2025-10-22",
    "2025-10-23", "2025-10-24", "2025-10-27", "2025-10-28", "2025-10-29", "2025-10-30", "2025-12-01", "2025-12-02",
    "2025-12-03", "2025-12-04", "2025-12-05", "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12",
    "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19", "2025-12-23", "2025-12-24", 
    
]
TEST_DAYS = [
    "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19", "2025-12-23", "2025-12-24", 
    "2025-12-29", "2025-12-30", "2025-12-31",
    '2026-01-01', '2026-01-02', '2026-01-05'
]
TRAIN_MODEL = True
SHOW_PLOT = not True
SHOW_HISTORY = not True

PROBABILITY_THRESHOLD = 0.55
TOKEN = 738561
PP = 0.0028
VOL_LOW = 0.0002
VOL_UP = 0.0006
RSI_LOW = 49
RSI_UP = 51
OP_SIDE_ACC_MAX_SLOPE = 0
QTY = 1

EXIT_ON_REV_SIGNAL = False
OPEN_POS_AFTER_REV_SIGNAL_EXIT = False

MARKET_DATA_HOME_PATH='/Users/manav/Projects/hope/hmd'

CONFIGS = [
    {
        'instrument_token': TOKEN,
        "profit_target_pct": pp,
        "initial_stop_loss_pct": PP-0.0012,
        "trailing_stop_pct": 0.0045,
        "RSI_LOW": rsi_low,
        "RSI_UP": rsi_up,
        "VOL_LOW": vol_low,
        "VOL_UP": vol_up,
        "TIMEFRAME": "1m",
    } for pp in [PP]
      for rsi_low, rsi_up in [(RSI_LOW, RSI_UP)]
      for vol_up in [VOL_UP]
      for vol_low in [VOL_LOW]
]
CONFIG = CONFIGS[0]

def add_rsi_indicator(
    df: pl.DataFrame,
    rsi_period: int = 14,
    price_col: str = "last_price"
) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(f"rsi")
        )

    # Ensure DataFrame is sorted by instrument and timestamp for correct calculations
    df = df.sort("instrument_token", "timestamp")

    df = df.with_columns(
        pl.col(price_col).diff(1).over("instrument_token").alias("_price_change")
    )
    df = df.with_columns(
        pl.when(pl.col("_price_change") > 0)
          .then(pl.col("_price_change"))
          .otherwise(0.0) # Use 0.0 to ensure float type
          .alias("_gain"),
        
        pl.when(pl.col("_price_change") < 0)
          .then(-pl.col("_price_change")) # Loss as a positive value
          .otherwise(0.0)
          .alias("_loss")
    )
    avg_gain_col = f"_avg_gain_{rsi_period}"
    avg_loss_col = f"_avg_loss_{rsi_period}"

    df = df.with_columns(
        pl.col("_gain")
          .ewm_mean(alpha=1.0/rsi_period, adjust=False, min_samples=1)
          .over("instrument_token")
          .alias(avg_gain_col),
        
        pl.col("_loss")
          .ewm_mean(alpha=1.0/rsi_period, adjust=False, min_samples=1)
          .over("instrument_token")
          .alias(avg_loss_col)
    )
    rsi_col_name = f"rsi"
    df = df.with_columns(
        pl.when(pl.col(avg_loss_col) == 0)
        .then(
            pl.when(pl.col(avg_gain_col) == 0)
            .then(pl.lit(50.0))  # Neutral RSI if no gains and no losses
            .otherwise(pl.lit(100.0)) # RSI is 100 if gains exist but no losses
        )
        .otherwise(
            # Ensure avg_gain_col is not null before division, ewm_mean handles this with min_periods
            # If avg_gain_col / avg_loss_col results in NaN (e.g. due to initial nulls before min_periods met),
            # the whole RSI expression will be null, which is correct.
            100.0 - (100.0 / (1.0 + (pl.col(avg_gain_col) / pl.col(avg_loss_col))))
        )
        .alias(rsi_col_name)
    )
    
    # Optional: Clean up intermediate columns
    df = df.drop(["_price_change", "_gain", "_loss", avg_gain_col, avg_loss_col])
    
    return df

def add_macd_indicator(
    df: pl.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    price_col: str = "last_price"
) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("macd"),
            pl.lit(None, dtype=pl.Float64).alias("signal")
        )

    # Ensure DataFrame is sorted by instrument and timestamp for correct calculations
    df = df.sort("instrument_token", "timestamp")

    ema_fast = df.select(
        pl.col(price_col)
        .ewm_mean(span=macd_fast, adjust=False, min_samples=1)
        .over("instrument_token")
        .alias("ema_fast")
    )

    ema_slow = df.select(
        pl.col(price_col)
        .ewm_mean(span=macd_slow, adjust=False, min_samples=1)
        .over("instrument_token")
        .alias("ema_slow")
    )

    df = df.with_columns(
        (ema_fast["ema_fast"] - ema_slow["ema_slow"]).alias("macd")
    )

    df = df.with_columns(
        pl.col("macd")
        .ewm_mean(span=macd_signal, adjust=False, min_samples=1)
        .over("instrument_token")
        .alias("signal")
    )

    return df

def add_bollinger_bands(
    df: pl.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    price_col: str = "last_price"
) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("bb_upper"),
            pl.lit(None, dtype=pl.Float64).alias("bb_lower"),
            pl.lit(None, dtype=pl.Float64).alias("bb_middle")
        )

    # Ensure DataFrame is sorted by instrument and timestamp for correct calculations
    df = df.sort("instrument_token", "timestamp")

    rolling_mean = df.select(
        pl.col(price_col)
        .rolling_mean(window_size=bb_period, min_samples=1)
        .over("instrument_token")
        .alias("bb_middle")
    )

    rolling_std = df.select(
        pl.col(price_col)
        .rolling_std(window_size=bb_period, min_samples=1)
        .over("instrument_token")
        .alias("bb_std_dev")
    )

    df = df.with_columns(
        (rolling_mean["bb_middle"] + bb_std * rolling_std["bb_std_dev"]).alias("bb_upper"),
        (rolling_mean["bb_middle"] - bb_std * rolling_std["bb_std_dev"]).alias("bb_lower"),
        rolling_mean["bb_middle"]
    )

    return df

def add_atr_indicator(df: pl.DataFrame, atr_period: int = 14) -> pl.DataFrame:
    # True Range (TR) is the max of:
    # 1. High - Low
    # 2. |High - Prev Close|
    # 3. |Low - Prev Close|
    df = df.with_columns([
        pl.col("price_close").shift(1).alias("prev_close")
    ]).with_columns([
        pl.max_horizontal([
            (pl.col("price_high") - pl.col("price_low")),
            (pl.col("price_high") - pl.col("prev_close")).abs(),
            (pl.col("price_low") - pl.col("prev_close")).abs()
        ]).alias("true_range")
    ])
    
    # ATR is the Wilder's Smoothing of TR
    return df.with_columns([
        pl.col("true_range").ewm_mean(span=atr_period, adjust=False).alias(f"atr_{atr_period}")
    ]).drop(["prev_close", "true_range"])

def add_adx_indicator(df: pl.DataFrame, adx_period: int = 14) -> pl.DataFrame:
    eps = 1e-9

    df = df.with_columns([
        (pl.col("price_high") - pl.col("price_high").shift(1)).alias("up_move"),
        (pl.col("price_low").shift(1) - pl.col("price_low")).alias("down_move"),
        pl.col("price_close").shift(1).alias("prev_close"),
    ])

    # True Range (FIXED max_horizontal)
    df = df.with_columns(
        pl.max_horizontal(
            pl.col("price_high") - pl.col("price_low"),
            (pl.col("price_high") - pl.col("prev_close")).abs(),
            (pl.col("price_low") - pl.col("prev_close")).abs(),
        ).alias("tr")
    )

    # Directional Movement
    df = df.with_columns([
        pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
        .then(pl.col("up_move")).otherwise(0.0).alias("plus_dm"),

        pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
        .then(pl.col("down_move")).otherwise(0.0).alias("minus_dm"),
    ])

    # Wilder smoothing (ensure min_periods)
    df = df.with_columns([
        pl.col("tr")
        .ewm_mean(alpha=1/adx_period, adjust=False, min_samples=adx_period)
        .alias("tr_smooth"),

        pl.col("plus_dm")
        .ewm_mean(alpha=1/adx_period, adjust=False, min_samples=adx_period)
        .alias("plus_dm_smooth"),

        pl.col("minus_dm")
        .ewm_mean(alpha=1/adx_period, adjust=False, min_samples=adx_period)
        .alias("minus_dm_smooth"),
    ])

    # DI calculations (guard division)
    df = df.with_columns([
        (100 * pl.col("plus_dm_smooth") / (pl.col("tr_smooth") + eps)).alias("plus_di"),
        (100 * pl.col("minus_dm_smooth") / (pl.col("tr_smooth") + eps)).alias("minus_di"),
    ])

    # DX (guard 0/0)
    df = df.with_columns(
        pl.when((pl.col("plus_di") + pl.col("minus_di")) > 0)
        .then(
            100 * (pl.col("plus_di") - pl.col("minus_di")).abs()
            / (pl.col("plus_di") + pl.col("minus_di"))
        )
        .otherwise(None)
        .alias("dx")
    )

    # Final ADX
    df = df.with_columns(
        pl.col("dx")
        .ewm_mean(alpha=1/adx_period, adjust=False, min_samples=adx_period)
        .alias(f"adx_{adx_period}")
    )

    return df.drop([
        "up_move", "down_move", "prev_close",
        "tr", "plus_dm", "minus_dm",
        "tr_smooth", "plus_dm_smooth", "minus_dm_smooth",
        "plus_di", "minus_di", "dx",
    ])

def add_stochastic_oscillator(df: pl.DataFrame, stoch_period: int = 14, stoch_smooth: int = 3) -> pl.DataFrame:
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    df = df.with_columns([
        pl.col("price_low").rolling_min(window_size=stoch_period).alias("l_low"),
        pl.col("price_high").rolling_max(window_size=stoch_period).alias("h_high")
    ]).with_columns([
        (100 * (pl.col("price_close") - pl.col("l_low")) / (pl.col("h_high") - pl.col("l_low"))).alias("stoch_k")
    ])
    
    # %D is the moving average of %K
    return df.with_columns([
        pl.col("stoch_k").rolling_mean(window_size=stoch_smooth).alias("stoch_d")
    ]).drop(["l_low", "h_high"])        

def add_vwap_indicator(df: pl.DataFrame) -> pl.DataFrame:
    # Typical Price * Volume
    # If your data spans multiple days, ensure you have a 'date' column to group by
    return df.with_columns([
        (((pl.col("price_high") + pl.col("price_low") + pl.col("price_close")) / 3) * pl.col("volume")).alias("pv")
    ]).with_columns([
        (pl.col("pv").cum_sum() / pl.col("volume").cum_sum()).alias("vwap")
    ]).drop("pv")

def add_obv_indicator(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.when(pl.col("price_close") > pl.col("price_close").shift(1)).then(pl.col("volume"))
        .when(pl.col("price_close") < pl.col("price_close").shift(1)).then(-1*pl.col("volume"))
        .otherwise(0).alias("direction_vol")
    ]).with_columns([
        pl.col("direction_vol").cum_sum().alias("obv")
    ]).drop("direction_vol")

def preprocess_data(
    ticks_df: pl.DataFrame,
    instrument_token_val: int,
    day: str,
    config: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Basic preprocessing: filtering, type casting, sorting, and deriving trade_volume.
    """
    #print("Preprocessing data...")
    d1 = datetime.datetime.strptime(f"{day} 09:15:00", '%Y-%m-%d %H:%M:%S')
    d2 = datetime.datetime.strptime(f"{day} 15:20:00", '%Y-%m-%d %H:%M:%S')
    ticks_df = ticks_df.filter((pl.col('timestamp') >= d1) & (pl.col("timestamp") <= d2))

    # Filter for a single instrument if multiple exist
    ticks_df = ticks_df.filter(pl.col("instrument_token") == instrument_token_val)
    open_price = ticks_df.sort("timestamp").select(pl.col("last_price").first()).to_series()[0]
    if open_price is None:
        raise ValueError(f"Open price not found for instrument {instrument_token_val} on {day}")

    # Cast timestamps and sort
    ticks_df = ticks_df.with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="us"))
    ).sort("timestamp")

    # Use 'last_quantity' as 'trade_volume'
    ticks_df = ticks_df.rename({"last_quantity": "trade_volume"})

    # Create 1-minute bars (taking the last price of each minute)
    candles_df = ticks_df.group_by_dynamic(
        index_column="timestamp",     # Renamed from 'index_column' to 'timestamp' in newer Polars, or 'by' if not index
        every=config['TIMEFRAME'],               # 1-minute intervals
        group_by="instrument_token",
        include_boundaries=True    # To get '_lower_boundary' for joining back accurately
    ).agg(
        price_high=pl.col("last_price").max() / open_price,
        price_open=pl.col("last_price").first() / open_price,
        price_close=pl.col("last_price").last() / open_price,
        price_low=pl.col("last_price").min() / open_price,
        price_high_real=pl.col("last_price").max(),
        price_open_real=pl.col("last_price").first(),
        price_close_real=pl.col("last_price").last(),
        price_low_real=pl.col("last_price").min(),
        volume=pl.col("trade_volume").sum(),
        price_return=(
            (pl.col("last_price").last() - pl.col("last_price").first()) / open_price
        ),
        price_volatility=(
            (pl.col("last_price").max() - pl.col("last_price").min()) / open_price
        ),
        price_log_return=(
            (pl.col("last_price").last() / pl.col("last_price").first()).log()
        ),
        obi=(pl.col("buy_quantity").cast(pl.Float64).mean() /
              (pl.col("sell_quantity").cast(pl.Float64).mean() + 1e-9) # epsilon to avoid div by zero
        ).fill_null(0)
    )

    candles_df = candles_df.with_columns(
        volatility_std_over_15m=(
            pl.col("price_volatility")
            .rolling_std(window_size=15, min_samples=1) # 15 periods for 15 minutes of 1-min data
            .over("instrument_token")
        ),
        volatility_std_over_3m=(
            pl.col("price_volatility")
            .rolling_std(window_size=3, min_samples=1) # 3 periods for 3 minutes of 1-min data
            .over("instrument_token")
        ),
        volatility_mean_over_15m=(
            pl.col("price_volatility")
            .ewm_mean(span=15, min_samples=1) # 15 periods for 15 minutes of 1-min data
            .over("instrument_token")
        ),
        volatility_mean_over_3m=(
            pl.col("price_volatility")
            .ewm_mean(span=3, min_samples=1) # 3 periods for 3 minutes of 1-min data
            .over("instrument_token")
        ),
        volatility_mean=pl.col('price_volatility').rolling_mean(window_size=1000000000, min_samples=1),
        obi_ma_1=pl.col("obi").ewm_mean(span=1, adjust=False, min_samples=1),
        obi_ma_3=pl.col("obi").ewm_mean(span=3, adjust=False, min_samples=1),
        obi_ma_5=pl.col("obi").ewm_mean(span=5, adjust=False, min_samples=1),
        obi_ma_15=pl.col("obi").ewm_mean(span=15, adjust=False, min_samples=1),


    ).rename({"_lower_boundary": "timestamp_1m_start"}) # Use lower boundary as the key for the minute

    # ADD INDICATORS now refer, claude repo
    # rsi, macd, bollinger bands, atr, adx, stochastic oscillator, vwap, obv.
    candles_df = add_rsi_indicator(candles_df, rsi_period=14, price_col="price_close")
    candles_df = add_macd_indicator(candles_df, macd_fast=12, macd_slow=26, macd_signal=9, price_col="price_close")
    candles_df = add_bollinger_bands(candles_df, bb_period=20, bb_std=2, price_col="price_close")
    candles_df = add_atr_indicator(candles_df, atr_period=14)
    candles_df = add_adx_indicator(candles_df, adx_period=14)
    candles_df = add_stochastic_oscillator(candles_df, stoch_period=14, stoch_smooth=3)
    candles_df = add_vwap_indicator(candles_df)
    candles_df = add_obv_indicator(candles_df)

    candles_df = candles_df.with_columns(
        obi_diff_315=pl.col("obi_ma_3") - pl.col("obi_ma_15"),
        obi_diff_315_abs=(pl.col("obi_ma_3") - pl.col("obi_ma_15")).abs(),
        obi_diff_315_slope=((pl.col("obi_ma_3") - pl.col("obi_ma_15")) - (pl.col("obi_ma_3").shift(1) - pl.col("obi_ma_15").shift(1))),
        obi_diff_35=pl.col("obi_ma_3") - pl.col("obi_ma_5"),
        obi_diff_35_abs=(pl.col("obi_ma_3") - pl.col("obi_ma_5")).abs(),
        obi_diff_35_slope=((pl.col("obi_ma_3") - pl.col("obi_ma_5")) - (pl.col("obi_ma_3").shift(1) - pl.col("obi_ma_5").shift(1))),
        volatility_ratio=pl.col("volatility_std_over_3m") / (pl.col("volatility_std_over_15m") + 1e-9),
        macd_minus_signal=pl.col("macd") - pl.col("signal"),
        rsi_distance_from_50=(pl.col("rsi") - 50).abs(),
        vwap_distance=(pl.col("price_close") - pl.col("vwap")) / pl.col("vwap"),
        volume_surge=pl.col("volume") / (pl.col("volume").rolling_mean(window_size=15, min_samples=1) + 1e-9)
    )
    # obi related features
    candles_df = candles_df.with_columns(
        obi_acceleration_3_15=pl.col("obi_diff_315_slope") - pl.col("obi_diff_315_slope").shift(1),
        obi_acceleration_3_5=pl.col("obi_diff_35_slope") - pl.col("obi_diff_35_slope").shift(1),
        obi_dir_3_15=pl.col("obi_diff_315").sign(),
        obi_dir_3_5=pl.col("obi_diff_35").sign(),
    )
    candles_df = candles_df.with_columns(
        obi_dir_rle_3_15=pl.col("obi_dir_3_15").rle_id(),
        obi_dir_rle_3_5=pl.col("obi_dir_3_5").rle_id()
    )
    # price candle related features
    candles_df = candles_df.with_columns(
        candle_body_ratio=pl.col("price_return") / pl.col("price_volatility"),
        upper_wick_ratio=(pl.col("price_high") - pl.max_horizontal("price_close", "price_open")) / (pl.col("price_volatility") + 1e-9),
        lower_wick_ratio=(pl.min_horizontal("price_close", "price_open") - pl.col("price_low")) / (pl.col("price_volatility") + 1e-9),
        bullish_candle=(pl.col("price_close") > pl.col("price_open")).cast(pl.Int8),
        bearish_candle=(pl.col("price_close") < pl.col("price_open")).cast(pl.Int8),
    )
    candles_df = candles_df.with_columns(
        bullish_candle_count_5=pl.col("bullish_candle").rolling_sum(window_size=5),
        bearish_candle_count_5=pl.col("bearish_candle").rolling_sum(window_size=5),
        bullish_candle_count_15=pl.col("bullish_candle").rolling_sum(window_size=15),
        bearish_candle_count_15=pl.col("bearish_candle").rolling_sum(window_size=15),
        uptrend_5=(pl.col("price_close").rolling_mean(window_size=5) > pl.col("price_close").rolling_mean(window_size=15)).cast(pl.Int8),
        uptrend_15=(pl.col("price_close").rolling_mean(window_size=15) > pl.col("price_close").rolling_mean(window_size=50)).cast(pl.Int8), 
    )

    # volatility features
    candles_df = candles_df.with_columns(
        volatility_percentile=pl.col("price_volatility").rank(method="average") / pl.col("price_volatility").len(),
        volatility_expanding=(pl.col("volatility_std_over_3m") > pl.col('volatility_std_over_15m')).cast(pl.Int8),
        atr_percentile=pl.col("atr_14").rank(method="average") / pl.col("atr_14").len(),
        bb_position=(pl.col("price_close") - pl.col("bb_middle")) / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-9),
    )

    # time based features
    candles_df = candles_df.with_columns(
        minutes_since_midnight=(pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()).cast(pl.Int32),
        minutes_since_open=(pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute() - 9*60 - 15).cast(pl.Int32),
        minutes_until_close=(15*60 + 20 - (pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute())).cast(pl.Int32),
    )
    candles_df = candles_df.with_columns(
        session_segment=pl.when(pl.col("minutes_since_open") < 30).then(1).when(pl.col("minutes_until_close") < 30).then(3).otherwise(2).cast(pl.Int8)
    )

    # interaction features
    candles_df = candles_df.with_columns(
        obi_trend_strength_315=pl.col("obi_diff_315_abs") * pl.col("adx_14") / 100,
        obi_volume_interaction_315=pl.col("obi_diff_315") * pl.col("volume_surge"),
        obi_rsi_interaction_315=pl.col("rsi_distance_from_50") * pl.col("obi_diff_315") / 50,
        obi_volatility_interaction_315=pl.col("obi_diff_315") / (pl.col("volatility_std_over_3m") + 1e-9),
        obi_macd_interaction_315=pl.col("obi_diff_315") * pl.col("macd_minus_signal"),
        obi_bb_interaction_315=pl.col("obi_diff_315") * pl.col("bb_position"),
        obi_trend_strength_35=pl.col("obi_diff_35_abs") * pl.col("adx_14") / 100,
        obi_volume_interaction_35=pl.col("obi_diff_35") * pl.col("volume_surge"),
        obi_rsi_interaction_35=pl.col("rsi_distance_from_50") * pl.col("obi_diff_35") / 50,
        obi_volatility_interaction_35=pl.col("obi_diff_35") / (pl.col("volatility_std_over_3m") + 1e-9),
        obi_macd_interaction_35=pl.col("obi_diff_35") * pl.col("macd_minus_signal"),
        obi_bb_interaction_35=pl.col("obi_diff_35") * pl.col("bb_position"),
    )

    return candles_df


def generate_signals(sdf: pl.DataFrame, config_dict: dict) -> pl.DataFrame:
    df = sdf.sort('timestamp')

    obi_long_entry = (
        ((df['obi_ma_3'] - df['obi_ma_15']) > 0) &
        ((df['obi_ma_3'].shift(1) - df['obi_ma_15'].shift(1)) < 0)
        #& (df['sq_ma_3_acc'] < OP_SIDE_ACC_MAX_SLOPE) 
        #& (df['bq_ma_3_velo'] > 0)
        # & (df['rsi'] < RSI_UP)
        # & (df['smooth_price'] > df['sma_3'])
        # & (df['volume_ma_3'] > df['volume_ma_3'].shift(60*3))
        # & (df['volatility_1m_interval_over_15m'] >= df['last_price'] * VOL_LOW)
        # & ((df['volatility_1m_interval_over_15m'] < df['last_price'] * VOL_UP) | (df['volatility_mean_1m_interval_over_3m'] < df['volatility_mean']))
        #& (df['volatility_mean_1m_interval_over_3m'] < df['volatility_mean'])
    )

    obi_short_entry = (
        ((df['obi_ma_3'] - df['obi_ma_15']) < 0) &
        ((df['obi_ma_3'].shift(1) - df['obi_ma_15'].shift(1)) > 0) 
        # & (df['bq_ma_3_acc'] < OP_SIDE_ACC_MAX_SLOPE) 
        #& (df['sq_ma_3_velo'] > 0)
        # & (df['rsi'] > RSI_LOW)
        # & (df['smooth_price'] < df['sma_3']) 
        # & (df['volume_ma_3'] > df['volume_ma_3'].shift(60*3))
        # & (df['volatility_1m_interval_over_15m'] >= df['last_price'] * VOL_LOW)
        # & ((df['volatility_1m_interval_over_15m'] < df['last_price'] * VOL_UP) | (df['volatility_mean_1m_interval_over_3m'] < df['volatility_mean']))
        #& (df['volatility_mean_1m_interval_over_3m'] < df['volatility_mean'])
    )

    long_entry_trigger = obi_long_entry.fill_null(False)
    short_entry_trigger = obi_short_entry.fill_null(False)

    df = df.with_columns(
        buy_signal=long_entry_trigger,
        sell_signal=short_entry_trigger,
        trade_signal=pl.when(long_entry_trigger).then(1).when(short_entry_trigger).then(-1).otherwise(0),
    )
    return df


def generate_good_trade_label(df: pl.DataFrame,
                              profit_threshold_pct: float,
                              loss_threshold_pct: float,
                              tradeable_col: str):
    eod_cutoff_time_str: str = "15:16:00"
    EOD_CUTOFF_TIME = datetime.datetime.strptime(eod_cutoff_time_str, "%H:%M:%S").time()
    def calculate_label_for_single_tick(
        current_idx: int,
        group_prices: pl.Series,
        group_timestamps: pl.Series,
        signal_type: str
    ) -> int:
        current_price = group_prices[current_idx]
        current_timestamp = group_timestamps[current_idx]

        eod_datetime_on_current_day = datetime.datetime.combine(current_timestamp.date(), EOD_CUTOFF_TIME)

        # If the current tick itself is at or after EOD, no forward evaluation for labeling.
        if current_timestamp >= eod_datetime_on_current_day:
            return 0 

        if signal_type == "BUY":
            upper_barrier = current_price * (1 + profit_threshold_pct)
            lower_barrier = current_price * (1 - loss_threshold_pct)
        else:
            upper_barrier = current_price * (1 + loss_threshold_pct)
            lower_barrier = current_price * (1 - profit_threshold_pct)

        # Iterate through subsequent ticks within this group (instrument's data for the day)
        # The loop goes from the next tick (current_idx + 1) to the end of the group's data.
        for j in range(current_idx + 1, len(group_prices)):
            future_price = group_prices[j]
            future_path_timestamp = group_timestamps[j]
            
            # future_price should not be None if from a valid column, but good to be safe
            if future_price is None: 
                raise Exception("Invalid future price entry")
            
            # Check if EOD cutoff time is reached or passed by this future_path_timestamp
            if future_path_timestamp >= eod_datetime_on_current_day:
                # buy pos
                if signal_type == "BUY":
                    if future_price > current_price:
                        return 1 # PT hit on or by EOD
                else:
                    # sell pos
                    if future_price < current_price:
                        return 1 # PT hit on or by EOD
                return 0
            if signal_type == "BUY":
                # If EOD not yet hit, check PT/SL barriers
                if future_price > upper_barrier:
                    return 1 # Profit target hit before EOD
                elif future_price <= lower_barrier:
                    return 0 # Stop loss hit before EOD
            else:
                # sell pos
                if future_price < lower_barrier:
                    return 1
                elif future_price >= upper_barrier:
                    return 0

        return 0

    # This function will be applied to each group (instrument's sub-DataFrame)
    def apply_labeling_to_group(group_df: pl.DataFrame) -> pl.DataFrame:
        # Extract series once per group for efficiency
        group_prices = group_df.get_column("price_close_real")
        group_timestamps = group_df.get_column("timestamp")
        
        labels = []
        for i in range(len(group_df)): # Iterate through each row of the current group
            if group_df['buy_signal'][i]:
                trade_signal = "BUY"
            elif group_df['sell_signal'][i]:
                trade_signal = "SELL"
            else:
                labels.append(0) # Not a trade signal row
                continue
            labels.append(calculate_label_for_single_tick(i, group_prices, group_timestamps, trade_signal))
        
        return group_df.with_columns(tradeable=pl.Series(labels, dtype=pl.Int8))

    df_with_labels = df.sort("instrument_token", "timestamp").group_by(
        "instrument_token", maintain_order=True
    ).map_groups(apply_labeling_to_group)
    
    return df_with_labels

def balance_df(df, label_col: str="tradeable"):
    print("Balancing label rows")
    # Count occurrences per label
    label_counts = df.group_by(label_col).len().sort("len", descending=True)
    print(f'Label count: {label_counts}')

    # Determine max count to balance to
    max_count = label_counts["len"][0]

    # Prepare list to hold balanced chunks
    balanced_chunks = []

    # For each label, duplicate rows as needed
    for label_value in df[label_col].unique().to_list():
        subset = df.filter(pl.col(label_col) == label_value)
        current_count = subset.height
        if current_count < max_count:
            # Calculate how many more rows are needed
            needed = max_count - current_count
            # Repeat the subset rows as needed (with random sampling if needed)
            additional_rows = subset.sample(n=needed, with_replacement=True)
            balanced_subset = pl.concat([subset, additional_rows])
        else:
            balanced_subset = subset
        balanced_chunks.append(balanced_subset)

    # Combine all into one balanced DataFrame
    balanced_df = pl.concat(balanced_chunks).with_row_index(name="new_id").sample(fraction=1.0,
                                                                                  shuffle=True,
                                                                                  with_replacement=False)
    return balanced_df

class BrokerageCalculator(object):
    # self.total_tax holds total brokerage + tax incurred on the squared off trade
    def __init__(self, bp, sp, qty=1, is_nse=True):
        self.bp = bp # buy price
        self.sp = sp # sell price
        self.qty = qty # trade quantity

        if self.bp is None:
            self.bp = 0
            self.bse_tran_charge_buy = 0
        if self.sp is None:
            self.sp = 0
            self.bse_tran_charge_sell = 0

        self.brokerage_buy = 20 if ((self.bp * self.qty * 0.0003) > 20) else round(self.bp * self.qty * 0.0003, 2)
        self.brokerage_sell = 20 if ((self.sp * self.qty * 0.0003) > 20) else round(self.sp * self.qty * 0.0003, 2)
        self.brokerage = round(self.brokerage_buy + self.brokerage_sell, 2)
        self.turnover = round((self.bp + self.sp) * self.qty, 2)
        self.stt_total = round(round((self.sp * self.qty) * 0.00025, 2))
        self.exc_trans_charge = round(0.0000325 * self.turnover, 2) if is_nse else round(0.0000375 * self.turnover, 2)
        self.nse_ipft = round(0.000001 * self.turnover, 2) if is_nse else 0
        self.exc_trans_charge = round(self.exc_trans_charge + self.nse_ipft, 2)
        self.cc = 0
        self.stax = round(0.18 * (self.brokerage + self.exc_trans_charge), 2)
        self.sebi_charges = round(self.turnover * 0.000001, 2)
        self.stamp_charges = round(round(self.bp * self.qty * 0.00003, 2))
        self.total_tax = round(self.brokerage + self.stt_total + self.exc_trans_charge + self.cc + self.stax + self.sebi_charges + self.stamp_charges, 2)
        self.breakeven = round(self.total_tax / self.qty, 2)
        if not self.breakeven:
            self.breakeven = 0
        self.net_profit = round(((self.sp - self.bp) * self.qty) - self.total_tax, 2)


# --- PnL Evaluation Function with Fixed Profit Target and Trailing Stop-Loss ---
def evaluate_pnl_with_trailing_stop(
    df_with_signals: pl.DataFrame,
    config_dict: dict,
    trade_qty: int = QTY,
    no_new_trade_after_time_str: str = "15:15:00",
    mandatory_sq_off_time_str: str = "15:16:00",
    enable_trailing_sl: bool=False
) -> dict:
    if df_with_signals.is_empty():
        print("Warning: Input DataFrame for PnL evaluation is empty.")
        return {"total_pnl": 0.0, "trade_count": 0, "winning_trades": 0, "losing_trades": 0, "trade_history": []}

    NO_NEW_TRADE_TIME = datetime.datetime.strptime(no_new_trade_after_time_str, "%H:%M:%S").time()
    MANDATORY_SQ_OFF_TIME = datetime.datetime.strptime(mandatory_sq_off_time_str, "%H:%M:%S").time()

    current_position = 0  # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    entry_timestamp = None
    
    highest_price_since_entry = 0.0 
    lowest_price_since_entry = float('inf') 
    current_trailing_stop_price = 0.0

    total_pnl = 0.0
    trade_history = []
    winning_trades = 0
    losing_trades = 0
    
    df_sorted = df_with_signals.sort("timestamp")
    last_available_price_for_eod_sq_off = df_sorted[-1]['price_close_real'][0] if not df_sorted.is_empty() else 0
    last_available_timestamp_for_eod_sq_off = df_sorted[-1]['timestamp'] if not df_sorted.is_empty() else None
    # eval tradable price, as mid of open and close of next candle
    df_sorted = df_sorted.with_columns(
        to_trade_price=(pl.col("price_open_real").shift(-1) + pl.col("price_close_real").shift(-1)) / 2
    )
    for tick in df_sorted.iter_rows(named=True):
        current_timestamp = tick['timestamp']
        current_price = tick['to_trade_price'] if tick['to_trade_price'] is not None else last_available_price_for_eod_sq_off
        current_time = current_timestamp.time()

        is_buy_entry_signal = tick.get('buy_signal', False)
        is_sell_entry_signal = tick.get('sell_signal', False)

        exit_reason = None
        close_position_this_tick = False
        
        if current_position != 0: # If a position is open
            profit_pct_for_target_calc = 0.0
            # Update trailing stop state first, regardless of P/L target check
            if current_position == 1: # Long
                if entry_price != 0: profit_pct_for_target_calc = (current_price - entry_price) / entry_price
                if enable_trailing_sl:    
                    highest_price_since_entry = max(highest_price_since_entry, current_price)
                    new_potential_ts_price = highest_price_since_entry * (1 - config_dict['trailing_stop_pct'])
                    current_trailing_stop_price = max(current_trailing_stop_price, new_potential_ts_price)
            elif current_position == -1: # Short
                if entry_price != 0: profit_pct_for_target_calc = (entry_price - current_price) / entry_price
                if enable_trailing_sl:
                    lowest_price_since_entry = min(lowest_price_since_entry, current_price)
                    new_potential_ts_price = lowest_price_since_entry * (1 + config_dict['trailing_stop_pct'])
                    current_trailing_stop_price = min(current_trailing_stop_price, new_potential_ts_price)
            
            # Check Exit Conditions in order of precedence
            # 1. Fixed Profit Target
            if profit_pct_for_target_calc >= config_dict['profit_target_pct']:
                exit_reason = f"profit_target_{'long' if current_position == 1 else 'short'}"
                close_position_this_tick = True
            
            # 2. Trailing Stop-Loss (if not closing for profit target)
            if not close_position_this_tick:
                if current_position == 1 and current_price <= current_trailing_stop_price:
                    exit_reason = "trailing_stop_long"
                    close_position_this_tick = True
                elif current_position == -1 and current_price >= current_trailing_stop_price:
                    exit_reason = "trailing_stop_short"
                    close_position_this_tick = True
            
            # 3. Mandatory Square Off Time (if not already closing)
            if not close_position_this_tick and current_time >= MANDATORY_SQ_OFF_TIME:
                exit_reason = "mandatory_eod_time_sq_off"
                close_position_this_tick = True
            
            # 4. S&R by opposite signal (if not already closing)
            if not close_position_this_tick and EXIT_ON_REV_SIGNAL:
                if current_position == 1 and is_sell_entry_signal:
                    exit_reason = "signal_reverse_to_short"
                    close_position_this_tick = True
                elif current_position == -1 and is_buy_entry_signal:
                    exit_reason = "signal_reverse_to_long"
                    close_position_this_tick = True
        
        if close_position_this_tick:
            exit_price = current_price             
            bp_calc, sp_calc = 0.0, 0.0
            position_type_str = "Long" if current_position == 1 else "Short"

            if current_position == 1: 
                bp_calc = entry_price; sp_calc = exit_price
            elif current_position == -1: 
                bp_calc = exit_price; sp_calc = entry_price 
            
            if entry_price > 0: 
                brokerage_instance = BrokerageCalculator(bp=bp_calc, sp=sp_calc, qty=trade_qty)
                pnl_for_this_trade = brokerage_instance.net_profit
                total_pnl += pnl_for_this_trade
                
                if pnl_for_this_trade > 0: winning_trades += 1
                elif pnl_for_this_trade < 0: losing_trades += 1

                trade_history.append({
                    "entry_timestamp": entry_timestamp, "exit_timestamp": current_timestamp,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "position_type": position_type_str,
                    "stop_price_at_exit": current_trailing_stop_price,
                    "gross_pnl_pts": round((sp_calc - bp_calc) if position_type_str == "Long" else (sp_calc - bp_calc), 4),
                    "brokerage_and_charges": brokerage_instance.total_tax,
                    "net_pnl": pnl_for_this_trade, "exit_reason": exit_reason
                })
            
            original_position_closed = current_position 
            current_position = 0; entry_price = 0.0; entry_timestamp = None
            highest_price_since_entry = 0.0; lowest_price_since_entry = float('inf'); current_trailing_stop_price = 0.0

            can_open_new_trade_after_SR = (current_time < NO_NEW_TRADE_TIME) and OPEN_POS_AFTER_REV_SIGNAL_EXIT
            if can_open_new_trade_after_SR:
                if exit_reason == "signal_reverse_to_short" and original_position_closed == 1:
                    current_position = -1; entry_price = current_price; entry_timestamp = current_timestamp
                    lowest_price_since_entry = current_price
                    current_trailing_stop_price = entry_price * (1 + config_dict['initial_stop_loss_pct'])
                    highest_price_since_entry = 0.0 
                elif exit_reason == "signal_reverse_to_long" and original_position_closed == -1:
                    current_position = 1; entry_price = current_price; entry_timestamp = current_timestamp
                    highest_price_since_entry = current_price
                    current_trailing_stop_price = entry_price * (1 - config_dict['initial_stop_loss_pct'])
                    lowest_price_since_entry = float('inf')
            continue 

        if current_position == 0 and current_time < NO_NEW_TRADE_TIME and tick['tradeable'] == 1:
            if is_buy_entry_signal and is_sell_entry_signal: pass 
            elif is_buy_entry_signal:
                current_position = 1; entry_price = current_price; entry_timestamp = current_timestamp
                highest_price_since_entry = current_price
                current_trailing_stop_price = entry_price * (1 - config_dict['initial_stop_loss_pct'])
                lowest_price_since_entry = float('inf')
            elif is_sell_entry_signal:
                current_position = -1; entry_price = current_price; entry_timestamp = current_timestamp
                lowest_price_since_entry = current_price
                current_trailing_stop_price = entry_price * (1 + config_dict['initial_stop_loss_pct'])
                highest_price_since_entry = 0.0 
                
    if current_position != 0 and entry_price > 0 and last_available_timestamp_for_eod_sq_off is not None:
        exit_price = last_available_price_for_eod_sq_off; exit_timestamp = last_available_timestamp_for_eod_sq_off
        exit_reason = "final_eod_data_end_square_off"
        bp_calc, sp_calc = 0.0, 0.0
        position_type_str = "Long" if current_position == 1 else "Short"
        if current_position == 1: bp_calc = entry_price; sp_calc = exit_price
        elif current_position == -1: bp_calc = exit_price; sp_calc = entry_price
        #print(exit_price, entry_price)
        brokerage_instance = BrokerageCalculator(bp=bp_calc, sp=sp_calc, qty=trade_qty)
        pnl_for_this_trade = brokerage_instance.net_profit
        total_pnl += pnl_for_this_trade
        if pnl_for_this_trade > 0: winning_trades += 1
        elif pnl_for_this_trade < 0: losing_trades += 1
        trade_history.append({
            "entry_timestamp": entry_timestamp, "exit_timestamp": exit_timestamp,
            "entry_price": entry_price, "exit_price": exit_price,
            "position_type": position_type_str, "stop_price_at_exit": current_trailing_stop_price,
            "gross_pnl_pts": round((sp_calc - bp_calc) if position_type_str == "Long" else (sp_calc - bp_calc), 4),
            "brokerage_and_charges": brokerage_instance.total_tax,
            "net_pnl": pnl_for_this_trade, "exit_reason": exit_reason
        })
            
    return {
        "total_pnl": round(total_pnl, 2), "trade_count": len(trade_history),
        "winning_trades": winning_trades, "losing_trades": losing_trades,
        "trade_history": trade_history
    }


# --- Main Execution ---
if __name__ == "__main__":
    ticks_df_map = {}
    config = CONFIG
    output_df = None
    all_trade_df = None
    optimal_threshold = 0.5
    max_pr_score = 0.0

    if TRAIN_MODEL:
        for day in TRAIN_DAYS:
            ticks_df = ticks_df_map.get(day, pl.read_parquet(os.path.join(MARKET_DATA_HOME_PATH, f"tick/{config['instrument_token']}/{day}.parquet")))
            ticks_df_map[day] = ticks_df
            n_ticks = ticks_df.clone()
            # 1. Preprocess
            try:
                final_df = preprocess_data(n_ticks, config['instrument_token'], day, config)
            except ValueError as ve:
                print(f"Skipping day {day} due to error: {ve}")
                continue
            final_df = final_df.drop_nulls() # After join and rolling calculations, nulls might appear
            if final_df.is_empty():
                print("DataFrame is empty after join_asof and drop_nulls. Check data alignment or window sizes.")
                continue
            
            # 4. Generate Signals
            output_df = generate_signals(final_df, config)
            output_df = generate_good_trade_label(output_df,
                                                  CONFIG['profit_target_pct'],
                                                  CONFIG['initial_stop_loss_pct'], "tradeable")
            # filter buy/sell entries
            output_df = output_df.filter((pl.col("buy_signal").is_not_null() & pl.col('buy_signal')) |
                                        (pl.col("sell_signal").is_not_null() & pl.col('sell_signal')))
            
            if all_trade_df is None:
                all_trade_df = output_df
            else:
                all_trade_df = pl.concat([all_trade_df, output_df])

        feature_columns = ['volatility_std_over_15m', 'volatility_std_over_3m', 'volatility_mean_over_15m',
                           'volatility_mean_over_3m',
                            'volatility_ratio',
                            'macd_minus_signal', 'rsi_distance_from_50',
                            'vwap_distance', 'volume_surge',  
                            'obv', 'trade_signal',
                            'obi_acceleration_3_15', 'obi_acceleration_3_5',
                            'obi_dir_rle_3_15', 'obi_dir_rle_3_5',
                            'candle_body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
                            'bullish_candle_count_5', 'bearish_candle_count_5',
                            'bullish_candle_count_15', 'bearish_candle_count_15',
                            'uptrend_5', 'uptrend_15',
                            'volatility_percentile', 'volatility_expanding',
                            'atr_percentile', 'bb_position',
                            'minutes_since_midnight', 'minutes_since_open', 'minutes_until_close',
                            'session_segment', 'obi_diff_315', 'obi_diff_315_abs', 'obi_diff_315_slope',
                            'obi_diff_35', 'obi_diff_35_abs', 'obi_diff_35_slope'
                           ]
        target_column_name = ["tradeable"]

        # traing model & eval accuracy
        train_split_ratio = 0.8
        all_df_samples = len(all_trade_df)
        all_trade_train_df = all_trade_df[:int(train_split_ratio * all_df_samples)]
        all_trade_test_df = all_trade_df[int(train_split_ratio * all_df_samples):]
        #import ipdb; ipdb.set_trace()

        # replace with scale_pos_weight in xgboost
        #all_trade_train_df = balance_df(all_trade_train_df, label_col="tradeable")
        pos_count = all_trade_train_df.filter(pl.col("tradeable") == 1).height
        neg_count = all_trade_train_df.filter(pl.col("tradeable") == 0).height
        spw = 1.0
        if pos_count > 0:
            spw = neg_count / pos_count

        X = all_trade_train_df.select(feature_columns)
        y = all_trade_train_df.select(target_column_name)

        # Convert to Pandas for scikit-learn
        X_train_pd = X.to_pandas()
        y_train_pd = y.to_pandas().squeeze() # .squeeze() to convert DataFrame column to Serie

        X_test_pd = all_trade_test_df.select(feature_columns).to_pandas()
        y_test_pd = all_trade_test_df.select(target_column_name).to_pandas().squeeze()

        print(f"Training data shape: {X_train_pd.shape}, Test data shape: {X_test_pd.shape}")
        if X_train_pd.empty:
            print("Training data is empty. Cannot train model.")
            all_trade_df.with_columns(dt_predicted_signal=pl.lit(None, dtype=pl.Int8))
        else:
            model = XGBClassifier(**{'learning_rate': 0.03,
                                     'max_depth': 4,
                                     'n_estimators': 5000,
                                     'subsample': 0.7,
                                     'colsample_bytree': 0.7,
                                     'objective': 'binary:logistic',
                                     'eval_metric': ['logloss'],
                                     'min_child_weight': 30,
                                     'gamma': 3.0,
                                     'reg_alpha':0.5,
                                     'reg_lambda': 7.0,
                                     'scale_pos_weight': spw
                                     })
            # try:
            #     model.load_model("rel_xgbmodel.json")
            # except:
            #     pass
            model.fit(X_train_pd, y_train_pd, eval_set=[(X_test_pd, y_test_pd)],
                       early_stopping_rounds=100, verbose=False)
            model.save_model('rel_xgbmodel.json')
            feature_importances = model.feature_importances_
            feature_importance_dict = dict(zip(feature_columns, feature_importances))
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            print("\nFeature Importances:")
            for feature, importance in sorted_features:
                print(f"{feature}: {importance}")
            # Predict and evaluate the model
            y_pred_proba = model.predict_proba(X_test_pd)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_test_pd, y_pred_proba)
            min_recall = 0.2
            valid = recalls[:-1] >= min_recall
            best_idx = np.argmax(precisions[:-1][valid])
            optimal_threshold = thresholds[valid][best_idx]
            print(f'Optimal Probability Threshold based on Precision-Recall Curve: {optimal_threshold}')

            y_pred = (y_pred_proba > optimal_threshold).astype(int)
            accuracy = accuracy_score(y_test_pd, y_pred)
            print(f'Test Data Accuracy: {accuracy}')
            #import ipdb; ipdb.set_trace()

            # --- Evaluate on Test Set (if test set is not empty) ---
            if not X_test_pd.empty:
                print("\nModel Evaluation on Test Set:")
                y_pred_test_proba = model.predict_proba(X_test_pd)[:, 1]
                y_pred_test = (y_pred_test_proba > optimal_threshold).astype(int)
                print(classification_report(y_test_pd, y_pred_test, zero_division=0))
                print("Confusion Matrix (Test Set):")
                print(confusion_matrix(y_test_pd, y_pred_test, labels=[0, 1])) # Specify labels for order
            else:
                print("\nTest set is empty. No evaluation metrics to display.")

    # --- Testing/Evaluation Phase ---
    day_wise_pnl = []
    all_day_pnl = 0.0
    model = XGBClassifier()
    model.load_model("rel_xgbmodel.json") #load('decision_tree_model.joblib')
    for day in TEST_DAYS:
        ticks_df = ticks_df_map.get(day, pl.read_parquet(os.path.join(MARKET_DATA_HOME_PATH, f"tick/{config['instrument_token']}/{day}.parquet")))
        ticks_df_map[day] = ticks_df
        n_ticks = ticks_df.clone()
        # 1. Preprocess
        try:
            final_df = preprocess_data(n_ticks, config['instrument_token'], day, config)
        except ValueError as ve:
            print(f"Skipping day {day} due to error: {ve}")
            continue
        final_df = final_df.drop_nulls() # After join and rolling calculations, nulls might appear
        if final_df.is_empty():
            print("DataFrame is empty after join_asof and drop_nulls. Check data alignment or window sizes.")
            continue
        output_df = generate_signals(final_df, config)
        feature_columns =['volatility_std_over_15m', 'volatility_std_over_3m', 'volatility_mean_over_15m',
                           'volatility_mean_over_3m',
                            'volatility_ratio',
                            'macd_minus_signal', 'rsi_distance_from_50',
                            'vwap_distance', 'volume_surge',  
                            'obv', 'trade_signal',
                            'obi_acceleration_3_15', 'obi_acceleration_3_5',
                            'obi_dir_rle_3_15', 'obi_dir_rle_3_5',
                            'candle_body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
                            'bullish_candle_count_5', 'bearish_candle_count_5',
                            'bullish_candle_count_15', 'bearish_candle_count_15',
                            'uptrend_5', 'uptrend_15',
                            'volatility_percentile', 'volatility_expanding',
                            'atr_percentile', 'bb_position',
                            'minutes_since_midnight', 'minutes_since_open', 'minutes_until_close',
                            'session_segment', 'obi_diff_315', 'obi_diff_315_abs', 'obi_diff_315_slope',
                            'obi_diff_35', 'obi_diff_35_abs', 'obi_diff_35_slope'
                           ]
        predict_proba = model.predict_proba(output_df.select(feature_columns).to_pandas())[:, 1]
        all_predictions_on_model_subset = (predict_proba > optimal_threshold).astype(int)
        output_df = output_df.with_columns(
            tradeable=pl.Series(values=all_predictions_on_model_subset, dtype=pl.Int8)
        )
        # output_df = output_df.with_columns(
        #     buy_signal=pl.col('buy_signal') & pl.col('tradeable').eq(1),
        #     sell_signal=pl.col('sell_signal') & pl.col('tradeable').eq(1)
        # )

        pnl_results = evaluate_pnl_with_trailing_stop(
            output_df,
            config
        )
        all_day_pnl += pnl_results['total_pnl']
        print("\n--- PnL Evaluation Results ---")
        print(f"Instrument: {config['instrument_token']}  Date: {day}")
        print(f"Total Net PnL: {pnl_results['total_pnl']}")
        print(f"Total Trades: {pnl_results['trade_count']}")
        print(f"Winning Trades: {pnl_results['winning_trades']}")
        print(f"Losing Trades: {pnl_results['losing_trades']}")

        if SHOW_HISTORY and pnl_results['trade_history']:
            print("\nTrade History:")
            for i, trade in enumerate(pnl_results['trade_history']):
                print(f"  Trade {i+1}:")
                for key, value in trade.items():
                    print(f"    {key}: {value}")

        if SHOW_PLOT:
            # PLOT data & signal pts........
            df = output_df.to_pandas()[20:]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
            # Plot each variable
            # axs[0].plot(df.index, df['sma_3'], label='SMA3', color='red')
            # axs[0].plot(df.index, df['sma_5'], label='SMA5', color='blue')s
            # axs[0].set_ylabel('SMAs')
            axs[0].plot(df.index, df['price_close_real'], label='Price', color='black')
            axs[0].set_ylabel('Smooth Price')
            # resis = df[(df['volatility_1m_interval_over_3m'] < 0.2) &
            #            (df['bq_ma_3_acc'] > 1000) &
            #            (df['sq_ma_3_acc']<-3000) & (df['volume_ma_1'] > df['volume_ma_3'])]
            # axs[0].scatter(resis.index, resis['last_price'], color='blue', marker='o', s=100, zorder=3, alpha=0.7)
            # supp = df[(df['volatility_1m_interval_over_3m'] < 0.2) & (df['sq_ma_3_acc'] > 1000)
            #           & (df['bq_ma_3_acc']<-3000)  & (df['volume_ma_1'] > df['volume_ma_3'])]
            # axs[0].scatter(supp.index, supp['last_price'], color='yellow', marker='o', s=100, zorder=3, alpha=0.7)
            # Plot Buy Entry Signals (Green Dots)
            buy_signals = df[df['buy_signal'] & (df['tradeable'] == 1)]
            axs[0].scatter(buy_signals.index, buy_signals['price_close_real'], color='green', marker='o', s=100, zorder=3, alpha=0.7)
            # buy_signals = df[df['buy_signal1']]
            # axs[0].scatter(buy_signals.index, buy_signals['last_price'], color='green', marker='$1$', s=100, zorder=3, alpha=0.7)
            # buy_signals = df[df['buy_signal3']]
            # axs[0].scatter(buy_signals.index, buy_signals['last_price'], color='green', marker='$3$', s=100, zorder=3, alpha=0.7)
            # buy_signals = df[df['buy_signal5']]
            # axs[0].scatter(buy_signals.index, buy_signals['last_price'],  color='green', marker='$5 $', s=100, zorder=3, alpha=0.7)
            
            # Plot Sell Entry Signals (Red Dots)
            sell_signals = df[df['sell_signal'] & (df['tradeable'] == 1)]
            axs[0].scatter(sell_signals.index, sell_signals['price_close_real'],  color='red', marker='o', s=100, zorder=3, alpha=0.7)
            # sell_signals = df[df['sell_signal1']]
            # axs[0].scatter(sell_signals.index, sell_signals['last_price'],  color='red', marker='$1$', s=100, zorder=3, alpha=0.7)
            # sell_signals = df[df['sell_signal3']]
            # axs[0].scatter(sell_signals.index, sell_signals['last_price'],  color='red', marker='$3$', s=100, zorder=3, alpha=0.7)
            # sell_signals = df[df['sell_signal5']]
            # axs[0].scatter(sell_signals.index, sell_signals['last_price'],  color='red', marker='$5$', s=100, zorder=3, alpha=0.7)
            axs[0].legend()
            axs[0].grid(True, linestyle=':', alpha=0.7)

            # axs[5].plot(df.index, df['sq_ma_3'], label='SQ MA 3', color='red')
            # axs[5].plot(df.index, df['sq_ma_5'], label='SQ MA 5', color='blue')
            # axs[5].plot(df.index, df['sq_ma_15'], label='SQ MA 15', color='purple')
            # axs[5].set_ylabel('SQ MA')
            # axs[5].grid(True, linestyle=':', alpha=0.7)
            # axs[5].legend()

            # axs[4].plot(df.index, df['macd'], label='MACD', color="red")
            # axs[4].plot(df.index, df['signal'], label='Signal', color="black")
            # axs[4].set_ylabel('MACD')
            # axs[4].grid(True, linestyle=':', alpha=0.7)
            # axs[4].legend()
        
            plt.xlabel('Timestamp')
            plt.tight_layout()
            plt.show()

        day_wise_pnl.append(pnl_results)
    print(f"Total pnl for all days: {all_day_pnl}")
