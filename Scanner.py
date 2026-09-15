# ============================================================
# CRYPTO MASTER AI v1
# Multi-Exchange Crypto Scanner
# LONG / SHORT / NO TRADE
# ============================================================

import os
import time
import math
import traceback
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAMES = {
    "15m": 120,
    "1h": 150,
    "4h": 150,
    "1d": 150,
}

TOP_COINS_PER_EXCHANGE = 30
MIN_QUOTE_VOLUME = 500_000
MIN_SCORE = 65

# Telegram is optional.
# Later put these in GitHub Secrets:
# TELEGRAM_BOT_TOKEN
# TELEGRAM_CHAT_ID

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================================
# EXCHANGES
# ============================================================

EXCHANGE_NAMES = [
    "okx",
    "kucoin",
    "bybit",
    "binance",
]


def create_exchange(name):
    try:
        exchange_class = getattr(ccxt, name)

        exchange = exchange_class({
            "enableRateLimit": True,
            "timeout": 20000,
        })

        return exchange

    except Exception as e:
        print(f"[ERROR] Cannot create {name}: {e}")
        return None


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    return result.fillna(50)


def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)

    macd_line = fast - slow
    signal = ema(macd_line, 9)

    histogram = macd_line - signal

    return macd_line, signal, histogram


def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def volume_ratio(df, period=20):
    avg = df["volume"].rolling(period).mean()

    return (
        df["volume"] / avg.replace(0, np.nan)
    ).fillna(1)


# ============================================================
# OHLCV
# ============================================================

def get_ohlcv(exchange, symbol, timeframe, limit=150):

    try:

        data = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not data or len(data) < 60:
            return None

        df = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df.dropna(inplace=True)

        if len(df) < 60:
            return None

        return df

    except Exception:
        return None


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def analyze_timeframe(df):

    if df is None or len(df) < 60:
        return None

    close = df["close"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    macd_line, macd_signal, macd_hist = macd(close)

    rsi_value = float(rsi(close).iloc[-1])

    vol_ratio = float(
        volume_ratio(df).iloc[-1]
    )

    atr_value = float(
        atr(df).iloc[-1]
    )

    price = float(close.iloc[-1])

    long_points = 0
    short_points = 0

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if price > ema20.iloc[-1]:
        long_points += 8
    else:
        short_points += 8

    if price > ema50.iloc[-1]:
        long_points += 8
    else:
        short_points += 8

    if ema20.iloc[-1] > ema50.iloc[-1]:
        long_points += 8
    else:
        short_points += 8

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if 50 <= rsi_value <= 68:
        long_points += 10

    elif 32 <= rsi_value < 50:
        short_points += 8

    elif rsi_value > 75:
        short_points += 5

    elif rsi_value < 25:
        long_points += 5

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if macd_hist.iloc[-1] > 0:
        long_points += 8
    else:
        short_points += 8

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if vol_ratio > 1.5:

        if long_points > short_points:
            long_points += 5
        else:
            short_points += 5

    # --------------------------------------------------------
    # Candle momentum
    # --------------------------------------------------------

    recent_return = (
        close.iloc[-1] / close.iloc[-5] - 1
    )

    if recent_return > 0.01:
        long_points += 5

    elif recent_return < -0.01:
        short_points += 5

    return {
        "price": price,
        "rsi": rsi_value,
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "volume_ratio": vol_ratio,
        "atr": atr_value,
        "long": long_points,
        "short": short_points,
    }


# ============================================================
# MULTI TIMEFRAME ANALYSIS
# ============================================================

def multi_timeframe_analysis(exchange, symbol):

    results = {}

    for tf, limit in TIMEFRAMES.items():

        df = get_ohlcv(
            exchange,
            symbol,
            tf,
            limit
        )

        analysis = analyze_timeframe(df)

        if analysis:
            results[tf] = analysis

        time.sleep(0.05)

    if not results:
        return None

    long_score = 0
    short_score = 0

    # More weight on higher timeframes
    weights = {
        "15m": 1.0,
        "1h": 1.5,
        "4h": 2.0,
        "1d": 2.5,
    }

    for tf, data in results.items():

        w = weights.get(tf, 1)

        long_score += data["long"] * w
        short_score += data["short"] * w

    return {
        "timeframes": results,
        "long_score": long_score,
        "short_score": short_score,
    }


# ============================================================
# BTC MARKET REGIME
# ============================================================

def btc_regime(exchange):

    try:

        symbol = "BTC/USDT"

        df = get_ohlcv(
            exchange,
            symbol,
            "1h",
            200
        )

        if df is None:
            return {
                "regime": "UNKNOWN",
                "score": 0
            }

        close = df["close"]

        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        e200 = ema(close, 200).iloc[-1]

        rsi_value = float(
            rsi(close).iloc[-1]
        )

        price = float(
            close.iloc[-1]
        )

        if price > e20 and e20 > e50 and e50 > e200 and rsi_value >= 55:
            regime = "BULLISH"
            score = 80

        elif price < e20 and e20 < e50 and e50 < e200 and rsi_value <= 45:
            regime = "BEARISH"
            score = 80

        else:
            regime = "NEUTRAL"
            score = 50

        return {
            "regime": regime,
            "score": score
        }

    except Exception:
        return {
            "regime": "UNKNOWN",
            "score": 0
        }
