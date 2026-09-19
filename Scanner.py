import os
import json
import math
import re
import time
import traceback
from datetime import datetime, timezone

import requests
import ccxt
import pandas as pd
import numpy as np


# ============================================================
# CRYPTO MASTER AI V9.1
# Binance + OKX + Bybit + KuCoin
# Spot + Futures + CMC + News + Order Book + Learning
# ============================================================

UTC = timezone.utc

CMC_BASE = "https://pro-api.coinmarketcap.com/public-api"

MAX_COINS = 30
OHLCV_LIMIT = 220
HTTP_TIMEOUT = 15

STABLES = {
    "USDT", "USDC", "FDUSD", "DAI", "USDE", "USDS",
    "USDD", "TUSD", "USDP", "PYUSD", "USDG", "RLUSD",
    "EURC", "EURT", "USTC"
}

# Binance is included here
SPOT_IDS = [
    "binance",
    "okx",
    "bybit",
    "kucoin"
]

# Binance USD-M Futures is included here
FUTURE_IDS = [
    "binanceusdm",
    "okx",
    "bybit",
    "kucoin"
]

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://news.bitcoin.com/feed/",
    "https://www.theblock.co/rss.xml",
]

session = requests.Session()

session.headers.update({
    "User-Agent": "Crypto-Master-AI/9.1",
    "Accept": "application/json,text/xml,application/xml,*/*"
})


# ============================================================
# BASIC HELPERS
# ============================================================

def now():
    return datetime.now(UTC)


def log(message):
    print(
        f"[{now().isoformat()}] {message}",
        flush=True
    )


def number(value, default=np.nan):
    try:
        if value is None or value == "":
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clip(value, low, high):
    try:
        return max(
            low,
            min(high, float(value))
        )
    except Exception:
        return low


def text(value, default=""):
    if value is None:
        return default

    return str(value).strip()


# ============================================================
# EXCHANGE CONNECTION
# ============================================================

def create_exchange(exchange_id, market_type):

    try:

        exchange_class = getattr(ccxt, exchange_id)

        options = {
            "enableRateLimit": True,
            "timeout": 15000
        }

        options["options"] = {
            "defaultType": market_type
        }

        exchange = exchange_class(options)

        exchange.load_markets()

        log(
            f"EXCHANGE OK: {exchange_id} | "
            f"markets={len(exchange.markets)}"
        )

        return exchange

    except Exception as e:

        log(
            f"EXCHANGE FAILED: {exchange_id} | "
            f"{type(e).__name__}: {e}"
        )

        return None


def build_exchanges():

    spot = []
    futures = []

    # Spot
    for exchange_id in SPOT_IDS:

        exchange = create_exchange(
            exchange_id,
            "spot"
        )

        if exchange:
            spot.append(
                (exchange_id, exchange)
            )

    # Futures
    for exchange_id in FUTURE_IDS:

        exchange = create_exchange(
            exchange_id,
            "swap"
        )

        if exchange:
            futures.append(
                (exchange_id, exchange)
            )

    return spot, futures


# ============================================================
# MARKET DISCOVERY
# ============================================================

def discover_candidates(spot):

    markets = {}

    for exchange_id, exchange in spot:

        try:

            tickers = exchange.fetch_tickers()

            for symbol, ticker in tickers.items():

                market = exchange.markets.get(symbol)

                if not market:
                    continue

                if not market.get("spot"):
                    continue

                if market.get("quote") != "USDT":
                    continue

                base = market.get("base")

                if not base:
                    continue

                base = base.upper()

                if base in STABLES:
                    continue

                last = number(
                    ticker.get("last")
                )

                quote_volume = number(
                    ticker.get("quoteVolume")
                )

                if not math.isfinite(
                    quote_volume
                ):

                    base_volume = number(
                        ticker.get("baseVolume")
                    )

                    if (
                        math.isfinite(base_volume)
                        and math.isfinite(last)
                    ):
                        quote_volume = (
                            base_volume * last
                        )
                    else:
                        quote_volume = 0

                if quote_volume <= 0:
                    continue

                pair = f"{base}/USDT"

                if pair not in markets:

                    markets[pair] = {
                        "symbol": pair,
                        "market_volume": 0.0,
                        "exchange_volumes": {}
                    }

                markets[pair][
                    "market_volume"
                ] += quote_volume

                markets[pair][
                    "exchange_volumes"
                ][exchange_id] = quote_volume

        except Exception as e:

            log(
                f"DISCOVERY FAILED: "
                f"{exchange_id} | "
                f"{type(e).__name__}: {e}"
            )

    results = []

    for item in markets.values():

        if item["market_volume"] < 250000:
            continue

        exchanges = item[
            "exchange_volumes"
        ]

        results.append({
            "symbol": item["symbol"],
            "market_volume": round(
                item["market_volume"],
                2
            ),
            "exchange_count": len(exchanges),
            "exchanges": ",".join(
                exchanges.keys()
            )
        })

    results.sort(
        key=lambda x: x["market_volume"],
        reverse=True
    )

    return results[:MAX_COINS]


# ============================================================
# FIND SPOT EXCHANGE
# ============================================================

def find_spot_exchange(spot, symbol):

    # Binance first
    preferred = [
        "binance",
        "okx",
        "bybit",
        "kucoin"
    ]

    for preferred_id in preferred:

        for exchange_id, exchange in spot:

            if exchange_id != preferred_id:
                continue

            if symbol in exchange.markets:
                return exchange_id, exchange

    # fallback
    for exchange_id, exchange in spot:

        if symbol in exchange.markets:
            return exchange_id, exchange

    return None, None


# ============================================================
# OHLCV
# ============================================================

def get_ohlcv(exchange, symbol, timeframe):

    try:

        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=OHLCV_LIMIT
        )

        if not candles:
            return None

        if len(candles) < 60:
            return None

        df = pd.DataFrame(
            candles,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna()

        return df.reset_index(
            drop=True
        )

    except Exception:

        return None


# ============================================================
# INDICATORS
# ============================================================

def EMA(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def RSI(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


def ATR(df, period=14):

    previous_close = df[
        "close"
    ].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def analyze_timeframe(df):

    if df is None or len(df) < 60:

        return {
            "score": 0.0,
            "trend": "UNKNOWN",
            "atr": np.nan,
            "rsi": np.nan
        }

    close = df["close"]

    ema20 = EMA(close, 20)
    ema50 = EMA(close, 50)
    ema100 = EMA(close, 100)

    rsi_value = RSI(close).iloc[-1]

    atr_value = ATR(df).iloc[-1]

    score = 0.0

    # Price vs EMA20
    if close.iloc[-1] > ema20.iloc[-1]:
        score += 0.25
    else:
        score -= 0.25

    # EMA20 vs EMA50
    if ema20.iloc[-1] > ema50.iloc[-1]:
        score += 0.30
    else:
        score -= 0.30

    # EMA50 vs EMA100
    if ema50.iloc[-1] > ema100.iloc[-1]:
        score += 0.20
    else:
        score -= 0.20

    # RSI
    if 55 <= rsi_value <= 70:
        score += 0.15

    elif 70 < rsi_value <= 78:
        score += 0.05

    elif rsi_value > 78:
        score -= 0.08

    elif 30 <= rsi_value < 45:
        score -= 0.15

    elif rsi_value < 30:
        score += 0.05

    # EMA slope
    old_ema = ema20.iloc[-6]

    slope = (
        ema20.iloc[-1] - old_ema
    ) / max(
        abs(old_ema),
        1e-12
    )

    score += clip(
        slope * 8,
        -0.12,
        0.12
    )

    score = clip(
        score,
        -1,
        1
    )

    if score >= 0.18:

        trend = "BULLISH"

    elif score <= -0.18:

        trend = "BEARISH"

    else:

        trend = "NEUTRAL"

    return {
        "score": score,
        "trend": trend,
        "atr": atr_value,
        "rsi": rsi_value
    }


# ============================================================
# PRICE ACTION
# ============================================================

def price_action(df):

    if df is None or len(df) < 5:

        return 0.0, "UNKNOWN"

    current = df.iloc[-1]
    previous = df.iloc[-2]

    candle_range = max(
        current["high"] - current["low"],
        1e-12
    )

    body = (
        current["close"]
        - current["open"]
    ) / candle_range

    score = clip(
        body * 0.9,
        -1,
        1
    )

    if current["close"] > previous["close"]:

        score += 0.15

    elif current["close"] < previous["close"]:

        score -= 0.15

    score = clip(
        score,
        -1,
        1
    )

    if score > 0.55:
        label = "STRONG_BULLISH"

    elif score > 0.15:
        label = "BULLISH"

    elif score < -0.55:
        label = "STRONG_BEARISH"

    elif score < -0.15:
        label = "BEARISH"

    else:
        label = "NEUTRAL"

    return score, label


# ============================================================
# BREAKOUT
# ============================================================

def breakout(df):

    if df is None or len(df) < 30:

        return 0.0, "UNKNOWN"

    previous_high = df[
        "high"
    ].iloc[-21:-1].max()

    previous_low = df[
        "low"
    ].iloc[-21:-1].min()

    close = df[
        "close"
    ].iloc[-1]

    if close > previous_high * 1.002:

        return 1.0, "BULLISH_BREAKOUT"

    if close < previous_low * 0.998:

        return -1.0, "BEARISH_BREAKOUT"

    return 0.0, "NO_BREAKOUT"


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if df is None or len(df) < 25:

        return 0.0, np.nan

    average_volume = df[
        "volume"
    ].iloc[-21:-1].mean()

    ratio = (
        df["volume"].iloc[-1]
        / max(average_volume, 1e-12)
    )

    pa_score, _ = price_action(df)

    score = clip(
        np.tanh(
            (ratio - 1) / 1.2
        ) * pa_score,
        -1,
        1
    )

    return score, ratio


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance(df):

    if df is None or len(df) < 50:

        return (
            np.nan,
            np.nan,
            0.0
        )

    support = float(
        df["low"].iloc[-51:-1].min()
    )

    resistance = float(
        df["high"].iloc[-51:-1].max()
    )

    close = float(
        df["close"].iloc[-1]
    )

    width = max(
        resistance - support,
        1e-12
    )

    position = (
        close - support
    ) / width

    if position < 0.20:

        score = 0.25

    elif position > 0.80:

        score = -0.25

    else:

        score = 0.0

    return (
        support,
        resistance,
        score
    )


# ============================================================
# FULL TECHNICAL ANALYSIS
# ============================================================

def technical_analysis(
    exchange,
    symbol
):

    timeframes = [
        "15m",
        "1h",
        "4h",
        "1d"
    ]

    weights = {
        "15m": 0.15,
        "1h": 0.25,
        "4h": 0.35,
        "1d": 0.25
    }

    details = {}
    dataframes = {}

    for timeframe in timeframes:

        df = get_ohlcv(
            exchange,
            symbol,
            timeframe
        )

        dataframes[timeframe] = df

        details[timeframe] = (
            analyze_timeframe(df)
        )

    timeframe_score = 0.0

    for timeframe in timeframes:

        timeframe_score += (
            weights[timeframe]
            * details[timeframe]["score"]
        )

    base_df = dataframes.get("1h")

    if base_df is None:

        base_df = dataframes.get("4h")

    if base_df is None:

        base_df = dataframes.get("15m")

    pa_score, pa_label = price_action(
        base_df
    )

    breakout_score, breakout_label = breakout(
        base_df
    )

    volume_score, volume_ratio = volume_analysis(
        base_df
    )

    support, resistance, sr_score = (
        support_resistance(base_df)
    )

    technical_raw = (
        timeframe_score * 28
        + pa_score * 7
        + breakout_score * 7
        + volume_score * 4
        + sr_score * 4
    )

    technical_score = clip(
        50 + technical_raw,
        0,
        100
    )

    atr_1h = details["1h"]["atr"]

    if not math.isfinite(atr_1h):

        atr_1h = details["4h"]["atr"]

    return {

        "tf_score": timeframe_score,

        "technical_score": technical_score,

        "15m_trend":
            details["15m"]["trend"],

        "1h_trend":
            details["1h"]["trend"],

        "4h_trend":
            details["4h"]["trend"],

        "1d_trend":
            details["1d"]["trend"],

        "atr": atr_1h,

        "price_action": pa_label,

        "price_action_score": pa_score,

        "breakout": breakout_label,

        "breakout_score": breakout_score,

        "volume_ratio": volume_ratio,

        "volume_score": volume_score,

        "support": support,

        "resistance": resistance,

        "sr_score": sr_score
    }


# ============================================================
# BTC REGIME
# ============================================================

def btc_regime(spot):

    exchange_id, exchange = (
        find_spot_exchange(
            spot,
            "BTC/USDT"
        )
    )

    if exchange is None:

        return 0.0, "UNKNOWN"

    df = get_ohlcv(
        exchange,
        "BTC/USDT",
        "4h"
    )

    analysis = analyze_timeframe(df)

    return (
        analysis["score"],
        analysis["trend"]
    )


# ============================================================
# FUTURES SYMBOL
# ============================================================

def futures_symbol(
    exchange,
    base
):

    possible = [
        f"{base}/USDT:USDT",
        f"{base}/USDT"
    ]

    for symbol in possible:

        market = exchange.markets.get(
            symbol
        )

        if not market:
            continue

        if (
            market.get("swap")
            or market.get("future")
        ):

            return symbol

    return None


# ============================================================
# FUNDING + OPEN INTEREST
# ============================================================

def derivatives(
    futures,
    base
):

    funding_rates = []
    open_interests = []

    funding_names = []
    oi_names = []

    # maximum 3 futures exchanges
    for exchange_id, exchange in futures[:3]:

        symbol = futures_symbol(
            exchange,
            base
        )

        if not symbol:
            continue

        # Funding
        try:

            if exchange.has.get(
                "fetchFundingRate"
            ):

                result = (
                    exchange.fetch_funding_rate(
                        symbol
                    )
                )

                rate = number(
                    result.get(
                        "fundingRate"
                    )
                )

                if math.isfinite(rate):

                    funding_rates.append(rate)

                    funding_names.append(
                        exchange_id.upper()
                    )

        except Exception as e:

            log(
                f"FUNDING FAILED "
                f"{exchange_id} {base}: "
                f"{type(e).__name__}"
            )

        # Open interest
        try:

            if exchange.has.get(
                "fetchOpenInterest"
            ):

                result = (
                    exchange.fetch_open_interest(
                        symbol
                    )
                )

                oi_value = number(
                    result.get(
                        "openInterestValue"
                    )
                )

                if not math.isfinite(
                    oi_value
                ):

                    oi_value = number(
                        result.get(
                            "openInterestAmount"
                        )
                    )

                if math.isfinite(
                    oi_value
                ):

                    open_interests.append(
                        oi_value
                    )

                    oi_names.append(
                        exchange_id.upper()
                    )

        except Exception as e:

            log(
                f"OI FAILED "
                f"{exchange_id} {base}: "
                f"{type(e).__name__}"
            )

    if funding_rates:

        average_funding = float(
            np.mean(funding_rates)
        )

    else:

        average_funding = 0.0

    # Positive funding = long crowding
    # Negative funding = short crowding

    if average_funding > 0.0008:

        funding_signal = "BEARISH"

    elif average_funding < -0.0008:

        funding_signal = "BULLISH"

    else:

        funding_signal = "NEUTRAL"

    total_oi = (
        float(np.sum(open_interests))
        if open_interests
        else 0.0
    )

    all_names = sorted(
        set(
            funding_names
            + oi_names
        )
    )

    return {

        "funding": average_funding,

        "funding_signal":
            funding_signal,

        "open_interest":
            total_oi,

        "futures_exchanges":
            len(all_names),

        "futures_names":
            ",".join(all_names)
    }


# ============================================================
# ORDER BOOK
# ============================================================

def orderbook(
    spot,
    symbol
):
    imbalances = []
    names = []

    preferred = [
        "binance",
        "okx",
        "bybit",
        "kucoin"
    ]

    for preferred_id in preferred:

        for exchange in spot:

            exchange_id = getattr(
                exchange,
                "id",
                ""
            ).lower()

            if exchange_id != preferred_id:
                continue

            try:
                markets = getattr(
                    exchange,
                    "markets",
                    {}
                ) or {}

                if symbol not in markets:
                    continue

                has = getattr(
                    exchange,
                    "has",
                    {}
                ) or {}

                if not isinstance(has, dict):
                    has = {}

                if has.get("fetchOrderBook") is False:
                    continue

                book = exchange.fetch_order_book(
                    symbol,
                    limit=20
                )

                if not isinstance(book, dict):
                    log(
                        f"ORDERBOOK EMPTY "
                        f"{exchange_id} {symbol}"
                    )
                    continue

                bids = book.get(
                    "bids",
                    []
                ) or []

                asks = book.get(
                    "asks",
                    []
                ) or []

                bids = bids[:20]
                asks = asks[:20]

                if not bids or not asks:
                    continue

                bid_value = sum(
                    number(price, 0)
                    * number(amount, 0)
                    for price, amount in bids
                )

                ask_value = sum(
                    number(price, 0)
                    * number(amount, 0)
                    for price, amount in asks
                )

                total = (
                    bid_value
                    + ask_value
                )

                if total <= 0:
                    continue

                imbalance = (
                    bid_value
                    - ask_value
                ) / total

                imbalances.append(
                    imbalance
                )

                names.append(
                    exchange_id.upper()
                )

            except Exception as e:
                log(
                    f"ORDERBOOK FAILED "
                    f"{exchange_id} {symbol}: "
                    f"{type(e).__name__}: {e}"
                )

    if not imbalances:
        return {
            "orderbook_imbalance": 0.0,
            "orderbook_signal": "UNKNOWN",
            "orderbook_exchanges": 0,
            "orderbook_names": ""
        }

    average = float(
        np.mean(imbalances)
    )

    if average > 0.10:
        signal = "BULLISH"

    elif average < -0.10:
        signal = "BEARISH"

    else:
        signal = "NEUTRAL"

    return {
        "orderbook_imbalance": average,
        "orderbook_signal": signal,
        "orderbook_exchanges": len(names),
        "orderbook_names": ",".join(names) 
    }

# ============================================================
# CMC HTTP
# ============================================================

def cmc_get(endpoint, params=None, tries=3):
    url = CMC_BASE + endpoint

    for attempt in range(tries):
        try:
            response = session.get(
                url,
                params=params or {},
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 429:
                if attempt < tries - 1:
                    time.sleep(
                        min(2 ** attempt, 15)
                    )
                    continue

                return None

            if not response.ok:
                log(
                    f"CMC HTTP {response.status_code} "
                    f"{endpoint}: "
                    f"{response.text[:500]}"
                )

                if response.status_code in {
                    400,
                    401,
                    403,
                    404
                }:
                    return None

                if attempt < tries - 1:
                    time.sleep(
                        2 ** attempt
                    )
                    continue

                return None

            data = response.json()

            status = data.get(
                "status",
                {}
            )

            error_code = status.get(
                "error_code"
            )

            error_message = status.get(
                "error_message"
            )

            if error_code not in (
                None,
                0,
                "0"
            ):
                log(
                    f"CMC API ERROR "
                    f"{endpoint}: "
                    f"code={error_code} "
                    f"message={error_message}"
                )
                return None

            return data

        except Exception as e:
            log(
                f"CMC ERROR "
                f"{endpoint}: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < tries - 1:
                time.sleep(
                    2 ** attempt
                )

    return None
# ============================================================
# FUNDAMENTALS
# ============================================================

def calculate_fundamental_score(row):

    score = 50.0

    reasons = []

    rank = number(
        row.get("cmc_rank")
    )

    market_cap = number(
        row.get("market_cap")
    )

    fdv = number(
        row.get("fdv")
    )

    circulating = number(
        row.get("circulating_supply")
    )

    maximum = number(
        row.get("max_supply")
    )

    pairs = number(
        row.get("market_pairs")
    )

    age = number(
        row.get("asset_age_days")
    )

    # Rank
    if math.isfinite(rank):

        if rank <= 50:

            score += 8
            reasons.append("TOP50")

        elif rank <= 100:

            score += 6
            reasons.append("TOP100")

        elif rank <= 250:

            score += 4
            reasons.append("TOP250")

        elif rank <= 500:

            score += 2

        elif rank > 1000:

            score -= 3

    # Market cap / FDV
    if (
        math.isfinite(market_cap)
        and math.isfinite(fdv)
        and fdv > 0
    ):

        ratio = (
            market_cap / fdv
        )

        row["mc_fdv_ratio"] = ratio

        if ratio >= 0.80:

            score += 5
            reasons.append("LOW_FDV_GAP")

        elif ratio >= 0.50:

            score += 3

        elif ratio < 0.25:

            score -= 5
            reasons.append("HIGH_FDV_RISK")

    # Circulating / max supply
    if (
        math.isfinite(circulating)
        and math.isfinite(maximum)
        and maximum > 0
    ):

        supply_ratio = (
            circulating / maximum
        )

        row["supply_ratio"] = supply_ratio

        if supply_ratio >= 0.80:

            score += 4

        elif supply_ratio >= 0.50:

            score += 2

        elif supply_ratio < 0.25:

            score -= 4

    # Market pairs
    if math.isfinite(pairs):

        if pairs >= 100:

            score += 3

        elif pairs >= 50:

            score += 2

        elif pairs < 10:

            score -= 2

    # Age
    if math.isfinite(age):

        if age > 365:

            score += 1

        elif age < 90:

            score -= 1

    score = clip(
        score,
        0,
        100
    )

    if score >= 70:

        rating = "STRONG"

    elif score >= 45:

        rating = "NEUTRAL"

    else:

        rating = "WEAK"

    row["fundamental_score"] = round(
        score,
        2
    )

    row["fundamental_rating"] = rating

    row["fundamental_reason"] = (
        "|".join(reasons)
        if reasons
        else "NEUTRAL"
    )

    return row


# ============================================================
# CMC FUNDAMENTAL DATA
# ============================================================

def fundamentals(symbols):

    output = {}

    for symbol in symbols:

        output[symbol] = {

            "name":
                symbol.split("/")[0],

            "cmc_rank":
                np.nan,

            "market_cap":
                np.nan,

            "circulating_supply":
                np.nan,

            "total_supply":
                np.nan,

            "max_supply":
                np.nan,

            "fdv":
                np.nan,

            "mc_fdv_ratio":
                np.nan,

            "supply_ratio":
                np.nan,

            "market_pairs":
                np.nan,

            "asset_age_days":
                np.nan,

            "fundamental_score":
                50,

            "fundamental_rating":
                "UNKNOWN",

            "fundamental_status":
                "UNAVAILABLE",

            "fundamental_reason":
                "CMC_UNAVAILABLE"
        }

    base_symbols = [
        s.split("/")[0].upper()
        for s in symbols
    ]

    # --------------------------------------------------------
    # CMC MAP
    # --------------------------------------------------------

    map_response = cmc_get(
        "/v1/cryptocurrency/map",
        {
            "symbol":
                ",".join(base_symbols),

            "listing_status":
                "active"
        }
    )

    map_data = []

    if isinstance(
        map_response,
        dict
    ):

        map_data = map_response.get(
            "data",
            []
        )

    selected = {}

    for item in map_data:

        symbol = text(
            item.get("symbol")
        ).upper()

        if symbol not in base_symbols:
            continue

        if not item.get(
            "is_active",
            1
        ):
            continue

        rank = number(
            item.get(
                "rank"
            ),
            999999
        )

        if (
            symbol not in selected
            or rank < selected[symbol][0]
        ):

            selected[symbol] = (
                rank,
                item
            )

    # --------------------------------------------------------
    # CMC QUOTES
    # --------------------------------------------------------

    ids = []

    for value in selected.values():

        item = value[1]

        if item.get("id"):

            ids.append(
                str(item["id"])
            )

    quotes = {}

    if ids:

        quote_response = cmc_get(
            "/v3/cryptocurrency/quotes/latest",
            {
                "id":
                    ",".join(ids),

                "convert":
                    "USD"
            }
        )

        if isinstance(
            quote_response,
            dict
        ):

            quotes = quote_response.get(
                "data",
                {}
            )

    # --------------------------------------------------------
    # BUILD FUNDAMENTALS
    # --------------------------------------------------------

    for symbol in symbols:

        base = symbol.split(
            "/"
        )[0].upper()

        if base not in selected:
            continue

        metadata = selected[
            base
        ][1]

        coin_id = str(
            metadata.get("id")
        )

        quote = {}

        if isinstance(
            quotes,
            dict
        ):

            quote = quotes.get(
                coin_id,
                {}
            )

        usd = quote.get(
            "quote",
            {}
        ).get(
            "USD",
            {}
        )

        if not usd:

            continue

        age_days = np.nan

        date_added = metadata.get(
            "date_added"
        )

        if date_added:
            try:
                added = datetime.fromisoformat(
                    date_added.replace(
                        "Z",
                        "+00:00"
                    )
                )
                age_days = (datetime.now(timezone.utc) - added).days
            except Exception:
                age_days = np.nanp

# ============================================================
# GLOBAL MARKET DATA
# ============================================================

def global_market_data():

    result = {

        "global_market_cap":
            np.nan,

        "btc_dominance":
            np.nan,

        "market_volume":
            np.nan,

        "fear_greed":
            np.nan,

        "fear_greed_class":
            "UNKNOWN"
    }

    # Global market
    response = cmc_get(
        "/v1/global-metrics/quotes/latest",
        {
            "convert":
                "USD"
        }
    )

    try:

        data = response[
            "data"
        ]

        usd = data[
            "quote"
        ][
            "USD"
        ]

        result[
            "global_market_cap"
        ] = number(
            usd.get(
                "total_market_cap"
            )
        )

        result[
            "market_volume"
        ] = number(
            usd.get(
                "total_volume_24h"
            )
        )

        result[
            "btc_dominance"
        ] = number(
            data.get(
                "btc_dominance"
            )
        )

    except Exception as e:

        log(
            f"GLOBAL CMC PARSE ERROR: "
            f"{type(e).__name__}: {e}"
        )

    # Fear & Greed
    response = cmc_get(
        "/v3/fear-and-greed/latest"
    )

    try:

        data = response.get(
            "data"
        )

        if isinstance(
            data,
            list
        ):

            data = data[0]

        result[
            "fear_greed"
        ] = number(
            data.get(
                "value"
            )
        )

        result[
            "fear_greed_class"
        ] = text(
            data.get(
                "value_classification"
            ),
            "UNKNOWN"
        )

    except Exception as e:

        log(
            f"FEAR GREED PARSE ERROR: "
            f"{type(e).__name__}: {e}"
        )

    return result


# ============================================================
# NEWS
# ============================================================

POSITIVE_WORDS = [
    "approval",
    "approved",
    "adoption",
    "adopt",
    "partnership",
    "integration",
    "launch",
    "launched",
    "upgrade",
    "inflow",
    "bullish",
    "record high",
    "breakout",
    "listing",
    "listed",
    "institutional",
    "mainnet",
    "airdrop"
]

NEGATIVE_WORDS = [
    "hack",
    "hacked",
    "exploit",
    "breach",
    "lawsuit",
    "ban",
    "banned",
    "delist",
    "delisted",
    "insolvency",
    "bankrupt",
    "outage",
    "attack",
    "stolen",
    "scam",
    "unlock",
    "liquidation",
    "collapse"
]

HIGH_IMPACT_WORDS = [
    "hack",
    "exploit",
    "breach",
    "lawsuit",
    "ban",
    "delist",
    "insolvency",
    "bankrupt",
    "stolen",
    "sec",
    "etf approval",
    "approval"
]

MEDIUM_IMPACT_WORDS = [
    "partnership",
    "listing",
    "delisted",
    "unlock",
    "outage",
    "upgrade",
    "launch",
    "inflow",
    "liquidation"
]


def clean_html(value):

    value = text(value)

    value = re.sub(
        r"<[^>]+>",
        " ",
        value
    )

    value = value.replace(
        "&amp;",
        "&"
    )

    return value.strip()


def load_news():

    news = []

    for feed_url in RSS_FEEDS:

        try:

            response = session.get(
                feed_url,
                timeout=8
            )

            response.raise_for_status()

            import xml.etree.ElementTree as ET

            root = ET.fromstring(
                response.content
            )

            items = root.findall(
                ".//item"
            )

            if items:

                for item in items[:40]:

                    title = clean_html(
                        item.findtext(
                            "title",
                            default=""
                        )
                    )

                    summary = clean_html(
                        item.findtext(
                            "description",
                            default=""
                        )
                    )

                    if title:

                        news.append({
                            "title": title,
                            "summary": summary
                        })

            else:

                namespace = {
                    "a":
                    "http://www.w3.org/2005/Atom"
                }

                entries = root.findall(
                    ".//a:entry",
                    namespace
                )

                for entry in entries[:40]:

                    title = clean_html(
                        entry.findtext(
                            "a:title",
                            default="",
                            namespaces=namespace
                        )
                    )

                    summary = clean_html(
                        entry.findtext(
                            "a:summary",
                            default="",
                            namespaces=namespace
                        )
                    )

                    if title:

                        news.append({
                            "title": title,
                            "summary": summary
                        })

        except Exception as e:

            log(
                f"NEWS FAILED: "
                f"{feed_url} | "
                f"{type(e).__name__}"
            )

    # Remove duplicates
    seen = set()
    cleaned = []

    for item in news:

        key = item[
            "title"
        ].lower()

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(item)

    return cleaned[:250]


def coin_news(
    coin_name,
    symbol,
    news
):

    coin_name = text(
        coin_name
    )

    base = symbol.split(
        "/"
    )[0].upper()

    matched = []

    # Avoid using short symbols like ONE,
    # UNI and RAY because they create false matches.
    symbol_pattern = None

    if len(base) >= 4:

        symbol_pattern = re.compile(
            r"\b"
            + re.escape(base)
            + r"\b",
            re.IGNORECASE
        )

    for item in news:

        combined = (
            item["title"]
            + " "
            + item["summary"]
        )

        name_match = False

        if len(coin_name) >= 4:

            name_match = bool(
                re.search(
                    re.escape(coin_name),
                    combined,
                    re.IGNORECASE
                )
            )

        symbol_match = False

        if symbol_pattern:

            symbol_match = bool(
                symbol_pattern.search(
                    combined
                )
            )

        if (
            name_match
            or symbol_match
        ):

            matched.append(item)

    if not matched:

        return {

            "news_score": 0.0,

            "news_sentiment":
                "NO_NEWS",

            "news_impact":
                "NONE",

            "news_count": 0,

            "news_headlines":
                ""
        }

    scores = []
    weights = []
    impacts = []

    for item in matched[:10]:

        combined = (
            item["title"]
            + " "
            + item["summary"]
        ).lower()

        positive_count = sum(
            combined.count(word)
            for word in POSITIVE_WORDS
        )

        negative_count = sum(
            combined.count(word)
            for word in NEGATIVE_WORDS
        )

        if positive_count > negative_count:

            score = 0.7

        elif negative_count > positive_count:

            score = -0.7

        else:

            score = 0.0

        if any(
            word in combined
            for word in HIGH_IMPACT_WORDS
        ):

            impact = "HIGH"

        elif any(
            word in combined
            for word in MEDIUM_IMPACT_WORDS
        ):

            impact = "MEDIUM"

        else:

            impact = "LOW"

        if impact == "HIGH":

            weight = 1.5

        elif impact == "MEDIUM":

            weight = 1.0

        else:

            weight = 0.5

        scores.append(score)
        weights.append(weight)
        impacts.append(impact)

    final_score = float(
        np.average(
            scores,
            weights=weights
        )
    )

    if final_score > 0.55:

        sentiment = "STRONGLY_BULLISH"

    elif final_score > 0.15:

        sentiment = "BULLISH"

    elif final_score < -0.55:

        sentiment = "STRONGLY_BEARISH"

    elif final_score < -0.15:

        sentiment = "BEARISH"

    else:

        sentiment = "NEUTRAL"

    if "HIGH" in impacts:

        impact = "HIGH"

    elif "MEDIUM" in impacts:

        impact = "MEDIUM"

    else:

        impact = "LOW"

    return {

        "news_score":
            round(final_score, 3),

        "news_sentiment":
            sentiment,

        "news_impact":
            impact,

        "news_count":
            len(matched),

        "news_headlines":
            " || ".join(
                x["title"]
                for x in matched[:5]
            )
    }


# ============================================================
# LEARNING MEMORY
# ============================================================

LEARNING_FILE = "ai_learning.json"


def load_learning():

    try:

        with open(
            LEARNING_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        data.setdefault(
            "version",
            1
        )

        data.setdefault(
            "pending",
            []
        )

        data.setdefault(
            "total",
            0
        )

        data.setdefault(
            "hits",
            0
        )

        return data

    except Exception:

        return {

            "version": 1,

            "pending": [],

            "total": 0,

            "hits": 0
        }


def save_learning(data):

    with open(
        LEARNING_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )


def update_learning(
    data,
    current_prices
):

    current_time = time.time()

    remaining = []

    for prediction in data.get(
        "pending",
        []
    ):

        created = number(
            prediction.get(
                "created"
            ),
            current_time
        )

        age_hours = (
            current_time - created
        ) / 3600

        symbol = prediction.get(
            "symbol"
        )

        if (
            age_hours >= 4
            and symbol in current_prices
        ):

            entry = number(
                prediction.get(
                    "entry"
                )
            )

            current = number(
                current_prices[
                    symbol
                ]
            )

            signal = prediction.get(
                "signal"
            )

            if (
                math.isfinite(entry)
                and entry > 0
                and math.isfinite(current)
            ):

                return_pct = (
                    current - entry
                ) / entry

                if signal == "LONG":

                    hit = (
                        return_pct > 0.001
                    )

                else:

                    hit = (
                        return_pct < -0.001
                    )

                data["total"] += 1

                if hit:

                    data["hits"] += 1

                continue

        # Remove very old unresolved predictions
        if age_hours < 12:

            remaining.append(
                prediction
            )

    data["pending"] = remaining[-5000:]

    return data


def learning_stats(data):

    total = int(
        data.get(
            "total",
            0
        )
    )

    hits = int(
        data.get(
            "hits",
            0
        )
    )

    if total:

        hit_rate = hits / total

    else:

        hit_rate = 0.5

    # Learning only starts after enough history.
    # Maximum adjustment is +/-10%.
    if total >= 20:

        adjustment = clip(
            (hit_rate - 0.5) * 0.2,
            -0.10,
            0.10
        )

    else:

        adjustment = 0.0

    return (
        hit_rate,
        adjustment
    )


def add_predictions(
    data,
    results
):

    existing = {
        (
            item.get("symbol"),
            item.get("signal")
        )
        for item in data.get(
            "pending",
            []
        )
    }

    actionable = [
        result
        for result in results
        if result["signal"]
        in ("LONG", "SHORT")
    ]

    actionable.sort(
        key=lambda x:
        x["signal_strength"],
        reverse=True
    )

    for result in actionable[:10]:

        key = (
            result["symbol"],
            result["signal"]
        )

        if key in existing:
            continue

        data["pending"].append({

            "symbol":
                result["symbol"],

            "signal":
                result["signal"],

            "entry":
                result["price"],

            "created":
                time.time()
        })

    data["pending"] = data[
        "pending"
    ][-5000:]

    return data


# ============================================================
# FINAL AI SCORE
# ============================================================

def create_result(
    base_row,
    technical,
    fundamental,
    news,
    order_book,
    derivatives_data,
    btc,
    global_data,
    learning_adjustment
):

    price = base_row["price"]

    technical_direction = (
        technical["technical_score"]
        - 50
    ) / 50

    fundamental_score = number(
        fundamental.get(
            "fundamental_score"
        ),
        50
    )

    fundamental_direction = (
        fundamental_score
        - 50
    ) / 50

    orderbook_imbalance = number(
        order_book.get(
            "orderbook_imbalance"
        ),
        0
    )

    funding_signal = (
        derivatives_data[
            "funding_signal"
        ]
    )

    if funding_signal == "BEARISH":

        funding_direction = -1

    elif funding_signal == "BULLISH":

        funding_direction = 1

    else:

        funding_direction = 0

    btc_score = number(
        btc[0],
        0
    )

    fear_greed = number(
        global_data.get(
            "fear_greed"
        )
    )

    fear_greed_direction = 0

    if (
        math.isfinite(fear_greed)
        and fear_greed >= 80
    ):

        fear_greed_direction = -0.5

    elif (
        math.isfinite(fear_greed)
        and fear_greed <= 20
        and fear_greed > 0
    ):

        fear_greed_direction = 0.5

    # --------------------------------------------------------
    # FINAL RAW SCORE
    # --------------------------------------------------------

    raw_score = (

        technical_direction
        * 32
        * (1 + learning_adjustment)

        +

        fundamental_direction
        * 7

        +

        news["news_score"]
        * 6

        +

        orderbook_imbalance
        * 5

        +

        funding_direction
        * 3

        +

        btc_score
        * 4

        +

        fear_greed_direction
        * 2
    )

    raw_score = clip(
        raw_score,
        -45,
        45
    )

    long_score = clip(
        50 + raw_score,
        0,
        95
    )

    short_score = clip(
        50 - raw_score,
        0,
        95
    )

    trends = [
        technical["15m_trend"],
        technical["1h_trend"],
        technical["4h_trend"],
        technical["1d_trend"]
    ]

    bullish_count = sum(
        trend in (
            "BULLISH",
            "STRONG_BULLISH"
        )
        for trend in trends
    )

    bearish_count = sum(
        trend in (
            "BEARISH",
            "STRONG_BEARISH"
        )
        for trend in trends
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    if (
        long_score >= 63
        and long_score
        > short_score + 5
        and bullish_count >= 2
    ):

        signal = "LONG"

    elif (
        short_score >= 63
        and short_score
        > long_score + 5
        and bearish_count >= 2
    ):

        signal = "SHORT"

    else:

        signal = "NO TRADE"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    agreement = (
        abs(
            bullish_count
            - bearish_count
        ) / 4
    )

    confidence = (
        50
        + abs(raw_score) * 0.75
        + agreement * 6
    )

    if news["news_impact"] == "HIGH":

        confidence += 2

    # Unknown fundamental data slightly reduces confidence
    if fundamental.get(
        "fundamental_status"
    ) != "OK":

        confidence -= 3

    confidence = clip(
        confidence,
        50,
        95
    )

    # --------------------------------------------------------
    # STOP LOSS / TAKE PROFIT
    # --------------------------------------------------------

    atr_value = number(
        technical.get("atr")
    )

    if (
        not math.isfinite(atr_value)
        or atr_value <= 0
    ):

        atr_value = price * 0.02

    if signal == "LONG":

        stop_loss = (
            price
            - 1.5 * atr_value
        )

        take_profit = (
            price
            + 3 * atr_value
        )

    elif signal == "SHORT":

        stop_loss = (
            price
            + 1.5 * atr_value
        )

        take_profit = (
            price
            - 3 * atr_value
        )

        if take_profit <= 0:

            take_profit = price * 0.95

    else:

        stop_loss = np.nan
        take_profit = np.nan

    result = {

        **base_row,

        **technical,

        **fundamental,

        **news,

        **order_book,

        **derivatives_data,

        "btc_score":
            btc[0],

        "btc_regime":
            btc[1],

        **global_data,

        "signal":
            signal,

        "long_score":
            round(
                long_score,
                2
            ),

        "short_score":
            round(
                short_score,
                2
            ),

        "signal_strength":
            round(
                max(
                    long_score,
                    short_score
                ),
                2
            ),

        "confidence":
            round(
                confidence,
                2
            ),

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "learning_adjustment":
            learning_adjustment,

        "reason":
            (
                f"TF:{technical['tf_score']:.2f} | "
                f"TECH:{technical['technical_score']:.1f} | "
                f"FUND:{fundamental.get('fundamental_score', 50)} | "
                f"NEWS:{news['news_sentiment']} | "
                f"OB:{order_book['orderbook_signal']} | "
                f"FUNDING:{derivatives_data['funding_signal']} | "
                f"BTC:{btc[1]} | "
                f"LEARN:{learning_adjustment:+.3f}"
            ),

        "timestamp":
            now().isoformat()
    }

    return result


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(results):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not token or not chat_id:

        return

    actionable = [
        result
        for result in results
        if result["signal"]
        in ("LONG", "SHORT")
    ]

    actionable.sort(
        key=lambda x:
        x["signal_strength"],
        reverse=True
    )

    actionable = actionable[:5]

    if not actionable:

        return

    lines = [
        "Crypto Master AI V9.1",
        ""
    ]

    for result in actionable:

        lines.append(
            f"{result['signal']} "
            f"{result['symbol']} | "
            f"Strength "
            f"{result['signal_strength']} | "
            f"Confidence "
            f"{result['confidence']}"
        )

        lines.append(
            f"Price: {result['price']}"
        )

        lines.append(
            f"SL: {result['stop_loss']}"
        )

        lines.append(
            f"TP: {result['take_profit']}"
        )

        lines.append("")

    message = "\n".join(
        lines
    )

    try:

        requests.post(
            f"https://api.telegram.org/bot"
            f"{token}/sendMessage",

            json={
                "chat_id":
                    chat_id,

                "text":
                    message
            },

            timeout=10
        )

    except Exception as e:

        log(
            f"TELEGRAM ERROR: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    log(
        "======================================"
    )

    log(
        "CRYPTO MASTER AI V9.1 START"
    )

    log(
        "======================================"
    )

    # --------------------------------------------------------
    # EXCHANGES
    # --------------------------------------------------------

    spot, futures = build_exchanges()

    if not spot:

        raise RuntimeError(
            "NO SPOT EXCHANGE AVAILABLE"
        )

    log(
        "SPOT EXCHANGES: "
        + ",".join(
            x[0].upper()
            for x in spot
        )
    )

    log(
        "FUTURES EXCHANGES: "
        + ",".join(
            x[0].upper()
            for x in futures
        )
    )

    # --------------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------------

    candidates = discover_candidates(
        spot
    )

    if not candidates:

        raise RuntimeError(
            "NO LIQUID USDT CANDIDATES"
        )

    log(
        f"CANDIDATES FOUND: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # GLOBAL DATA
    # --------------------------------------------------------

    global_data = (
        global_market_data()
    )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    news_data = load_news()

    log(
        f"NEWS ITEMS: "
        f"{len(news_data)}"
    )

    # --------------------------------------------------------
    # FUNDAMENTALS
    # --------------------------------------------------------

    symbols = [
        item["symbol"]
        for item in candidates
    ]

    fundamentals_data = fundamentals(
        symbols
    )

    # --------------------------------------------------------
    # BTC
    # --------------------------------------------------------

    btc = btc_regime(
        spot
    )

    log(
        f"BTC REGIME: "
        f"{btc[1]}"
    )

    # --------------------------------------------------------
    # LEARNING
    # --------------------------------------------------------

    learning = load_learning()

    current_prices = {}

    results = []

    # --------------------------------------------------------
    # ANALYZE COINS
    # --------------------------------------------------------

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        symbol = candidate[
            "symbol"
        ]

        exchange_id, exchange = (
            find_spot_exchange(
                spot,
                symbol
            )
        )

        if exchange is None:

            log(
                f"SKIP {symbol}: "
                f"NO SPOT EXCHANGE"
            )

            continue

        try:

            ticker = exchange.fetch_ticker(
                symbol
            )

            price = number(
                ticker.get(
                    "last"
                )
            )

            if (
                not math.isfinite(price)
                or price <= 0
            ):

                continue

            base = symbol.split(
                "/"
            )[0]

            base_row = {

                **candidate,

                "price":
                    price,

                "analysis_exchange":
                    exchange_id.upper()
            }

            # Technical
            technical = technical_analysis(
                exchange,
                symbol
            )

            # Order book
            order_book = orderbook(
                spot,
                symbol
            )

            # Futures
            derivative_data = derivatives(
                futures,
                base
            )

            # Fundamental
            fundamental = (
                (fundamentals_data or {}).get(
                    symbol,
                    {
                        "name": base,
                        "fundamental_score": 50,
                        "fundamental_rating":
                            "UNKNOWN",
                        "fundamental_status":
                            "UNAVAILABLE",
                        "fundamental_reason":
                            "CMC_UNAVAILABLE"
                    }
                )
            )

            coin_name = fundamental.get(
                "name",
                base
            )

            # News
            news = coin_news(
                coin_name,
                symbol,
                news_data
            )

            # Initial result
            result = create_result(
                base_row,
                technical,
                fundamental,
                news,
                order_book,
                derivative_data,
                btc,
                global_data,
                0.0
            )

            current_prices[
                symbol
            ] = price

            results.append(
                result
            )

            log(
                f"[{index}/"
                f"{len(candidates)}] "
                f"{symbol} -> "
                f"{result['signal']} "
                f"{result['signal_strength']}"
            )

        except Exception as e:

            log(
                f"COIN FAILED "
                f"{symbol}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    if not results:

        raise RuntimeError(
            "SCAN PRODUCED ZERO RESULTS"
        )

    # --------------------------------------------------------
    # UPDATE LEARNING
    # --------------------------------------------------------

    learning = update_learning(
        learning,
        current_prices
    )

    hit_rate, learning_adjustment = (
        learning_stats(
            learning
        )
    )

    # --------------------------------------------------------
    # APPLY LEARNING
    # --------------------------------------------------------

    for result in results:

        technical_score = number(
            result.get(
                "technical_score"
            ),
            50
        )

        fundamental_score = number(
            result.get(
                "fundamental_score"
            ),
            50
        )

        news_score = number(
            result.get(
                "news_score"
            ),
            0
        )

        orderbook_imbalance = number(
            result.get(
                "orderbook_imbalance"
            ),
            0
        )

        funding_signal = result.get(
            "funding_signal"
        )

        if funding_signal == "BEARISH":

            funding_direction = -1

        elif funding_signal == "BULLISH":

            funding_direction = 1

        else:

            funding_direction = 0

        btc_score = number(
            result.get(
                "btc_score"
            ),
            0
        )

        raw = (

            (
                (technical_score - 50)
                / 50
            )
            * 32
            * (
                1
                + learning_adjustment
            )

            +

            (
                (fundamental_score - 50)
                / 50
            )
            * 7

            +

            news_score * 6

            +

            orderbook_imbalance * 5

            +

            funding_direction * 3

            +

            btc_score * 4
        )

        raw = clip(
            raw,
            -45,
            45
        )

        result["long_score"] = round(
            clip(
                50 + raw,
                0,
                95
            ),
            2
        )

        result["short_score"] = round(
            clip(
                50 - raw,
                0,
                95
            ),
            2
        )

        result["signal_strength"] = round(
            max(
                result["long_score"],
                result["short_score"]
            ),
            2
        )

        result[
            "learning_total"
        ] = learning["total"]

        result[
            "learning_hit_rate"
        ] = round(
            hit_rate,
            4
        )

        result[
            "learning_adjustment"
        ] = round(
            learning_adjustment,
            4
        )

    # --------------------------------------------------------
    # SAVE NEW PREDICTIONS
    # --------------------------------------------------------

    learning = add_predictions(
        learning,
        results
    )

    save_learning(
        learning
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        results
    )

    order = {
        "LONG": 0,
        "SHORT": 1,
        "NO TRADE": 2
    }

    dataframe["_order"] = (
        dataframe["signal"]
        .map(order)
        .fillna(9)
    )

    dataframe = dataframe.sort_values(
        [
            "_order",
            "signal_strength"
        ],
        ascending=[
            True,
            False
        ]
    )

    dataframe = dataframe.drop(
        columns=["_order"]
    )

    dataframe.to_csv(
        "crypto_scan_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    send_telegram(
        results
    )

    log(
        "======================================"
    )

    log(
        f"SCAN COMPLETE | "
        f"ROWS={len(dataframe)} | "
        f"LEARNING_TOTAL="
        f"{learning['total']} | "
        f"HIT_RATE="
        f"{hit_rate:.2%}"
    )

    log(
        "======================================"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        log(
            f"FATAL ERROR: "
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        raise
