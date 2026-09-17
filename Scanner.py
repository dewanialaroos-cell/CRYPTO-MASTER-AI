import os
import re
import time
import math
import requests
import ccxt
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


# ============================================================
# CRYPTO MASTER AI V9.1
#
# Technical
# Fundamental
# News
# Order Book
# Futures
# Binance
# OKX
# Bybit
# KuCoin
# BTC Regime
# Fear & Greed
#
# IMPORTANT:
# confidence != win probability
# ============================================================


MAX_COINS = 60

TIMEFRAMES = [
    "15m",
    "1h",
    "4h",
    "1d"
]


# ============================================================
# STABLECOINS
# ============================================================

STABLECOINS = {
    "USDT",
    "USDC",
    "FDUSD",
    "DAI",
    "TUSD",
    "USDE",
    "USDD",
    "USDP",
    "PYUSD",
    "USDS",
    "USDG",
    "RLUSD",
    "EURC"
}


# ============================================================
# NEWS FEEDS
# ============================================================

NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://cryptoslate.com/feed/"
]


POSITIVE_WORDS = [
    "approval",
    "approved",
    "adoption",
    "partnership",
    "launch",
    "listing",
    "buy",
    "buying",
    "inflow",
    "bullish",
    "surge",
    "growth",
    "upgrade",
    "integration",
    "institutional",
    "etf",
    "funding",
    "investment",
    "record",
    "mainnet",
    "expansion"
]


NEGATIVE_WORDS = [
    "hack",
    "hacked",
    "exploit",
    "attack",
    "lawsuit",
    "ban",
    "banned",
    "delist",
    "delisting",
    "outflow",
    "fraud",
    "scam",
    "bearish",
    "crash",
    "collapse",
    "liquidation",
    "sec",
    "investigation",
    "sanction",
    "stolen",
    "vulnerability",
    "bankruptcy",
    "shutdown",
    "rug pull"
]


HIGH_IMPACT_WORDS = [
    "hack",
    "exploit",
    "sec",
    "etf",
    "approval",
    "approved",
    "ban",
    "banned",
    "lawsuit",
    "liquidation",
    "bankruptcy",
    "delisting",
    "regulation",
    "fed",
    "cpi",
    "interest rate"
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        result = float(value)

        if math.isnan(result):
            return default

        if math.isinf(result):
            return default

        return result

    except Exception:

        return default


def clamp(value, low, high):

    return max(
        low,
        min(high, value)
    )


def now_utc():

    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def base_symbol(symbol):

    if "/" in symbol:
        return symbol.split("/")[0].upper()

    return symbol.upper()


# ============================================================
# EXCHANGE CREATION
# ============================================================

def create_exchange(exchange_id):

    try:

        cls = getattr(
            ccxt,
            exchange_id
        )

        return cls({
            "enableRateLimit": True,
            "timeout": 20000
        })

    except Exception as e:

        print(
            f"{exchange_id} setup error: {e}"
        )

        return None


def create_futures_exchange(exchange_id):

    try:

        cls = getattr(
            ccxt,
            exchange_id
        )

        return cls({
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {
                "defaultType": "swap"
            }
        })

    except Exception as e:

        print(
            f"{exchange_id} futures setup error: {e}"
        )

        return None


# ============================================================
# SPOT EXCHANGES
# ============================================================

SPOT_EXCHANGES = {

    "BINANCE":
        create_exchange("binance"),

    "OKX":
        create_exchange("okx"),

    "BYBIT":
        create_exchange("bybit"),

    "KUCOIN":
        create_exchange("kucoin")
}


# ============================================================
# FUTURES EXCHANGES
# ============================================================

FUTURES_EXCHANGES = {

    "BINANCE":
        create_futures_exchange("binanceusdm"),

    "OKX":
        create_futures_exchange("okx"),

    "BYBIT":
        create_futures_exchange("bybit"),

    "KUCOIN":
        create_futures_exchange("kucoin")
}


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):

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

    result = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    return result.fillna(50)


def atr(df, period=14):

    high_low = (
        df["high"] -
        df["low"]
    )

    high_close = abs(
        df["high"] -
        df["close"].shift()
    )

    low_close = abs(
        df["low"] -
        df["close"].shift()
    )

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(
        period
    ).mean()


# ============================================================
# OHLCV
# ============================================================

def get_ohlcv(
    exchange,
    symbol,
    timeframe,
    limit=220
):

    if exchange is None:
        return None

    try:

        data = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not data:
            return None

        if len(data) < 80:
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

        return df

    except Exception:

        return None


# ============================================================
# MARKET DISCOVERY
# ============================================================

def discover_market():

    coins = {}

    for exchange_name, exchange in SPOT_EXCHANGES.items():

        if exchange is None:
            continue

        print(
            f"Discovering {exchange_name}..."
        )

        try:

            markets = exchange.load_markets()

        except Exception as e:

            print(
                f"{exchange_name} markets error: {e}"
            )

            continue

        for symbol, market in markets.items():

            try:

                if not market.get(
                    "spot",
                    False
                ):
                    continue

                if not symbol.endswith(
                    "/USDT"
                ):
                    continue

                base = base_symbol(
                    symbol
                )

                if base in STABLECOINS:
                    continue

                ticker = exchange.fetch_ticker(
                    symbol
                )

                volume = safe_float(
                    ticker.get(
                        "quoteVolume"
                    )
                )

                price = safe_float(
                    ticker.get(
                        "last"
                    )
                )

                if price <= 0:
                    continue

                if volume < 250000:
                    continue

                if symbol not in coins:

                    coins[symbol] = {
                        "volume": 0,
                        "exchanges": set()
                    }

                coins[symbol][
                    "volume"
                ] += volume

                coins[symbol][
                    "exchanges"
                ].add(
                    exchange_name
                )

            except Exception:

                continue

    ranked = sorted(
        coins.items(),
        key=lambda x:
            x[1]["volume"],
        reverse=True
    )

    result = []

    for symbol, info in ranked[
        :MAX_COINS
    ]:

        result.append({

            "symbol": symbol,

            "market_volume":
                info["volume"],

            "exchange_count":
                len(
                    info["exchanges"]
                ),

            "exchanges":
                ",".join(
                    sorted(
                        info["exchanges"]
                    )
                )
        })

    print(
        f"Discovered {len(result)} coins"
    )

    return result


# ============================================================
# BTC REGIME
# ============================================================

def btc_regime():

    preferred = [
        "BINANCE",
        "OKX",
        "BYBIT",
        "KUCOIN"
    ]

    for name in preferred:

        exchange = SPOT_EXCHANGES.get(
            name
        )

        if exchange is None:
            continue

        df = get_ohlcv(
            exchange,
            "BTC/USDT",
            "4h",
            220
        )

        if df is None:
            continue

        close = df["close"]

        e20 = ema(
            close,
            20
        ).iloc[-1]

        e50 = ema(
            close,
            50
        ).iloc[-1]

        e100 = ema(
            close,
            100
        ).iloc[-1]

        current = close.iloc[-1]

        current_rsi = safe_float(
            rsi(
                close
            ).iloc[-1],
            50
        )

        score = 0

        if current > e20:
            score += 1
        else:
            score -= 1

        if e20 > e50:
            score += 1
        else:
            score -= 1

        if e50 > e100:
            score += 1
        else:
            score -= 1

        if current_rsi > 55:
            score += 1

        elif current_rsi < 45:
            score -= 1

        if score >= 3:
            regime = "BULLISH"

        elif score <= -3:
            regime = "BEARISH"

        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "score": score
        }

    return {
        "regime": "UNKNOWN",
        "score": 0
    }


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def timeframe_analysis(df):

    if df is None:
        return {
            "trend": "UNKNOWN",
            "score": 0,
            "rsi": 50
        }

    close = df["close"]

    e20 = ema(
        close,
        20
    ).iloc[-1]

    e50 = ema(
        close,
        50
    ).iloc[-1]

    e100 = ema(
        close,
        100
    ).iloc[-1]

    price = close.iloc[-1]

    current_rsi = safe_float(
        rsi(
            close
        ).iloc[-1],
        50
    )

    score = 0

    if price > e20:
        score += 1
    else:
        score -= 1

    if e20 > e50:
        score += 1
    else:
        score -= 1

    if e50 > e100:
        score += 1
    else:
        score -= 1

    if current_rsi > 55:
        score += 1

    elif current_rsi < 45:
        score -= 1

    if score >= 3:
        trend = "BULLISH"

    elif score <= -3:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "score": score,
        "rsi": current_rsi
    }


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if df is None:
        return {
            "ratio": 1,
            "score": 0
        }

    current = safe_float(
        df["volume"].iloc[-1]
    )

    average = safe_float(
        df["volume"].iloc[-21:-1].mean(),
        1
    )

    if average <= 0:
        return {
            "ratio": 1,
            "score": 0
        }

    ratio = current / average

    if ratio >= 3:
        score = 4

    elif ratio >= 2:
        score = 3

    elif ratio >= 1.5:
        score = 2

    elif ratio >= 1.2:
        score = 1

    elif ratio < 0.7:
        score = -1

    else:
        score = 0

    return {
        "ratio": ratio,
        "score": score
    }


# ============================================================
# PRICE ACTION
# ============================================================

def price_action(df):

    recent = df.tail(5)

    green = (
        recent["close"] >
        recent["open"]
    ).sum()

    red = (
        recent["close"] <
        recent["open"]
    ).sum()

    if green >= 4:
        return {
            "label": "STRONG_BULLISH",
            "score": 3
        }

    if red >= 4:
        return {
            "label": "STRONG_BEARISH",
            "score": -3
        }

    if green > red:
        return {
            "label": "BULLISH",
            "score": 1
        }

    if red > green:
        return {
            "label": "BEARISH",
            "score": -1
        }

    return {
        "label": "NEUTRAL",
        "score": 0
    }


# ============================================================
# BREAKOUT
# ============================================================

def breakout_analysis(df):

    previous_high = df[
        "high"
    ].iloc[
        -21:-1
    ].max()

    previous_low = df[
        "low"
    ].iloc[
        -21:-1
    ].min()

    price = df[
        "close"
    ].iloc[-1]

    if price > previous_high:

        return {
            "label": "BULLISH_BREAKOUT",
            "score": 4
        }

    if price < previous_low:

        return {
            "label": "BEARISH_BREAKDOWN",
            "score": -4
        }

    return {
        "label": "NO_BREAKOUT",
        "score": 0
    }


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance(df):

    recent = df.tail(50)

    support = safe_float(
        recent["low"].min()
    )

    resistance = safe_float(
        recent["high"].max()
    )

    price = safe_float(
        df["close"].iloc[-1]
    )

    score = 0

    if price > 0:

        support_distance = (
            price - support
        ) / price

        resistance_distance = (
            resistance - price
        ) / price

        if support_distance < 0.01:
            score += 2

        if resistance_distance < 0.01:
            score -= 2

    return {
        "support": support,
        "resistance": resistance,
        "score": score
    }


# ============================================================
# ORDER BOOK
# ============================================================

def get_orderbook(exchange, symbol):

    try:

        book = exchange.fetch_order_book(
            symbol,
            limit=20
        )

        bids = book.get(
            "bids",
            []
        )

        asks = book.get(
            "asks",
            []
        )

        if not bids or not asks:
            return None

        bid_value = sum(
            safe_float(price) *
            safe_float(amount)

            for price, amount
            in bids
        )

        ask_value = sum(
            safe_float(price) *
            safe_float(amount)

            for price, amount
            in asks
        )

        total = (
            bid_value +
            ask_value
        )

        if total <= 0:
            return None

        imbalance = (
            bid_value -
            ask_value
        ) / total

        return {
            "imbalance": imbalance,
            "value": total
        }

    except Exception:

        return None


def multi_exchange_orderbook(symbol):

    values = []
    weights = []
    exchanges_used = []

    for name, exchange in SPOT_EXCHANGES.items():

        if exchange is None:
            continue

        result = get_orderbook(
            exchange,
            symbol
        )

        if result is None:
            continue

        values.append(
            result["imbalance"]
        )

        weights.append(
            result["value"]
        )

        exchanges_used.append(
            name
        )

    if not values:

        return {
            "imbalance": 0,
            "score": 0,
            "signal": "UNKNOWN",
            "exchanges": 0
        }

    imbalance = float(
        np.average(
            values,
            weights=weights
        )
    )

    imbalance = clamp(
        imbalance,
        -1,
        1
    )

    if imbalance >= 0.20:
        signal = "BULLISH"

    elif imbalance <= -0.20:
        signal = "BEARISH"

    else:
        signal = "NEUTRAL"

    score = int(
        clamp(
            imbalance * 8,
            -5,
            5
        )
    )

    return {
        "imbalance": imbalance,
        "score": score,
        "signal": signal,
        "exchanges": len(
            exchanges_used
        ),
        "names": ",".join(
            exchanges_used
        )
    }


# ============================================================
# FUTURES
# ============================================================

def futures_analysis(symbol):

    funding_values = []
    oi_values = []
    used = []

    for name, exchange in FUTURES_EXCHANGES.items():

        if exchange is None:
            continue

        # ----------------------------------------------------
        # Convert spot symbol:
        #
        # BTC/USDT
        #
        # into futures:
        #
        # BTC/USDT:USDT
        # ----------------------------------------------------

        futures_symbol = (
            symbol.replace(
                "/USDT",
                "/USDT:USDT"
            )
        )

        try:

            markets = exchange.load_markets()

            if futures_symbol not in markets:

                # Some exchanges use the plain symbol
                if symbol in markets:
                    futures_symbol = symbol

                else:
                    continue

            funding = exchange.fetch_funding_rate(
                futures_symbol
            )

            fr = safe_float(
                funding.get(
                    "fundingRate"
                )
            )

            funding_values.append(
                fr
            )

            try:

                oi = exchange.fetch_open_interest(
                    futures_symbol
                )

                oi_value = safe_float(
                    oi.get(
                        "openInterestValue"
                    )
                )

                if oi_value > 0:
                    oi_values.append(
                        oi_value
                    )

            except Exception:

                pass

            used.append(name)

        except Exception:

            continue

    if funding_values:

        average_funding = float(
            np.mean(
                funding_values
            )
        )

    else:

        average_funding = 0

    if average_funding > 0.0005:

        funding_signal = (
            "LONGS_CROWDED"
        )

        funding_score = -2

    elif average_funding < -0.0005:

        funding_signal = (
            "SHORTS_CROWDED"
        )

        funding_score = 2

    else:

        funding_signal = "NEUTRAL"
        funding_score = 0

    return {

        "funding":
            average_funding,

        "funding_signal":
            funding_signal,

        "funding_score":
            funding_score,

        "open_interest":
            sum(oi_values),

        "futures_exchanges":
            len(used),

        "futures_names":
            ",".join(used)
    }


# ============================================================
# CMC KEYLESS API
# ============================================================

CMC_BASE = (
    "https://pro-api.coinmarketcap.com"
    "/public-api"
)


def cmc_get(
    endpoint,
    params=None
):

    try:

        response = requests.get(
            CMC_BASE + endpoint,
            params=params or {},
            headers={
                "Accept":
                "application/json"
            },
            timeout=20
        )

        if response.status_code != 200:

            print(
                f"CMC HTTP error: "
                f"{response.status_code}"
            )

            return None

        payload = response.json()

        status = payload.get(
            "status",
            {}
        )

        error_code = status.get(
            "error_code",
            0
        )

        if error_code not in [
            0,
            None
        ]:

            print(
                "CMC API error:",
                status.get(
                    "error_message"
                )
            )

            return None

        return payload.get(
            "data"
        )

    except Exception as e:

        print(
            f"CMC request error: {e}"
        )

        return None


# ============================================================
# CMC ID MAP
# ============================================================

def cmc_id_map(symbols):

    mapping = {}

    unique = []

    for symbol in symbols:

        base = base_symbol(
            symbol
        )

        if base not in unique:
            unique.append(base)

    for start in range(
        0,
        len(unique),
        25
    ):

        batch = unique[
            start:start + 25
        ]

        data = cmc_get(
            "/v1/cryptocurrency/map",
            {
                "symbol":
                    ",".join(batch),

                "listing_status":
                    "active"
            }
        )

        if not isinstance(
            data,
            list
        ):
            continue

        for item in data:

            sym = str(
                item.get(
                    "symbol",
                    ""
                )
            ).upper()

            if sym not in mapping:
                mapping[sym] = []

            mapping[sym].append(
                item
            )

        time.sleep(
            0.3
        )

    return mapping


# ============================================================
# CMC FUNDAMENTALS
# ============================================================

def get_fundamentals(symbols):

    print(
        "Loading CMC ID map..."
    )

    mapping = cmc_id_map(
        symbols
    )

    ids = {}

    # Select the first active matching asset.
    # If multiple assets share a ticker, prefer
    # the one with the lowest CMC rank later.

    for symbol, items in mapping.items():

        if not items:
            continue

        selected = items[0]

        best_rank = 10**12

        for item in items:

            rank = safe_float(
                item.get(
                    "rank"
                ),
                10**12
            )

            if rank < best_rank:

                best_rank = rank
                selected = item

        coin_id = selected.get(
            "id"
        )

        if coin_id:

            ids[symbol] = int(
                coin_id
            )

    result = {}

    id_list = list(
        ids.values()
    )

    reverse_ids = {
        value: key
        for key, value
        in ids.items()
    }

    print(
        f"CMC IDs found: "
        f"{len(id_list)}"
    )

    for start in range(
        0,
        len(id_list),
        25
    ):

        batch = id_list[
            start:start + 25
        ]

        data = cmc_get(
            "/v3/cryptocurrency/quotes/latest",
            {
                "id":
                    ",".join(
                        str(x)
                        for x in batch
                    ),

                "convert":
                    "USD"
            }
        )

        if not isinstance(
            data,
            dict
        ):
            continue

        for key, item in data.items():

            try:
                coin_id = int(key)
            except Exception:
                continue

            sym = reverse_ids.get(
                coin_id
            )

            if not sym:
                continue

            quote = (
                item.get(
                    "quote",
                    {}
                )
                .get(
                    "USD",
                    {}
                )
            )

            result[sym] = {

                "id":
                    coin_id,

                "name":
                    item.get(
                        "name",
                        ""
                    ),

                "symbol":
                    item.get(
                        "symbol",
                        sym
                    ),

                "rank":
                    safe_float(
                        item.get(
                            "cmc_rank"
                        )
                    ),

                "market_cap":
                    safe_float(
                        quote.get(
                            "market_cap"
                        )
                    ),

                "volume_24h":
                    safe_float(
                        quote.get(
                            "volume_24h"
                        )
                    ),

                "change_24h":
                    safe_float(
                        quote.get(
                            "percent_change_24h"
                        )
                    ),

                "change_7d":
                    safe_float(
                        quote.get(
                            "percent_change_7d"
                        )
                    ),

                "change_30d":
                    safe_float(
                        quote.get(
                            "percent_change_30d"
                        )
                    ),

                "circulating_supply":
                    safe_float(
                        item.get(
                            "circulating_supply"
                        )
                    ),

                "total_supply":
                    safe_float(
                        item.get(
                            "total_supply"
                        )
                    ),

                "max_supply":
                    safe_float(
                        item.get(
                            "max_supply"
                        )
                    ),

                "fdv":
                    safe_float(
                        quote.get(
                            "fully_diluted_market_cap"
                        )
                    ),

                "market_pairs":
                    safe_float(
                        item.get(
                            "num_market_pairs"
                        )
                    ),

                "date_added":
                    item.get(
                        "date_added"
                    )
            }

        time.sleep(
            0.3
        )

    print(
        f"CMC fundamentals received: "
        f"{len(result)}"
    )

    return result


# ============================================================
# FUNDAMENTAL SCORE
# ============================================================

def fundamental_analysis(data):

    if not data:

        return {

            "score": 50,

            "rating":
                "DATA_UNAVAILABLE",

            "market_cap": 0,
            "rank": 0,
            "circulating_supply": 0,
            "total_supply": 0,
            "max_supply": 0,
            "fdv": 0,
            "mc_fdv_ratio": 0,
            "supply_ratio": 0,
            "market_pairs": 0,
            "age_days": 0,

            "reason":
                "CMC_DATA_UNAVAILABLE"
        }

    score = 50
    reasons = []

    rank = safe_float(
        data.get(
            "rank"
        )
    )

    market_cap = safe_float(
        data.get(
            "market_cap"
        )
    )

    circulating = safe_float(
        data.get(
            "circulating_supply"
        )
    )

    total_supply = safe_float(
        data.get(
            "total_supply"
        )
    )

    max_supply = safe_float(
        data.get(
            "max_supply"
        )
    )

    fdv = safe_float(
        data.get(
            "fdv"
        )
    )

    pairs = safe_float(
        data.get(
            "market_pairs"
        )
    )

    # --------------------------------------------------------
    # MARKET CAP RANK
    # --------------------------------------------------------

    if rank > 0:

        if rank <= 50:

            score += 5
            reasons.append(
                "TOP50"
            )

        elif rank <= 100:

            score += 3
            reasons.append(
                "TOP100"
            )

        elif rank <= 250:

            score += 1

        elif rank > 1000:

            score -= 3
            reasons.append(
                "LOW_RANK"
            )

    # --------------------------------------------------------
    # FDV DILUTION
    # --------------------------------------------------------

    mc_fdv_ratio = 0

    if (
        market_cap > 0
        and fdv > 0
    ):

        mc_fdv_ratio = (
            market_cap /
            fdv
        )

        if mc_fdv_ratio >= 0.80:

            score += 4
            reasons.append(
                "LOW_DILUTION"
            )

        elif mc_fdv_ratio >= 0.50:

            score += 2

        elif mc_fdv_ratio < 0.25:

            score -= 4
            reasons.append(
                "HIGH_DILUTION"
            )

    # --------------------------------------------------------
    # SUPPLY
    # --------------------------------------------------------

    supply_ratio = 0

    if (
        max_supply > 0
        and circulating > 0
    ):

        supply_ratio = (
            circulating /
            max_supply
        )

        if supply_ratio >= 0.80:

            score += 3
            reasons.append(
                "HIGH_CIRCULATION"
            )

        elif supply_ratio >= 0.50:

            score += 1

        elif supply_ratio < 0.25:

            score -= 3
            reasons.append(
                "LOW_CIRCULATION"
            )

    # --------------------------------------------------------
    # MARKET PAIRS
    # --------------------------------------------------------

    if pairs >= 200:

        score += 3
        reasons.append(
            "WIDE_MARKET_ACCESS"
        )

    elif pairs >= 100:

        score += 2

    elif pairs >= 50:

        score += 1

    elif pairs > 0 and pairs < 10:

        score -= 2

    # --------------------------------------------------------
    # UNKNOWN MAX SUPPLY
    # --------------------------------------------------------

    if max_supply <= 0:

        score -= 1

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    age_days = 0

    date_added = data.get(
        "date_added"
    )

    if date_added:

        try:

            dt = datetime.fromisoformat(
                date_added.replace(
                    "Z",
                    "+00:00"
                )
            )

            age_days = (
                datetime.now(
                    timezone.utc
                ) - dt
            ).days

        except Exception:

            age_days = 0

    score = int(
        clamp(
            score,
            0,
            100
        )
    )

    if score >= 70:

        rating = "STRONG"

    elif score >= 50:

        rating = "NEUTRAL"

    else:

        rating = "WEAK"

    return {

        "score": score,

        "rating": rating,

        "market_cap":
            market_cap,

        "rank":
            rank,

        "circulating_supply":
            circulating,

        "total_supply":
            total_supply,

        "max_supply":
            max_supply,

        "fdv":
            fdv,

        "mc_fdv_ratio":
            mc_fdv_ratio,

        "supply_ratio":
            supply_ratio,

        "market_pairs":
            pairs,

        "age_days":
            age_days,

        "reason":
            ",".join(
                reasons
            )
            if reasons
            else "NEUTRAL"
    }


# ============================================================
# NEWS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return " ".join(
        str(text).split()
    )


def load_rss(url):

    articles = []

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent":
                "Mozilla/5.0 CryptoMasterAI"
            },
            timeout=15
        )

        if response.status_code != 200:
            return []

        root = ET.fromstring(
            response.content
        )

        for item in root.iter():

            if not item.tag.lower().endswith(
                "item"
            ):
                continue

            title = ""
            link = ""
            date = ""

            for child in list(item):

                tag = child.tag.lower()

                if tag.endswith(
                    "title"
                ):

                    title = clean_text(
                        child.text
                    )

                elif tag.endswith(
                    "link"
                ):

                    link = clean_text(
                        child.text
                    )

                elif tag.endswith(
                    "pubdate"
                ):

                    date = clean_text(
                        child.text
                    )

            if title:

                articles.append({

                    "title":
                        title,

                    "link":
                        link,

                    "date":
                        date,

                    "source":
                        url
                })

    except Exception:

        pass

    return articles


def load_all_news():

    all_articles = []

    for feed in NEWS_FEEDS:

        articles = load_rss(
            feed
        )

        all_articles.extend(
            articles[:50]
        )

        time.sleep(
            0.2
        )

    unique = {}

    for article in all_articles:

        key = article[
            "title"
        ].lower().strip()

        unique[key] = article

    return list(
        unique.values()
    )


# ============================================================
# NEWS MATCHING
# ============================================================

def keyword_in_title(
    title,
    keyword
):

    keyword = keyword.strip().lower()

    if len(keyword) <= 3:

        pattern = (
            r"\b" +
            re.escape(keyword) +
            r"\b"
        )

        return re.search(
            pattern,
            title
        ) is not None

    return keyword in title


def news_analysis(
    symbol,
    fundamentals,
    articles
):

    base = base_symbol(
        symbol
    )

    coin_name = ""

    if fundamentals:

        coin_name = str(
            fundamentals.get(
                "name",
                ""
            )
        ).lower()

    keywords = [
        base.lower()
    ]

    if coin_name:

        keywords.append(
            coin_name
        )

    matched = []

    for article in articles:

        title = article[
            "title"
        ].lower()

        found = False

        for keyword in keywords:

            if keyword_in_title(
                title,
                keyword
            ):

                found = True
                break

        if found:

            matched.append(
                article
            )

    matched = matched[:10]

    if not matched:

        return {

            "score": 0,

            "sentiment":
                "NO_NEWS",

            "impact":
                "NONE",

            "count": 0,

            "headlines": "",

            "reason":
                "NO_MATCHING_NEWS"
        }

    total_score = 0
    high_impact = 0
    headlines = []

    for article in matched:

        title = article[
            "title"
        ].lower()

        positive = 0
        negative = 0

        for word in POSITIVE_WORDS:

            if word in title:
                positive += 1

        for word in NEGATIVE_WORDS:

            if word in title:
                negative += 1

        article_score = (
            positive -
            negative
        )

        total_score += (
            article_score
        )

        if any(
            word in title
            for word in HIGH_IMPACT_WORDS
        ):

            high_impact += 1

        headlines.append(
            article["title"]
        )

    score = clamp(
        total_score * 12,
        -100,
        100
    )

    if score >= 25:

        sentiment = "BULLISH"

    elif score <= -25:

        sentiment = "BEARISH"

    else:

        sentiment = "NEUTRAL"

    if high_impact >= 2:

        impact = "CRITICAL"

    elif high_impact == 1:

        impact = "HIGH"

    elif len(matched) >= 3:

        impact = "MEDIUM"

    else:

        impact = "LOW"

    return {

        "score":
            score,

        "sentiment":
            sentiment,

        "impact":
            impact,

        "count":
            len(matched),

        "headlines":
            " || ".join(
                headlines[:5]
            ),

        "reason":
            sentiment +
            "_" +
            impact
    }


# ============================================================
# GLOBAL MARKET
# ============================================================

def global_market():

    data = cmc_get(
        "/v1/global-metrics/quotes/latest",
        {
            "convert":
                "USD"
        }
    )

    if not data:

        return {}

    quote = (
        data.get(
            "quote",
            {}
        )
        .get(
            "USD",
            {}
        )
    )

    return {

        "market_cap":
            safe_float(
                quote.get(
                    "total_market_cap"
                )
            ),

        "volume_24h":
            safe_float(
                quote.get(
                    "total_volume_24h"
                )
            ),

        "btc_dominance":
            safe_float(
                data.get(
                    "btc_dominance"
                )
            ),

        "eth_dominance":
            safe_float(
                data.get(
                    "eth_dominance"
                )
            )
    }


# ============================================================
# FEAR & GREED
# ============================================================

def fear_greed():

    data = cmc_get(
        "/v3/fear-and-greed/latest"
    )

    if not data:

        return {

            "value": 0,

            "classification":
                "UNKNOWN"
        }

    return {

        "value":
            safe_float(
                data.get(
                    "value"
                )
            ),

        "classification":
            data.get(
                "value_classification",
                "UNKNOWN"
            )
    }


# ============================================================
# COIN ANALYSIS
# ============================================================

def analyze_coin(
    item,
    fundamental_data,
    articles,
    btc,
    global_data,
    fg
):

    symbol = item[
        "symbol"
    ]

    # --------------------------------------------------------
    # Use Binance first, then other exchanges
    # --------------------------------------------------------

    exchange_order = [
        "BINANCE",
        "OKX",
        "BYBIT",
        "KUCOIN"
    ]

    exchange = None
    exchange_name = ""

    for name in exchange_order:

        candidate = (
            SPOT_EXCHANGES.get(
                name
            )
        )

        if candidate is None:
            continue

        try:

            markets = candidate.load_markets()

            if symbol in markets:

                exchange = candidate
                exchange_name = name
                break

        except Exception:

            continue

    if exchange is None:
        return None

    # --------------------------------------------------------
    # Timeframes
    # --------------------------------------------------------

    tf = {}

    for timeframe in TIMEFRAMES:

        df = get_ohlcv(
            exchange,
            symbol,
            timeframe,
            220
        )

        tf[timeframe] = (
            timeframe_analysis(
                df
            )
        )

    valid = [
        value["score"]
        for value in tf.values()
        if value["trend"] != "UNKNOWN"
    ]

    if not valid:
        return None

    weights = {

        "15m":
            0.15,

        "1h":
            0.30,

        "4h":
            0.35,

        "1d":
            0.20
    }

    tf_score = 0

    for timeframe, weight in weights.items():

        tf_score += (
            tf[timeframe]["score"]
            * weight
        )

    # --------------------------------------------------------
    # 1H
    # --------------------------------------------------------

    df = get_ohlcv(
        exchange,
        symbol,
        "1h",
        220
    )

    if df is None:
        return None

    price = safe_float(
        df["close"].iloc[-1]
    )

    volume = volume_analysis(
        df
    )

    action = price_action(
        df
    )

    breakout = breakout_analysis(
        df
    )

    sr = support_resistance(
        df
    )

    # --------------------------------------------------------
    # Order Book
    # --------------------------------------------------------

    orderbook = (
        multi_exchange_orderbook(
            symbol
        )
    )

    # --------------------------------------------------------
    # Futures
    # --------------------------------------------------------

    futures = futures_analysis(
        symbol
    )

    # --------------------------------------------------------
    # Fundamental
    # --------------------------------------------------------

    fundamental = (
        fundamental_analysis(
            fundamental_data
        )
    )

    # --------------------------------------------------------
    # News
    # --------------------------------------------------------

    news = news_analysis(
        symbol,
        fundamental_data,
        articles
    )

    # --------------------------------------------------------
    # Technical
    # --------------------------------------------------------

    technical_raw = (

        tf_score

        + volume["score"] * 0.8

        + action["score"] * 0.8

        + breakout["score"] * 0.9

        + sr["score"] * 0.5
    )

    technical_score = clamp(
        50 +
        technical_raw * 4,
        0,
        100
    )

    # --------------------------------------------------------
    # Derivatives
    # --------------------------------------------------------

    derivatives_score = clamp(
        50 +
        futures["funding_score"] * 8,
        0,
        100
    )

    # --------------------------------------------------------
    # Order Flow
    # --------------------------------------------------------

    orderflow_score = clamp(
        50 +
        orderbook["score"] * 6,
        0,
        100
    )

    # --------------------------------------------------------
    # BTC
    # --------------------------------------------------------

    btc_bias = 0

    if btc["regime"] == "BULLISH":

        btc_bias = 5

    elif btc["regime"] == "BEARISH":

        btc_bias = -5

    # --------------------------------------------------------
    # Fundamental Bias
    # --------------------------------------------------------

    fundamental_bias = clamp(
        (
            fundamental["score"]
            - 50
        ) * 0.20,
        -10,
        10
    )

    # --------------------------------------------------------
    # News Bias
    # --------------------------------------------------------

    news_bias = (
        news["score"] /
        10
    )

    if news["impact"] == "CRITICAL":

        news_bias *= 1.5

    news_bias = clamp(
        news_bias,
        -15,
        15
    )

    # --------------------------------------------------------
    # FINAL BIAS
    # --------------------------------------------------------

    final_bias = (

        (technical_score - 50)
        * 0.55

        + (derivatives_score - 50)
        * 0.10

        + (orderflow_score - 50)
        * 0.10

        + fundamental_bias

        + news_bias

        + btc_bias
    )

    long_score = clamp(
        50 + final_bias,
        0,
        100
    )

    short_score = clamp(
        50 - final_bias,
        0,
        100
    )

    # --------------------------------------------------------
    # TIMEFRAME CONFIRMATION
    # --------------------------------------------------------

    bullish_tf = sum(
        1
        for x in tf.values()
        if x["trend"] == "BULLISH"
    )

    bearish_tf = sum(
        1
        for x in tf.values()
        if x["trend"] == "BEARISH"
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal = "NO TRADE"

    if (
        long_score >= 65
        and bullish_tf >= 2
        and long_score >
        short_score + 8
    ):

        signal = "LONG"

    elif (
        short_score >= 65
        and bearish_tf >= 2
        and short_score >
        long_score + 8
    ):

        signal = "SHORT"

    # --------------------------------------------------------
    # Critical news protection
    # --------------------------------------------------------

    if news["impact"] == "CRITICAL":

        if (
            signal == "LONG"
            and long_score < 75
        ):

            signal = "NO TRADE"

        if (
            signal == "SHORT"
            and short_score < 75
        ):

            signal = "NO TRADE"

    # --------------------------------------------------------
    # Strength
    # --------------------------------------------------------

    if signal == "LONG":

        strength = long_score

    elif signal == "SHORT":

        strength = short_score

    else:

        strength = max(
            long_score,
            short_score
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confirmation = 0

    confirmation += min(
        abs(tf_score) * 5,
        20
    )

    confirmation += min(
        abs(volume["score"]) * 3,
        10
    )

    confirmation += min(
        abs(orderbook["score"]) * 2,
        10
    )

    confirmation += min(
        abs(
            futures["funding_score"]
        ) * 2,
        8
    )

    confirmation += min(
        abs(news["score"]) * 0.10,
        10
    )

    confidence = clamp(
        50 + confirmation,
        0,
        100
    )

    # --------------------------------------------------------
    # SL / TP
    # --------------------------------------------------------

    atr_value = safe_float(
        atr(df).iloc[-1]
    )

    if atr_value <= 0:

        atr_value = (
            price * 0.01
        )

    if signal == "LONG":

        stop_loss = (
            price -
            atr_value * 1.5
        )

        take_profit = (
            price +
            atr_value * 3
        )

    elif signal == "SHORT":

        stop_loss = (
            price +
            atr_value * 1.5
        )

        take_profit = (
            price -
            atr_value * 3
        )

    else:

        stop_loss = 0
        take_profit = 0

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "symbol":
            symbol,

        "signal":
            signal,

        "signal_strength":
            round(
                strength,
                2
            ),

        "confidence":
            round(
                confidence,
                2
            ),

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

        "price":
            price,

        "analysis_exchange":
            exchange_name,

        "btc_regime":
            btc["regime"],

        "btc_regime_score":
            btc["score"],

        "15m_trend":
            tf["15m"]["trend"],

        "1h_trend":
            tf["1h"]["trend"],

        "4h_trend":
            tf["4h"]["trend"],

        "1d_trend":
            tf["1d"]["trend"],

        "technical_score":
            round(
                technical_score,
                2
            ),

        "fundamental_score":
            fundamental["score"],

        "fundamental_rating":
            fundamental["rating"],

        "market_cap":
            fundamental["market_cap"],

        "cmc_rank":
            fundamental["rank"],

        "circulating_supply":
            fundamental[
                "circulating_supply"
            ],

        "total_supply":
            fundamental[
                "total_supply"
            ],

        "max_supply":
            fundamental[
                "max_supply"
            ],

        "fdv":
            fundamental["fdv"],

        "mc_fdv_ratio":
            round(
                fundamental[
                    "mc_fdv_ratio"
                ],
                4
            ),

        "supply_ratio":
            round(
                fundamental[
                    "supply_ratio"
                ],
                4
            ),

        "market_pairs":
            fundamental[
                "market_pairs"
            ],

        "asset_age_days":
            fundamental[
                "age_days"
            ],

        "fundamental_reason":
            fundamental[
                "reason"
            ],

        "volume_ratio":
            round(
                volume["ratio"],
                3
            ),

        "price_action":
            action["label"],

        "breakout":
            breakout["label"],

        "support":
            sr["support"],

        "resistance":
            sr["resistance"],

        "funding":
            futures["funding"],

        "funding_signal":
            futures[
                "funding_signal"
            ],

        "open_interest":
            futures[
                "open_interest"
            ],

        "futures_exchanges":
            futures[
                "futures_exchanges"
            ],

        "futures_names":
            futures[
                "futures_names"
            ],

        "orderbook_imbalance":
            round(
                orderbook[
                    "imbalance"
                ],
                4
            ),

        "orderbook_signal":
            orderbook[
                "signal"
            ],

        "orderbook_exchanges":
            orderbook[
                "exchanges"
            ],

        "orderbook_names":
            orderbook.get(
                "names",
                ""
            ),

        "news_score":
            round(
                news["score"],
                2
            ),

        "news_sentiment":
            news["sentiment"],

        "news_impact":
            news["impact"],

        "news_count":
            news["count"],

        "news_headlines":
            news["headlines"],

        "global_market_cap":
            global_data.get(
                "market_cap",
                0
            ),

        "btc_dominance":
            global_data.get(
                "btc_dominance",
                0
            ),

        "fear_greed":
            fg.get(
                "value",
                0
            ),

        "fear_greed_class":
            fg.get(
                "classification",
                "UNKNOWN"
            ),

        "market_volume":
            item["market_volume"],

        "exchange_count":
            item["exchange_count"],

        "exchanges":
            item["exchanges"],

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "reason":
            (
                f"TF:{tf_score:.1f} | "
                f"TECH:{technical_score:.1f} | "
                f"FUND:{fundamental['score']} | "
                f"NEWS:{news['sentiment']} | "
                f"NEWS_IMPACT:{news['impact']} | "
                f"OB:{orderbook['signal']} | "
                f"FUNDING:{futures['funding_signal']} | "
                f"BTC:{btc['regime']}"
            ),

        "timestamp":
            now_utc()
    }


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not token or not chat_id:
        return

    try:

        requests.post(

            f"https://api.telegram.org/"
            f"bot{token}/sendMessage",

            data={

                "chat_id":
                    chat_id,

                "text":
                    message
            },

            timeout=15
        )

    except Exception:

        pass


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CRYPTO MASTER AI V9.1"
    )

    print(
        "BINANCE + OKX + BYBIT + KUCOIN"
    )

    print(
        "TECHNICAL + FUNDAMENTAL + "
        "FUTURES + ORDER BOOK + NEWS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    coins = discover_market()

    if not coins:

        print(
            "No coins discovered."
        )

        return

    symbols = [
        item["symbol"]
        for item in coins
    ]

    # --------------------------------------------------------
    # FUNDAMENTALS
    # --------------------------------------------------------

    fundamentals = get_fundamentals(
        symbols
    )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    print(
        "Loading crypto news..."
    )

    articles = load_all_news()

    print(
        f"News loaded: "
        f"{len(articles)}"
    )

    # --------------------------------------------------------
    # BTC
    # --------------------------------------------------------

    btc = btc_regime()

    print(
        "BTC regime:",
        btc["regime"]
    )

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    global_data = (
        global_market()
    )

    fg = fear_greed()

    print(
        "Fear & Greed:",
        fg["value"],
        fg["classification"]
    )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    results = []

    for index, item in enumerate(
        coins,
        start=1
    ):

        symbol = item[
            "symbol"
        ]

        print(
            f"[{index}/{len(coins)}] "
            f"{symbol}"
        )

        try:

            result = analyze_coin(

                item,

                fundamentals.get(
                    base_symbol(
                        symbol
                    )
                ),

                articles,

                btc,

                global_data,

                fg
            )

            if result:

                results.append(
                    result
                )

        except Exception as e:

            print(
                f"{symbol} analysis error: "
                f"{e}"
            )

    if not results:

        print(
            "No results."
        )

        return

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    order = {

        "LONG": 0,

        "SHORT": 1,

        "NO TRADE": 2
    }

    results.sort(

        key=lambda x: (

            order.get(
                x["signal"],
                9
            ),

            -x[
                "signal_strength"
            ]
        )
    )

    df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        "crypto_scan_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "TOP RESULTS"
    )
    print("=" * 70)

    columns = [

        "symbol",

        "signal",

        "signal_strength",

        "confidence",

        "technical_score",

        "fundamental_score",

        "fundamental_rating",

        "news_sentiment",

        "news_impact",

        "orderbook_signal",

        "futures_exchanges",

        "btc_regime"
    ]

    available = [

        column

        for column in columns

        if column in df.columns
    ]

    print(
        df[
            available
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    long_count = (
        df["signal"] ==
        "LONG"
    ).sum()

    short_count = (
        df["signal"] ==
        "SHORT"
    ).sum()

    no_trade_count = (
        df["signal"] ==
        "NO TRADE"
    ).sum()

    print("\n")
    print("=" * 70)
    print(
        "SUMMARY"
    )
    print("=" * 70)

    print(
        f"LONG: {long_count}"
    )

    print(
        f"SHORT: {short_count}"
    )

    print(
        f"NO TRADE: {no_trade_count}"
    )

    print(
        f"TOTAL: {len(df)}"
    )

    # --------------------------------------------------------
    # DATA HEALTH
    # --------------------------------------------------------

    fundamental_ok = (
        df[
            "fundamental_rating"
        ] !=
        "DATA_UNAVAILABLE"
    ).sum()

    futures_ok = (
        df[
            "futures_exchanges"
        ] > 0
    ).sum()

    orderbook_ok = (
        df[
            "orderbook_exchanges"
        ] > 0
    ).sum()

    news_found = (
        df[
            "news_count"
        ] > 0
    ).sum()

    print("\n")
    print("=" * 70)
    print(
        "DATA HEALTH"
    )
    print("=" * 70)

    print(
        f"Fundamental data: "
        f"{fundamental_ok}/{len(df)}"
    )

    print(
        f"Futures data: "
        f"{futures_ok}/{len(df)}"
    )

    print(
        f"Order book data: "
        f"{orderbook_ok}/{len(df)}"
    )

    print(
        f"Coins with news: "
        f"{news_found}/{len(df)}"
    )

    print("\n")
    print(
        "IMPORTANT: confidence is a model "
        "setup score, NOT win probability."
    )

    print(
        "Saved: crypto_scan_results.csv"
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    strong = df[
        (
            df["signal"].isin(
                [
                    "LONG",
                    "SHORT"
                ]
            )
        )
        &
        (
            df[
                "signal_strength"
            ] >= 75
        )
    ].head(10)

    if len(strong) > 0:

        message = (
            "🚀 CRYPTO MASTER AI V9.1\n\n"
        )

        for _, row in strong.iterrows():

            message += (

                f"{row['symbol']} "
                f"{row['signal']}\n"

                f"Strength: "
                f"{row['signal_strength']}\n"

                f"Confidence: "
                f"{row['confidence']}\n"

                f"Fundamental: "
                f"{row['fundamental_score']} "
                f"{row['fundamental_rating']}\n"

                f"News: "
                f"{row['news_sentiment']} "
                f"{row['news_impact']}\n"

                f"OrderBook: "
                f"{row['orderbook_signal']}\n"

                f"Futures: "
                f"{row['futures_exchanges']} "
                f"exchanges\n\n"
            )

        message += (
            "⚠️ Model score is not "
            "guaranteed profit probability."
        )

        send_telegram(
            message
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
