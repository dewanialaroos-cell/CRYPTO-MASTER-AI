import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# CRYPTO MASTER AI V7
# ADVANCED SIGNAL + CONFIDENCE ENGINE
# ============================================================

MAX_COINS = 60
MIN_VOLUME_USDT = 250000

LONG_THRESHOLD = 68
SHORT_THRESHOLD = 68

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

EXCHANGE_NAMES = ["okx", "bybit", "kucoin"]


# ============================================================
# EXCHANGE
# ============================================================

def create_exchange(name, futures=False):
    try:
        config = {
            "enableRateLimit": True,
            "timeout": 20000
        }

        if futures:
            config["options"] = {
                "defaultType": "swap"
            }

        exchange_class = getattr(ccxt, name)

        return exchange_class(config)

    except Exception:
        return None


spot_exchanges = {}
futures_exchanges = {}

for name in EXCHANGE_NAMES:

    spot = create_exchange(name, False)

    if spot:
        spot_exchanges[name] = spot

    futures = create_exchange(name, True)

    if futures:
        futures_exchanges[name] = futures


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

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):

    hl = df["high"] - df["low"]

    hc = abs(
        df["high"] -
        df["close"].shift()
    )

    lc = abs(
        df["low"] -
        df["close"].shift()
    )

    tr = pd.concat(
        [hl, hc, lc],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


# ============================================================
# OHLCV
# ============================================================

def get_ohlcv(exchange, symbol, timeframe, limit=180):

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

        return df

    except Exception:
        return None


# ============================================================
# DISCOVER COINS
# ============================================================

def discover_coins():

    candidates = {}

    for name, exchange in spot_exchanges.items():

        try:

            markets = exchange.load_markets()
            tickers = exchange.fetch_tickers()

            for symbol, market in markets.items():

                try:

                    if not market.get("spot"):
                        continue

                    if not market.get(
                        "active",
                        True
                    ):
                        continue

                    if not symbol.endswith("/USDT"):
                        continue

                    ticker = tickers.get(symbol)

                    if not ticker:
                        continue

                    volume = ticker.get(
                        "quoteVolume"
                    )

                    if volume is None:
                        continue

                    volume = float(volume)

                    if volume < MIN_VOLUME_USDT:
                        continue

                    if symbol not in candidates:

                        candidates[symbol] = {
                            "symbol": symbol,
                            "volume": 0,
                            "exchanges": set()
                        }

                    candidates[symbol][
                        "volume"
                    ] += volume

                    candidates[symbol][
                        "exchanges"
                    ].add(name)

                except Exception:
                    continue

        except Exception:
            continue

    results = []

    for symbol, item in candidates.items():

        item["exchange_count"] = len(
            item["exchanges"]
        )

        results.append(item)

    results.sort(
        key=lambda x: x["volume"],
        reverse=True
    )

    return results[:MAX_COINS]


# ============================================================
# BTC REGIME
# ============================================================

def btc_regime():

    exchange = spot_exchanges.get("okx")

    if not exchange:
        return "UNKNOWN", 0

    df = get_ohlcv(
        exchange,
        "BTC/USDT",
        "4h",
        200
    )

    if df is None:
        return "UNKNOWN", 0

    df["ema20"] = ema(
        df["close"],
        20
    )

    df["ema50"] = ema(
        df["close"],
        50
    )

    df["ema100"] = ema(
        df["close"],
        100
    )

    df["rsi"] = rsi(
        df["close"]
    )

    last = df.iloc[-1]

    score = 0

    if last["close"] > last["ema20"]:
        score += 1
    else:
        score -= 1

    if last["ema20"] > last["ema50"]:
        score += 1
    else:
        score -= 1

    if last["ema50"] > last["ema100"]:
        score += 1
    else:
        score -= 1

    if last["rsi"] > 55:
        score += 1

    elif last["rsi"] < 45:
        score -= 1

    if score >= 3:
        return "BULL", score

    if score <= -3:
        return "BEAR", score

    return "NEUTRAL", score


# ============================================================
# TIMEFRAME ENGINE
# ============================================================

def timeframe_analysis(df):

    if df is None or len(df) < 60:

        return {
            "trend": "UNKNOWN",
            "score": 0,
            "momentum": 0,
            "rsi": 50,
            "atr": 0
        }

    df = df.copy()

    df["ema20"] = ema(
        df["close"],
        20
    )

    df["ema50"] = ema(
        df["close"],
        50
    )

    df["rsi"] = rsi(
        df["close"]
    )

    df["atr"] = atr(df)

    last = df.iloc[-1]

    score = 0

    if last["close"] > last["ema20"]:
        score += 1
    else:
        score -= 1

    if last["ema20"] > last["ema50"]:
        score += 1
    else:
        score -= 1

    rsi_value = float(
        last["rsi"]
    )

    if rsi_value > 55:
        score += 1

    elif rsi_value < 45:
        score -= 1

    momentum = (
        (
            last["close"] /
            df["close"].iloc[-10]
        ) - 1
    ) * 100

    if momentum > 0.5:
        score += 1

    elif momentum < -0.5:
        score -= 1

    if score >= 2:
        trend = "BULL"

    elif score <= -2:
        trend = "BEAR"

    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "score": score,
        "momentum": momentum,
        "rsi": rsi_value,
        "atr": float(last["atr"])
    }


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if df is None or len(df) < 30:
        return 1, 0, "UNKNOWN"

    current_volume = float(
        df["volume"].iloc[-1]
    )

    average_volume = float(
        df["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    if average_volume <= 0:
        return 1, 0, "UNKNOWN"

    ratio = (
        current_volume /
        average_volume
    )

    price_change = (
        (
            df["close"].iloc[-1] /
            df["close"].iloc[-2]
        ) - 1
    ) * 100

    score = 0

    if ratio >= 2:

        if price_change > 0:
            score = 8
            direction = "BUY_VOLUME"

        elif price_change < 0:
            score = -8
            direction = "SELL_VOLUME"

        else:
            direction = "NEUTRAL"

    elif ratio >= 1.5:

        if price_change > 0:
            score = 4
            direction = "BUY_VOLUME"

        elif price_change < 0:
            score = -4
            direction = "SELL_VOLUME"

        else:
            direction = "NEUTRAL"

    else:
        direction = "LOW_VOLUME"

    return ratio, score, direction


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance(df):

    if df is None or len(df) < 60:
        return 0, 0, 0.5

    recent = df.tail(50)

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    price = float(
        df["close"].iloc[-1]
    )

    if resistance <= support:
        return support, resistance, 0.5

    position = (
        price - support
    ) / (
        resistance - support
    )

    return support, resistance, position


# ============================================================
# BREAKOUT
# ============================================================

def breakout_analysis(df):

    if df is None or len(df) < 30:
        return "NONE", 0

    previous_high = float(
        df["high"]
        .iloc[-11:-1]
        .max()
    )

    previous_low = float(
        df["low"]
        .iloc[-11:-1]
        .min()
    )

    last_close = float(
        df["close"].iloc[-1]
    )

    last_high = float(
        df["high"].iloc[-1]
    )

    last_low = float(
        df["low"].iloc[-1]
    )

    average_volume = float(
        df["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    current_volume = float(
        df["volume"].iloc[-1]
    )

    # Strong bullish breakout
    if last_close > previous_high:

        if current_volume >= average_volume * 1.3:
            return "BULL_BREAKOUT", 7

        return "WEAK_BULL_BREAKOUT", 2

    # Strong bearish breakout
    if last_close < previous_low:

        if current_volume >= average_volume * 1.3:
            return "BEAR_BREAKOUT", -7

        return "WEAK_BEAR_BREAKOUT", -2

    # Fake bullish breakout
    if (
        last_high > previous_high
        and last_close < previous_high
    ):
        return "FAKE_BULL_BREAKOUT", -6

    # Fake bearish breakout
    if (
        last_low < previous_low
        and last_close > previous_low
    ):
        return "FAKE_BEAR_BREAKOUT", 6

    return "NONE", 0


# ============================================================
# PRICE ACTION
# ============================================================

def price_action(df):

    if df is None or len(df) < 20:
        return "UNKNOWN", 0

    highs = df["high"].tail(10).values
    lows = df["low"].tail(10).values

    higher_high = (
        highs[-1] > highs[-3]
    )

    higher_low = (
        lows[-1] > lows[-3]
    )

    lower_high = (
        highs[-1] < highs[-3]
    )

    lower_low = (
        lows[-1] < lows[-3]
    )

    if higher_high and higher_low:
        return "HH_HL", 7

    if lower_high and lower_low:
        return "LH_LL", -7

    return "MIXED", 0


# ============================================================
# ORDER BOOK
# ============================================================

def orderbook_analysis(
    exchange,
    symbol
):

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
            return 0, 0, "UNKNOWN"

        bid_value = sum(
            float(price) *
            float(amount)
            for price, amount in bids
        )

        ask_value = sum(
            float(price) *
            float(amount)
            for price, amount in asks
        )

        total = (
            bid_value +
            ask_value
        )

        if total <= 0:
            return 0, 0, "UNKNOWN"

        imbalance = (
            bid_value -
            ask_value
        ) / total

        if imbalance > 0.15:

            return (
                imbalance,
                6,
                "BUY"
            )

        if imbalance < -0.15:

            return (
                imbalance,
                -6,
                "SELL"
            )

        return (
            imbalance,
            0,
            "NEUTRAL"
        )

    except Exception:

        return 0, 0, "UNKNOWN"


# ============================================================
# FUTURES
# ============================================================

def futures_analysis(symbol):

    funding_values = []
    oi_values = []
    exchanges = []

    for name, exchange in futures_exchanges.items():

        try:

            markets = exchange.load_markets()

            futures_symbol = symbol

            if futures_symbol not in markets:

                alternative = symbol.replace(
                    "/USDT",
                    "/USDT:USDT"
                )

                if alternative in markets:
                    futures_symbol = alternative

                else:
                    continue

            funding = None
            oi = None

            try:

                funding_data = (
                    exchange
                    .fetch_funding_rate(
                        futures_symbol
                    )
                )

                funding = (
                    funding_data
                    .get("fundingRate")
                )

            except Exception:
                pass

            try:

                oi_data = (
                    exchange
                    .fetch_open_interest(
                        futures_symbol
                    )
                )

                oi = (
                    oi_data.get(
                        "openInterestAmount"
                    )
                    or
                    oi_data.get(
                        "openInterestValue"
                    )
                    or
                    oi_data.get(
                        "openInterest"
                    )
                )

            except Exception:
                pass

            if funding is not None:

                funding_values.append(
                    float(funding)
                )

            if oi is not None:

                try:
                    oi_values.append(
                        float(oi)
                    )
                except Exception:
                    pass

            if (
                funding is not None
                or oi is not None
            ):
                exchanges.append(name)

        except Exception:
            continue

    funding_avg = (
        float(np.mean(funding_values))
        if funding_values
        else 0
    )

    oi_avg = (
        float(np.mean(oi_values))
        if oi_values
        else 0
    )

    futures_score = 0

    # Negative funding can support long bias
    if funding_avg < -0.0008:
        futures_score += 5

    # Very positive funding can warn of crowded longs
    elif funding_avg > 0.0008:
        futures_score -= 5

    return (
        funding_avg,
        oi_avg,
        futures_score,
        ",".join(exchanges)
    )


# ============================================================
# REASON
# ============================================================

def build_reason(
    direction,
    tf_scores,
    volume_score,
    breakout_score,
    pa_score,
    sr_score,
    futures_score,
    orderbook_score,
    btc_regime
):

    reasons = []

    total_tf = sum(tf_scores)

    if total_tf >= 5:
        reasons.append(
            "MTF_BULL_CONFIRM"
        )

    elif total_tf <= -5:
        reasons.append(
            "MTF_BEAR_CONFIRM"
        )

    if volume_score >= 4:
        reasons.append(
            "STRONG_BUY_VOLUME"
        )

    elif volume_score <= -4:
        reasons.append(
            "STRONG_SELL_VOLUME"
        )

    if breakout_score >= 5:
        reasons.append(
            "BULL_BREAKOUT"
        )

    elif breakout_score <= -5:
        reasons.append(
            "BEAR_BREAKOUT"
        )

    if pa_score > 0:
        reasons.append("HH_HL")

    elif pa_score < 0:
        reasons.append("LH_LL")

    if sr_score > 0:
        reasons.append("SUPPORT_ZONE")

    elif sr_score < 0:
        reasons.append("RESISTANCE_ZONE")

    if futures_score > 0:
        reasons.append(
            "FUNDING_LONG_SUPPORT"
        )

    elif futures_score < 0:
        reasons.append(
            "FUNDING_LONG_WARNING"
        )

    if orderbook_score > 0:
        reasons.append(
            "ORDERBOOK_BUY"
        )

    elif orderbook_score < 0:
        reasons.append(
            "ORDERBOOK_SELL"
        )

    if btc_regime == "BULL":
        reasons.append(
            "BTC_BULL_REGIME"
        )

    elif btc_regime == "BEAR":
        reasons.append(
            "BTC_BEAR_REGIME"
        )

    if not reasons:
        reasons.append(
            "WEAK_OR_MIXED_SETUP"
        )

    return "|".join(reasons)


# ============================================================
# COIN ANALYSIS
# ============================================================

def analyze_coin(
    item,
    btc_regime_value
):

    symbol = item["symbol"]

    exchange = spot_exchanges.get(
        "okx"
    )

    if not exchange:
        return None

    data = {}

    # --------------------------------------------------------
    # 4 TIMEFRAMES
    # --------------------------------------------------------

    for tf in TIMEFRAMES:

        df = get_ohlcv(
            exchange,
            symbol,
            tf,
            180
        )

        if df is None:
            return None

        data[tf] = df

    a15 = timeframe_analysis(
        data["15m"]
    )

    a1h = timeframe_analysis(
        data["1h"]
    )

    a4h = timeframe_analysis(
        data["4h"]
    )

    a1d = timeframe_analysis(
        data["1d"]
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price = float(
        data["1h"]["close"].iloc[-1]
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = (
        a15["momentum"] +
        a1h["momentum"] +
        a4h["momentum"] +
        a1d["momentum"]
    ) / 4

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    (
        volume_ratio,
        volume_score,
        volume_signal
    ) = volume_analysis(
        data["1h"]
    )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    (
        support,
        resistance,
        position
    ) = support_resistance(
        data["1h"]
    )

    sr_score = 0

    if position <= 0.25:
        sr_score = 5

    elif position >= 0.75:
        sr_score = -5

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    (
        breakout,
        breakout_score
    ) = breakout_analysis(
        data["1h"]
    )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    (
        pa_signal,
        pa_score
    ) = price_action(
        data["1h"]
    )

    # --------------------------------------------------------
    # FUTURES
    # --------------------------------------------------------

    (
        funding,
        oi,
        futures_score,
        futures_exchange
    ) = futures_analysis(
        symbol
    )

    # --------------------------------------------------------
    # ORDER BOOK
    # --------------------------------------------------------

    (
        orderbook_imbalance,
        orderbook_score,
        orderbook_signal
    ) = orderbook_analysis(
        exchange,
        symbol
    )

    # --------------------------------------------------------
    # BTC
    # --------------------------------------------------------

    btc_score = 0

    if btc_regime_value == "BULL":
        btc_score = 8

    elif btc_regime_value == "BEAR":
        btc_score = -8

    # --------------------------------------------------------
    # MULTI TIMEFRAME
    # --------------------------------------------------------

    tf_raw = (
        a15["score"] * 4 +
        a1h["score"] * 7 +
        a4h["score"] * 9 +
        a1d["score"] * 9
    )

    tf_raw = max(
        -29,
        min(29, tf_raw)
    )

    # --------------------------------------------------------
    # COIN REGIME
    # --------------------------------------------------------

    if (
        a4h["trend"] == "BULL"
        and a1d["trend"] == "BULL"
    ):

        coin_regime = "BULL"
        coin_regime_score = 8

    elif (
        a4h["trend"] == "BEAR"
        and a1d["trend"] == "BEAR"
    ):

        coin_regime = "BEAR"
        coin_regime_score = -8

    else:

        coin_regime = "NEUTRAL"
        coin_regime_score = 0

    # --------------------------------------------------------
    # RAW DIRECTION SCORE
    # --------------------------------------------------------

    raw_direction_score = (
        tf_raw +
        volume_score +
        breakout_score +
        pa_score +
        sr_score +
        futures_score +
        orderbook_score +
        btc_score +
        coin_regime_score
    )

    # --------------------------------------------------------
    # LONG / SHORT SCORES
    # --------------------------------------------------------

    long_score = (
        50 +
        raw_direction_score
    )

    short_score = (
        50 -
        raw_direction_score
    )

    long_score = max(
        0,
        min(100, long_score)
    )

    short_score = max(
        0,
        min(100, short_score)
    )

    # --------------------------------------------------------
    # CONFIRMATION
    # --------------------------------------------------------

    bull_tf_count = sum([
        a15["trend"] == "BULL",
        a1h["trend"] == "BULL",
        a4h["trend"] == "BULL",
        a1d["trend"] == "BULL"
    ])

    bear_tf_count = sum([
        a15["trend"] == "BEAR",
        a1h["trend"] == "BEAR",
        a4h["trend"] == "BEAR",
        a1d["trend"] == "BEAR"
    ])

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal = "NO TRADE"

    if (
        long_score >= LONG_THRESHOLD
        and long_score > short_score
        and bull_tf_count >= 3
        and breakout != "FAKE_BULL_BREAKOUT"
        and volume_ratio >= 0.8
    ):

        signal = "LONG"

    elif (
        short_score >= SHORT_THRESHOLD
        and short_score > long_score
        and bear_tf_count >= 3
        and breakout != "FAKE_BEAR_BREAKOUT"
        and volume_ratio >= 0.8
    ):

        signal = "SHORT"

    # --------------------------------------------------------
    # SIGNAL STRENGTH
    # --------------------------------------------------------

    if signal == "LONG":

        signal_strength = long_score

    elif signal == "SHORT":

        signal_strength = short_score

    else:

        # IMPORTANT:
        # NO TRADE is NOT given a fake 90/100 score.
        signal_strength = 0

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if signal == "LONG":

        confidence = (
            long_score * 0.60 +
            bull_tf_count * 7 +
            max(0, volume_score) * 0.8 +
            max(0, pa_score) * 0.5 +
            max(0, orderbook_score) * 0.5
        )

    elif signal == "SHORT":

        confidence = (
            short_score * 0.60 +
            bear_tf_count * 7 +
            max(0, -volume_score) * 0.8 +
            max(0, -pa_score) * 0.5 +
            max(0, -orderbook_score) * 0.5
        )

    else:

        confidence = 0

    confidence = max(
        0,
        min(100, confidence)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr_value = a1h["atr"]

    if atr_value <= 0:
        atr_value = price * 0.01

    # --------------------------------------------------------
    # STOP LOSS / TAKE PROFIT
    # --------------------------------------------------------

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
    # REASON
    # --------------------------------------------------------

    reason = build_reason(
        signal,
        [
            a15["score"],
            a1h["score"],
            a4h["score"],
            a1d["score"]
        ],
        volume_score,
        breakout_score,
        pa_score,
        sr_score,
        futures_score,
        orderbook_score,
        btc_regime_value
    )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "symbol": symbol,

        "signal": signal,

        "signal_strength": round(
            signal_strength,
            2
        ),

        "confidence": round(
            confidence,
            2
        ),

        "long_score": round(
            long_score,
            2
        ),

        "short_score": round(
            short_score,
            2
        ),

        "price": round(
            price,
            8
        ),

        "btc_regime": btc_regime_value,

        "coin_regime": coin_regime,

        "15m_trend": a15["trend"],

        "1h_trend": a1h["trend"],

        "4h_trend": a4h["trend"],

        "1d_trend": a1d["trend"],

        "momentum": round(
            momentum,
            3
        ),

        "volume_ratio": round(
            volume_ratio,
            3
        ),

        "volume_signal": volume_signal,

        "price_action": pa_signal,

        "breakout": breakout,

        "support": round(
            support,
            8
        ),

        "resistance": round(
            resistance,
            8
        ),

        "funding_rate": round(
            funding,
            8
        ),

        "open_interest": round(
            oi,
            4
        ),

        "future_exchange": futures_exchange,

        "orderbook_imbalance": round(
            orderbook_imbalance,
            4
        ),

        "orderbook_signal": orderbook_signal,

        "market_volume": round(
            item["volume"],
            2
        ),

        "exchange_count": item[
            "exchange_count"
        ],

        "stop_loss": round(
            stop_loss,
            8
        ),

        "take_profit": round(
            take_profit,
            8
        ),

        "reason": reason,

        "timestamp": timestamp
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("CRYPTO MASTER AI V7")
    print("ADVANCED SIGNAL + CONFIDENCE ENGINE")
    print("=" * 75)

    print("\nLoading BTC market regime...")

    btc_regime_value, btc_score = btc_regime()

    print(
        f"BTC REGIME: {btc_regime_value}"
    )

    print(
        f"BTC SCORE: {btc_score}"
    )

    print("\nDiscovering coins...")

    candidates = discover_coins()

    print(
        f"Candidate coins: {len(candidates)}"
    )

    results = []

    for index, item in enumerate(
        candidates,
        start=1
    ):

        symbol = item["symbol"]

        print(
            f"[{index}/{len(candidates)}] "
            f"{symbol}"
        )

        try:

            result = analyze_coin(
                item,
                btc_regime_value
            )

            if result:
                results.append(
                    result
                )

        except Exception as e:

            print(
                f"ERROR {symbol}: {e}"
            )

        time.sleep(0.05)

    if not results:

        print(
            "\nNo results generated."
        )

        return

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # SORT
    # ========================================================

    # LONG/SHORT first,
    # NO TRADE after them.
    signal_order = {
        "LONG": 0,
        "SHORT": 1,
        "NO TRADE": 2
    }

    df["signal_order"] = (
        df["signal"]
        .map(signal_order)
    )

    df = df.sort_values(
        by=[
            "signal_order",
            "signal_strength",
            "confidence"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )

    df = df.drop(
        columns=["signal_order"]
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    output_file = (
        "crypto_scan_results.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 75)
    print("TOP V7 SIGNALS")
    print("=" * 75)

    columns = [
        "symbol",
        "signal",
        "signal_strength",
        "confidence",
        "long_score",
        "short_score",
        "price",
        "btc_regime",
        "15m_trend",
        "1h_trend",
        "4h_trend",
        "1d_trend",
        "momentum",
        "volume_ratio",
        "price_action",
        "breakout",
        "funding_rate",
        "open_interest",
        "stop_loss",
        "take_profit"
    ]

    print(
        df[columns]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("SCAN SUMMARY")
    print("=" * 75)

    print(
        "Total analyzed:",
        len(df)
    )

    print(
        "LONG:",
        len(
            df[
                df["signal"] == "LONG"
            ]
        )
    )

    print(
        "SHORT:",
        len(
            df[
                df["signal"] == "SHORT"
            ]
        )
    )

    print(
        "NO TRADE:",
        len(
            df[
                df["signal"] == "NO TRADE"
            ]
        )
    )

    print(
        "\nCSV saved:",
        output_file
    )

    print(
        "\nCrypto Master AI V7 scan completed."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
