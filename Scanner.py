import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# CRYPTO MASTER AI V8
# BALANCED MULTI-FACTOR SIGNAL ENGINE
# ============================================================

MAX_COINS = 60
MIN_VOLUME_USDT = 250000

LONG_THRESHOLD = 65
SHORT_THRESHOLD = 65

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
EXCHANGES = ["okx", "bybit", "kucoin"]


# ============================================================
# EXCHANGE SETUP
# ============================================================

def make_exchange(name, futures=False):
    try:
        config = {
            "enableRateLimit": True,
            "timeout": 20000
        }

        if futures:
            config["options"] = {
                "defaultType": "swap"
            }

        return getattr(ccxt, name)(config)

    except Exception:
        return None


spot_exchanges = {}
futures_exchanges = {}

for name in EXCHANGES:

    spot = make_exchange(name, False)
    if spot:
        spot_exchanges[name] = spot

    futures = make_exchange(name, True)
    if futures:
        futures_exchanges[name] = futures


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

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def ATR(df, period=14):

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

def get_data(exchange, symbol, timeframe, limit=180):

    try:

        data = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            limit=limit
        )

        if not data or len(data) < 60:
            return None

        return pd.DataFrame(
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

    except Exception:
        return None


# ============================================================
# MARKET DISCOVERY
# ============================================================

def discover():

    coins = {}

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

                    if symbol not in coins:

                        coins[symbol] = {
                            "symbol": symbol,
                            "volume": 0,
                            "exchanges": set()
                        }

                    coins[symbol]["volume"] += volume

                    coins[symbol][
                        "exchanges"
                    ].add(name)

                except Exception:
                    continue

        except Exception as e:

            print(
                f"Discovery error {name}: {e}"
            )

    result = []

    for item in coins.values():

        item["exchange_count"] = len(
            item["exchanges"]
        )

        result.append(item)

    result.sort(
        key=lambda x: x["volume"],
        reverse=True
    )

    return result[:MAX_COINS]


# ============================================================
# BTC REGIME
# ============================================================

def get_btc_regime():

    exchange = spot_exchanges.get("okx")

    if not exchange:
        return "UNKNOWN", 0

    df = get_data(
        exchange,
        "BTC/USDT",
        "4h",
        200
    )

    if df is None:
        return "UNKNOWN", 0

    df["ema20"] = EMA(
        df["close"], 20
    )

    df["ema50"] = EMA(
        df["close"], 50
    )

    df["ema100"] = EMA(
        df["close"], 100
    )

    df["rsi"] = RSI(
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
# TIMEFRAME ANALYSIS
# ============================================================

def analyze_tf(df):

    if df is None or len(df) < 60:

        return {
            "trend": "UNKNOWN",
            "direction": 0,
            "momentum": 0,
            "rsi": 50,
            "atr": 0
        }

    df = df.copy()

    df["ema20"] = EMA(
        df["close"], 20
    )

    df["ema50"] = EMA(
        df["close"], 50
    )

    df["rsi"] = RSI(
        df["close"]
    )

    df["atr"] = ATR(df)

    last = df.iloc[-1]

    direction = 0

    if last["close"] > last["ema20"]:
        direction += 1
    else:
        direction -= 1

    if last["ema20"] > last["ema50"]:
        direction += 1
    else:
        direction -= 1

    if last["rsi"] > 55:
        direction += 1

    elif last["rsi"] < 45:
        direction -= 1

    momentum = (
        (
            last["close"] /
            df["close"].iloc[-10]
        ) - 1
    ) * 100

    if momentum > 0.5:
        direction += 1

    elif momentum < -0.5:
        direction -= 1

    if direction >= 2:
        trend = "BULL"

    elif direction <= -2:
        trend = "BEAR"

    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "direction": direction,
        "momentum": momentum,
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"])
    }


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if df is None or len(df) < 30:
        return 1, 0, "UNKNOWN"

    current = float(
        df["volume"].iloc[-1]
    )

    average = float(
        df["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    if average <= 0:
        return 1, 0, "UNKNOWN"

    ratio = current / average

    price_change = (
        (
            df["close"].iloc[-1] /
            df["close"].iloc[-2]
        ) - 1
    ) * 100

    if ratio >= 2:

        if price_change > 0:
            return ratio, 8, "STRONG_BUY_VOLUME"

        if price_change < 0:
            return ratio, -8, "STRONG_SELL_VOLUME"

    if ratio >= 1.5:

        if price_change > 0:
            return ratio, 5, "BUY_VOLUME"

        if price_change < 0:
            return ratio, -5, "SELL_VOLUME"

    if ratio >= 1.1:

        if price_change > 0:
            return ratio, 2, "MILD_BUY_VOLUME"

        if price_change < 0:
            return ratio, -2, "MILD_SELL_VOLUME"

    return ratio, 0, "NORMAL_VOLUME"


# ============================================================
# PRICE ACTION
# ============================================================

def price_action(df):

    if df is None or len(df) < 20:
        return "UNKNOWN", 0

    highs = df["high"].tail(10).values
    lows = df["low"].tail(10).values

    hh = highs[-1] > highs[-3]
    hl = lows[-1] > lows[-3]

    lh = highs[-1] < highs[-3]
    ll = lows[-1] < lows[-3]

    if hh and hl:
        return "HH_HL", 6

    if lh and ll:
        return "LH_LL", -6

    return "MIXED", 0


# ============================================================
# BREAKOUT
# ============================================================

def breakout(df):

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

    close = float(
        df["close"].iloc[-1]
    )

    high = float(
        df["high"].iloc[-1]
    )

    low = float(
        df["low"].iloc[-1]
    )

    avg_volume = float(
        df["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    current_volume = float(
        df["volume"].iloc[-1]
    )

    if close > previous_high:

        if current_volume >= avg_volume * 1.3:
            return "BULL_BREAKOUT", 8

        return "WEAK_BULL_BREAKOUT", 3

    if close < previous_low:

        if current_volume >= avg_volume * 1.3:
            return "BEAR_BREAKOUT", -8

        return "WEAK_BEAR_BREAKOUT", -3

    if (
        high > previous_high
        and close < previous_high
    ):
        return "FAKE_BULL_BREAKOUT", -6

    if (
        low < previous_low
        and close > previous_low
    ):
        return "FAKE_BEAR_BREAKOUT", 6

    return "NONE", 0


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
# ORDER BOOK
# ============================================================

def orderbook(exchange, symbol):

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
            float(p) * float(a)
            for p, a in bids
        )

        ask_value = sum(
            float(p) * float(a)
            for p, a in asks
        )

        total = bid_value + ask_value

        if total <= 0:
            return 0, 0, "UNKNOWN"

        imbalance = (
            bid_value - ask_value
        ) / total

        if imbalance > 0.15:
            return imbalance, 5, "BUY"

        if imbalance < -0.15:
            return imbalance, -5, "SELL"

        return imbalance, 0, "NEUTRAL"

    except Exception:
        return 0, 0, "UNKNOWN"


# ============================================================
# FUTURES
# ============================================================

def futures_data(symbol):

    funding_list = []
    oi_list = []
    used = []

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

                data = exchange.fetch_funding_rate(
                    futures_symbol
                )

                funding = data.get(
                    "fundingRate"
                )

            except Exception:
                pass

            try:

                data = exchange.fetch_open_interest(
                    futures_symbol
                )

                oi = (
                    data.get(
                        "openInterestAmount"
                    )
                    or
                    data.get(
                        "openInterestValue"
                    )
                    or
                    data.get(
                        "openInterest"
                    )
                )

            except Exception:
                pass

            if funding is not None:
                funding_list.append(
                    float(funding)
                )

            if oi is not None:

                try:
                    oi_list.append(
                        float(oi)
                    )
                except Exception:
                    pass

            if (
                funding is not None
                or oi is not None
            ):
                used.append(name)

        except Exception:
            continue

    funding = (
        float(np.mean(funding_list))
        if funding_list
        else 0
    )

    oi = (
        float(np.mean(oi_list))
        if oi_list
        else 0
    )

    # Funding interpretation
    if funding < -0.0008:
        funding_score = 5
        funding_signal = "LONG_SUPPORT"

    elif funding > 0.0008:
        funding_score = -5
        funding_signal = "LONG_CROWDED"

    else:
        funding_score = 0
        funding_signal = "NORMAL"

    return (
        funding,
        oi,
        funding_score,
        funding_signal,
        ",".join(used)
    )


# ============================================================
# MASTER ANALYSIS
# ============================================================

def analyze_coin(item, btc_regime):

    exchange = spot_exchanges.get("okx")

    if not exchange:
        return None

    symbol = item["symbol"]

    frames = {}

    for tf in TIMEFRAMES:

        df = get_data(
            exchange,
            symbol,
            tf,
            180
        )

        if df is None:
            return None

        frames[tf] = df

    # --------------------------------------------------------
    # TIMEFRAMES
    # --------------------------------------------------------

    a15 = analyze_tf(
        frames["15m"]
    )

    a1h = analyze_tf(
        frames["1h"]
    )

    a4h = analyze_tf(
        frames["4h"]
    )

    a1d = analyze_tf(
        frames["1d"]
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price = float(
        frames["1h"]["close"].iloc[-1]
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
        frames["1h"]
    )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    (
        pa_signal,
        pa_score
    ) = price_action(
        frames["1h"]
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    (
        breakout_signal,
        breakout_score
    ) = breakout(
        frames["1h"]
    )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    (
        support,
        resistance,
        position
    ) = support_resistance(
        frames["1h"]
    )

    sr_score = 0

    if position <= 0.25:
        sr_score = 5

    elif position >= 0.75:
        sr_score = -5

    # --------------------------------------------------------
    # ORDER BOOK
    # --------------------------------------------------------

    (
        imbalance,
        orderbook_score,
        orderbook_signal
    ) = orderbook(
        exchange,
        symbol
    )

    # --------------------------------------------------------
    # FUTURES
    # --------------------------------------------------------

    (
        funding,
        oi,
        funding_score,
        funding_signal,
        futures_exchange
    ) = futures_data(
        symbol
    )

    # --------------------------------------------------------
    # TIMEFRAME BALANCE
    # --------------------------------------------------------

    tf_score = (
        a15["direction"] * 2 +
        a1h["direction"] * 5 +
        a4h["direction"] * 7 +
        a1d["direction"] * 6
    )

    tf_score = max(
        -40,
        min(40, tf_score)
    )

    # --------------------------------------------------------
    # BTC REGIME
    # --------------------------------------------------------

    btc_score = 0

    if btc_regime == "BULL":
        btc_score = 5

    elif btc_regime == "BEAR":
        btc_score = -5

    # --------------------------------------------------------
    # COIN REGIME
    # --------------------------------------------------------

    if (
        a4h["trend"] == "BULL"
        and a1d["trend"] == "BULL"
    ):

        coin_regime = "BULL"
        coin_regime_score = 5

    elif (
        a4h["trend"] == "BEAR"
        and a1d["trend"] == "BEAR"
    ):

        coin_regime = "BEAR"
        coin_regime_score = -5

    else:

        coin_regime = "NEUTRAL"
        coin_regime_score = 0

    # ========================================================
    # RAW DIRECTION
    # ========================================================

    raw = (
        tf_score +
        volume_score +
        pa_score +
        breakout_score +
        sr_score +
        orderbook_score +
        funding_score +
        btc_score +
        coin_regime_score
    )

    # Keep within useful range
    raw = max(
        -50,
        min(50, raw)
    )

    # ========================================================
    # LONG / SHORT SCORE
    # ========================================================

    long_score = 50 + raw
    short_score = 50 - raw

    long_score = max(
        0,
        min(100, long_score)
    )

    short_score = max(
        0,
        min(100, short_score)
    )

    # ========================================================
    # CONFIRMATION
    # ========================================================

    bull_count = sum([
        a15["trend"] == "BULL",
        a1h["trend"] == "BULL",
        a4h["trend"] == "BULL",
        a1d["trend"] == "BULL"
    ])

    bear_count = sum([
        a15["trend"] == "BEAR",
        a1h["trend"] == "BEAR",
        a4h["trend"] == "BEAR",
        a1d["trend"] == "BEAR"
    ])

    # ========================================================
    # SIGNAL
    # ========================================================

    signal = "NO TRADE"

    # LONG: 2+ bullish TFs + score
    if (
        long_score >= LONG_THRESHOLD
        and bull_count >= 2
        and long_score > short_score
        and breakout_signal != "FAKE_BULL_BREAKOUT"
    ):

        signal = "LONG"

    # SHORT: 2+ bearish TFs + score
    elif (
        short_score >= SHORT_THRESHOLD
        and bear_count >= 2
        and short_score > long_score
        and breakout_signal != "FAKE_BEAR_BREAKOUT"
    ):

        signal = "SHORT"

    # ========================================================
    # EXTRA MARKET FILTER
    # ========================================================

    if volume_ratio < 0.65:

        signal = "NO TRADE"

    # ========================================================
    # SIGNAL STRENGTH
    # ========================================================

    if signal == "LONG":
        strength = long_score

    elif signal == "SHORT":
        strength = short_score

    else:
        strength = max(
            long_score,
            short_score
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if signal == "LONG":

        confidence = (
            long_score * 0.55
            + bull_count * 8
            + max(0, volume_score) * 0.7
            + max(0, pa_score) * 0.6
            + max(0, orderbook_score) * 0.5
        )

    elif signal == "SHORT":

        confidence = (
            short_score * 0.55
            + bear_count * 8
            + max(0, -volume_score) * 0.7
            + max(0, -pa_score) * 0.6
            + max(0, -orderbook_score) * 0.5
        )

    else:

        confidence = 0

    confidence = max(
        0,
        min(100, confidence)
    )

    # ========================================================
    # SL / TP
    # ========================================================

    atr_value = a1h["atr"]

    if atr_value <= 0:
        atr_value = price * 0.01

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

    # ========================================================
    # REASONS
    # ========================================================

    reasons = []

    if tf_score >= 10:
        reasons.append("MTF_BULL")

    elif tf_score <= -10:
        reasons.append("MTF_BEAR")

    if volume_score >= 4:
        reasons.append("BUY_VOLUME")

    elif volume_score <= -4:
        reasons.append("SELL_VOLUME")

    if pa_score > 0:
        reasons.append("HH_HL")

    elif pa_score < 0:
        reasons.append("LH_LL")

    if breakout_score >= 5:
        reasons.append("BULL_BREAKOUT")

    elif breakout_score <= -5:
        reasons.append("BEAR_BREAKOUT")

    if sr_score > 0:
        reasons.append("NEAR_SUPPORT")

    elif sr_score < 0:
        reasons.append("NEAR_RESISTANCE")

    if orderbook_score > 0:
        reasons.append("BUY_ORDERBOOK")

    elif orderbook_score < 0:
        reasons.append("SELL_ORDERBOOK")

    if funding_score > 0:
        reasons.append("NEGATIVE_FUNDING")

    elif funding_score < 0:
        reasons.append("POSITIVE_FUNDING")

    if btc_regime == "BULL":
        reasons.append("BTC_BULL")

    elif btc_regime == "BEAR":
        reasons.append("BTC_BEAR")

    if not reasons:
        reasons.append("MIXED_SETUP")

    reason = "|".join(reasons)

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "symbol": symbol,

        "signal": signal,

        "signal_strength": round(
            strength,
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

        "btc_regime": btc_regime,

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

        "breakout": breakout_signal,

        "support": round(
            support,
            8
        ),

        "resistance": round(
            resistance,
            8
        ),

        "future_exchange": futures_exchange,

        "funding_rate": round(
            funding,
            8
        ),

        "funding_signal": funding_signal,

        "open_interest": round(
            oi,
            4
        ),

        "orderbook_imbalance": round(
            imbalance,
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

    print("=" * 80)
    print("CRYPTO MASTER AI V8")
    print("BALANCED MULTI-FACTOR SIGNAL ENGINE")
    print("=" * 80)

    # BTC
    print("\nChecking BTC regime...")

    btc_regime, btc_score = get_btc_regime()

    print(
        f"BTC REGIME: {btc_regime}"
    )

    print(
        f"BTC SCORE: {btc_score}"
    )

    # Discover
    print("\nDiscovering market...")

    candidates = discover()

    print(
        f"Coins found: {len(candidates)}"
    )

    results = []

    for i, item in enumerate(
        candidates,
        start=1
    ):

        symbol = item["symbol"]

        print(
            f"[{i}/{len(candidates)}] "
            f"Analyzing {symbol}"
        )

        try:

            result = analyze_coin(
                item,
                btc_regime
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
            "\nNO RESULTS GENERATED"
        )

        return

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # SORT
    # ========================================================

    order = {
        "LONG": 0,
        "SHORT": 1,
        "NO TRADE": 2
    }

    df["sort_order"] = (
        df["signal"].map(order)
    )

    df = df.sort_values(
        by=[
            "sort_order",
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
        columns=["sort_order"]
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        "crypto_scan_results.csv",
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 80)
    print("CRYPTO MASTER AI V8 RESULTS")
    print("=" * 80)

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

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

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
        "\nCSV: crypto_scan_results.csv"
    )

    print(
        "\nCRYPTO MASTER AI V8 COMPLETE"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
