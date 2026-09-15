import ccxt
import pandas as pd
import numpy as np
import time
import os
import requests
from datetime import datetime, timezone


# ============================================================
# CRYPTO MASTER AI — FUTURES INTELLIGENCE v4
# ============================================================

EXCHANGES = {
    "OKX": ccxt.okx({"enableRateLimit": True}),
    "BYBIT": ccxt.bybit({"enableRateLimit": True}),
    "KUCOIN": ccxt.kucoin({"enableRateLimit": True}),
}

FUTURES_EXCHANGES = {
    "OKX": ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    }),
    "BYBIT": ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    }),
    "KUCOIN": ccxt.kucoin({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    }),
}

TIMEFRAMES = {
    "15m": 120,
    "1h": 120,
    "4h": 120,
    "1d": 120,
}

MIN_VOLUME_USDT = 250000
MAX_COINS = 80


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


# ============================================================
# OHLCV
# ============================================================

def get_ohlcv(exchange, symbol, timeframe, limit=120):

    try:
        data = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not data:
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
# TECHNICAL ANALYSIS
# ============================================================

def analyze_timeframe(df):

    if df is None or len(df) < 60:
        return None

    close = df["close"]

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)

    rsi_value = rsi(close).iloc[-1]
    atr_value = atr(df).iloc[-1]

    price = close.iloc[-1]

    momentum = (
        (price / close.iloc[-20]) - 1
    ) * 100

    if price > e20.iloc[-1] > e50.iloc[-1]:
        trend_score = 1
    elif price < e20.iloc[-1] < e50.iloc[-1]:
        trend_score = -1
    else:
        trend_score = 0

    return {
        "price": float(price),
        "ema20": float(e20.iloc[-1]),
        "ema50": float(e50.iloc[-1]),
        "ema200": float(e200.iloc[-1]),
        "rsi": float(rsi_value),
        "atr": float(atr_value),
        "momentum": float(momentum),
        "trend": trend_score
    }


# ============================================================
# BTC REGIME
# ============================================================

def btc_regime(exchange):

    try:

        df = get_ohlcv(
            exchange,
            "BTC/USDT",
            "4h",
            250
        )

        if df is None:
            return "UNKNOWN", 0

        close = df["close"]

        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        e200 = ema(close, 200).iloc[-1]

        rsi_value = rsi(close).iloc[-1]
        price = close.iloc[-1]

        if (
            price > e20
            and e20 > e50
            and e50 > e200
            and rsi_value >= 55
        ):
            return "BULLISH", 80

        elif (
            price < e20
            and e20 < e50
            and e50 < e200
            and rsi_value <= 45
        ):
            return "BEARISH", 80

        else:
            return "NEUTRAL", 50

    except Exception:
        return "UNKNOWN", 0


# ============================================================
# MARKET DISCOVERY
# ============================================================

def discover_markets():

    candidates = {}

    for name, exchange in EXCHANGES.items():

        try:

            print(f"\nLoading {name} markets...")

            exchange.load_markets()
            tickers = exchange.fetch_tickers()

            count = 0

            for symbol, ticker in tickers.items():

                try:

                    market = exchange.markets.get(symbol)

                    if not market:
                        continue

                    if not market.get("spot"):
                        continue

                    if not symbol.endswith("/USDT"):
                        continue

                    base = symbol.split("/")[0]

                    if base in [
                        "USDT",
                        "USDC",
                        "USD",
                        "EUR",
                        "DAI"
                    ]:
                        continue

                    quote_volume = ticker.get("quoteVolume")

                    if quote_volume is None:
                        continue

                    quote_volume = float(quote_volume)

                    if quote_volume < MIN_VOLUME_USDT:
                        continue

                    key = base + "/USDT"

                    if key not in candidates:

                        candidates[key] = {
                            "symbol": key,
                            "exchanges": [],
                            "volume": 0
                        }

                    candidates[key]["exchanges"].append(name)
                    candidates[key]["volume"] += quote_volume

                    count += 1

                except Exception:
                    continue

            print(
                f"{name}: {count} liquid USDT markets found"
            )

        except Exception as e:

            print(
                f"{name} error: {str(e)[:120]}"
            )

    markets = list(candidates.values())

    markets.sort(
        key=lambda x: x["volume"],
        reverse=True
    )

    markets = markets[:MAX_COINS]

    print(
        f"\nMASTER MARKET UNIVERSE: {len(markets)} coins"
    )

    return markets


# ============================================================
# FUTURES INTELLIGENCE
# ============================================================

def get_futures_data(symbol):

    best = None

    for name, exchange in FUTURES_EXCHANGES.items():

        try:

            exchange.load_markets()

            futures_symbol = symbol + ":USDT"

            if futures_symbol not in exchange.markets:

                continue

            funding_rate = np.nan
            open_interest = np.nan

            # Funding
            try:

                funding = exchange.fetch_funding_rate(
                    futures_symbol
                )

                funding_rate = funding.get(
                    "fundingRate"
                )

                if funding_rate is not None:
                    funding_rate = float(
                        funding_rate
                    )

            except Exception:
                pass

            # Open Interest
            try:

                oi = exchange.fetch_open_interest(
                    futures_symbol
                )

                open_interest = oi.get(
                    "openInterestValue"
                )

                if open_interest is None:
                    open_interest = oi.get(
                        "openInterestAmount"
                    )

                if open_interest is not None:
                    open_interest = float(
                        open_interest
                    )

            except Exception:
                pass

            if (
                not pd.isna(funding_rate)
                or not pd.isna(open_interest)
            ):

                best = {
                    "futures_exchange": name,
                    "funding_rate": funding_rate,
                    "open_interest": open_interest
                }

                break

        except Exception:
            continue

    if best is None:

        return {
            "futures_exchange": "N/A",
            "funding_rate": np.nan,
            "open_interest": np.nan
        }

    return best


# ============================================================
# FUTURES SCORE
# ============================================================

def futures_score(funding_rate):

    if pd.isna(funding_rate):

        return 0, 0

    # Funding is usually expressed as decimal.
    # Example: 0.0001 = 0.01%

    long_score = 0
    short_score = 0

    # Strong negative funding
    if funding_rate <= -0.0005:

        long_score += 8

    # Mild negative funding
    elif funding_rate < 0:

        long_score += 3

    # Strong positive funding
    elif funding_rate >= 0.0005:

        short_score += 8

    # Mild positive funding
    elif funding_rate > 0:

        short_score += 3

    return long_score, short_score


# ============================================================
# ANALYZE COIN
# ============================================================

def analyze_coin(exchange, symbol, btc_regime_name):

    try:

        analyses = {}

        for tf in TIMEFRAMES:

            df = get_ohlcv(
                exchange,
                symbol,
                tf,
                TIMEFRAMES[tf]
            )

            analysis = analyze_timeframe(df)

            if analysis is None:
                return None

            analyses[tf] = analysis

        a15 = analyses["15m"]
        a1h = analyses["1h"]
        a4h = analyses["4h"]
        a1d = analyses["1d"]

        long_score = 0
        short_score = 0

        # ----------------------------------------------------
        # TIMEFRAME TREND
        # ----------------------------------------------------

        if a15["trend"] > 0:
            long_score += 8

        elif a15["trend"] < 0:
            short_score += 8

        if a1h["trend"] > 0:
            long_score += 15

        elif a1h["trend"] < 0:
            short_score += 15

        if a4h["trend"] > 0:
            long_score += 20

        elif a4h["trend"] < 0:
            short_score += 20

        if a1d["trend"] > 0:
            long_score += 20

        elif a1d["trend"] < 0:
            short_score += 20

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        if 55 <= a1h["rsi"] <= 70:
            long_score += 10

        if 30 <= a1h["rsi"] <= 45:
            short_score += 10

        # ----------------------------------------------------
        # MOMENTUM
        # ----------------------------------------------------

        if a1h["momentum"] > 0:
            long_score += 7

        elif a1h["momentum"] < 0:
            short_score += 7

        # ----------------------------------------------------
        # BTC REGIME
        # ----------------------------------------------------

        if btc_regime_name == "BULLISH":
            long_score += 10

        elif btc_regime_name == "BEARISH":
            short_score += 10

        # ----------------------------------------------------
        # FUTURES
        # ----------------------------------------------------

        futures = get_futures_data(symbol)

        f_long, f_short = futures_score(
            futures["funding_rate"]
        )

        long_score += f_long
        short_score += f_short

        # ----------------------------------------------------
        # FINAL SIGNAL
        # ----------------------------------------------------

        if long_score >= 60 and long_score > short_score:

            signal = "LONG"
            score = long_score

        elif short_score >= 60 and short_score > long_score:

            signal = "SHORT"
            score = short_score

        else:

            signal = "NO TRADE"
            score = max(
                long_score,
                short_score
            )

        price = a1h["price"]
        atr_value = a1h["atr"]

        if signal == "LONG":

            stop_loss = price - (
                1.5 * atr_value
            )

            take_profit = price + (
                3 * atr_value
            )

        elif signal == "SHORT":

            stop_loss = price + (
                1.5 * atr_value
            )

            take_profit = price - (
                3 * atr_value
            )

        else:

            stop_loss = np.nan
            take_profit = np.nan

        return {

            "symbol": symbol,
            "signal": signal,
            "score": round(score, 2),

            "price": price,

            "btc_regime": btc_regime_name,

            "rsi_15m": round(
                a15["rsi"], 2
            ),

            "rsi_1h": round(
                a1h["rsi"], 2
            ),

            "rsi_4h": round(
                a4h["rsi"], 2
            ),

            "rsi_1d": round(
                a1d["rsi"], 2
            ),

            "momentum_1h": round(
                a1h["momentum"], 2
            ),

            "futures_exchange":
                futures["futures_exchange"],

            "funding_rate":
                futures["funding_rate"],

            "open_interest":
                futures["open_interest"],

            "stop_loss":
                round(stop_loss, 8)
                if not pd.isna(stop_loss)
                else np.nan,

            "take_profit":
                round(take_profit, 8)
                if not pd.isna(take_profit)
                else np.nan,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }

    except Exception as e:

        print(
            f"{symbol} error: {str(e)[:100]}"
        )

        return None


# ============================================================
# MAIN ENGINE
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MASTER AI — FUTURES INTELLIGENCE")
    print("=" * 70)

    markets = discover_markets()

    if not markets:

        print(
            "\nERROR: No markets discovered."
        )

        return

    exchange = EXCHANGES["OKX"]

    btc_name, btc_score = btc_regime(
        exchange
    )

    print(
        f"\nBTC REGIME: {btc_name}"
    )

    print(
        f"BTC REGIME SCORE: {btc_score}"
    )

    results = []

    for i, market in enumerate(markets):

        symbol = market["symbol"]

        print(
            f"[{i + 1}/{len(markets)}] "
            f"Analyzing {symbol}"
        )

        result = analyze_coin(
            exchange,
            symbol,
            btc_name
        )

        if result:

            result["market_volume"] = round(
                market["volume"],
                2
            )

            result["exchange_count"] = len(
                market["exchanges"]
            )

            results.append(result)

        time.sleep(0.15)

    if not results:

        print(
            "\nNo analysis results generated."
        )

        return

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="score",
        ascending=False
    )

    df.to_csv(
        "crypto_scan_results.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("TOP CRYPTO MASTER AI SETUPS")
    print("=" * 70)

    columns = [
        "symbol",
        "signal",
        "score",
        "btc_regime",
        "funding_rate",
        "open_interest",
        "rsi_1h",
        "market_volume"
    ]

    print(
        df[columns].head(20).to_string(
            index=False
        )
    )

    print("\nCSV saved successfully.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
