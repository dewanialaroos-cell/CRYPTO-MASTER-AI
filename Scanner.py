import ccxt
import pandas as pd
import numpy as np
import time
import os
import requests
from datetime import datetime, timezone


# ============================================================
# CRYPTO MASTER AI — FOUNDATION v3
# ============================================================

EXCHANGES = {
    "OKX": ccxt.okx({"enableRateLimit": True}),
    "BYBIT": ccxt.bybit({"enableRateLimit": True}),
    "KUCOIN": ccxt.kucoin({"enableRateLimit": True}),
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
# BTC MARKET REGIME
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
# ANALYZE COIN
# ============================================================

def analyze_coin(exchange, symbol, btc_regime_name):

    result = {
        "symbol": symbol,
        "exchange": exchange.id,
        "signal": "NO TRADE",
        "score": 0
    }

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

        # 15m
        if a15["trend"] > 0:
            long_score += 8

        if a15["trend"] < 0:
            short_score += 8

        # 1H
        if a1h["trend"] > 0:
            long_score += 15

        if a1h["trend"] < 0:
            short_score += 15

        # 4H
        if a4h["trend"] > 0:
            long_score += 20

        if a4h["trend"] < 0:
            short_score += 20

        # 1D
        if a1d["trend"] > 0:
            long_score += 20

        if a1d["trend"] < 0:
            short_score += 20

        # RSI
        if 55 <= a1h["rsi"] <= 70:
            long_score += 10

        if 30 <= a1h["rsi"] <= 45:
            short_score += 10

        # Momentum
        if a1h["momentum"] > 0:
            long_score += 7

        if a1h["momentum"] < 0:
            short_score += 7

        # BTC regime
        if btc_regime_name == "BULLISH":
            long_score += 10

        elif btc_regime_name == "BEARISH":
            short_score += 10

        # Decide
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

            stop_loss = price - (1.5 * atr_value)
            take_profit = price + (3 * atr_value)

        elif signal == "SHORT":

            stop_loss = price + (1.5 * atr_value)
            take_profit = price - (3 * atr_value)

        else:

            stop_loss = np.nan
            take_profit = np.nan

        result.update({

            "price": price,

            "btc_regime": btc_regime_name,

            "signal": signal,

            "score": round(score, 2),

            "rsi_15m": round(a15["rsi"], 2),
            "rsi_1h": round(a1h["rsi"], 2),
            "rsi_4h": round(a4h["rsi"], 2),
            "rsi_1d": round(a1d["rsi"], 2),

            "momentum_1h": round(
                a1h["momentum"], 2
            ),

            "stop_loss": (
                round(stop_loss, 8)
                if not pd.isna(stop_loss)
                else np.nan
            ),

            "take_profit": (
                round(take_profit, 8)
                if not pd.isna(take_profit)
                else np.nan
            ),

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat()

        })

        return result

    except Exception as e:

        print(
            f"{symbol} error: {str(e)[:100]}"
        )

        return None


# ============================================================
# MAIN ENGINE
# ============================================================

def main():

    print("=" * 60)

    print("CRYPTO MASTER AI — MARKET ENGINE")

    print("=" * 60)

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

    print("\n" + "=" * 60)

    print("TOP CRYPTO MASTER AI SETUPS")

    print("=" * 60)

    print(
        df[
            [
                "symbol",
                "signal",
                "score",
                "btc_regime",
                "price",
                "rsi_1h",
                "market_volume"
            ]
        ].head(20).to_string(
            index=False
        )
    )

    print("\nCSV saved successfully.")

    # ========================================================
    # TELEGRAM ALERT
    # ========================================================

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if token and chat_id:

        top = df[
            df["signal"].isin(
                ["LONG", "SHORT"]
            )
        ].head(3)

        if not top.empty:

            message = (
                "🤖 CRYPTO MASTER AI\n\n"
                f"BTC Regime: {btc_name}\n\n"
            )

            for _, row in top.iterrows():

                message += (
                    f"{row['symbol']} "
                    f"{row['signal']} "
                    f"Score {row['score']}\n"
                )

            try:

                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": message
                    },
                    timeout=10
                )

                print(
                    "Telegram alert sent."
                )

            except Exception as e:

                print(
                    f"Telegram error: {e}"
                )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
