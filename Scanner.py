# ============================================================
# CRYPTO MASTER AI v2
# CoinMarketCap Full Market Universe
# Multi-Exchange Scanner
# LONG / SHORT / NO TRADE
# Spot + Futures + Funding + Open Interest
# Multi-Timeframe Technical Analysis
# ============================================================

import os
import time
import traceback
from datetime import datetime, timezone

import requests
import ccxt
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

EXCHANGES = ["okx", "bybit", "binance", "kucoin"]

TIMEFRAMES = {
    "15m": 120,
    "1h": 150,
    "4h": 150,
    "1d": 150,
}

# CoinMarketCap market universe
CMC_LIMIT = 5000

# Deep technical scan
MAX_COINS_PER_EXCHANGE = 20

MIN_24H_VOLUME = 500_000

MIN_SCORE = 65

OUTPUT_FILE = "crypto_scan_results.csv"
CMC_FILE = "cmc_full_market.csv"

REQUEST_TIMEOUT = 20


# ============================================================
# COINMARKETCAP
# ============================================================

def get_cmc_market():

    print("\n==============================")
    print("COINMARKETCAP MARKET SCAN")
    print("==============================")

    url = (
        "https://pro-api.coinmarketcap.com"
        "/public-api/v3/cryptocurrency/listings/latest"
    )

    params = {
        "start": 1,
        "limit": CMC_LIMIT,
        "convert": "USD",
        "sort": "market_cap",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        coins = data.get("data", [])

        rows = []

        for coin in coins:

            quote = coin.get("quote", {}).get("USD", {})

            rows.append({
                "cmc_rank": coin.get("cmc_rank"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "slug": coin.get("slug"),
                "price_usd": quote.get("price"),
                "market_cap": quote.get("market_cap"),
                "volume_24h": quote.get("volume_24h"),
                "change_1h": quote.get("percent_change_1h"),
                "change_24h": quote.get("percent_change_24h"),
                "change_7d": quote.get("percent_change_7d"),
                "market_cap_dominance": quote.get(
                    "market_cap_dominance"
                ),
                "num_market_pairs": coin.get(
                    "num_market_pairs"
                ),
            })

        df = pd.DataFrame(rows)

        if not df.empty:

            df.to_csv(
                CMC_FILE,
                index=False
            )

        print(
            f"CMC coins loaded: {len(df)}"
        )

        return df

    except Exception as e:

        print(
            "CMC error:",
            str(e)
        )

        return pd.DataFrame()


# ============================================================
# EXCHANGE
# ============================================================

def create_exchange(name):

    try:

        cls = getattr(ccxt, name)

        exchange = cls({
            "enableRateLimit": True,
            "timeout": 20000,
        })

        exchange.load_markets()

        return exchange

    except Exception as e:

        print(
            f"{name}: exchange load failed:",
            str(e)
        )

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

    value = 100 - (
        100 / (1 + rs)
    )

    return value.fillna(50)


def atr(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def macd(series):

    fast = ema(series, 12)
    slow = ema(series, 26)

    macd_line = fast - slow

    signal = ema(
        macd_line,
        9
    )

    histogram = (
        macd_line - signal
    )

    return (
        macd_line,
        signal,
        histogram
    )


# ============================================================
# OHLCV
# ============================================================

def get_ohlcv(
    exchange,
    symbol,
    timeframe,
    limit=150
):

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
                "volume",
            ]
        )

        return df

    except Exception:

        return None


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def analyze_timeframe(df):

    if df is None:
        return None

    if len(df) < 60:
        return None

    close = df["close"]

    e20 = ema(
        close,
        20
    )

    e50 = ema(
        close,
        50
    )

    e200 = ema(
        close,
        min(200, len(df))
    )

    rsi_values = rsi(
        close
    )

    macd_line, signal, histogram = macd(
        close
    )

    atr_values = atr(
        df
    )

    volume_avg = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    volume_ratio = (
        df["volume"].iloc[-1]
        /
        max(
            volume_avg.iloc[-1],
            1e-12
        )
    )

    price = float(
        close.iloc[-1]
    )

    rsi_value = float(
        rsi_values.iloc[-1]
    )

    macd_value = float(
        histogram.iloc[-1]
    )

    atr_value = float(
        atr_values.iloc[-1]
    )

    long_score = 0
    short_score = 0

    # -------------------------
    # TREND
    # -------------------------

    if price > e20.iloc[-1]:
        long_score += 8
    else:
        short_score += 8

    if e20.iloc[-1] > e50.iloc[-1]:
        long_score += 8
    else:
        short_score += 8

    if price > e200.iloc[-1]:
        long_score += 8
    else:
        short_score += 8

    # -------------------------
    # RSI
    # -------------------------

    if 52 <= rsi_value <= 70:
        long_score += 7

    if 30 <= rsi_value <= 48:
        short_score += 7

    # -------------------------
    # MACD
    # -------------------------

    if macd_value > 0:
        long_score += 7
    else:
        short_score += 7

    # -------------------------
    # VOLUME
    # -------------------------

    if volume_ratio >= 1.2:

        if long_score > short_score:
            long_score += 4
        else:
            short_score += 4

    # -------------------------
    # MOMENTUM
    # -------------------------

    recent_return = (
        close.iloc[-1]
        /
        close.iloc[-6]
        - 1
    ) * 100

    if recent_return > 0:
        long_score += 4
    else:
        short_score += 4

    return {
        "price": price,
        "long": long_score,
        "short": short_score,
        "rsi": rsi_value,
        "atr": atr_value,
        "volume_ratio": volume_ratio,
    }


# ============================================================
# MULTI TIMEFRAME
# ============================================================

def multi_timeframe_analysis(
    exchange,
    symbol
):

    weights = {
        "15m": 1.0,
        "1h": 1.5,
        "4h": 2.0,
        "1d": 2.5,
    }

    results = {}

    total_long = 0
    total_short = 0

    total_weight = sum(
        weights.values()
    )

    for timeframe, limit in TIMEFRAMES.items():

        df = get_ohlcv(
            exchange,
            symbol,
            timeframe,
            limit
        )

        analysis = analyze_timeframe(
            df
        )

        if analysis is None:
            continue

        results[timeframe] = analysis

        total_long += (
            analysis["long"]
            * weights[timeframe]
        )

        total_short += (
            analysis["short"]
            * weights[timeframe]
        )

        time.sleep(
            exchange.rateLimit / 1000
        )

    if not results:
        return None

    max_points = (
        46 * total_weight
    )

    long_score = (
        total_long
        / max_points
        * 100
    )

    short_score = (
        total_short
        / max_points
        * 100
    )

    return {
        "long_score": round(
            long_score,
            2
        ),
        "short_score": round(
            short_score,
            2
        ),
        "details": results,
    }


# ============================================================
# BTC MARKET REGIME
# ============================================================

def btc_regime(exchange):

    try:

        df = get_ohlcv(
            exchange,
            "BTC/USDT",
            "1h",
            220
        )

        if df is None:
            return {
                "regime": "UNKNOWN",
                "score": 0
            }

        close = df["close"]

        e20 = ema(
            close,
            20
        )

        e50 = ema(
            close,
            50
        )

        e200 = ema(
            close,
            200
        )

        rsi_value = float(
            rsi(close).iloc[-1]
        )

        price = float(
            close.iloc[-1]
        )

        if (
            price > e20.iloc[-1]
            and e20.iloc[-1] > e50.iloc[-1]
            and e50.iloc[-1] > e200.iloc[-1]
            and rsi_value >= 55
        ):

            return {
                "regime": "BULLISH",
                "score": 80
            }

        elif (
            price < e20.iloc[-1]
            and e20.iloc[-1] < e50.iloc[-1]
            and e50.iloc[-1] < e200.iloc[-1]
            and rsi_value <= 45
        ):

            return {
                "regime": "BEARISH",
                "score": 80
            }

        else:

            return {
                "regime": "NEUTRAL",
                "score": 50
            }

    except Exception:

        return {
            "regime": "UNKNOWN",
            "score": 0
        }


# ============================================================
# FUTURES DATA
# ============================================================

def get_futures_data(
    exchange,
    base
):

    funding = None
    open_interest = None

    try:

        futures_symbol = (
            f"{base}/USDT:USDT"
        )

        if futures_symbol not in exchange.markets:
            return funding, open_interest

        market = exchange.markets[
            futures_symbol
        ]

        if not market.get("swap"):
            return funding, open_interest

        # Funding
        try:

            funding_data = (
                exchange.fetch_funding_rate(
                    futures_symbol
                )
            )

            funding = funding_data.get(
                "fundingRate"
            )

        except Exception:
            pass

        # Open interest
        try:

            oi_data = (
                exchange.fetch_open_interest(
                    futures_symbol
                )
            )

            open_interest = oi_data.get(
                "openInterestAmount"
            )

        except Exception:
            pass

    except Exception:
        pass

    return funding, open_interest


# ============================================================
# MARKET DISCOVERY
# ============================================================

def get_liquid_usdt_symbols(
    exchange,
    cmc_df
):

    try:

        tickers = (
            exchange.fetch_tickers()
        )

    except Exception as e:

        print(
            "Ticker error:",
            str(e)
        )

        return []

    # CMC symbols
    cmc_symbols = set()

    if not cmc_df.empty:

        cmc_symbols = set(
            cmc_df[
                "symbol"
            ]
            .astype(str)
            .str.upper()
            .tolist()
        )

    candidates = []

    for symbol, ticker in tickers.items():

        try:

            market = exchange.markets.get(
                symbol
            )

            if not market:
                continue

            if market.get("spot") is not True:
                continue

            if market.get("quote") != "USDT":
                continue

            base = market.get(
                "base",
                ""
            ).upper()

            if base not in cmc_symbols:
                continue

            quote_volume = (
                ticker.get(
                    "quoteVolume"
                )
            )

            if quote_volume is None:

                last = ticker.get(
                    "last"
                )

                volume = ticker.get(
                    "baseVolume"
                )

                if (
                    last is not None
                    and volume is not None
                ):

                    quote_volume = (
                        last * volume
                    )

            if quote_volume is None:
                continue

            quote_volume = float(
                quote_volume
            )

            if quote_volume < MIN_24H_VOLUME:
                continue

            candidates.append({
                "symbol": symbol,
                "base": base,
                "volume": quote_volume,
            })

        except Exception:
            continue

    candidates.sort(
        key=lambda x: x["volume"],
        reverse=True
    )

    return candidates[
        :MAX_COINS_PER_EXCHANGE
    ]


# ============================================================
# MASTER DECISION
# ============================================================

def master_decision(
    long_score,
    short_score,
    btc_regime_value,
    funding
):

    long_final = float(
        long_score
    )

    short_final = float(
        short_score
    )

    # BTC regime
    if btc_regime_value == "BULLISH":

        long_final += 8
        short_final -= 8

    elif btc_regime_value == "BEARISH":

        short_final += 8
        long_final -= 8

    # Funding
    if funding is not None:

        try:

            funding = float(
                funding
            )

            # Positive funding =
            # crowded longs
            if funding > 0.0005:

                long_final -= 4
                short_final += 2

            # Negative funding =
            # crowded shorts
            elif funding < -0.0005:

                short_final += 4
                long_final += 2

        except Exception:
            pass

    long_final = max(
        0,
        min(100, long_final)
    )

    short_final = max(
        0,
        min(100, short_final)
    )

    if (
        long_final >= MIN_SCORE
        and
        long_final - short_final >= 10
    ):

        decision = "LONG"

    elif (
        short_final >= MIN_SCORE
        and
        short_final - long_final >= 10
    ):

        decision = "SHORT"

    else:

        decision = "NO TRADE"

    confidence = max(
        long_final,
        short_final
    )

    return (
        decision,
        round(long_final, 2),
        round(short_final, 2),
        round(confidence, 2)
    )


# ============================================================
# RISK LEVELS
# ============================================================

def risk_levels(
    price,
    atr_value,
    decision
):

    if atr_value is None:
        return None, None, None

    atr_value = max(
        float(atr_value),
        price * 0.001
    )

    if decision == "LONG":

        stop_loss = (
            price - 1.5 * atr_value
        )

        tp1 = (
            price + 2 * atr_value
        )

        tp2 = (
            price + 3 * atr_value
        )

    elif decision == "SHORT":

        stop_loss = (
            price + 1.5 * atr_value
        )

        tp1 = (
            price - 2 * atr_value
        )

        tp2 = (
            price - 3 * atr_value
        )

    else:

        return None, None, None

    return (
        round(stop_loss, 8),
        round(tp1, 8),
        round(tp2, 8)
    )


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

        url = (
            f"https://api.telegram.org/bot"
            f"{token}/sendMessage"
        )

        requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": message,
            },
            timeout=15
        )

    except Exception as e:

        print(
            "Telegram error:",
            str(e)
        )


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    start_time = time.time()

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print("\n")
    print("=" * 70)
    print("CRYPTO MASTER AI v2")
    print("FULL CMC MARKET + MULTI EXCHANGE")
    print(timestamp)
    print("=" * 70)

    # --------------------------------------------------------
    # FULL CMC MARKET
    # --------------------------------------------------------

    cmc_df = get_cmc_market()

    # --------------------------------------------------------
    # RESULT COLUMNS
    # --------------------------------------------------------

    columns = [
        "timestamp",
        "cmc_rank",
        "coin",
        "exchange",
        "symbol",
        "price",
        "volume_24h",
        "btc_regime",
        "long_score",
        "short_score",
        "confidence",
        "decision",
        "funding",
        "open_interest",
        "rsi_15m",
        "rsi_1h",
        "rsi_4h",
        "rsi_1d",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
    ]

    all_results = []

    # --------------------------------------------------------
    # EXCHANGES
    # --------------------------------------------------------

    for exchange_name in EXCHANGES:

        print("\n")
        print("=" * 70)
        print(
            f"EXCHANGE: {exchange_name.upper()}"
        )
        print("=" * 70)

        exchange = create_exchange(
            exchange_name
        )

        if exchange is None:
            continue

        try:

            regime_data = btc_regime(
                exchange
            )

            regime = regime_data[
                "regime"
            ]

            print(
                "BTC REGIME:",
                regime
            )

            candidates = (
                get_liquid_usdt_symbols(
                    exchange,
                    cmc_df
                )
            )

            print(
                "Tradable liquid coins:",
                len(candidates)
            )

            for index, item in enumerate(
                candidates,
                start=1
            ):

                symbol = item[
                    "symbol"
                ]

                base = item[
                    "base"
                ]

                print(
                    f"[{index}/{len(candidates)}]"
                    f" {symbol}"
                )

                try:

                    analysis = (
                        multi_timeframe_analysis(
                            exchange,
                            symbol
                        )
                    )

                    if analysis is None:
                        continue

                    funding, oi = (
                        get_futures_data(
                            exchange,
                            base
                        )
                    )

                    decision, long_final, short_final, confidence = (
                        master_decision(
                            analysis[
                                "long_score"
                            ],
                            analysis[
                                "short_score"
                            ],
                            regime,
                            funding
                        )
                    )

                    details = analysis[
                        "details"
                    ]

                    price = None

                    if "1h" in details:
                        price = details[
                            "1h"
                        ]["price"]

                    else:

                        first_tf = next(
                            iter(details)
                        )

                        price = details[
                            first_tf
                        ]["price"]

                    atr_value = None

                    if "1h" in details:

                        atr_value = details[
                            "1h"
                        ]["atr"]

                    stop_loss, tp1, tp2 = (
                        risk_levels(
                            price,
                            atr_value,
                            decision
                        )
                    )

                    cmc_rank = None
                    coin_name = base
                    volume_24h = item[
                        "volume"
                    ]

                    if not cmc_df.empty:

                        match = cmc_df[
                            cmc_df[
                                "symbol"
                            ].str.upper()
                            == base.upper()
                        ]

                        if not match.empty:

                            # If duplicate symbols exist,
                            # use first matching record.
                            row = match.iloc[0]

                            cmc_rank = row[
                                "cmc_rank"
                            ]

                            coin_name = row[
                                "name"
                            ]

                    result = {

                        "timestamp":
                            timestamp,

                        "cmc_rank":
                            cmc_rank,

                        "coin":
                            coin_name,

                        "exchange":
                            exchange_name,

                        "symbol":
                            symbol,

                        "price":
                            price,

                        "volume_24h":
                            volume_24h,

                        "btc_regime":
                            regime,

                        "long_score":
                            long_final,

                        "short_score":
                            short_final,

                        "confidence":
                            confidence,

                        "decision":
                            decision,

                        "funding":
                            funding,

                        "open_interest":
                            oi,

                        "rsi_15m":
                            details.get(
                                "15m",
                                {}
                            ).get(
                                "rsi"
                            ),

                        "rsi_1h":
                            details.get(
                                "1h",
                                {}
                            ).get(
                                "rsi"
                            ),

                        "rsi_4h":
                            details.get(
                                "4h",
                                {}
                            ).get(
                                "rsi"
                            ),

                        "rsi_1d":
                            details.get(
                                "1d",
                                {}
                            ).get(
                                "rsi"
                            ),

                        "stop_loss":
                            stop_loss,

                        "take_profit_1":
                            tp1,

                        "take_profit_2":
                            tp2,
                    }

                    all_results.append(
                        result
                    )

                    if decision != "NO TRADE":

                        print(
                            "   >>>",
                            decision,
                            "|",
                            "Confidence:",
                            confidence
                        )

                except Exception as e:

                    print(
                        "   Coin error:",
                        str(e)
                    )

        except Exception as e:

            print(
                f"{exchange_name} error:",
                str(e)
            )

        finally:

            try:
                exchange.close()
            except Exception:
                pass

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results,
        columns=columns
    )

    if not results_df.empty:

        results_df = results_df.sort_values(
            by="confidence",
            ascending=False
        )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY TOP SIGNALS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP AI SIGNALS")
    print("=" * 70)

    if results_df.empty:

        print(
            "No results generated."
        )

    else:

        signals = results_df[
            results_df[
                "decision"
            ] != "NO TRADE"
        ]

        if signals.empty:

            print(
                "NO TRADE signals currently."
            )

        else:

            top_signals = signals.head(
                15
            )

            for _, row in top_signals.iterrows():

                print(
                    f"{row['decision']:8} "
                    f"{row['exchange']:8} "
                    f"{row['symbol']:18} "
                    f"Confidence={row['confidence']}"
                )

    # --------------------------------------------------------
    # TELEGRAM SUMMARY
    # --------------------------------------------------------

    if not results_df.empty:

        signals = results_df[
            results_df[
                "decision"
            ] != "NO TRADE"
        ]

        if not signals.empty:

            message = (
                "🤖 CRYPTO MASTER AI\n\n"
                f"Time: {timestamp}\n\n"
            )

            for _, row in signals.head(
                5
            ).iterrows():

                message += (
                    f"{row['decision']} "
                    f"{row['symbol']} "
                    f"({row['exchange']})\n"
                    f"Confidence: "
                    f"{row['confidence']}\n"
                    f"Price: "
                    f"{row['price']}\n"
                    f"SL: "
                    f"{row['stop_loss']}\n"
                    f"TP1: "
                    f"{row['take_profit_1']}\n"
                    f"TP2: "
                    f"{row['take_profit_2']}\n\n"
                )

            send_telegram(
                message
            )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    elapsed = round(
        time.time() - start_time,
        2
    )

    print("\n")
    print("=" * 70)
    print(
        "SCAN COMPLETE"
    )
    print(
        f"Coins analyzed: {len(results_df)}"
    )
    print(
        f"Saved: {OUTPUT_FILE}"
    )
    print(
        f"Runtime: {elapsed} seconds"
    )
    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            "\nFATAL ERROR:",
            str(e)
        )

        traceback.print_exc()

        # Still create CSV so GitHub
        # Actions does not fail because
        # of a missing output file.

        pd.DataFrame(
            columns=[
                "timestamp",
                "cmc_rank",
                "coin",
                "exchange",
                "symbol",
                "price",
                "volume_24h",
                "btc_regime",
                "long_score",
                "short_score",
                "confidence",
                "decision",
                "funding",
                "open_interest",
                "rsi_15m",
                "rsi_1h",
                "rsi_4h",
                "rsi_1d",
                "stop_loss",
                "take_profit_1",
                "take_profit_2",
            ]
        ).to_csv(
            OUTPUT_FILE,
            index=False
        )
