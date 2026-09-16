import os
import time
import math
import requests
import ccxt
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# ============================================================
# CRYPTO MASTER AI V9
# Technical + Fundamental + Derivatives + Order Flow + News
# ============================================================

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_COINS = 60

CMC_BASE = "https://pro-api.coinmarketcap.com/public-api"

STABLECOINS = {
    "USDT/USDT", "USDC/USDT", "FDUSD/USDT", "DAI/USDT",
    "USDE/USDT", "USDD/USDT", "TUSD/USDT", "USDP/USDT",
    "PYUSD/USDT", "USDS/USDT", "USDG/USDT", "RLUSD/USDT"
}

# Public RSS feeds
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://cryptoslate.com/feed/",
]

NEWS_KEYWORDS_POSITIVE = [
    "approval", "approved", "adoption", "partnership", "launch",
    "listing", "buy", "buying", "inflow", "bullish", "surge",
    "growth", "upgrade", "integration", "institutional",
    "etf", "funding", "investment", "record high", "mainnet"
]

NEWS_KEYWORDS_NEGATIVE = [
    "hack", "hacked", "exploit", "attack", "lawsuit", "ban",
    "banned", "delist", "delisting", "outflow", "fraud",
    "scam", "bearish", "crash", "collapse", "liquidation",
    "sec", "investigation", "sanction", "stolen", "vulnerability",
    "bankruptcy", "shutdown", "rug pull"
]

HIGH_IMPACT_WORDS = [
    "hack", "exploit", "sec", "etf", "approval", "approved",
    "ban", "banned", "lawsuit", "liquidation", "bankruptcy",
    "delisting", "regulation", "fed", "cpi", "rate decision"
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def clamp(x, low, high):
    return max(low, min(high, x))


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ============================================================
# EXCHANGES
# ============================================================

def make_exchange(exchange_id, futures=False):
    try:
        options = {}

        if futures:
            options["defaultType"] = "swap"

        exchange_class = getattr(ccxt, exchange_id)

        exchange = exchange_class({
            "enableRateLimit": True,
            "timeout": 15000,
            "options": options
        })

        return exchange

    except Exception as e:
        print(f"{exchange_id} setup error: {e}")
        return None


SPOT_EXCHANGES = {
    "OKX": make_exchange("okx"),
    "BYBIT": make_exchange("bybit"),
    "KUCOIN": make_exchange("kucoin"),
}

FUTURES_EXCHANGES = {
    "OKX": make_exchange("okx", True),
    "BYBIT": make_exchange("bybit", True),
    "KUCOIN": make_exchange("kucoin", True),
}


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    return result.fillna(50)


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

def get_data(exchange, symbol, timeframe, limit=250):

    if exchange is None:
        return None

    try:
        data = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not data or len(data) < 80:
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

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms"
        )

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        df = df.dropna()

        return df

    except Exception:
        return None


# ============================================================
# DISCOVER MARKET
# ============================================================

def discover():

    market_data = {}

    for name, exchange in SPOT_EXCHANGES.items():

        if exchange is None:
            continue

        try:
            markets = exchange.load_markets()

            for symbol, market in markets.items():

                if not market.get("spot"):
                    continue

                if not symbol.endswith("/USDT"):
                    continue

                if symbol in STABLECOINS:
                    continue

                base = symbol.split("/")[0]

                if base in {
                    "USDT", "USDC", "DAI", "FDUSD",
                    "TUSD", "USDE", "USDD"
                }:
                    continue

                try:
                    ticker = exchange.fetch_ticker(symbol)

                    quote_volume = safe_float(
                        ticker.get("quoteVolume")
                    )

                    last = safe_float(
                        ticker.get("last")
                    )

                    if last <= 0:
                        continue

                    if quote_volume < 250000:
                        continue

                    if symbol not in market_data:
                        market_data[symbol] = {
                            "volume": 0,
                            "exchanges": set()
                        }

                    market_data[symbol]["volume"] += quote_volume
                    market_data[symbol]["exchanges"].add(name)

                except Exception:
                    continue

        except Exception as e:
            print(f"Discovery error {name}: {e}")

    ranked = sorted(
        market_data.items(),
        key=lambda x: x[1]["volume"],
        reverse=True
    )

    result = []

    for symbol, info in ranked[:MAX_COINS]:
        result.append({
            "symbol": symbol,
            "market_volume": info["volume"],
            "exchange_count": len(info["exchanges"]),
            "exchange_names": ",".join(info["exchanges"])
        })

    print(f"Discovered {len(result)} coins")

    return result


# ============================================================
# BTC REGIME
# ============================================================

def get_btc_regime():

    exchange = SPOT_EXCHANGES.get("OKX")

    if exchange is None:
        return {
            "regime": "UNKNOWN",
            "score": 0
        }

    df = get_data(
        exchange,
        "BTC/USDT",
        "4h",
        250
    )

    if df is None:
        return {
            "regime": "UNKNOWN",
            "score": 0
        }

    close = df["close"]

    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e100 = ema(close, 100).iloc[-1]

    current = close.iloc[-1]

    r = rsi(close).iloc[-1]

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

    if r > 55:
        score += 1
    elif r < 45:
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


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def analyze_tf(df):

    if df is None or len(df) < 80:
        return {
            "trend": "UNKNOWN",
            "score": 0,
            "rsi": 50
        }

    close = df["close"]

    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e100 = ema(close, 100).iloc[-1]

    current = close.iloc[-1]

    r = safe_float(
        rsi(close).iloc[-1],
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

    if r > 55:
        score += 1

    if r < 45:
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
        "rsi": r
    }


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if df is None or len(df) < 30:
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

    if df is None or len(df) < 10:
        return {
            "score": 0,
            "label": "UNKNOWN"
        }

    recent = df.tail(5)

    green = (
        recent["close"] > recent["open"]
    ).sum()

    red = (
        recent["close"] < recent["open"]
    ).sum()

    if green >= 4:
        return {
            "score": 3,
            "label": "STRONG_BULLISH"
        }

    if red >= 4:
        return {
            "score": -3,
            "label": "STRONG_BEARISH"
        }

    if green > red:
        return {
            "score": 1,
            "label": "BULLISH"
        }

    if red > green:
        return {
            "score": -1,
            "label": "BEARISH"
        }

    return {
        "score": 0,
        "label": "NEUTRAL"
    }


# ============================================================
# BREAKOUT
# ============================================================

def breakout(df):

    if df is None or len(df) < 30:
        return {
            "score": 0,
            "label": "UNKNOWN"
        }

    current = df["close"].iloc[-1]

    previous_high = df["high"].iloc[-21:-1].max()
    previous_low = df["low"].iloc[-21:-1].min()

    if current > previous_high:
        return {
            "score": 4,
            "label": "BULLISH_BREAKOUT"
        }

    if current < previous_low:
        return {
            "score": -4,
            "label": "BEARISH_BREAKDOWN"
        }

    return {
        "score": 0,
        "label": "NO_BREAKOUT"
    }


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance(df):

    if df is None or len(df) < 50:
        return {
            "score": 0,
            "support": 0,
            "resistance": 0
        }

    recent = df.tail(50)

    support = recent["low"].min()
    resistance = recent["high"].max()
    price = df["close"].iloc[-1]

    score = 0

    if resistance > 0:
        resistance_distance = (
            resistance - price
        ) / price

        if resistance_distance < 0.01:
            score -= 2

    if support > 0:
        support_distance = (
            price - support
        ) / price

        if support_distance < 0.01:
            score += 2

    return {
        "score": score,
        "support": support,
        "resistance": resistance
    }


# ============================================================
# ORDER BOOK - MULTI EXCHANGE
# ============================================================

def orderbook_multi(symbol):

    imbalances = []
    total_values = []

    for name, exchange in SPOT_EXCHANGES.items():

        if exchange is None:
            continue

        try:
            book = exchange.fetch_order_book(
                symbol,
                limit=20
            )

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            if not bids or not asks:
                continue

            bid_value = sum(
                safe_float(p) * safe_float(v)
                for p, v in bids
            )

            ask_value = sum(
                safe_float(p) * safe_float(v)
                for p, v in asks
            )

            total = bid_value + ask_value

            if total <= 0:
                continue

            imbalance = (
                bid_value - ask_value
            ) / total

            imbalances.append(imbalance)
            total_values.append(total)

        except Exception:
            continue

    if not imbalances:
        return {
            "imbalance": 0,
            "score": 0,
            "signal": "UNKNOWN",
            "exchanges": 0
        }

    weighted = np.average(
        imbalances,
        weights=total_values
    )

    weighted = clamp(
        weighted,
        -1,
        1
    )

    score = int(
        clamp(
            weighted * 8,
            -5,
            5
        )
    )

    if weighted >= 0.20:
        signal = "BULLISH"

    elif weighted <= -0.20:
        signal = "BEARISH"

    else:
        signal = "NEUTRAL"

    return {
        "imbalance": weighted,
        "score": score,
        "signal": signal,
        "exchanges": len(imbalances)
    }


# ============================================================
# FUTURES
# ============================================================

def futures_data(symbol):

    funding_values = []
    oi_values = []

    for name, exchange in FUTURES_EXCHANGES.items():

        if exchange is None:
            continue

        try:

            try:
                funding = exchange.fetch_funding_rate(
                    symbol
                )

                fr = safe_float(
                    funding.get("fundingRate")
                )

                funding_values.append(fr)

            except Exception:
                pass

            try:
                oi = exchange.fetch_open_interest(
                    symbol
                )

                oi_value = safe_float(
                    oi.get("openInterestValue")
                )

                if oi_value > 0:
                    oi_values.append(oi_value)

            except Exception:
                pass

        except Exception:
            continue

    if funding_values:
        avg_funding = float(
            np.mean(funding_values)
        )
    else:
        avg_funding = 0

    if avg_funding > 0.0005:
        funding_score = -2
        funding_signal = "LONGS_CROWDED"

    elif avg_funding < -0.0005:
        funding_score = 2
        funding_signal = "SHORTS_CROWDED"

    else:
        funding_score = 0
        funding_signal = "NEUTRAL"

    total_oi = sum(oi_values)

    return {
        "funding": avg_funding,
        "funding_score": funding_score,
        "funding_signal": funding_signal,
        "open_interest": total_oi,
        "futures_exchanges": len(funding_values)
    }


# ============================================================
# CMC FUNDAMENTAL DATA
# ============================================================

def cmc_request(path, params=None):

    try:

        url = CMC_BASE + path

        response = requests.get(
            url,
            params=params or {},
            headers={
                "Accept": "application/json"
            },
            timeout=15
        )

        if response.status_code != 200:
            return None

        data = response.json()

        status = data.get("status", {})

        if str(
            status.get("error_code", 0)
        ) != "0":

            return None

        return data.get("data")

    except Exception as e:
        print(f"CMC error: {e}")
        return None


def get_cmc_fundamentals(symbols):

    clean_symbols = []

    for s in symbols:

        base = s.split("/")[0]

        if base not in clean_symbols:
            clean_symbols.append(base)

    result = {}

    # CMC symbol queries are batched to reduce requests.
    # Production-grade workflows should eventually map
    # CMC numeric IDs because symbols can collide.

    for start in range(0, len(clean_symbols), 25):

        batch = clean_symbols[start:start + 25]

        try:

            data = cmc_request(
                "/v3/cryptocurrency/quotes/latest",
                {
                    "symbol": ",".join(batch),
                    "convert": "USD"
                }
            )

            if not isinstance(data, dict):
                continue

            for key, item in data.items():

                quote = (
                    item.get("quote", {})
                    .get("USD", {})
                )

                result[key.upper()] = {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "symbol": item.get("symbol"),
                    "rank": item.get("cmc_rank"),
                    "market_cap": safe_float(
                        quote.get("market_cap")
                    ),
                    "volume_24h": safe_float(
                        quote.get("volume_24h")
                    ),
                    "percent_24h": safe_float(
                        quote.get("percent_change_24h")
                    ),
                    "percent_7d": safe_float(
                        quote.get("percent_change_7d")
                    ),
                    "percent_30d": safe_float(
                        quote.get("percent_change_30d")
                    ),
                    "circulating_supply": safe_float(
                        item.get("circulating_supply")
                    ),
                    "total_supply": safe_float(
                        item.get("total_supply")
                    ),
                    "max_supply": safe_float(
                        item.get("max_supply")
                    ),
                    "fdv": safe_float(
                        quote.get("fully_diluted_market_cap")
                    ),
                    "market_pairs": safe_float(
                        item.get("num_market_pairs")
                    ),
                    "date_added": item.get("date_added")
                }

        except Exception as e:
            print(f"CMC batch error: {e}")

        time.sleep(0.4)

    return result


# ============================================================
# FUNDAMENTAL SCORE
# ============================================================

def fundamental_analysis(data):

    if not data:
        return {
            "score": 50,
            "rating": "UNKNOWN",
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
            "reason": "FUNDAMENTAL_DATA_UNAVAILABLE"
        }

    score = 50
    reasons = []

    rank = safe_float(data.get("rank"))
    market_cap = safe_float(data.get("market_cap"))
    circulating = safe_float(
        data.get("circulating_supply")
    )
    total_supply = safe_float(
        data.get("total_supply")
    )
    max_supply = safe_float(
        data.get("max_supply")
    )
    fdv = safe_float(data.get("fdv"))
    pairs = safe_float(
        data.get("market_pairs")
    )

    # Market cap rank
    if rank > 0:

        if rank <= 50:
            score += 5
            reasons.append("TOP_50")

        elif rank <= 100:
            score += 3
            reasons.append("TOP_100")

        elif rank <= 250:
            score += 1

        elif rank > 1000:
            score -= 3
            reasons.append("LOW_RANK")

    # Market cap / FDV
    mc_fdv_ratio = 0

    if fdv > 0 and market_cap > 0:

        mc_fdv_ratio = market_cap / fdv

        if mc_fdv_ratio >= 0.80:
            score += 4
            reasons.append("LOW_FDV_DILUTION")

        elif mc_fdv_ratio >= 0.50:
            score += 2

        elif mc_fdv_ratio < 0.25:
            score -= 4
            reasons.append("HIGH_FDV_DILUTION")

    # Circulating / max supply
    supply_ratio = 0

    if max_supply > 0 and circulating > 0:

        supply_ratio = (
            circulating / max_supply
        )

        if supply_ratio >= 0.80:
            score += 3
            reasons.append("HIGH_CIRCULATION")

        elif supply_ratio >= 0.50:
            score += 1

        elif supply_ratio < 0.25:
            score -= 3
            reasons.append("LOW_CIRCULATION")

    # Market pairs / liquidity ecosystem
    if pairs >= 200:
        score += 3
        reasons.append("WIDE_MARKET_ACCESS")

    elif pairs >= 100:
        score += 2

    elif pairs >= 50:
        score += 1

    elif pairs < 10:
        score -= 2

    # Infinite / unknown max supply
    if max_supply <= 0:
        score -= 1

    score = int(
        clamp(score, 0, 100)
    )

    if score >= 70:
        rating = "STRONG"

    elif score >= 50:
        rating = "NEUTRAL"

    else:
        rating = "WEAK"

    date_added = data.get("date_added")
    age_days = 0

    if date_added:
        try:
            dt = datetime.fromisoformat(
                date_added.replace("Z", "+00:00")
            )

            age_days = (
                datetime.now(timezone.utc) - dt
            ).days

        except Exception:
            age_days = 0

    return {
        "score": score,
        "rating": rating,
        "market_cap": market_cap,
        "rank": rank,
        "circulating_supply": circulating,
        "total_supply": total_supply,
        "max_supply": max_supply,
        "fdv": fdv,
        "mc_fdv_ratio": mc_fdv_ratio,
        "supply_ratio": supply_ratio,
        "market_pairs": pairs,
        "age_days": age_days,
        "reason": ",".join(reasons) if reasons else "NEUTRAL"
    }


# ============================================================
# NEWS ENGINE
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return " ".join(
        str(text).split()
    )


def parse_rss(url):

    articles = []

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent":
                "Mozilla/5.0 CryptoMasterAI/9.0"
            },
            timeout=12
        )

        if response.status_code != 200:
            return articles

        root = ET.fromstring(
            response.content
        )

        for item in root.iter():

            tag = item.tag.lower()

            if not tag.endswith("item"):
                continue

            title = ""
            link = ""
            pubdate = ""

            for child in list(item):

                ctag = child.tag.lower()

                if ctag.endswith("title"):
                    title = clean_text(
                        child.text
                    )

                elif ctag.endswith("link"):
                    link = clean_text(
                        child.text
                    )

                elif ctag.endswith("pubdate"):
                    pubdate = clean_text(
                        child.text
                    )

            if title:
                articles.append({
                    "title": title,
                    "link": link,
                    "date": pubdate,
                    "source": url
                })

    except Exception:
        pass

    return articles


def load_news():

    all_news = []

    for feed in NEWS_FEEDS:

        articles = parse_rss(feed)

        all_news.extend(
            articles[:40]
        )

        time.sleep(0.2)

    # Remove duplicate titles
    unique = {}
    for article in all_news:
        key = article["title"].lower().strip()
        unique[key] = article

    return list(unique.values())


def news_for_coin(symbol, articles):

    base = symbol.split("/")[0].upper()

    aliases = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"],
        "BNB": ["binance coin", "bnb"],
        "DOGE": ["dogecoin", "doge"],
        "ADA": ["cardano", "ada"],
        "AVAX": ["avalanche", "avax"],
        "DOT": ["polkadot", "dot"],
        "LINK": ["chainlink", "link"],
        "SUI": ["sui"],
        "TRX": ["tron", "trx"],
        "TON": ["toncoin", "ton"],
        "LTC": ["litecoin", "ltc"],
        "BCH": ["bitcoin cash", "bch"],
    }

    words = aliases.get(
        base,
        [base.lower()]
    )

    matched = []

    for article in articles:

        title = article["title"].lower()

        if any(
            w in title
            for w in words
        ):
            matched.append(article)

    return matched[:10]


def analyze_news(symbol, articles):

    matched = news_for_coin(
        symbol,
        articles
    )

    if not matched:
        return {
            "score": 0,
            "sentiment": "NO_NEWS",
            "impact": "NONE",
            "count": 0,
            "headlines": "",
            "reason": "NO_MATCHING_NEWS"
        }

    score = 0
    high_count = 0

    selected_titles = []

    for article in matched:

        title = article["title"].lower()

        pos = sum(
            1 for word in NEWS_KEYWORDS_POSITIVE
            if word in title
        )

        neg = sum(
            1 for word in NEWS_KEYWORDS_NEGATIVE
            if word in title
        )

        article_score = pos - neg

        score += article_score

        if any(
            word in title
            for word in HIGH_IMPACT_WORDS
        ):
            high_count += 1

        selected_titles.append(
            article["title"]
        )

    score = clamp(
        score * 12,
        -100,
        100
    )

    if score >= 25:
        sentiment = "BULLISH"

    elif score <= -25:
        sentiment = "BEARISH"

    else:
        sentiment = "NEUTRAL"

    if high_count >= 2:
        impact = "CRITICAL"

    elif high_count == 1:
        impact = "HIGH"

    elif len(matched) >= 3:
        impact = "MEDIUM"

    else:
        impact = "LOW"

    return {
        "score": score,
        "sentiment": sentiment,
        "impact": impact,
        "count": len(matched),
        "headlines": " || ".join(
            selected_titles[:5]
        ),
        "reason": (
            f"{sentiment}_{impact}"
        )
    }


# ============================================================
# GLOBAL MARKET / FEAR GREED
# ============================================================

def get_global_market():

    data = cmc_request(
        "/v1/global-metrics/quotes/latest",
        {
            "convert": "USD"
        }
    )

    if not data:
        return {}

    quote = (
        data.get("quote", {})
        .get("USD", {})
    )

    return {
        "market_cap": safe_float(
            quote.get("total_market_cap")
        ),
        "volume_24h": safe_float(
            quote.get("total_volume_24h")
        ),
        "btc_dominance": safe_float(
            data.get("btc_dominance")
        ),
        "eth_dominance": safe_float(
            data.get("eth_dominance")
        )
    }


def get_fear_greed():

    data = cmc_request(
        "/v3/fear-and-greed/latest"
    )

    if not data:
        return {
            "value": 0,
            "classification": "UNKNOWN"
        }

    return {
        "value": safe_float(
            data.get("value")
        ),
        "classification": data.get(
            "value_classification",
            "UNKNOWN"
        )
    }


# ============================================================
# COMPLETE COIN ANALYSIS
# ============================================================

def analyze_coin(
    symbol,
    market_info,
    cmc_data,
    news_articles,
    btc_regime,
    global_market,
    fear_greed
):

    exchange = SPOT_EXCHANGES.get("OKX")

    if exchange is None:
        return None

    timeframe_results = {}

    for tf in TIMEFRAMES:

        df = get_data(
            exchange,
            symbol,
            tf,
            250
        )

        timeframe_results[tf] = analyze_tf(df)

        time.sleep(0.05)

    valid_scores = [
        v["score"]
        for v in timeframe_results.values()
        if v["trend"] != "UNKNOWN"
    ]

    if not valid_scores:
        return None

    # Weighted technical timeframe score
    weights = {
        "15m": 0.15,
        "1h": 0.30,
        "4h": 0.35,
        "1d": 0.20
    }

    tf_score = 0

    for tf, weight in weights.items():
        tf_score += (
            timeframe_results[tf]["score"]
            * weight
        )

    # Main 1h data
    df = get_data(
        exchange,
        symbol,
        "1h",
        250
    )

    if df is None:
        return None

    price = safe_float(
        df["close"].iloc[-1]
    )

    volume = volume_analysis(df)
    action = price_action(df)
    br = breakout(df)
    sr = support_resistance(df)

    ob = orderbook_multi(symbol)
    fut = futures_data(symbol)

    fundamental = fundamental_analysis(
        cmc_data
    )

    news = analyze_news(
        symbol,
        news_articles
    )

    # ========================================================
    # TECHNICAL SCORE
    # ========================================================

    technical_raw = (
        tf_score
        + volume["score"] * 0.9
        + action["score"] * 0.9
        + br["score"] * 1.0
        + sr["score"] * 0.6
    )

    technical_score = clamp(
        50 + technical_raw * 4,
        0,
        100
    )

    # ========================================================
    # DERIVATIVES + ORDER FLOW
    # ========================================================

    derivatives_score = clamp(
        50 + fut["funding_score"] * 7,
        0,
        100
    )

    orderflow_score = clamp(
        50 + ob["score"] * 6,
        0,
        100
    )

    # ========================================================
    # BTC REGIME
    # ========================================================

    btc_bias = 0

    if btc_regime["regime"] == "BULLISH":
        btc_bias = 5

    elif btc_regime["regime"] == "BEARISH":
        btc_bias = -5

    # ========================================================
    # NEWS BIAS
    # ========================================================

    news_bias = (
        news["score"] / 10
    )

    if news["impact"] == "CRITICAL":
        news_bias *= 1.5

    news_bias = clamp(
        news_bias,
        -15,
        15
    )

    # ========================================================
    # FUNDAMENTAL BIAS
    # Keep fundamentals moderate for short-term trading.
    # ========================================================

    fundamental_bias = (
        fundamental["score"] - 50
    ) * 0.20

    fundamental_bias = clamp(
        fundamental_bias,
        -10,
        10
    )

    # ========================================================
    # FINAL RAW DIRECTION SCORE
    # ========================================================

    final_bias = (
        (technical_score - 50) * 0.55
        + (derivatives_score - 50) * 0.10
        + (orderflow_score - 50) * 0.10
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

    # ========================================================
    # TREND CONFIRMATION
    # ========================================================

    bullish_tf = sum(
        1 for x in timeframe_results.values()
        if x["trend"] == "BULLISH"
    )

    bearish_tf = sum(
        1 for x in timeframe_results.values()
        if x["trend"] == "BEARISH"
    )

    # ========================================================
    # SIGNAL
    # ========================================================

    signal = "NO TRADE"

    if (
        long_score >= 65
        and bullish_tf >= 2
        and long_score > short_score + 8
    ):
        signal = "LONG"

    elif (
        short_score >= 65
        and bearish_tf >= 2
        and short_score > long_score + 8
    ):
        signal = "SHORT"

    # Extreme news risk can cancel normal setups.
    if news["impact"] == "CRITICAL":
        # Do not automatically reverse the trade.
        # Only require stronger confirmation.
        if signal == "LONG" and long_score < 75:
            signal = "NO TRADE"

        if signal == "SHORT" and short_score < 75:
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
    # SETUP CONFIDENCE
    # NOT WIN PROBABILITY
    # ========================================================

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
        abs(ob["score"]) * 2,
        10
    )

    confirmation += min(
        abs(fut["funding_score"]) * 2,
        8
    )

    confirmation += min(
        abs(news["score"]) * 0.10,
        10
    )

    setup_confidence = clamp(
        50 + confirmation,
        0,
        100
    )

    # ========================================================
    # STOP LOSS / TAKE PROFIT
    # ========================================================

    atr_value = safe_float(
        atr(df).iloc[-1]
    )

    if atr_value <= 0:
        atr_value = price * 0.01

    if signal == "LONG":

        stop_loss = price - (
            atr_value * 1.5
        )

        take_profit = price + (
            atr_value * 3
        )

    elif signal == "SHORT":

        stop_loss = price + (
            atr_value * 1.5
        )

        take_profit = price - (
            atr_value * 3
        )

    else:

        stop_loss = 0
        take_profit = 0

    # ========================================================
    # REASON
    # ========================================================

    reasons = [
        f"TF:{tf_score:.1f}",
        f"TECH:{technical_score:.1f}",
        f"FUND:{fundamental['score']}",
        f"NEWS:{news['sentiment']}",
        f"NEWS_IMPACT:{news['impact']}",
        f"OB:{ob['signal']}",
        f"FUNDING:{fut['funding_signal']}",
        f"BTC:{btc_regime['regime']}"
    ]

    return {
        "symbol": symbol,
        "signal": signal,
        "signal_strength": round(strength, 2),
        "confidence": round(
            setup_confidence,
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

        "price": price,

        "btc_regime":
            btc_regime["regime"],

        "btc_regime_score":
            btc_regime["score"],

        "15m_trend":
            timeframe_results["15m"]["trend"],

        "1h_trend":
            timeframe_results["1h"]["trend"],

        "4h_trend":
            timeframe_results["4h"]["trend"],

        "1d_trend":
            timeframe_results["1d"]["trend"],

        "technical_score":
            round(technical_score, 2),

        "fundamental_score":
            fundamental["score"],

        "fundamental_rating":
            fundamental["rating"],

        "market_cap":
            fundamental["market_cap"],

        "cmc_rank":
            fundamental["rank"],

        "circulating_supply":
            fundamental["circulating_supply"],

        "total_supply":
            fundamental["total_supply"],

        "max_supply":
            fundamental["max_supply"],

        "fdv":
            fundamental["fdv"],

        "mc_fdv_ratio":
            round(
                fundamental["mc_fdv_ratio"],
                4
            ),

        "supply_ratio":
            round(
                fundamental["supply_ratio"],
                4
            ),

        "market_pairs":
            fundamental["market_pairs"],

        "asset_age_days":
            fundamental["age_days"],

        "fundamental_reason":
            fundamental["reason"],

        "volume_ratio":
            round(
                volume["ratio"],
                3
            ),

        "price_action":
            action["label"],

        "breakout":
            br["label"],

        "support":
            sr["support"],

        "resistance":
            sr["resistance"],

        "funding":
            fut["funding"],

        "funding_signal":
            fut["funding_signal"],

        "open_interest":
            fut["open_interest"],

        "futures_exchanges":
            fut["futures_exchanges"],

        "orderbook_imbalance":
            round(
                ob["imbalance"],
                4
            ),

        "orderbook_signal":
            ob["signal"],

        "orderbook_exchanges":
            ob["exchanges"],

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
            global_market.get(
                "market_cap",
                0
            ),

        "btc_dominance":
            global_market.get(
                "btc_dominance",
                0
            ),

        "fear_greed":
            fear_greed.get(
                "value",
                0
            ),

        "fear_greed_class":
            fear_greed.get(
                "classification",
                "UNKNOWN"
            ),

        "market_volume":
            market_info["market_volume"],

        "exchange_count":
            market_info["exchange_count"],

        "exchanges":
            market_info["exchange_names"],

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "reason":
            " | ".join(reasons),

        "timestamp":
            now_utc()
    }


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):

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
                "text": text
            },
            timeout=10
        )

    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MASTER AI V9")
    print("Technical + Fundamental + News + Order Flow")
    print("=" * 70)

    # 1. Discover market
    coins = discover()

    if not coins:
        print("No coins discovered.")
        return

    # 2. Fundamental data
    symbols = [
        x["symbol"]
        for x in coins
    ]

    print("Loading CoinMarketCap fundamentals...")

    cmc_data = get_cmc_fundamentals(
        symbols
    )

    print(
        f"CMC data received: "
        f"{len(cmc_data)} assets"
    )

    # 3. News
    print("Loading crypto news...")

    news_articles = load_news()

    print(
        f"News articles loaded: "
        f"{len(news_articles)}"
    )

    # 4. BTC regime
    btc_regime = get_btc_regime()

    print(
        f"BTC regime: "
        f"{btc_regime['regime']}"
    )

    # 5. Global market
    global_market = get_global_market()

    # 6. Fear & Greed
    fear_greed = get_fear_greed()

    print(
        f"Fear & Greed: "
        f"{fear_greed['value']} "
        f"{fear_greed['classification']}"
    )

    # 7. Analyze
    results = []

    for index, item in enumerate(
        coins,
        start=1
    ):

        symbol = item["symbol"]

        print(
            f"[{index}/{len(coins)}] "
            f"Analyzing {symbol}"
        )

        try:

            result = analyze_coin(
                symbol,
                item,
                cmc_data.get(
                    symbol.split("/")[0]
                ),
                news_articles,
                btc_regime,
                global_market,
                fear_greed
            )

            if result:
                results.append(result)

        except Exception as e:
            print(
                f"{symbol} error: {e}"
            )

    if not results:
        print("No analysis results.")
        return

    # ========================================================
    # SORT
    # ========================================================

    signal_order = {
        "LONG": 0,
        "SHORT": 1,
        "NO TRADE": 2
    }

    results.sort(
        key=lambda x: (
            signal_order.get(
                x["signal"],
                9
            ),
            -x["signal_strength"]
        )
    )

    df = pd.DataFrame(
        results
    )

    # Save CSV
    df.to_csv(
        "crypto_scan_results.csv",
        index=False
    )

    # ========================================================
    # PRINT TOP RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TOP CRYPTO MASTER AI V9 RESULTS")
    print("=" * 70)

    display_columns = [
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
        "funding_signal",
        "btc_regime"
    ]

    available = [
        c for c in display_columns
        if c in df.columns
    ]

    print(
        df[available]
        .head(20)
        .to_string(index=False)
    )

    # ========================================================
    # SIGNAL COUNTS
    # ========================================================

    long_count = (
        df["signal"] == "LONG"
    ).sum()

    short_count = (
        df["signal"] == "SHORT"
    ).sum()

    no_trade = (
        df["signal"] == "NO TRADE"
    ).sum()

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"LONG: {long_count}"
    )

    print(
        f"SHORT: {short_count}"
    )

    print(
        f"NO TRADE: {no_trade}"
    )

    print(
        f"Total analyzed: {len(df)}"
    )

    print(
        "\nIMPORTANT: confidence is a model setup score, "
        "NOT a guaranteed win probability."
    )

    # ========================================================
    # TELEGRAM TOP SIGNALS
    # ========================================================

    strong = df[
        (
            df["signal"].isin(
                ["LONG", "SHORT"]
            )
        )
        &
        (
            df["signal_strength"] >= 75
        )
    ].head(10)

    if len(strong) > 0:

        message = (
            "🚀 CRYPTO MASTER AI V9\n\n"
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
                f"Funding: "
                f"{row['funding_signal']}\n\n"
            )

        message += (
            "⚠️ AI score is not a guaranteed "
            "profit probability."
        )

        send_telegram(
            message
        )

    print("\nScan completed.")
    print(
        f"Saved: crypto_scan_results.csv"
    )


if __name__ == "__main__":
    main()
