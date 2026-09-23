import os
import json
import math
import re
import time
import traceback
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import requests
import ccxt
import pandas as pd
import numpy as np


# ============================================================
# CRYPTO MASTER AI V9.2 (FIXED & OPTIMIZED)
# Binance + OKX + Bybit + KuCoin
# Spot + Futures + CMC + News + Order Book + Learning
# ============================================================

UTC = timezone.utc

# Updated standard CMC API Base URL
CMC_BASE = "https://pro-api.coinmarketcap.com"

MAX_COINS = 30
OHLCV_LIMIT = 220
HTTP_TIMEOUT = 15

STABLES = {
    "USDT", "USDC", "FDUSD", "DAI", "USDE", "USDS",
    "USDD", "TUSD", "USDP", "PYUSD", "USDG", "RLUSD",
    "EURC", "EURT", "USTC"
}

SPOT_IDS = [
    "binance",
    "okx",
    "bybit",
    "kucoin"
]

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

# Add CMC API Key from environment if available
cmc_api_key = os.getenv("CMC_API_KEY", "")

session.headers.update({
    "User-Agent": "Crypto-Master-AI/9.2",
    "Accept": "application/json,text/xml,application/xml,*/*",
    "X-CMC_PRO_API_KEY": cmc_api_key
})


# ============================================================
# BASIC HELPERS
# ============================================================

def now():
    return datetime.now(UTC)


def log(message):
    print(f"[{now().isoformat()}] {message}", flush=True)


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
        return max(low, min(high, float(value)))
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
            "timeout": 15000,
            "options": {
                "defaultType": market_type
            }
        }

        exchange = exchange_class(options)
        exchange.load_markets()

        log(f"EXCHANGE OK: {exchange_id} | markets={len(exchange.markets)}")
        return exchange
    except Exception as e:
        log(f"EXCHANGE FAILED: {exchange_id} | {type(e).__name__}: {e}")
        return None


def build_exchanges():
    spot = []
    futures = []

    for exchange_id in SPOT_IDS:
        exchange = create_exchange(exchange_id, "spot")
        if exchange:
            spot.append((exchange_id, exchange))

    for exchange_id in FUTURE_IDS:
        exchange = create_exchange(exchange_id, "swap")
        if exchange:
            futures.append((exchange_id, exchange))

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
                if not market or not market.get("spot"):
                    continue

                if market.get("quote") != "USDT":
                    continue

                base = market.get("base")
                if not base:
                    continue

                base = base.upper()
                if base in STABLES:
                    continue

                last = number(ticker.get("last"))
                quote_volume = number(ticker.get("quoteVolume"))

                if not math.isfinite(quote_volume):
                    base_volume = number(ticker.get("baseVolume"))
                    if math.isfinite(base_volume) and math.isfinite(last):
                        quote_volume = base_volume * last
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

                markets[pair]["market_volume"] += quote_volume
                markets[pair]["exchange_volumes"][exchange_id] = quote_volume

        except Exception as e:
            log(f"DISCOVERY FAILED: {exchange_id} | {type(e).__name__}: {e}")

    results = []
    for item in markets.values():
        if item["market_volume"] < 250000:
            continue

        exchanges = item["exchange_volumes"]
        results.append({
            "symbol": item["symbol"],
            "market_volume": round(item["market_volume"], 2),
            "exchange_count": len(exchanges),
            "exchanges": ",".join(exchanges.keys())
        })

    results.sort(key=lambda x: x["market_volume"], reverse=True)
    return results[:MAX_COINS]


def find_spot_exchange(spot, symbol):
    preferred = ["binance", "okx", "bybit", "kucoin"]

    for preferred_id in preferred:
        for exchange_id, exchange in spot:
            if exchange_id != preferred_id:
                continue
            if symbol in exchange.markets:
                return exchange_id, exchange

    for exchange_id, exchange in spot:
        if symbol in exchange.markets:
            return exchange_id, exchange

    return None, None


# ============================================================
# OHLCV & INDICATORS
# ============================================================

def get_ohlcv(exchange, symbol, timeframe):
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=OHLCV_LIMIT)
        if not candles or len(candles) < 60:
            return None

        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)
        return df
    except Exception:
        return None


def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()


def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ATR(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ============================================================
# ANALYSIS LOGIC
# ============================================================

def analyze_timeframe(df):
    if df is None or len(df) < 60:
        return {"score": 0.0, "trend": "UNKNOWN", "atr": np.nan, "rsi": np.nan}

    close = df["close"]
    ema20 = EMA(close, 20)
    ema50 = EMA(close, 50)
    ema100 = EMA(close, 100)

    rsi_value = RSI(close).iloc[-1]
    atr_value = ATR(df).iloc[-1]

    score = 0.0

    if close.iloc[-1] > ema20.iloc[-1]: score += 0.25
    else: score -= 0.25

    if ema20.iloc[-1] > ema50.iloc[-1]: score += 0.30
    else: score -= 0.30

    if ema50.iloc[-1] > ema100.iloc[-1]: score += 0.20
    else: score -= 0.20

    if 55 <= rsi_value <= 70: score += 0.15
    elif 70 < rsi_value <= 78: score += 0.05
    elif rsi_value > 78: score -= 0.08
    elif 30 <= rsi_value < 45: score -= 0.15
    elif rsi_value < 30: score += 0.05

    old_ema = ema20.iloc[-6]
    slope = (ema20.iloc[-1] - old_ema) / max(abs(old_ema), 1e-12)
    score += clip(slope * 8, -0.12, 0.12)

    score = clip(score, -1, 1)
    trend = "BULLISH" if score >= 0.18 else "BEARISH" if score <= -0.18 else "NEUTRAL"

    return {"score": score, "trend": trend, "atr": atr_value, "rsi": rsi_value}


def price_action(df):
    if df is None or len(df) < 5:
        return 0.0, "UNKNOWN"

    curr, prev = df.iloc[-1], df.iloc[-2]
    rng = max(curr["high"] - curr["low"], 1e-12)
    body = (curr["close"] - curr["open"]) / rng

    score = clip(body * 0.9, -1, 1)
    if curr["close"] > prev["close"]: score += 0.15
    elif curr["close"] < prev["close"]: score -= 0.15

    score = clip(score, -1, 1)
    label = "STRONG_BULLISH" if score > 0.55 else "BULLISH" if score > 0.15 else \
            "STRONG_BEARISH" if score < -0.55 else "BEARISH" if score < -0.15 else "NEUTRAL"

    return score, label


def breakout(df):
    if df is None or len(df) < 30:
        return 0.0, "UNKNOWN"

    prev_high = df["high"].iloc[-21:-1].max()
    prev_low = df["low"].iloc[-21:-1].min()
    close = df["close"].iloc[-1]

    if close > prev_high * 1.002: return 1.0, "BULLISH_BREAKOUT"
    if close < prev_low * 0.998: return -1.0, "BEARISH_BREAKOUT"
    return 0.0, "NO_BREAKOUT"


def volume_analysis(df):
    if df is None or len(df) < 25:
        return 0.0, np.nan

    avg_vol = df["volume"].iloc[-21:-1].mean()
    ratio = df["volume"].iloc[-1] / max(avg_vol, 1e-12)
    pa_score, _ = price_action(df)
    score = clip(np.tanh((ratio - 1) / 1.2) * pa_score, -1, 1)

    return score, ratio


def support_resistance(df):
    if df is None or len(df) < 50:
        return np.nan, np.nan, 0.0

    sup = float(df["low"].iloc[-51:-1].min())
    res = float(df["high"].iloc[-51:-1].max())
    close = float(df["close"].iloc[-1])
    width = max(res - sup, 1e-12)
    pos = (close - sup) / width

    score = 0.25 if pos < 0.20 else -0.25 if pos > 0.80 else 0.0
    return sup, res, score


def technical_analysis(exchange, symbol):
    timeframes = ["15m", "1h", "4h", "1d"]
    weights = {"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}

    details, dataframes = {}, {}
    for tf in timeframes:
        df = get_ohlcv(exchange, symbol, tf)
        dataframes[tf] = df
        details[tf] = analyze_timeframe(df)

    tf_score = sum(weights[tf] * details[tf]["score"] for tf in timeframes)

    base_df = dataframes.get("1h") or dataframes.get("4h") or dataframes.get("15m")

    pa_score, pa_label = price_action(base_df)
    bo_score, bo_label = breakout(base_df)
    vol_score, vol_ratio = volume_analysis(base_df)
    sup, res, sr_score = support_resistance(base_df)

    raw = (tf_score * 28 + pa_score * 7 + bo_score * 7 + vol_score * 4 + sr_score * 4)
    tech_score = clip(50 + raw, 0, 100)

    atr_1h = details["1h"]["atr"]
    if not math.isfinite(atr_1h):
        atr_1h = details["4h"]["atr"]

    return {
        "tf_score": tf_score, "technical_score": tech_score,
        "15m_trend": details["15m"]["trend"], "1h_trend": details["1h"]["trend"],
        "4h_trend": details["4h"]["trend"], "1d_trend": details["1d"]["trend"],
        "atr": atr_1h, "price_action": pa_label, "price_action_score": pa_score,
        "breakout": bo_label, "breakout_score": bo_score,
        "volume_ratio": vol_ratio, "volume_score": vol_score,
        "support": sup, "resistance": res, "sr_score": sr_score
    }


def btc_regime(spot):
    _, exchange = find_spot_exchange(spot, "BTC/USDT")
    if exchange is None: return 0.0, "UNKNOWN"
    df = get_ohlcv(exchange, "BTC/USDT", "4h")
    analysis = analyze_timeframe(df)
    return analysis["score"], analysis["trend"]


def futures_symbol(exchange, base):
    possible = [f"{base}/USDT:USDT", f"{base}/USDT"]
    for sym in possible:
        m = exchange.markets.get(sym)
        if m and (m.get("swap") or m.get("future")):
            return sym
    return None


def derivatives(futures, base):
    funding_rates, open_interests = [], []
    funding_names, oi_names = [], []

    for ex_id, exchange in futures[:3]:
        symbol = futures_symbol(exchange, base)
        if not symbol: continue

        try:
            if exchange.has.get("fetchFundingRate"):
                res = exchange.fetch_funding_rate(symbol)
                rate = number(res.get("fundingRate"))
                if math.isfinite(rate):
                    funding_rates.append(rate)
                    funding_names.append(ex_id.upper())
        except Exception: pass

        try:
            if exchange.has.get("fetchOpenInterest"):
                res = exchange.fetch_open_interest(symbol)
                oi_val = number(res.get("openInterestValue"))
                if not math.isfinite(oi_val):
                    oi_val = number(res.get("openInterestAmount"))
                if math.isfinite(oi_val):
                    open_interests.append(oi_val)
                    oi_names.append(ex_id.upper())
        except Exception: pass

    avg_funding = float(np.mean(funding_rates)) if funding_rates else 0.0
    funding_signal = "BEARISH" if avg_funding > 0.0008 else "BULLISH" if avg_funding < -0.0008 else "NEUTRAL"
    total_oi = float(np.sum(open_interests)) if open_interests else 0.0
    all_names = sorted(set(funding_names + oi_names))

    return {
        "funding": avg_funding, "funding_signal": funding_signal,
        "open_interest": total_oi, "futures_exchanges": len(all_names),
        "futures_names": ",".join(all_names)
    }


def orderbook(spot, symbol):
    imbalances, names = [], []
    preferred = ["binance", "okx", "bybit", "kucoin"]

    for pref_id in preferred:
        for ex_id, exchange in spot:
            if text(ex_id).lower() != pref_id: continue
            try:
                if symbol not in exchange.markets: continue
                book = exchange.fetch_order_book(symbol, limit=20)
                bids, asks = book.get("bids", [])[:20], book.get("asks", [])[:20]
                if not bids or not asks: continue

                bid_v = sum(number(p, 0) * number(a, 0) for p, a in bids)
                ask_v = sum(number(p, 0) * number(a, 0) for p, a in asks)
                tot = bid_v + ask_v
                if tot <= 0: continue

                imbalances.append((bid_v - ask_v) / tot)
                names.append(ex_id.upper())
            except Exception: pass

    if not imbalances:
        return {"orderbook_imbalance": 0.0, "orderbook_signal": "UNKNOWN", "orderbook_exchanges": 0, "orderbook_names": ""}

    avg = float(np.mean(imbalances))
    sig = "BULLISH" if avg > 0.10 else "BEARISH" if avg < -0.10 else "NEUTRAL"

    return {
        "orderbook_imbalance": avg, "orderbook_signal": sig,
        "orderbook_exchanges": len(names), "orderbook_names": ",".join(names)
    }


# ============================================================
# CMC & FUNDAMENTALS
# ============================================================

def cmc_get(endpoint, params=None, tries=3):
    url = CMC_BASE + endpoint
    for attempt in range(tries):
        try:
            res = session.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
            if res.status_code == 429:
                time.sleep(min(2 ** attempt, 15))
                continue
            if not res.ok:
                return None
            data = res.json()
            if data.get("status", {}).get("error_code") in (None, 0, "0"):
                return data
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def calculate_fundamental_score(row):
    score, reasons = 50.0, []
    rank = number(row.get("cmc_rank"))
    mcap = number(row.get("market_cap"))
    fdv = number(row.get("fdv"))

    if math.isfinite(rank):
        if rank <= 50: score += 8; reasons.append("TOP50")
        elif rank <= 100: score += 6; reasons.append("TOP100")
        elif rank <= 250: score += 4; reasons.append("TOP250")
        elif rank > 1000: score -= 3

    if math.isfinite(mcap) and math.isfinite(fdv) and fdv > 0:
        ratio = mcap / fdv
        row["mc_fdv_ratio"] = ratio
        if ratio >= 0.80: score += 5; reasons.append("LOW_FDV_GAP")
        elif ratio < 0.25: score -= 5; reasons.append("HIGH_FDV_RISK")

    score = clip(score, 0, 100)
    row["fundamental_score"] = round(score, 2)
    row["fundamental_rating"] = "STRONG" if score >= 70 else "NEUTRAL" if score >= 45 else "WEAK"
    row["fundamental_reason"] = "|".join(reasons) if reasons else "NEUTRAL"
    return row


def fundamentals(symbols):
    output = {}
    for sym in symbols:
        output[sym] = {
            "name": sym.split("/")[0], "cmc_rank": np.nan, "market_cap": np.nan,
            "fdv": np.nan, "fundamental_score": 50.0, "fundamental_rating": "UNKNOWN",
            "fundamental_status": "UNAVAILABLE", "fundamental_reason": "CMC_UNAVAILABLE"
        }

    base_symbols = list(dict.fromkeys([s.split("/")[0].upper() for s in symbols]))
    if not base_symbols: return output

    map_res = cmc_get("/v1/cryptocurrency/map", {"symbol": ",".join(base_symbols)})
    if not map_res or "data" not in map_res: return output

    selected = {}
    for item in map_res.get("data", []):
        sym = text(item.get("symbol")).upper()
        if sym in base_symbols:
            rank = number(item.get("rank"), 999999)
            if sym not in selected or rank < selected[sym][0]:
                selected[sym] = (rank, item)

    ids = [str(v[1]["id"]) for v in selected.values() if v[1].get("id")]
    if not ids: return output

    quote_res = cmc_get("/v1/cryptocurrency/quotes/latest", {"id": ",".join(ids), "convert": "USD"})
    quotes = quote_res.get("data", {}) if quote_res else {}

    for sym in symbols:
        base = sym.split("/")[0].upper()
        if base not in selected: continue
        coin_id = str(selected[base][1].get("id"))
        q = quotes.get(coin_id, {})
        usd = q.get("quote", {}).get("USD", {}) if q else {}

        if usd:
            output[sym].update({
                "name": text(q.get("name"), base),
                "cmc_rank": number(q.get("cmc_rank")),
                "market_cap": number(usd.get("market_cap")),
                "fdv": number(usd.get("fully_diluted_market_cap")),
                "fundamental_status": "OK"
            })
            output[sym] = calculate_fundamental_score(output[sym])

    return output


def global_market_data():
    res = cmc_get("/v1/global-metrics/quotes/latest")
    data = res.get("data", {}) if res else {}
    usd = data.get("quote", {}).get("USD", {})

    return {
        "global_market_cap": number(usd.get("total_market_cap")),
        "btc_dominance": number(data.get("btc_dominance")),
        "fear_greed": 50, "fear_greed_class": "NEUTRAL"
    }


# ============================================================
# NEWS
# ============================================================

def clean_html(val):
    return re.sub(r"<[^>]+>", " ", text(val)).replace("&amp;", "&").strip()


def load_news():
    news = []
    for feed_url in RSS_FEEDS:
        try:
            res = session.get(feed_url, timeout=8)
            res.raise_for_status()
            root = ET.fromstring(res.content)
            for item in root.findall(".//item")[:20]:
                title = clean_html(item.findtext("title", default=""))
                summary = clean_html(item.findtext("description", default=""))
                if title: news.append({"title": title, "summary": summary})
        except Exception: pass

    seen, cleaned = set(), []
    for item in news:
        k = item["title"].lower()
        if k and k not in seen:
            seen.add(k)
            cleaned.append(item)
    return cleaned[:150]


def coin_news(coin_name, symbol, news):
    base = symbol.split("/")[0].upper()
    matched = [
        item for item in news
        if re.search(r"\b" + re.escape(base) + r"\b", item["title"] + " " + item["summary"], re.I)
    ]

    if not matched:
        return {"news_score": 0.0, "news_sentiment": "NO_NEWS", "news_impact": "NONE", "news_count": 0}

    return {"news_score": 0.2, "news_sentiment": "BULLISH", "news_impact": "MEDIUM", "news_count": len(matched)}


# ============================================================
# AI CALCULATIONS & LEARNING
# ============================================================

LEARNING_FILE = "ai_learning.json"


def load_learning():
    try:
        with open(LEARNING_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "pending": [], "total": 0, "hits": 0}


def save_learning(data):
    with open(LEARNING_FILE, "w") as f:
        json.dump(data, f, indent=2)


def update_learning(data, current_prices):
    now_t = time.time()
    rem = []
    for pred in data.get("pending", []):
        age = (now_t - number(pred.get("created"), now_t)) / 3600
        sym = pred.get("symbol")

        if age >= 4 and sym in current_prices:
            entry, curr = number(pred.get("entry")), number(current_prices[sym])
            if entry > 0 and math.isfinite(curr):
                ret = (curr - entry) / entry
                hit = ret > 0.001 if pred.get("signal") == "LONG" else ret < -0.001
                data["total"] += 1
                if hit: data["hits"] += 1
                continue
        if age < 12: rem.append(pred)

    data["pending"] = rem
    return data


def learning_stats(data):
    tot, hits = data.get("total", 0), data.get("hits", 0)
    rate = hits / tot if tot else 0.5
    adj = clip((rate - 0.5) * 0.2, -0.10, 0.10) if tot >= 20 else 0.0
    return rate, adj


def create_result(base_row, tech, fund, news, ob, deriv, btc, gdata, adj):
    p = base_row["price"]
    raw = (tech["technical_score"] - 50) / 50 * 32 * (1 + adj) + news["news_score"] * 6
    long_s, short_s = clip(50 + raw, 0, 95), clip(50 - raw, 0, 95)

    sig = "LONG" if long_s >= 63 else "SHORT" if short_s >= 63 else "NO TRADE"
    atr_v = number(tech.get("atr"), p * 0.02)

    sl = p - 1.5 * atr_v if sig == "LONG" else p + 1.5 * atr_v if sig == "SHORT" else np.nan
    tp = p + 3 * atr_v if sig == "LONG" else p - 3 * atr_v if sig == "SHORT" else np.nan

    return {
        **base_row, **tech, **fund, **news, **ob, **deriv, **gdata,
        "signal": sig, "long_score": round(long_s, 2), "short_score": round(short_s, 2),
        "signal_strength": round(max(long_s, short_s), 2), "confidence": 70.0,
        "stop_loss": sl, "take_profit": tp, "learning_adjustment": adj
    }


# ============================================================
# MAIN SCANNER WORKFLOW
# ============================================================

def main():
    log("======================================")
    log("CRYPTO MASTER AI V9.2 START")
    log("======================================")

    spot, futures = build_exchanges()
    if not spot: raise RuntimeError("NO SPOT EXCHANGE AVAILABLE")

    candidates = discover_candidates(spot)
    if not candidates: raise RuntimeError("NO LIQUID USDT CANDIDATES")

    gdata = global_market_data()
    news_data = load_news()
    fundamentals_data = fundamentals([c["symbol"] for c in candidates])
    btc = btc_regime(spot)
    learning = load_learning()

    results, current_prices = [], {}

    for idx, cand in enumerate(candidates, start=1):
        sym = cand["symbol"]
        ex_id, exchange = find_spot_exchange(spot, sym)
        if not exchange: continue

        try:
            ticker = exchange.fetch_ticker(sym)
            price = number(ticker.get("last"))
            if not math.isfinite(price) or price <= 0: continue

            base = sym.split("/")[0]
            base_row = {**cand, "price": price, "analysis_exchange": ex_id.upper()}

            tech = technical_analysis(exchange, sym)
            ob = orderbook(spot, sym)
            deriv = derivatives(futures, base)
            fund = fundamentals_data.get(sym, {})
            news = coin_news(fund.get("name", base), sym, news_data)

            res = create_result(base_row, tech, fund, news, ob, deriv, btc, gdata, 0.0)
            current_prices[sym] = price
            results.append(res)
            log(f"[{idx}/{len(candidates)}] {sym} -> {res['signal']} Strength:{res['signal_strength']}")

        except Exception as e:
            log(f"COIN FAILED {sym}: {type(e).__name__}: {e}")

    learning = update_learning(learning, current_prices)
    rate, adj = learning_stats(learning)

    for r in results:
        r["learning_adjustment"] = adj

    pd.DataFrame(results).to_csv("crypto_scan_results.csv", index=False)
    save_learning(learning)
    log("SCAN COMPLETE & SAVED TO crypto_scan_results.csv")


if __name__ == "__main__":
    main()
