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
# CRYPTO MASTER AI V12.0 (FULL ADVANCED UNIFIED ENGINE)
# Multi-Exchange + Anti-Stop-Hunt + Dynamic RR + Full AI Learning
# ============================================================

UTC = timezone.utc
CMC_BASE = "https://pro-api.coinmarketcap.com"

MAX_COINS = 30
OHLCV_LIMIT = 220
HTTP_TIMEOUT = 15
LEARNING_FILE = "ai_learning_v2.json"

STABLES = {
    "USDT", "USDC", "FDUSD", "DAI", "USDE", "USDS",
    "USDD", "TUSD", "USDP", "PYUSD", "USDG", "RLUSD",
    "EURC", "EURT", "USTC"
}

SPOT_IDS = ["binance", "okx", "bybit", "kucoin"]
FUTURE_IDS = ["binanceusdm", "okx", "bybit", "kucoin"]

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://news.bitcoin.com/feed/",
    "https://www.theblock.co/rss.xml",
]

session = requests.Session()
cmc_api_key = os.getenv("CMC_API_KEY", "67404c0257fd4d2894796d7022e63eb9")


session.headers.update({
    "User-Agent": "Crypto-Master-AI/12.0",
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
        if value is None or value == "": return default
        val = float(value)
        return val if math.isfinite(val) else default
    except Exception:
        return default

def clip(value, low, high):
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low

def text(value, default=""):
    return str(value).strip() if value is not None else default


# ============================================================
# ADVANCED AI SELF-LEARNING ENGINE
# ============================================================

def load_advanced_learning():
    default_structure = {
        "version": 2,
        "weights": {
            "technical": 0.40,
            "orderbook": 0.20,
            "derivatives": 0.20,
            "sentiment": 0.20
        },
        "thresholds": {"long_min": 62.0, "short_max": 38.0},
        "performance": {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.50},
        "pending_evaluations": []
    }
    try:
        if os.path.exists(LEARNING_FILE):
            with open(LEARNING_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        log(f"LEARNING LOAD ERROR: {e}")
    return default_structure

def save_advanced_learning(data):
    try:
        with open(LEARNING_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log(f"LEARNING SAVE ERROR: {e}")

def update_learning_feedback(learning_data, current_prices):
    now_ts = time.time()
    remaining_pending = []
    
    total_trades = learning_data["performance"]["total_trades"]
    wins = learning_data["performance"]["wins"]
    losses = learning_data["performance"]["losses"]

    for item in learning_data.get("pending_evaluations", []):
        sym = item.get("symbol")
        created = item.get("created", now_ts)
        age_hours = (now_ts - created) / 3600.0
        
        if sym in current_prices:
            curr_p = current_prices[sym]
            entry_p = item.get("entry_price")
            sl = item.get("stop_loss")
            tp = item.get("take_profit")
            sig = item.get("signal")

            is_resolved = False
            hit_win = False

            if sig == "LONG":
                if math.isfinite(tp) and curr_p >= tp: hit_win, is_resolved = True, True
                elif math.isfinite(sl) and curr_p <= sl: hit_win, is_resolved = False, True
            elif sig == "SHORT":
                if math.isfinite(tp) and curr_p <= tp: hit_win, is_resolved = True, True
                elif math.isfinite(sl) and curr_p >= sl: hit_win, is_resolved = False, True

            if not is_resolved and age_hours >= 24:
                ret = (curr_p - entry_p) / entry_p if sig == "LONG" else (entry_p - curr_p) / entry_p
                hit_win = ret > 0.005
                is_resolved = True

            if is_resolved:
                total_trades += 1
                if hit_win:
                    wins += 1
                    learning_data["weights"]["technical"] = clip(learning_data["weights"]["technical"] + 0.01, 0.20, 0.60)
                else:
                    losses += 1
                    learning_data["weights"]["technical"] = clip(learning_data["weights"]["technical"] - 0.01, 0.20, 0.60)
                continue

        if age_hours < 24:
            remaining_pending.append(item)

    win_rate = wins / total_trades if total_trades > 0 else 0.50
    learning_data["performance"] = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4)
    }

    if win_rate < 0.45 and total_trades >= 10:
        learning_data["thresholds"]["long_min"] = clip(learning_data["thresholds"]["long_min"] + 0.5, 60.0, 75.0)
        learning_data["thresholds"]["short_max"] = clip(learning_data["thresholds"]["short_max"] - 0.5, 25.0, 40.0)
    elif win_rate > 0.60 and total_trades >= 10:
        learning_data["thresholds"]["long_min"] = clip(learning_data["thresholds"]["long_min"] - 0.2, 58.0, 70.0)
        learning_data["thresholds"]["short_max"] = clip(learning_data["thresholds"]["short_max"] + 0.2, 30.0, 42.0)

    learning_data["pending_evaluations"] = remaining_pending
    return learning_data


# ============================================================
# EXCHANGE CONNECTION & CANDIDATE DISCOVERY
# ============================================================

def create_exchange(exchange_id, market_type):
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {"defaultType": market_type}
        })
        exchange.load_markets()
        return exchange
    except Exception:
        return None

def build_exchanges():
    spot = [(eid, create_exchange(eid, "spot")) for eid in SPOT_IDS]
    futures = [(eid, create_exchange(eid, "swap")) for eid in FUTURE_IDS]
    return [x for x in spot if x[1]], [x for x in futures if x[1]]

def discover_candidates(spot):
    markets = {}
    for exchange_id, exchange in spot:
        try:
            tickers = exchange.fetch_tickers()
            for symbol, ticker in tickers.items():
                market = exchange.markets.get(symbol)
                if not market or not market.get("spot") or market.get("quote") != "USDT":
                    continue
                base = (market.get("base") or "").upper()
                if not base or base in STABLES:
                    continue

                last = number(ticker.get("last"))
                quote_volume = number(ticker.get("quoteVolume"))
                if not math.isfinite(quote_volume) or quote_volume <= 0:
                    base_volume = number(ticker.get("baseVolume"))
                    quote_volume = base_volume * last if math.isfinite(base_volume) and math.isfinite(last) else 0

                pair = f"{base}/USDT"
                if pair not in markets:
                    markets[pair] = {"symbol": pair, "market_volume": 0.0, "exchange_volumes": {}}

                markets[pair]["market_volume"] += quote_volume
                markets[pair]["exchange_volumes"][exchange_id] = quote_volume
        except Exception:
            pass

    results = []
    for item in markets.values():
        if item["market_volume"] < 50000:
            continue
        results.append({
            "symbol": item["symbol"],
            "market_volume": round(item["market_volume"], 2),
            "exchange_count": len(item["exchange_volumes"]),
            "exchanges": ",".join(item["exchange_volumes"].keys())
        })

    results.sort(key=lambda x: x["market_volume"], reverse=True)
    return results[:MAX_COINS]

def find_spot_exchange(spot, symbol):
    for pref_id in ["binance", "okx", "bybit", "kucoin"]:
        for ex_id, exchange in spot:
            if ex_id == pref_id and symbol in exchange.markets:
                return ex_id, exchange
    return None, None


# ============================================================
# TECHNICAL & TIME-FRAME ANALYSIS ENGINE
# ============================================================

def get_ohlcv(exchange, symbol, timeframe):
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=OHLCV_LIMIT)
        if not candles or len(candles) < 60: return None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna().reset_index(drop=True)
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

def technical_analysis(exchange, symbol):
    df = get_ohlcv(exchange, symbol, "1h")
    if df is None:
        return {"technical_score": 50.0, "atr": np.nan, "support": np.nan, "resistance": np.nan, "tf_score": 0.0}

    close = df["close"]
    ema20, ema50 = EMA(close, 20), EMA(close, 50)
    rsi_v = RSI(close).iloc[-1]
    atr_v = ATR(df).iloc[-1]

    score = 50.0
    if close.iloc[-1] > ema20.iloc[-1]: score += 12
    if ema20.iloc[-1] > ema50.iloc[-1]: score += 15
    if 55 <= rsi_v <= 70: score += 10
    elif rsi_v > 78: score -= 8

    sup = float(df["low"].iloc[-50:].min())
    res = float(df["high"].iloc[-50:].max())

    return {
        "technical_score": clip(score, 0, 100),
        "atr": atr_v,
        "support": sup,
        "resistance": res,
        "tf_score": (score - 50.0) / 50.0
    }


# ============================================================
# DERIVATIVES, ORDERBOOK & SENTIMENT ENGINES
# ============================================================

def orderbook(spot, symbol):
    imbalances = []
    for pref_id in ["binance", "okx", "bybit", "kucoin"]:
        for ex_id, exchange in spot:
            if ex_id != pref_id or symbol not in exchange.markets: continue
            try:
                book = exchange.fetch_order_book(symbol, limit=20)
                bids, asks = book.get("bids", [])[:20], book.get("asks", [])[:20]
                if not bids or not asks: continue

                bid_v = sum(number(p, 0) * number(a, 0) for p, a in bids)
                ask_v = sum(number(p, 0) * number(a, 0) for p, a in asks)
                tot = bid_v + ask_v
                if tot > 0: imbalances.append((bid_v - ask_v) / tot)
            except Exception: pass

    avg = float(np.mean(imbalances)) if imbalances else 0.0
    return {"orderbook_imbalance": avg, "orderbook_score": clip(50 + (avg * 50), 0, 100)}

def derivatives(futures, base):
    funding_rates = []
    for ex_id, exchange in futures[:3]:
        sym = f"{base}/USDT:USDT" if f"{base}/USDT:USDT" in exchange.markets else f"{base}/USDT"
        if sym not in exchange.markets: continue
        try:
            if exchange.has.get("fetchFundingRate"):
                res = exchange.fetch_funding_rate(sym)
                rate = number(res.get("fundingRate"))
                if math.isfinite(rate): funding_rates.append(rate)
        except Exception: pass

    avg_funding = float(np.mean(funding_rates)) if funding_rates else 0.0
    sig_score = 50.0 - (avg_funding * 10000)
    return {"funding_rate": avg_funding, "derivatives_score": clip(sig_score, 0, 100)}

def load_news():
    news = []
    for feed_url in RSS_FEEDS:
        try:
            res = session.get(feed_url, timeout=8)
            res.raise_for_status()
            root = ET.fromstring(res.content)
            for item in root.findall(".//item")[:20]:
                title = re.sub(r"<[^>]+>", " ", text(item.findtext("title", default=""))).strip()
                summary = re.sub(r"<[^>]+>", " ", text(item.findtext("description", default=""))).strip()
                if title: news.append({"title": title, "summary": summary})
        except Exception: pass
    return news[:200]

def coin_news_score(symbol, news_list):
    base = symbol.split("/")[0].upper()
    matched = [
        item for item in news_list
        if re.search(r"\b" + re.escape(base) + r"\b", item["title"] + " " + item["summary"], re.I)
    ]
    if not matched: return 50.0

    scores = []
    pos_words = ["approval", "approved", "adoption", "partnership", "launch", "bullish", "breakout"]
    neg_words = ["hack", "hacked", "exploit", "lawsuit", "ban", "delist", "bankrupt", "stolen"]

    for item in matched[:10]:
        comb = (item["title"] + " " + item["summary"]).lower()
        pos = sum(comb.count(w) for w in pos_words)
        neg = sum(comb.count(w) for w in neg_words)
        scores.append(0.8 if pos > neg else -0.8 if neg > pos else 0.0)

    avg_score = float(np.mean(scores))
    return clip(50 + (avg_score * 50), 0, 100)


# ============================================================
# LIQUIDITY ANTI-HUNT SL & DYNAMIC RR TP
# ============================================================

def find_deep_liquidity_levels(spot, symbol, price, direction):
    _, exchange = find_spot_exchange(spot, symbol)
    if not exchange: return None

    try:
        book = exchange.fetch_order_book(symbol, limit=50)
        orders = book["bids"] if direction == "LONG" else book["asks"]
        if not orders: return None

        avg_vol = np.mean([o[1] for o in orders])
        big_walls = [o[0] for o in orders if o[1] > avg_vol * 2.2]

        if big_walls:
            return min(big_walls) if direction == "LONG" else max(big_walls)
    except Exception:
        pass
    return None

def calculate_advanced_risk_management(price, signal, tech_data, spot, symbol):
    if signal not in ["LONG", "SHORT"]:
        return np.nan, np.nan, 0.0

    atr = number(tech_data.get("atr"), price * 0.02)
    sup = number(tech_data.get("support"), price * 0.95)
    res = number(tech_data.get("resistance"), price * 1.05)
    tf_score = abs(number(tech_data.get("tf_score"), 0.5))

    liquidity_wall = find_deep_liquidity_levels(spot, symbol, price, signal)

    if signal == "LONG":
        base_sl = price - (1.8 * atr)
        sr_sl = sup * 0.993
        sl = min(base_sl, sr_sl)
        if liquidity_wall and liquidity_wall < price:
            sl = min(sl, liquidity_wall * 0.995)
    else:
        base_sl = price + (1.8 * atr)
        sr_sl = res * 1.007
        sl = max(base_sl, sr_sl)
        if liquidity_wall and liquidity_wall > price:
            sl = max(sl, liquidity_wall * 1.005)

    risk_distance = abs(price - sl)
    dynamic_rr = 2.0 + (tf_score * 2.5)

    tp = price + (risk_distance * dynamic_rr) if signal == "LONG" else price - (risk_distance * dynamic_rr)
    return round(sl, 6), round(tp, 6), round(dynamic_rr, 2)


# ============================================================
# MAIN SCANNER RUNNER
# ============================================================

def main():
    log("======================================")
    log("CRYPTO MASTER AI V12.0 (FULL UNIFIED ENGINE START)")
    log("======================================")

    learning_data = load_advanced_learning()
    log(f"AI ENGINE ONLINE | Win Rate: {learning_data['performance']['win_rate']*100:.1f}% | Total Analyzed Trades: {learning_data['performance']['total_trades']}")

    spot, futures = build_exchanges()
    if not spot: raise RuntimeError("NO SPOT EXCHANGE AVAILABLE")

    candidates = discover_candidates(spot)
    news_list = load_news()
    results, current_prices = [], {}

    weights = learning_data["weights"]
    long_thresh = learning_data["thresholds"]["long_min"]
    short_thresh = learning_data["thresholds"]["short_max"]

    for idx, cand in enumerate(candidates, start=1):
        sym = cand["symbol"]
        base = sym.split("/")[0]
        ex_id, exchange = find_spot_exchange(spot, sym)
        if not exchange: continue

        try:
            ticker = exchange.fetch_ticker(sym)
            price = number(ticker.get("last"))
            if not math.isfinite(price) or price <= 0: continue
            current_prices[sym] = price

            # Module Score Calculations
            tech = technical_analysis(exchange, sym)
            ob = orderbook(spot, sym)
            deriv = derivatives(futures, base)
            news_s = coin_news_score(sym, news_list)

            # AI Multi-Factor Score Aggregation
            signal_score = (
                tech["technical_score"] * weights["technical"] +
                ob["orderbook_score"] * weights["orderbook"] +
                deriv["derivatives_score"] * weights["derivatives"] +
                news_s * weights["sentiment"]
            )

            sig = "LONG" if signal_score >= long_thresh else "SHORT" if signal_score <= short_thresh else "NO TRADE"
            sl, tp, rr_ratio = calculate_advanced_risk_management(price, sig, tech, spot, sym)

            base_row = {**cand, "price": price, "analysis_exchange": ex_id.upper()}
            
            res = {
                **base_row,
                "signal": sig,
                "signal_strength": round(signal_score, 2),
                "stop_loss": sl,
                "take_profit": tp,
                "dynamic_rr_ratio": rr_ratio,
                "funding_rate": deriv["funding_rate"],
                "orderbook_imbalance": ob["orderbook_imbalance"],
                "sentiment_score": news_s,
                "ai_winrate_factor": learning_data["performance"]["win_rate"],
                "timestamp": now().isoformat()
            }
            
            results.append(res)

            if sig in ["LONG", "SHORT"]:
                learning_data["pending_evaluations"].append({
                    "symbol": sym,
                    "signal": sig,
                    "entry_price": price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "created": time.time()
                })

            log(f"[{idx}/{len(candidates)}] {sym} | Sig: {sig} ({signal_score:.1f}) | SL: {sl} | TP: {tp} (RR 1:{rr_ratio})")

        except Exception as e:
            log(f"FAILED {sym}: {e}")

    learning_data = update_learning_feedback(learning_data, current_prices)
    save_advanced_learning(learning_data)

    df = pd.DataFrame(results)
    df.to_csv("crypto_scan_results.csv", index=False)
    
    log("======================================")
    log(f"SCAN COMPLETE | RESULTS SAVED TO crypto_scan_results.csv")
    log("======================================")


if __name__ == "__main__":
    main()
