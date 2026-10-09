
import os
import time
import csv
import json
from datetime import datetime, timezone

import ccxt

# ==============================
# CRYPTO MASTER AI DATA DOWNLOADER
# ==============================

DATA_DIR = "training_data"
EXCHANGE_ID = os.getenv("DATA_EXCHANGE", "okx")
YEARS = int(os.getenv("DATA_YEARS", "5"))

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
]

COLUMNS = [
    "timestamp", "datetime", "symbol", "timeframe",
    "open", "high", "low", "close", "volume"
]


def make_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    return exchange_class({
        "enableRateLimit": True,
        "timeout": 30000,
    })


def download_candles(exchange, symbol, timeframe):
    now_ms = exchange.milliseconds()
    start_ms = now_ms - YEARS * 365 * 24 * 60 * 60 * 1000

    safe_symbol = symbol.replace("/", "_")
    filename = f"{safe_symbol}_{timeframe}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    os.makedirs(DATA_DIR, exist_ok=True)

    existing = {}

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        ts = int(row["timestamp"])
                        existing[ts] = [
                            ts,
                            row.get("datetime", ""),
                            symbol,
                            timeframe,
                            float(row["open"]),
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                            float(row["volume"]),
                        ]
                    except (ValueError, KeyError):
                        continue
        except OSError as exc:
            print(f"Existing file read error {filename}: {exc}")

    since = max(start_ms, max(existing.keys(), default=start_ms))
    limit = 300
    downloaded = 0

    print(f"\nDownloading {symbol} {timeframe} from {EXCHANGE_ID}...")

    while since < now_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            if not candles:
                break

            latest_ts = since

            for candle in candles:
                ts, open_, high, low, close, volume = candle

                if ts < start_ms or ts > now_ms:
                    continue

                dt = datetime.fromtimestamp(
                    ts / 1000, tz=timezone.utc
                ).isoformat()

                existing[ts] = [
                    ts, dt, symbol, timeframe,
                    open_, high, low, close, volume
                ]
                latest_ts = max(latest_ts, ts)
                downloaded += 1

            if latest_ts < since or len(candles) < 2:
                break

            since = latest_ts + 1
            time.sleep(exchange.rateLimit / 1000)

        except Exception as exc:
            print(f"Download warning {symbol} {timeframe}: {exc}")
            break

    rows = [existing[k] for k in sorted(existing)]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(
        f"Saved {filename}: {len(rows)} candles "
        f"(new/updated fetched: {downloaded})"
    )

    return len(rows)


def main():
    print("====================================")
    print(" CRYPTO MASTER AI - DATA DOWNLOADER")
    print("====================================")
    print(f"Exchange: {EXCHANGE_ID}")
    print(f"History target: {YEARS} years")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {TIMEFRAMES}")

    exchange = make_exchange()
    exchange.load_markets()

    total_files = 0
    total_candles = 0
    errors = []

    for symbol in SYMBOLS:
        if symbol not in exchange.markets:
            print(f"SKIP: {symbol} not available on {EXCHANGE_ID}")
            continue

        for timeframe in TIMEFRAMES:
            try:
                count = download_candles(
                    exchange, symbol, timeframe
                )
                total_files += 1
                total_candles += count
            except Exception as exc:
                message = f"{symbol} {timeframe}: {exc}"
                errors.append(message)
                print("ERROR:", message)

    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": EXCHANGE_ID,
        "target_years": YEARS,
        "symbols_requested": len(SYMBOLS),
        "timeframes": TIMEFRAMES,
        "files_saved": total_files,
        "total_candles": total_candles,
        "errors": errors,
    }

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(
        os.path.join(DATA_DIR, "download_report.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\nDOWNLOAD SUMMARY")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
