"""Download available historical CoinMarketCap quote data for Crypto Master AI.
API key must be stored as GitHub Actions secret CMC_API_KEY. Never commit API keys.
Historical endpoint/coverage depends on the user's CMC plan; this script records unavailable symbols honestly.
"""
import os, json, time
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

OUT = Path("specialist_data/fundamentals")
REPORT = Path("data_downloads/cmc_download_summary.json")
SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE"]
BASE = "https://pro-api.coinmarketcap.com/v1"
def main():
    key = os.getenv("CMC_API_KEY", "").strip()
    if not key:
        raise SystemExit("CMC_API_KEY missing. Add it in GitHub Settings > Secrets and variables > Actions.")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"X-CMC_PRO_API_KEY": key, "Accept": "application/json"})
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "source":"CoinMarketCap API", "domains":{}, "symbols":{}, "note":"Available history depends on CMC plan and endpoint permissions. No synthetic data generated."}
    all_frames = []
    # First resolve IDs and validate the key using the map endpoint.
    try:
        r = s.get(BASE + "/cryptocurrency/map", params={"symbol": ",".join(SYMBOLS), "limit":5000}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get("status", {}).get("error_code", 0) != 0:
            raise RuntimeError(payload["status"].get("error_message", "CMC map API error"))
        map_rows = payload.get("data", [])
        by_symbol = {}
        for row in map_rows:
            sym = row.get("symbol")
            if sym in SYMBOLS and sym not in by_symbol:
                by_symbol[sym] = row
    except Exception as e:
        raise SystemExit(f"CMC API validation failed: {type(e).__name__}: {e}")
    for sym in SYMBOLS:
        coin = by_symbol.get(sym)
        if not coin:
            report["symbols"][sym] = {"status":"unavailable","rows":0,"reason":"CoinMarketCap map did not return this symbol"}
            continue
        params = {"id": coin["id"], "interval":"daily", "count":3650, "convert":"USD"}
        try:
            r = s.get(BASE + "/cryptocurrency/quotes/historical", params=params, timeout=45)
            if r.status_code in (401,403,429):
                report["symbols"][sym] = {"status":"unavailable","rows":0,"http_status":r.status_code,"reason":r.text[:500]}
                print(f"{sym}: CMC access unavailable HTTP {r.status_code}: {r.text[:180]}")
                time.sleep(1)
                continue
            r.raise_for_status()
            body = r.json()
            status = body.get("status", {})
            if status.get("error_code", 0) != 0:
                report["symbols"][sym] = {"status":"unavailable","rows":0,"reason":status.get("error_message","CMC API error")}
                print(f"{sym}: {status.get('error_message')}")
                time.sleep(1)
                continue
            data = body.get("data", [])
            # CMC response is normally a list of quote snapshots; tolerate wrapped response.
            if isinstance(data, dict):
                data = data.get("quotes", [])
            rows = []
            for item in data:
                quote = item.get("quote", {}).get("USD", {})
                ts = item.get("timestamp") or item.get("time_open")
                if not ts: continue
                rows.append({
                    "timestamp": pd.to_datetime(ts, utc=True, errors="coerce"),
                    "price": quote.get("price"),
                    "volume_24h": quote.get("volume_24h"),
                    "market_cap": quote.get("market_cap"),
                    "circulating_supply": item.get("circulating_supply"),
                    "total_supply": item.get("total_supply"),
                    "max_supply": item.get("max_supply"),
                    "symbol": sym,
                })
            df = pd.DataFrame(rows).dropna(subset=["timestamp","price"]).sort_values("timestamp")
            if len(df) >= 10:
                for col in ["price","volume_24h","market_cap","circulating_supply","total_supply","max_supply"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["price_return_1d"] = df.price.pct_change()
                df["market_cap_return_1d"] = df.market_cap.pct_change()
                df["volume_to_market_cap"] = df.volume_24h / df.market_cap.replace(0, pd.NA)
                df["future_return_4d"] = df.price.shift(-4) / df.price - 1
                df["label"] = (df.future_return_4d > 0).astype("Int64")
                df.loc[df.future_return_4d.isna(), "label"] = pd.NA
                df = df.replace([float("inf"), float("-inf")], pd.NA)
                path = OUT / f"{sym}_cmc_historical.csv"
                df.to_csv(path, index=False)
                all_frames.append(df)
                report["symbols"][sym] = {"status":"downloaded","rows":len(df),"start":str(df.timestamp.min()),"end":str(df.timestamp.max()),"file":str(path)}
            else:
                report["symbols"][sym] = {"status":"insufficient_history","rows":len(df),"reason":"CMC returned fewer than 10 usable daily rows"}
            print(f"{sym}: {report['symbols'][sym]}")
        except Exception as e:
            report["symbols"][sym] = {"status":"unavailable","rows":0,"reason":f"{type(e).__name__}: {e}"}
            print(f"{sym}: error {type(e).__name__}: {e}")
        time.sleep(1.2)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True).sort_values(["symbol","timestamp"])
        combined.to_csv(OUT / "cmc_fundamentals_combined.csv", index=False)
        report["domains"]["fundamentals"] = {"status":"downloaded","rows":len(combined),"files":len(all_frames)+1}
    else:
        report["domains"]["fundamentals"] = {"status":"no_usable_data","rows":0,"files":0}
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
if __name__ == "__main__":
    main()
