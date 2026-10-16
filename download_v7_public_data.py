"""Download real public crypto data and build timestamped V7 specialist datasets.

Sources:
- Binance public OHLCV via ccxt (historical hourly candles)
- CoinGecko public market_chart endpoint (historical price, market cap, volume)
- Crypto RSS headlines (recent headlines only; labels only when a 4h future candle exists)

This script does NOT synthesize rows. Order-book and liquidation history are NOT fabricated:
those domains require real historical snapshots from a suitable provider.
"""
from __future__ import annotations
import time, json, re, html, io, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import ccxt
import xml.etree.ElementTree as ET

ROOT = Path("specialist_data")
RAW = Path("data_downloads")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT"]
COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin"
}
DAYS = 3650
HORIZON_HOURS = 4
TIMEOUT = 25
HEADERS = {"User-Agent": "CryptoMasterAI-ResearchDataBot/1.0"}
POSITIVE = {"surge","surges","surged","rally","rallies","bull","bullish","gain","gains","growth","adoption","approve","approved","approval","breakout","record","high","inflow","inflows","rise","rises","rising","upgrade","partnership"}
NEGATIVE = {"crash","crashes","crashed","bear","bearish","drop","drops","fall","falls","falling","hack","hacked","exploit","lawsuit","ban","banned","selloff","outflow","outflows","liquidation","liquidations","decline","risk","risks","warning","investigation"}

def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

def fetch_ohlcv(exchange, symbol, since_ms):
    """Read Binance's public monthly archive files; does not call the restricted exchange API."""
    rows = []
    session = get_session()
    market_symbol = symbol.replace("/", "")
    start = pd.Timestamp(since_ms, unit="ms", tz="UTC").to_period("M")
    end = pd.Timestamp.now(tz="UTC").to_period("M")
    months = pd.period_range(start=start, end=end, freq="M")
    for month in months:
        ym = str(month)
        url = f"https://data.binance.vision/data/spot/monthly/klines/{market_symbol}/1h/{market_symbol}-1h-{ym}.zip"
        try:
            response = session.get(url, timeout=TIMEOUT)
            if response.status_code in (403, 404):
                print(f"  archive unavailable for {market_symbol} {ym} (HTTP {response.status_code})")
                continue
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                part = pd.read_csv(zf.open(names[0]), header=None)
            if part.empty:
                continue
            # Binance monthly kline archive columns.
            part = part.iloc[:, :6]
            part.columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
            rows.extend(part.values.tolist())
            print(f"  archive {market_symbol} {ym}: {len(part)} candles")
        except Exception as e:
            print(f"  archive error {market_symbol} {ym}: {type(e).__name__}: {e}")
        time.sleep(0.08)
    df = pd.DataFrame(rows, columns=["timestamp_ms","open","high","low","close","volume"])
    if df.empty:
        return df
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp_ms"]).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
    df["timestamp"] = pd.to_datetime(df.timestamp_ms, unit="ms", utc=True)
    df = df[df.timestamp >= pd.Timestamp(since_ms, unit="ms", tz="UTC")]
    cutoff = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.timestamp < cutoff].copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(df):
    x = df.copy()
    close, high, low, vol = x.close, x.high, x.low, x.volume
    for n in [1,2,3,4,6,12,24,48,72,168]:
        x[f"return_{n}"] = close.pct_change(n)
    for n in [5,9,12,20,26,50,100,200]:
        ema = close.ewm(span=n, adjust=False, min_periods=min(n, 20)).mean()
        x[f"ema_distance_{n}"] = close / ema - 1
    ema12=close.ewm(span=12,adjust=False).mean(); ema26=close.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False).mean()
    x["macd_pct"]=macd/close; x["macd_signal_pct"]=sig/close; x["macd_hist_pct"]=(macd-sig)/close
    for n in [7,14,21]:
        prev=close.shift(1)
        tr=pd.concat([(high-low),(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
        x[f"atr_{n}"]=tr.rolling(n).mean()/close
        x[f"volatility_{n}"]=x["return_1"].rolling(n).std()
        x[f"rsi_{n}"]=rsi(close,n)
        x[f"roc_{n}"]=close.pct_change(n)
    x["adx_proxy_14"]=(high.diff().abs().rolling(14).mean()+low.diff().abs().rolling(14).mean())/close
    x["trend_direction"]=np.sign(x["ema_distance_20"].fillna(0))
    x["trend_strength"]=x["ema_distance_50"].abs()
    x["volume_ratio_20"]=vol/(vol.rolling(20).mean()+1e-12)
    x["volume_ratio_50"]=vol/(vol.rolling(50).mean()+1e-12)
    x["volume_change_1"]=vol.pct_change().replace([np.inf,-np.inf],np.nan)
    x["volume_zscore_20"]=(vol-vol.rolling(20).mean())/(vol.rolling(20).std()+1e-12)
    x["obv_slope_20"]=(np.sign(close.diff()).fillna(0)*vol).rolling(20).sum()/(vol.rolling(20).sum()+1e-12)
    typical=(high+low+close)/3
    x["vwap_distance_20"]=close/((typical*vol).rolling(20).sum()/(vol.rolling(20).sum()+1e-12))-1
    for n in [10,20,50]:
        x[f"support_distance_{n}"]=close/(low.rolling(n).min()+1e-12)-1
        x[f"resistance_distance_{n}"]=high.rolling(n).max()/(close+1e-12)-1
        mid=close.rolling(n).mean(); sd=close.rolling(n).std()
        x[f"bb_width_{n}"]=4*sd/(mid+1e-12)
        x[f"bb_position_{n}"]=(close-(mid-2*sd))/(4*sd+1e-12)
    candle_range=(high-low).replace(0,np.nan)
    x["body_pct"]=(close-x.open)/x.open
    x["body_range_ratio"]=(close-x.open)/candle_range
    x["upper_wick_pct"]=(high-pd.concat([x.open,close],axis=1).max(axis=1))/close
    x["lower_wick_pct"]=(pd.concat([x.open,close],axis=1).min(axis=1)-low)/close
    x["close_position"]=(close-low)/candle_range
    x["inside_bar"]=((high<high.shift(1))&(low>low.shift(1))).astype(int)
    x["outside_bar"]=((high>high.shift(1))&(low<low.shift(1))).astype(int)
    x["hour_utc"]=x.timestamp.dt.hour
    x["weekday_utc"]=x.timestamp.dt.dayofweek
    x["hour_sin"]=np.sin(2*np.pi*x.hour_utc/24); x["hour_cos"]=np.cos(2*np.pi*x.hour_utc/24)
    x["weekday_sin"]=np.sin(2*np.pi*x.weekday_utc/7); x["weekday_cos"]=np.cos(2*np.pi*x.weekday_utc/7)
    x["future_return_4h"]=x.close.shift(-HORIZON_HOURS)/x.close-1
    # Binary label is defined only after the future close exists; no future feature is retained.
    x["label"]=(x.future_return_4h>0).astype("Int64")
    x.loc[x.future_return_4h.isna(),"label"]=pd.NA
    return x.replace([np.inf,-np.inf],np.nan)

def specialist_export(feat, folder, symbol):
    folder.mkdir(parents=True,exist_ok=True)
    # Export numeric features available at signal time, with a future-derived label only.
    excluded={"timestamp_ms","open","high","low","close","volume","future_return_4h","label"}
    cols=[c for c in feat.columns if c not in excluded and pd.api.types.is_numeric_dtype(feat[c])]
    out=feat[["timestamp","label"]+cols].dropna(subset=["label"]).copy()
    out["label"]=out.label.astype(int)
    out["symbol"]=symbol.replace("/","_")
    # Save per-symbol file so source identity and chronology remain auditable.
    path=folder/f"{symbol.replace('/','_')}_1h.csv"
    out.to_csv(path,index=False)
    return len(out),str(path)

def download_coingecko(session, coin_id):
    url=f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    response=session.get(url,params={"vs_currency":"usd","days":str(DAYS),"interval":"daily"},timeout=TIMEOUT)
    response.raise_for_status()
    data=response.json()
    prices=pd.DataFrame(data.get("prices",[]),columns=["timestamp_ms","cg_price"])
    caps=pd.DataFrame(data.get("market_caps",[]),columns=["timestamp_ms","market_cap"])
    volumes=pd.DataFrame(data.get("total_volumes",[]),columns=["timestamp_ms","market_volume"])
    if prices.empty: return pd.DataFrame()
    d=prices.merge(caps,on="timestamp_ms",how="left").merge(volumes,on="timestamp_ms",how="left")
    d["timestamp"]=pd.to_datetime(d.timestamp_ms,unit="ms",utc=True)
    d=d.sort_values("timestamp")
    d["price_return_1d"]=d.cg_price.pct_change()
    d["market_cap_return_1d"]=d.market_cap.pct_change()
    d["volume_change_1d"]=d.market_volume.pct_change()
    d["volume_to_market_cap"]=d.market_volume/(d.market_cap+1e-12)
    d["price_volatility_7d"]=d.price_return_1d.rolling(7).std()
    d["future_return_4d"]=d.cg_price.shift(-4)/d.cg_price-1
    d["label"]=(d.future_return_4d>0).astype("Int64")
    d.loc[d.future_return_4d.isna(),"label"]=pd.NA
    return d.replace([np.inf,-np.inf],np.nan)

def download_news(session):
    feeds=[
      "https://www.coindesk.com/arc/outboundfeeds/rss/",
      "https://cointelegraph.com/rss",
    ]
    rows=[]
    for url in feeds:
        try:
            res=session.get(url,timeout=TIMEOUT); res.raise_for_status()
            root=ET.fromstring(res.content)
            for item in root.findall(".//item"):
                title=html.unescape((item.findtext("title") or "").strip())
                pub=(item.findtext("pubDate") or "").strip()
                if not title or not pub: continue
                try: ts=pd.to_datetime(pub,utc=True,errors="raise")
                except Exception: continue
                words=set(re.findall(r"[a-z]+",title.lower()))
                rows.append({"timestamp":ts,"positive_word_count":sum(w in POSITIVE for w in words),
                  "negative_word_count":sum(w in NEGATIVE for w in words),
                  "headline_length":len(words),"source_domain":url.split("/")[2]})
        except Exception as e:
            print(f"RSS unavailable {url}: {e}")
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows).drop_duplicates(["timestamp","headline_length"]).sort_values("timestamp")
    return d

def main():
    RAW.mkdir(exist_ok=True)
    session=get_session()
    exchange=ccxt.binance({"enableRateLimit":True,"timeout":30000,"options":{"defaultType":"spot"}})
    since=int((pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=DAYS)).timestamp()*1000)
    candle_frames={}
    summary={"created_at":datetime.now(timezone.utc).isoformat(),"source":"Binance public OHLCV + CoinGecko public market chart + RSS headlines",
      "days_requested":DAYS,"requested_years":round(DAYS/365,2),"history_note":"OHLCV history goes back as far as each asset was listed on the exchange; not every coin has a full 10 years. Public CoinGecko endpoint may restrict long-range history on its free tier.","symbols":SYMBOLS,"domains":{},"warnings":[]}
    all_ind=[]; all_tv=[]
    for symbol in SYMBOLS:
        try:
            print(f"Downloading {symbol} 1h candles...")
            df=fetch_ohlcv(exchange,symbol,since)
            if len(df)<300:
                summary["warnings"].append(f"{symbol}: insufficient candle history ({len(df)} rows)")
                continue
            df.to_csv(RAW/f"{symbol.replace('/','_')}_1h.csv",index=False)
            feat=build_features(df)
            n1,p1=specialist_export(feat,ROOT/"indicators",symbol)
            # TradingView-style features are calculated from real exchange candles, not claimed to be TV-exported data.
            tvcols=["timestamp","label","symbol","return_1","return_4","return_24","ema_distance_9","ema_distance_20","ema_distance_50","macd_pct","macd_signal_pct","macd_hist_pct","rsi_7","rsi_14","roc_14","atr_14","bb_width_20","bb_position_20","volume_ratio_20","volume_zscore_20","support_distance_20","resistance_distance_20","body_pct","upper_wick_pct","lower_wick_pct","hour_sin","hour_cos","weekday_sin","weekday_cos"]
            tv=feat[[c for c in tvcols if c in feat.columns]].dropna(subset=["label"]).copy()
            tv["symbol"]=symbol.replace("/","_"); tv["label"]=tv.label.astype(int)
            (ROOT/"tradingview").mkdir(parents=True,exist_ok=True)
            tv.to_csv(ROOT/"tradingview"/f"{symbol.replace('/','_')}_1h.csv.gz",index=False,compression="gzip")
            all_ind.append(n1); all_tv.append(len(tv))
            candle_frames[symbol.split("/")[0]]=df
            print(f"  saved {n1} indicator rows and {len(tv)} TradingView-style rows")
        except Exception as e:
            summary["warnings"].append(f"OHLCV {symbol}: {type(e).__name__}: {e}")
            print(f"FAILED {symbol}: {e}")
    summary["domains"]["indicator_ai"]={"rows":int(sum(all_ind)),"files":len(all_ind),"status":"downloaded" if all_ind else "no data"}
    summary["domains"]["tradingview_ai"]={"rows":int(sum(all_tv)),"files":len(all_tv),"status":"downloaded" if all_tv else "no data","note":"TradingView-style indicators derived from exchange candles; not an official TradingView export."}
    # Download historical public market cap/volume/price data. This is market fundamentals proxy data, not company financial statements.
    fund_counts=[]
    for sym,cid in COINGECKO_IDS.items():
        try:
            print(f"Downloading CoinGecko historical market data for {sym}...")
            d=download_coingecko(session,cid)
            if d.empty: continue
            keep=["timestamp","label","price_return_1d","market_cap_return_1d","volume_change_1d","volume_to_market_cap","price_volatility_7d"]
            out=d[keep].dropna(subset=["label"]).copy(); out["label"]=out.label.astype(int); out["symbol"]=sym+"_USDT"
            (ROOT/"fundamentals").mkdir(parents=True,exist_ok=True)
            out.to_csv(ROOT/"fundamentals"/f"{sym}_market_history.csv.gz",index=False,compression="gzip")
            fund_counts.append(len(out)); time.sleep(2)
        except Exception as e:
            summary["warnings"].append(f"CoinGecko {sym}: {type(e).__name__}: {e}")
            print(f"CoinGecko failed {sym}: {e}")
    summary["domains"]["fundamentals_ai"]={"rows":int(sum(fund_counts)),"files":len(fund_counts),"status":"downloaded" if fund_counts else "no data","note":"Historical crypto market cap/volume/price proxies, not corporate financial statements."}
    # RSS gives recent headline-only data. Label only items at least 4h old and where a future candle exists.
    try:
        news=download_news(session)
        news_rows=[]
        if not news.empty:
            for _,row in news.iterrows():
                # Match BTC candle close nearest to or after publication time; require 4h future price.
                btc=candle_frames.get("BTC")
                if btc is None: continue
                ix=btc.timestamp.searchsorted(row.timestamp)
                if ix<0 or ix+HORIZON_HOURS>=len(btc): continue
                if btc.timestamp.iloc[ix] < row.timestamp: ix+=1
                if ix+HORIZON_HOURS>=len(btc): continue
                # The article must predate the labeling close; no future news is used as a feature.
                ret=float(btc.close.iloc[ix+HORIZON_HOURS]/btc.close.iloc[ix]-1)
                news_rows.append({"timestamp":row.timestamp,"positive_word_count":int(row.positive_word_count),
                  "negative_word_count":int(row.negative_word_count),"headline_length":int(row.headline_length),
                  "label":int(ret>0),"symbol":"BTC_USDT"})
        if news_rows:
            (ROOT/"news").mkdir(parents=True,exist_ok=True)
            pd.DataFrame(news_rows).drop_duplicates("timestamp").to_csv(ROOT/"news"/"rss_headlines_labeled.csv.gz",index=False,compression="gzip")
        summary["domains"]["news_ai"]={"rows":len(news_rows),"files":1 if news_rows else 0,"status":"downloaded" if news_rows else "insufficient recent labeled RSS data","note":"Recent RSS only; this is not a historical news archive and keyword features are a weak baseline."}
    except Exception as e:
        summary["warnings"].append(f"RSS news: {type(e).__name__}: {e}")
        summary["domains"]["news_ai"]={"rows":0,"status":"download failed"}
    summary["domains"]["order_book_ai"]={"rows":0,"status":"NOT DOWNLOADED","note":"Historical snapshots require a real archive/provider. No historical depth is fabricated."}
    summary["domains"]["liquidation_ai"]={"rows":0,"status":"NOT DOWNLOADED","note":"Historical liquidation events require a real archive/provider. No events are fabricated."}
    (ROOT/"order_book").mkdir(parents=True,exist_ok=True)
    (ROOT/"liquidations").mkdir(parents=True,exist_ok=True)
    (RAW/"download_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nDOWNLOAD SUMMARY\n"+json.dumps(summary,indent=2))
if __name__=="__main__":
    main()
