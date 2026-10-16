"""Download timestamped crypto news from GDELT and label it with later CMC returns.
Only articles actually returned by the source are used; sentiment is a transparent lexical baseline.
"""
import json,re,time
from pathlib import Path
from datetime import datetime,timezone
import requests
import pandas as pd

ROOT=Path("specialist_data/news"); REPORT=Path("data_downloads/news_download_summary.json")
CMC=Path("specialist_data/fundamentals")
COINS={"BTC":"bitcoin OR BTC","ETH":"ethereum OR ETH","SOL":"solana OR SOL","XRP":"ripple OR XRP","ADA":"cardano OR ADA","DOGE":"dogecoin OR DOGE"}
POS={"surge","rally","bullish","gain","growth","adoption","approve","approved","approval","breakout","record","high","inflow","rise","rising","upgrade","partnership","rebound","outperform"}
NEG={"crash","bearish","drop","fall","hack","exploit","lawsuit","ban","banned","selloff","outflow","liquidation","decline","risk","warning","investigation","fraud","collapse","plunge"}
def main():
 ROOT.mkdir(parents=True,exist_ok=True); REPORT.parent.mkdir(parents=True,exist_ok=True)
 s=requests.Session(); s.headers["User-Agent"]="CryptoMasterAI-Research/1.0 (historical news analysis)"
 report={"created_at":datetime.now(timezone.utc).isoformat(),"source":"GDELT DOC API","symbols":{},"note":"Coverage and API limits vary; lexical sentiment is a baseline, not a trained language model."}
 end=pd.Timestamp.now(tz="UTC").floor("D"); start=end-pd.Timedelta(days=365)
 months=pd.period_range(start=start.to_period("M"),end=end.to_period("M"),freq="M")
 for coin,query in COINS.items():
  path=CMC/f"{coin}_cmc_historical.csv"
  if not path.exists():
   report["symbols"][coin]={"status":"no_price_labels","rows":0}; continue
  prices=pd.read_csv(path); prices["timestamp"]=pd.to_datetime(prices.timestamp,utc=True,errors="coerce")
  prices=prices.dropna(subset=["timestamp","price"]).sort_values("timestamp")
  rows=[]
  for month in months:
   ms=max(month.start_time.tz_localize("UTC"),start); me=min((month.end_time+pd.Timedelta(seconds=1)).tz_localize("UTC"),end)
   params={"query":f'({query}) crypto','mode':'ArtList','format':'json','maxrecords':250,'sort':'HybridRel',
    'startdatetime':ms.strftime("%Y%m%d%H%M%S"),'enddatetime':me.strftime("%Y%m%d%H%M%S")}
   try:
    payload=None
    for attempt in range(5):
     r=s.get("https://api.gdeltproject.org/api/v2/doc/doc",params=params,timeout=30)
     if r.status_code==429 or r.status_code>=500:
      wait=min(15*(attempt+1),60)
      print(f"{coin} {month}: HTTP {r.status_code}; retry {attempt+1}/5 after {wait}s")
      time.sleep(wait)
      continue
     r.raise_for_status()
     payload=r.json()
     break
    if payload is None:
     print(f"{coin} {month}: skipped after repeated rate limits/server errors")
     time.sleep(8)
     continue
    for a in payload.get("articles",[]):
     title=(a.get("title") or "").strip()
     raw=a.get("seendate") or ""
     ts=pd.to_datetime(raw,format="%Y%m%dT%H%M%SZ",utc=True,errors="coerce")
     if pd.isna(ts) or not title: continue
     words=set(re.findall(r"[a-z]+",title.lower()))
     rows.append({"timestamp":ts,"positive_word_count":sum(w in POS for w in words),
       "negative_word_count":sum(w in NEG for w in words),"headline_word_count":len(words),
       "title_char_count":len(title),"source_domain":str(a.get("domain") or ""),
       "source_country":str(a.get("sourcecountry") or ""),"symbol":coin})
   except Exception as e: print(f"{coin} {month}: {type(e).__name__}: {str(e)[:160]}")
   time.sleep(3.5)
  d=pd.DataFrame(rows)
  if d.empty:
   report["symbols"][coin]={"status":"no_articles","rows":0}; continue
  d=d.drop_duplicates(["timestamp","source_domain","headline_word_count"]).sort_values("timestamp")
  # Label each article using next available daily CMC quote at or after 4 days from publication.
  future=prices[["timestamp","price"]].copy().rename(columns={"timestamp":"future_timestamp","price":"future_price"})
  future["future_timestamp"]-=pd.Timedelta(days=4)
  base=prices[["timestamp","price"]].rename(columns={"price":"base_price"}).sort_values("timestamp")
  d=pd.merge_asof(d.sort_values("timestamp"),base,on="timestamp",direction="backward")
  d=pd.merge_asof(d.sort_values("timestamp"),future.sort_values("future_timestamp"),left_on="timestamp",right_on="future_timestamp",direction="forward")
  d=d.dropna(subset=["base_price","future_price"])
  d["future_return_4d"]=d.future_price/d.base_price-1
  d["label"]=(d.future_return_4d>0).astype(int)
  d["sentiment_net"]=d.positive_word_count-d.negative_word_count
  out=ROOT/f"{coin}_gdelt_news.csv"; d.to_csv(out,index=False)
  report["symbols"][coin]={"status":"downloaded","rows":len(d),"start":str(d.timestamp.min()),"end":str(d.timestamp.max()),"file":str(out)}
  print(coin,report["symbols"][coin])
 report["total_labeled_rows"]=sum(x.get("rows",0) for x in report["symbols"].values())
 report["files"]=sum(1 for x in report["symbols"].values() if x.get("rows",0)>0)
 REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
