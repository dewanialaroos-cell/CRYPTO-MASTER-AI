"""Collect genuine market snapshots from multiple public exchange APIs; no fabricated history.
Order-book snapshots are point-in-time only and must not be misrepresented as historical depth.
"""
import json, time, os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import ccxt
OUT=Path("specialist_data")
EXCHANGES=["binance","coinbase","kraken","okx","bybit","kucoin","gateio","bitget","mexc","htx","bitstamp","cryptocom"]
SYMBOLS=["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT"]
def main():
 OUT.mkdir(exist_ok=True); (OUT/"order_book").mkdir(exist_ok=True); (OUT/"fundamentals").mkdir(exist_ok=True)
 summary={"created_at":datetime.now(timezone.utc).isoformat(),"exchanges":{},"order_book_rows":0,"fundamentals_rows":0,"note":"Public snapshots are current observations, not historical order-book or liquidation records."}
 books=[]; tickers=[]
 for name in EXCHANGES:
  try:
   cls=getattr(ccxt,name,None)
   if cls is None: summary["exchanges"][name]={"status":"unsupported"}; continue
   ex=cls({"enableRateLimit":True,"timeout":15000})
   markets=ex.load_markets()
   successes=0; failures=0
   for sym in SYMBOLS:
    market=sym if sym in markets else sym.replace("/USDT","/USD")
    if market not in markets: continue
    try:
     ticker=ex.fetch_ticker(market)
     tickers.append({"timestamp":datetime.now(timezone.utc).isoformat(),"exchange":name,"symbol":market,"last":ticker.get("last"),"bid":ticker.get("bid"),"ask":ticker.get("ask"),"baseVolume":ticker.get("baseVolume"),"quoteVolume":ticker.get("quoteVolume"),"high":ticker.get("high"),"low":ticker.get("low")})
     book=ex.fetch_order_book(market,limit=20)
     bids=book.get("bids",[]); asks=book.get("asks",[])
     if bids and asks and bids[0][0] and asks[0][0]:
      bid_qty=sum(float(x[1]) for x in bids[:20]); ask_qty=sum(float(x[1]) for x in asks[:20]); mid=(bids[0][0]+asks[0][0])/2
      books.append({"timestamp":datetime.now(timezone.utc).isoformat(),"exchange":name,"symbol":market,"best_bid":bids[0][0],"best_ask":asks[0][0],"spread_bps":(asks[0][0]-bids[0][0])/mid*10000,"bid_depth_20":bid_qty,"ask_depth_20":ask_qty,"depth_imbalance":(bid_qty-ask_qty)/(bid_qty+ask_qty) if bid_qty+ask_qty else 0,"mid_price":mid,"label":None})
     successes+=1
    except Exception as e: failures+=1; print(name,market,"unavailable",type(e).__name__,str(e)[:120])
    time.sleep(max(getattr(ex,"rateLimit",1000)/1000,0.1))
   summary["exchanges"][name]={"status":"queried","markets_ok":successes,"failures":failures}
  except Exception as e: summary["exchanges"][name]={"status":"unavailable","reason":f"{type(e).__name__}: {str(e)[:200]}"}
 book_path=OUT/"order_book"/"multi_exchange_orderbook_snapshots.csv"
 old_books=pd.read_csv(book_path) if book_path.exists() else pd.DataFrame()
 new_books=pd.DataFrame(books)
 combined=pd.concat([old_books,new_books],ignore_index=True) if not old_books.empty else new_books
 if not combined.empty:
  combined["timestamp"]=pd.to_datetime(combined["timestamp"],utc=True,errors="coerce")
  combined=combined.dropna(subset=["timestamp","mid_price"]).drop_duplicates(["timestamp","exchange","symbol"],keep="last").sort_values("timestamp")
  # Only label when a real snapshot from the same exchange/market exists 4-6 hours later.
  for idx,row in combined[combined["label"].isna()].iterrows():
   later=combined[(combined.exchange==row.exchange)&(combined.symbol==row.symbol)&(combined.timestamp>=row.timestamp+pd.Timedelta(hours=4))&(combined.timestamp<=row.timestamp+pd.Timedelta(hours=6))]
   if not later.empty:
    combined.loc[idx,"label"]=int(float(later.iloc[0].mid_price)>float(row.mid_price))
  combined.to_csv(book_path,index=False)
  summary["order_book_rows"]=int(len(combined))
  summary["order_book_labeled_rows"]=int(combined["label"].notna().sum())
 ticker_path=OUT/"fundamentals"/"multi_exchange_market_snapshots.csv"
 old_tickers=pd.read_csv(ticker_path) if ticker_path.exists() else pd.DataFrame()
 new_tickers=pd.DataFrame(tickers)
 combined_tickers=pd.concat([old_tickers,new_tickers],ignore_index=True) if not old_tickers.empty else new_tickers
 if not combined_tickers.empty:
  combined_tickers=combined_tickers.drop_duplicates(["timestamp","exchange","symbol"],keep="last").sort_values("timestamp")
  combined_tickers.to_csv(ticker_path,index=False)
  summary["fundamentals_rows"]=int(len(combined_tickers))
 Path("data_downloads").mkdir(exist_ok=True); Path("data_downloads/multi_exchange_snapshot_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 print(json.dumps(summary,indent=2))
if __name__=="__main__": main()