"""Download real public Binance USD-M historical metrics and liquidation archives where available."""
import io, time, zipfile, json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd, requests
DAYS=3650
SYMBOLS=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT"]
ROOT=Path("specialist_data"); RAW=Path("data_downloads"); S=requests.Session(); S.headers["User-Agent"]="CryptoMasterAI-Research/1.0"
def archive(url):
 try:
  r=S.get(url,timeout=25)
  if r.status_code in (403,404): return None
  r.raise_for_status()
  with zipfile.ZipFile(io.BytesIO(r.content)) as z:
   files=[n for n in z.namelist() if n.lower().endswith(".csv")]
   if not files: return None
   return pd.read_csv(z.open(files[0]),header=None)
 except Exception as e:
  print("unavailable",url,type(e).__name__,str(e)[:160]); return None
def main():
 ROOT.mkdir(exist_ok=True); RAW.mkdir(exist_ok=True)
 start=(pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=DAYS)).to_period("M")
 end=pd.Timestamp.now(tz="UTC").to_period("M"); months=pd.period_range(start,end,freq="M")
 summary={"created_at":datetime.now(timezone.utc).isoformat(),"days_requested":DAYS,"domains":{},"warnings":[]}
 # Real futures liquidation snapshots, if Binance archive exposes them.
 liquidations=[]
 for sym in SYMBOLS:
  for month in months:
   ym=str(month); url="https://data.binance.vision/data/futures/um/monthly/liquidationSnapshot/"+sym+"/"+sym+"-liquidationSnapshot-"+ym+".zip"
   d=archive(url)
   if d is None or d.empty or d.shape[1]<5: continue
   d=d.iloc[:,:5].copy(); d.columns=["symbol","side","price","quantity","timestamp_ms"]
   d["price"]=pd.to_numeric(d.price,errors="coerce"); d["quantity"]=pd.to_numeric(d.quantity,errors="coerce"); d["timestamp_ms"]=pd.to_numeric(d.timestamp_ms,errors="coerce")
   d=d.dropna(subset=["price","quantity","timestamp_ms"])
   if d.empty: continue
   d["timestamp"]=pd.to_datetime(d.timestamp_ms,unit="ms",utc=True); d["notional"]=d.price*d.quantity
   d["sell_side_events"]=d.side.astype(str).str.lower().isin(["sell","s"]).astype(int); d["asset"]=sym
   liquidations.append(d[["timestamp","asset","side","price","quantity","notional","sell_side_events"]])
  time.sleep(.1)
 if liquidations:
  out=pd.concat(liquidations,ignore_index=True).sort_values("timestamp"); out.to_csv(ROOT/"liquidations"/"historical_liquidation_events.csv.gz",index=False,compression="gzip")
  summary["domains"]["liquidations"]={"rows":len(out),"files":1,"status":"downloaded","start":str(out.timestamp.min()),"end":str(out.timestamp.max())}
 else:
  summary["domains"]["liquidations"]={"rows":0,"files":0,"status":"unavailable","note":"Public archive did not expose compatible liquidation snapshots; no fake data created."}
 # Real derivatives metrics: open interest, long/short ratios and taker buy/sell volume.
 metrics=[]
 for sym in SYMBOLS:
  for month in months:
   ym=str(month); url="https://data.binance.vision/data/futures/um/monthly/metrics/"+sym+"/"+sym+"-metrics-"+ym+".zip"
   d=archive(url)
   if d is None or d.empty: continue
   d.columns=[str(c).strip().lower().replace(" ","_") for c in d.columns]
   if not any("timestamp" in c for c in d.columns): d.rename(columns={d.columns[0]:"timestamp"},inplace=True)
   tc=next((c for c in d.columns if "timestamp" in c),None)
   if tc is None: continue
   d["timestamp"]=pd.to_datetime(d[tc],utc=True,errors="coerce")
   d["asset"]=sym; metrics.append(d)
  time.sleep(.1)
 if metrics:
  out=pd.concat(metrics,ignore_index=True,sort=False).dropna(subset=["timestamp"]).sort_values("timestamp")
  out.to_csv(ROOT/"order_book"/"historical_derivatives_metrics.csv.gz",index=False,compression="gzip")
  summary["domains"]["derivatives_metrics"]={"rows":len(out),"files":1,"status":"downloaded","note":"Open interest and market positioning metrics; NOT order-book depth.","start":str(out.timestamp.min()),"end":str(out.timestamp.max())}
 else: summary["domains"]["derivatives_metrics"]={"rows":0,"files":0,"status":"unavailable"}
 (RAW/"derivatives_download_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 print(json.dumps(summary,indent=2))
if __name__=="__main__": main()