"""Collect real recent OKX liquidation events and label after a 4-6h observed price horizon.
This is prospective/recent coverage only; it does not create fake 8-10 year liquidation history.
"""
import json,time
from pathlib import Path
from datetime import datetime,timezone
import requests,pandas as pd
ROOT=Path("specialist_data/liquidations"); REPORT=Path("data_downloads/liquidation_download_summary.json")
INSTS={"BTC-USDT-SWAP":"BTC","ETH-USDT-SWAP":"ETH","SOL-USDT-SWAP":"SOL","XRP-USDT-SWAP":"XRP","ADA-USDT-SWAP":"ADA","DOGE-USDT-SWAP":"DOGE"}
BASE="https://www.okx.com"
def get_json(url,params):
 r=requests.get(url,params=params,timeout=20,headers={"User-Agent":"CryptoMasterAI-Research/1.0"})
 if not r.ok:
  raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
 body=r.json()
 if body.get("code") not in (None,"0"):
  raise RuntimeError(f"OKX API code={body.get('code')} msg={body.get('msg')}")
 return body
def main():
 ROOT.mkdir(parents=True,exist_ok=True); REPORT.parent.mkdir(parents=True,exist_ok=True)
 path=ROOT/"okx_liquidation_events.csv"; old=pd.read_csv(path) if path.exists() else pd.DataFrame()
 events=[]
 for inst,asset in INSTS.items():
  try:
   # OKX liquidation-orders requires instFamily or uly; instId alone returns code 50015.
   family=inst.replace("-SWAP","")
   body=get_json(BASE+"/api/v5/public/liquidation-orders",{"instType":"SWAP","instFamily":family,"state":"filled","limit":"100"})
   for group in body.get("data",[]):
    for d in group.get("details",[]):
     ts=pd.to_datetime(d.get("ts"),unit="ms",utc=True,errors="coerce")
     px=pd.to_numeric(d.get("bkPx") or d.get("px"),errors="coerce")
     qty=pd.to_numeric(d.get("sz"),errors="coerce")
     if pd.isna(ts) or pd.isna(px) or pd.isna(qty) or px<=0: continue
     events.append({"timestamp":ts,"asset":asset,"instrument":inst,"side":str(d.get("side","")),"price":float(px),"quantity":float(qty),"notional":float(px*qty),"label":None})
  except Exception as e: print(inst,"unavailable",type(e).__name__,str(e)[:180])
  time.sleep(.25)
 fresh=pd.DataFrame(events)
 if not old.empty or not fresh.empty:
  all_df=pd.concat([old,fresh],ignore_index=True) if not old.empty else fresh
  all_df["timestamp"]=pd.to_datetime(all_df["timestamp"],utc=True,errors="coerce")
  all_df=all_df.dropna(subset=["timestamp","price"]).drop_duplicates(["timestamp","instrument","side","price","quantity"],keep="last").sort_values("timestamp")
  now=pd.Timestamp.now(tz="UTC")
  for idx,row in all_df[all_df["label"].isna()].iterrows():
   age=(now-row.timestamp).total_seconds()/3600
   if 4<=age<=6:
    try:
     t=get_json(BASE+"/api/v5/market/ticker",{"instId":row.instrument})
     last=float(t["data"][0]["last"])
     all_df.loc[idx,"label"]=int(last>float(row.price))
     all_df.loc[idx,"label_price"]=last
     all_df.loc[idx,"label_timestamp"]=now.isoformat()
    except Exception as e: print("label unavailable",row.instrument,str(e)[:100])
  all_df.to_csv(path,index=False)
  labeled=all_df[all_df["label"].notna()]
  summary={"created_at":datetime.now(timezone.utc).isoformat(),"source":"OKX public liquidation-orders + ticker","raw_events":len(all_df),"labeled_rows":len(labeled),"new_events_this_run":len(fresh),"start":str(all_df.timestamp.min()),"end":str(all_df.timestamp.max()),"status":"downloaded" if len(labeled)>=100 else "collecting_more_rows","note":"Recent/prospective records only; not historical 8-10 year liquidation history."}
 else: summary={"created_at":datetime.now(timezone.utc).isoformat(),"source":"OKX public API","raw_events":0,"labeled_rows":0,"status":"unavailable","note":"No events returned by public endpoint; no synthetic rows generated."}
 REPORT.write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
