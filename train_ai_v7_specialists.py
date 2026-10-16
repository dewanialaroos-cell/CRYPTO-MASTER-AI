"""V7 multi-domain specialist trainer for optional external datasets.
Expected directories: specialist_data/<domain>/*.csv. Each CSV must contain a binary
'label' column (0/1) and numeric feature columns captured as-of signal time.
No synthetic data is generated. Missing domains are reported as NOT TRAINED.
Research-only: does not modify Scanner.py or place trades.
"""
import json, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path("specialist_data")
MODEL_DIR=Path("models")
DOMAINS={
 "fundamentals_ai":"fundamentals",
 "order_book_ai":"order_book",
 "liquidation_ai":"liquidations",
 "news_ai":"news",
 "indicator_ai":"indicators",
 "tradingview_ai":"tradingview",
}
SEED=42
def model():
 return VotingClassifier(estimators=[
  ("extra",ExtraTreesClassifier(n_estimators=80,max_depth=10,min_samples_leaf=10,class_weight="balanced",n_jobs=-1,random_state=SEED)),
  ("hist",HistGradientBoostingClassifier(max_iter=80,learning_rate=.05,l2_regularization=2,random_state=SEED)),
  ("logit",make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED)))
 ],voting="soft",n_jobs=1)
def load_domain(folder):
 frames=[]
 for p in sorted(list(folder.glob("*.csv")) + list(folder.glob("*.csv.gz"))):
  try:
   d=pd.read_csv(p)
   d.columns=[str(c).strip().lower() for c in d.columns]
   if "label" not in d or len(d)<100: continue
   d=d.replace([np.inf,-np.inf],np.nan)
   d["label"]=pd.to_numeric(d["label"],errors="coerce")
   d=d[d.label.isin([0,1])]
   d["_source_file"]=p.name
   if len(d)>=100: frames.append(d)
  except Exception as e: print(f"SKIP {p}: {e}")
 return pd.concat(frames,ignore_index=True) if frames else None
def main():
 warnings.filterwarnings("ignore")
 models={}; report={}
 for model_name,folder_name in DOMAINS.items():
  folder=ROOT/folder_name
  data=load_domain(folder) if folder.exists() else None
  if data is None:
   report[model_name]={"status":"NOT TRAINED","reason":f"No valid labeled CSV data in {folder}/","required":"At least 100 rows with label=0/1, numeric as-of-time features, and timestamp/symbol columns recommended."}
   print(f"{model_name}: NOT TRAINED — add labeled data to {folder}/")
   continue
  # Use timestamp ordering when available; otherwise stable row order is retained and explicitly noted.
  tcol=next((c for c in ("timestamp","time","datetime","date") if c in data.columns),None)
  if tcol:
   data["_sort_time"]=pd.to_datetime(data[tcol],utc=True,errors="coerce")
   data=data.sort_values("_sort_time",kind="stable")
  excluded={"label","_source_file","_sort_time","timestamp","time","datetime","date","symbol","asset","coin","target_time","future_return_4d","future_return_1d","future_price","label_price","label_timestamp","future_timestamp"}
  cols=[c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c]) and data[c].notna().any()]
  data=data.dropna(subset=["label"])
  if len(cols)<2 or data.label.nunique()<2:
   report[model_name]={"status":"NOT TRAINED","reason":"Need at least two numeric features and both label classes."}; continue
  X=data[cols].replace([np.inf,-np.inf],np.nan); y=data.label.astype(int)
  n=len(data); a=int(n*.60); b=int(n*.80)
  # Purge a small boundary window to reduce near-boundary target overlap.
  tr_end=max(0,a-3); va_start=min(n,a+3); va_end=max(va_start,b-3); te_start=min(n,b+3)
  Xtr,ytr=X.iloc[:tr_end],y.iloc[:tr_end]
  Xva,yva=X.iloc[va_start:va_end],y.iloc[va_start:va_end]
  Xte,yte=X.iloc[te_start:],y.iloc[te_start:]
  if min(len(Xtr),len(Xva),len(Xte))<30 or any(z.nunique()<2 for z in (ytr,yva,yte)):
   report[model_name]={"status":"NOT TRAINED","reason":"Chronological train/validation/test split too small or missing a class.","rows":n}; continue
  clf=model(); clf.fit(Xtr,ytr)
  prob=clf.predict_proba(Xte)[:,list(clf.classes_).index(1)]; pred=(prob>=.5).astype(int)
  try: auc=round(float(roc_auc_score(yte,prob)),5)
  except Exception: auc=None
  metrics={"rows":n,"train_rows":len(Xtr),"validation_rows":len(Xva),"test_rows":len(Xte),
   "feature_count":len(cols),"accuracy":round(float(accuracy_score(yte,pred)),5),
   "balanced_accuracy":round(float(balanced_accuracy_score(yte,pred)),5),
   "precision":round(float(precision_score(yte,pred,zero_division=0)),5),
   "recall":round(float(recall_score(yte,pred,zero_division=0)),5),"roc_auc":auc}
  models[model_name]={"model":clf,"features":cols,"source_folder":str(folder),"trained_at":datetime.now(timezone.utc).isoformat()}
  report[model_name]={"status":"TRAINED","metrics":metrics,"features":cols,"time_ordered":bool(tcol)}
  print(f"{model_name}: TRAINED | test rows={len(Xte)} | ROC-AUC={auc}")
 MODEL_DIR.mkdir(exist_ok=True)
 bundle={"version":"V7","created_at":datetime.now(timezone.utc).isoformat(),"models":models,
  "domain_names":DOMAINS,"research_only":True}
 joblib.dump(bundle,MODEL_DIR/"crypto_master_ai_v7_specialists.joblib")
 payload={"version":"V7","created_at":bundle["created_at"],"trained_count":len(models),
  "domain_count":len(DOMAINS),"domains":report,"scikit_learn":sklearn.__version__,
  "warnings":["No synthetic training examples are created.","Fundamental, order-book, liquidation, news and TradingView domains need real labeled historical data.",
  "TradingView AI uses exported/webhook-derived features; this code does not claim an official TradingView API connection.",
  "News features must be timestamped to avoid using articles published after the signal.",
  "Liquidation and order-book data must be historical snapshots aligned to each signal time.",
  "Metrics are classification metrics, not net trading returns. No live trading; Scanner.py remains unchanged."]}
 (MODEL_DIR/"crypto_master_ai_v7_report.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
 print("\\nV7 DOMAIN SUMMARY\\n"+json.dumps({"trained_count":len(models),"domains":report,
  "model":"models/crypto_master_ai_v7_specialists.joblib","report":"models/crypto_master_ai_v7_report.json"},indent=2))
if __name__=="__main__": main()
