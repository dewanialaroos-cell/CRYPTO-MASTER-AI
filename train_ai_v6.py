"""Crypto Master AI V6: domain specialists + Head AI.
Research-only: independent of Scanner.py; no live orders or alerts.
Uses chronological train/validation/test partitions and a conservative ensemble.
"""
import json, warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_ai_v3 import DATA_DIR, MODEL_DIR, TIMEFRAME_CODES, load_candles, make_features

OUT_MODEL = MODEL_DIR / "crypto_master_ai_v6.joblib"
OUT_REPORT = MODEL_DIR / "crypto_master_ai_v6_report.json"
MAX_ROWS_PER_FILE = 12000
SEED = 42
DOMAINS = {
 "trend": ["return_", "ema_distance_", "macd_", "adx_", "trend_", "regime_trend"],
 "momentum": ["rsi_", "stoch_", "roc_", "cci_", "williams_", "momentum_"],
 "volatility": ["atr_", "volatility_", "bb_", "range_", "true_range_", "realized_volatility_", "regime_volatility"],
 "volume_flow": ["volume_", "obv_", "mfi_", "cmf_", "vwap_", "accumulation_"],
 "candlestick": ["body_", "upper_wick_", "lower_wick_", "close_position", "gap_", "inside_bar", "outside_bar", "bullish_engulf", "bearish_engulf", "doji_", "pin_bar"],
 "support_resistance": ["high_distance_", "low_distance_", "support_", "resistance_", "breakout_", "drawdown_"],
 "market_regime": ["regime_", "timeframe_code", "hour_", "weekday_", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"],
}
def domain_cols(cols, prefixes):
    return [c for c in cols if any(c == p or c.startswith(p) for p in prefixes)]
def make_specialist():
    return VotingClassifier(estimators=[
      ("extra", ExtraTreesClassifier(n_estimators=70,max_depth=10,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=SEED)),
      ("hist", HistGradientBoostingClassifier(max_iter=80,learning_rate=.05,l2_regularization=2,random_state=SEED)),
      ("logit", make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED)))
    ], voting="soft", n_jobs=1)
def proba(model, X):
    classes=list(model.classes_)
    return model.predict_proba(X)[:,classes.index(1)] if 1 in classes else np.zeros(len(X))
def main():
    warnings.filterwarnings("ignore")
    files=sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    if not files: raise FileNotFoundError("No training_data/*.csv files")
    train, val, test, used, skipped=[],[],[],[],[]
    for path in files:
        tf=path.stem.rsplit("_",1)[-1].lower()
        if tf not in TIMEFRAME_CODES: continue
        candles=load_candles(path)
        if candles is None: skipped.append({"file":path.name,"reason":"invalid/short OHLCV"}); continue
        f=make_features(candles,tf)
        pos=f["_position"]
        n=len(candles); a=int(n*.60); b=int(n*.80)
        # Purge HORIZON rows at boundaries to prevent target overlap.
        parts=[f[pos < a-3], f[(pos >= a+3)&(pos < b-3)], f[pos >= b+3]]
        if any(len(x)<100 or x.label.nunique()<2 for x in parts):
            skipped.append({"file":path.name,"reason":"insufficient split/classes"}); continue
        def cap(x,limit):
            return x if len(x)<=limit else x.iloc[np.linspace(0,len(x)-1,limit).astype(int)]
        train.append(cap(parts[0],MAX_ROWS_PER_FILE)); val.append(cap(parts[1],4000)); test.append(cap(parts[2],4000)); used.append(path.name)
        print(f"{path.name}: train={len(parts[0])}, validation={len(parts[1])}, test={len(parts[2])}")
    if not train: raise RuntimeError("No valid files after chronological split")
    tr=pd.concat(train,ignore_index=True); va=pd.concat(val,ignore_index=True); te=pd.concat(test,ignore_index=True)
    # Remove internal positional index; use only engineered features.
    excluded={"label","_position"}
    features=[c for c in tr.columns if c not in excluded]
    groups={k:domain_cols(features,v) for k,v in DOMAINS.items()}
    groups={k:v for k,v in groups.items() if len(v)>=3}
    if len(groups)<4: raise RuntimeError(f"Too few specialist domains: {list(groups)}")
    models={}; validation_meta=pd.DataFrame(index=va.index); test_meta=pd.DataFrame(index=te.index)
    report_domains={}
    for name, cols in groups.items():
        model=make_specialist()
        model.fit(tr[cols],tr.label.astype(int))
        validation_meta[name]=proba(model,va[cols])
        test_meta[name]=proba(model,te[cols])
        report_domains[name]={"feature_count":len(cols),"features":cols,"validation_auc":None}
        try: report_domains[name]["validation_auc"]=round(float(roc_auc_score(va.label.astype(int),validation_meta[name])),5)
        except Exception: pass
        models[name]={"model":model,"features":cols}
        print(f"Specialist {name}: {len(cols)} features")
    # Train Head AI only on validation-period specialist predictions; evaluate once on later test.
    head=VotingClassifier(estimators=[
      ("logit",make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED))),
      ("extra",ExtraTreesClassifier(n_estimators=100,max_depth=5,min_samples_leaf=30,class_weight="balanced",n_jobs=-1,random_state=SEED))
    ],voting="soft",n_jobs=1)
    head.fit(validation_meta,va.label.astype(int))
    p=proba(head,test_meta); y=te.label.astype(int).to_numpy(); pred=(p>=.5).astype(int)
    try: auc=float(roc_auc_score(y,p))
    except Exception: auc=None
    metrics={"test_rows":int(len(y)),"accuracy":round(float(accuracy_score(y,pred)),5),
      "balanced_accuracy":round(float(balanced_accuracy_score(y,pred)),5),
      "precision":round(float(precision_score(y,pred,zero_division=0)),5),
      "recall":round(float(recall_score(y,pred,zero_division=0)),5),
      "roc_auc":round(auc,5) if auc is not None else None,
      "positive_prediction_rate_pct":round(float(pred.mean()*100),3)}
    bundle={"version":"V6","created_at":datetime.now(timezone.utc).isoformat(),"domains":models,
      "head_model":head,"head_features":list(validation_meta.columns),"features":features,
      "target_note":"Binary direction label from train_ai_v3.make_features; research classification, not a guaranteed profitable trade signal.",
      "timeframe_codes":TIMEFRAME_CODES}
    MODEL_DIR.mkdir(exist_ok=True); joblib.dump(bundle,OUT_MODEL)
    report={"version":"V6","created_at":bundle["created_at"],"status":"trained","specialist_domain_count":len(models),
      "specialist_domains":report_domains,"feature_count":len(features),"metrics":metrics,
      "training_rows":int(len(tr)),"validation_rows":int(len(va)),"test_rows":int(len(te)),
      "files_used":used,"files_skipped":skipped,"scikit_learn":sklearn.__version__,
      "limitations":["Research-only; Scanner.py unchanged.","Test is chronological per file but files may represent correlated assets/timeframes; not a portfolio backtest.",
      "Head model is fit on validation specialist outputs; no cross-fitting, so validate independently before trusting results.",
      "Metrics classify future direction labels and are not net returns. Fees, slippage, funding, risk sizing, and drawdown require separate execution backtests.",
      "Features use only current/past candle data; future return is used only to construct labels."]}
    OUT_REPORT.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("\\nCRYPTO MASTER AI V6 REPORT")
    print(json.dumps({"domains":list(models),"feature_count":len(features),"metrics":metrics,
      "model":str(OUT_MODEL),"report":str(OUT_REPORT)},indent=2))
if __name__=="__main__": main()
