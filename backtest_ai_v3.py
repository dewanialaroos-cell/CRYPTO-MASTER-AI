"""V3 Head AI trading-style holdout backtest.
Research only: per-file chronological final 20%, next-open entry, fixed horizon close exit,
non-overlapping trades, configurable fee/slippage assumptions. Does not alter Scanner.py.
"""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train_ai_v3 import DATA_DIR, HORIZON, MODEL_DIR, TIMEFRAME_CODES, load_candles, make_features

MODEL_PATH = MODEL_DIR / "crypto_ai_model_v3.joblib"
REPORT_PATH = MODEL_DIR / "backtest_ai_v3_report.json"
TRADES_PATH = MODEL_DIR / "backtest_ai_v3_trades.csv"

def predict_probabilities(bundle, features):
    probs = pd.DataFrame(index=features.index)
    for key, spec in bundle["specialists"].items():
        model = spec["model"]
        probs[key] = model.predict_proba(features[spec["features"]].astype(float))[:, list(model.classes_).index(1)]
    for group, keys in bundle["specialist_groups"].items():
        probs[group + "__mean"] = probs[keys].mean(axis=1)
    cols = bundle["head_features"]
    return bundle["head_model"].predict_proba(probs[cols])[:, list(bundle["head_model"].classes_).index(1)]

def backtest_file(path, bundle, fee_pct, slippage_pct, threshold):
    tf = path.stem.rsplit("_", 1)[-1].lower()
    if tf not in TIMEFRAME_CODES:
        return [], {"file": path.name, "status": "skipped", "reason": "unsupported timeframe"}
    candles = load_candles(path)
    if candles is None:
        return [], {"file": path.name, "status": "skipped", "reason": "missing/invalid OHLCV"}
    feat = make_features(candles, tf)
    feat = feat[feat["_position"] >= int(len(candles) * 0.80)].copy()
    if len(feat) < 10:
        return [], {"file": path.name, "status": "skipped", "reason": "test segment too small"}
    feat["prob_long"] = predict_probabilities(bundle, feat[bundle["features"]])
    feat["entry"] = candles["open"].shift(-1).reindex(feat.index)
    feat["exit"] = candles["close"].shift(-HORIZON).reindex(feat.index)
    feat = feat.dropna(subset=["prob_long", "entry", "exit"])
    rows, i, idxs = [], 0, list(feat.index)
    while i < len(idxs):
        idx = idxs[i]
        p = float(feat.at[idx, "prob_long"])
        side = "LONG" if p >= threshold else ("SHORT" if p <= 1-threshold else None)
        if side is None:
            i += 1
            continue
        entry, exit_price = float(feat.at[idx, "entry"]), float(feat.at[idx, "exit"])
        if entry <= 0 or exit_price <= 0:
            i += 1
            continue
        gross = (exit_price / entry - 1) * (1 if side == "LONG" else -1)
        net = gross - 2 * (fee_pct + slippage_pct) / 100
        rows.append({"file":path.name,"symbol":path.stem.rsplit("_",1)[0],"timeframe":tf,
          "timestamp":str(candles.at[idx,"_time"]),"side":side,"prob_long":p,"entry":entry,
          "exit":exit_price,"horizon_candles":HORIZON,"gross_return_pct":gross*100,
          "net_return_pct":net*100,"fee_round_trip_pct":2*fee_pct,
          "slippage_round_trip_pct":2*slippage_pct})
        target = int(feat.at[idx, "_position"]) + HORIZON
        while i < len(idxs) and int(feat.at[idxs[i], "_position"]) <= target:
            i += 1
    return rows, {"file":path.name,"status":"tested","test_signals":len(feat),"trades":len(rows)}

def summarize(records, equity):
    if not records:
        return {"trades":0,"win_rate_pct":None,"avg_net_return_pct":None,"profit_factor":None,
                "compounded_return_pct":0.0,"max_drawdown_pct":0.0,"ending_equity":equity}
    d = pd.DataFrame(records)
    r = d["net_return_pct"].astype(float)/100
    wins, losses = r[r>0].sum(), -r[r<0].sum()
    curves, drawdowns, endings = [], [], []
    for _, g in d.groupby("file", sort=False):
        curve = equity * (1 + g["net_return_pct"].astype(float)/100).cumprod()
        curves.extend(curve.tolist())
        drawdowns.extend(((curve/curve.cummax())-1).mul(100).tolist())
        endings.append(float(curve.iloc[-1]))
    return {"trades":int(len(d)),"win_rate_pct":round(float((r>0).mean()*100),4),
      "avg_net_return_pct":round(float(r.mean()*100),6),"median_net_return_pct":round(float(r.median()*100),6),
      "profit_factor":round(float(wins/losses),6) if losses>0 else None,
      "mean_per_file_compounded_return_pct":round(float(np.mean(endings)/equity-1)*100,4),
      "worst_file_max_drawdown_pct":round(float(min(drawdowns)),4) if drawdowns else 0.0,
      "mean_ending_equity_per_file":round(float(np.mean(endings)),2),"starting_equity_per_file":equity}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fee-pct",type=float,default=0.04,help="fee per side, percent")
    p.add_argument("--slippage-pct",type=float,default=0.02,help="slippage per side, percent")
    p.add_argument("--threshold",type=float,default=0.60)
    p.add_argument("--equity",type=float,default=1000.0)
    a=p.parse_args()
    if not 0.5<a.threshold<1: raise ValueError("threshold must be between 0.5 and 1")
    if min(a.fee_pct,a.slippage_pct)<0: raise ValueError("costs cannot be negative")
    if not MODEL_PATH.exists(): raise FileNotFoundError("Run V3 training first: "+str(MODEL_PATH))
    bundle=joblib.load(MODEL_PATH)
    if bundle.get("model_version")!="v3-specialist-head-experimental": raise ValueError("Saved model is not expected V3 format")
    all_trades, file_reports=[],[]
    for path in sorted(DATA_DIR.glob("*.csv")):
        rows,rep=backtest_file(path,bundle,a.fee_pct,a.slippage_pct,a.threshold)
        all_trades.extend(rows); file_reports.append(rep)
        print(f'{path.name}: {rep.get("trades",0)} trades' if rep["status"]=="tested" else f'{path.name}: SKIP {rep["reason"]}')
    overall=summarize(all_trades,a.equity); by_side={}; by_tf={}; by_file={}
    if all_trades:
        d=pd.DataFrame(all_trades)
        by_side={k:summarize(g.to_dict("records"),a.equity) for k,g in d.groupby("side")}
        by_tf={k:summarize(g.to_dict("records"),a.equity) for k,g in d.groupby("timeframe")}
        by_file={k:summarize(g.to_dict("records"),a.equity) for k,g in d.groupby("file")}
    report={"model_version":bundle["model_version"],"created_at_utc":pd.Timestamp.now(tz="UTC").isoformat(),
      "method":"Per-file final 20% chronological holdout; next candle open entry; fixed HORIZON close exit; non-overlapping trades.",
      "threshold":a.threshold,"fees":{"fee_pct_per_side":a.fee_pct,"slippage_pct_per_side":a.slippage_pct,
      "round_trip_cost_pct":2*(a.fee_pct+a.slippage_pct)},"overall":overall,"by_side":by_side,"by_timeframe":by_tf,
      "by_file":by_file,"files":file_reports,"warnings":[
      "Paper backtest only; no scanner or live trading integration.",
      "Fee and slippage are configurable assumptions, not verified exchange rates.",
      "Fixed horizon only; stop loss, take profit, funding, liquidation, partial fills and latency are not simulated.",
      "Equity is compounded independently per file; this is not a single combined portfolio curve.",
      "The head was trained on validation-period specialist outputs, so stacking is temporal holdout, not cross-fitted stacking."]}
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(all_trades,columns=["file","symbol","timeframe","timestamp","side","prob_long","entry","exit","horizon_candles","gross_return_pct","net_return_pct","fee_round_trip_pct","slippage_round_trip_pct"]).to_csv(TRADES_PATH,index=False)
    print("\nV3 HEAD AI BACKTEST\n"+json.dumps({"overall":overall,"by_side":by_side,"by_timeframe":by_tf,"report":str(REPORT_PATH),"trades":len(all_trades)},indent=2))
if __name__=="__main__": main()
