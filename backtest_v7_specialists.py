"""Out-of-sample trading simulation for trained V7 specialists.
Uses the chronological final 20% test segment, model probabilities, observed future returns,
and conservative round-trip fee/slippage assumptions. Research only; never places orders.
"""
import json, warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path("specialist_data")
MODEL_PATH = Path("models/crypto_master_ai_v7_specialists.joblib")
REPORT_PATH = Path("models/backtest_v7_specialists_report.json")
TRADES_PATH = Path("models/backtest_v7_specialists_trades.csv")
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
FEE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.02
DOMAIN_FOLDERS = {
    "fundamentals_ai": "fundamentals",
    "news_ai": "news",
    "indicator_ai": "indicators",
    "tradingview_ai": "tradingview",
}

def read_domain(folder):
    frames = []
    for path in sorted(list(folder.glob("*.csv")) + list(folder.glob("*.csv.gz"))):
        try:
            d = pd.read_csv(path)
            d.columns = [str(c).strip().lower() for c in d.columns]
            if "label" not in d:
                continue
            d["_source_file"] = path.name
            frames.append(d)
        except Exception as exc:
            print(f"SKIP {path}: {type(exc).__name__}: {exc}")
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[data["label"].isin([0, 1])].copy()
    tcol = next((c for c in ("timestamp", "time", "datetime", "date") if c in data.columns), None)
    if tcol:
        data["_sort_time"] = pd.to_datetime(data[tcol], utc=True, errors="coerce")
        data = data.sort_values("_sort_time", kind="stable", na_position="last").reset_index(drop=True)
    return data

def score_trades(prob, future_return):
    costs_pct = 2 * (FEE_PER_SIDE_PCT + SLIPPAGE_PER_SIDE_PCT)
    records = []
    for i, (p, ret) in enumerate(zip(prob, future_return)):
        if not np.isfinite(p) or not np.isfinite(ret):
            continue
        side = "LONG" if p >= LONG_THRESHOLD else ("SHORT" if p <= SHORT_THRESHOLD else "FLAT")
        if side == "FLAT":
            continue
        gross_pct = float(ret) * 100 * (1 if side == "LONG" else -1)
        net_pct = gross_pct - costs_pct
        records.append({"test_row": i, "side": side, "probability_up": float(p),
                        "future_return_pct": float(ret) * 100,
                        "gross_return_pct": gross_pct, "net_return_pct": net_pct})
    return records

def summarize(records):
    if not records:
        return {"status": "NO QUALIFYING TRADES", "trades": 0}
    r = np.array([x["net_return_pct"] / 100 for x in records], dtype=float)
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    equity = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(equity)
    dd = equity / peaks - 1
    return {
        "status": "BACKTESTED",
        "trades": int(len(r)),
        "long_trades": int(sum(x["side"] == "LONG" for x in records)),
        "short_trades": int(sum(x["side"] == "SHORT" for x in records)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 3),
        "avg_net_return_pct": round(float(r.mean() * 100), 5),
        "profit_factor": round(wins / losses, 4) if losses else ("infinite" if wins else None),
        "compounded_return_pct_trade_sequence": round(float((equity[-1] - 1) * 100), 3),
        "max_drawdown_pct_trade_sequence": round(float(dd.min() * 100), 3),
        "mean_probability_up": round(float(np.mean([x["probability_up"] for x in records])), 5),
    }

def main():
    warnings.filterwarnings("ignore")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"V7 model bundle missing: {MODEL_PATH}; train V7 first")
    bundle = joblib.load(MODEL_PATH)
    models = bundle.get("models", {})
    reports = {}
    all_trades = []
    round_trip_cost = 2 * (FEE_PER_SIDE_PCT + SLIPPAGE_PER_SIDE_PCT)
    for model_name, folder_name in DOMAIN_FOLDERS.items():
        if model_name not in models:
            reports[model_name] = {"status": "NOT BACKTESTED", "reason": "Model not trained"}
            continue
        folder = ROOT / folder_name
        data = read_domain(folder) if folder.exists() else None
        if data is None or len(data) < 100:
            reports[model_name] = {"status": "NOT BACKTESTED", "reason": "No usable labeled dataset"}
            continue
        spec = models[model_name]
        features = spec.get("features", [])
        missing = [c for c in features if c not in data.columns]
        if missing:
            reports[model_name] = {"status": "ERROR", "reason": "Missing model features", "missing_features": missing}
            continue
        # Match the trainer's chronological 60/20/20 split and boundary purge.
        n = len(data)
        te_start = min(n, int(n * .80) + 3)
        test = data.iloc[te_start:].copy()
        if len(test) < 30 or test["label"].nunique() < 2:
            reports[model_name] = {"status": "NOT BACKTESTED", "reason": "Insufficient final chronological holdout", "test_rows": len(test)}
            continue
        model = spec["model"]
        X = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
        # Prefer actual forward returns from the same source rows. If unavailable, do not fake PnL.
        return_col = next((c for c in ("future_return_4h", "future_return_4d", "future_return_1d") if c in test.columns), None)
        if return_col is None:
            y = test["label"].astype(int).to_numpy()
            pred = (prob >= .5).astype(int)
            reports[model_name] = {
                "status": "CLASSIFICATION ONLY — NO PRICE-RETURN BACKTEST",
                "test_rows": int(len(test)), "roc_auc": None,
                "reason": "No observed forward-return column; trade PnL is not fabricated."
            }
            continue
        actual = pd.to_numeric(test[return_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(prob)
        test_prob = prob[valid]
        actual = actual[valid]
        raw = score_trades(test_prob, actual)
        # Map trade rows back to original file and timestamp for audit.
        valid_test = test.loc[valid].reset_index(drop=True)
        for rec in raw:
            j = rec.pop("test_row")
            rec.update({
                "model": model_name,
                "source_file": str(valid_test.iloc[j].get("_source_file", "")),
                "timestamp": str(valid_test.iloc[j].get("timestamp", valid_test.iloc[j].get("date", ""))),
                "return_column": return_col,
                "round_trip_cost_pct": round_trip_cost,
            })
        summary = summarize(raw)
        summary.update({
            "test_rows": int(len(test)),
            "test_start": str(test.iloc[0].get("timestamp", test.iloc[0].get("date", ""))),
            "test_end": str(test.iloc[-1].get("timestamp", test.iloc[-1].get("date", ""))),
            "forward_return_column": return_col,
            "long_threshold": LONG_THRESHOLD,
            "short_threshold": SHORT_THRESHOLD,
            "fee_per_side_pct": FEE_PER_SIDE_PCT,
            "slippage_per_side_pct": SLIPPAGE_PER_SIDE_PCT,
            "round_trip_cost_pct": round_trip_cost,
            "source_files": sorted(test["_source_file"].dropna().unique().tolist()) if "_source_file" in test else [],
        })
        reports[model_name] = summary
        all_trades.extend(raw)
        print(f"{model_name}: {summary.get('status')} | trades={summary.get('trades', 0)} | avg net={summary.get('avg_net_return_pct')}")
    trades = pd.DataFrame(all_trades)
    if trades.empty:
        trades = pd.DataFrame(columns=["model","source_file","timestamp","side","probability_up","future_return_pct","gross_return_pct","net_return_pct","return_column","round_trip_cost_pct"])
    TRADES_PATH.parent.mkdir(exist_ok=True)
    trades.to_csv(TRADES_PATH, index=False)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "V7 domain specialists",
        "status": "COMPLETED",
        "method": "Each trained model is evaluated on the chronological final 20% holdout, matching the V7 trainer split. Trades only use observed forward returns; thresholds are 0.55 LONG and 0.45 SHORT. Fee and slippage assumptions are deducted.",
        "warning": "Research simulation only. The trade sequence combines different symbols/timeframes and is not a simultaneous portfolio simulation. Dataset label/return horizons vary by domain. Results are not guarantees of future performance. Scanner.py and live trading are unchanged.",
        "thresholds": {"long": LONG_THRESHOLD, "short": SHORT_THRESHOLD},
        "costs": {"fee_per_side_pct": FEE_PER_SIDE_PCT, "slippage_per_side_pct": SLIPPAGE_PER_SIDE_PCT},
        "models": reports,
        "trades_csv": str(TRADES_PATH),
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
