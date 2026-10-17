"""V8 conservative per-specialist backtest.
Uses chronological holdout, forward returns only for PnL, non-overlapping trades per source file,
fixed fractional exposure and costs. Research only; no exchange orders.
Stop-loss/take-profit are intentionally not simulated because source datasets do not provide
intraperiod OHLC paths aligned to each external-domain signal.
"""
import json, warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

ROOT = Path("specialist_data")
MODEL_PATH = Path("models/crypto_master_ai_v7_specialists.joblib")
REPORT_PATH = Path("models/backtest_v8_risk_report.json")
TRADES_PATH = Path("models/backtest_v8_risk_trades.csv")
DOMAINS = {
    "fundamentals_ai": "fundamentals",
    "news_ai": "news",
    "indicator_ai": "indicators",
    "tradingview_ai": "tradingview",
}
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
FEE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.02
MAX_EXPOSURE_PCT = 1.0
INITIAL_EQUITY = 100.0
HORIZON_ROWS = 4

def load_files(folder):
    frames = []
    for path in sorted(list(folder.glob("*.csv")) + list(folder.glob("*.csv.gz"))):
        try:
            d = pd.read_csv(path)
            d.columns = [str(c).strip().lower() for c in d.columns]
            if "label" not in d:
                continue
            d["_source_file"] = path.name
            tcol = next((c for c in ("timestamp", "time", "datetime", "date") if c in d.columns), None)
            if tcol:
                d["_sort_time"] = pd.to_datetime(d[tcol], utc=True, errors="coerce")
                d = d.sort_values("_sort_time", kind="stable", na_position="last")
            frames.append(d)
        except Exception as e:
            print(f"SKIP {path}: {type(e).__name__}: {e}")
    return frames

def summarize(trades):
    if not trades:
        return {"status": "NO QUALIFYING TRADES", "trades": 0}
    r = np.array([t["net_return_pct"] / 100.0 for t in trades], dtype=float)
    # Fixed 1% of current equity is allocated per trade; this is exposure sizing,
    # not 1% stop-loss risk. Returns are scaled by that exposure fraction.
    equity = [INITIAL_EQUITY]
    peak = INITIAL_EQUITY
    max_dd = 0.0
    for ret in r:
        equity.append(equity[-1] * (1.0 + MAX_EXPOSURE_PCT / 100.0 * ret))
        peak = max(peak, equity[-1])
        max_dd = min(max_dd, equity[-1] / peak - 1.0)
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    return {
        "status": "BACKTESTED",
        "trades": int(len(r)),
        "long_trades": int(sum(t["side"] == "LONG" for t in trades)),
        "short_trades": int(sum(t["side"] == "SHORT" for t in trades)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 3),
        "avg_net_return_per_full_exposure_pct": round(float(r.mean() * 100), 5),
        "profit_factor": round(wins / losses, 4) if losses else ("infinite" if wins else None),
        "equity_start": INITIAL_EQUITY,
        "equity_end": round(float(equity[-1]), 5),
        "equity_return_pct_at_1pct_exposure": round(float((equity[-1] / INITIAL_EQUITY - 1) * 100), 5),
        "max_drawdown_pct_at_1pct_exposure": round(float(max_dd * 100), 5),
    }

def main():
    warnings.filterwarnings("ignore")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bundle missing: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    models = bundle.get("models", {})
    report_models = {}
    all_trades = []
    round_trip_cost_pct = 2 * (FEE_PER_SIDE_PCT + SLIPPAGE_PER_SIDE_PCT)
    for model_name, folder_name in DOMAINS.items():
        spec = models.get(model_name)
        if not spec:
            report_models[model_name] = {"status": "NOT TRAINED"}
            continue
        folder = ROOT / folder_name
        frames = load_files(folder) if folder.exists() else []
        if not frames:
            report_models[model_name] = {"status": "NO DATA"}
            continue
        data = pd.concat(frames, ignore_index=True, sort=False)
        data["label"] = pd.to_numeric(data["label"], errors="coerce")
        data = data[data["label"].isin([0, 1])].copy()
        tcol = next((c for c in ("timestamp", "time", "datetime", "date") if c in data.columns), None)
        if tcol:
            data["_sort_time"] = pd.to_datetime(data[tcol], utc=True, errors="coerce")
            data = data.sort_values(["_source_file", "_sort_time"], kind="stable", na_position="last")
        features = spec.get("features", [])
        forbidden = [c for c in features if str(c).lower().startswith(("future_", "target_", "label_"))
                     or "future_return" in str(c).lower()
                     or str(c).lower() in {"label", "target", "outcome", "y_true"}]
        if forbidden:
            report_models[model_name] = {"status": "INVALID MODEL — TARGET LEAKAGE", "features": forbidden}
            continue
        missing = [c for c in features if c not in data.columns]
        if missing:
            report_models[model_name] = {"status": "MISSING FEATURES", "missing_features": missing}
            continue
        return_col = next((c for c in ("future_return_4d", "future_return_1d", "future_return_4h") if c in data.columns), None)
        if not return_col:
            report_models[model_name] = {"status": "CLASSIFICATION ONLY — NO FORWARD RETURNS"}
            continue
        trades = []
        for source, group in data.groupby("_source_file", sort=False):
            group = group.reset_index(drop=True)
            split = int(len(group) * 0.8) + 3
            if len(group) - split < 30:
                continue
            test = group.iloc[split:].copy().reset_index(drop=True)
            X = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            try:
                p = spec["model"].predict_proba(X)[:, list(spec["model"].classes_).index(1)]
            except Exception as e:
                print(f"{model_name}/{source} scoring skipped: {e}")
                continue
            ret = pd.to_numeric(test[return_col], errors="coerce").to_numpy(dtype=float)
            next_available = 0
            for i in range(len(test)):
                if i < next_available or not np.isfinite(p[i]) or not np.isfinite(ret[i]):
                    continue
                side = "LONG" if p[i] >= LONG_THRESHOLD else ("SHORT" if p[i] <= SHORT_THRESHOLD else "FLAT")
                if side == "FLAT":
                    continue
                gross_pct = float(ret[i] * 100 * (1 if side == "LONG" else -1))
                net_pct = gross_pct - round_trip_cost_pct
                trades.append({
                    "model": model_name, "source_file": source,
                    "timestamp": str(test.iloc[i].get(tcol, "")) if tcol else "",
                    "side": side, "probability_up": float(p[i]),
                    "forward_return_pct": float(ret[i] * 100),
                    "gross_return_pct": gross_pct, "net_return_pct": net_pct,
                    "return_column": return_col, "round_trip_cost_pct": round_trip_cost_pct,
                    "hold_rows": HORIZON_ROWS, "test_row": i
                })
                # Prevent overlapping horizon trades within each source file.
                next_available = i + HORIZON_ROWS
        summary = summarize(trades)
        summary.update({
            "classification_test_rows": int(sum(max(0, len(g) - (int(len(g)*0.8)+3)) for _, g in data.groupby("_source_file"))),
            "forward_return_column": return_col,
            "long_threshold": LONG_THRESHOLD, "short_threshold": SHORT_THRESHOLD,
            "fee_per_side_pct": FEE_PER_SIDE_PCT, "slippage_per_side_pct": SLIPPAGE_PER_SIDE_PCT,
            "round_trip_cost_pct": round_trip_cost_pct,
            "max_exposure_pct_of_equity_per_trade": MAX_EXPOSURE_PCT,
            "hold_rows": HORIZON_ROWS,
            "stop_loss_take_profit": "NOT SIMULATED: intraperiod OHLC path unavailable/alignment not verified",
            "method_note": "Non-overlapping trades per source file; per-model results remain separate, not a joint portfolio."
        })
        report_models[model_name] = summary
        all_trades.extend(trades)
        print(f"{model_name}: {summary.get('status')} trades={summary.get('trades',0)} PF={summary.get('profit_factor')}")
    REPORT_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(all_trades).to_csv(TRADES_PATH, index=False)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "V8 risk-controlled research backtest",
        "status": "COMPLETED",
        "models": report_models,
        "assumptions": {
            "long_threshold": LONG_THRESHOLD, "short_threshold": SHORT_THRESHOLD,
            "fee_per_side_pct": FEE_PER_SIDE_PCT, "slippage_per_side_pct": SLIPPAGE_PER_SIDE_PCT,
            "max_exposure_pct_of_equity_per_trade": MAX_EXPOSURE_PCT,
            "initial_equity": INITIAL_EQUITY,
            "overlap_policy": "Skip new signals until previous HORIZON_ROWS trade horizon ends, separately per source file",
            "warning": "Research simulation only. 1% exposure is not 1% stop-loss risk. SL/TP not simulated without aligned intraperiod OHLC. No joint portfolio or live orders."
        },
        "trades_csv": str(TRADES_PATH)
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
