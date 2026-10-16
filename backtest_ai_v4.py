"""V4 audited research backtest for Crypto Master AI.
Keeps the production Scanner.py untouched. Uses chronological per-file holdouts,
next-open entry, close-at-horizon exit, non-overlapping trades, cost stress tests,
and correctly labelled per-file (not combined-portfolio) equity statistics.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_ai_v3 import DATA_DIR, HORIZON, MODEL_DIR, TIMEFRAME_CODES, load_candles, make_features

MODEL_PATH = MODEL_DIR / "crypto_ai_model_v3.joblib"
REPORT_PATH = MODEL_DIR / "backtest_ai_v4_audit_report.json"
TRADES_PATH = MODEL_DIR / "backtest_ai_v4_audit_trades.csv"


def positive_class_probability(model, frame):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model has no positive class (1)")
    return model.predict_proba(frame)[:, classes.index(1)]


def predict_probabilities(bundle, features):
    probs = pd.DataFrame(index=features.index)
    for key, spec in bundle["specialists"].items():
        probs[key] = positive_class_probability(
            spec["model"], features[spec["features"]].astype(float)
        )
    for group, keys in bundle["specialist_groups"].items():
        probs[group + "__mean"] = probs[keys].mean(axis=1)
    return positive_class_probability(bundle["head_model"], probs[bundle["head_features"]])


def collect_candidates(path, bundle):
    tf = path.stem.rsplit("_", 1)[-1].lower()
    if tf not in TIMEFRAME_CODES:
        return [], {"file": path.name, "status": "skipped", "reason": "unsupported timeframe"}
    candles = load_candles(path)
    if candles is None:
        return [], {"file": path.name, "status": "skipped", "reason": "missing/invalid OHLCV"}

    feat = make_features(candles, tf)
    # Use the final 20% by original candle position, not by the shorter labelled frame.
    feat = feat[feat["_position"] >= int(len(candles) * 0.80)].copy()
    if len(feat) < 10:
        return [], {"file": path.name, "status": "skipped", "reason": "test segment too small"}

    feat["prob_long"] = predict_probabilities(bundle, feat[bundle["features"]])
    feat["entry"] = candles["open"].shift(-1).reindex(feat.index)
    feat["exit"] = candles["close"].shift(-HORIZON).reindex(feat.index)
    feat["entry_time"] = candles["_time"].shift(-1).reindex(feat.index)
    feat["exit_time"] = candles["_time"].shift(-HORIZON).reindex(feat.index)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prob_long", "entry", "exit", "entry_time", "exit_time"]
    )
    rows = []
    for idx, row in feat.iterrows():
        rows.append({
            "file": path.name,
            "symbol": path.stem.rsplit("_", 1)[0],
            "timeframe": tf,
            "signal_time": str(candles.at[idx, "_time"]),
            "entry_time": str(row["entry_time"]),
            "exit_time": str(row["exit_time"]),
            "signal_position": int(row["_position"]),
            "prob_long": float(row["prob_long"]),
            "entry": float(row["entry"]),
            "exit": float(row["exit"]),
        })
    return rows, {"file": path.name, "status": "tested", "test_rows": len(rows)}


def select_trades(candidates, threshold, fee_pct, slippage_pct):
    """Choose trades independently per file; no overlapping HORIZON windows."""
    selected = []
    by_file = {}
    for c in candidates:
        by_file.setdefault(c["file"], []).append(c)
    for file_name, rows in by_file.items():
        rows.sort(key=lambda x: x["signal_position"])
        next_allowed_position = -1
        for row in rows:
            pos = row["signal_position"]
            if pos <= next_allowed_position:
                continue
            p = row["prob_long"]
            side = "LONG" if p >= threshold else ("SHORT" if p <= 1 - threshold else None)
            if side is None:
                continue
            entry, exit_price = row["entry"], row["exit"]
            if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0 or exit_price <= 0:
                continue
            gross = (exit_price / entry - 1.0) * (1 if side == "LONG" else -1)
            cost = 2 * (fee_pct + slippage_pct) / 100.0
            net = gross - cost
            selected.append({
                **row, "side": side, "horizon_candles": HORIZON,
                "gross_return_pct": gross * 100,
                "net_return_pct": net * 100,
                "fee_round_trip_pct": 2 * fee_pct,
                "slippage_round_trip_pct": 2 * slippage_pct,
            })
            next_allowed_position = pos + HORIZON
    return selected


def summarize(records, starting_equity=1000.0):
    if not records:
        return {"trades": 0, "win_rate_pct": None, "avg_net_return_pct": None,
                "profit_factor": None, "mean_per_file_compounded_return_pct": None,
                "worst_file_max_drawdown_pct": None}
    d = pd.DataFrame(records)
    r = pd.to_numeric(d["net_return_pct"], errors="coerce").dropna() / 100.0
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    file_returns, file_drawdowns, ending_equities = [], [], []
    for _, group in d.groupby("file", sort=False):
        gr = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna() / 100.0
        curve = starting_equity * (1.0 + gr).cumprod()
        equity_path = pd.concat([pd.Series([starting_equity]), curve.reset_index(drop=True)], ignore_index=True)
        drawdown = (equity_path / equity_path.cummax() - 1.0) * 100.0
        file_returns.append((float(curve.iloc[-1]) / starting_equity - 1.0) * 100.0)
        file_drawdowns.append(float(drawdown.min()))
        ending_equities.append(float(curve.iloc[-1]))
    return {
        "trades": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 4),
        "avg_net_return_pct": round(float(r.mean() * 100), 6),
        "median_net_return_pct": round(float(r.median() * 100), 6),
        "profit_factor": round(gains / losses, 6) if losses > 0 else (None if gains == 0 else "infinite"),
        "gross_avg_return_pct": round(float(pd.to_numeric(d["gross_return_pct"]).mean()), 6),
        "total_fees_pct_sum_of_trade_returns": round(float(pd.to_numeric(d["fee_round_trip_pct"]).sum()), 6),
        "total_slippage_pct_sum_of_trade_returns": round(float(pd.to_numeric(d["slippage_round_trip_pct"]).sum()), 6),
        "mean_per_file_compounded_return_pct": round(float(np.mean(file_returns)), 4),
        "worst_file_max_drawdown_pct": round(float(min(file_drawdowns)), 4),
        "mean_ending_equity_per_file": round(float(np.mean(ending_equities)), 2),
        "starting_equity_per_file": starting_equity,
        "equity_note": "Each CSV/file is compounded independently from the same starting equity; this is NOT one combined portfolio.",
    }


def grouped_summary(records, key, starting_equity):
    if not records:
        return {}
    d = pd.DataFrame(records)
    return {str(name): summarize(group.to_dict("records"), starting_equity)
            for name, group in d.groupby(key, sort=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--fee-pct", type=float, default=0.04, help="fee per side, percent")
    parser.add_argument("--slippage-pct", type=float, default=0.02, help="slippage per side, percent")
    parser.add_argument("--equity", type=float, default=1000.0)
    args = parser.parse_args()
    if not 0.50 < args.threshold < 1.0:
        raise ValueError("threshold must be greater than 0.50 and less than 1.00")
    if args.fee_pct < 0 or args.slippage_pct < 0 or args.equity <= 0:
        raise ValueError("costs must be non-negative and equity must be positive")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Run V3 training first: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    if bundle.get("model_version") != "v3-specialist-head-experimental":
        raise ValueError("Saved model is not the expected V3 format")

    candidates, file_reports = [], []
    for path in sorted(DATA_DIR.glob("*.csv")):
        rows, report = collect_candidates(path, bundle)
        candidates.extend(rows)
        file_reports.append(report)
        print(f"{path.name}: {len(rows)} test rows" if report["status"] == "tested"
              else f"{path.name}: SKIP {report['reason']}")

    trades = select_trades(candidates, args.threshold, args.fee_pct, args.slippage_pct)
    overall = summarize(trades, args.equity)
    threshold_sensitivity = {}
    for threshold in (0.55, 0.60, 0.65, 0.70):
        ts = select_trades(candidates, threshold, args.fee_pct, args.slippage_pct)
        threshold_sensitivity[str(threshold)] = summarize(ts, args.equity)

    cost_sensitivity = {}
    for fee, slip, label in [
        (args.fee_pct, args.slippage_pct, "base_cost"),
        (args.fee_pct * 1.5, args.slippage_pct * 1.5, "1_5x_cost"),
        (args.fee_pct * 2, args.slippage_pct * 2, "2x_cost"),
    ]:
        ts = select_trades(candidates, args.threshold, fee, slip)
        cost_sensitivity[label] = {
            "fee_pct_per_side": fee, "slippage_pct_per_side": slip,
            "metrics": summarize(ts, args.equity),
        }

    report = {
        "model_version": bundle["model_version"],
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "Per-file chronological final 20% holdout; signal after candle close; entry at next candle open; exit at close HORIZON candles after signal; non-overlapping positions per CSV.",
        "threshold": args.threshold,
        "horizon_candles": HORIZON,
        "fees": {"fee_pct_per_side": args.fee_pct, "slippage_pct_per_side": args.slippage_pct,
                 "round_trip_cost_pct": 2 * (args.fee_pct + args.slippage_pct)},
        "overall": overall,
        "by_side": grouped_summary(trades, "side", args.equity),
        "by_timeframe": grouped_summary(trades, "timeframe", args.equity),
        "by_file": grouped_summary(trades, "file", args.equity),
        "threshold_sensitivity": threshold_sensitivity,
        "cost_sensitivity": cost_sensitivity,
        "files": file_reports,
        "audit_notes": [
            "Research-only; production Scanner.py is not modified.",
            "Trade timing matches the model label's close-to-close HORIZON target direction, but realized trade return is next-open to future close.",
            "The backtest is a per-file holdout, not a live forward test; the model has been trained using historical data.",
            "Per-file equity curves are independent and must not be interpreted as one portfolio's compounded return.",
            "No stop loss, take profit, funding, liquidation, partial fills, market impact, latency, or cross-symbol capital constraints are simulated.",
            "Threshold/cost sensitivity is diagnostic and must not be used to select a threshold on the same holdout and then claim unbiased performance.",
            "Fees/slippage are assumptions, not verified exchange-specific execution costs.",
            "Head model uses validation-period specialist outputs; stacking is not cross-fitted."
        ],
    }
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["file", "symbol", "timeframe", "signal_time", "entry_time", "exit_time",
               "signal_position", "side", "prob_long", "entry", "exit", "horizon_candles",
               "gross_return_pct", "net_return_pct", "fee_round_trip_pct", "slippage_round_trip_pct"]
    pd.DataFrame(trades, columns=columns).to_csv(TRADES_PATH, index=False)
    print("\nV4 AUDITED BACKTEST SUMMARY\n" + json.dumps({
        "overall": overall, "by_timeframe": report["by_timeframe"],
        "threshold_sensitivity": threshold_sensitivity,
        "cost_sensitivity": cost_sensitivity,
        "report": str(REPORT_PATH), "trades_csv": str(TRADES_PATH)
    }, indent=2))


if __name__ == "__main__":
    main()
