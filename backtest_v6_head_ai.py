"""Combined V6 Head AI chronological trading backtest.
Research only: combines the available V6 specialists through the trained Head AI.
Uses final 20% of each OHLCV file, next-open entry, fixed horizon exit,
non-overlapping trades per file, and fee/slippage assumptions.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from train_ai_v3 import DATA_DIR, HORIZON, MODEL_DIR, TIMEFRAME_CODES, load_candles, make_features

MODEL_PATH = MODEL_DIR / "crypto_master_ai_v6.joblib"
REPORT_PATH = MODEL_DIR / "backtest_v6_head_ai_report.json"
TRADES_PATH = MODEL_DIR / "backtest_v6_head_ai_trades.csv"
THRESHOLD = 0.55
FEE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.02


def positive_probability(model, frame):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(frame))
    return model.predict_proba(frame)[:, classes.index(1)]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"V6 model missing: {MODEL_PATH}; train V6 first")
    bundle = joblib.load(MODEL_PATH)
    domains = bundle.get("domains", {})
    head = bundle.get("head_model")
    head_features = bundle.get("head_features", [])
    if not domains or head is None or not head_features:
        raise ValueError("V6 bundle must contain specialists, head_model, and head_features")

    all_trades = []
    file_reports = []
    costs = 2 * (FEE_PER_SIDE_PCT + SLIPPAGE_PER_SIDE_PCT) / 100
    for path in sorted(DATA_DIR.glob("*.csv")):
        tf = path.stem.rsplit("_", 1)[-1].lower()
        if tf not in TIMEFRAME_CODES:
            continue
        candles = load_candles(path)
        if candles is None:
            file_reports.append({"file": path.name, "status": "SKIPPED", "reason": "invalid OHLCV"})
            continue
        feat = make_features(candles, tf)
        feat = feat[feat["_position"] >= int(len(candles) * 0.80)].copy()
        if len(feat) < 10:
            file_reports.append({"file": path.name, "status": "SKIPPED", "reason": "short test segment"})
            continue

        specialist_probs = {}
        missing_domains = []
        for domain, spec in domains.items():
            cols = spec.get("features", [])
            if not cols or any(c not in feat.columns for c in cols):
                missing_domains.append(domain)
                continue
            specialist_probs[domain] = positive_probability(
                spec["model"], feat[cols].astype(float)
            )
        available = [name for name in head_features if name in specialist_probs]
        if len(available) != len(head_features):
            file_reports.append({
                "file": path.name, "status": "SKIPPED",
                "reason": "head specialist features unavailable",
                "missing_domains": [name for name in head_features if name not in specialist_probs],
            })
            continue
        head_frame = pd.DataFrame({name: specialist_probs[name] for name in head_features}, index=feat.index)
        head_probs = positive_probability(head, head_frame[head_features])
        candidates = []
        for idx, p in zip(feat.index, head_probs):
            pos = int(feat.at[idx, "_position"])
            entry_idx = pos + 1
            exit_idx = pos + HORIZON
            if exit_idx >= len(candles):
                continue
            entry = float(candles.iloc[entry_idx]["open"])
            exit_price = float(candles.iloc[exit_idx]["close"])
            if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0 or exit_price <= 0:
                continue
            side = "LONG" if p >= THRESHOLD else ("SHORT" if p <= 1 - THRESHOLD else None)
            if side is None:
                continue
            gross = (exit_price / entry - 1.0) * (1 if side == "LONG" else -1)
            net = (gross - costs) * 100
            candidates.append({
                "model": "head_ai", "file": path.name,
                "symbol": path.stem.rsplit("_", 1)[0], "timeframe": tf,
                "signal_position": pos, "side": side, "head_probability_up": float(p),
                "entry": entry, "exit": exit_price,
                "gross_return_pct": gross * 100, "net_return_pct": net,
                "fee_round_trip_pct": 2 * FEE_PER_SIDE_PCT,
                "slippage_round_trip_pct": 2 * SLIPPAGE_PER_SIDE_PCT,
            })
        candidates.sort(key=lambda r: r["signal_position"])
        selected = []
        last_exit = -1
        for trade in candidates:
            if trade["signal_position"] <= last_exit:
                continue
            selected.append(trade)
            last_exit = trade["signal_position"] + HORIZON
        all_trades.extend(selected)
        file_reports.append({
            "file": path.name, "status": "TESTED", "trades": len(selected),
            "long_trades": sum(t["side"] == "LONG" for t in selected),
            "short_trades": sum(t["side"] == "SHORT" for t in selected),
            "skipped_specialists": missing_domains,
        })

    trades = pd.DataFrame(all_trades)
    if trades.empty:
        trades = pd.DataFrame(columns=[
            "model", "file", "symbol", "timeframe", "signal_position", "side",
            "head_probability_up", "entry", "exit", "gross_return_pct", "net_return_pct",
            "fee_round_trip_pct", "slippage_round_trip_pct",
        ])
    r = pd.to_numeric(trades["net_return_pct"], errors="coerce").dropna() / 100
    wins = float(r[r > 0].sum()) if len(r) else 0.0
    losses = float(-r[r < 0].sum()) if len(r) else 0.0
    equity = (1 + r).cumprod() if len(r) else pd.Series(dtype=float)
    drawdown = equity / equity.cummax() - 1 if len(equity) else pd.Series(dtype=float)
    summary = {
        "status": "BACKTESTED" if len(r) else "NO TRADES",
        "trades": int(len(r)),
        "target_minimum_trades": 200,
        "target_met": bool(len(r) >= 200),
        "long_trades": int((trades["side"] == "LONG").sum()) if len(trades) else 0,
        "short_trades": int((trades["side"] == "SHORT").sum()) if len(trades) else 0,
        "win_rate_pct": round(float((r > 0).mean() * 100), 3) if len(r) else None,
        "avg_net_return_pct": round(float(r.mean() * 100), 5) if len(r) else None,
        "profit_factor": round(wins / losses, 4) if losses else (None if not wins else "infinite"),
        "compounded_return_pct_single_trade_sequence": round(float((equity.iloc[-1] - 1) * 100), 3) if len(equity) else None,
        "max_drawdown_pct_single_trade_sequence": round(float(drawdown.min() * 100), 3) if len(drawdown) else None,
        "fees_per_side_pct": FEE_PER_SIDE_PCT,
        "slippage_per_side_pct": SLIPPAGE_PER_SIDE_PCT,
        "threshold": THRESHOLD, "horizon_candles": HORIZON,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    trades.to_csv(TRADES_PATH, index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "V6 Head AI",
        "method": "Combined Head AI over specialist probabilities; chronological final 20% holdout per file; next-open entry; close at configured horizon; non-overlapping trades per file; fees and slippage included.",
        "warning": "Aggregated across symbols/timeframes; not a simultaneous portfolio simulation. Backtest results are historical simulations, not guarantees. Minimum 200 trades is a target, not forced.",
        "head_ai_backtest": summary, "files": file_reports,
        "trades_csv": str(TRADES_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
