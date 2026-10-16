"""Independent chronological trading backtests for each V6 candle specialist.
Research only. Uses next-candle open entry, HORIZON-candle close exit, non-overlapping
trades, and fee/slippage assumptions. Results are per file, not a combined portfolio.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from train_ai_v3 import DATA_DIR, HORIZON, MODEL_DIR, TIMEFRAME_CODES, load_candles, make_features

MODEL_PATH = MODEL_DIR / "crypto_master_ai_v6.joblib"
REPORT_PATH = MODEL_DIR / "backtest_v6_specialists_report.json"
TRADES_PATH = MODEL_DIR / "backtest_v6_specialists_trades.csv"
THRESHOLD = 0.60
FEE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.02


def positive_probability(model, frame):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(frame))
    return model.predict_proba(frame)[:, classes.index(1)]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"V6 model missing: {MODEL_PATH}; run train_ai_v6.py first")
    bundle = joblib.load(MODEL_PATH)
    domains = bundle.get("domains", {})
    if not domains:
        raise ValueError("V6 bundle has no specialist domains")

    all_trades = []
    file_reports = []
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
        row_count = 0
        for domain, spec in domains.items():
            cols = spec["features"]
            missing = [c for c in cols if c not in feat.columns]
            if missing:
                continue
            probs = positive_probability(spec["model"], feat[cols].astype(float))
            candidates = []
            for idx, p in zip(feat.index, probs):
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
                costs = 2 * (FEE_PER_SIDE_PCT + SLIPPAGE_PER_SIDE_PCT) / 100
                candidates.append({
                    "model": domain, "file": path.name, "symbol": path.stem.rsplit("_", 1)[0],
                    "timeframe": tf, "signal_position": pos, "side": side,
                    "probability_up": float(p), "entry": entry, "exit": exit_price,
                    "gross_return_pct": gross * 100, "net_return_pct": (gross - costs) * 100,
                    "fee_round_trip_pct": 2 * FEE_PER_SIDE_PCT,
                    "slippage_round_trip_pct": 2 * SLIPPAGE_PER_SIDE_PCT,
                })
            candidates.sort(key=lambda r: r["signal_position"])
            last_exit = -1
            for trade in candidates:
                if trade["signal_position"] <= last_exit:
                    continue
                all_trades.append(trade)
                last_exit = trade["signal_position"] + HORIZON
                row_count += 1
        file_reports.append({"file": path.name, "status": "TESTED", "specialist_trades": row_count})

    trades_df = pd.DataFrame(all_trades)
    summaries = {}
    for domain in domains:
        d = trades_df[trades_df["model"] == domain] if not trades_df.empty else pd.DataFrame()
        if d.empty:
            summaries[domain] = {"status": "NO TRADES", "trades": 0, "reason": "No test signals crossed the probability threshold"}
            continue
        r = pd.to_numeric(d["net_return_pct"], errors="coerce").dropna() / 100
        wins = float(r[r > 0].sum())
        losses = float(-r[r < 0].sum())
        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax() - 1
        summaries[domain] = {
            "status": "BACKTESTED", "trades": int(len(r)),
            "win_rate_pct": round(float((r > 0).mean() * 100), 3),
            "avg_net_return_pct": round(float(r.mean() * 100), 5),
            "profit_factor": round(wins / losses, 4) if losses else (None if not wins else "infinite"),
            "compounded_return_pct_single_trade_sequence": round(float((equity.iloc[-1] - 1) * 100), 3),
            "max_drawdown_pct_single_trade_sequence": round(float(drawdown.min() * 100), 3),
            "fees_per_side_pct": FEE_PER_SIDE_PCT, "slippage_per_side_pct": SLIPPAGE_PER_SIDE_PCT,
            "threshold": THRESHOLD, "horizon_candles": HORIZON,
        }
    MODEL_DIR.mkdir(exist_ok=True)
    if not trades_df.empty:
        trades_df.to_csv(TRADES_PATH, index=False)
    else:
        pd.DataFrame(columns=["model","file","symbol","timeframe","side","net_return_pct"]).to_csv(TRADES_PATH, index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "V6", "status": "completed",
        "method": "Independent specialist signals; chronological final 20% holdout; next-open entry; close at configured horizon; non-overlapping trades per model/file; fees and slippage included.",
        "not_portfolio": "Per-specialist trade sequences are aggregated across files and are not a simultaneous portfolio simulation.",
        "specialist_backtests": summaries, "files": file_reports,
        "trades_csv": str(TRADES_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
