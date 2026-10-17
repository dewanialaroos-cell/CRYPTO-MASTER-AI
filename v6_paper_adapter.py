"""V6 Head AI paper-only inference adapter for Scanner.py."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/crypto_master_ai_v6.joblib")
PAPER_LOG = Path("paper_trading_v6.jsonl")
THRESHOLD_LONG = float(os.getenv("V6_LONG_THRESHOLD", "0.60"))
THRESHOLD_SHORT = float(os.getenv("V6_SHORT_THRESHOLD", "0.40"))
FEE_PER_SIDE = float(os.getenv("PAPER_FEE_PER_SIDE", "0.0004"))
SLIPPAGE_PER_SIDE = float(os.getenv("PAPER_SLIPPAGE_PER_SIDE", "0.0002"))

def _positive_probability(model, frame):
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return 0.5
    return float(model.predict_proba(frame)[0, classes.index(1)])

def v6_paper_signal(exchange, symbol, timeframe="1h"):
    """Return a fail-closed paper signal. Never submits exchange orders."""
    result = {
        "v6_status": "UNAVAILABLE",
        "v6_probability": None,
        "v6_signal": "HOLD",
        "v6_model_version": "V6",
        "v6_paper_only": True,
    }
    if not MODEL_PATH.exists():
        result["v6_status"] = "MODEL_FILE_MISSING"
        return result
    try:
        import train_ai_v3
        bundle = joblib.load(MODEL_PATH)
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
        if not candles or len(candles) < 200:
            result["v6_status"] = "INSUFFICIENT_OHLCV"
            return result
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["_time", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
        features = train_ai_v3.make_features(df, timeframe, include_unlabeled=True)
        if features.empty:
            result["v6_status"] = "FEATURES_EMPTY"
            return result
        row = features.iloc[[-1]]
        domains = bundle.get("domains", {})
        meta = {}
        for name, item in domains.items():
            cols = item.get("features", [])
            model = item.get("model")
            if model is None or not cols or any(col not in row.columns for col in cols):
                result["v6_status"] = "FEATURE_SCHEMA_MISMATCH"
                return result
            meta[name] = _positive_probability(model, row[cols])
        head = bundle.get("head_model")
        head_cols = bundle.get("head_features", [])
        if head is None or not head_cols or any(col not in meta for col in head_cols):
            result["v6_status"] = "HEAD_SCHEMA_MISMATCH"
            return result
        head_frame = pd.DataFrame([{col: meta[col] for col in head_cols}], columns=head_cols)
        probability = _positive_probability(head, head_frame)
        signal = "LONG" if probability >= THRESHOLD_LONG else "SHORT" if probability <= THRESHOLD_SHORT else "HOLD"
        result.update({
            "v6_status": "OK",
            "v6_probability": round(probability, 6),
            "v6_signal": signal,
            "v6_timeframe": timeframe,
            "v6_domain_probabilities": {k: round(v, 6) for k, v in meta.items()},
            "v6_paper_only": True,
        })
        # Append the observation only; no exchange order is ever placed here.
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "price": float(df["close"].iloc[-1]),
            "probability": probability,
            "signal": signal,
            "fee_per_side": FEE_PER_SIDE,
            "slippage_per_side": SLIPPAGE_PER_SIDE,
            "paper_only": True,
        }
        with PAPER_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return result
    except Exception as exc:
        result["v6_status"] = "ERROR:" + type(exc).__name__
        return result
