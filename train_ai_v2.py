"""Experimental, leakage-aware Crypto Master AI V2 trainer.

This is deliberately separate from Scanner.py. It creates a model and report only;
it does not place trades or send alerts. Review held-out results and backtest before
any integration with the live scanner.
"""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
DATA_DIR = Path("training_data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "crypto_ai_model_v2.joblib"
REPORT_PATH = MODEL_DIR / "ai_training_report_v2.json"
HORIZON = 3
MAX_ROWS_PER_FILE = 150000
TIMEFRAME_CODES = {"15m": 0, "1h": 1, "4h": 2, "1d": 3}


def load_candles(path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Skip {path.name}: read error: {exc}")
        return None
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"Skip {path.name}: missing OHLCV columns")
        return None
    tcol = next((c for c in ("timestamp", "time", "datetime", "date") if c in df.columns), None)
    if not tcol:
        print(f"Skip {path.name}: no timestamp")
        return None
    raw = df[tcol]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.8:
        unit = "ms" if numeric.dropna().median() > 100_000_000_000 else "s"
        df["_time"] = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        df["_time"] = pd.to_datetime(raw, utc=True, errors="coerce")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_time", *required])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["volume"] >= 0)]
    df = df.sort_values("_time").drop_duplicates("_time", keep="last").reset_index(drop=True)
    return df if len(df) >= 300 else None


def make_features(df, timeframe):
    o, h, l, c, v = (df[x].astype(float) for x in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=df.index)
    for n in (1, 2, 3, 6, 12, 24):
        f[f"return_{n}"] = c.pct_change(n)
    for n in (9, 20, 50, 100, 200):
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"ema_distance_{n}"] = c / ema - 1
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)
    previous = c.shift(1)
    tr = pd.concat([(h - l), (h - previous).abs(), (l - previous).abs()], axis=1).max(axis=1)
    f["atr_pct_14"] = tr.rolling(14).mean() / c
    f["atr_pct_50"] = tr.rolling(50).mean() / c
    for n in (10, 20, 50):
        f[f"volatility_{n}"] = c.pct_change().rolling(n).std()
        avg_volume = v.rolling(n).mean()
        f[f"volume_ratio_{n}"] = v / avg_volume.replace(0, np.nan)
    f["range_pct"] = (h - l) / c
    f["body_pct"] = (c - o) / o
    candle_range = (h - l).replace(0, np.nan)
    f["close_position"] = (c - l) / candle_range
    f["upper_wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    f["lower_wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c
    f["high_distance_20"] = c / h.rolling(20).max() - 1
    f["low_distance_20"] = c / l.rolling(20).min() - 1
    f["high_distance_50"] = c / h.rolling(50).max() - 1
    f["low_distance_50"] = c / l.rolling(50).min() - 1
    f["timeframe_code"] = TIMEFRAME_CODES[timeframe]
    future_return = c.shift(-HORIZON) / c - 1
    threshold = np.maximum(0.0015, f["atr_pct_14"].fillna(0) * 0.5)
    f["label"] = np.where(future_return > threshold, 1, np.where(future_return < -threshold, 0, np.nan))
    f["_pos"] = np.arange(len(df))
    f = f.replace([np.inf, -np.inf], np.nan)
    return f.dropna(subset=["label", "_pos"]).copy()


def main():
    print("CRYPTO MASTER AI V2 — chronological model comparison")
    files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    if not files:
        raise FileNotFoundError("No training_data/*.csv files found")
    train_parts, val_parts, test_parts, used, skipped = [], [], [], [], []
    for path in files:
        timeframe = path.stem.rsplit("_", 1)[-1].lower()
        if timeframe not in TIMEFRAME_CODES:
            continue
        candles = load_candles(path)
        if candles is None:
            skipped.append(path.name)
            continue
        prepared = make_features(candles, timeframe)
        n = len(candles)
        first, second = int(n * 0.60), int(n * 0.80)
        positions = prepared["_pos"]
        train = prepared[positions < first - HORIZON]
        validation = prepared[(positions >= first) & (positions < second - HORIZON)]
        test = prepared[positions >= second]
        if min(len(train), len(validation), len(test)) < 100 or any(part["label"].nunique() < 2 for part in (train, validation, test)):
            skipped.append(path.name)
            print(f"Skip {path.name}: insufficient rows/classes in chronological split")
            continue

        def cap(frame, limit):
            if len(frame) <= limit:
                return frame
            indices = np.linspace(0, len(frame) - 1, limit).astype(int)
            return frame.iloc[indices]

        train_parts.append(cap(train, MAX_ROWS_PER_FILE))
        val_parts.append(cap(validation, MAX_ROWS_PER_FILE // 3))
        test_parts.append(cap(test, MAX_ROWS_PER_FILE // 3))
        used.append(path.name)
        print(f"{path.name}: train={len(train):,} validation={len(validation):,} test={len(test):,}")
    if not train_parts:
        raise RuntimeError("No CSVs had enough usable data and both classes in all chronological splits")
    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    features = [col for col in train.columns if col not in ("label", "_pos")]
    X_train, y_train = train[features].astype(float), train["label"].astype(int)
    X_val, y_val = validation[features].astype(float), validation["label"].astype(int)
    X_test, y_test = test[features].astype(float), test["label"].astype(int)
    candidates = {
        "logistic_regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, l2_regularization=1.0, random_state=42))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=160, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
    }
    validation_scores, fitted = {}, {}
    for name, model in candidates.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        validation_probability = model.predict_proba(X_val)[:, 1]
        validation_scores[name] = float(roc_auc_score(y_val, validation_probability))
        fitted[name] = model
        print(f"  validation ROC AUC={validation_scores[name]:.4f}")
    selected_name = max(validation_scores, key=validation_scores.get)
    selected_model = fitted[selected_name]
    predictions = selected_model.predict(X_test)
    probabilities = selected_model.predict_proba(X_test)[:, 1]
    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "v2-experimental",
        "selected_model": selected_name,
        "validation_roc_auc_by_model": validation_scores,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "long_precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
            "short_precision": float(precision_score(y_test, predictions, pos_label=0, zero_division=0)),
            "long_recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
            "short_recall": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        },
        "train_rows": int(len(train)), "validation_rows": int(len(validation)), "test_rows": int(len(test)),
        "features": features, "horizon_candles": HORIZON, "files_used": used, "files_skipped": skipped,
        "classes": {"0": "SHORT direction", "1": "LONG direction"},
        "limitations": "Direction classification only, not proof of profitability. Fees, slippage, funding, liquidation, position sizing and execution are not modeled. V2 is not connected to Scanner.py.",
        "scikit_learn_version": sklearn.__version__,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": selected_model, "features": features, "timeframe_codes": TIMEFRAME_CODES, "horizon": HORIZON, "classes": {0: "SHORT", 1: "LONG"}, "model_version": "v2-experimental"}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nHeld-out test (not used for model selection):")
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Selected model: {selected_name}; report: {REPORT_PATH}; model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
