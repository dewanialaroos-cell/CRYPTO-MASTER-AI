import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ==========================================
# CRYPTO MASTER AI - HISTORICAL TRAINING
# ==========================================

DATA_DIR = Path("training_data")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "crypto_ai_model.joblib"
REPORT_PATH = MODEL_DIR / "ai_training_report.json"

MAX_ROWS_PER_FILE = 40000
HORIZON = 3

FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "ema_distance_20",
    "ema_distance_50",
    "rsi_14",
    "atr_pct",
    "volume_ratio",
    "range_pct",
    "body_pct",
    "close_position",
    "volatility_20",
    "timeframe_code",
]

TIMEFRAME_CODES = {
    "15m": 0,
    "1h": 1,
    "4h": 2,
    "1d": 3,
}


def load_candles(path):
    """Load and validate one historical OHLCV CSV file."""

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read {path.name}: {exc}")
        return None

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required = {
        "open", "high", "low", "close", "volume"
    }

    if not required.issubset(df.columns):
        print(f"Skipping {path.name}: missing OHLCV columns")
        return None

    timestamp_column = next(
        (
            name for name in ("timestamp", "time", "datetime", "date")
            if name in df.columns
        ),
        None,
    )

    if timestamp_column:
        raw_time = df[timestamp_column]

        numeric_time = pd.to_numeric(raw_time, errors="coerce")

        if numeric_time.notna().mean() > 0.8:
            median_time = numeric_time.dropna().median()
            unit = "ms" if median_time > 100_000_000_000 else "s"
            df["_time"] = pd.to_datetime(
                numeric_time, unit=unit, utc=True, errors="coerce"
            )
        else:
            df["_time"] = pd.to_datetime(
                raw_time, utc=True, errors="coerce"
            )

        df = df.sort_values("_time").drop_duplicates(
            subset=["_time"], keep="last"
        )
    else:
        print(f"Skipping {path.name}: no timestamp column")
        return None

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["_time", "open", "high", "low", "close", "volume"]
    )

    df = df[
        (df["close"] > 0)
        & (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["volume"] >= 0)
    ].copy()

    if len(df) < 200:
        print(f"Skipping {path.name}: not enough candles")
        return None

    return df.reset_index(drop=True)


def build_features(df, timeframe):
    """Build indicators using current and previous candles only."""

    close = df["close"]
    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_6"] = close.pct_change(6)
    features["return_12"] = close.pct_change(12)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    features["ema_distance_20"] = close / ema20 - 1
    features["ema_distance_50"] = close / ema50 - 1

    change = close.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    features["rsi_14"] = rsi.fillna(50)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(14).mean()
    features["atr_pct"] = atr / close

    average_volume = volume.rolling(20).mean()
    features["volume_ratio"] = (
        volume / average_volume.replace(0, np.nan)
    )

    features["range_pct"] = (high - low) / close
    features["body_pct"] = (close - open_price) / open_price

    candle_range = (high - low).replace(0, np.nan)
    features["close_position"] = (close - low) / candle_range

    features["volatility_20"] = (
        close.pct_change().rolling(20).std()
    )

    features["timeframe_code"] = TIMEFRAME_CODES[timeframe]

    # Future return is used only to create the historical training label.
    # It is NOT included among prediction features.
    future_return = close.shift(-HORIZON) / close - 1

    # Ignore very small moves that may mostly be market noise.
    threshold = np.maximum(
        0.0015,
        features["atr_pct"].fillna(0) * 0.5,
    )

    # 1 = LONG direction; 0 = SHORT direction.
    # Neutral moves are excluded from this first model.
    label = np.where(
        future_return > threshold,
        1.0,
        np.where(future_return < -threshold, 0.0, np.nan),
    )

    result = features.copy()
    result["label"] = label
    result["_position"] = np.arange(len(df))

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(
        subset=FEATURES + ["label", "_position"]
    ).copy()

    result["label"] = result["label"].astype(int)

    return result


def main():
    print("=" * 55)
    print(" CRYPTO MASTER AI - MODEL TRAINING ")
    print("=" * 55)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "training_data folder not found in repository root."
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No historical CSV files found in training_data/."
        )

    train_parts = []
    test_parts = []
    files_used = []
    skipped_files = []

    for path in files:
        timeframe = path.stem.rsplit("_", 1)[-1].lower()

        if timeframe not in TIMEFRAME_CODES:
            continue

        print(f"\nPreparing {path.name} ...")

        candles = load_candles(path)

        if candles is None:
            skipped_files.append(path.name)
            continue

        split_position = int(len(candles) * 0.80)

        prepared = build_features(candles, timeframe)

        # Keep the time-based split: older candles train the model,
        # newer candles test it. Do not randomly mix the two.
        train = prepared[
            prepared["_position"] < split_position - HORIZON
        ].copy()

        test = prepared[
            prepared["_position"] >= split_position
        ].copy()

        # Keep a representative sample across the history to control
        # memory use without selecting only the most recent candles.
        if len(train) > MAX_ROWS_PER_FILE:
            selected = np.linspace(
                0, len(train) - 1, MAX_ROWS_PER_FILE
            ).astype(int)
            train = train.iloc[selected].copy()

        if len(test) > MAX_ROWS_PER_FILE // 4:
            selected = np.linspace(
                0, len(test) - 1, MAX_ROWS_PER_FILE // 4
            ).astype(int)
            test = test.iloc[selected].copy()

        if len(train) and len(test):
            train_parts.append(train)
            test_parts.append(test)
            files_used.append(path.name)

            print(
                f"Train rows: {len(train):,} | "
                f"Test rows: {len(test):,}"
            )
        else:
            skipped_files.append(path.name)
            print("Skipped: insufficient usable training/test rows.")

    if not train_parts or not test_parts:
        raise RuntimeError(
            "Not enough usable history. Check training_data CSV files."
        )

    train_data = pd.concat(train_parts, ignore_index=True)
    test_data = pd.concat(test_parts, ignore_index=True)

    X_train = train_data[FEATURES].astype(float)
    y_train = train_data["label"].astype(int)

    X_test = test_data[FEATURES].astype(float)
    y_test = test_data["label"].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training data contains only one class. Cannot train."
        )

    if y_test.nunique() < 2:
        raise RuntimeError(
            "Test data contains only one class. Cannot evaluate reliably."
        )

    print("\nTraining Logistic Regression model ...")
    print(f"Training examples: {len(X_train):,}")
    print(f"Testing examples: {len(X_test):,}")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "StandardScaler + LogisticRegression",
        "scikit_learn_version": sklearn.__version__,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_files_used": files_used,
        "skipped_files": skipped_files,
        "feature_names": FEATURES,
        "prediction_horizon_candles": HORIZON,
        "classes": {
            "0": "SHORT direction",
            "1": "LONG direction",
        },
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "long_precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "short_precision": float(
            precision_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "long_recall": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "short_recall": float(
            recall_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["SHORT", "LONG"],
            zero_division=0,
            output_dict=True,
        ),
        "warning": (
            "Historical direction classification only. "
            "Results do not include fees, slippage, funding, "
            "liquidations, or a complete trading-strategy backtest."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "timeframe_codes": TIMEFRAME_CODES,
            "horizon": HORIZON,
            "classes": {0: "SHORT", 1: "LONG"},
            "scikit_learn_version": sklearn.__version__,
        },
        MODEL_PATH,
    )

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n===== HELD-OUT HISTORICAL TEST =====")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"LONG precision:    {metrics['long_precision']:.4f}")
    print(f"SHORT precision:   {metrics['short_precision']:.4f}")
    print(f"ROC AUC:           {metrics['roc_auc']:.4f}")

    print("\nSaved trained model:", MODEL_PATH)
    print("Saved training report:", REPORT_PATH)
    print("\nTraining finished. These results are not a profit guarantee.")


if __name__ == "__main__":
    main()
