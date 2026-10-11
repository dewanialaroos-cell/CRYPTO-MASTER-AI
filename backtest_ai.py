import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

# =====================================================
# CRYPTO MASTER AI - HISTORICAL BACKTEST ENGINE
# =====================================================

DATA_DIR = Path("training_data")
MODEL_PATH = Path("models/crypto_ai_model.joblib")
OUTPUT_DIR = Path("models")

RESULTS_PATH = OUTPUT_DIR / "backtest_results.csv"
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# Match the training configuration.
HORIZON = 3
TEST_SPLIT = 0.80

# Assumed costs per side. Replace these with costs suitable
# for your intended exchange and market.
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0003

# Only take signals with at least this predicted probability.
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45

TIMEFRAME_CODES = {
    "15m": 0,
    "1h": 1,
    "4h": 2,
    "1d": 3,
}

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


def load_candles(path):
    """Read, clean and chronologically sort OHLCV data."""

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}

    if not required.issubset(df.columns):
        raise ValueError(f"{path.name}: missing OHLCV columns")

    timestamp_col = next(
        (
            c for c in ("timestamp", "time", "datetime", "date")
            if c in df.columns
        ),
        None,
    )

    if timestamp_col is None:
        raise ValueError(f"{path.name}: timestamp column missing")

    raw_time = df[timestamp_col]
    numeric_time = pd.to_numeric(raw_time, errors="coerce")

    if numeric_time.notna().mean() > 0.8:
        median_value = numeric_time.dropna().median()
        unit = "ms" if median_value > 100_000_000_000 else "s"

        df["_time"] = pd.to_datetime(
            numeric_time,
            unit=unit,
            utc=True,
            errors="coerce",
        )
    else:
        df["_time"] = pd.to_datetime(
            raw_time,
            utc=True,
            errors="coerce",
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["_time", "open", "high", "low", "close", "volume"]
    )

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ]

    df = (
        df.sort_values("_time")
        .drop_duplicates("_time", keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 200:
        raise ValueError(f"{path.name}: insufficient candle history")

    return df


def build_features(df, timeframe):
    """Use the same feature definitions as train_ai.py."""

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

    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)

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

    avg_volume = volume.rolling(20).mean()
    features["volume_ratio"] = (
        volume / avg_volume.replace(0, np.nan)
    )

    features["range_pct"] = (high - low) / close
    features["body_pct"] = (close - open_price) / open_price

    candle_range = (high - low).replace(0, np.nan)
    features["close_position"] = (close - low) / candle_range

    features["volatility_20"] = close.pct_change().rolling(20).std()
    features["timeframe_code"] = TIMEFRAME_CODES[timeframe]

    result = features.replace([np.inf, -np.inf], np.nan).copy()
    result["_position"] = np.arange(len(df))

    result = result.dropna(subset=FEATURES).copy()

    return result


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    print("=" * 55)
    print(" CRYPTO MASTER AI - HISTORICAL BACKTEST ")
    print("=" * 55)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run the training workflow first."
        )

    if not DATA_DIR.exists():
        raise FileNotFoundError("training_data folder not found.")

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Unexpected model format; retrain the model.")

    model = bundle["model"]
    model_features = bundle.get("features", FEATURES)
    horizon = int(bundle.get("horizon", HORIZON))

    if list(model_features) != FEATURES:
        raise ValueError(
            "Model features do not match this backtester. "
            "Use the matching training code."
        )

    if horizon != HORIZON:
        raise ValueError(
            f"Model horizon is {horizon}, but backtester horizon is {HORIZON}."
        )

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV history found in training_data.")

    all_trades = []
    skipped_files = []

    total_test_candidates = 0
    processed_files = 0

    for path in files:
        timeframe = path.stem.rsplit("_", 1)[-1].lower()

        if timeframe not in TIMEFRAME_CODES:
            continue

        symbol = path.stem.rsplit("_", 1)[0]

        print(f"\nBacktesting {symbol} {timeframe} ...")

        try:
            candles = load_candles(path)

            split_position = int(len(candles) * TEST_SPLIT)

            prepared = build_features(candles, timeframe)

            # Only use the chronological holdout section.
            test = prepared[
                prepared["_position"] >= split_position
            ].copy()

            if test.empty:
                raise ValueError("No usable rows in held-out period.")

            X_test = test[FEATURES].astype(float)

            probabilities = model.predict_proba(X_test)
            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError("Trained model has no LONG class.")

            long_index = classes.index(1)
            prob_long = probabilities[:, long_index]

            positions = test["_position"].to_numpy(dtype=int)

            total_test_candidates += len(test)

            # Avoid overlapping trades in the same symbol/timeframe file.
            next_available_position = split_position
            file_trade_count = 0

            for position, probability in zip(positions, prob_long):
                signal_position = int(position)

                if signal_position < next_available_position:
                    continue

                # Entry is the next candle's open, not the signal candle's close.
                entry_position = signal_position + 1

                # Exit after HORIZON candles have elapsed from entry.
                exit_position = signal_position + HORIZON

                if entry_position >= len(candles):
                    continue

                if exit_position >= len(candles):
                    continue

                if probability >= LONG_THRESHOLD:
                    side = "LONG"
                elif probability <= SHORT_THRESHOLD:
                    side = "SHORT"
                else:
                    continue

                entry_price = float(
                    candles.iloc[entry_position]["open"]
                )
                exit_price = float(
                    candles.iloc[exit_position]["close"]
                )

                if entry_price <= 0 or exit_price <= 0:
                    continue

                if side == "LONG":
                    gross_return = (exit_price / entry_price) - 1
                else:
                    gross_return = (entry_price - exit_price) / entry_price

                # Round-trip fee and slippage assumptions.
                total_cost = 2 * (
                    FEE_PER_SIDE + SLIPPAGE_PER_SIDE
                )

                net_return = gross_return - total_cost

                all_trades.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "side": side,
                        "signal_time": candles.iloc[
                            signal_position
                        ]["_time"].isoformat(),
                        "entry_time": candles.iloc[
                            entry_position
                        ]["_time"].isoformat(),
                        "exit_time": candles.iloc[
                            exit_position
                        ]["_time"].isoformat(),
                        "probability_long": float(probability),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return_pct": gross_return * 100,
                        "estimated_cost_pct": total_cost * 100,
                        "net_return_pct": net_return * 100,
                        "result": "WIN" if net_return > 0 else "LOSS",
                    }
                )

                file_trade_count += 1
                next_available_position = exit_position + 1

            processed_files += 1

            print(
                f"Test rows: {len(test):,} | "
                f"Non-overlapping signals: {file_trade_count}"
            )

        except Exception as exc:
            skipped_files.append(
                {"file": path.name, "reason": str(exc)}
            )
            print(f"Skipped {path.name}: {exc}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "symbol", "timeframe", "side",
        "signal_time", "entry_time", "exit_time",
        "probability_long", "entry_price", "exit_price",
        "gross_return_pct", "estimated_cost_pct",
        "net_return_pct", "result",
    ]

    results = pd.DataFrame(all_trades, columns=columns)

    if not results.empty:
        results = results.sort_values("exit_time").reset_index(drop=True)

    results.to_csv(RESULTS_PATH, index=False)

    if len(results):
        returns = results["net_return_pct"].to_numpy() / 100.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        # This is an event-sequence illustration, not a portfolio
        # drawdown calculation; trades across different files may overlap.
        equity = np.cumprod(1 + returns)
        equity_with_start = np.concatenate(([1.0], equity))
        running_peak = np.maximum.accumulate(equity_with_start)

        drawdowns = (
            equity_with_start - running_peak
        ) / running_peak

        gross_profit = float(wins.sum())
        gross_loss = float(abs(losses.sum()))

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else None
        )

        compounded_return_pct = (
            float(equity[-1] - 1) * 100
        )

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(MODEL_PATH),
            "holdout_fraction": 1 - TEST_SPLIT,
            "prediction_horizon_candles": horizon,
            "long_probability_threshold": LONG_THRESHOLD,
            "short_probability_threshold": SHORT_THRESHOLD,
            "fee_per_side_pct": FEE_PER_SIDE * 100,
            "slippage_per_side_pct": SLIPPAGE_PER_SIDE * 100,
            "files_processed": processed_files,
            "test_candidates": int(total_test_candidates),
            "completed_trades": int(len(results)),
            "long_trades": int((results["side"] == "LONG").sum()),
            "short_trades": int((results["side"] == "SHORT").sum()),
            "wins": int((returns > 0).sum()),
            "losses": int((returns <= 0).sum()),
            "win_rate_pct": float((returns > 0).mean() * 100),
            "average_gross_return_pct": float(
                results["gross_return_pct"].mean()
            ),
            "average_net_return_pct": float(
                results["net_return_pct"].mean()
            ),
            "profit_factor": safe_float(profit_factor)
            if profit_factor is not None else None,
            "compounded_trade_sequence_return_pct": compounded_return_pct,
            "event_sequence_max_drawdown_pct": float(
                abs(drawdowns.min()) * 100
            ),
            "skipped_files": skipped_files,
            "limitations": [
                "Signal-level test, not a complete portfolio simulation.",
                "Trades from different symbols and timeframes can overlap.",
                "Fees and slippage are assumptions, not live exchange quotes.",
                "Funding, leverage, liquidation, stop-loss and take-profit "
                "execution are not simulated.",
                "Historical performance does not guarantee future results."
            ],
        }

    else:
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files_processed": processed_files,
            "test_candidates": int(total_test_candidates),
            "completed_trades": 0,
            "skipped_files": skipped_files,
            "message": (
                "No qualifying trades were found. Review model outputs, "
                "probability thresholds and skipped files."
            ),
        }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\n" + "=" * 55)
    print(" BACKTEST SUMMARY ")
    print("=" * 55)

    print(f"Files processed: {processed_files}")
    print(f"Test candidates: {total_test_candidates}")
    print(f"Completed trades: {len(results)}")

    if len(results):
        print(f"Win rate: {report['win_rate_pct']:.2f}%")
        print(
            "Average net return per trade: "
            f"{report['average_net_return_pct']:.4f}%"
        )
        print(
            "Profit factor: "
            f"{report['profit_factor']}"
        )
        print(
            "Event-sequence drawdown estimate: "
            f"{report['event_sequence_max_drawdown_pct']:.2f}%"
        )

    print(f"\nSaved trades: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print("\nBacktest completed. Results are historical, not a profit guarantee.")


if __name__ == "__main__":
    main()
