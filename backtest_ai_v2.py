"""Out-of-sample backtest for the experimental Crypto Master AI V2 model.

Uses the exact feature definitions stored by train_ai_v2.py, a chronological
80% holdout, next-candle-open entries, fixed-horizon exits, estimated round-trip
fees/slippage, and non-overlapping trades per symbol/timeframe stream.

This is research tooling, not financial advice or a profitability guarantee.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("training_data")
MODEL_PATH = Path("models/crypto_ai_model_v2.joblib")
OUTPUT_DIR = Path("models")
TRADES_PATH = OUTPUT_DIR / "backtest_v2_trades.csv"
REPORT_PATH = OUTPUT_DIR / "backtest_v2_report.json"

TEST_SPLIT = 0.80
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0003
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
TIMEFRAME_CODES = {"15m": 0, "1h": 1, "4h": 2, "1d": 3}


def load_candles(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError("Missing OHLCV columns")
    timestamp_col = next((c for c in ("timestamp", "time", "datetime", "date") if c in df.columns), None)
    if timestamp_col is None:
        raise ValueError("Missing timestamp column")
    raw = df[timestamp_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.8:
        unit = "ms" if numeric.dropna().median() > 100_000_000_000 else "s"
        df["_time"] = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        df["_time"] = pd.to_datetime(raw, utc=True, errors="coerce")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_time", *required])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)]
    df = df.sort_values("_time").drop_duplicates("_time", keep="last").reset_index(drop=True)
    if len(df) < 300:
        raise ValueError(f"Only {len(df)} valid candles; need at least 300")
    return df


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
    f["_position"] = np.arange(len(df))
    return f.replace([np.inf, -np.inf], np.nan)


def stream_metrics(frame):
    if frame.empty:
        return {"trades": 0}
    returns = frame["net_return_pct"].to_numpy(dtype=float) / 100.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    total_loss = float(abs(losses.sum()))
    equity = np.cumprod(1.0 + returns)
    equity_with_start = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(equity_with_start)
    drawdown = (equity_with_start - peaks) / np.where(peaks == 0, 1.0, peaks)
    pf = float(wins.sum() / total_loss) if total_loss > 0 else (None if len(wins) == 0 else "infinite")
    return {
        "trades": int(len(frame)),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 4),
        "average_net_return_pct": round(float(returns.mean() * 100), 6),
        "median_net_return_pct": round(float(np.median(returns) * 100), 6),
        "total_net_return_pct_compounded_within_stream": round(float((equity[-1] - 1) * 100), 4),
        "profit_factor": pf,
        "max_drawdown_pct_within_stream": round(float(abs(drawdown.min()) * 100), 4),
        "gross_profit_pct_sum": round(float(wins.sum() * 100), 6),
        "gross_loss_pct_sum": round(float(total_loss * 100), 6),
    }


def main():
    print("CRYPTO MASTER AI V2 — OUT-OF-SAMPLE BACKTEST")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"V2 model missing: {MODEL_PATH}. Run the V2 training workflow first.")
    if not DATA_DIR.exists():
        raise FileNotFoundError("training_data folder not found")
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("Unexpected V2 model bundle")
    model = bundle["model"]
    features = list(bundle["features"])
    horizon = int(bundle.get("horizon", 3))
    timeframe_codes = bundle.get("timeframe_codes", TIMEFRAME_CODES)
    if horizon < 1:
        raise ValueError("Invalid horizon in model bundle")
    if set(features) != set(make_features(load_candles(next(iter(sorted(DATA_DIR.glob("*.csv"))))), "15m").columns) - {"_position"}:
        # Do not silently run a model against a different feature schema.
        raise ValueError("Model feature schema does not match V2 backtester features")

    fee_slippage_roundtrip = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
    trades = []
    skipped = []
    processed_files = 0
    total_test_rows = 0
    files = sorted(DATA_DIR.glob("*.csv"))
    for path in files:
        timeframe = path.stem.rsplit("_", 1)[-1].lower()
        if timeframe not in timeframe_codes:
            continue
        symbol = path.stem.rsplit("_", 1)[0]
        try:
            candles = load_candles(path)
            prepared = make_features(candles, timeframe)
            split = int(len(candles) * TEST_SPLIT)
            test = prepared[(prepared["_position"] >= split)].dropna(subset=features).copy()
            if test.empty:
                raise ValueError("No usable rows in chronological holdout")
            probabilities = model.predict_proba(test[features].astype(float))
            classes = list(model.classes_)
            if 1 not in classes:
                raise ValueError("Model does not contain LONG class label 1")
            long_probabilities = probabilities[:, classes.index(1)]
            positions = test["_position"].to_numpy(dtype=int)
            total_test_rows += len(test)
            next_free_position = split
            stream_trade_count = 0
            for signal_position, probability in zip(positions, long_probabilities):
                signal_position = int(signal_position)
                if signal_position < next_free_position:
                    continue
                if probability >= LONG_THRESHOLD:
                    side = "LONG"
                elif probability <= SHORT_THRESHOLD:
                    side = "SHORT"
                else:
                    continue
                entry_position = signal_position + 1
                exit_position = signal_position + horizon
                if entry_position >= len(candles) or exit_position >= len(candles):
                    continue
                entry = float(candles.iloc[entry_position]["open"])
                exit_price = float(candles.iloc[exit_position]["close"])
                if entry <= 0 or exit_price <= 0:
                    continue
                gross = (exit_price / entry - 1) if side == "LONG" else (entry / exit_price - 1)
                net = gross - fee_slippage_roundtrip
                trades.append({
                    "symbol": symbol, "timeframe": timeframe, "side": side,
                    "signal_time": candles.iloc[signal_position]["_time"].isoformat(),
                    "entry_time": candles.iloc[entry_position]["_time"].isoformat(),
                    "exit_time": candles.iloc[exit_position]["_time"].isoformat(),
                    "probability_long": round(float(probability), 8),
                    "entry_price": entry, "exit_price": exit_price,
                    "gross_return_pct": gross * 100,
                    "estimated_round_trip_cost_pct": fee_slippage_roundtrip * 100,
                    "net_return_pct": net * 100,
                    "result": "WIN" if net > 0 else "LOSS",
                })
                stream_trade_count += 1
                next_free_position = exit_position + 1
            processed_files += 1
            print(f"{path.name}: holdout rows={len(test):,}; trades={stream_trade_count}")
        except Exception as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            print(f"SKIP {path.name}: {exc}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["symbol", "timeframe", "side", "signal_time", "entry_time", "exit_time", "probability_long", "entry_price", "exit_price", "gross_return_pct", "estimated_round_trip_cost_pct", "net_return_pct", "result"]
    results = pd.DataFrame(trades, columns=columns)
    if not results.empty:
        results = results.sort_values(["symbol", "timeframe", "exit_time"]).reset_index(drop=True)
    results.to_csv(TRADES_PATH, index=False)

    by_stream = {}
    if not results.empty:
        for (symbol, timeframe), frame in results.groupby(["symbol", "timeframe"], sort=True):
            by_stream[f"{symbol}_{timeframe}"] = stream_metrics(frame)
        returns = results["net_return_pct"].to_numpy(dtype=float)
        long_metrics = stream_metrics(results[results.side == "LONG"])
        short_metrics = stream_metrics(results[results.side == "SHORT"])
        all_returns = returns / 100.0
        wins = all_returns[all_returns > 0]
        losses = all_returns[all_returns < 0]
        loss_sum = float(abs(losses.sum()))
        profit_factor = float(wins.sum() / loss_sum) if loss_sum > 0 else (None if len(wins) == 0 else "infinite")
        summary = {
            "trades": int(len(results)),
            "long_trades": int((results.side == "LONG").sum()),
            "short_trades": int((results.side == "SHORT").sum()),
            "win_rate_pct": round(float((all_returns > 0).mean() * 100), 4),
            "average_net_return_pct_per_trade": round(float(all_returns.mean() * 100), 6),
            "median_net_return_pct_per_trade": round(float(np.median(all_returns) * 100), 6),
            "profit_factor_trade_return_sums": profit_factor,
            "long_metrics": long_metrics,
            "short_metrics": short_metrics,
            "note": "Pooled trade-return sums are descriptive only: different symbols/timeframes may overlap and are not a single investable portfolio. Drawdown is calculated separately within each symbol/timeframe stream.",
        }
    else:
        summary = {"trades": 0, "note": "No qualifying trades. Do not infer strategy performance from an empty sample."}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(MODEL_PATH),
        "data_files_found": len(files),
        "files_processed": processed_files,
        "files_skipped": skipped,
        "holdout_start_fraction": TEST_SPLIT,
        "horizon_candles": horizon,
        "entry_rule": "Signal after candle close; enter at next candle open.",
        "exit_rule": "Exit at close of signal_position + horizon candle.",
        "non_overlapping": "Trades do not overlap within each symbol/timeframe stream; separate streams may overlap.",
        "long_probability_threshold": LONG_THRESHOLD,
        "short_probability_threshold": SHORT_THRESHOLD,
        "assumed_fee_per_side_pct": FEE_PER_SIDE * 100,
        "assumed_slippage_per_side_pct": SLIPPAGE_PER_SIDE * 100,
        "assumed_round_trip_cost_pct": fee_slippage_roundtrip * 100,
        "test_rows_scored": int(total_test_rows),
        "summary": summary,
        "per_symbol_timeframe": by_stream,
        "limitations": [
            "This is a simplified fixed-horizon research backtest, not live execution.",
            "Fee/slippage values are assumptions; actual costs vary by exchange, liquidity and order type.",
            "Funding, liquidation, leverage, spread dynamics, stop-loss/take-profit, market impact and partial fills are not modeled.",
            "No aggregate portfolio drawdown is reported because streams can overlap and capital allocation is unspecified.",
            "Historical performance does not guarantee future results. V2 is not connected to Scanner.py.",
        ],
        "outputs": {"trades_csv": str(TRADES_PATH), "report_json": str(REPORT_PATH)},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("\nBACKTEST SUMMARY")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved trades: {TRADES_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
