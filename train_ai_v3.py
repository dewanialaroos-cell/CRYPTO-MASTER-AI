"""Crypto Master AI V3: feature specialists + a meta/head model.

Research-only. Does not connect to Scanner.py, send alerts, or place trades.
Chronological per-file splits reduce look-ahead leakage; validation trains the
head model and the final holdout is used only for evaluation.
"""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
DATA_DIR = Path("training_data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "crypto_ai_model_v3.joblib"
REPORT_PATH = MODEL_DIR / "ai_training_report_v3.json"
HORIZON = 3
MAX_ROWS_PER_FILE = 20_000
TIMEFRAME_CODES = {"15m": 0, "1h": 1, "4h": 2, "1d": 3}
FEATURE_GROUPS = {
    "trend": ["return_", "ema_distance_", "macd_", "adx_", "trend_"],
    "momentum": ["rsi_", "stoch_", "roc_", "cci_", "williams_", "momentum_"],
    "volatility": ["atr_", "volatility_", "bb_", "range_", "true_range_", "realized_"],
    "volume_flow": ["volume_", "obv_", "mfi_", "cmf_", "vwap_", "accumulation_"],
    "candle_structure": ["body_", "upper_wick_", "lower_wick_", "close_position", "gap_", "inside_bar", "outside_bar", "engulf_", "doji_", "pin_bar"],
    "levels_context": ["high_distance_", "low_distance_", "support_", "resistance_", "breakout_", "drawdown_", "timeframe_code", "hour_", "weekday_", "regime_"],
}


def load_candles(path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"SKIP {path.name}: {exc}")
        return None
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"SKIP {path.name}: missing OHLCV")
        return None
    tcol = next((x for x in ("timestamp", "time", "datetime", "date") if x in df.columns), None)
    if tcol is None:
        print(f"SKIP {path.name}: timestamp missing")
        return None
    raw = df[tcol]
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().mean() > 0.8:
        unit = "ms" if num.dropna().median() > 100_000_000_000 else "s"
        df["_time"] = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    else:
        df["_time"] = pd.to_datetime(raw, utc=True, errors="coerce")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_time", *required])
    df = df[(df.open > 0) & (df.high > 0) & (df.low > 0) & (df.close > 0) & (df.volume >= 0)]
    df = df.sort_values("_time").drop_duplicates("_time", keep="last").reset_index(drop=True)
    return df if len(df) >= 300 else None


def make_features(df, timeframe):
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=df.index)
    ret = c.pct_change()
    for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48):
        f[f"return_{n}"] = c.pct_change(n)
        f[f"roc_{n}"] = c.pct_change(n)
    for n in (5, 9, 12, 20, 26, 50, 100, 200):
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"ema_distance_{n}"] = c / ema - 1
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_pct"] = macd / c
    f["macd_signal_pct"] = signal / c
    f["macd_hist_pct"] = (macd - signal) / c
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)
    f["rsi_7"] = (100 - 100 / (1 + (delta.clip(lower=0).ewm(alpha=1/7, adjust=False).mean() / (-delta.clip(upper=0)).ewm(alpha=1/7, adjust=False).mean().replace(0, np.nan)))).fillna(50)
    for n in (10, 14, 20):
        lowest, highest = l.rolling(n).min(), h.rolling(n).max()
        stoch = (c - lowest) / (highest - lowest).replace(0, np.nan)
        f[f"stoch_{n}"] = stoch
        f[f"williams_{n}"] = -100 * (highest - c) / (highest - lowest).replace(0, np.nan)
        typical = (h + l + c) / 3
        mean = typical.rolling(n).mean()
        mad = typical.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        f[f"cci_{n}"] = (typical - mean) / (0.015 * mad.replace(0, np.nan))
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    for n in (7, 14, 20, 21, 50):
        f[f"atr_{n}"] = tr.rolling(n).mean() / c
        f[f"volatility_{n}"] = ret.rolling(n).std()
        f[f"realized_volatility_{n}"] = np.sqrt((ret.pow(2)).rolling(n).sum())
    for n in (10, 20, 50):
        mid = c.rolling(n).mean()
        sd = c.rolling(n).std()
        f[f"bb_width_{n}"] = (4 * sd) / mid
        f[f"bb_position_{n}"] = (c - (mid - 2*sd)) / (4*sd).replace(0, np.nan)
        f[f"range_{n}"] = (h.rolling(n).max() - l.rolling(n).min()) / c
        avgvol = v.rolling(n).mean()
        f[f"volume_ratio_{n}"] = v / avgvol.replace(0, np.nan)
    f["true_range_pct"] = tr / c
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14 = tr.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    f["adx_14"] = dx.rolling(14).mean()
    f["trend_direction"] = np.sign(plus_di - minus_di)
    f["trend_strength"] = (c / c.ewm(span=50, adjust=False).mean() - 1).abs()
    candle_range = (h-l).replace(0, np.nan)
    f["body_pct"] = (c-o) / o
    f["body_range_ratio"] = (c-o) / candle_range
    f["upper_wick_pct"] = (h - pd.concat([o,c], axis=1).max(axis=1)) / c
    f["lower_wick_pct"] = (pd.concat([o,c], axis=1).min(axis=1) - l) / c
    f["close_position"] = (c-l) / candle_range
    f["gap_pct"] = (o-prev) / prev
    f["doji_score"] = (c-o).abs() / candle_range
    f["pin_bar_score"] = (h - pd.concat([o,c], axis=1).max(axis=1) + pd.concat([o,c], axis=1).min(axis=1) - l) / candle_range
    f["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)
    f["outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(float)
    f["bullish_engulf"] = ((c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))).astype(float)
    f["bearish_engulf"] = ((c < o) & (c.shift(1) > o.shift(1)) & (o >= c.shift(1)) & (c <= o.shift(1))).astype(float)
    for n in (10, 20, 50, 100):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"high_distance_{n}"] = c / hi - 1
        f[f"low_distance_{n}"] = c / lo - 1
        f[f"support_{n}"] = (c - lo) / c
        f[f"resistance_{n}"] = (hi - c) / c
        f[f"breakout_up_{n}"] = (c > hi.shift(1)).astype(float)
        f[f"breakout_down_{n}"] = (c < lo.shift(1)).astype(float)
        peak = c.rolling(n).max()
        f[f"drawdown_{n}"] = c / peak - 1
    direction = np.sign(ret.fillna(0))
    signed_vol = direction * v
    f["obv_slope_10"] = signed_vol.rolling(10).sum() / v.rolling(10).sum().replace(0, np.nan)
    f["obv_slope_30"] = signed_vol.rolling(30).sum() / v.rolling(30).sum().replace(0, np.nan)
    typical = (h+l+c)/3
    raw_money = typical * v
    positive = raw_money.where(typical.diff() > 0, 0).rolling(14).sum()
    negative = raw_money.where(typical.diff() < 0, 0).rolling(14).sum()
    f["mfi_14"] = (100 - 100/(1 + positive / negative.replace(0, np.nan))).fillna(50)
    mf_multiplier = ((c-l) - (h-c)) / candle_range
    f["cmf_20"] = (mf_multiplier * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    f["accumulation_distribution"] = (mf_multiplier.fillna(0) * v).cumsum() / (v.cumsum().replace(0, np.nan))
    f["vwap_distance_20"] = c / ((typical*v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)) - 1
    f["volume_change_1"] = v.pct_change().replace([np.inf,-np.inf], np.nan)
    f["volume_zscore_20"] = (v-v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    f["timeframe_code"] = TIMEFRAME_CODES[timeframe]
    f["hour_utc"] = df["_time"].dt.hour
    f["weekday_utc"] = df["_time"].dt.dayofweek
    f["hour_sin"] = np.sin(2*np.pi*f["hour_utc"]/24)
    f["hour_cos"] = np.cos(2*np.pi*f["hour_utc"]/24)
    f["weekday_sin"] = np.sin(2*np.pi*f["weekday_utc"]/7)
    f["weekday_cos"] = np.cos(2*np.pi*f["weekday_utc"]/7)
    f["regime_trend"] = np.sign(c / c.ewm(span=100, adjust=False).mean() - 1)
    f["regime_volatility"] = f["volatility_20"] / f["volatility_50"].replace(0, np.nan)
    future_return = c.shift(-HORIZON) / c - 1
    threshold = np.maximum(0.0015, f["atr_14"].fillna(0) * 0.5)
    f["label"] = np.where(future_return > threshold, 1, np.where(future_return < -threshold, 0, np.nan))
    f["_position"] = np.arange(len(df))
    return f.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"]).copy()


def build_model(kind, seed=42):
    if kind == "logistic":
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed))])
    elif kind == "random_forest":
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=25, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))])
    elif kind == "extra_trees":
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(n_estimators=50, max_depth=10, min_samples_leaf=25, class_weight="balanced", n_jobs=-1, random_state=seed))])
    else:
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=80, learning_rate=0.06, l2_regularization=1.5, random_state=seed))])
    return estimator


def main():
    print("CRYPTO MASTER AI V3 — 60+ features, specialist teams, head model")
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
        a, b = int(n*0.60), int(n*0.80)
        pos = prepared["_position"]
        parts = [prepared[pos < a-HORIZON], prepared[(pos >= a) & (pos < b-HORIZON)], prepared[pos >= b]]
        if any(len(x) < 100 or x.label.nunique() < 2 for x in parts):
            skipped.append(path.name)
            print(f"SKIP {path.name}: split too small or only one class")
            continue
        def cap(frame, limit):
            return frame if len(frame) <= limit else frame.iloc[np.linspace(0, len(frame)-1, limit).astype(int)]
        train_parts.append(cap(parts[0], MAX_ROWS_PER_FILE))
        val_parts.append(cap(parts[1], 5_000))
        test_parts.append(cap(parts[2], 5_000))
        used.append(path.name)
        print(f"{path.name}: train={len(parts[0])} val={len(parts[1])} test={len(parts[2])}")
    if not train_parts:
        raise RuntimeError("No usable CSV splits")
    train, val, test = (pd.concat(x, ignore_index=True) for x in (train_parts, val_parts, test_parts))
    all_features = [x for x in train.columns if x not in ("label", "_position")]
    groups = {name: [f for f in all_features if any(f.startswith(prefix) if prefix.endswith("_") else (f == prefix or f.startswith(prefix+"_")) for prefix in prefixes)] for name, prefixes in FEATURE_GROUPS.items()}
    groups = {name: cols for name, cols in groups.items() if len(cols) >= 3}
    if len(all_features) < 60 or len(groups) < 5:
        raise RuntimeError(f"Feature coverage check failed: {len(all_features)} features, {len(groups)} groups")
    y_train, y_val, y_test = (x.label.astype(int) for x in (train, val, test))
    specialists, specialist_meta = {}, {}
    val_meta = pd.DataFrame(index=val.index)
    test_meta = pd.DataFrame(index=test.index)
    model_kinds = ("logistic", "random_forest", "extra_trees", "hist_gradient_boosting")
    for group, cols in groups.items():
        print(f"Specialist {group}: {len(cols)} features")
        fitted = []
        for kind in model_kinds:
            key = f"{group}__{kind}"
            model = build_model(kind)
            model.fit(train[cols].astype(float), y_train)
            val_prob = model.predict_proba(val[cols].astype(float))[:, list(model.classes_).index(1)]
            test_prob = model.predict_proba(test[cols].astype(float))[:, list(model.classes_).index(1)]
            val_meta[key], test_meta[key] = val_prob, test_prob
            specialists[key] = {"model": model, "features": cols, "group": group, "kind": kind}
            fitted.append(key)
        specialist_meta[group] = fitted
        val_meta[f"{group}__mean"] = val_meta[fitted].mean(axis=1)
        test_meta[f"{group}__mean"] = test_meta[fitted].mean(axis=1)
    head_features = list(val_meta.columns)
    head = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("head", LogisticRegression(max_iter=800, class_weight="balanced", random_state=17))])
    head.fit(val_meta[head_features], y_val)
    pred = head.predict(test_meta[head_features])
    prob = head.predict_proba(test_meta[head_features])[:, list(head.classes_).index(1)]
    metrics = {"accuracy": float(accuracy_score(y_test, pred)), "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)), "long_precision": float(precision_score(y_test, pred, pos_label=1, zero_division=0)), "short_precision": float(precision_score(y_test, pred, pos_label=0, zero_division=0)), "long_recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)), "short_recall": float(recall_score(y_test, pred, pos_label=0, zero_division=0))}
    metrics["roc_auc"] = float(roc_auc_score(y_test, prob)) if y_test.nunique() == 2 else None
    report = {
        "model_version": "v3-specialist-head-experimental",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_count": len(all_features), "feature_names": all_features,
        "feature_groups": {k: {"count": len(v), "features": v} for k,v in groups.items()},
        "specialist_model_count": len(specialists), "specialist_models": list(specialists.keys()),
        "head_model": "StandardScaler + LogisticRegression trained on validation-period specialist predictions",
        "split": {"train": "first 60% per file, purged by horizon", "validation": "next 20% per file; trains head model", "test": "final 20% per file; untouched for final evaluation"},
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "test_metrics": metrics, "files_used": used, "files_skipped": skipped,
        "horizon_candles": HORIZON, "timeframe_codes": TIMEFRAME_CODES,
        "limitations": ["Experimental directional classifier, not proof of profitability.", "Head learns from validation-period specialist outputs; this is a temporal holdout stack, not cross-fitted stacking.", "Trading fees, slippage, funding, liquidation, position sizing and execution are not modeled in training metrics.", "Not integrated into Scanner.py."],
        "scikit_learn_version": sklearn.__version__,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model_version": "v3-specialist-head-experimental", "specialists": specialists, "specialist_groups": specialist_meta, "head_model": head, "head_features": head_features, "feature_groups": groups, "features": all_features, "timeframe_codes": TIMEFRAME_CODES, "horizon": HORIZON, "classes": {0:"SHORT",1:"LONG"}}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("\nV3 HELD-OUT TEST REPORT")
    print(json.dumps({"features_count":len(all_features),"specialist_model_count":len(specialists),"test_metrics":metrics,"files_used":len(used),"files_skipped":len(skipped)}, indent=2))
    print(f"Saved {MODEL_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
