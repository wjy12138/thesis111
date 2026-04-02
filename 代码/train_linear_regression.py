"""
Train Linear Regression (LR) baseline on the same wind farm data and pipeline
as the deep learning models (LSTM, CNN, CNN-LSTM).

Produces CSV output files in the same format as the existing results so they
can be directly compared in tables and figures.

Usage:
    python train_linear_regression.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --------------- Configuration (mirrors config.py) -------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "数据清洗部分"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "result" / "figure_csv" / "lr_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Time(year-month-day h:m:s)"
TARGET_COL = "Power (MW)"

LOOKBACK = 96
LOOK_FORWARDS = [1, 2, 3, 4, 5, 6]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # validation set exists but LR doesn't use it

RANDOM_SEED = 42

SITE_FILES = [
    "Wind farm site 1 (Nominal capacity-99MW).xlsx",
    "Wind farm site 2 (Nominal capacity-200MW).xlsx",
    "Wind farm site 3 (Nominal capacity-99MW).xlsx",
    "Wind farm site 4 (Nominal capacity-66MW).xlsx",
    "Wind farm site 5 (Nominal capacity-36MW).xlsx",
    "Wind farm site 6 (Nominal capacity-96MW).xlsx",
]


def normalize_column_name(col_name):
    import re
    col_name = str(col_name).strip().lower()
    col_name = col_name.replace("\n", " ")
    col_name = re.sub(r"\s+", " ", col_name)
    return col_name


def infer_columns(columns):
    """Find time and target columns, same logic as data_utils."""
    time_col = None
    target_col = None
    for col in columns:
        normalized = normalize_column_name(col)
        if "time" in normalized and time_col is None:
            time_col = col
        if "power" in normalized and target_col is None:
            target_col = col
    if time_col is None:
        time_col = columns[0]
    if target_col is None:
        target_col = columns[-1]
    return time_col, target_col


def add_time_features(df, time_col, target_col):
    """Same time features as data_utils.add_time_features."""
    df = df.copy()
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["dayofyear"] = df[time_col].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]
    return df, feature_cols


def clean_dataframe(df, time_col, target_col):
    """Minimal cleaning: same as data_utils.clean_site_dataframe."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    for col in df.columns:
        if col != time_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def create_sequences(features, targets, lookback, look_forward, start_idx, end_idx, use_history_target=True):
    """Same sliding window as data_utils.create_sequences."""
    x_list = []
    y_list = []

    for target_start_idx in range(start_idx, end_idx - look_forward + 1):
        input_start_idx = target_start_idx - lookback
        if input_start_idx < 0:
            continue

        x_feat_seq = features[input_start_idx:target_start_idx]
        if use_history_target:
            x_target_hist = targets[input_start_idx:target_start_idx].reshape(-1, 1)
            x_seq = np.concatenate([x_feat_seq, x_target_hist], axis=1)
        else:
            x_seq = x_feat_seq

        y_seq = targets[target_start_idx:target_start_idx + look_forward].reshape(-1)
        x_list.append(x_seq)
        y_list.append(y_seq)

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_lr_for_site_lf(site_file, look_forward):
    """Train Linear Regression for one site and one look_forward value."""
    file_path = DATA_DIR / site_file
    print(f"\n{'='*60}")
    print(f"Site: {site_file}, LF={look_forward}")
    print(f"{'='*60}")

    # Load and clean data
    df = pd.read_excel(file_path)
    time_col, target_col = infer_columns(list(df.columns))
    df = clean_dataframe(df, time_col, target_col)
    df, feature_cols = add_time_features(df, time_col, target_col)

    # Split
    total_size = len(df)
    train_end = int(total_size * TRAIN_RATIO)
    val_end = int(total_size * (TRAIN_RATIO + VAL_RATIO))

    # Scale (fit only on train)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_df = df.iloc[:train_end]
    feature_scaler.fit(train_df[feature_cols].values)
    target_scaler.fit(train_df[[target_col]].values)

    features_scaled = feature_scaler.transform(df[feature_cols].values)
    targets_scaled = target_scaler.transform(df[[target_col]].values)

    # Create sequences
    x_train, y_train = create_sequences(
        features_scaled, targets_scaled, LOOKBACK, look_forward,
        start_idx=LOOKBACK, end_idx=train_end, use_history_target=True
    )
    x_test, y_test = create_sequences(
        features_scaled, targets_scaled, LOOKBACK, look_forward,
        start_idx=val_end, end_idx=total_size, use_history_target=True
    )

    print(f"  x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"  x_test shape:  {x_test.shape},  y_test shape:  {y_test.shape}")

    # Flatten the 3D input (samples, lookback, features) -> (samples, lookback*features)
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    x_train_flat = x_train.reshape(n_train, -1)
    x_test_flat = x_test.reshape(n_test, -1)

    # Train Linear Regression
    start_time = time.time()
    lr_model = LinearRegression()
    lr_model.fit(x_train_flat, y_train)
    train_time = time.time() - start_time

    # Predict
    y_pred_scaled = lr_model.predict(x_test_flat)

    # Inverse transform to original scale
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

    # Compute metrics (flatten all horizons together, same as existing code)
    y_true_flat = y_test_orig.reshape(-1)
    y_pred_flat = y_pred_orig.reshape(-1)

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    print(f"  Results: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    print(f"  Train time: {train_time:.4f}s")

    return {
        "site_name": site_file.replace(".xlsx", ""),
        "model_name": "LR",
        "lookback": LOOKBACK,
        "look_forward": look_forward,
        "batch_size": "N/A",
        "learning_rate": "N/A",
        "hidden_size": "N/A",
        "num_layers": "N/A",
        "epochs": "N/A",
        "dropout": "N/A",
        "checkpoint_selection": "N/A",
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_time": train_time,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    np.random.seed(RANDOM_SEED)

    all_results = []

    # ---------- 1. Site1 across all LFs (matches site1_lookforward_metrics) ----------
    site1_results = []
    for lf in LOOK_FORWARDS:
        result = train_lr_for_site_lf(SITE_FILES[0], lf)
        site1_results.append(result)
        all_results.append(result)

    # Save site1 lookforward metrics
    site1_csv = OUTPUT_DIR / "site1_lookforward_metrics_lb96_lr.csv"
    fieldnames = list(site1_results[0].keys())
    with open(site1_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(site1_results)
    print(f"\nSaved: {site1_csv}")

    # ---------- 2. Cross-site for each LF (matches cross_site_metrics_lf*.csv) ----------
    for lf in LOOK_FORWARDS:
        lf_results = []
        for site_idx, site_file in enumerate(SITE_FILES):
            # Check if already computed for site1
            if site_idx == 0:
                result = [r for r in site1_results if r["look_forward"] == lf][0].copy()
            else:
                result = train_lr_for_site_lf(site_file, lf)
                all_results.append(result)
            result["site_id"] = site_idx + 1
            lf_results.append(result)

        cross_csv = OUTPUT_DIR / f"cross_site_metrics_lf{lf}_lr.csv"
        fieldnames_cross = list(lf_results[0].keys())
        with open(cross_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_cross)
            writer.writeheader()
            writer.writerows(lf_results)
        print(f"Saved: {cross_csv}")

    # ---------- 3. Summary table ----------
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION BASELINE SUMMARY")
    print("=" * 80)

    # Print Site1 metrics table
    print("\n--- Site 1 Metrics (LF=1 to LF=6) ---")
    print(f"{'LF':>4} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Time(s)':>10}")
    for r in site1_results:
        print(f"{r['look_forward']:>4} {r['mse']:>10.3f} {r['rmse']:>10.3f} {r['mae']:>10.3f} {r['r2']:>10.4f} {r['train_time']:>10.4f}")

    # Print cross-site averages for each LF
    print("\n--- Cross-Site Average (all 6 farms) ---")
    for lf in LOOK_FORWARDS:
        lf_rows = [r for r in all_results if r["look_forward"] == lf]
        # Deduplicate: keep one per site
        seen_sites = set()
        unique_rows = []
        for r in lf_rows:
            if r["site_name"] not in seen_sites:
                seen_sites.add(r["site_name"])
                unique_rows.append(r)
        if unique_rows:
            avg_mse = np.mean([r["mse"] for r in unique_rows])
            avg_mae = np.mean([r["mae"] for r in unique_rows])
            avg_r2 = np.mean([r["r2"] for r in unique_rows])
            print(f"  LF={lf}: Avg MSE={avg_mse:.3f}, Avg MAE={avg_mae:.3f}, Avg R²={avg_r2:.4f}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
