"""
Train ARIMA baseline on the same wind farm data pipeline as deep learning models.

ARIMA is a classic univariate time series model. It uses only the historical
power output to predict future values, making it a natural statistical baseline
for comparison with CNN, LSTM, and CNN-LSTM.

Approach:
  1. Load & clean data identically to the DL pipeline
  2. Fit ARIMA on training set power series only
  3. Append validation observations to update model state (no refit)
  4. Rolling forecast through test set: predict LF steps ahead, then
     append actual observation and advance
  5. Compute metrics (MSE, RMSE, MAE, R²) in the same format as DL results

Usage:
    python train_arima.py
"""
import csv
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# --------------- Configuration -------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "数据清洗部分"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "result" / "figure_csv" / "arima_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Power (MW)"
LOOKBACK = 96
MAX_LF = 6
LOOK_FORWARDS = list(range(1, MAX_LF + 1))

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

RANDOM_SEED = 42

# ARIMA order: (p, d, q)
# p=5: AR terms covering ~75 min at 15-min resolution
# d=1: first differencing for non-stationarity
# q=2: MA terms
ARIMA_ORDER = (5, 1, 2)

# For speed: use only last N training points to fit ARIMA
# Full 49k points can be very slow; 10k captures ~100 days of patterns
ARIMA_TRAIN_WINDOW = 10000

SITE_FILES = [
    "Wind farm site 1 (Nominal capacity-99MW).xlsx",
    "Wind farm site 2 (Nominal capacity-200MW).xlsx",
    "Wind farm site 3 (Nominal capacity-99MW).xlsx",
    "Wind farm site 4 (Nominal capacity-66MW).xlsx",
    "Wind farm site 5 (Nominal capacity-36MW).xlsx",
    "Wind farm site 6 (Nominal capacity-96MW).xlsx",
]


def normalize_column_name(col_name):
    col_name = str(col_name).strip().lower()
    col_name = col_name.replace("\n", " ")
    col_name = re.sub(r"\s+", " ", col_name)
    return col_name


def infer_columns(columns):
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


def clean_dataframe(df, time_col, target_col):
    """Same cleaning as data_utils.clean_site_dataframe."""
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


def fit_arima_for_site(power_series, train_end, val_end):
    """
    Fit ARIMA on the training portion, then append val data to update state.
    Returns the fitted model ready for test-set rolling forecast.
    """
    # Use the last ARIMA_TRAIN_WINDOW points of training data for speed
    train_start = max(0, train_end - ARIMA_TRAIN_WINDOW)
    train_power = power_series[train_start:train_end].values.astype(float)

    print(f"  Fitting ARIMA{ARIMA_ORDER} on {len(train_power)} training points...")
    start_fit = time.time()

    model = ARIMA(train_power, order=ARIMA_ORDER)
    fit_result = model.fit(method_kwargs={"maxiter": 200})

    fit_time = time.time() - start_fit
    print(f"  ARIMA fit completed in {fit_time:.1f}s (AIC={fit_result.aic:.1f})")

    # Append validation data to update model state without refitting
    val_power = power_series[train_end:val_end].values.astype(float)
    print(f"  Appending {len(val_power)} validation observations...")
    start_append = time.time()
    fit_result = fit_result.append(val_power, refit=False)
    append_time = time.time() - start_append
    print(f"  Validation append done in {append_time:.1f}s")

    return fit_result, fit_time


def rolling_forecast(fit_result, test_power, max_lf, progress_interval=1000):
    """
    Rolling forecast through the test set.
    At each position: forecast max_lf steps ahead, then append actual observation.

    Returns: forecasts array of shape (n_test_points, max_lf)
    """
    n_test = len(test_power)
    forecasts = np.zeros((n_test, max_lf))

    print(f"  Rolling forecast through {n_test} test points...")
    start_roll = time.time()

    for i in range(n_test):
        # Forecast max_lf steps ahead
        fc = fit_result.forecast(steps=max_lf)
        forecasts[i, :] = fc

        # Append actual observation (update state, no refit)
        fit_result = fit_result.append([test_power[i]], refit=False)

        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_roll
            rate = (i + 1) / elapsed
            remaining = (n_test - i - 1) / rate
            print(f"    [{i+1}/{n_test}] {rate:.0f} pts/s, ~{remaining:.0f}s remaining")

    roll_time = time.time() - start_roll
    print(f"  Rolling forecast done in {roll_time:.1f}s ({n_test/roll_time:.0f} pts/s)")

    return forecasts, roll_time


def compute_metrics_for_lf(forecasts, test_power, lf):
    """
    Compute metrics for a specific look_forward value.
    Matches the DL evaluation: sliding window of LF consecutive predictions.
    """
    n_samples = len(test_power) - lf + 1
    if n_samples <= 0:
        return None

    # Build prediction and truth arrays matching DL shape (n_samples, lf)
    y_pred = np.zeros((n_samples, lf))
    y_true = np.zeros((n_samples, lf))

    for i in range(n_samples):
        y_pred[i, :] = forecasts[i, :lf]
        y_true[i, :] = test_power[i:i + lf]

    # Flatten and compute metrics (same as DL code)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "n_samples": n_samples}


def process_site(site_file):
    """Process one wind farm site: fit ARIMA and evaluate for all LF values."""
    file_path = DATA_DIR / site_file
    site_name = site_file.replace(".xlsx", "")

    print(f"\n{'='*70}")
    print(f"Site: {site_name}")
    print(f"{'='*70}")

    # Load and clean
    df = pd.read_excel(file_path)
    time_col, target_col = infer_columns(list(df.columns))
    df = clean_dataframe(df, time_col, target_col)

    power_series = df[target_col]
    total_size = len(power_series)
    train_end = int(total_size * TRAIN_RATIO)
    val_end = int(total_size * (TRAIN_RATIO + VAL_RATIO))

    print(f"  Total: {total_size}, Train: {train_end}, Val end: {val_end}, "
          f"Test: {total_size - val_end}")

    # Fit ARIMA
    fit_result, fit_time = fit_arima_for_site(power_series, train_end, val_end)

    # Test data
    test_power = power_series[val_end:].values.astype(float)

    # Rolling forecast (predict MAX_LF steps at each position)
    forecasts, roll_time = rolling_forecast(fit_result, test_power, MAX_LF)

    # Compute metrics for each LF
    results = []
    for lf in LOOK_FORWARDS:
        metrics = compute_metrics_for_lf(forecasts, test_power, lf)
        if metrics is None:
            continue

        result = {
            "site_name": site_name,
            "model_name": "ARIMA",
            "lookback": LOOKBACK,
            "look_forward": lf,
            "batch_size": "N/A",
            "learning_rate": "N/A",
            "hidden_size": "N/A",
            "num_layers": "N/A",
            "epochs": "N/A",
            "dropout": "N/A",
            "checkpoint_selection": "N/A",
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "train_time": fit_time,
        }
        results.append(result)
        print(f"  LF={lf}: MSE={metrics['mse']:.3f}, RMSE={metrics['rmse']:.3f}, "
              f"MAE={metrics['mae']:.3f}, R²={metrics['r2']:.4f} "
              f"(n={metrics['n_samples']})")

    return results


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    np.random.seed(RANDOM_SEED)
    print(f"ARIMA Order: {ARIMA_ORDER}")
    print(f"Training window: {ARIMA_TRAIN_WINDOW} points")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    all_site_results = {}  # site_idx -> list of results for all LFs

    for site_idx, site_file in enumerate(SITE_FILES):
        results = process_site(site_file)
        all_site_results[site_idx + 1] = results

    # ---- Save site1 lookforward metrics ----
    site1_results = all_site_results[1]
    site1_csv = OUTPUT_DIR / "site1_lookforward_metrics_lb96_arima.csv"
    fieldnames = list(site1_results[0].keys())
    with open(site1_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(site1_results)
    print(f"\nSaved: {site1_csv}")

    # ---- Save cross-site metrics for each LF ----
    for lf in LOOK_FORWARDS:
        lf_rows = []
        for site_idx in range(1, 7):
            for r in all_site_results[site_idx]:
                if r["look_forward"] == lf:
                    row = r.copy()
                    row["site_id"] = site_idx
                    lf_rows.append(row)
                    break

        cross_csv = OUTPUT_DIR / f"cross_site_metrics_lf{lf}_arima.csv"
        fieldnames_cross = list(lf_rows[0].keys())
        with open(cross_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_cross)
            writer.writeheader()
            writer.writerows(lf_rows)
        print(f"Saved: {cross_csv}")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("ARIMA BASELINE SUMMARY")
    print("=" * 80)

    print("\n--- Site 1 Metrics (LF=1 to LF=6) ---")
    print(f"{'LF':>4} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    for r in site1_results:
        print(f"{r['look_forward']:>4} {r['mse']:>10.3f} {r['rmse']:>10.3f} "
              f"{r['mae']:>10.3f} {r['r2']:>10.4f}")

    print("\n--- Cross-Site Average (all 6 farms) ---")
    for lf in LOOK_FORWARDS:
        lf_rows = []
        for site_idx in range(1, 7):
            for r in all_site_results[site_idx]:
                if r["look_forward"] == lf:
                    lf_rows.append(r)
                    break
        if lf_rows:
            avg_mse = np.mean([r["mse"] for r in lf_rows])
            avg_rmse = np.mean([r["rmse"] for r in lf_rows])
            avg_mae = np.mean([r["mae"] for r in lf_rows])
            avg_r2 = np.mean([r["r2"] for r in lf_rows])
            print(f"  LF={lf}: Avg MSE={avg_mse:.3f}, Avg RMSE={avg_rmse:.3f}, "
                  f"Avg MAE={avg_mae:.3f}, Avg R²={avg_r2:.4f}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
