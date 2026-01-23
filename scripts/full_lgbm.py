import polars as pl
import numpy as np
from datetime import timedelta
from lightgbm import LGBMRegressor

DF_PATH = "data_lake/gold/feature_store_2023.parquet"

TRAIN_DAYS = 180
TEST_DAYS = 14
STEP_DAYS = 14

FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "t2m_c", "wind_ms", "rh_pct", "msl_hpa", "tp_mm",
    "lag_1", "lag_24", "lag_168",
    "roll_mean_24", "roll_mean_168",
]
TARGET = "demand_mw"

def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    mape = np.mean(np.abs((y - p) / y)) * 100.0
    return mae, rmse, mape

df = pl.read_parquet(DF_PATH).sort("ts_utc")

# Baseline prediction for comparison
df = df.with_columns(pl.col(TARGET).shift(168).alias("y_pred_baseline"))

# Drop rows that cannot be used for model training/prediction
df_model = df.drop_nulls([TARGET] + FEATURES)

min_ts = df_model.select(pl.col("ts_utc").min()).item()
max_ts = df_model.select(pl.col("ts_utc").max()).item()

start_test = min_ts + timedelta(days=TRAIN_DAYS)
end_limit = max_ts - timedelta(days=TEST_DAYS)

rows = []
k = 0
t = start_test

while t <= end_limit:
    train_start = t - timedelta(days=TRAIN_DAYS)
    train_end = t
    test_start = t
    test_end = t + timedelta(days=TEST_DAYS)

    train = df_model.filter((pl.col("ts_utc") >= train_start) & (pl.col("ts_utc") < train_end))
    test  = df_model.filter((pl.col("ts_utc") >= test_start) & (pl.col("ts_utc") < test_end))

    if train.height == 0 or test.height == 0:
        break

    X_train = train.select(FEATURES).to_pandas()
    y_train = train.select(TARGET).to_numpy().ravel()

    X_test = test.select(FEATURES).to_pandas()
    y_test = test.select(TARGET).to_numpy().ravel()

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=7,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae, rmse, mape = metrics(y_test, pred)

    # Baseline on the same test period
    baseline_test = df.filter((pl.col("ts_utc") >= test_start) & (pl.col("ts_utc") < test_end))
    baseline_test = baseline_test.filter(pl.col("y_pred_baseline").is_not_null() & pl.col(TARGET).is_not_null())
    yb = baseline_test[TARGET].to_numpy()
    pb = baseline_test["y_pred_baseline"].to_numpy()
    b_mae, b_rmse, b_mape = metrics(yb, pb)

    rows.append({
        "fold": k,
        "test_start": test_start,
        "test_end": test_end,
        "rows": test.height,
        "mae_baseline": b_mae,
        "mae_lgbm": mae,
        "mape_baseline": b_mape,
        "mape_lgbm": mape,
    })

    k += 1
    t = t + timedelta(days=STEP_DAYS)

res = pl.DataFrame(rows)

print(res)

print("\nSUMMARY (mean across folds)")
print(res.select([
    pl.col("mae_baseline").mean().alias("mae_baseline_mean"),
    pl.col("mae_lgbm").mean().alias("mae_lgbm_mean"),
    pl.col("mape_baseline").mean().alias("mape_baseline_mean"),
    pl.col("mape_lgbm").mean().alias("mape_lgbm_mean"),
]).row(0))

print("\nIMPROVEMENT (mean)")
print(res.select([
    (pl.col("mae_baseline") - pl.col("mae_lgbm")).mean().alias("mae_gain_mw"),
    (pl.col("mape_baseline") - pl.col("mape_lgbm")).mean().alias("mape_gain_pct"),
]).row(0))
