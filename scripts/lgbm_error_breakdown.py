import polars as pl
import numpy as np
from datetime import timedelta
from lightgbm import LGBMRegressor

DF_PATH = "data_lake/gold/feature_store_2023.parquet"

FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "t2m_c", "wind_ms", "rh_pct", "msl_hpa", "tp_mm",
    "lag_1", "lag_24", "lag_168",
    "roll_mean_24", "roll_mean_168",
]
TARGET = "demand_mw"

df = (
    pl.read_parquet(DF_PATH)
    .sort("ts_utc")
    .drop_nulls([TARGET] + FEATURES)
)

max_ts = df.select(pl.col("ts_utc").max()).item()
split_ts = max_ts - timedelta(days=30)

train = df.filter(pl.col("ts_utc") < split_ts)
test  = df.filter(pl.col("ts_utc") >= split_ts)

print("max_ts   :", max_ts)
print("split_ts :", split_ts)
print("train rows:", train.height)
print("test rows :", test.height)

if test.height == 0:
    raise RuntimeError("Test split is empty. Check split_ts logic.")

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

pred_df = (
    test.select(["ts_utc", "demand_mw"])
    .with_columns([
        pl.Series("y_pred", pred),
        pl.col("ts_utc").dt.month().alias("month"),
        pl.col("ts_utc").dt.hour().alias("hour"),
    ])
    .with_columns([
        (pl.col("demand_mw") - pl.col("y_pred")).abs().alias("abs_err"),
    ])
)

by_month = pred_df.group_by("month").agg(pl.mean("abs_err").alias("mae_mw")).sort("month")
by_hour  = pred_df.group_by("hour").agg(pl.mean("abs_err").alias("mae_mw")).sort("hour")

print("\nMAE by month (test window):")
print(by_month)
print("\nMAE by hour (test window):")
print(by_hour)


"""
------------------------------------------------------------------------------------------------
OUTPUT:
------------------------------------------------------------------------------------------------

MAE by month (test window):
shape: (1, 2)
┌───────┬───────────┐
│ month ┆ mae_mw    │
│ ---   ┆ ---       │
│ i8    ┆ f64       │
╞═══════╪═══════════╡
│ 12    ┆ 561.22995 │
└───────┴───────────┘

MAE by hour (test window):
shape: (24, 2)
┌──────┬────────────┐
│ hour ┆ mae_mw     │
│ ---  ┆ ---        │
│ i8   ┆ f64        │
╞══════╪════════════╡
│ 0    ┆ 777.391987 │
│ 1    ┆ 547.24432  │
│ 2    ┆ 459.759681 │
│ 3    ┆ 389.927508 │
│ 4    ┆ 437.407912 │
│ …    ┆ …          │
│ 19   ┆ 780.443993 │
│ 20   ┆ 513.00993  │
│ 21   ┆ 464.160348 │
│ 22   ┆ 376.61694  │
│ 23   ┆ 425.273363 │
└──────┴────────────┘

------------------------------------------------------------------------------------------------

This is a believable demand forcast error profile:

- MAE for Decenber approx 561 MW (consistent with the overall ~ 596 MW mean)

- Error spikes at evening peaks (19:00) and midnight (00:00) with troughs in the

03:00 to 05:00 interval (makes sense since it the dead of night)

-------------------------------------------------------------------------------------------------
"""