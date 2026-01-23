import pandas as pd # Using pandas to remove sklearn warning on run
import polars as pl
import numpy as np
from datetime import datetime
from lightgbm import LGBMRegressor

DF_PATH = "data_lake/gold/feature_store_2023.parquet"

FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "t2m_c", "wind_ms", "rh_pct", "msl_hpa", "tp_mm",
    "lag_1", "lag_24", "lag_168",
    "roll_mean_24", "roll_mean_168",
]

TARGET = "demand_mw"

df = pl.read_parquet(DF_PATH).sort("ts_utc")

# Drop rows with missing target or missing required features
df = df.drop_nulls([TARGET] + FEATURES)

# one fold similar to baseline fold 10 worst period
train_start = datetime(2023, 5, 28, 0, 0, 0)
train_end   = datetime(2023, 11, 24, 0, 0, 0)
test_start  = datetime(2023, 11, 24, 0, 0, 0)
test_end    = datetime(2023, 12, 8, 0, 0, 0)

train = df.filter((pl.col("ts_utc") >= train_start) & (pl.col("ts_utc") < train_end))
test  = df.filter((pl.col("ts_utc") >= test_start) & (pl.col("ts_utc") < test_end))

print("train rows:", train.height)
print("test rows :", test.height)

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

mae = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean((y_test - pred) ** 2))
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100.0

print("LGBM MAE  (MW):", float(mae))
print("LGBM RMSE (MW):", float(rmse))
print("LGBM MAPE  (%):", float(mape))

"""
--------------------------------------------------------------------------------------------------
OUTPUT:
--------------------------------------------------------------------------------------------------

train rows: 4320
test rows : 336

[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000362 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2592
[LightGBM] [Info] Number of data points in the train set: 4320, number of used features: 14
[LightGBM] [Info] Start training from score 23852.830266

LGBM MAE  (MW): 1502.1974388450194
LGBM RMSE (MW): 1927.382657962094
LGBM MAPE  (%): 4.2158339195167915

--------------------------------------------------------------------------------------------------


On the worst baseline fold (Nov 24 to Dec 8) Baseline MAE: 3527 MW, baseline MAPE: 10.80 %

Using LightGBM -> MAE: 1502 MW (very good improvement by approx 2000 MW)
MAPE: 4.22 % 

That is substantial

--------------------------------------------------------------------------------------------------
"""