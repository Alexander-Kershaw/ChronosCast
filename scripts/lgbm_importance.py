import polars as pl
import numpy as np
from datetime import timedelta
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance

GOLD_PATH = "data_lake/gold/feature_store_2023.parquet"

FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "t2m_c", "wind_ms", "rh_pct", "msl_hpa", "tp_mm",
    "lag_1", "lag_24", "lag_168",
    "roll_mean_24", "roll_mean_168",
]
TARGET = "demand_mw"

df = (
    pl.read_parquet(GOLD_PATH)
    .sort("ts_utc")
    .drop_nulls([TARGET] + FEATURES)
)

max_ts = df.select(pl.col("ts_utc").max()).item()
split_ts = max_ts - timedelta(days=30)

train = df.filter(pl.col("ts_utc") < split_ts)
test  = df.filter(pl.col("ts_utc") >= split_ts)

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

# Permutation importance using MAE
result = permutation_importance(
    model, X_test, y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=7,
)

importances = result.importances_mean  # this is in "score units" (negative MAE)
# Convert to MAE increase in MW: baseline MAE - permuted MAE (negative sign flips)
mae_increase = -importances

order = np.argsort(mae_increase)[::-1]

print("Permutation importance (MAE increase in MW when permuted):\n")
for i in order[:15]:
    v = mae_increase[i]
    sign = "+" if v >= 0 else ""
    print(f"{FEATURES[i]:>14s}  {sign}{v:.2f} MW") 


"""
----------------------------------------------------------------------------------------------------
OUTPUT:
----------------------------------------------------------------------------------------------------

        rh_pct  +5.77 MW
         month  +-0.00 MW
    is_weekend  -2.85 MW
       msl_hpa  -4.84 MW
 roll_mean_168  -4.98 MW
         t2m_c  -12.09 MW
         tp_mm  -16.22 MW
       wind_ms  -34.65 MW
  roll_mean_24  -60.03 MW
           dow  -109.36 MW
       lag_168  -494.79 MW
        lag_24  -515.00 MW
          hour  -984.30 MW
         lag_1  -5569.20 MW

-----------------------------------------------------------------------------------------------------         

What is driving UK energy demand:

- lag_1 is dominant, with hour, lag_24, lag_168, being secondary dominators of importance.
This is good, it means the model is predominantly using a very stable causative structure

- weather features matter, however they are not of massively significant importance relative
to demand inertia and schedules. This makes a lot of sense and seems realistic.


- month, msl_hpa, roll_mean_168, and is_weekend are essentially redundant, this is probably
because dow + and lags encode most of it already.

-----------------------------------------------------------------------------------------------------
"""