import polars as pl
from datetime import timedelta

DF_PATH = "data_lake/gold/feature_store_2023.parquet"
"""
-------------------------------------------------------------------------------------------
Walk-forward backtest
-------------------------------------------------------------------------------------------

Some default parameters:

- Training window: 180 days
- Test horizon: 14 days
- steps: 14 days (should have no overlapping tests)

Results in multiple folds over the entire temporal span of 2023

Note: Is model agonstic so models (LightGBM/XGBoost) can be implemented easily

-------------------------------------------------------------------------------------------
"""
TRAIN_DAYS = 180
TEST_DAYS = 14
STEP_DAYS = 14

df = pl.read_parquet(DF_PATH).sort("ts_utc")

# backtest where we have all required baseline inputs
df = df.with_columns([
    pl.col("demand_mw").shift(168).alias("y_pred_baseline"),
])

# Extract candidate evaluation rows (exclude null target/pred)
eval_df = df.filter(
    pl.col("demand_mw").is_not_null() & pl.col("y_pred_baseline").is_not_null()
).select(["ts_utc", "demand_mw", "y_pred_baseline"])

min_ts = eval_df.select(pl.col("ts_utc").min()).item()
max_ts = eval_df.select(pl.col("ts_utc").max()).item()

start_test = min_ts + timedelta(days=TRAIN_DAYS)
end_limit = max_ts - timedelta(days=TEST_DAYS)

folds = []
k = 0
t = start_test
while t <= end_limit:
    train_start = t - timedelta(days=TRAIN_DAYS)
    train_end = t 
    test_start = t
    test_end = t + timedelta(days=TEST_DAYS)

    folds.append((k, train_start, train_end, test_start, test_end))
    k += 1
    t = t + timedelta(days=STEP_DAYS)

print("min_ts:", min_ts)
print("max_ts:", max_ts)
print("n_folds:", len(folds))

print("\nFirst 5 folds:")
for row in folds[:5]:
    print(row)

print("\nLast 2 folds:")
for row in folds[-2:]:
    print(row)

"""
-------------------------------------------------------------------------------------------
OUTPUT:
-------------------------------------------------------------------------------------------

min_ts: 2023-01-08 00:00:00
max_ts: 2023-12-31 23:00:00
n_folds: 12

First 5 folds:
(0, datetime.datetime(2023, 1, 8, 0, 0), datetime.datetime(2023, 7, 7, 0, 0), datetime.datetime(2023, 7, 7, 0, 0), datetime.datetime(2023, 7, 21, 0, 0))
(1, datetime.datetime(2023, 1, 22, 0, 0), datetime.datetime(2023, 7, 21, 0, 0), datetime.datetime(2023, 7, 21, 0, 0), datetime.datetime(2023, 8, 4, 0, 0))
(2, datetime.datetime(2023, 2, 5, 0, 0), datetime.datetime(2023, 8, 4, 0, 0), datetime.datetime(2023, 8, 4, 0, 0), datetime.datetime(2023, 8, 18, 0, 0))
(3, datetime.datetime(2023, 2, 19, 0, 0), datetime.datetime(2023, 8, 18, 0, 0), datetime.datetime(2023, 8, 18, 0, 0), datetime.datetime(2023, 9, 1, 0, 0))
(4, datetime.datetime(2023, 3, 5, 0, 0), datetime.datetime(2023, 9, 1, 0, 0), datetime.datetime(2023, 9, 1, 0, 0), datetime.datetime(2023, 9, 15, 0, 0))

Last 2 folds:
(10, datetime.datetime(2023, 5, 28, 0, 0), datetime.datetime(2023, 11, 24, 0, 0), datetime.datetime(2023, 11, 24, 0, 0), datetime.datetime(2023, 12, 8, 0, 0))
(11, datetime.datetime(2023, 6, 11, 0, 0), datetime.datetime(2023, 12, 8, 0, 0), datetime.datetime(2023, 12, 8, 0, 0), datetime.datetime(2023, 12, 22, 0, 0))

-------------------------------------------------------------------------------------------

12 folds with clear boundaries, no leakage (train is always before test), with folds going
forward in 2 week chunks, so this is all in order as intended

--------------------------------------------------------------------------------------------
"""