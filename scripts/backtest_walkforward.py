import polars as pl
import numpy as np
from datetime import timedelta

DF_PATH = "data_lake/gold/feature_store_2023.parquet"

TRAIN_DAYS = 180
TEST_DAYS = 14
STEP_DAYS = 14

df = (
    pl.read_parquet(DF_PATH)
    .sort("ts_utc")
    .with_columns(pl.col("demand_mw").shift(168).alias("y_pred_baseline"))
)

# only rows where y and baseline exist
eval_df = df.filter(
    pl.col("demand_mw").is_not_null() & pl.col("y_pred_baseline").is_not_null()
).select(["ts_utc", "demand_mw", "y_pred_baseline"])

min_ts = eval_df.select(pl.col("ts_utc").min()).item()
max_ts = eval_df.select(pl.col("ts_utc").max()).item()

start_test = min_ts + timedelta(days=TRAIN_DAYS)
end_limit = max_ts - timedelta(days=TEST_DAYS)

def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    mape = np.mean(np.abs((y - p) / y)) * 100.0
    return mae, rmse, mape

rows = []
k = 0
t = start_test

while t <= end_limit:
    test_start = t
    test_end = t + timedelta(days=TEST_DAYS)

    test = eval_df.filter((pl.col("ts_utc") >= test_start) & (pl.col("ts_utc") < test_end))
    if test.height == 0:
        break

    y = test["demand_mw"].to_numpy()
    p = test["y_pred_baseline"].to_numpy()
    mae, rmse, mape = metrics(y, p)

    rows.append({
        "fold": k,
        "test_start": test_start,
        "test_end": test_end,
        "rows": test.height,
        "mae_mw": mae,
        "rmse_mw": rmse,
        "mape_pct": mape,
    })

    k += 1
    t = t + timedelta(days=STEP_DAYS)

res = pl.DataFrame(rows)

print(res)

# summary across folds
print("\nSUMMARY (mean across folds)")
print(res.select([
    pl.col("mae_mw").mean().alias("mae_mw_mean"),
    pl.col("rmse_mw").mean().alias("rmse_mw_mean"),
    pl.col("mape_pct").mean().alias("mape_pct_mean"),
]).row(0))

# Worst fold (by MAE)
worst = res.sort("mae_mw", descending=True).head(1)
print("\nWORST FOLD (highest MAE)")
print(worst)



"""
-----------------------------------------------------------------------------------------------

OUTPUT:

-----------------------------------------------------------------------------------------------


shape: (12, 7)
┌──────┬─────────────┬────────────┬──────┬────────────┬────────────┬───────────┐
│ fold ┆ test_start  ┆ test_end   ┆ rows ┆ mae_mw     ┆ rmse_mw    ┆ mape_pct  │
│ ---  ┆ ---         ┆ ---        ┆ ---  ┆ ---        ┆ ---        ┆ ---       │
│ i64  ┆ datetime[μs ┆ datetime[μ ┆ i64  ┆ f64        ┆ f64        ┆ f64       │
│      ┆ ]           ┆ s]         ┆      ┆            ┆            ┆           │
╞══════╪═════════════╪════════════╪══════╪════════════╪════════════╪═══════════╡
│ 0    ┆ 2023-07-07  ┆ 2023-07-21 ┆ 336  ┆ 1780.42261 ┆ 2352.94965 ┆ 8.276117  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 9          ┆ 1          ┆           │
│ 1    ┆ 2023-07-21  ┆ 2023-08-04 ┆ 336  ┆ 1392.09672 ┆ 1948.21549 ┆ 6.473591  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 6          ┆            ┆           │
│ 2    ┆ 2023-08-04  ┆ 2023-08-18 ┆ 336  ┆ 1214.86458 ┆ 1748.83315 ┆ 5.527664  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 3          ┆ 1          ┆           │
│ 3    ┆ 2023-08-18  ┆ 2023-09-01 ┆ 336  ┆ 1150.12202 ┆ 1566.13825 ┆ 5.136591  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 4          ┆ 2          ┆           │
│ 4    ┆ 2023-09-01  ┆ 2023-09-15 ┆ 336  ┆ 1773.51339 ┆ 2192.64180 ┆ 7.357772  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 3          ┆ 9          ┆           │
│ …    ┆ …           ┆ …          ┆ …    ┆ …          ┆ …          ┆ …         │
│ 7    ┆ 2023-10-13  ┆ 2023-10-27 ┆ 336  ┆ 1744.65178 ┆ 2180.24624 ┆ 6.663094  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 6          ┆ 6          ┆           │
│ 8    ┆ 2023-10-27  ┆ 2023-11-10 ┆ 336  ┆ 1315.60565 ┆ 1643.58391 ┆ 4.666097  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 5          ┆ 4          ┆           │
│ 9    ┆ 2023-11-10  ┆ 2023-11-24 ┆ 336  ┆ 1910.47916 ┆ 2393.37463 ┆ 6.75082   │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 7          ┆            ┆           │
│ 10   ┆ 2023-11-24  ┆ 2023-12-08 ┆ 336  ┆ 3527.48214 ┆ 4140.50380 ┆ 10.803985 │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 3          ┆ 2          ┆           │
│ 11   ┆ 2023-12-08  ┆ 2023-12-22 ┆ 336  ┆ 2808.66964 ┆ 3362.96892 ┆ 9.667079  │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 3          ┆ 4          ┆           │
└──────┴─────────────┴────────────┴──────┴────────────┴────────────┴───────────┘

SUMMARY (mean across folds)
(1803.3058035714284, 2281.614035669984, 7.036612414530398)

WORST FOLD (highest MAE)
shape: (1, 7)
┌──────┬─────────────┬────────────┬──────┬────────────┬────────────┬───────────┐
│ fold ┆ test_start  ┆ test_end   ┆ rows ┆ mae_mw     ┆ rmse_mw    ┆ mape_pct  │
│ ---  ┆ ---         ┆ ---        ┆ ---  ┆ ---        ┆ ---        ┆ ---       │
│ i64  ┆ datetime[μs ┆ datetime[μ ┆ i64  ┆ f64        ┆ f64        ┆ f64       │
│      ┆ ]           ┆ s]         ┆      ┆            ┆            ┆           │
╞══════╪═════════════╪════════════╪══════╪════════════╪════════════╪═══════════╡
│ 10   ┆ 2023-11-24  ┆ 2023-12-08 ┆ 336  ┆ 3527.48214 ┆ 4140.50380 ┆ 10.803985 │
│      ┆ 00:00:00    ┆ 00:00:00   ┆      ┆ 3          ┆ 2          ┆           │
└──────┴─────────────┴────────────┴──────┴────────────┴────────────┴───────────┘


-----------------------------------------------------------------------------------------------
Baseline is strongest in the later part of summer, and is minimized in early winter,
which echos the volitility in demand and shows seasonal weather sensitivity.

This is an appropriate baseline to evaluate models against 
-----------------------------------------------------------------------------------------------
"""