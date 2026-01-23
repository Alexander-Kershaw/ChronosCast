import polars as pl
import numpy as np

"""
---------------------------------------------------------------------------------
Backtesting framework
---------------------------------------------------------------------------------

Random train/test split is an inccorect approach since a time series has
a temporal ordering. Therefore, a random split would cause randomosed train/test
samples to be taken across the entire temporal range. This essentially means
the model would see future patterns (with demand seasonal fluctuations cause
a change in the overarching time-energy demand pattern), meaning the
performance becomes invalid.

Implementation: A leakage resilient rolling validation process
 - training is done using past temporal data
 - a consecitive chink is predicted using the previous chunk
 - this porcess is rolled forward and repeated


Baseline: The lie detector test

- example: seasonaly naivety -> demand(t) ~ demand(t-168) (same hour last week)
if temporally shifted timestamps show a consistency regardless of temporal shifts
then it implies that the feature engineering or model set up is incorrect.

---------------------------------------------------------------------------------
"""

GOLD_FEATURE_STR_PATH = "data_lake/gold/feature_store_2023.parquet"

# Gold feature store polars df
df = (
    pl.read_parquet(GOLD_FEATURE_STR_PATH)
    .sort("ts_utc")
)

# Baseline for same hour last week
df = df.with_columns(
    pl.col("demand_mw").shift(168).alias("y_pred_baseline")
)

# Evaluate for only where both y and pred exist
eval_df = df.filter(
    pl.col("demand_mw").is_not_null() & pl.col("y_pred_baseline").is_not_null()
)

y = eval_df["demand_mw"].to_numpy()
p = eval_df["y_pred_baseline"].to_numpy()

# Performance baseline
mae = np.mean(np.abs(y - p))
rmse = np.sqrt(np.mean((y - p) ** 2))
mape = np.mean(np.abs((y - p) / y)) * 100.0

print("rows_total:", df.height)
print("rows_eval :", eval_df.height)
print("MAE  (MW):", float(mae))
print("RMSE (MW):", float(rmse))
print("MAPE  (%):", float(mape))

# Error breakdown by hour of the day
by_hour = (
    eval_df
    .with_columns((pl.col("demand_mw") - pl.col("y_pred_baseline")).abs().alias("abs_err"))
    .group_by(pl.col("ts_utc").dt.hour().alias("hour"))
    .agg(pl.mean("abs_err").alias("mae_mw"))
    .sort("hour")
)

print("\nMAE by hour (first 10 rows):")
print(by_hour.head(10))


"""
OUTPUT:

rows_total: 8760
rows_eval : 8590
MAE  (MW): 2025.1742142025612
RMSE (MW): 2698.9473691282337
MAPE  (%): 7.840789083119592

MAE by hour (first 10 rows):
shape: (10, 2)
┌──────┬─────────────┐
│ hour ┆ mae_mw      │
│ ---  ┆ ---         │
│ i8   ┆ f64         │
╞══════╪═════════════╡
│ 0    ┆ 1482.448324 │
│ 1    ┆ 1501.494413 │
│ 2    ┆ 1552.238827 │
│ 3    ┆ 1553.861732 │
│ 4    ┆ 1522.150838 │
│ 5    ┆ 1534.050279 │
│ 6    ┆ 1695.111732 │
│ 7    ┆ 1973.812849 │
│ 8    ┆ 2142.146648 │
│ 9    ┆ 2374.980447 │
└──────┴─────────────┘

Pipeline is behaving like real demand (error clumbs into morning ramp hours (7-10),
which is where weekly-naive may stuggle.)



"""