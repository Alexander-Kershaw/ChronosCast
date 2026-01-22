from pathlib import Path
import polars as pl

"""
------------------------------------------------------------
Feature store table:
------------------------------------------------------------
Key aspect of the ChronosCast project, the feature store
is a single hourly table keyed by ts_utc containing the
following:

- target: demand_mw
- drivers (weather features)
- time structure (calendar features)
- autoregressive memory: lag features
- stability contexts: rolling statistics

Rules:

- Leakage (for any arbitrary timestamp t, every feauture is
to be computed using data <= to t exclusively). This
ensures lag features are foundationally safely constructed.

- Rolling features have a window ending at t, not centered 
around it.


Examples of features included:

- Calendar: hour, dow, month, is_weekend
- Lags: t-1, t-24, t-168
- Rolling means: 24h and 168h
- Weather at timestamp t (for the various weather features)
------------------------------------------------------------
"""

GOLD = Path("data_lake/gold")
OUT_PATH = GOLD / "feature_store_2023.parquet"

BASE_PATH = GOLD / "feature_store_base_2023.parquet"
df = pl.read_parquet(BASE_PATH).sort("ts_utc")

# Calendar features
df = df.with_columns([
    pl.col("ts_utc").dt.hour().alias("hour"),
    pl.col("ts_utc").dt.weekday().alias("dow"),
    pl.col("ts_utc").dt.month().alias("month"),
    (pl.col("ts_utc").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
])

# Lag features 
df = df.with_columns([
    pl.col("demand_mw").shift(1).alias("lag_1"),
    pl.col("demand_mw").shift(24).alias("lag_24"),
    pl.col("demand_mw").shift(168).alias("lag_168"),
])

# Rolling means 
df = df.with_columns([
    pl.col("demand_mw").rolling_mean(window_size=24).alias("roll_mean_24"),
    pl.col("demand_mw").rolling_mean(window_size=168).alias("roll_mean_168"),
])

# Data checks
print("rows:", df.height)
print("null target:", df.select(pl.col("demand_mw").is_null().sum()).item())
print("null lag_168:", df.select(pl.col("lag_168").is_null().sum()).item())

df.write_parquet(OUT_PATH)
print("wrote:", OUT_PATH)
