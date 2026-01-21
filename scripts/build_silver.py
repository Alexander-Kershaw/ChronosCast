from pathlib import Path
import polars as pl

BRONZE_DIR = Path("data_lake/bronze/neso")
SILVER_DIR = Path("data_lake/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BRONZE_DIR / "demanddata_2023.csv"
OUT_PATH = SILVER_DIR / "demand_uk_hourly.parquet"

df = pl.read_csv(CSV_PATH)

# Columns of interest
date_col = "SETTLEMENT_DATE"
period_col = "SETTLEMENT_PERIOD"
demand_col = "ND"  # National Demand (MW)

missing = [c for c in [date_col, period_col, demand_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {df.columns}")

# Parse half hour rows
half_hour = (
    df.select([
        pl.col(date_col).cast(pl.Utf8).alias("settlement_date_raw"),
        pl.col(period_col).cast(pl.Int32).alias("settlement_period"),
        pl.col(demand_col).cast(pl.Float64).alias("demand_mw"),
    ])
    .with_columns([
        pl.col("settlement_date_raw").str.strptime(pl.Date, "%d-%b-%y", strict=False).alias("date"),
    ])
    .with_columns([
        (
            pl.col("date").cast(pl.Datetime)
            + (pl.col("settlement_period") - 1) * pl.duration(minutes=30)
        ).alias("ts_utc")
    ])
)

# Diagnose bad timestamp rows
bad_ts = half_hour.filter(pl.col("ts_utc").is_null()).height
print("Bad rows (null ts_utc):", bad_ts)
if bad_ts:
    print(
        half_hour.filter(pl.col("ts_utc").is_null())
        .select(["settlement_date_raw", "settlement_period", "demand_mw"])
        .head(10)
    )

# Filter to valid rows 
half_hour = (
    half_hour
    .filter(pl.col("ts_utc").is_not_null() & pl.col("demand_mw").is_not_null())
    .select(["ts_utc", "demand_mw"])
    .sort("ts_utc")
)

# Resample to hourly mean
hourly = (
    half_hour
    .group_by_dynamic("ts_utc", every="1h", closed="left", label="left")
    .agg(pl.col("demand_mw").mean().alias("demand_mw"))
    .sort("ts_utc")
)

# Checks
dupes = hourly.select(pl.col("ts_utc").is_duplicated().sum()).item()

diffs = (
    hourly.with_columns(pl.col("ts_utc").diff().alias("dt"))
    .select("dt")
    .drop_nulls()
)
bad_steps = diffs.filter(pl.col("dt") != pl.duration(hours=1)).height
nulls = hourly.select(pl.col("demand_mw").is_null().sum()).item()

min_ts, max_ts = (
    hourly
    .select([
        pl.col("ts_utc").min().alias("min_ts"),
        pl.col("ts_utc").max().alias("max_ts"),
    ])
    .row(0)
)

print("Half-hour rows:", half_hour.height)
print("Hourly rows:", hourly.height)
print("Range:", (min_ts, max_ts))
print("Duplicates:", dupes)
print("Bad step count (dt != 1h):", bad_steps)
print("Null demand count:", nulls)

hourly.write_parquet(OUT_PATH)
print("Wrote:", OUT_PATH)
