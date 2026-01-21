import polars as pl

df = pl.read_parquet("data_lake/silver/demand_uk_hourly.parquet").sort("ts_utc")

gaps = (
    df.with_columns(pl.col("ts_utc").diff().alias("dt"))
    .filter(pl.col("dt").is_not_null() & (pl.col("dt") != pl.duration(hours=1)))
    .select(["ts_utc", "dt"])
)

print("gap_rows: ", gaps.height)
print(gaps)


"""
OUTPUT: 


gap_rows:  1
shape: (1, 2)
┌─────────────────────┬──────────────┐
│ ts_utc              ┆ dt           │
│ ---                 ┆ ---          │
│ datetime[μs]        ┆ duration[μs] │
╞═════════════════════╪══════════════╡
│ 2023-03-27 00:00:00 ┆ 2h           │


2023-03-26 at 23:00:00 is missing


"""