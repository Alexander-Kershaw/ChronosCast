import polars as pl

df = pl.read_parquet("data_lake/gold/feature_store_base_2023.parquet").sort("ts_utc")

null_weather = df.filter(
    pl.any_horizontal(
        [
            pl.col("t2m_c").is_null(),
            pl.col("wind_ms").is_null(),
            pl.col("rh_pct").is_null(),
            pl.col("msl_hpa").is_null(),
            pl.col("tp_mm").is_null(),
        ]
    )
).select(["ts_utc", "demand_mw", "t2m_c", "wind_ms", "rh_pct", "msl_hpa", "tp_mm"])

print("null_weather_rows:", null_weather.height)
print(null_weather)

null_demand = df.filter(pl.col("demand_mw").is_null()).select(["ts_utc"])
print("\nnull_demand_rows:", null_demand.height)
print(null_demand)
