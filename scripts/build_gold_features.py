from pathlib import Path
import polars as pl

SILVER = Path("data_lake/silver")
GOLD = Path("data_lake/gold")
GOLD.mkdir(parents=True, exist_ok=True)

DEMAND_PATH = SILVER / "demand_uk_hourly.parquet"
WEATHER_PATH = SILVER / "weather_uk_hourly.parquet"
OUT_PATH = GOLD / "feature_store_base_2023.parquet"

demand = (
    pl.read_parquet(DEMAND_PATH)
    .select(["ts_utc", "demand_mw"])
    .with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))
)

weather = (
    pl.read_parquet(WEATHER_PATH)
    .with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))
)

# Joining demand and weather to timestamps
base = (
    demand.join(weather, on="ts_utc", how="left")
    .sort("ts_utc")
)

print("rows:", base.height)
print("null demand:", base.select(pl.col("demand_mw").is_null().sum()).item())
weather_cols = ["t2m_c", "wind_ms", "rh_pct", "msl_hpa", "tp_mm"]
print(
    "null weather rows:",
    base.select(pl.any_horizontal([pl.col(c).is_null() for c in weather_cols]).sum()).item()
)

base.write_parquet(OUT_PATH)
print("wrote:", OUT_PATH)
