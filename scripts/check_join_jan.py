import polars as pl
from datetime import datetime

# UK energy demand data
demand = (
    pl.read_parquet("data_lake/silver/demand_uk_hourly.parquet")
    .select(["ts_utc", "demand_mw"])
)

# UK weather data
weather = pl.read_parquet("data_lake/silver/weather_uk_hourly_2023_01.parquet")

start = datetime(2023, 1, 1, 0, 0, 0)
end   = datetime(2023, 2, 1, 0, 0, 0)

d_jan = demand.filter((pl.col("ts_utc") >= start) & (pl.col("ts_utc") < end))
w_jan = weather.filter((pl.col("ts_utc") >= start) & (pl.col("ts_utc") < end))

print("demand_jan rows:", d_jan.height)
print("weather_jan rows:", w_jan.height)

joined = d_jan.join(w_jan, on="ts_utc", how="inner")

print("joined rows:", joined.height)

# Check for any nulls after join
nulls = joined.select(pl.all().null_count()).row(0)
print("null_counts:", dict(zip(joined.columns, nulls)))

# Check timestamp coverage 
missing_in_weather = d_jan.join(w_jan.select("ts_utc"), on="ts_utc", how="anti").height
missing_in_demand  = w_jan.join(d_jan.select("ts_utc"), on="ts_utc", how="anti").height
print("missing_in_weather:", missing_in_weather)
print("missing_in_demand :", missing_in_demand)
