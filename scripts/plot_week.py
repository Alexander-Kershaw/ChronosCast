import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime

df = (
    pl.read_parquet("data_lake/silver/demand_uk_hourly.parquet")
    .sort("ts_utc")
)

# picked a week away from the missing demand date
start = datetime(2023, 2, 1, 0, 0, 0)
end   = datetime(2023, 2, 8, 0, 0, 0)

week = df.filter((pl.col("ts_utc") >= start) & (pl.col("ts_utc") < end))

print("week rows:", week.height)
print("week nulls:", week.select(pl.col("demand_mw").is_null().sum()).item())

x = week["ts_utc"].to_list()
y = week["demand_mw"].to_list()

plt.figure()
plt.plot(x, y)
plt.title("UK Demand (ND) hourly, 1-week")
plt.xlabel("Time (UTC index)")
plt.ylabel("Demand (MW)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
