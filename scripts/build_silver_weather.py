from pathlib import Path
import numpy as np
import polars as pl
import xarray as xr

BRONZE_EX = Path("data_lake/bronze/era5/extracted")
SILVER = Path("data_lake/silver")
SILVER.mkdir(parents=True, exist_ok=True)

# Got two era5 files for each month
NC_INSTANT = BRONZE_EX / "data_stream-oper_stepType-instant.nc"
NC_ACCUM   = BRONZE_EX / "data_stream-oper_stepType-accum.nc"

OUT_PATH = SILVER / "weather_uk_hourly_2023_01.parquet"

for p in [NC_INSTANT, NC_ACCUM]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

# Open both extracted files from era5 download (instant/ accum)
ds_i = xr.open_dataset(NC_INSTANT, engine="netcdf4")
ds_a = xr.open_dataset(NC_ACCUM, engine="netcdf4")

# Merge variables aligning on time/lat/lon
ds = xr.merge([ds_i, ds_a], compat="override")

# Identify dimension names
if "time" in ds.coords:
    time_coord = "time"
elif "valid_time" in ds.coords:
    time_coord = "valid_time"
else:
    raise ValueError(f"No time coordinate found. Coords: {list(ds.coords)}")
lat_dim = "latitude" if "latitude" in ds.dims else "lat"
lon_dim = "longitude" if "longitude" in ds.dims else "lon"

need = ["t2m", "d2m", "u10", "v10", "msl", "tp"]
missing = [v for v in need if v not in ds.variables]
if missing:
    raise ValueError(f"Missing variables: {missing}. Found: {list(ds.variables)}")

# Spatial mean over the UK box grid
mean = ds[need].mean(dim=[lat_dim, lon_dim], skipna=True)

# Pull the arrays
t2m = mean["t2m"].values
d2m = mean["d2m"].values
u10 = mean["u10"].values
v10 = mean["v10"].values
msl = mean["msl"].values
tp  = mean["tp"].values

# Unit conversions and derived features
t2m_c = t2m - 273.15
td_c  = d2m - 273.15
wind_ms = np.sqrt(u10**2 + v10**2)
msl_hpa = msl / 100.0
tp_mm = tp * 1000.0

# Magnus approximation
a, b = 17.625, 243.04
rh = 100.0 * np.exp((a * td_c) / (b + td_c)) / np.exp((a * t2m_c) / (b + t2m_c))
rh = np.clip(rh, 0.0, 100.0)

# Time axis
ts = mean.coords[time_coord].values.astype("datetime64[ns]")

df = (
    pl.DataFrame(
        {
            "ts_utc": ts,
            "t2m_c": t2m_c,
            "wind_ms": wind_ms,
            "rh_pct": rh,
            "msl_hpa": msl_hpa,
            "tp_mm": tp_mm,
        }
    )
    .sort("ts_utc")
)

# Some checks
dupes = df.select(pl.col("ts_utc").is_duplicated().sum()).item()
null_counts = df.select(pl.all().null_count()).row(0)

min_ts, max_ts = (
    df.select([
        pl.col("ts_utc").min().alias("min_ts"),
        pl.col("ts_utc").max().alias("max_ts"),
    ]).row(0)
)

print("rows:", df.height)
print("range:", (min_ts, max_ts))
print("duplicates:", dupes)
print("null_counts:", dict(zip(df.columns, null_counts)))

# Fix of polars complaining so creating alias for column names
def minmax(col: str):
    return df.select([
        pl.col(col).min().alias("min"),
        pl.col(col).max().alias("max"),
    ]).row(0)

print("t2m_c min/max:", minmax("t2m_c"))
print("wind_ms min/max:", minmax("wind_ms"))
print("msl_hpa min/max:", minmax("msl_hpa"))
print("tp_mm min/max:", minmax("tp_mm"))
print("rh_pct min/max:", minmax("rh_pct"))

df.write_parquet(OUT_PATH)
print("wrote:", OUT_PATH)
