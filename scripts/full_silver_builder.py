from pathlib import Path
import re
import numpy as np
import polars as pl
import xarray as xr
import zipfile

# Define paths for bronze and silver, make directories
BRONZE = Path("data_lake/bronze/era5")
EXTRACT_ROOT = BRONZE / "extracted_monthly"
SILVER = Path("data_lake/silver")
SILVER.mkdir(parents=True, exist_ok=True)
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_PATH = SILVER / "weather_uk_hourly.parquet"

a, b = 17.625, 243.04  # Magnus constants

# Extraction for era5 compressed downloads
def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return sorted([p for p in out_dir.glob("*.nc")])

def open_month(nc_files: list[Path]) -> xr.Dataset:
    # Expect two files per extraction (instant and accumulated weather data)
    if len(nc_files) < 1:
        raise ValueError("No .nc files extracted")

    # Open all and merge
    dsets = [xr.open_dataset(p, engine="netcdf4") for p in nc_files]
    return xr.merge(dsets, compat="override")

# To polars 
def month_to_polars(ds: xr.Dataset) -> pl.DataFrame:
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    # search for time coordinate name
    if "time" in ds.coords:
        time_coord = "time"
    elif "valid_time" in ds.coords:
        time_coord = "valid_time"
    else:
        raise ValueError(f"No time coordinate found. Coords: {list(ds.coords)}")

    # The initial features, checking for missing variables
    need = ["t2m", "d2m", "u10", "v10", "msl", "tp"]
    missing = [v for v in need if v not in ds.variables]
    if missing:
        raise ValueError(f"Missing variables: {missing}. Available: {list(ds.variables)}")

    mean = ds[need].mean(dim=[lat_dim, lon_dim], skipna=True)

    # pull variables and engineer features
    t2m = mean["t2m"].values
    d2m = mean["d2m"].values
    u10 = mean["u10"].values
    v10 = mean["v10"].values
    msl = mean["msl"].values
    tp  = mean["tp"].values

    t2m_c = t2m - 273.15
    td_c  = d2m - 273.15
    wind_ms = np.sqrt(u10**2 + v10**2)
    msl_hpa = msl / 100.0
    tp_mm = tp * 1000.0

    rh = 100.0 * np.exp((a * td_c) / (b + td_c)) / np.exp((a * t2m_c) / (b + t2m_c))
    rh = np.clip(rh, 0.0, 100.0)

    # make sure consistent datetime unit (ns)
    ts = mean.coords[time_coord].values.astype("datetime64[ns]")

    # cast to polars dataframe
    df = pl.DataFrame(
        {
            "ts_utc": ts,
            "t2m_c": t2m_c,
            "wind_ms": wind_ms,
            "rh_pct": rh,
            "msl_hpa": msl_hpa,
            "tp_mm": tp_mm,
        }
    ).with_columns(
        pl.col("ts_utc").cast(pl.Datetime("us"))
    ).sort("ts_utc")

    return df

def main():
    # Expect files like: era5_uk_2023_01.nc (note the .nc are zips) sort into chronological order
    zips = sorted(BRONZE.glob("era5_uk_2023_??.nc"))
    if not zips:
        raise FileNotFoundError(f"No monthly files found in {BRONZE}")

    # process individual era5 zips
    frames = []
    for zp in zips:
        m = re.search(r"2023_(\d{2})", zp.name)
        mm = m.group(1) if m else "xx"
        out_dir = EXTRACT_ROOT / f"2023_{mm}"
        print("Processing:", zp.name, "->", out_dir)

        nc_files = extract_zip(zp, out_dir)
        ds = open_month(nc_files)
        dfm = month_to_polars(ds)

        print("  rows:", dfm.height, "range:", dfm.select(pl.min("ts_utc"), pl.max("ts_utc")).row(0))
        frames.append(dfm)

    df = pl.concat(frames).sort("ts_utc")

    # print some summaries of duplicates, missing values, and ranges
    dupes = df.select(pl.col("ts_utc").is_duplicated().sum()).item()
    nulls = df.select(pl.all().null_count()).row(0)
    print("\nFULL YEAR")
    print("rows:", df.height)
    print("range:", df.select(pl.min("ts_utc"), pl.max("ts_utc")).row(0))
    print("duplicates:", dupes)
    print("null_counts:", dict(zip(df.columns, nulls)))

    # write to parquet
    df.write_parquet(OUT_PATH)
    print("wrote:", OUT_PATH)

if __name__ == "__main__":
    main()
