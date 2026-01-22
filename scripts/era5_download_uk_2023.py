from pathlib import Path
import cdsapi

OUT_DIR = Path("data_lake/bronze/era5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box for UK
AREA = [60, -8, 49, 2]

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
]

def download_month(year: int, month: int):
    c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api")


    target = OUT_DIR / f"era5_uk_{year}_{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        print("Skip (exists):", target.name)
        return

    req = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": AREA,
    }

    print("Requesting:", target.name)
    c.retrieve("reanalysis-era5-single-levels", req, str(target))

if __name__ == "__main__":
    for m in range(1, 13):
        download_month(2023, m)
