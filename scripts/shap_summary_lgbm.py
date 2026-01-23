import polars as pl
import numpy as np
from datetime import timedelta
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt
from pathlib import Path

GOLD_PATH = "data_lake/gold/feature_store_2023.parquet"
OUTDIR = Path("reports/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "t2m_c", "wind_ms", "rh_pct", "msl_hpa", "tp_mm",
    "lag_1", "lag_24", "lag_168",
    "roll_mean_24", "roll_mean_168",
]
TARGET = "demand_mw"


df = (
    pl.read_parquet(GOLD_PATH)
    .sort("ts_utc")
    .drop_nulls([TARGET] + FEATURES)
)

# Using last 30 days as test window for explainability
max_ts = df.select(pl.col("ts_utc").max()).item()
split_ts = max_ts - timedelta(days=30)

train = df.filter(pl.col("ts_utc") < split_ts)
test  = df.filter(pl.col("ts_utc") >= split_ts)

X_train = train.select(FEATURES).to_pandas()
y_train = train.select(TARGET).to_numpy().ravel()

X_test = test.select(FEATURES).to_pandas()
y_test = test.select(TARGET).to_numpy().ravel()

print("train rows:", len(X_train))
print("test rows :", len(X_test))

# Train
model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=7,
)
model.fit(X_train, y_train)

# SHAP (TreeExplainer)
explainer = shap.TreeExplainer(model)

# sample up to 1000 rows for performance
n = min(1000, len(X_test))
X_sample = X_test.sample(n=n, random_state=7)

shap_values = explainer.shap_values(X_sample)

# Summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
out1 = OUTDIR / "shap_summary_beeswarm.png"
plt.tight_layout()
plt.savefig(out1, dpi=200)
plt.close()

# Summary plot (bar)
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
out2 = OUTDIR / "shap_summary_bar.png"
plt.tight_layout()
plt.savefig(out2, dpi=200)
plt.close()

print("Wrote:")
print(" -", out1)
print(" -", out2)
