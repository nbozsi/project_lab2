import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from datetime import datetime
from smape_error import smape

TIMELAGS = range(1, 6)  # in steps (1 step is 15 minutes)


TARGET_COLS = (
    "Negative Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
    "Positive Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
    "System Direction (kWh)",
)


joined_df = pl.read_parquet("data/joined_df.parquet").filter(pl.col("UTCdate").dt.replace_time_zone(None) >= datetime(2024, 1, 1))

results = []
for lag in TIMELAGS:
    for target in TARGET_COLS:
        df = joined_df.select(pl.col(target).alias("y_true"), pl.col(target).shift(lag).alias("y_pred")).slice(lag)
        y_true = df["y_true"]
        y_pred = df["y_pred"]

        results.append(
            {
                "Target": f"{target}_t+{lag*15}min",
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": root_mean_squared_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
                "SMAPE": smape(y_true.to_numpy(), y_pred.to_numpy()),
            }
        )

naive_results = pl.from_dicts(results)
naive_results.write_csv("model_results/naive_model.csv")
