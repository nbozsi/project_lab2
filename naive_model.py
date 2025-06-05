import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

TIMELAG = 8  # in steps (1 step is 15 minutes)


target_cols = (
    "Negatív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
    "Pozitív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
    "Rendszer-irány (kWh)",
)
w = max(map(len, target_cols))

joined_df = pl.read_parquet("data/joined_df.parquet")

print(f" Naive model with timelag {TIMELAG/4:>5} hours" + " |     MAE    |    RMSE    ")
for target in target_cols:
    df = joined_df.select(pl.col(target).alias("y_true"), pl.col(target).shift(TIMELAG).alias("y_pred")).slice(TIMELAG)

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    print(f"{target:>{w}} | {mean_absolute_error(y_true,y_pred):^10.2f} | {root_mean_squared_error(y_true,y_pred):^10.2f}")
