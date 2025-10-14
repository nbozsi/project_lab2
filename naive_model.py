import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import altair as alt
from datetime import datetime

TIMELAGS = range(1, 6)  # in steps (1 step is 15 minutes)


TARGET_COLS = (
    "Negative Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
    "Positive Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
    "System Direction (kWh)",
)


def lag_chart(df, colors=["red", "green"], model_name=None, width=800, height=400):
    # Base encodings
    if "model" in df.columns:
        base = alt.Chart(df).encode(
            x=alt.X("lag:T", axis=alt.Axis(title="Lag", labelAngle=0, labelFontSize=14, titleFontSize=14, tickCount=4, format="%H:%M")),
            y=alt.Y("error:Q", axis=alt.Axis(title="Error", labelFontSize=14, titleFontSize=14)),
            color=alt.Color("Target:N", scale=alt.Scale(range=colors)).legend(
                orient="bottom", direction="vertical", labelFontSize=13, titleFontSize=13, labelLimit=600
            ),
            strokeDash=alt.StrokeDash("model:N", sort=(model_name, "Naive model")).legend(
                orient="bottom", labelFontSize=13, titleFontSize=13
            ),
        )
    else:
        base = alt.Chart(df).encode(
            x=alt.X("lag:T", axis=alt.Axis(title="Lag", labelAngle=0, labelFontSize=14, titleFontSize=14, tickCount=4, format="%H:%M")),
            y=alt.Y("error:Q", axis=alt.Axis(title="Error", labelFontSize=14, titleFontSize=14)),
            color=alt.Color("Target:N", scale=alt.Scale(range=colors)).legend(
                orient="bottom", direction="vertical", labelFontSize=13, titleFontSize=13, labelLimit=600
            ),
            strokeDash=alt.StrokeDash("error measure:N").legend(orient="bottom", labelFontSize=13, titleFontSize=13),
        )

    # Line chart
    line = base.mark_line(size=2)

    # Point markers
    points = base.mark_point(size=50, filled=False)

    # Combine
    return (points + line).properties(width=800, height=300)


joined_df = pl.read_parquet("data/joined_df.parquet")

results = []
for lag in TIMELAGS:
    for target in TARGET_COLS:
        df = (
            joined_df.filter(pl.col("UTCdate").dt.replace_time_zone(None) >= datetime(2024, 1, 1))
            .select(pl.col(target).alias("y_true"), pl.col(target).shift(lag).alias("y_pred"))
            .slice(lag)
        )
        y_true = df["y_true"]
        y_pred = df["y_pred"]
        print("shape", df.height)

        results.append(
            {
                "Target": target,
                "lag": lag,
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": root_mean_squared_error(y_true, y_pred),
                "R²": r2_score(y_true, y_pred),
            }
        )

naive_results = (
    pl.from_dicts(results)
    .unpivot(["MAE", "RMSE", "R²"], index=["Target", "lag"], variable_name="error measure", value_name="error")
    .with_columns(
        (pl.col("lag") * 15 * 60 * 1000).cast(pl.Datetime("ms")),
    )
)


sys_dir = naive_results.filter((pl.col("Target") == "System Direction (kWh)"))
unit_prices = naive_results.filter((pl.col("Target") != "System Direction (kWh)"))
print(sys_dir)
chart = lag_chart(sys_dir.filter(pl.col("error measure").is_in(("MAE", "RMSE"))), ["blue"])
chart.save("figures/naive_model_rmse_system_direction.png")

chart = lag_chart(unit_prices.filter(pl.col("error measure").is_in(("MAE", "RMSE"))))
chart.save("figures/naive_model_rmse_unit_price.png")

chart = lag_chart(naive_results.filter(pl.col("error measure") == "R²"), ["red", "green", "blue"])
chart.save("figures/naive_model_R2.png")


def unpivot_errors(df):
    return df.with_columns(
        pl.col("Target").str.strip_chars_end("t+1234567890min").str.strip_suffix("_"),
        (pl.col("Target").str.extract(r"_t\+(\d*)min").cast(pl.Int32) * 60 * 1000).cast(pl.Datetime("ms")).alias("lag"),
    ).unpivot(["MAE", "RMSE", "R²"], index=["Target", "lag"], variable_name="error measure", value_name="error")


def compare_with_naive(results, naive_results, model_name):
    compare = pl.concat(
        (
            results.with_columns(model=pl.lit(model_name)),
            naive_results.with_columns(model=pl.lit("Naive model")),
        ),
        how="vertical",
    )

    sys_dir = compare.filter((pl.col("Target") == "System Direction (kWh)"))
    unit_prices = compare.filter((pl.col("Target") != "System Direction (kWh)"))

    chart = lag_chart(sys_dir.filter(pl.col("error measure") == "MAE"), ["blue"], model_name=model_name)
    chart.save(f"figures/{model_name}_mae_system_direction.png")
    chart = lag_chart(sys_dir.filter(pl.col("error measure") == "RMSE"), ["blue"], model_name=model_name)
    chart.save(f"figures/{model_name}_rmse_system_direction.png")

    chart = lag_chart(unit_prices.filter(pl.col("error measure") == "MAE"), model_name=model_name)
    chart.save(f"figures/{model_name}_mae_unit_price.png")
    chart = lag_chart(unit_prices.filter(pl.col("error measure") == "RMSE"), model_name=model_name)
    chart.save(f"figures/{model_name}_rmse_unit_price.png")

    chart = lag_chart(compare.filter(pl.col("error measure") == "R²"), ["red", "green", "blue"], model_name=model_name)
    chart.save(f"figures/{model_name}_R2.png")


results = unpivot_errors(pl.read_parquet("xgboost_separate_results.parquet"))
compare_with_naive(results, naive_results, "xgboost")

results = unpivot_errors(pl.read_csv("evaluation_metrics_joined_df_with_temp.csv"))
compare_with_naive(results, naive_results, "NN_with_temp")

results = unpivot_errors(pl.read_csv("evaluation_metrics_joined_df_with_weather.csv"))
compare_with_naive(results, naive_results, "NN_with_weather")

results = unpivot_errors(pl.read_csv("evaluation_metrics_joined_df.csv"))
compare_with_naive(results, naive_results, "NN")
