import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import altair as alt


TIMELAGS = range(1, 25)  # in steps (1 step is 15 minutes)


TARGET_COLS = (
    "Negative Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
    "Positive Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
    "System Direction (kWh)",
)


def lag_chart(df, colors=["red", "green"], width=800, height=400):
    # Base encodings
    base = alt.Chart(df).encode(
        x=alt.X("lag:T", axis=alt.Axis(title="Lag", labelAngle=0, labelFontSize=14, titleFontSize=14, tickCount=4, format="%H:%M")),
        y=alt.Y("error:Q", axis=alt.Axis(title="Error", labelFontSize=14, titleFontSize=14)),
        color=alt.Color("target", scale=alt.Scale(range=colors)).legend(
            orient="bottom", labelFontSize=11, titleFontSize=13, labelLimit=310
        ),
        strokeDash=alt.StrokeDash("error measure").legend(orient="bottom", labelFontSize=11, titleFontSize=13),
    )

    # Line chart
    line = base.mark_line(size=2)

    # Point markers
    points = base.mark_point(size=50, filled=False)

    # Combine
    return (points + line).properties(width=800, height=400)


joined_df = pl.read_parquet("data/joined_df.parquet")

results = []
for lag in TIMELAGS:
    for target in TARGET_COLS:
        df = joined_df.select(pl.col(target).alias("y_true"), pl.col(target).shift(lag).alias("y_pred")).slice(lag)

        y_true = df["y_true"]
        y_pred = df["y_pred"]

        results.append(
            {
                "target": target,
                "lag": lag,
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": root_mean_squared_error(y_true, y_pred),
            }
        )

results = (
    pl.from_dicts(results)
    .unpivot(["mae", "rmse"], index=["target", "lag"], variable_name="error measure", value_name="error")
    .with_columns(
        (pl.col("lag") * 15 * 60 * 1000).cast(pl.Datetime("ms")),
    )
)

chart = lag_chart(results.filter(pl.col("target") == "System Direction (kWh)"), ["blue"])
chart.save("figures/naive_model_performance_system_direction.png")

chart = lag_chart(results.filter(pl.col("target") != "System Direction (kWh)"))
chart.save("figures/naive_model_performance_unit_price.png")
