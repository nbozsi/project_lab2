import altair as alt
import polars as pl


def lag_chart(df, width=800, height=400):

    # Base encodings
    base = (
        alt.Chart(df)
        .encode(
            x=alt.X("lag:T", axis=alt.Axis(title="Lag", labelAngle=0, labelFontSize=14, titleFontSize=14, tickCount=4, format="%H:%M")),
            y=alt.Y("error:Q", axis=alt.Axis(title=df["error measure"][0], labelFontSize=14, titleFontSize=14)),
            color=alt.Color("model:N")
            .scale(scheme="category10")
            .legend(orient="bottom", direction="horizontal", labelFontSize=13, titleFontSize=13, labelLimit=600),
        )
        .properties(title=df["Target"][0])
    )

    bar = base.mark_bar()
    # Line chart
    line = base.mark_line(size=2)

    # Point markers
    points = base.mark_point(size=50, filled=False)

    # Combine
    return (points + line).properties(width=800, height=300)


def lag_chart(df, width=800, height=400):

    # Base encodings
    base = (
        alt.Chart(df)
        .encode(
            x=alt.X("model:N"),
            y=alt.Y("error:Q", axis=alt.Axis(title=df["error measure"][0], labelFontSize=14, titleFontSize=14)),
            color=alt.Color("model:N")
            .scale(scheme="category10")
            .legend(orient="bottom", direction="horizontal", labelFontSize=13, titleFontSize=13, labelLimit=600),
            column=alt.Column("lag:T", header=alt.Header(format="%H:%M")),
        )
        .properties(title=df["Target"][0])
    )

    bar = base.mark_bar()

    # Combine
    return bar  # .properties(width=800, height=300)


def unpivot_errors(df):
    return df.with_columns(
        pl.col("Target").str.strip_chars_end("t+1234567890min").str.strip_suffix("_"),
        (pl.col("Target").str.extract(r"_t\+(\d*)min").cast(pl.Int32) * 60 * 1000).cast(pl.Datetime("ms")).alias("lag"),
    ).unpivot(set(df.columns) - {"Target", "lag"}, index=["Target", "lag"], variable_name="error measure", value_name="error")


def compare_results(results):

    compare = pl.concat(
        [result_df.with_columns(model=pl.lit(model_name)) for model_name, result_df in results.items()],
        how="vertical",
    )

    sys_dir = compare.filter((pl.col("Target") == "System Direction (kWh)"))
    pos_unit_prices = compare.filter((pl.col("Target") == "Positive Balancing Energy Unit Price for Balance Groups (HUF/kWh)"))
    neg_unit_prices = compare.filter((pl.col("Target") == "Negative Balancing Energy Unit Price for Balance Groups (HUF/kWh)"))

    for meas in compare["error measure"].unique():
        chart = lag_chart(sys_dir.filter(pl.col("error measure") == meas))
        chart.save(f"figures/{meas}_system_direction.png")
        chart = lag_chart(pos_unit_prices.filter(pl.col("error measure") == meas))
        chart.save(f"figures/{meas}_pos_unit_price.png")
        chart = lag_chart(neg_unit_prices.filter(pl.col("error measure") == meas))
        chart.save(f"figures/{meas}_neg_unit_price.png")


results = {
    "xgboost_w_weather": unpivot_errors(pl.read_csv("model_results/xgb_results_with_weather.csv")),
    "NN": unpivot_errors(pl.read_csv("model_results/NN_100_seeds_joined_df.csv")),
    "NN_w_temp": unpivot_errors(pl.read_csv("model_results/NN_100_seeds_joined_df_with_temp.csv")),
    "NN_w_weather": unpivot_errors(pl.read_csv("model_results/NN_100_seeds_joined_df_with_weather.csv")),
    "NNH": unpivot_errors(pl.read_csv("model_results/NN_100_seeds_Huber_joined_df.csv")),
    "NNH_w_temp": unpivot_errors(pl.read_csv("model_results/NN_100_seeds_Huber_joined_df_with_temp.csv")),
    "NNH_w_weather": unpivot_errors(pl.read_csv("model_results/NN_100_seeds_Huber_joined_df_with_weather.csv")),
    "naive": unpivot_errors(pl.read_csv("model_results/naive_model.csv")),
}
compare_results(results)
