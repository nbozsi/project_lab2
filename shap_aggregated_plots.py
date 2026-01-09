import polars as pl
import altair as alt

pl.Config.set_tbl_width_chars(-1)
pl.Config.set_fmt_str_lengths(120)


def shap_agg_plot(df):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("sum(mean_abs_shap)", axis=alt.Axis(title="Sum of Shap Value ", labelFontSize=16, titleFontSize=20, format="~s")),
            y=alt.Y(
                "Feature:N",
                sort=alt.EncodingSortField(
                    field="mean_abs_shap",  # field to sort by
                    op="sum",  # aggregate operation
                    order="descending",  # descending = biggest on top
                ),
                axis=alt.Axis(title="Feature", labelFontSize=18, titleFontSize=20),
            ),
            color=alt.Color("lag:Q").scale(scheme="plasma").legend(values=range(-600, 721, 30)),
            order=alt.Order(
                # Sort the segments of the bars by this field
                "lag:O",
                sort="ascending",
            ),
        )
    ).properties(width=1600, height=800)


targets = [
    "System Direction (kWh)",
    "Positive Balancing Energy Unit Price for Balance Groups (HUFkWh)",
    "Negative Balancing Energy Unit Price for Balance Groups (HUFkWh)",
]

label2idx = pl.DataFrame()
combined_chart = None
for target in targets:
    df = pl.read_csv(f"shap_results/SHAP_Aggregated_{target}.csv")
    df = df.with_columns(
        pl.col("feature")
        .str.replace(r"(.*)(\_t(\-|\+)\d{2,3}min)", "${1}")
        .replace(
            {
                "temp_mean": "Temperature Mean",
                "temp_std": "Temperature Std",
                "ssrd_mean": "Surface Solar Radiation Downwards Mean",
                "ssrd_std": "Surface Solar Radiation Downwards Std",
                "100m_wind_speed_mean": "Wind Speed (100m) Mean",
                "100m_wind_speed_std": "Wind Speed (100m) Std",
                "10m_wind_speed_mean": "Wind Speed (10m) Mean",
                "10m_wind_speed_std": "Wind Speed (10m) Std",
            }
        )
        .alias("base_feature"),
        pl.col("feature").str.extract(r"_t((?:\+|\-)\d*)min").cast(pl.Int32).fill_null(0).alias("lag"),
    )
    if label2idx.is_empty():
        label2idx = (
            df.group_by("base_feature")
            .agg((pl.col("lag").sort_by(pl.col("lag").abs()).last() / 60).alias("lag"))
            .sort("base_feature")
            .with_row_index(offset=1)
            .rename({"index": "Feature"})
        )
        # label2idx = df.select(pl.col("base_feature").unique().sort()).with_row_index(offset=1).rename({"index": "Feature"})
        label2idx.write_csv(f"figurelabels.csv", separator="&", line_terminator=" \\\\\n", include_header=False)
    df = df.join(label2idx, on="base_feature")
    chart = shap_agg_plot(df)
    chart.configure_legend(gradientLength=1600, gradientThickness=20, orient="bottom", direction="horizontal").save(
        f"figures/shap_aggregated_{target}plot.png"
    )

    combined_chart = combined_chart & chart if combined_chart else chart

combined_chart.configure_legend(gradientLength=1600, gradientThickness=20, orient="bottom", direction="horizontal").save(
    f"figures/shap_aggregated_plot.png", ppi=200
)
