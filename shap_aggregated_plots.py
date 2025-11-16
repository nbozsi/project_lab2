import polars as pl
import altair as alt

pl.Config.set_tbl_width_chars(-1)
pl.Config.set_fmt_str_lengths(120)
df = pl.read_csv("shap_results/shap_feature_importance_all_features.csv")
# df = pl.read_csv("shap_aggregated_top5.csv")
df = df.with_columns(
    pl.col("feature").str.replace(r"(.*)(\_t(\-|\+)\d{2,3}min)", "${1}").alias("base_feature"),
    pl.col("feature").str.extract(r"_t((?:\+|\-)\d*)min").cast(pl.Int32).alias("lag"),
)
print(df)
print(df["base_feature"].value_counts().sort(by="count"))

chart = (
    # alt.Chart(df)
    alt.Chart(df.filter(pl.col("base_feature") != "System Direction (kWh)"))
    .mark_bar()
    .encode(
        x="sum(mean_abs_shap)",
        y=alt.Y(
            "base_feature",
            sort=alt.EncodingSortField(
                field="mean_abs_shap",  # field to sort by
                op="sum",  # aggregate operation
                order="descending",  # descending = biggest on top
            ),
        ),
        color=alt.Color("lag:O").scale(scheme="viridis"),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "lag",
            sort="ascending",
        ),
    )
).properties(width=1600, height=800)
# chart = df.plot.bar(x="feature", y="mean_abs_shap", column="group")
chart.save("miz2.png")
