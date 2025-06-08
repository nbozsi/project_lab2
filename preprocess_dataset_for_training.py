import polars as pl


def timelag_expressions(lags, cols=None):
    if isinstance(lags, int):
        if lags > 0:
            lags = range(1, lags + 1)
        else:
            lags = range(lags, 0)
    if not cols:
        for i in lags:
            yield (pl.all().exclude("Időpont").shift(-i).name.suffix(f"_t{i*15:+}min"))
    else:
        for i in lags:
            for colname in cols:
                yield (pl.col(colname).shift(-i).name.suffix(f"_t{i*15:+}min"))


def create_training_data(df):

    col_stats = (
        df.describe()
        .transpose(column_names="statistic", include_header=True)
        .with_columns((pl.col("null_count") / pl.col("count")).alias("null_ratio"))
    )

    keep_cols = col_stats.filter(pl.col("null_ratio") <= 0.1)["column"]

    final_df = df.select(keep_cols.to_list()).fill_null(strategy="forward")

    cols_10_h = (
        "Estimated Wind Power Production (Current)",
        "Estimated Wind Power Production (Day-Ahead)",
        "Estimated Wind Power Production (Intraday)",
        "Estimated Solar Power Production (Current)",
        "Estimated Solar Power Production (Intraday)",
        "Estimated Solar Power Production (Day-Ahead)",
    )

    cols_12_h = (
        "Gross Planned Power Plant Generation",
        "Gross System Load Forecast (Day-Ahead)",
        "HU-AT Schedule",
        "HU-HR Schedule",
        "HU-SK Schedule",
        "HU-RS Schedule",
        "HU-UA Schedule",
        "HU-RO Schedule",
    )

    target_cols = (
        "Negative Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
        "Positive Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
        "System Direction (kWh)",
    )

    X = final_df.select(
        pl.all(),
        *timelag_expressions(40, cols_10_h),
        *timelag_expressions(48, cols_12_h),
        *timelag_expressions(-40, set(keep_cols) - set(cols_10_h) - set(cols_12_h) - {"Időpont"}),
    )
    y = final_df.select(
        *timelag_expressions(range(16, 21), target_cols),
    )

    X = X.drop_nulls()
    y = y.slice(40, y.height - 48 - 40)

    return (X, y)


if __name__ == "__main__":
    joined_df = pl.read_parquet("data/joined_df.parquet")
    X, y = create_training_data(joined_df)
    print(y.describe())
    print(X.describe())
