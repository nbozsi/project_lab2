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

    cols_5_steps = (
        "temp_mean",
        "temp_std",
        "ssrd_mean",
        "ssrd_std",
        "10m_wind_speed_mean",
        "10m_wind_speed_std",
        "100m_wind_speed_mean",
        "100m_wind_speed_std",
    )

    target_cols = (
        "Negative Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
        "Positive Balancing Energy Unit Price for Balance Groups (HUF/kWh)",
        "System Direction (kWh)",
    )

    exclude_cols = {
        "valid_time", "number", "expver", "Óraátállítás",
        "Net System Load (Actual - Operational Control)",
        "Activated Negative mFRR and RR Balancing Energy Cost (HUF)",
        "Activated Positive mFRR and RR Balancing Energy Cost (HUF)",
        "Activated Positive mFRR and RR Balancing Energy (kWh)",
        "Activated Negative mFRR and RR Balancing Energy (kWh)",
        "Gross Actual System Load",
        "Gross Planned System Load",
        "Net Actual System Load - Net Commercial Settlement Measurement",
        "Net Domestic Generation (Actual)",
        "Net Load",
        "Net Planned Power Plant Generation",
        "Net Planned System Generation",
        "Net Planned System Load",
        "Net System Load Forecast (Day-Ahead)",
        "100m_wind_speed_mean_right",
        "hour",
        "dayofweek",
        "dayofweek_sin",
        "dayofweek_cos",
        "day_of_year",
        "is_holiday",
    }

    existing_cols = set(final_df.columns)

    cols_10_h_exist = [c for c in cols_10_h if c in existing_cols]
    cols_12_h_exist = [c for c in cols_12_h if c in existing_cols]
    cols_5_steps_exist = [c for c in cols_5_steps if c in existing_cols]
    target_cols_exist = [c for c in target_cols if c in existing_cols]

    time_features = ["hour", "dayofweek", "dayofweek_sin", "dayofweek_cos", "day_of_year", "is_holiday"]

    # Columns to include without duplication
    non_time_cols = set(final_df.columns) - set(time_features) - exclude_cols

    X = final_df.select(
        *(pl.col(c) for c in time_features if c in final_df.columns),  # time features as-is
        *(pl.col(c) for c in non_time_cols),                           # everything else that isn’t excluded
        *(timelag_expressions(40, cols_10_h_exist) if cols_10_h_exist else []),
        *(timelag_expressions(48, cols_12_h_exist) if cols_12_h_exist else []),
        *(timelag_expressions(5, cols_5_steps_exist) if cols_5_steps_exist else []),
        *timelag_expressions(
            -40,
            set(keep_cols)
            - set(cols_10_h_exist)
            - set(cols_12_h_exist)
            - set(cols_5_steps_exist)
            - {"UTCdate"}
            - exclude_cols,
        ),
    )

    y = final_df.select(
        *(
            timelag_expressions(range(1, 6), target_cols_exist)
            if target_cols_exist else []
        ),
    )

    X = X.drop_nulls()
    y = y.slice(40, y.height - 48 - 40)

    return X, y



if __name__ == "__main__":
    joined_df = pl.read_parquet("data/joined_df_with_weather.parquet")
    X, y = create_training_data(joined_df)
    print(y.describe())
    print(X.describe())
