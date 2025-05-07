import polars as pl


def create_training_data(df):

    col_stats = (
        df.describe()
        .transpose(column_names="statistic", include_header=True)
        .with_columns((pl.col("null_count") / pl.col("count")).alias("null_ratio"))
    )

    keep_cols = col_stats.filter(pl.col("null_ratio") <= 0.1)["column"]

    final_df = df.select(keep_cols.to_list()).fill_null(strategy="forward")

    cols_10_h = (
        "Szélerőművek becsült termelése (aktuális)",
        "Szélerőművek becsült termelése (dayahead)",
        "Szélerőművek becsült termelése (intraday)",
        "Naperőművek becsült termelése (aktuális)",
        "Naperőművek becsült termelése (intraday)",
        "Naperőművek becsült termelése (dayahead)",
    )

    cols_12_h = (
        "Bruttó terv erőművi termelés",
        "Bruttó rendszerterhelés becslés (dayahead)",
        "HU-AT menetrend",
        "HU-HR menetrend",
        # "HU-SI menetrend (RIR NT)",
        "HU-SK menetrend",
        "HU-RS menetrend",
        "HU-UK menetrend",
        "HU-RO menetrend",
    )
    target_cols = (
        "Negatív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
        "Pozitív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
        "Rendszer-irány (kWh)",
    )

    def timelag_expressions(lags, cols=None):
        if isinstance(lags, int):
            lags = range(1, lags + 1)
        if not cols:
            for i in lags:
                yield (pl.all().exclude("Időpont").shift(-i).name.suffix(f"_t+{i*15}min"))
        else:
            for i in lags:
                for colname in cols:
                    yield (pl.col(colname).shift(-i).name.suffix(f"_t+{i*15}min"))

    final_df = final_df.select(
        pl.all(),
        *timelag_expressions(40, cols_10_h),
        *timelag_expressions(48, cols_12_h),
        *timelag_expressions(range(16, 21), target_cols),
    )
    return final_df


if __name__ == "__main__":
    joined_df = pl.read_parquet("data/joined_df.parquet")
    final_df = create_training_data(joined_df)
    print(final_df)
