import polars as pl
from english_terms import hun2eng


COMMON_DATECOL_NAME = "UTCdate"
mavir_neg = pl.read_parquet("data/mavir_Neg_data.parquet").with_columns(
    pl.col("Dátum")
    .dt.combine(pl.col("Kiegyenlítő-energia-elszámolási időszak"))
    .dt.replace_time_zone("Europe/Budapest", ambiguous="earliest", non_existent="null")
    .alias(COMMON_DATECOL_NAME)
)
# Remove the original columns "Dátum" and "Kiegyenlítő-energia-elszámolási időszak"
mavir_neg = mavir_neg.drop(["Dátum", "Kiegyenlítő-energia-elszámolási időszak"])
mavir_neg = mavir_neg.rename(
    {"Mérlegköri kiegyenlítő energia egységára (HUF/kWh)": "Negatív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)"}
)
# Remove one of the Rendszer-irány (kWh) columns, in this case, from neg, and keep it in poz, since they are the same
mavir_neg = mavir_neg.drop(["Rendszer-irány (kWh)"])

mavir_poz = pl.read_parquet("data/mavir_Poz_data.parquet").with_columns(
    pl.col("Dátum")
    .dt.combine(pl.col("Kiegyenlítő-energia-elszámolási időszak"))
    .dt.replace_time_zone("Europe/Budapest", ambiguous="earliest", non_existent="null")
    # .dt.convert_time_zone("UTC")
    .alias(COMMON_DATECOL_NAME)
)
mavir_poz = mavir_poz.drop(["Dátum", "Kiegyenlítő-energia-elszámolási időszak"])
mavir_poz = mavir_poz.rename(
    {"Mérlegköri kiegyenlítő energia egységára (HUF/kWh)": "Pozitív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)"}
)

PV = pl.read_parquet("data/PV.parquet").rename({"Időpont": "UTCdate"})
hatar_aramlas = pl.read_parquet("data/hatar_aramlas.parquet").rename({"Időpont": "UTCdate"})
real_time_aggregated = pl.read_parquet("data/real_time_aggregated.parquet").rename({"Időpont": "UTCdate"})
rendszerterheles = pl.read_parquet("data/rendszerterheles.parquet").rename({"Időpont": "UTCdate"})
wind = pl.read_parquet("data/wind.parquet").rename({"Időpont": "UTCdate"})


def delete_cols_with_nulls(df, null_th=0.1):
    # Identify all columns except the join key COMMON_DATECOL_NAME
    non_id_cols = [col for col in df.columns if col != COMMON_DATECOL_NAME]

    # Sum non-null indicators for these columns; if the sum is 0, then all are null.
    cols_with_nulls = list(col for col, nc in df.select(pl.all().null_count()).to_dict().items() if nc[0] / df.height > null_th)
    print(cols_with_nulls)
    # Filter rows where at least one non-COMMON_DATECOL_NAME column is not null.
    df_clean = df.drop(cols_with_nulls)

    # print(f"{name}: shape before cleaning {df.shape} -> after cleaning {df_clean.shape}")
    return df_clean.fill_null(strategy="forward")


mavir_neg = delete_cols_with_nulls(mavir_neg)
mavir_poz = delete_cols_with_nulls(mavir_poz)
PV = delete_cols_with_nulls(PV)
hatar_aramlas = delete_cols_with_nulls(hatar_aramlas)
real_time_aggregated = delete_cols_with_nulls(real_time_aggregated)
rendszerterheles = delete_cols_with_nulls(rendszerterheles)
wind = delete_cols_with_nulls(wind)

# List of DataFrames to join
dataframes = [mavir_neg, mavir_poz, PV, hatar_aramlas, real_time_aggregated, rendszerterheles, wind]
joined_df = dataframes[0].with_columns(pl.col(COMMON_DATECOL_NAME).cum_count().over(COMMON_DATECOL_NAME).alias("Óraátállítás"))
for df in dataframes[1:]:
    joined_df = joined_df.join(
        df.with_columns(pl.col(COMMON_DATECOL_NAME).cum_count().over(COMMON_DATECOL_NAME).alias("Óraátállítás")),
        on=(COMMON_DATECOL_NAME, "Óraátállítás"),
        how="inner",
    )
    print(joined_df.height)


# Move 'Időpont' to the first column
joined_df = joined_df.select([COMMON_DATECOL_NAME] + [col for col in joined_df.columns if col != COMMON_DATECOL_NAME])

joined_df.write_parquet("data/joined_df_HU.parquet")
joined_df.rename(hun2eng).write_parquet("data/joined_df.parquet")
