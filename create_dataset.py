import polars as pl

from era5csv_reader import read_all_era5
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

# Add new time-based features
joined_df = joined_df.with_columns([
    # Hour of day
    pl.col(COMMON_DATECOL_NAME).dt.hour().alias("hour"),
    
    # Day of week (0=Monday, 6=Sunday) with sin and cosine encoding
    pl.col(COMMON_DATECOL_NAME).dt.weekday().alias("dayofweek"),
    (pl.col(COMMON_DATECOL_NAME).dt.weekday() * 2 * pl.lit(3.14159) / 7).sin().alias("dayofweek_sin"),
    (pl.col(COMMON_DATECOL_NAME).dt.weekday() * 2 * pl.lit(3.14159) / 7).cos().alias("dayofweek_cos"),
    
    # Day of year
    (pl.col(COMMON_DATECOL_NAME).dt.ordinal_day() / 366).alias("day_of_year"),
    
    # Hungarian holidays (hardcoded dates including movable holidays)
    pl.when(
        # Fixed holidays
        (pl.col(COMMON_DATECOL_NAME).dt.month() == 1) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 1) |  # New Year's Day
        (pl.col(COMMON_DATECOL_NAME).dt.month() == 3) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 15) |  # Revolution Day
        (pl.col(COMMON_DATECOL_NAME).dt.month() == 5) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 1) |  # Labour Day
        (pl.col(COMMON_DATECOL_NAME).dt.month() == 8) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 20) |  # St Stephen's Day
        (pl.col(COMMON_DATECOL_NAME).dt.month() == 10) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 23) |  # Republic Day
        (pl.col(COMMON_DATECOL_NAME).dt.month() == 11) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 1) |  # All Saints
        (pl.col(COMMON_DATECOL_NAME).dt.month() == 12) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([25, 26])) |  # Christmas
        
        # 2019 movable holidays
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2019) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 4) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([19, 21, 22])) |  # Good Friday, Easter Sunday, Easter Monday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2019) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 6) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([9, 10])) |  # Pentecost Sunday, Pentecost Monday
        
        # 2020 movable holidays
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2020) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 4) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([10, 12, 13])) |  # Good Friday, Easter Sunday, Easter Monday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2020) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 5) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 31) |  # Pentecost Sunday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2020) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 6) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 1) |  # Pentecost Monday
        
        # 2021 movable holidays
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2021) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 4) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([2, 4, 5])) |  # Good Friday, Easter Sunday, Easter Monday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2021) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 5) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([23, 24])) |  # Pentecost Sunday, Pentecost Monday
        
        # 2022 movable holidays
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2022) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 3) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 14) |  # Bridge day
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2022) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 4) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([15, 17, 18])) |  # Good Friday, Easter Sunday, Easter Monday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2022) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 6) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([5, 6])) |  # Pentecost Sunday, Pentecost Monday
        
        # 2023 movable holidays
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2023) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 4) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([7, 9, 10])) |  # Good Friday, Easter Sunday, Easter Monday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2023) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 5) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 29) |  # Pentecost Monday
        
        # 2024 movable holidays
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2024) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 3) & (pl.col(COMMON_DATECOL_NAME).dt.day().is_in([29, 31])) |  # Good Friday, Easter Sunday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2024) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 4) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 1) |  # Easter Monday
        (pl.col(COMMON_DATECOL_NAME).dt.year() == 2024) & (pl.col(COMMON_DATECOL_NAME).dt.month() == 5) & (pl.col(COMMON_DATECOL_NAME).dt.day() == 20)  # Pentecost Monday
    ).then(1).otherwise(0).alias("is_holiday")
])

joined_df.write_parquet("data/joined_df_HU.parquet")
joined_df = joined_df.rename(hun2eng)
joined_df.write_parquet("data/joined_df.parquet")


# adding weather data
weather_data = read_all_era5("data/era5_weather")

# creating 1 with only adding temperature
joined_df_temp = joined_df.join(weather_data.select("UTCdate", "temp_mean", "temp_std"), on=COMMON_DATECOL_NAME, how="left").write_parquet(
    "data/joined_df_with_temp.parquet"
)

joined_df_weather = joined_df.join(weather_data, on=COMMON_DATECOL_NAME, how="left")
print(joined_df.height)
joined_df_weather.write_parquet("data/joined_df_with_weather.parquet")
