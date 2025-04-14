import polars as pl
from datetime import datetime, timezone

mavir_neg = pl.read_parquet("data/mavir_Neg_data.parquet").with_columns(
    pl.col("Dátum")
    .dt.combine(pl.col("Kiegyenlítő-energia-elszámolási időszak"))
    .dt.convert_time_zone(time_zone="Europe/Budapest")
    .alias("Időpont")
)
mavir_poz = pl.read_parquet("data/mavir_Poz_data.parquet").with_columns(
    pl.col("Dátum")
    .dt.combine(pl.col("Kiegyenlítő-energia-elszámolási időszak"))
    .dt.convert_time_zone(time_zone="Europe/Budapest")
    .dt.convert_time_zone("UTC")
    .alias("Időpont")
)
PV = pl.read_parquet("data/PV.parquet")
hatar_aramlas = pl.read_parquet("data/hatar_aramlas.parquet")
real_time_aggregated = pl.read_parquet("data/real_time_aggregated.parquet")
rendszerterheles = pl.read_parquet("data/rendszerterheles.parquet")
wind = pl.read_parquet("data/wind.parquet")

print(mavir_neg.shape)
print(mavir_poz.shape)
print(PV.shape)
print(hatar_aramlas.shape)
print(real_time_aggregated.shape)
print(rendszerterheles.shape)
print(wind.shape)
print()

print(mavir_neg["Időpont"].min())
print(mavir_poz["Időpont"].min())
print(PV["Időpont"].min())
print(hatar_aramlas["Időpont"].min())
print(real_time_aggregated["Időpont"].min())
print(rendszerterheles["Időpont"].min())
print(wind["Időpont"].min())
print()

print(mavir_neg["Időpont"].max())
print(mavir_poz["Időpont"].max())
print(PV["Időpont"].max())
print(hatar_aramlas["Időpont"].max())
print(real_time_aggregated["Időpont"].max())
print(rendszerterheles["Időpont"].max())
print(wind["Időpont"].max())

df = pl.concat(
    [
        mavir_neg.select(pl.col("Időpont").alias("Időpont_")),
        PV.select(pl.col("Időpont")),
    ],
    how="horizontal",
)

# for year in range(2019, 2025):
#    # print(PV.filter(pl.col("Naperőművek nettó üzemirányítási").is_null()))
#    print(
#        PV.filter(
#            (pl.col("Időpont").dt.year() == year)
#            & (pl.col("Időpont").dt.month() == 12)
#            & (pl.col("Időpont").dt.day() == 31)
#            & (pl.col("Időpont").dt.hour() == 23)
#            & (pl.col("Időpont").dt.minute() == 15)
#        )
#    )

print(df.filter(pl.col("Időpont") == pl.col("Időpont_")).tail())
print(df.filter(pl.col("Időpont") != pl.col("Időpont_")).head())
