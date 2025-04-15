import polars as pl

mavir_neg = pl.read_parquet("data/mavir_Neg_data.parquet").with_columns(
    pl.col("Dátum")
    .dt.combine(pl.col("Kiegyenlítő-energia-elszámolási időszak"))
    .dt.replace_time_zone("Europe/Budapest", ambiguous="earliest", non_existent="null")
    .alias("Időpont")
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
    .alias("Időpont")
)
mavir_poz = mavir_poz.drop(["Dátum", "Kiegyenlítő-energia-elszámolási időszak"])
mavir_poz = mavir_poz.rename(
    {"Mérlegköri kiegyenlítő energia egységára (HUF/kWh)": "Pozitív Mérlegköri kiegyenlítő energia egységára (HUF/kWh)"}
)

PV = pl.read_parquet("data/PV.parquet")
hatar_aramlas = pl.read_parquet("data/hatar_aramlas.parquet")
real_time_aggregated = pl.read_parquet("data/real_time_aggregated.parquet")
rendszerterheles = pl.read_parquet("data/rendszerterheles.parquet")
wind = pl.read_parquet("data/wind.parquet")

# print(mavir_neg.shape)
# print(mavir_poz.shape)
# print(PV.shape)
# print(hatar_aramlas.shape)
# print(real_time_aggregated.shape)
# print(rendszerterheles.shape)
# print(wind.shape)
# print()

# print(mavir_neg["Időpont"].min())
# print(mavir_poz["Időpont"].min())
# print(PV["Időpont"].min())
# print(hatar_aramlas["Időpont"].min())
# print(real_time_aggregated["Időpont"].min())
# print(rendszerterheles["Időpont"].min())
# print(wind["Időpont"].min())
# print()

# print(mavir_neg["Időpont"].max())
# print(mavir_poz["Időpont"].max())
# print(PV["Időpont"].max())
# print(hatar_aramlas["Időpont"].max())
# print(real_time_aggregated["Időpont"].max())
# print(rendszerterheles["Időpont"].max())
# print(wind["Időpont"].max())
# print()

# print(f"mavir_neg Időpont type: {type(mavir_neg['Időpont'][0])}")
# print(f"mavir_poz Időpont type: {type(mavir_poz['Időpont'][0])}")
# print(f"PV Időpont type: {type(PV['Időpont'][0])}")
# print(f"hatar_aramlas Időpont type: {type(hatar_aramlas['Időpont'][0])}")
# print(f"real_time_aggregated Időpont type: {type(real_time_aggregated['Időpont'][0])}")
# print(f"rendszerterheles Időpont type: {type(rendszerterheles['Időpont'][0])}")
# print(f"wind Időpont type: {type(wind['Időpont'][0])}")
# print()

# print("mavir_neg Időpont sample:")
# print(mavir_neg.select("Időpont").head())
# print("mavir_poz Időpont sample:")
# print(mavir_poz.select("Időpont").head())
# print("PV Időpont sample:")
# print(PV.select("Időpont").head())
# print("hatar_aramlas Időpont sample:")
# print(hatar_aramlas.select("Időpont").head())
# print("real_time_aggregated Időpont sample:")
# print(real_time_aggregated.select("Időpont").head())
# print("rendszerterheles Időpont sample:")
# print(rendszerterheles.select("Időpont").head())
# print("wind Időpont sample:")
# print(wind.select("Időpont").head())

# Dictionary mapping names to dataframes
dataframes_dict = {
    "mavir_neg": mavir_neg,
    "mavir_poz": mavir_poz,
    "PV": PV,
    "hatar_aramlas": hatar_aramlas,
    "real_time_aggregated": real_time_aggregated,
    "rendszerterheles": rendszerterheles,
    "wind": wind,
}

for name, df in dataframes_dict.items():
    # Identify all columns except the join key "Időpont"
    non_id_cols = [col for col in df.columns if col != "Időpont"]
    # Sum non-null indicators for these columns; if the sum is 0, then all are null.
    non_null_sum = sum([pl.col(c).is_not_null().cast(pl.Int8) for c in non_id_cols])
    # Filter rows where at least one non-"Időpont" column is not null.
    df_clean = df.filter(non_null_sum > 0)
    # print(f"{name}: shape before cleaning {df.shape} -> after cleaning {df_clean.shape}")
    dataframes_dict[name] = df_clean

mavir_neg = dataframes_dict["mavir_neg"]
mavir_poz = dataframes_dict["mavir_poz"]
PV = dataframes_dict["PV"]
hatar_aramlas = dataframes_dict["hatar_aramlas"]
real_time_aggregated = dataframes_dict["real_time_aggregated"]
rendszerterheles = dataframes_dict["rendszerterheles"]
wind = dataframes_dict["wind"]


# List of DataFrames to join
dataframes = [mavir_neg, mavir_poz, PV, hatar_aramlas, real_time_aggregated, rendszerterheles, wind]

join_key = "Időpont"

# Sequentially join all DataFrames on the exact match of the 'Időpont' column
final_df = dataframes[0]
for df in dataframes[1:]:
    final_df = final_df.join(df, on=join_key, how="inner")

final_df = final_df.drop("Óraátállítás")

# Move 'Időpont' to the first column
final_df = final_df.select(["Időpont"] + [col for col in final_df.columns if col != "Időpont"])


final_df.write_parquet("data/joined_df.parquet")
