import polars as pl
from glob import glob
from pprint import pprint
from datetime import timedelta


reportType = "Neg"

path_pattern = f"data/kiegyenlito_arak/**/{reportType}_*.xlsx"

schema = {
    "Neg": {
        "Dátum": pl.Date(),
        "Kiegyenlítő-energia-elszámolási időszak": pl.String(),
        "Igényelt Negatív aFRR kiegyenlítő szabályozási energia (kWh)": pl.Float64(),
        "Negatív GCC energia (kWh)": pl.Float64(),
        "Negatív aFRR aktivált kiegyenlítő szabályozási energia (kWh)": pl.Float64(),
        "Negatív aFRR aktivált kiegyenlítő szabályozási energia díja (HUF)": pl.Float64(),
        "Negatív mFRR és RR aktivált kiegyenlítő szabályozási energia (kWh)": pl.Float64(),
        "Negatív mFRR és RR aktivált kiegyenlítő szabályozási energia díja (HUF)": pl.Float64(),
        "Negatív nemzetközi kisegítés (kWh)": pl.Float64(),
        "Negatív rendszer-irányítói menetrend-módosítás (kWh)": pl.Float64(),
        "Mérlegköri kiegyenlítő energia egységára (HUF/kWh)": pl.Float64(),
        "Rendszer-irány (kWh)": pl.Float64(),
    },
    "Poz": {
        "Dátum": pl.Date(),
        "Kiegyenlítő-energia-elszámolási időszak": pl.String(),
        "Igényelt Pozitív aFRR kiegyenlítő szabályozási energia (kWh)": pl.Float64(),
        "Pozitív GCC energia (kWh)": pl.Float64(),
        "Pozitív aFRR aktivált kiegyenlítő szabályozási energia (kWh)": pl.Float64(),
        "Pozitív aFRR aktivált kiegyenlítő szabályozási energia díja (HUF)": pl.Float64(),
        "Pozitív mFRR és RR aktivált kiegyenlítő szabályozási energia (kWh)": pl.Float64(),
        "Pozitív mFRR és RR aktivált kiegyenlítő szabályozási energia díja (HUF)": pl.Float64(),
        "Pozitív nemzetközi kisegítés (kWh)": pl.Float64(),
        "Pozitív rendszer-irányítói menetrend-módosítás (kWh)": pl.Float64(),
        "Mérlegköri kiegyenlítő energia egységára (HUF/kWh)": pl.Float64(),
        "Rendszer-irány (kWh)": pl.Float64(),
    },
}
schema = schema[reportType]

renames = {
    "Rendszer-állapot (kWh)": "Rendszer-irány (kWh)",
    "Rendszerállapot (kWh)": "Rendszer-irány (kWh)",
    "Negatív mérlegköri kiegyenlítő energia egységára (HUF/kWh)": "Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
    "Pozitív mérlegköri kiegyenlítő energia egységára (HUF/kWh)": "Mérlegköri kiegyenlítő energia egységára (HUF/kWh)",
    "Kiegyenlítő- energia-elszámolási időszak": "Kiegyenlítő-energia-elszámolási időszak",
    "Pozitív rendszerirányítói menetrend-módosítás (kWh)": "Pozitív rendszer-irányítói menetrend-módosítás (kWh)",
    "Pozitív rendszerirányítói menetrend-módosítás díja (HUF)": "Pozitív rendszer-irányítói menetrend-módosítás díja (HUF)",
}
read_opts = dict(
    header_row=6,
)

files_in_order = sorted(glob(path_pattern))

dfs = tuple(
    pl.read_excel(file, read_options=read_opts).rename(renames, strict=False).select(schema.keys()).cast(schema) for file in files_in_order
)

df = pl.concat(dfs)
df = df.with_columns(
    pl.col("Kiegyenlítő-energia-elszámolási időszak").str.slice(0, 5).str.replace(r"\+", ":").str.to_time("%H:%M"),
).with_columns(
    (
        (pl.col("Kiegyenlítő-energia-elszámolási időszak").diff() != timedelta(minutes=15))
        & (pl.col("Kiegyenlítő-energia-elszámolási időszak").dt.hour() != 0)
    ).alias("Óraátállítás"),
)
assert df.height == 2192 * 24 * 4
print(df.filter(pl.col("Óraátállítás")))
df.write_parquet(f"data/mavir_{reportType}_data.parquet")
print(df)
