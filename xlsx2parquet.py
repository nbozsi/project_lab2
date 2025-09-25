import polars as pl
import polars.selectors as cs
import os
from typing import Dict

excel_folder = "excels"
tables: Dict[str, pl.DataFrame] = {}

for subfolder in os.listdir(excel_folder):
    dfs = []
    print(subfolder)
    for path in os.listdir(os.path.join(excel_folder, subfolder)):
        fullpath = os.path.join(excel_folder, subfolder, path)
        # Parse the datetime including the timezone information,
        # then convert it to "Europe/Budapest" so the saved value matches the original Excel's local time.
        df = pl.read_excel(fullpath).with_columns(
            (cs.all() - cs.by_name("Időpont")).cast(pl.Float64),
            pl.col("Időpont")
            .str.to_datetime("%Y.%m.%d %H:%M:%S %z")
            .dt.convert_time_zone("Europe/Budapest")  # Convert to Budapest timezone
            .alias("Időpont"),
        )
        dfs.append(df)
    tables[subfolder] = pl.concat(dfs).sort(by="Időpont")

out_folder = "data"
for name, df in tables.items():
    df.write_parquet(os.path.join(out_folder, f"{name}.parquet"))
