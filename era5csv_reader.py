import polars as pl
import polars.selectors as cs
from glob import glob
import os.path


def read_era5_csv(path):
    df = pl.read_csv(path, try_parse_dates=True).drop(cs.starts_with("expver", "number"))

    return df.rename({"valid_time": "datetime"}, strict=False)

    # df =
    return df


def read_all_era5(dir):
    dfs = list(map(read_era5_csv, glob(os.path.join(dir, "*.csv"))))

    joined_df = dfs[0]
    for df in dfs:
        joined_df = joined_df.join(df, on="datetime", how="inner")

    return (
        joined_df.upsample("datetime", every="15m")
        .fill_null(strategy="forward")
        .with_columns(
            pl.col("datetime").dt.replace_time_zone("Europe/Budapest", ambiguous="earliest", non_existent="null").alias("UTCdate")
        )
        .drop("datetime")
    )


if __name__ == "__main__":
    df = read_all_era5("data/era5_weather")
