import glob
import xarray as xr
import polars as pl


# Open all downloaded files as a single xarray dataset
files = sorted(glob.glob("./era5/era5_hungary_*.nc"))
ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")

# Convert 2m temperature from Kelvin to Celsius
temp_c = ds["t2m"] - 273.15

# Build a dataset directly from temp_c
ds_reduced = xr.Dataset({"temp_avg": temp_c.mean(dim=["latitude", "longitude"]), "temp_std": temp_c.std(dim=["latitude", "longitude"])})

df = ds_reduced.to_dataframe().reset_index()

df = pl.from_dataframe(df).with_columns(
    pl.col("valid_time").dt.cast_time_unit("us").dt.replace_time_zone("Europe/Budapest", ambiguous="earliest", non_existent="null")
)

joined_time_col = pl.read_parquet("data/joined_df.parquet", columns="UTCdate")
print(joined_time_col.height)
df = df.sort(by="valid_time").upsample("valid_time", every="15m", maintain_order=True).select(pl.all().fill_null(strategy="forward"))
print(df.height)


df.write_parquet("hungary_hourly_temperature_2019_2024.parquet")
joined = pl.read_parquet("data/joined_df.parquet")
full = df.join(joined, how="inner", left_on="valid_time", right_on="UTCdate")
full.write_parquet("data/joined_df_with_weather.parquet")
