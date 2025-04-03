import polars as pl


df = pl.read_excel("data/kiegyenlito_arak_mappa/**/Poz_*.xlsx")
print(df.shape)
