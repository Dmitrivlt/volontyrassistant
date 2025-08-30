import pandas as pd

df = pd.read_parquet("data/bybit/SOLUSDT_15m.parquet")
print("Rows:", len(df))
print("First:", df["timestamp"].min())
print("Last:", df["timestamp"].max())
