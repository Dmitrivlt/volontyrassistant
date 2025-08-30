import pandas as pd

# путь к твоему файлу с историей или стримом
df = pd.read_parquet("data/binance/SOLUSDT_15m.parquet")

print("=== Первые свечи ===")
print(df.head())

print("\n=== Последние свечи ===")
print(df.tail())

print("\nВсего строк:", len(df))
print("Диапазон дат:", df["timestamp"].min(), "->", df["timestamp"].max())
