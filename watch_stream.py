import time
import pandas as pd

PATH = "data/bybit/SOLUSDT_15m.parquet"  # замени путь, если у тебя Binance или другой символ

def watch_file():
    last_ts = None
    while True:
        try:
            df = pd.read_parquet(PATH)
            if not df.empty:
                last_rows = df.tail(3)
                ts_new = last_rows["timestamp"].iloc[-1]
                if ts_new != last_ts:
                    print("="*40)
                    print("Обновление:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print(last_rows)
                    last_ts = ts_new
        except Exception as e:
            print("Ошибка чтения:", e)
        time.sleep(10)  # проверяем каждые 10 секунд

if __name__ == "__main__":
    watch_file()
