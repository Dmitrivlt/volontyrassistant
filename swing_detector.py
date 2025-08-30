#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Swing Detector (упрощённый - без фильтра ATR/gap)
- Вход: OHLCV из .parquet (data_pipeline.py)
- Выход: .parquet со свингами (H/L точки)
"""

import argparse, os, sys
import numpy as np
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def true_range(df: pd.DataFrame) -> np.ndarray:
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
         np.maximum((df["high"] - prev_close).abs(),
                    (df["low"]  - prev_close).abs()))
    return tr.to_numpy()

def atr_series(df: pd.DataFrame, n: int) -> np.ndarray:
    tr = true_range(df)
    atr = pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()
    return atr

def local_extrema_indices(high: np.ndarray, low: np.ndarray, lb: int) -> tuple[list[int], list[int]]:
    """
    Пивот High в i: high[i] > max(high[i-lb:i]) и high[i] >= max(high[i+1:i+1+lb])
    Пивот Low  в i: low[i]  < min(low[i-lb:i])  и low[i]  <= min(low[i+1:i+1+lb])
    Требует i-lb >= 0 и i+lb < n.
    """
    n = len(high)
    pivH, pivL = [], []
    for i in range(lb, n - lb):
        leftH  = high[i-lb:i]
        rightH = high[i+1:i+1+lb]
        leftL  = low[i-lb:i]
        rightL = low[i+1:i+1+lb]
        h = high[i]; l = low[i]
        if h > leftH.max() and h >= rightH.max():
            pivH.append(i)
        if l < leftL.min() and l <= rightL.min():
            pivL.append(i)
    return pivH, pivL

def build_raw_pivots(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    high = df["high"].to_numpy()
    low  = df["low"].to_numpy()
    pivH, pivL = local_extrema_indices(high, low, lookback)
    rows = []
    for i in pivH:
        rows.append({"idx": i, "timestamp": int(df["timestamp"].iloc[i]), "type": "H", "price": float(high[i])})
    for i in pivL:
        rows.append({"idx": i, "timestamp": int(df["timestamp"].iloc[i]), "type": "L", "price": float(low[i])})

    if not rows:  # если нет пивотов на заданном lookback
        return pd.DataFrame(columns=["idx","timestamp","type","price"])

    piv = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    return piv

# УПРОЩЁННО: возвращаем все пивоты без фильтрации ATR/gap
def passthrough(piv: pd.DataFrame) -> pd.DataFrame:
    return piv.reset_index(drop=True) if not piv.empty else piv

def main():
    ap = argparse.ArgumentParser(description="Swing Detector (упрощённый)")
    ap.add_argument("--input", required=True, help="data/<exchange>/<SYMBOL>_<interval>.parquet")
    ap.add_argument("--output", help="swings/<...>_swings.parquet (по умолчанию рядом)")
    ap.add_argument("--lookback", type=int, default=1, help="окно фрактала (1..5)")
    ap.add_argument("--atr_n", type=int, default=14)  # оставляем на будущее
    args = ap.parse_args()

    df = pd.read_parquet(args.input).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        print("Файл со свечами пуст.")
        sys.exit(0)

    piv_raw = build_raw_pivots(df, args.lookback)
    print(f"[debug] найдено пивотов (сырых): {len(piv_raw)} при lookback={args.lookback}")

    piv = passthrough(piv_raw)
    print(f"[debug] сохранится пивотов: {len(piv)}")

    out = piv[["timestamp","type","price"]].copy() if not piv.empty else pd.DataFrame(columns=["timestamp","type","price"])

    if args.output:
        out_path = args.output
    else:
        base = os.path.dirname(args.input)
        ensure_dir(os.path.join(base, "swings"))
        fname = os.path.basename(args.input).replace(".parquet", "_swings.parquet")
        out_path = os.path.join(base, "swings", fname)

    out.to_parquet(out_path, index=False)
    print(f"Свинги сохранены в {out_path} | всего: {len(out)}")

if __name__ == "__main__":
    main()
