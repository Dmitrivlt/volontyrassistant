#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZigZag swings (ATR-or-% mode, or Percent-only mode)
- Вход: OHLCV .parquet из data_pipeline.py
- Режимы:
  * --mode atr : порог = max(k_atr * ATR, k_pct * close), с авто-ослаблением
  * --mode pct : простой % порог по close (очень мягкий, чтобы зажечь поток свингов)
- Выход: .parquet со свингами (timestamp, type: H/L, price)

Примеры:
  python zigzag_atr.py --input data/bybit/SOLUSDT_15m.parquet --mode pct --k_pct 0.001 --min_bars_gap 1
  python zigzag_atr.py --input data/bybit/SOLUSDT_15m.parquet --mode atr --atr_n 14 --k_atr 1.2 --k_pct 0.003 --min_bars_gap 2 --target_min 150
"""

import argparse, os, sys
import numpy as np
import pandas as pd

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- базовые вычисления ----------

def true_range(df: pd.DataFrame) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs()
    ], axis=1).max(axis=1)
    return tr.to_numpy()

def atr_series(df: pd.DataFrame, n: int) -> np.ndarray:
    tr = true_range(df)
    atr = pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()
    return atr

# ---------- режим ATR-or-% (гибридный) ----------

def zigzag_close_atr_or_pct(df: pd.DataFrame, atr_n: int, k_atr: float, k_pct: float,
                            min_bars_gap: int) -> pd.DataFrame:
    """
    Разворот по CLOSE фиксируется, когда откат от экстремума >= max(k_atr*ATR[i], k_pct*close[i]).
    Возвращает df[timestamp, type(H/L), price].
    """
    d = df.sort_values("timestamp").reset_index(drop=True)
    if len(d) < max(atr_n + 2, 50):
        return pd.DataFrame(columns=["timestamp","type","price"])

    atr = atr_series(d, atr_n)
    close = d["close"].to_numpy()
    ts = d["timestamp"].to_numpy()

    # стартовый индекс, где ATR валиден
    valid = np.where(~np.isnan(atr))[0]
    if len(valid) == 0:
        return pd.DataFrame(columns=["timestamp","type","price"])
    start = int(valid[0])
    start = max(start, atr_n)

    # начальный тренд по наклону close
    if start + 1 >= len(d):
        return pd.DataFrame(columns=["timestamp","type","price"])
    trend = "up" if close[start+1] >= close[start] else "down"
    ext_idx = start
    ext_price = close[ext_idx]
    last_pivot_i = None

    pivots = []  # (idx, "H"/"L", price)

    for i in range(start+1, len(d)):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        thr = max(k_atr * a, k_pct * close[i])

        if trend == "up":
            # обновляем максимум по close
            if close[i] > ext_price:
                ext_price, ext_idx = close[i], i
            # разворот вниз (откат от max)
            if (ext_price - close[i]) >= thr:
                if last_pivot_i is None or (i - last_pivot_i) >= min_bars_gap:
                    pivots.append((ext_idx, "H", float(ext_price)))
                    last_pivot_i = i
                    trend = "down"
                    ext_price, ext_idx = close[i], i
        else:  # down
            if close[i] < ext_price:
                ext_price, ext_idx = close[i], i
            # разворот вверх (откат от min)
            if (close[i] - ext_price) >= thr:
                if last_pivot_i is None or (i - last_pivot_i) >= min_bars_gap:
                    pivots.append((ext_idx, "L", float(ext_price)))
                    last_pivot_i = i
                    trend = "up"
                    ext_price, ext_idx = close[i], i

    if not pivots:
        return pd.DataFrame(columns=["timestamp","type","price"])

    # Очистим подряд одинаковые типы, оставляя более "крайний"
    cleaned = []
    for idx, t, p in sorted(pivots, key=lambda x: x[0]):
        if cleaned and cleaned[-1][1] == t:
            ci, ct, cp = cleaned[-1]
            if (t == "H" and p > cp) or (t == "L" and p < cp):
                cleaned[-1] = (idx, t, p)
        else:
            cleaned.append((idx, t, p))

    out = pd.DataFrame([{"timestamp": int(ts[i]), "type": t, "price": float(p)} for i, t, p in cleaned])
    return out.reset_index(drop=True)

def auto_relax_atr(df: pd.DataFrame, atr_n: int,
                   k_atr: float, k_pct: float, min_bars_gap: int,
                   target_min: int = 100, max_steps: int = 8) -> pd.DataFrame:
    """
    Авто-ослабление порогов в режиме ATR-or-%:
    1) уменьшаем k_atr
    2) уменьшаем k_pct
    3) уменьшаем gap
    """
    cur_k_atr = k_atr
    cur_k_pct = k_pct
    cur_gap = min_bars_gap

    for step in range(max_steps + 1):
        swings = zigzag_close_atr_or_pct(df, atr_n, cur_k_atr, cur_k_pct, cur_gap)
        print(f"[debug] step {step}: k_atr={cur_k_atr:.4f}, k_pct={cur_k_pct:.5f}, gap={cur_gap}, swings={len(swings)}")
        if len(swings) >= target_min or step == max_steps:
            return swings

        # ослабляем
        if cur_k_atr > 0.2:
            cur_k_atr = round(max(0.2, cur_k_atr * 0.7), 4)
        elif cur_k_pct > 0.0005:
            cur_k_pct = round(max(0.0005, cur_k_pct * 0.7), 5)
        elif cur_gap > 1:
            cur_gap = cur_gap - 1
        else:
            continue

# ---------- режим Percent-only (очень мягкий) ----------

def zigzag_percent(df: pd.DataFrame, k_pct: float = 0.001, min_bars_gap: int = 1) -> pd.DataFrame:
    """
    Простейший ZigZag по CLOSE: разворот, если |close - экстремум| >= k_pct * close.
    Возвращает df[timestamp, type(H/L), price].
    """
    d = df.sort_values("timestamp").reset_index(drop=True)
    if len(d) < 3:
        return pd.DataFrame(columns=["timestamp","type","price"])

    close = d["close"].to_numpy()
    ts = d["timestamp"].to_numpy()

    trend = "up" if close[1] >= close[0] else "down"
    ext_idx = 0
    ext_price = close[0]
    last_pivot_i = None

    pivots = []

    for i in range(1, len(d)):
        thr = k_pct * close[i]

        if trend == "up":
            if close[i] > ext_price:
                ext_price, ext_idx = close[i], i
            if (ext_price - close[i]) >= thr:
                if last_pivot_i is None or (i - last_pivot_i) >= min_bars_gap:
                    pivots.append((ext_idx, "H", float(ext_price)))
                    last_pivot_i = i
                    trend = "down"
                    ext_price, ext_idx = close[i], i
        else:
            if close[i] < ext_price:
                ext_price, ext_idx = close[i], i
            if (close[i] - ext_price) >= thr:
                if last_pivot_i is None or (i - last_pivot_i) >= min_bars_gap:
                    pivots.append((ext_idx, "L", float(ext_price)))
                    last_pivot_i = i
                    trend = "up"
                    ext_price, ext_idx = close[i], i

    if not pivots:
        return pd.DataFrame(columns=["timestamp","type","price"])

    # Очистка подряд одинаковых типов
    cleaned = []
    for idx, t, p in sorted(pivots, key=lambda x: x[0]):
        if cleaned and cleaned[-1][1] == t:
            ci, ct, cp = cleaned[-1]
            if (t == "H" and p > cp) or (t == "L" and p < cp):
                cleaned[-1] = (idx, t, p)
        else:
            cleaned.append((idx, t, p))

    out = pd.DataFrame([{"timestamp": int(ts[i]), "type": t, "price": float(p)} for i, t, p in cleaned])
    return out.reset_index(drop=True)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="ZigZag swings (ATR-or-% / Percent-only)")
    ap.add_argument("--input", required=True, help="data/<exchange>/<SYMBOL>_<interval>.parquet")
    ap.add_argument("--output", help="по умолчанию: рядом в папке swings/")
    ap.add_argument("--mode", choices=["atr","pct"], default="pct", help="atr = гибридный, pct = процентный")
    # параметры ATR-режима
    ap.add_argument("--atr_n", type=int, default=14)
    ap.add_argument("--k_atr", type=float, default=2.0)
    ap.add_argument("--k_pct", type=float, default=0.002)
    ap.add_argument("--min_bars_gap", type=int, default=3)
    ap.add_argument("--target_min", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if df.empty:
        print("Файл со свечами пуст.")
        sys.exit(0)

    if args.mode == "atr":
        swings = auto_relax_atr(df, args.atr_n, args.k_atr, args.k_pct, args.min_bars_gap,
                                target_min=args.target_min, max_steps=8)
    else:
        # percent-only
        swings = zigzag_percent(df, k_pct=args.k_pct, min_bars_gap=args.min_bars_gap)
        print(f"[debug] pct-mode: swings={len(swings)} (k_pct={args.k_pct}, gap={args.min_bars_gap})")

    # сохранение
    if args.output:
        out_path = args.output
    else:
        base = os.path.dirname(args.input)
        ensure_dir(os.path.join(base, "swings"))
        fname = os.path.basename(args.input).replace(".parquet", "_swings.parquet")
        out_path = os.path.join(base, "swings", fname)

    swings.to_parquet(out_path, index=False)
    print(f"Свинги сохранены: {out_path} | всего: {len(swings)}")

if __name__ == "__main__":
    main()
