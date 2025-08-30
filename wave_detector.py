#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wave Detector (LONG-only после 2 и 4) на базе свингов ZigZag.
Вход:
  - OHLCV .parquet (из data_pipeline.py) — для ATR/проверок
  - Swings .parquet (из zigzag_atr.py) — колонки: timestamp, type ('H'/'L'), price

Выход:
  - Signals .parquet: входы после 2 и 4 c уровнями entry/stop/tp, RR, опорными точками волн
"""

import argparse, os
import pandas as pd
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    return atr

def nearest_atr(df_ohlcv: pd.DataFrame, ts: int, atr_col: str = "atr") -> float:
    # берём ATR на (или перед) timestamp сигнала
    i = df_ohlcv["timestamp"].searchsorted(ts, side="right") - 1
    if i < 0: 
        return float("nan")
    return float(df_ohlcv[atr_col].iloc[i])

def load_swings(path: str) -> pd.DataFrame:
    sw = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    assert set(["timestamp","type","price"]).issubset(sw.columns), \
        "Swings parquet должен иметь колонки: timestamp,type,price"
    # оставим только нужные столбцы и типы
    sw = sw[["timestamp","type","price"]].copy()
    sw = sw[sw["type"].isin(["H","L"])]
    return sw.reset_index(drop=True)

def detect_bull_impulses(sw: pd.DataFrame,
                         overlap_tol: float = 0.0,
                         retr2_rng = (0.236, 0.786),
                         retr4_rng = (0.236, 0.786),
                         min_len3_vs1: float = 0.618) -> pd.DataFrame:
    """
    Ищем L-H-L-H-L шаблоны и валидируем их базовыми правилами.
    Возвращает DataFrame с колонками волновых точек и рассчитанными метриками.
    """
    rows = []
    # оставим индексы только чёткой очередности L/H
    sw_clean = sw.copy()
    # пробегаем окно из 5 свингов
    for i in range(len(sw_clean) - 4):
        a = sw_clean.iloc[i + 0]
        b = sw_clean.iloc[i + 1]
        c = sw_clean.iloc[i + 2]
        d = sw_clean.iloc[i + 3]
        e = sw_clean.iloc[i + 4]
        # шаблон L-H-L-H-L
        if not (a["type"] == "L" and b["type"] == "H" and c["type"] == "L" and d["type"] == "H" and e["type"] == "L"):
            continue

        L1_t, L1_p = int(a["timestamp"]), float(a["price"])
        H1_t, H1_p = int(b["timestamp"]), float(b["price"])
        L2_t, L2_p = int(c["timestamp"]), float(c["price"])
        H3_t, H3_p = int(d["timestamp"]), float(d["price"])
        L4_t, L4_p = int(e["timestamp"]), float(e["price"])

        # монотонность/базовые
        if not (H1_p > L1_p and H3_p > H1_p and L2_p > L1_p and L4_p > L2_p):
            continue

        # непересечение 1/4 (с допуском overlap_tol)
        if L4_p <= H1_p * (1 - overlap_tol):
            continue

        # фибо-откаты
        denom1 = (H1_p - L1_p)
        denom3 = (H3_p - L2_p)
        if denom1 <= 0 or denom3 <= 0:
            continue
        retr2 = (H1_p - L2_p) / denom1
        retr4 = (H3_p - L4_p) / denom3
        if not (retr2_rng[0] <= retr2 <= retr2_rng[1]):
            continue
        if not (retr4_rng[0] <= retr4 <= retr4_rng[1]):
            continue

        # импульсность 3
        len1 = (H1_p - L1_p)
        len3 = (H3_p - L2_p)
        if len3 < min_len3_vs1 * len1:
            continue

        rows.append({
            "L1_t": L1_t, "L1_p": L1_p,
            "H1_t": H1_t, "H1_p": H1_p,
            "L2_t": L2_t, "L2_p": L2_p,
            "H3_t": H3_t, "H3_p": H3_p,
            "L4_t": L4_t, "L4_p": L4_p,
            "retr2": retr2, "retr4": retr4,
            "len1": len1, "len3": len3
        })

    return pd.DataFrame(rows)

def build_long_signals(df_ohlcv: pd.DataFrame,
                       impulses: pd.DataFrame,
                       rr: float = 2.0,
                       atr_pad: float = 0.25,
                       atr_n: int = 14,
                       confirm_break: bool = True) -> pd.DataFrame:
    """
    На основе импульсных шаблонов строим два типа сигналов:
      - After2: вход после окончания 2 (пробой над H1)
      - After4: вход после окончания 4 (пробой над H3)
    entry = уровень пробоя (close/High), SL = L2/L4 - atr_pad*ATR, TP = entry + RR*(entry - SL)
    """
    df = df_ohlcv.sort_values("timestamp").reset_index(drop=True).copy()
    df["atr"] = atr_series(df, n=atr_n)

    sig_rows = []

    for _, r in impulses.iterrows():
        # --- после 2 ---
        entry2_level = r["H1_p"]
        ts2 = int(r["L2_t"])
        atr2 = nearest_atr(df, ts2, "atr")
        sl2 = r["L2_p"] - atr_pad * atr2 if np.isfinite(atr2) else r["L2_p"]
        if sl2 <= 0 or entry2_level <= sl2:
            pass
        else:
            sig_rows.append({
                "when": ts2, "type": "LONG_after2",
                "entry": float(entry2_level),
                "stop": float(sl2),
                "tp": float(entry2_level + rr * (entry2_level - sl2)),
                "RR": rr,
                "L1_t": int(r["L1_t"]), "L1_p": float(r["L1_p"]),
                "H1_t": int(r["H1_t"]), "H1_p": float(r["H1_p"]),
                "L2_t": int(r["L2_t"]), "L2_p": float(r["L2_p"]),
                "H3_t": int(r["H3_t"]), "H3_p": float(r["H3_p"]),
                "L4_t": int(r["L4_t"]), "L4_p": float(r["L4_p"]),
            })

        # --- после 4 ---
        entry4_level = r["H3_p"]
        ts4 = int(r["L4_t"])
        atr4 = nearest_atr(df, ts4, "atr")
        sl4 = r["L4_p"] - atr_pad * atr4 if np.isfinite(atr4) else r["L4_p"]
        if sl4 <= 0 or entry4_level <= sl4:
            pass
        else:
            sig_rows.append({
                "when": ts4, "type": "LONG_after4",
                "entry": float(entry4_level),
                "stop": float(sl4),
                "tp": float(entry4_level + rr * (entry4_level - sl4)),
                "RR": rr,
                "L1_t": int(r["L1_t"]), "L1_p": float(r["L1_p"]),
                "H1_t": int(r["H1_t"]), "H1_p": float(r["H1_p"]),
                "L2_t": int(r["L2_t"]), "L2_p": float(r["L2_p"]),
                "H3_t": int(r["H3_t"]), "H3_p": float(r["H3_p"]),
                "L4_t": int(r["L4_t"]), "L4_p": float(r["L4_p"]),
            })

    return pd.DataFrame(sig_rows).sort_values(["when","type"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Wave Detector (LONG-only after 2/4)")
    ap.add_argument("--ohlcv", required=True, help="Путь к OHLCV .parquet (для ATR/контекста)")
    ap.add_argument("--swings", required=True, help="Путь к Swings .parquet (из zigzag_atr.py)")
    ap.add_argument("--output", help="Путь для Signals .parquet; по умолчанию рядом: signals/<...>_signals.parquet")
    # настройки валидации
    ap.add_argument("--overlap_tol", type=float, default=0.0)
    ap.add_argument("--retr2_min", type=float, default=0.236)
    ap.add_argument("--retr2_max", type=float, default=0.786)
    ap.add_argument("--retr4_min", type=float, default=0.236)
    ap.add_argument("--retr4_max", type=float, default=0.786)
    ap.add_argument("--min_len3_vs1", type=float, default=0.618)
    # риск/таргет
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--atr_pad", type=float, default=0.25)
    ap.add_argument("--atr_n", type=int, default=14)
    args = ap.parse_args()

    df = pd.read_parquet(args.ohlcv).sort_values("timestamp").reset_index(drop=True)
    sw = load_swings(args.swings)

    impulses = detect_bull_impulses(
        sw,
        overlap_tol=args.overlap_tol,
        retr2_rng=(args.retr2_min, args.retr2_max),
        retr4_rng=(args.retr4_min, args.retr4_max),
        min_len3_vs1=args.min_len3_vs1
    )

    signals = build_long_signals(
        df, impulses,
        rr=args.rr, atr_pad=args.atr_pad, atr_n=args.atr_n
    )

    if args.output:
        out_path = args.output
    else:
        base = os.path.dirname(args.swings)
        ensure_dir(os.path.join(base, "..", "waves"))
        # имя берём из swings-файла
        fname = os.path.basename(args.swings).replace("_swings.parquet", "_signals.parquet")
        out_path = os.path.join(base, "..", "waves", fname)

    signals.to_parquet(out_path, index=False)
    print(f"Найдено импульсных шаблонов: {len(impulses)}")
    print(f"Сигналов (LONG after 2/4):   {len(signals)}")
    print(f"Сохранено: {out_path}")

if __name__ == "__main__":
    main()
