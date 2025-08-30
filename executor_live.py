#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live Executor (Bybit Testnet) — LONG only after 2/4

- Читает OHLCV и сигналы (parquet).
- На закрытии нового бара 15m: если high>=entry и approve=auto -> Market Buy.
- Выставляет:
   * TP_PARTIAL: reduceOnly лимит, 50% по цене entry + 2R
   * TP_FINAL:   reduceOnly лимит, остаток по tp
   * SL_INIT:    reduceOnly лимит по stop (упрощённо для тестнета)
- BE: при достижении 2R — SL переносится на entry, TP_FINAL восстанавливается.
- Округление цен/кол-ва по спецификации (tick_size / qty_step / min_order_qty).
- Авто-ограничение размера позиции по доступной марже, чтобы не ловить 110007.

ENV: BYBIT_API_KEY, BYBIT_API_SECRET
Approve: runtime/control.json => {"approve_mode":"auto"} или "manual"
"""

from __future__ import annotations
import os, time, json, argparse
import pandas as pd
from exchange_adapter import BybitAdapter, BybitError

# ---------- IO ----------

def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def read_control(path: str) -> dict:
    if not os.path.exists(path):
        return {"approve_mode": "auto"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"approve_mode": "auto"}

def read_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    need = {"timestamp", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"OHLCV должен содержать {need}")
    return df

def read_signals(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("when").reset_index(drop=True)
    need = {"when", "type", "entry", "stop", "tp"}
    if not need.issubset(df.columns):
        raise ValueError(f"Signals должен содержать {need}")
    df = df[df["type"].isin(["LONG_after2", "LONG_after4"])].reset_index(drop=True)
    return df

def price_with_slip(price: float, bps: float, side: str) -> float:
    slip = bps / 10_000.0
    return price * (1 + slip) if side == "buy" else price * (1 - slip)

# ---------- Live Executor ----------

def run_live(ohlcv_path: str,
             signals_path: str,
             symbol: str,
             initial_capital: float,
             risk_input: float,
             risk_unit: str,
             max_concurrent: int,
             leverage: float,
             fee_bps: float,
             slip_bps: float,
             control_path: str,
             poll_sec: float = 3.0):
    # ключи
    api_key = os.environ.get("BYBIT_API_KEY")
    api_secret = os.environ.get("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Нужно экспортировать BYBIT_API_KEY и BYBIT_API_SECRET (тестнет ключи).")

    bybit = BybitAdapter(api_key, api_secret)

    # спецификация инструмента
    tick_size, qty_step, min_qty, min_notional = bybit.get_symbol_spec(symbol)
    print(f"[SPEC] {symbol} tick={tick_size} qty_step={qty_step} min_qty={min_qty} min_notional={min_notional}")

    # плечо (не падаем, если уже установлено)
    bybit.set_leverage(symbol, buyLeverage=leverage)

    ohlcv = read_ohlcv(ohlcv_path)
    signals = read_signals(signals_path)
    if "active" not in signals.columns:
        signals["active"] = False

    opened = []  # [{entry_fill, stop_eff, tp_eff, qty, be_active}]
    equity = initial_capital
    last_rows = len(ohlcv)

    print(f"[LIVE] start: rows={last_rows}, eq={equity:.2f}, sym={symbol}, approve={read_control(control_path).get('approve_mode')}")

    # параметры сопровождения
    partial_frac = 0.5
    partial_tpR = 2.0
    be_fromR = 2.0

    def risk_per_trade() -> float:
        slots = max(1, min(int(max_concurrent), 10))
        if risk_unit == "pct":
            return equity * (risk_input / 100.0) / slots
        else:
            return float(risk_input) / slots

    while True:
        try:
            df = read_ohlcv(ohlcv_path)
            if len(df) > last_rows:
                new = df.iloc[last_rows:]
                for _, bar in new.iterrows():
                    t = int(bar["timestamp"])
                    high = float(bar["high"])
                    # low = float(bar["low"])  # не требуется здесь

                    approve_auto = (read_control(control_path).get("approve_mode", "auto") == "auto")

                    # входы
                    if approve_auto and len(opened) < max_concurrent:
                        cand = signals[(signals["when"] <= t) & (~signals["active"])]
                        for idx, sig in cand.iterrows():
                            entry = float(sig["entry"])
                            stop = float(sig["stop"])
                            tp = float(sig["tp"])
                            if high >= entry:
                                entry_fill = price_with_slip(entry, slip_bps, "buy")
                                stop_eff = price_with_slip(stop, slip_bps, "sell")
                                tp_eff = price_with_slip(tp, slip_bps, "sell")

                                risk_unit_usd = entry_fill - stop_eff
                                if risk_unit_usd <= 0:
                                    continue

                                risk_dollars = risk_per_trade()
                                qty_base = (risk_dollars * leverage) / risk_unit_usd
                                qty_base = bybit.round_qty(qty_base, qty_step, min_qty)
                                if qty_base <= 0:
                                    continue

                                # проверка notional (если биржа требует)
                                notional = entry_fill * qty_base
                                if min_notional and notional < min_notional:
                                    need = min_notional / max(entry_fill, 1e-12)
                                    qty_base = bybit.round_qty(need, qty_step, min_qty)

                                # --- ОГРАНИЧЕНИЕ ПО ДОСТУПНОЙ МАРЖЕ ---
                                avail = bybit.get_available_usdt()
                                req_margin = notional / max(leverage, 1e-9)
                                if req_margin > max(avail - 5, 0):  # оставить запас на комиссии/округления
                                    max_notional = max((avail - 5), 0) * leverage
                                    max_qty = max_notional / max(entry_fill, 1e-12)
                                    qty_base = bybit.round_qty(max_qty, qty_step, min_qty)
                                    if qty_base <= 0:
                                        print(f"[SKIP] Недостаточно баланса: avail={avail:.2f} USDT, нужно≈{req_margin:.2f} USDT")
                                        continue
                                # --------------------------------------

                                # маркет-вход
                                r1 = bybit.place_market_buy(symbol, qty=qty_base, reduce_only=False)
                                order_id = (r1.get("result") or {}).get("orderId", "")
                                print(f"[ENTER] t={t} qty={qty_base} entry≈{entry_fill:.6f} id={order_id}")

                                # partial TP (50% на 2R)
                                risk_per_unit = entry_fill - stop_eff
                                tp_partial_lvl = entry_fill + partial_tpR * risk_per_unit
                                tp_partial_eff = price_with_slip(tp_partial_lvl, slip_bps, "sell")
                                tp_partial_eff = bybit.round_price(tp_partial_eff, tick_size)
                                qty_partial = bybit.round_qty(qty_base * partial_frac, qty_step, min_qty)
                                if qty_partial > 0:
                                    bybit.place_reduce_only_limit(symbol, qty=qty_partial, price=tp_partial_eff, tag="TP_PARTIAL")

                                # final TP (остаток) и SL
                                qty_final = max(qty_base - qty_partial, 0.0)
                                tp_eff_r = bybit.round_price(tp_eff, tick_size)
                                if qty_final > 0:
                                    bybit.place_reduce_only_limit(symbol, qty=qty_final, price=tp_eff_r, tag="TP_FINAL")
                                sl_eff_r = bybit.round_price(stop_eff, tick_size)
                                bybit.place_reduce_only_limit(symbol, qty=qty_base, price=sl_eff_r, tag="SL_INIT")

                                opened.append(dict(
                                    entry_fill=entry_fill,
                                    stop_eff=sl_eff_r,
                                    tp_eff=tp_eff_r,
                                    qty=qty_base,
                                    be_active=False
                                ))
                                signals.at[idx, "active"] = True

                    # сопровождение BE
                    if opened:
                        for pos in opened:
                            if not pos["be_active"]:
                                risk_per_unit = pos["entry_fill"] - pos["stop_eff"]
                                be_trigger = pos["entry_fill"] + be_fromR * risk_per_unit
                                if high >= be_trigger:
                                    bybit.cancel_all_reduce_only(symbol)
                                    new_sl = bybit.round_price(pos["entry_fill"], tick_size)
                                    bybit.place_reduce_only_limit(symbol, qty=pos["qty"], price=new_sl, tag="SL_BE")
                                    bybit.place_reduce_only_limit(symbol, qty=pos["qty"] * 0.5, price=pos["tp_eff"], tag="TP_FINAL_RE")
                                    pos["stop_eff"] = new_sl
                                    pos["be_active"] = True
                                    print(f"[BE] moved SL->entry {new_sl:.6f}")

                last_rows = len(df)
            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print("STOP by user")
            break
        except BybitError as e:
            print(f"[BybitError] {e}")
            time.sleep(2.0)
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(2.0)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Live Executor (Bybit Testnet)")
    ap.add_argument("--ohlcv", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--initial_capital", type=float, default=10000.0)
    ap.add_argument("--risk_input", type=float, default=100.0, help="100 при pct => 100% депозита (делится на max_concurrent)")
    ap.add_argument("--risk_unit", choices=["pct", "usd"], default="pct")
    ap.add_argument("--max_concurrent", type=int, default=1)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--fee_bps", type=float, default=6.0)
    ap.add_argument("--slip_bps", type=float, default=2.0)
    ap.add_argument("--control", default="runtime/control.json")
    ap.add_argument("--poll_sec", type=float, default=3.0)
    args = ap.parse_args()

    run_live(
        ohlcv_path=args.ohlcv,
        signals_path=args.signals,
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        risk_input=args.risk_input,
        risk_unit=args.risk_unit,
        max_concurrent=args.max_concurrent,
        leverage=args.leverage,
        fee_bps=args.fee_bps,
        slip_bps=args.slip_bps,
        control_path=args.control,
        poll_sec=args.poll_sec
    )

if __name__ == "__main__":
    main()
