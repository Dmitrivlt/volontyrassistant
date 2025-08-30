#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtester для сигналов LONG (после 2/4) по OHLCV.
Поддержка реализма:
  - комиссии (--fee_bps) и проскальзывание (--slip_bps)
  - частичная фиксация (--partial_tpR, --partial_frac)
  - перевод в безубыток (--be_fromR)

Логика исполнения:
  • Вход по high>=entry (fill: entry*(1+slip)); комиссия на вход списывается сразу.
  • После входа проверяем события на каждом баре:
    - Частичная фиксация при достижении partial_tpR (по цене tp_partial_eff = entry_fill + partial_tpR*R_per_unit с учётом slip).
    - BE: как только достигнуто be_fromR, стоп остатка переносится на entry_fill.
    - Финальный SL/TP: как прежде; при одновременном касании действует fill_priority.
  • Все цены исполнения учитывают slip и fees.
"""

import argparse, os
import pandas as pd
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    need = {"timestamp","open","high","low","close"}
    assert need.issubset(df.columns), f"OHLCV должен содержать колонки: {need}"
    return df

def read_signals(path: str) -> pd.DataFrame:
    sig = pd.read_parquet(path).sort_values("when").reset_index(drop=True)
    need = {"when","type","entry","stop","tp"}
    assert need.issubset(sig.columns), f"Signals должен содержать колонки: {need}"
    return sig

def run_backtest(ohlcv: pd.DataFrame,
                 signals: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 risk_per_trade: float = 0.01,
                 max_concurrent: int = 10,
                 fill_priority: str = "sl_first",
                 start_ts: int | None = None,
                 end_ts: int | None = None,
                 queue: bool = False,
                 fee_bps: float = 0.0,
                 slip_bps: float = 0.0,
                 partial_tpR: float = 2.0,
                 partial_frac: float = 0.5,
                 be_fromR: float = 2.0) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Возвращает:
      - trades_df: сделки (каждое закрытие – отдельная строка с leg=partial|final|EOD)
      - stats: словарь метрик
      - equity_curve: эквити по времени закрытия ног
    """
    # фильтрация сигналов по времени
    sig = signals.copy()
    if start_ts is not None:
        sig = sig[sig["when"] >= start_ts]
    if end_ts is not None:
        sig = sig[sig["when"] <= end_ts]
    sig = sig.reset_index(drop=True)

    ts = ohlcv["timestamp"].to_numpy()
    high = ohlcv["high"].to_numpy()
    low  = ohlcv["low"].to_numpy()
    close = ohlcv["close"].to_numpy()

    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0

    equity = initial_capital
    total_fees = 0.0

    open_positions = []  # список словарей активных позиций
    trades = []
    trade_id = 0
    pending = []
    sig_idx = 0

    for bar_i in range(len(ohlcv)):
        t = int(ts[bar_i])

        # (1) добавляем сигналы, "созревшие" к текущему бару
        while sig_idx < len(sig) and sig.loc[sig_idx, "when"] <= t:
            s = sig.loc[sig_idx]
            pending.append({
                "sig_idx": sig_idx,
                "when": int(s["when"]),
                "entry": float(s["entry"]),
                "stop": float(s["stop"]),
                "tp": float(s["tp"]),
                "stype": str(s["type"])
            })
            sig_idx += 1

        # (2) попытка входа по pending
        if pending:
            new_pending = []
            for ps in pending:
                if len(open_positions) < max_concurrent:
                    if high[bar_i] >= ps["entry"]:
                        entry_fill = ps["entry"] * (1.0 + slip)
                        stop_eff   = ps["stop"]  * (1.0 - slip)  # SL ухудшен slip
                        tp_eff     = ps["tp"]    * (1.0 - slip)  # TP ухудшен slip

                        risk_dollars = equity * risk_per_trade
                        risk_per_unit = entry_fill - stop_eff
                        if risk_per_unit <= 0:
                            continue
                        size_full = risk_dollars / risk_per_unit
                        if size_full <= 0:
                            continue

                        # комиссия на вход
                        fee_entry = fee * entry_fill * size_full
                        total_fees += fee_entry
                        equity -= fee_entry

                        # подготовка параметров частичного выхода
                        have_partial = (partial_tpR > 0.0 and 0.0 < partial_frac < 1.0)
                        size_partial = size_full * partial_frac if have_partial else 0.0
                        size_final   = size_full - size_partial
                        # уровень для частичного TP в ценах:
                        tp_partial_lvl = entry_fill + partial_tpR * risk_per_unit
                        tp_partial_eff = tp_partial_lvl * (1.0 - slip)  # худший fill

                        # признак BE
                        be_active = False
                        be_level  = entry_fill  # переносим стоп на entry_fill при достижении be_fromR

                        trade_id += 1
                        open_positions.append({
                            "id": trade_id,
                            "open_time": t,
                            "open_bar": bar_i,
                            "entry": ps["entry"],
                            "entry_fill": entry_fill,
                            "stop_orig": ps["stop"],
                            "stop_eff": stop_eff,
                            "tp": ps["tp"],
                            "tp_eff": tp_eff,
                            "size_partial": size_partial,
                            "size_final": size_final,
                            "risk_dollars": risk_dollars,
                            "risk_per_unit": risk_per_unit,
                            "stype": ps["stype"],
                            "have_partial": have_partial,
                            "partial_done": False,
                            "be_fromR": be_fromR,
                            "be_active": False,
                            "be_level": be_level,
                            "tp_partial_eff": tp_partial_eff
                        })
                    else:
                        new_pending.append(ps)
                else:
                    if queue:
                        new_pending.append(ps)
            pending = new_pending

        # (3) обновляем открытые позиции: частичный TP, BE, финальный SL/TP
        if open_positions:
            still_open = []
            for pos in open_positions:
                # 3.1. активация BE, если достигнут требуемый R
                if (not pos["be_active"]) and (pos["be_fromR"] > 0.0):
                    # достигнут ли уровень entry_fill + be_fromR * risk_per_unit?
                    be_trigger = pos["entry_fill"] + pos["be_fromR"] * pos["risk_per_unit"]
                    if high[bar_i] >= be_trigger:
                        pos["be_active"] = True
                        # переносим стоп остатка (final) в безубыток
                        pos["stop_eff"] = max(pos["stop_eff"], pos["be_level"])

                # 3.2. частичная фиксация (если включена и ещё не сделана)
                if pos["have_partial"] and (not pos["partial_done"]):
                    # если high достиг уровня частичного тейка
                    if high[bar_i] >= pos["tp_partial_eff"]:
                        # закрываем size_partial по tp_partial_eff
                        size_leg = pos["size_partial"]
                        if size_leg > 0:
                            exit_fill = pos["tp_partial_eff"]
                            pnl_per_unit = exit_fill - pos["entry_fill"]
                            pnl = pnl_per_unit * size_leg
                            fee_exit = fee * exit_fill * size_leg
                            pnl_after = pnl - fee_exit
                            equity += pnl_after
                            total_fees += fee_exit

                            trades.append({
                                "id": pos["id"],
                                "leg": "partial",
                                "stype": pos["stype"],
                                "open_time": pos["open_time"],
                                "close_time": int(ts[bar_i]),
                                "entry_fill": pos["entry_fill"],
                                "exit_fill": exit_fill,
                                "exit_reason": "TP_PARTIAL",
                                "size": size_leg,
                                "risk_dollars": pos["risk_dollars"] * (size_leg / (pos["size_partial"] + pos["size_final"])),
                                "pnl": pnl_after,
                                "r_mult": pnl_after / (pos["risk_dollars"] * (size_leg / (pos["size_partial"] + pos["size_final"]))),
                                "equity_after": equity
                            })

                        pos["partial_done"] = True
                        pos["size_partial"] = 0.0  # всё закрыли частично

                # 3.3. финальная часть — обычные SL/TP
                # определим рабочие уровни SL/TP для остатка
                size_final = pos["size_final"]
                # если весь объём закрыт частично (редко, но вдруг partial_frac=1.0) — пропустим финал
                if size_final <= 0 and pos["size_partial"] <= 0:
                    continue

                # проверяем касание уровней остатком:
                hit_sl = (low[bar_i] <= pos["stop_eff"])
                hit_tp = (high[bar_i] >= pos["tp_eff"])

                exit_reason = None
                if hit_sl and hit_tp:
                    exit_reason = ("TP" if fill_priority == "tp_first" else "SL")
                elif hit_sl:
                    exit_reason = "SL"
                elif hit_tp:
                    exit_reason = "TP"

                if exit_reason is not None:
                    # для остатка считаем его текущий размер:
                    # остаток = size_final + то, что ещё не закрыто частично (на всякий случай)
                    size_remaining = pos["size_final"] + pos["size_partial"]
                    if size_remaining > 0:
                        exit_fill = pos["tp_eff"] if exit_reason == "TP" else pos["stop_eff"]
                        pnl_per_unit = exit_fill - pos["entry_fill"]
                        pnl = pnl_per_unit * size_remaining
                        fee_exit = fee * exit_fill * size_remaining
                        pnl_after = pnl - fee_exit
                        equity += pnl_after
                        total_fees += fee_exit

                        trades.append({
                            "id": pos["id"],
                            "leg": "final",
                            "stype": pos["stype"],
                            "open_time": pos["open_time"],
                            "close_time": int(ts[bar_i]),
                            "entry_fill": pos["entry_fill"],
                            "exit_fill": exit_fill,
                            "exit_reason": exit_reason,
                            "size": size_remaining,
                            "risk_dollars": pos["risk_dollars"] * (size_remaining / (pos["size_partial"] + pos["size_final"])),
                            "pnl": pnl_after,
                            "r_mult": pnl_after / (pos["risk_dollars"] * (size_remaining / (pos["size_partial"] + pos["size_final"]))),
                            "equity_after": equity
                        })
                    # позиция закрыта
                else:
                    # ещё открыта
                    still_open.append(pos)

            open_positions = still_open

    # (4) закрываем всё по последнему close (EOD)
    if open_positions:
        last_close = float(close[-1])
        last_t = int(ts[-1])
        for pos in open_positions:
            size_remaining = pos["size_final"] + pos["size_partial"]
            if size_remaining <= 0:
                continue
            exit_fill = last_close * (1.0 - slip)
            pnl_per_unit = exit_fill - pos["entry_fill"]
            pnl = pnl_per_unit * size_remaining
            fee_exit = fee * exit_fill * size_remaining
            pnl_after = pnl - fee_exit
            equity += pnl_after
            total_fees += fee_exit

            trades.append({
                "id": pos["id"],
                "leg": "final",
                "stype": pos["stype"],
                "open_time": pos["open_time"],
                "close_time": last_t,
                "entry_fill": pos["entry_fill"],
                "exit_fill": exit_fill,
                "exit_reason": "EOD",
                "size": size_remaining,
                "risk_dollars": pos["risk_dollars"] * (size_remaining / (pos["size_partial"] + pos["size_final"])),
                "pnl": pnl_after,
                "r_mult": pnl_after / (pos["risk_dollars"] * (size_remaining / (pos["size_partial"] + pos["size_final"]))),
                "equity_after": equity
            })

    trades_df = pd.DataFrame(trades).sort_values("close_time").reset_index(drop=True)

    # --- Метрики
    # агрегируем по trade_id: считаем суммарный pnl и r_mult по каждому id
    if not trades_df.empty:
        agg = trades_df.groupby("id", as_index=False).agg(
            pnl=("pnl","sum"),
            risk=("risk_dollars","sum")
        )
        agg["r_mult"] = agg["pnl"] / agg["risk"]
        n_trades = len(agg)
        wins = int((agg["r_mult"] > 0).sum())
        losses = n_trades - wins
        winrate = (wins / n_trades * 100.0) if n_trades else 0.0
        avg_r = float(agg["r_mult"].mean()) if n_trades else 0.0
    else:
        n_trades = wins = losses = 0
        winrate = avg_r = 0.0

    expectancy = avg_r

    # кривая эквити (по закрытиям ног)
    equity_curve = trades_df[["close_time","equity_after"]].copy()
    equity_curve = equity_curve.rename(columns={"close_time":"timestamp","equity_after":"equity"})

    # макс. просадка
    if not equity_curve.empty:
        peak = equity_curve["equity"].cummax()
        dd = (equity_curve["equity"] - peak) / peak
        max_dd = float(dd.min())  # отрицательное число
    else:
        max_dd = 0.0

    stats = dict(
        signals_total=int(len(sig)),
        trades_total=int(n_trades),
        wins=int(wins),
        losses=int(losses),
        winrate_pct=float(winrate),
        avg_R=float(avg_r),
        expectancy_R=float(expectancy),
        final_equity=float(equity),
        net_profit=float(equity - initial_capital),
        total_fees=float(total_fees),
        max_drawdown_pct=float(max_dd * 100.0),
        start_equity=float(initial_capital),
        fee_bps=float(fee_bps),
        slip_bps=float(slip_bps),
        partial_tpR=float(partial_tpR),
        partial_frac=float(partial_frac),
        be_fromR=float(be_fromR)
    )

    return trades_df, stats, equity_curve

def main():
    ap = argparse.ArgumentParser(description="Backtester LONG (после 2/4) + fees/slip + partial/BE")
    ap.add_argument("--ohlcv", required=True, help="data/<exchange>/<SYMBOL>_<interval>.parquet")
    ap.add_argument("--signals", required=True, help="data/<exchange>/waves/<SYMBOL>_<interval>_signals*.parquet")
    ap.add_argument("--output", help="куда сохранить trades.parquet (по умолчанию рядом в waves/)")

    ap.add_argument("--initial_capital", type=float, default=10000.0)
    ap.add_argument("--risk_per_trade", type=float, default=0.01)
    ap.add_argument("--max_concurrent", type=int, default=10)
    ap.add_argument("--fill_priority", choices=["sl_first","tp_first"], default="sl_first")
    ap.add_argument("--start_ts", type=int)
    ap.add_argument("--end_ts", type=int)
    ap.add_argument("--queue", action="store_true", default=False)

    # реализм
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--slip_bps", type=float, default=0.0)

    # управление позицией
    ap.add_argument("--partial_tpR", type=float, default=2.0, help="частичный выход на X R (0 = выкл)")
    ap.add_argument("--partial_frac", type=float, default=0.5, help="доля позиции для частичного TP (0..1)")
    ap.add_argument("--be_fromR", type=float, default=2.0, help="перенос стопа в BE после X R (0 = выкл)")

    args = ap.parse_args()

    ohlcv = read_ohlcv(args.ohlcv)
    signals = read_signals(args.signals)

    trades_df, stats, equity_curve = run_backtest(
        ohlcv=ohlcv,
        signals=signals,
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_concurrent=args.max_concurrent,
        fill_priority=args.fill_priority,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        queue=args.queue,
        fee_bps=args.fee_bps,
        slip_bps=args.slip_bps,
        partial_tpR=args.partial_tpR,
        partial_frac=args.partial_frac,
        be_fromR=args.be_fromR
    )

    # сохранение
    if args.output:
        out_path = args.output
    else:
        base = os.path.dirname(args.signals)
        ensure_dir(base)
        sig_base = os.path.splitext(os.path.basename(args.signals))[0]
        fn = f"{sig_base}_trades.parquet"
        out_path = os.path.join(base, fn)

    trades_df.to_parquet(out_path, index=False)

    print("=== Backtest Summary ===")
    for k,v in stats.items():
        if k.endswith("_pct"):
            print(f"{k}: {v:.2f}%")
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"trades saved to: {out_path}")

    if not equity_curve.empty:
        eq_first = equity_curve['equity'].iloc[0]
        eq_last  = equity_curve['equity'].iloc[-1]
        print(f"equity first/last: {eq_first:.2f} -> {eq_last:.2f}")

if __name__ == "__main__":
    main()
