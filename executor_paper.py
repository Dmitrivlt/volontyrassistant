#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper Executor (LONG-only after 2/4)
- Читает OHLCV (закрытые бары) из parquet, отслеживает новые строки.
- Читает сигналы (entry/stop/tp/when) из parquet.
- На закрытии бара активирует входы, рассчитывает размер, ведёт позицию:
    * partial 50% на 2R
    * BE с 2R
- Режим Approve: auto | manual
- Делит депозит по лимиту одновременных сделок (1..10) — как в твоём UI.

Файлы:
  OHLCV: data/<exchange>/<SYMBOL>_<interval>.parquet
  Signals: data/<exchange>/waves/<SYMBOL>_<interval>_signals_rr3.parquet  (или др.)
  Trades out: runtime/exe_trades.parquet
  Inbox (manual approve): runtime/inbox.jsonl
  Control: runtime/control.json  (approve_mode: "auto"|"manual")

Запуск (пример):
python executor_paper.py \
  --ohlcv data/bybit/SOLUSDT_15m.parquet \
  --signals data/bybit/waves/SOLUSDT_15m_signals_rr3.parquet \
  --initial_capital 10000 \
  --risk_input 100 --risk_unit pct \
  --max_concurrent 5 \
  --leverage 1 \
  --fill_priority sl_first \
  --fee_bps 6 --slip_bps 2
"""

from __future__ import annotations
import argparse, os, time, json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

UTC = timezone.utc

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- IO ----------

def read_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    need = {"timestamp","open","high","low","close"}
    if not need.issubset(df.columns):
        raise ValueError(f"OHLCV должен содержать {need}")
    return df

def read_signals(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("when").reset_index(drop=True)
    need = {"when","type","entry","stop","tp"}
    if not need.issubset(df.columns):
        raise ValueError(f"Signals должен содержать {need}")
    # берём только LONG after 2/4
    df = df[df["type"].isin(["LONG_after2","LONG_after4"])].reset_index(drop=True)
    return df

def to_ms(ts: int | float) -> int:
    return int(ts)

def ts_human(ms: int) -> str:
    return datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------- Risk & Sizing ----------

def compute_risk_per_trade(equity: float, risk_input: float, risk_unit: str, max_concurrent: int) -> float:
    """
    risk_input: число (например 100)
    risk_unit: 'pct' или 'usd'
    max_concurrent: 1..10 (делим депозит на N слотов — как ты описал)
    Возвращает risk_dollars на одну сделку.
    """
    slots = max(1, min(int(max_concurrent), 10))
    if risk_unit == "pct":
        # например 100% и slots=5 -> на сделку 20% эквити
        risk_total_pct = float(risk_input) / 100.0
        risk_per_slot = risk_total_pct / slots
        return equity * risk_per_slot
    else:
        # фикс. USD, делим на слоты
        return float(risk_input) / slots

def price_with_slip(price: float, slip_bps: float, side: str) -> float:
    slip = slip_bps / 10_000.0
    if side == "buy":
        return price * (1.0 + slip)
    else:
        return price * (1.0 - slip)

# ---------- Approve control ----------

def read_control(path: str) -> dict:
    # {"approve_mode": "auto"|"manual"}
    if not os.path.exists(path):
        return {"approve_mode":"auto"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"approve_mode":"auto"}

def inbox_append(path: str, item: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ---------- Core Executor ----------

class PaperExecutor:
    def __init__(self,
                 ohlcv_path: str,
                 signals_path: str,
                 initial_capital: float,
                 risk_input: float,
                 risk_unit: str,
                 max_concurrent: int,
                 leverage: float,
                 fee_bps: float,
                 slip_bps: float,
                 fill_priority: str,
                 control_path: str = "runtime/control.json",
                 inbox_path: str = "runtime/inbox.jsonl",
                 trades_out: str = "runtime/exe_trades.parquet",
                 log_path: str = "runtime/exe_events.log") -> None:
        self.ohlcv_path = ohlcv_path
        self.signals_path = signals_path
        self.equity = initial_capital
        self.risk_input = risk_input
        self.risk_unit = risk_unit
        self.max_concurrent = max_concurrent
        self.leverage = leverage
        self.fee = fee_bps / 10_000.0
        self.slip_bps = slip_bps
        self.fill_priority = fill_priority
        self.control_path = control_path
        self.inbox_path = inbox_path
        self.trades_out = trades_out
        self.log_path = log_path

        ensure_dir(os.path.dirname(trades_out) or ".")
        ensure_dir(os.path.dirname(log_path) or ".")

        self.ohlcv = read_ohlcv(ohlcv_path)
        self.signals = read_signals(signals_path)

        self.signals["active"] = False  # активирован ли вход
        self.signals["done"] = False    # сделка завершена (обе ноги)
        self.signals_idx = 0            # указатель для поступления сигналов по времени

        self.open_positions: list[dict] = []  # позиции (может быть 0..N)
        self.trades_rows = []  # будет писать ноги сделок
        self.next_trade_id = 0

        self._log(f"INIT equity={self.equity:.2f}, risk={self.risk_input}{self.risk_unit}, slots={self.max_concurrent}, lev={self.leverage}")

    def _log(self, msg: str) -> None:
        line = f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _save_trades(self):
        if self.trades_rows:
            df = pd.DataFrame(self.trades_rows)
            if os.path.exists(self.trades_out):
                old = pd.read_parquet(self.trades_out)
                df = pd.concat([old, df], ignore_index=True)
            df.sort_values("close_time", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.to_parquet(self.trades_out, index=False)

    def _enter_position(self, bar, sig) -> None:
        # approve?
        ctrl = read_control(self.control_path)
        if ctrl.get("approve_mode","auto") == "manual":
            inbox_item = {
                "ts": int(bar["timestamp"]),
                "symbol": os.path.basename(self.ohlcv_path).split("_")[0],
                "entry": float(sig["entry"]),
                "stop": float(sig["stop"]),
                "tp": float(sig["tp"]),
                "when": int(sig["when"]),
                "suggested": True
            }
            inbox_append(self.inbox_path, inbox_item)
            self._log(f"APPROVE REQUIRED -> inbox appended: {inbox_item}")
            return

        # auto mode: входим
        entry = float(sig["entry"])
        stop  = float(sig["stop"])
        tp    = float(sig["tp"])
        high  = float(bar["high"])
        low   = float(bar["low"])

        if high < entry:
            return  # на всякий случай

        entry_fill = price_with_slip(entry, self.slip_bps, "buy")
        stop_eff   = price_with_slip(stop,  self.slip_bps, "sell")
        tp_eff     = price_with_slip(tp,    self.slip_bps, "sell")

        risk_dollars = compute_risk_per_trade(self.equity, self.risk_input, self.risk_unit, self.max_concurrent)
        risk_per_unit = entry_fill - stop_eff
        if risk_per_unit <= 0:
            self._log("SKIP enter: bad levels (risk_per_unit<=0)")
            return
        size_full = (risk_dollars * self.leverage) / risk_per_unit
        if size_full <= 0:
            return

        # комиссия на вход
        fee_entry = self.fee * entry_fill * size_full
        self.equity -= fee_entry

        # partial/BE
        partial_frac = 0.5
        partial_tpR  = 2.0
        be_fromR     = 2.0
        have_partial = (0.0 < partial_frac < 1.0)
        size_partial = size_full * partial_frac if have_partial else 0.0
        size_final   = size_full - size_partial

        tp_partial_lvl = entry_fill + partial_tpR * risk_per_unit
        tp_partial_eff = price_with_slip(tp_partial_lvl, self.slip_bps, "sell")

        self.next_trade_id += 1
        pos = dict(
            id=self.next_trade_id,
            open_time=int(bar["timestamp"]),
            entry=entry, entry_fill=entry_fill,
            stop=stop, stop_eff=stop_eff,
            tp=tp, tp_eff=tp_eff,
            risk_dollars=risk_dollars,
            risk_per_unit=risk_per_unit,
            size_partial=size_partial,
            size_final=size_final,
            have_partial=have_partial,
            partial_done=False,
            be_fromR=be_fromR,
            be_active=False,
            be_level=entry_fill,
            tp_partial_eff=tp_partial_eff
        )
        self.open_positions.append(pos)
        self._log(f"ENTER id={pos['id']} entry_fill={entry_fill:.4f} size={size_full:.4f} fee_in={fee_entry:.4f} eq={self.equity:.2f}")

    def _maybe_close_positions_on_bar(self, bar, fill_priority: str):
        high = float(bar["high"])
        low  = float(bar["low"])
        t    = int(bar["timestamp"])
        still = []
        for pos in self.open_positions:
            # BE активация
            if (not pos["be_active"]) and pos["be_fromR"] > 0.0:
                be_trigger = pos["entry_fill"] + pos["be_fromR"] * pos["risk_per_unit"]
                if high >= be_trigger:
                    pos["be_active"] = True
                    pos["stop_eff"] = max(pos["stop_eff"], pos["be_level"])

            # partial
            if pos["have_partial"] and (not pos["partial_done"]) and high >= pos["tp_partial_eff"]:
                size_leg = pos["size_partial"]
                if size_leg > 0:
                    exit_fill = pos["tp_partial_eff"]
                    pnl = (exit_fill - pos["entry_fill"]) * size_leg
                    fee_exit = self.fee * exit_fill * size_leg
                    pnl_after = pnl - fee_exit
                    self.equity += pnl_after
                    self.trades_rows.append(dict(
                        id=pos["id"], leg="partial",
                        open_time=pos["open_time"], close_time=t,
                        entry_fill=pos["entry_fill"], exit_fill=exit_fill,
                        exit_reason="TP_PARTIAL",
                        size=size_leg, pnl=pnl_after,
                        r_mult=pnl_after / (pos["risk_dollars"] * (size_leg / (pos["size_partial"] + pos["size_final"]))),
                        equity_after=self.equity
                    ))
                    self._log(f"PARTIAL id={pos['id']} exit={exit_fill:.4f} pnl={pnl_after:.2f} eq={self.equity:.2f}")
                pos["partial_done"] = True
                pos["size_partial"] = 0.0

            # final
            size_remaining = pos["size_final"] + pos["size_partial"]
            if size_remaining <= 0:
                continue

            hit_sl = (low <= pos["stop_eff"])
            hit_tp = (high >= pos["tp_eff"])

            exit_reason = None
            if hit_sl and hit_tp:
                exit_reason = ("TP" if fill_priority == "tp_first" else "SL")
            elif hit_sl:
                exit_reason = "SL"
            elif hit_tp:
                exit_reason = "TP"

            if exit_reason is not None:
                exit_fill = pos["tp_eff"] if exit_reason == "TP" else pos["stop_eff"]
                pnl = (exit_fill - pos["entry_fill"]) * size_remaining
                fee_exit = self.fee * exit_fill * size_remaining
                pnl_after = pnl - fee_exit
                self.equity += pnl_after
                self.trades_rows.append(dict(
                    id=pos["id"], leg="final",
                    open_time=pos["open_time"], close_time=t,
                    entry_fill=pos["entry_fill"], exit_fill=exit_fill,
                    exit_reason=exit_reason,
                    size=size_remaining, pnl=pnl_after,
                    r_mult=pnl_after / (pos["risk_dollars"] * (size_remaining / (pos["size_partial"] + pos["size_final"]))),
                    equity_after=self.equity
                ))
                self._log(f"EXIT id={pos['id']} {exit_reason} exit={exit_fill:.4f} pnl={pnl_after:.2f} eq={self.equity:.2f}")
            else:
                still.append(pos)

        self.open_positions = still

    def run(self, poll_sec: float = 3.0):
        last_rows = len(self.ohlcv)
        self._log(f"START watching {self.ohlcv_path} (rows={last_rows})")

        while True:
            try:
                # перечитать OHLCV при росте файла
                df = read_ohlcv(self.ohlcv_path)
                if len(df) > last_rows:
                    new = df.iloc[last_rows:]
                    for _, bar in new.iterrows():
                        # активировать сигналы с when<=бар
                        # (выполняем вход если high >= entry)
                        # сначала добавим свежие сигналы в зону внимания
                        # (у нас все сигналы в файле, просто фильтруем по времени)
                        # вход:
                        # ограничение по кол-ву одновременных сделок:
                        if len(self.open_positions) < self.max_concurrent:
                            # для простоты: активируем все релевантные сигналы этого бара
                            cand = self.signals[(self.signals["done"] == False) & (self.signals["when"] <= bar["timestamp"])]
                            for idx, sig in cand.iterrows():
                                self._enter_position(bar, sig)
                                # помечать как активированные не обязательно; закрытие случится позже
                                self.signals.loc[idx, "active"] = True

                        # сопровождение позиций на баре
                        self._maybe_close_positions_on_bar(bar, self.fill_priority)

                    last_rows = len(df)
                    # периодически сохраняем сделки
                    self._save_trades()

                time.sleep(poll_sec)

            except KeyboardInterrupt:
                self._log("STOP by user")
                self._save_trades()
                break

            except Exception as e:
                self._log(f"ERROR: {e}")
                time.sleep(poll_sec)
