#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volontyr's Assistant - OHLCV Data Pipeline
- REST: исторические свечи Binance/Bybit
- WS: онлайн-стрим закрытых свечей Binance/Bybit
- Хранение: Parquet (дедупликция, строго закрытые бары => без look-ahead)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Iterable, List, Optional, Tuple

import httpx
import pandas as pd
from dateutil import parser as dtparser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import websockets
import pytz

# ---------- Константы и утилиты ----------

UTC = timezone.utc

BINANCE_REST = "https://api.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws"

BYBIT_REST = "https://api.bybit.com"
# Public WS Unified (Linear):
BYBIT_WS_LINEAR = "wss://stream.bybit.com/v5/public/linear"

# совместимый формат интервалов
SUPPORTED_INTERVALS = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}

def now_utc_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)

def interval_to_ms(interval: str) -> int:
    # '15m' -> 900000 ms, '1h' -> 3600000, etc.
    unit = interval[-1]
    val = int(interval[:-1])
    if unit == "m":
        return val * 60_000
    if unit == "h":
        return val * 60 * 60_000
    if unit == "d":
        return val * 24 * 60 * 60_000
    raise ValueError(f"Неподдерживаемый интервал: {interval}")

def align_to_closed_end(end_ms: int, interval: str) -> int:
    """Вернуть правую границу так, чтобы последний бар был закрыт."""
    step = interval_to_ms(interval)
    return (end_ms // step) * step  # конец последнего полностью закрытого бара

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_date_to_ms(x: str) -> int:
    # принимает 'YYYY-MM-DD' или ISO, возвращает ms (UTC)
    dt = dtparser.parse(x)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return int(dt.timestamp() * 1000)

def bybit_interval(interval: str) -> str:
    # Bybit v5 kline intervals: '1','3','5','15','30','60','120','240','360','720','D','W','M'
    mapping = {
        "1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
        "1h":"60","2h":"120","4h":"240","6h":"360","12h":"720",
        "1d":"D"
    }
    if interval not in mapping:
        raise ValueError(f"Интервал {interval} не поддерживается для Bybit.")
    return mapping[interval]

# ---------- Хранилище OHLCV ----------

@dataclass
class OHLCV:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class OHLCVStore:
    def __init__(self, root: str = "data") -> None:
        self.root = root

    def path(self, exchange: str, symbol: str, interval: str) -> str:
        d = os.path.join(self.root, exchange.lower())
        ensure_dir(d)
        return os.path.join(d, f"{symbol.upper()}_{interval}.parquet")

    def read(self, exchange: str, symbol: str, interval: str) -> pd.DataFrame:
        p = self.path(exchange, symbol, interval)
        if os.path.exists(p):
            return pd.read_parquet(p)
        cols = ["timestamp","open","high","low","close","volume","exchange","symbol","interval"]
        return pd.DataFrame(columns=cols)

    def append(self, exchange: str, symbol: str, interval: str, rows: List[OHLCV]) -> None:
        if not rows:
            return
        df_new = pd.DataFrame([{
            "timestamp": r.ts,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "exchange": exchange.lower(),
            "symbol": symbol.upper(),
            "interval": interval
        } for r in rows])
        p = self.path(exchange, symbol, interval)
        if os.path.exists(p):
            df_old = pd.read_parquet(p)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        # дедупликация по timestamp
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df.reset_index(drop=True, inplace=True)
        df.to_parquet(p, index=False)

# ---------- REST: Исторические свечи ----------

class BinanceREST:
    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self.client = client or httpx.Client(timeout=30.0)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> List[OHLCV]:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
            "startTime": start_ms,
            "endTime": end_ms
        }
        r = self.client.get(f"{BINANCE_REST}/api/v3/klines", params=params)
        r.raise_for_status()
        data = r.json()
        out: List[OHLCV] = []
        for row in data:
            # [ openTime, o, h, l, c, v, closeTime, ...]
            ot = int(row[0])
            ct = int(row[6])
            # Binance возвращает и «текущую» кандлу, если endTime попал внутрь; отфильтруем по закрытию
            is_closed = (ct <= align_to_closed_end(end_ms, interval))
            if not is_closed:
                continue
            out.append(OHLCV(
                ts=ot,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            ))
        return out

class BybitREST:
    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self.client = client or httpx.Client(timeout=30.0)

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> List[OHLCV]:
        # Bybit v5: /v5/market/kline?category=linear&symbol=SOLUSDT&interval=15&start=...&end=...&limit=...
        params = {
            "category": "linear",
            "symbol": symbol.upper(),
            "interval": bybit_interval(interval),
            "start": start_ms,
            "end": end_ms,
            "limit": limit
        }
        r = self.client.get(f"{BYBIT_REST}/v5/market/kline", params=params)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise httpx.HTTPError(f"Bybit error: {data}")
        result = data.get("result", {}) or {}
        rows = result.get("list", []) or []
        # Bybit часто отдаёт в обратном порядке — нормализуем
        rows = sorted(rows, key=lambda x: int(x[0] if isinstance(x, list) else x.get("start")))
        out: List[OHLCV] = []
        for row in rows:
            # форматы: может быть list или dict в зав-ти от SDK/версии
            if isinstance(row, list):
                # [ start, open, high, low, close, volume, turnover ]
                start = int(row[0])
                o, h, l, c = map(float, row[1:5])
                v = float(row[5])
            else:
                start = int(row["start"])
                o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                v = float(row.get("volume", row.get("turnover", 0.0)))
            # фильтруем незакрытые — на REST обычно уже закрытые, но на всякий случай синхронизируем по end
            if start >= align_to_closed_end(end_ms, interval):
                # если свеча начинается на последней границе, она ещё не закрыта
                continue
            out.append(OHLCV(ts=start, open=o, high=h, low=l, close=c, volume=v))
        return out

def daterange_fetch_rest(
    exchange: str,
    symbols: List[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    store: OHLCVStore
) -> None:
    assert interval in SUPPORTED_INTERVALS, f"Интервал {interval} не поддерживается."
    end_ms = align_to_closed_end(end_ms, interval)  # не захватываем «незакрытый» бар

    if exchange.lower() == "binance":
        client = BinanceREST()
        limit = 1000
        step = interval_to_ms(interval) * limit
        for sym in symbols:
            cur = start_ms
            while cur < end_ms:
                chunk_end = min(cur + step, end_ms)
                rows = client.klines(sym, interval, cur, chunk_end, limit=limit)
                store.append("binance", sym, interval, rows)
                cur = chunk_end
    elif exchange.lower() == "bybit":
        client = BybitREST()
        # у Bybit лимит тоже можно держать 1000; шагаем по времени
        limit = 1000
        step = interval_to_ms(interval) * limit
        for sym in symbols:
            cur = start_ms
            while cur < end_ms:
                chunk_end = min(cur + step, end_ms)
                rows = client.klines(sym, interval, cur, chunk_end, limit=limit)
                store.append("bybit", sym, interval, rows)
                cur = chunk_end
    else:
        raise ValueError("exchange должен быть binance|bybit")

# ---------- WS: Онлайн свечи (только закрытые) ----------

async def binance_ws_stream(symbols: List[str], interval: str, store: OHLCVStore):
    """
    Binance WS: по одному соединению на символ (надёжнее), пишем только закрытые kline (k['x'] == True)
    """
    assert interval in SUPPORTED_INTERVALS
    async def one_symbol(sym: str):
        stream = f"{sym.lower()}@kline_{interval}"
        url = f"{BINANCE_WS}/{stream}"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        k = data.get("k", {})
                        if not k:
                            continue
                        if bool(k.get("x")) is True:  # closed
                            row = OHLCV(
                                ts=int(k["t"]),
                                open=float(k["o"]),
                                high=float(k["h"]),
                                low=float(k["l"]),
                                close=float(k["c"]),
                                volume=float(k["v"]),
                            )
                            store.append("binance", sym, interval, [row])
            except Exception as e:
                print(f"[Binance WS][{sym}] error: {e}. Reconnect in 3s...")
                await asyncio.sleep(3)

    await asyncio.gather(*(one_symbol(s) for s in symbols))

async def bybit_ws_stream(symbols: List[str], interval: str, store: OHLCVStore):
    """
    Bybit Unified v5 WS (public linear). Подписка: kline.<interval>.<symbol>, запись только confirm==True.
    """
    iv = bybit_interval(interval)
    args = [f"kline.{iv}.{s.upper()}" for s in symbols]
    sub_msg = json.dumps({"op": "subscribe", "args": args})

    while True:
        try:
            async with websockets.connect(BYBIT_WS_LINEAR, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(sub_msg)
                async for msg in ws:
                    data = json.loads(msg)
                    if "topic" not in data or "data" not in data:
                        continue
                    topic: str = data["topic"]
                    if not topic.startswith("kline."):
                        continue
                    # topic = kline.<iv>.<SYMBOL>
                    parts = topic.split(".")
                    if len(parts) < 3:
                        continue
                    sym = parts[2]
                    entries = data.get("data", [])
                    for e in entries:
                        # e может быть dict со строками
                        confirm = e.get("confirm", False)
                        if confirm is True:
                            row = OHLCV(
                                ts=int(e.get("start")),
                                open=float(e.get("open")),
                                high=float(e.get("high")),
                                low=float(e.get("low")),
                                close=float(e.get("close")),
                                volume=float(e.get("volume") or e.get("turnover") or 0.0)
                            )
                            store.append("bybit", sym, interval, [row])
        except Exception as e:
            print(f"[Bybit WS] error: {e}. Reconnect in 3s...")
            await asyncio.sleep(3)

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Volontyr's Assistant - OHLCV pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # fetch-rest
    p_rest = sub.add_parser("fetch-rest", help="Загрузить историю свечей (REST)")
    p_rest.add_argument("--exchange", choices=["binance","bybit"], required=True)
    p_rest.add_argument("--symbols", required=True, help="Список через запятую, напр. SOLUSDT,OPUSDT")
    p_rest.add_argument("--interval", required=True, choices=sorted(SUPPORTED_INTERVALS))
    p_rest.add_argument("--start", required=True, help="Дата начала (YYYY-MM-DD или ISO)")
    p_rest.add_argument("--end", required=True, help="Дата конца (YYYY-MM-DD или ISO)")
    p_rest.add_argument("--data-dir", default="data")

    # stream
    p_ws = sub.add_parser("stream", help="Стримить закрытые свечи (WS)")
    p_ws.add_argument("--exchange", choices=["binance","bybit"], required=True)
    p_ws.add_argument("--symbols", required=True, help="Список через запятую")
    p_ws.add_argument("--interval", required=True, choices=sorted(SUPPORTED_INTERVALS))
    p_ws.add_argument("--data-dir", default="data")

    return p.parse_args()

def main() -> None:
    args = parse_args()
    store = OHLCVStore(root=args.data_dir)

    if args.cmd == "fetch-rest":
        start_ms = parse_date_to_ms(args.start)
        end_ms = parse_date_to_ms(args.end)
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        daterange_fetch_rest(args.exchange, symbols, args.interval, start_ms, end_ms, store)
        print("Готово.")
    elif args.cmd == "stream":
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.exchange == "binance":
            asyncio.run(binance_ws_stream(symbols, args.interval, store))
        else:
            asyncio.run(bybit_ws_stream(symbols, args.interval, store))
    else:
        raise RuntimeError("unknown cmd")

if __name__ == "__main__":
    main()
