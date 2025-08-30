#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bybit v5 Adapter (Testnet) — Linear USDT Perp

Функции:
  - get_server_time()
  - get_symbol_spec(symbol) -> (tick_size, qty_step, min_order_qty, min_notional)
  - set_leverage(symbol, buyLeverage)
  - get_wallet_balance()
  - get_available_usdt()
  - get_positions(symbol=None)
  - place_market_buy(...)
  - place_reduce_only_limit(...)
  - cancel_all_reduce_only(symbol)
  - round_price / round_qty

ENV: BYBIT_API_KEY, BYBIT_API_SECRET
"""

from __future__ import annotations
import time, hmac, hashlib, json
from typing import Dict, Any, Optional, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BYBIT_TESTNET = "https://api-testnet.bybit.com"

class BybitError(Exception):
    pass

def _ts_ms() -> int:
    return int(time.time() * 1000)

def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

class BybitAdapter:
    def __init__(self, api_key: str, api_secret: str, base_url: str = BYBIT_TESTNET, timeout: float = 15.0):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    # --------- low-level signed calls ---------

    def _headers(self, timestamp: int, sign: str) -> Dict[str, str]:
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }

    def _signed_get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = _ts_ms()
        recv = "5000"
        # порядок параметров важен для подписи: используем join в переданном порядке
        query = "&".join([f"{k}={v}" for k, v in params.items()]) if params else ""
        pre_sign = f"{timestamp}{self.api_key}{recv}{query}"
        sign = _sign(self.api_secret, pre_sign)
        url = f"{self.base_url}{path}"
        r = self.client.get(url, params=params, headers=self._headers(timestamp, sign))
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") not in (0, None):
            raise BybitError(data)
        return data

    def _signed_post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = _ts_ms()
        recv = "5000"
        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        pre_sign = f"{timestamp}{self.api_key}{recv}{payload}"
        sign = _sign(self.api_secret, pre_sign)
        url = f"{self.base_url}{path}"
        r = self.client.post(url, content=payload.encode("utf-8"), headers=self._headers(timestamp, sign))
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") not in (0, None):
            raise BybitError(data)
        return data

    # --------- public/account ---------

    def get_server_time(self) -> int:
        data = self._signed_get("/v5/market/time", {})
        return int(data.get("time", _ts_ms()))

    def get_symbol_spec(self, symbol: str) -> Tuple[float, float, float, float]:
        """
        Возвращает: (tick_size, qty_step, min_order_qty, min_notional)
        """
        params = {"category": "linear", "symbol": symbol}
        data = self._signed_get("/v5/market/instruments-info", params)
        result = ((data.get("result") or {}).get("list") or [])
        if not result:
            # дефолты (на случай отсутствия символа)
            return 0.01, 0.001, 0.0, 0.0
        item = result[0]
        pf = item.get("priceFilter", {})
        lf = item.get("lotSizeFilter", {})
        tick = float(pf.get("tickSize", 0.01))
        step = float(lf.get("qtyStep", 0.001))
        min_qty = float(lf.get("minOrderQty", 0.0))
        min_notional = float(lf.get("minOrderAmt", 0.0)) if lf.get("minOrderAmt") is not None else 0.0
        return tick, step, min_qty, min_notional

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), retry=retry_if_exception_type((httpx.HTTPError, BybitError)))
    def set_leverage(self, symbol: str, buyLeverage: int | float) -> Dict[str, Any]:
        body = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(buyLeverage),
            "sellLeverage": str(buyLeverage)
        }
        try:
            return self._signed_post("/v5/position/set-leverage", body)
        except BybitError as e:
            payload = e.args[0] if e.args else {}
            # 110043: leverage not modified — игнорируем
            if isinstance(payload, dict) and payload.get("retCode") == 110043:
                return {"retCode": 110043, "retMsg": "leverage not modified - ignored", "result": {}}
            raise

    def get_wallet_balance(self, coin: str = "USDT") -> Dict[str, Any]:
        params = {"accountType": "UNIFIED"}
        return self._signed_get("/v5/account/wallet-balance", params)

    def get_available_usdt(self) -> float:
        wb = self.get_wallet_balance()
        acc = (wb.get("result") or {}).get("list") or []
        for a in acc:
            for c in a.get("coin", []):
                if c.get("coin") == "USDT":
                    return float(c.get("availableToWithdraw", 0.0))
        return 0.0

    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return self._signed_get("/v5/position/list", params)

    # --------- orders ---------

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), retry=retry_if_exception_type((httpx.HTTPError, BybitError)))
    def place_market_buy(self, symbol: str, qty: float, reduce_only: bool = False) -> Dict[str, Any]:
        body = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy",
            "orderType": "Market",
            "qty": f"{qty}",
            "reduceOnly": reduce_only,
            "timeInForce": "ImmediateOrCancel"
        }
        return self._signed_post("/v5/order/create", body)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), retry=retry_if_exception_type((httpx.HTTPError, BybitError)))
    def place_reduce_only_limit(self, symbol: str, qty: float, price: float, tag: str = "TP") -> Dict[str, Any]:
        body = {
            "category": "linear",
            "symbol": symbol,
            "side": "Sell",
            "orderType": "Limit",
            "qty": f"{qty}",
            "price": f"{price}",
            "reduceOnly": True,
            "timeInForce": "GoodTillCancel",
            "orderLinkId": f"{tag}-{_ts_ms()}"
        }
        return self._signed_post("/v5/order/create", body)

    def cancel_all_reduce_only(self, symbol: str) -> Dict[str, Any]:
        body = {"category": "linear", "symbol": symbol}
        return self._signed_post("/v5/order/cancel-all", body)

    # --------- utils: rounding ---------

    @staticmethod
    def round_to_step(x: float, step: float) -> float:
        if step <= 0:
            return x
        return (int(x / step + 1e-12)) * step

    def round_price(self, price: float, tick_size: float) -> float:
        return self.round_to_step(price, tick_size)

    def round_qty(self, qty: float, qty_step: float, min_qty: float = 0.0) -> float:
        q = self.round_to_step(qty, qty_step)
        if min_qty and q < min_qty:
            q = min_qty
        return q
