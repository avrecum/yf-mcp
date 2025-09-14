#!/usr/bin/env python3
import os
import time
import math
import json
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone

import yfinance as yf
from fastmcp import FastMCP

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SERVER_NAME = "Yahoo Finance MCP Server"

# Yahoo chart accepted intervals and ranges (kept in sync with public endpoints)
ALLOWED_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo"
}
ALLOWED_RANGES = {
    "1d", "5d", "1mo", "3mo", "6mo",
    "1y", "2y", "5y", "10y", "ytd", "max"
}

# A minimal ISO 4217 currency allowlist. Extend as needed.
ISO_4217 = {
    "USD","EUR","GBP","JPY","CHF","CAD","AUD","NZD","CNY","SEK","NOK","DKK","PLN","HUF","CZK",
    "TRY","ZAR","MXN","BRL","INR","RUB","KRW","SGD","HKD","TWD","THB","IDR","MYR","PHP","ILS",
    "AED","SAR","CLP","COP","PEN","RON","BGN","ISK","KWD","QAR","VND","ARS","UAH","EGP","NGN",
    "PKR","LKR"
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_iso(ts: int | float | None) -> str | None:
    if ts is None or (isinstance(ts, float) and (math.isnan(ts) or math.isinf(ts))):
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None

def _require_interval(interval: str) -> str:
    if interval not in ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(ALLOWED_INTERVALS)}")
    return interval

def _require_range(range_: str) -> str:
    if range_ not in ALLOWED_RANGES:
        raise ValueError(f"Unsupported range '{range_}'. Allowed: {sorted(ALLOWED_RANGES)}")
    return range_

def _require_currency(code: str) -> str:
    code_u = (code or "").upper()
    if len(code_u) != 3 or not code_u.isalpha():
        raise ValueError(f"Currency code must be a 3-letter ISO 4217 code, got '{code}'.")
    # Do not hard block if not in allowlist; Yahoo may still support it.
    # We warn by raising on obviously bad codes only.
    return code_u

def _currency_symbol(base: str, quote: str) -> str:
    # Yahoo FX pairs use the "BASEQUOTE=X" convention, e.g., "EURUSD=X".
    return f"{base}{quote}=X"

def _yf_quote_payload(symbols: list[str]) -> dict:
    tickers = yf.Tickers(" ".join(symbols))
    results: list[dict] = []
    for sym in symbols:
        tk = tickers.tickers.get(sym)
        if not tk:
            continue
        entry: dict = {"symbol": sym}
        try:
            fi = tk.fast_info
            entry["regularMarketPrice"] = _safe_float(getattr(fi, "last_price", None))
            lpt = getattr(fi, "last_price_time", None)
            entry["regularMarketTime"] = int(lpt) if isinstance(lpt, (int, float)) else None
            entry["currency"] = getattr(fi, "currency", None)
            entry["exchange"] = getattr(fi, "exchange", None)
        except Exception:
            pass
        # Enrich with names when inexpensive; ignore errors
        try:
            info = getattr(tk, "get_info", None)
            info = info() if callable(info) else getattr(tk, "info", {})
            if isinstance(info, dict):
                entry["shortName"] = info.get("shortName")
                entry["longName"] = info.get("longName")
                entry["quoteType"] = info.get("quoteType")
                entry["fullExchangeName"] = info.get("fullExchangeName") or info.get("exchange")
                if entry.get("currency") is None:
                    entry["currency"] = info.get("currency")
        except Exception:
            pass
        results.append(entry)
    return {"quoteResponse": {"result": results, "error": None}}

def _safe_float(x: t.Any) -> float | None:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

def _yf_build_candles_from_history(hist) -> list[dict]:
    if hist is None or getattr(hist, "empty", True):
        return []
    # Ensure timezone aware UTC
    try:
        idx = hist.index.tz_convert("UTC")
    except Exception:
        idx = hist.index
    timestamps = [int(getattr(ts, "timestamp", lambda: int(ts.value // 1_000_000_000))()) if hasattr(ts, "timestamp") else int(ts.timestamp()) for ts in getattr(idx, "to_pydatetime", lambda: idx)()]
    candles = []
    opens = hist["Open"].tolist() if "Open" in hist else [None] * len(timestamps)
    highs = hist["High"].tolist() if "High" in hist else [None] * len(timestamps)
    lows = hist["Low"].tolist() if "Low" in hist else [None] * len(timestamps)
    closes = hist["Close"].tolist() if "Close" in hist else [None] * len(timestamps)
    adjs = hist["Adj Close"].tolist() if "Adj Close" in hist else closes
    vols = hist["Volume"].tolist() if "Volume" in hist else [None] * len(timestamps)
    for i, ts in enumerate(timestamps):
        candles.append({
            "time_utc": _to_iso(ts),
            "open": _safe_float(opens[i] if i < len(opens) else None),
            "high": _safe_float(highs[i] if i < len(highs) else None),
            "low": _safe_float(lows[i] if i < len(lows) else None),
            "close": _safe_float(closes[i] if i < len(closes) else None),
            "adjclose": _safe_float(adjs[i] if i < len(adjs) else None),
            "volume": int(vols[i]) if i < len(vols) and isinstance(vols[i], (int, float)) and not math.isnan(vols[i]) else None
        })
    return candles

# -----------------------------------------------------------------------------
# MCP Server
# -----------------------------------------------------------------------------
mcp = FastMCP(SERVER_NAME)

# -----------------------------------------------------------------------------
# Existing sample tools retained
# -----------------------------------------------------------------------------
@mcp.tool(description="Greet a user by name with a welcome message from the MCP server")
def greet(name: str) -> str:
    return f"Hello, {name}! Welcome to {SERVER_NAME}."

@mcp.tool(description="Get information about the MCP server including name, version, environment, and Python version")
def get_server_info() -> dict:
    return {
        "server_name": SERVER_NAME,
        "version": "1.1.0",
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "python_version": os.sys.version.split()[0],
        "time_utc": _now_iso()
    }

# -----------------------------------------------------------------------------
# Currency (Foreign Exchange) tools
# -----------------------------------------------------------------------------
@mcp.tool(description="Get the latest exchange rate for a currency pair (e.g., EUR to USD). Returns price and metadata.")
def get_currency_pair_rate(base_currency: str, quote_currency: str) -> dict:
    base = _require_currency(base_currency)
    quote = _require_currency(quote_currency)
    symbol = _currency_symbol(base, quote)

    data = _yf_quote_payload([symbol])
    result = (data.get("quoteResponse", {}) or {}).get("result", [])
    if not result:
        raise RuntimeError(f"No quote available for currency pair {base}/{quote} ({symbol}).")

    q = result[0]
    price = _safe_float(q.get("regularMarketPrice"))
    tsec = q.get("regularMarketTime")
    return {
        "pair": f"{base}/{quote}",
        "symbol": symbol,
        "price": price,
        "price_time_utc": _to_iso(tsec),
        "currency": q.get("currency") or quote,
        "exchange": q.get("fullExchangeName") or q.get("exchange") or "FX",
        "market_state": q.get("marketState")
    }

@mcp.tool(description="Convert an amount from one currency to another using the latest available rate.")
def convert_currency_amount(amount: float, base_currency: str, quote_currency: str) -> dict:
    if not isinstance(amount, (int, float)):
        raise ValueError("Amount must be numeric.")
    base = _require_currency(base_currency)
    quote = _require_currency(quote_currency)
    quote_data = get_currency_pair_rate(base, quote)
    price = quote_data["price"]
    if price is None:
        raise RuntimeError(f"Could not obtain an exchange rate for {base}/{quote}.")
    converted = float(amount) * float(price)
    return {
        "input": {"amount": amount, "base": base, "quote": quote},
        "rate": price,
        "converted_amount": converted,
        "as_of_utc": quote_data["price_time_utc"],
        "symbol": quote_data["symbol"]
    }

# @mcp.tool(description="Historical candles for a currency pair. Choose an allowed range and interval (e.g., range='1mo', interval='1d').")
def get_currency_pair_history(
    base_currency: str,
    quote_currency: str,
    range: str = "1mo",
    interval: str = "1d"
) -> dict:
    base = _require_currency(base_currency)
    quote = _require_currency(quote_currency)
    range_ = _require_range(range)
    interval_ = _require_interval(interval)
    symbol = _currency_symbol(base, quote)

    hist = yf.Ticker(symbol).history(period=range_, interval=interval_, auto_adjust=False)
    candles = _yf_build_candles_from_history(hist)
    if not candles:
        raise RuntimeError(f"No historical data for {symbol} with range={range_}, interval={interval_}.")
    return {
        "pair": f"{base}/{quote}",
        "symbol": symbol,
        "range": range_,
        "interval": interval_,
        "candles": candles
    }

# -----------------------------------------------------------------------------
# Quotes and market data tools (equities, indices, crypto)
# -----------------------------------------------------------------------------
@mcp.tool(description="Get the latest quote for a single symbol (equity, index like ^GSPC, currency pair like EURUSD=X, or crypto like BTC-USD).")
def get_quote(symbol: str) -> dict:
    sym = symbol.strip()
    data = _yf_quote_payload([sym])

    result = (data.get("quoteResponse", {}) or {}).get("result", [])
    if not result:
        raise RuntimeError(f"No quote available for symbol '{sym}'.")
    q = result[0]
    return {
        "symbol": q.get("symbol", sym),
        "short_name": q.get("shortName"),
        "long_name": q.get("longName"),
        "quote_type": q.get("quoteType"),
        "currency": q.get("currency"),
        "exchange": q.get("fullExchangeName") or q.get("exchange"),
        "market_state": q.get("marketState"),
        "price": _safe_float(q.get("regularMarketPrice")),
        "previous_close": _safe_float(q.get("regularMarketPreviousClose")),
        "open": _safe_float(q.get("regularMarketOpen")),
        "day_high": _safe_float(q.get("regularMarketDayHigh")),
        "day_low": _safe_float(q.get("regularMarketDayLow")),
        "volume": q.get("regularMarketVolume"),
        "time_utc": _to_iso(q.get("regularMarketTime"))
    }

@mcp.tool(description="Get latest quotes for multiple symbols at once. Returns a dict keyed by symbol.")
def get_batch_quotes(symbols: list[str]) -> dict:
    if not symbols:
        raise ValueError("Provide at least one symbol.")
    syms = [s.strip() for s in symbols if s and s.strip()]
    data = _yf_quote_payload(syms)

    out: dict[str, dict] = {}
    for q in (data.get("quoteResponse", {}) or {}).get("result", []):
        sym = q.get("symbol")
        out[sym] = {
            "symbol": sym,
            "price": _safe_float(q.get("regularMarketPrice")),
            "currency": q.get("currency"),
            "time_utc": _to_iso(q.get("regularMarketTime")),
            "market_state": q.get("marketState"),
            "exchange": q.get("fullExchangeName") or q.get("exchange"),
            "short_name": q.get("shortName"),
            "long_name": q.get("longName"),
        }
    # Ensure all requested symbols are present, filling missing with None
    for s in syms:
        if s not in out:
            out[s] = {"symbol": s, "price": None, "error": "No quote"}
    return out

@mcp.tool(description="Historical candles for any symbol (equity, index, currency pair, crypto). Choose a range and interval.")
def get_history(symbol: str, range: str = "1mo", interval: str = "1d") -> dict:
    sym = symbol.strip()
    range_ = _require_range(range)
    interval_ = _require_interval(interval)

    hist = yf.Ticker(sym).history(period=range_, interval=interval_, auto_adjust=False)
    candles = _yf_build_candles_from_history(hist)
    if not candles:
        raise RuntimeError(f"No historical data for {sym} with range={range_}, interval={interval_}.")
    return {
        "symbol": sym,
        "range": range_,
        "interval": interval_,
        "candles": candles
    }

@mcp.tool(description="Market summary for a region (e.g., 'US', 'GB', 'DE', 'JP'). Returns top indices and session state.")
def get_market_summary(region: str = "US") -> dict:
    region = (region or "US").upper()
    # Minimal curated index list per region; extend as needed.
    region_indices: dict[str, list[str]] = {
        "US": ["^GSPC", "^DJI", "^IXIC"],
        "GB": ["^FTSE"],
        "DE": ["^GDAXI"],
        "FR": ["^FCHI"],
        "JP": ["^N225"],
        "HK": ["^HSI"],
        "CN": ["000001.SS", "399001.SZ"],
        "IN": ["^BSESN", "^NSEI"],
    }
    syms = region_indices.get(region, ["^GSPC", "^DJI", "^IXIC"])
    data = _yf_quote_payload(syms)
    out = []
    for q in (data.get("quoteResponse", {}) or {}).get("result", []):
        out.append({
            "symbol": q.get("symbol"),
            "short_name": q.get("shortName"),
            "full_exchange_name": q.get("fullExchangeName") or q.get("exchange"),
            "market_state": q.get("marketState"),
            "price": _safe_float(q.get("regularMarketPrice")),
            "change": None,
            "change_percent": None,
            "time_utc": _to_iso(q.get("regularMarketTime"))
        })
    return {"region": region, "indices": out}

# -----------------------------------------------------------------------------
# Discovery and options tools ("stuff")
# -----------------------------------------------------------------------------
@mcp.tool(description="Search instruments by free text. Returns tickers Yahoo associates with the query.")
def search_instruments(query: str, count: int = 10, news_count: int = 0) -> dict:
    q = (query or "").strip()
    if not q:
        raise ValueError("Provide a non-empty query string.")
    # yfinance Search API
    s = yf.Search(q)
    items = getattr(s, "results", []) or []
    out = []
    for it in items[: int(count)]:
        out.append({
            "symbol": it.get("symbol"),
            "short_name": it.get("shortname") or it.get("shortName"),
            "long_name": it.get("longname") or it.get("longName"),
            "exchange": it.get("exchange") or it.get("exchDisp"),
            "quote_type": it.get("quoteType"),
            "score": _safe_float(it.get("score")),
            "sector": it.get("sector"),
            "industry": it.get("industry"),
            "is_currency": it.get("quoteType") == "CURRENCY"
        })
    return {"query": q, "results": out}

@mcp.tool(description="Get option chain for an equity symbol. If expiration is omitted, returns available expirations.")
def get_options_chain(symbol: str, expiration: str | None = None) -> dict:
    # expiration can be a UNIX seconds string; if omitted we return dates.
    sym = symbol.strip()
    tk = yf.Ticker(sym)
    # If expiration is omitted, return list of available dates
    exps = getattr(tk, "options", []) or []
    out: dict = {"symbol": sym}
    # Underlying price/currency
    try:
        fi = tk.fast_info
        out["underlying"] = {
            "price": _safe_float(getattr(fi, "last_price", None)),
            "currency": getattr(fi, "currency", None)
        }
    except Exception:
        out["underlying"] = {"price": None, "currency": None}
    if not expiration:
        out["expirations_utc"] = exps
        return out
    # Normalize YYYY-MM-DD
    exp = expiration.strip()
    if exp not in exps:
        raise ValueError("expiration must be one of Ticker.options values in YYYY-MM-DD.")
    chain = tk.option_chain(exp)
    def _fmt_row(row: dict) -> dict:
        return {
            "contract_symbol": row.get("contractSymbol"),
            "strike": _safe_float(row.get("strike")),
            "last_price": _safe_float(row.get("lastPrice")),
            "bid": _safe_float(row.get("bid")),
            "ask": _safe_float(row.get("ask")),
            "volume": row.get("volume"),
            "open_interest": row.get("openInterest"),
            "in_the_money": row.get("inTheMoney"),
            "implied_volatility": _safe_float(row.get("impliedVolatility")),
            "expiration_utc": exp
        }
    calls = [ _fmt_row(r) for r in chain.calls.to_dict("records") ] if hasattr(chain, "calls") else []
    puts =  [ _fmt_row(r) for r in chain.puts.to_dict("records") ] if hasattr(chain, "puts") else []
    return out | {"expiration_utc": exp, "calls": calls, "puts": puts}

# -----------------------------------------------------------------------------
# Convenience: crypto and indices helpers (wrappers over get_quote)
# -----------------------------------------------------------------------------
@mcp.tool(description="Get latest price for a crypto asset vs USD, e.g., symbol='BTC' -> 'BTC-USD'.")
def get_crypto_usd_quote(symbol: str) -> dict:
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("Provide a crypto base symbol, e.g., 'BTC'.")
    return get_quote(f"{sym}-USD")

@mcp.tool(description="Get the S&P 500 (^GSPC) index quote.")
def get_sp500_quote() -> dict:
    return get_quote("^GSPC")

@mcp.tool(description="Get the NASDAQ 100 (^NDX) index quote.")
def get_nasdaq100_quote() -> dict:
    return get_quote("^NDX")

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@mcp.tool(description="Simple liveness probe.")
def health_check() -> dict:
    return {"ok": True, "server": SERVER_NAME, "time_utc": _now_iso()}

# -----------------------------------------------------------------------------
# Server main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    print(f"Starting {SERVER_NAME} on {host}:{port}")
    mcp.run(
        transport="http",
        host=host,
        port=port
    )
