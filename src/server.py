#!/usr/bin/env python3
import os
import time
import math
import json
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from fastmcp import FastMCP

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SERVER_NAME = "Yahoo Finance MCP Server"
BASE_URL = os.environ.get("YAHOO_FINANCE_BASE_URL", "https://query1.finance.yahoo.com")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "10"))
USE_YFINANCE = os.environ.get("USE_YFINANCE", "0") == "1"

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
session = requests.Session()
session.headers.update({
    # A reasonable UA reduces 403s with some Yahoo frontends.
    "User-Agent": "Mozilla/5.0 (compatible; FastMCP/1.0; +https://github.com/)"
})

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_iso(ts: int | float | None) -> str | None:
    if ts is None or (isinstance(ts, float) and (math.isnan(ts) or math.isinf(ts))):
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None

def _yahoo_get(path: str, params: dict | None = None) -> dict:
    """GET helper that raises for non-OK responses and returns parsed JSON."""
    url = f"{BASE_URL}{path}"
    resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

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

def _quote_symbols(symbols: list[str]) -> dict:
    """Return Yahoo v7 quote payload for the given symbols list."""
    symbols_s = ",".join(symbols)
    data = _yahoo_get("/v7/finance/quote", {"symbols": symbols_s})
    return data

def _chart(symbol: str, range_: str, interval: str) -> dict:
    params = {"range": range_, "interval": interval, "events": "div,split"}
    data = _yahoo_get(f"/v8/finance/chart/{symbol}", params)
    return data

def _safe_float(x: t.Any) -> float | None:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

# Optional fallback via yfinance (only if requested)
def _yf_quote(symbols: list[str]) -> dict | None:
    if not USE_YFINANCE:
        return None
    try:
        import yfinance as yf
        tickers = yf.Tickers(" ".join(symbols))
        result = []
        for sym in symbols:
            tk = tickers.tickers.get(sym)
            if not tk:
                continue
            info = {}
            # fast_info is fast and robust for price/time fields
            try:
                fi = tk.fast_info
                info["symbol"] = sym
                info["regularMarketPrice"] = _safe_float(getattr(fi, "last_price", None))
                info["regularMarketTime"]  = int(getattr(fi, "last_price_time", 0)) if getattr(fi, "last_price_time", None) else None
                info["currency"] = getattr(fi, "currency", None)
                info["exchange"] = getattr(fi, "exchange", None)
            except Exception:
                pass
            if info:
                result.append(info)
        return {"quoteResponse": {"result": result, "error": None}}
    except Exception:
        return None

def _yf_chart(symbol: str, range_: str, interval: str) -> dict | None:
    if not USE_YFINANCE:
        return None
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=range_, interval=interval, auto_adjust=False)
        # Emulate the Yahoo chart structure minimally
        if hist.empty:
            return {"chart": {"result": None, "error": {"code": "Empty", "description": "No data"}}}
        timestamps = [int(ts.timestamp()) for ts in hist.index.tz_convert("UTC").to_pydatetime()]
        result = {
            "meta": {"symbol": symbol},
            "timestamp": timestamps,
            "indicators": {
                "quote": [{
                    "open":  [ _safe_float(v) for v in hist["Open"].tolist() ],
                    "high":  [ _safe_float(v) for v in hist["High"].tolist() ],
                    "low":   [ _safe_float(v) for v in hist["Low"].tolist() ],
                    "close": [ _safe_float(v) for v in hist["Close"].tolist() ],
                    "volume":[ int(v) if (isinstance(v, (int,float)) and not math.isnan(v)) else None for v in hist["Volume"].tolist() ],
                }],
                "adjclose": [{
                    "adjclose": [ _safe_float(v) for v in (hist["Adj Close"].tolist() if "Adj Close" in hist else hist["Close"].tolist()) ]
                }]
            }
        }
        return {"chart": {"result": [result], "error": None}}
    except Exception:
        return None

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

    data = None
    try:
        data = _quote_symbols([symbol])
    except Exception:
        # Optional fallback
        data = _yf_quote([symbol]) or {"quoteResponse": {"result": [], "error": "unavailable"}}

    result = (data.get("quoteResponse", {}) or {}).get("result", [])  # type: ignore
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

@mcp.tool(description="Historical candles for a currency pair. Choose an allowed range and interval (e.g., range='1mo', interval='1d').")
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

    data = None
    try:
        data = _chart(symbol, range_, interval_)
    except Exception:
        data = _yf_chart(symbol, range_, interval_)

    chart = (data or {}).get("chart", {})
    if not chart or chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {symbol}: {chart.get('error')}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"No historical data for {symbol} with range={range_}, interval={interval_}.")

    res = results[0]
    timestamps: list[int] = res.get("timestamp", []) or []
    ind = res.get("indicators", {}) or {}
    quote_arr = (ind.get("quote", []) or [{}])[0]
    adj_arr = (ind.get("adjclose", []) or [{}])[0]

    candles = []
    for i, ts in enumerate(timestamps):
        candles.append({
            "time_utc": _to_iso(ts),
            "open":  _safe_float((quote_arr.get("open") or [None])[i] if i < len(quote_arr.get("open", [])) else None),
            "high":  _safe_float((quote_arr.get("high") or [None])[i] if i < len(quote_arr.get("high", [])) else None),
            "low":   _safe_float((quote_arr.get("low") or [None])[i] if i < len(quote_arr.get("low", [])) else None),
            "close": _safe_float((quote_arr.get("close") or [None])[i] if i < len(quote_arr.get("close", [])) else None),
            "adjclose": _safe_float((adj_arr.get("adjclose") or [None])[i] if i < len(adj_arr.get("adjclose", [])) else None),
            "volume": int((quote_arr.get("volume") or [None])[i]) if i < len(quote_arr.get("volume", [])) and (quote_arr.get("volume") or [None])[i] is not None else None
        })

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
    data = None
    try:
        data = _quote_symbols([sym])
    except Exception:
        data = _yf_quote([sym]) or {"quoteResponse": {"result": [], "error": "unavailable"}}

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
    data = None
    try:
        data = _quote_symbols(syms)
    except Exception:
        data = _yf_quote(syms) or {"quoteResponse": {"result": [], "error": "unavailable"}}

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

    data = None
    try:
        data = _chart(sym, range_, interval_)
    except Exception:
        data = _yf_chart(sym, range_, interval_)

    chart = (data or {}).get("chart", {})
    if not chart or chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {sym}: {chart.get('error')}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"No historical data for {sym} with range={range_}, interval={interval_}.")

    res = results[0]
    timestamps: list[int] = res.get("timestamp", []) or []
    ind = res.get("indicators", {}) or {}
    quote_arr = (ind.get("quote", []) or [{}])[0]
    adj_arr = (ind.get("adjclose", []) or [{}])[0]

    candles = []
    for i, ts in enumerate(timestamps):
        candles.append({
            "time_utc": _to_iso(ts),
            "open":  _safe_float((quote_arr.get("open") or [None])[i] if i < len(quote_arr.get("open", [])) else None),
            "high":  _safe_float((quote_arr.get("high") or [None])[i] if i < len(quote_arr.get("high", [])) else None),
            "low":   _safe_float((quote_arr.get("low") or [None])[i] if i < len(quote_arr.get("low", [])) else None),
            "close": _safe_float((quote_arr.get("close") or [None])[i] if i < len(quote_arr.get("close", [])) else None),
            "adjclose": _safe_float((adj_arr.get("adjclose") or [None])[i] if i < len(adj_arr.get("adjclose", [])) else None),
            "volume": int((quote_arr.get("volume") or [None])[i]) if i < len(quote_arr.get("volume", [])) and (quote_arr.get("volume") or [None])[i] is not None else None
        })

    return {
        "symbol": sym,
        "range": range_,
        "interval": interval_,
        "candles": candles
    }

@mcp.tool(description="Market summary for a region (e.g., 'US', 'GB', 'DE', 'JP'). Returns top indices and session state.")
def get_market_summary(region: str = "US") -> dict:
    region = (region or "US").upper()
    data = _yahoo_get("/v6/finance/quote/marketSummary", {"lang": "en", "region": region})
    results = (data.get("marketSummaryResponse", {}) or {}).get("result", [])  # type: ignore
    out = []
    for r in results:
        out.append({
            "symbol": r.get("symbol"),
            "short_name": r.get("shortName"),
            "full_exchange_name": r.get("fullExchangeName"),
            "market_state": r.get("marketState"),
            "price": _safe_float((r.get("regularMarketPrice") or {}).get("raw")),
            "change": _safe_float((r.get("regularMarketChange") or {}).get("raw")),
            "change_percent": _safe_float((r.get("regularMarketChangePercent") or {}).get("raw")),
            "time_utc": _to_iso((r.get("regularMarketTime") or {}).get("raw"))
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
    params = {"q": q, "quotesCount": int(count), "newsCount": int(news_count)}
    data = _yahoo_get("/v1/finance/search", params)
    quotes = data.get("quotes", []) or []
    out = []
    for it in quotes:
        out.append({
            "symbol": it.get("symbol"),
            "short_name": it.get("shortname") or it.get("shortName"),
            "long_name": it.get("longname") or it.get("longName"),
            "exchange": it.get("exchange") or it.get("exchDisp"),
            "quote_type": it.get("quoteType"),
            "score": _safe_float(it.get("score")),
            "sector": it.get("sector"),
            "industry": it.get("industry"),
            "is_currency": bool(it.get("isYahooFinance", False)) and (it.get("quoteType") == "CURRENCY")
        })
    return {"query": q, "results": out}

@mcp.tool(description="Get option chain for an equity symbol. If expiration is omitted, returns available expirations.")
def get_options_chain(symbol: str, expiration: str | None = None) -> dict:
    # expiration can be a UNIX seconds string; if omitted we return dates.
    sym = symbol.strip()
    path = f"/v7/finance/options/{sym}"
    params = {}
    if expiration:
        # Accept YYYY-MM-DD for convenience, convert to epoch if needed.
        exp = expiration.strip()
        if exp.isdigit():
            params["date"] = exp
        else:
            # Try parse YYYY-MM-DD to epoch seconds (UTC midnight)
            try:
                dt = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                params["date"] = str(int(dt.timestamp()))
            except Exception:
                raise ValueError("expiration must be UNIX seconds or YYYY-MM-DD.")
    data = _yahoo_get(path, params or None)
    opt = (data.get("optionChain", {}) or {}).get("result", [])  # type: ignore
    if not opt:
        raise RuntimeError(f"No options data for {sym}.")
    meta = opt[0].get("quote", {}) or {}
    expirations = [ _to_iso(d) for d in opt[0].get("expirationDates", []) ]
    out: dict = {"symbol": sym, "underlying": {"price": _safe_float(meta.get("regularMarketPrice")), "currency": meta.get("currency")}}
    if not expiration:
        out["expirations_utc"] = expirations
        return out
    chains = opt[0].get("options", []) or []
    if not chains:
        return out | {"calls": [], "puts": []}
    chain = chains[0]
    calls = chain.get("calls", []) or []
    puts = chain.get("puts", []) or []
    def _fmt(o: dict) -> dict:
        return {
            "contract_symbol": o.get("contractSymbol"),
            "strike": _safe_float(o.get("strike")),
            "last_price": _safe_float(o.get("lastPrice")),
            "bid": _safe_float(o.get("bid")),
            "ask": _safe_float(o.get("ask")),
            "volume": o.get("volume"),
            "open_interest": o.get("openInterest"),
            "in_the_money": o.get("inTheMoney"),
            "implied_volatility": _safe_float(o.get("impliedVolatility")),
            "expiration_utc": _to_iso(chain.get("expiration"))
        }
    return out | {
        "expiration_utc": _to_iso(chain.get("expiration")),
        "calls": [_fmt(c) for c in calls],
        "puts":  [_fmt(p) for p in puts]
    }

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
