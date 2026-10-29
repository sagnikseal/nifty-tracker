#!/usr/bin/env python3
"""
Fetch NIFTY 50 Daily Market Data and compute trading metrics.
Tracks Daily High, Daily Low, Range, Open, Close, Change, Pivots (CPR), and Candlestick patterns.
Outputs structured JSON to data/nifty_data.json.
"""

import sys
import json
import os
import datetime
from pathlib import Path

# Try standard urllib first (zero dependency)
try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nifty_data.json"
SYMBOL = "^NSEI"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def fetch_from_yahoo_api(range_str="1y"):
    """Fetch OHLC daily data from Yahoo Finance Chart API."""
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?range={range_str}&interval=1d&includePrePost=false",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{SYMBOL}?range={range_str}&interval=1d&includePrePost=false"
    ]
    
    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data.get("chart", {}).get("result", [])
                if result:
                    return result[0]
        except Exception as e:
            last_err = e
            continue
            
    # Optional yfinance fallback if urllib failed
    try:
        import yfinance as yf
        print("Falling back to yfinance package...")
        ticker = yf.Ticker(SYMBOL)
        df = ticker.history(period=range_str, interval="1d")
        if not df.empty:
            timestamps = [int(ts.timestamp()) for ts in df.index]
            opens = df["Open"].tolist()
            highs = df["High"].tolist()
            lows = df["Low"].tolist()
            closes = df["Close"].tolist()
            volumes = df["Volume"].tolist() if "Volume" in df else [0] * len(df)
            meta = {
                "symbol": SYMBOL,
                "regularMarketPrice": closes[-1] if closes else None,
                "regularMarketDayHigh": highs[-1] if highs else None,
                "regularMarketDayLow": lows[-1] if lows else None,
            }
            return {
                "meta": meta,
                "timestamp": timestamps,
                "indicators": {"quote": [{"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}]}
            }
    except Exception as yf_err:
        print(f"yfinance fallback failed: {yf_err}")

    if last_err:
        raise RuntimeError(f"Failed to fetch market data: {last_err}")
    raise RuntimeError("No chart data received from Yahoo Finance")


def compute_pivots_and_metrics(raw_candles, meta=None):
    """
    Given sorted raw candles [{'date': 'YYYY-MM-DD', 'open', 'high', 'low', 'close', 'volume'}],
    computes daily range, change %, CPR, pivot points, inside/outside bars, and gap.
    """
    enriched = []
    
    for i, c in enumerate(raw_candles):
        o = round(float(c['open']), 2)
        h = round(float(c['high']), 2)
        l = round(float(c['low']), 2)
        cl = round(float(c['close']), 2)
        vol = int(c.get('volume', 0) or 0)
        dt_str = c['date']
        
        # Day of week
        dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d")
        day_name = dt_obj.strftime("%A")
        
        # Daily range
        d_range = round(h - l, 2)
        d_range_pct = round((d_range / o) * 100, 2) if o > 0 else 0.0
        
        # Change from previous close
        prev_close = None
        change = 0.0
        change_pct = 0.0
        gap = 0.0
        gap_type = "Flat"
        is_inside_day = False
        is_outside_day = False
        
        if i > 0:
            prev = enriched[i - 1]
            prev_close = prev['close']
            prev_high = prev['high']
            prev_low = prev['low']
            
            change = round(cl - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
            gap = round(o - prev_close, 2)
            
            if gap >= 25:
                gap_type = "Gap Up"
            elif gap <= -25:
                gap_type = "Gap Down"
            else:
                gap_type = "Flat"
                
            # Inside Day / Outside Day
            if h <= prev_high and l >= prev_low:
                is_inside_day = True
            elif h >= prev_high and l <= prev_low:
                is_outside_day = True
        else:
            change = round(cl - o, 2)
            change_pct = round((change / o) * 100, 2) if o else 0.0
            prev_close = o

        # Candle pattern
        if abs(cl - o) <= 12:
            candle_type = "Doji"
        elif cl > o:
            candle_type = "Bullish"
        else:
            candle_type = "Bearish"

        # Central Pivot Range (CPR) for this day's levels (based on High, Low, Close)
        pivot = round((h + l + cl) / 3.0, 2)
        bc = round((h + l) / 2.0, 2)
        tc = round((pivot - bc) + pivot, 2)
        cpr_width = round(abs(tc - bc), 2)
        cpr_type = "Narrow" if cpr_width <= 40 else ("Wide" if cpr_width >= 90 else "Average")
        
        # Standard Classic Pivots
        r1 = round((2 * pivot) - l, 2)
        s1 = round((2 * pivot) - h, 2)
        r2 = round(pivot + (h - l), 2)
        s2 = round(pivot - (h - l), 2)
        
        enriched.append({
            "date": dt_str,
            "day": day_name,
            "open": o,
            "high": h,
            "low": l,
            "close": cl,
            "volume": vol,
            "range": d_range,
            "range_pct": d_range_pct,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "gap": gap,
            "gap_type": gap_type,
            "candle_type": candle_type,
            "is_inside_day": is_inside_day,
            "is_outside_day": is_outside_day,
            "pivot": pivot,
            "tc": tc,
            "bc": bc,
            "cpr_width": cpr_width,
            "cpr_type": cpr_type,
            "r1": r1,
            "s1": s1,
            "r2": r2,
            "s2": s2
        })
        
    return enriched


def main():
    print("=" * 60)
    print("Fetching NIFTY 50 Market Data...")
    print("=" * 60)
    
    # Check if existing data file exists
    existing_records_dict = {}
    if DATA_PATH.exists():
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                for item in old_data.get("history", []):
                    existing_records_dict[item["date"]] = item
            print(f"Loaded {len(existing_records_dict)} existing records from {DATA_PATH.name}")
        except Exception as e:
            print(f"Warning: could not read existing data ({e}), starting fresh")

    # Fetch 1 year from Yahoo
    chart_res = fetch_from_yahoo_api(range_str="1y")
    meta = chart_res.get("meta", {})
    timestamps = chart_res.get("timestamp", [])
    quotes = chart_res.get("indicators", {}).get("quote", [{}])[0]
    
    opens = quotes.get("open", [])
    highs = quotes.get("high", [])
    lows = quotes.get("low", [])
    closes = quotes.get("close", [])
    volumes = quotes.get("volume", [])
    
    raw_by_date = {}
    for i in range(len(timestamps)):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        
        if o is None or h is None or l is None:
            continue
            
        # If today's candle close is not finalized in quote array yet, use regularMarketPrice
        if c is None and i == len(timestamps) - 1:
            c = meta.get("regularMarketPrice", meta.get("regularMarketDayHigh", h))
        if c is None:
            continue
            
        dt = datetime.datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d")
        vol = volumes[i] if (volumes and i < len(volumes) and volumes[i] is not None) else 0
        raw_by_date[dt] = {
            "date": dt,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol
        }
        
    # Merge existing records in case user has older historical data
    for dt, item in existing_records_dict.items():
        if dt not in raw_by_date:
            raw_by_date[dt] = {
                "date": dt,
                "open": item["open"],
                "high": item["high"],
                "low": item["low"],
                "close": item["close"],
                "volume": item.get("volume", 0)
            }
            
    # Sort chronologically
    sorted_raw = [raw_by_date[k] for k in sorted(raw_by_date.keys())]
    print(f"Total deduplicated trading days: {len(sorted_raw)}")
    
    enriched = compute_pivots_and_metrics(sorted_raw, meta)
    
    # Calculate 5-day & 20-day Average Daily Range (ADR)
    recent_20 = enriched[-20:] if len(enriched) >= 20 else enriched
    recent_5 = enriched[-5:] if len(enriched) >= 5 else enriched
    adr_20 = round(sum(d["range"] for d in recent_20) / len(recent_20), 2) if recent_20 else 0
    adr_5 = round(sum(d["range"] for d in recent_5) / len(recent_5), 2) if recent_5 else 0
    
    # 52-week High and Low
    all_highs = [d["high"] for d in enriched]
    all_lows = [d["low"] for d in enriched]
    high_52w = max(all_highs) if all_highs else meta.get("fiftyTwoWeekHigh", 0)
    low_52w = min(all_lows) if all_lows else meta.get("fiftyTwoWeekLow", 0)
    
    # Latest day summary
    latest = enriched[-1] if enriched else None
    prev_day = enriched[-2] if len(enriched) > 1 else None
    
    # Current timestamp in IST
    # UTC + 5:30
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    ist_str = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    output_payload = {
        "symbol": "NIFTY 50",
        "ticker": SYMBOL,
        "exchange": "NSE",
        "currency": "INR",
        "last_updated": ist_str,
        "last_updated_iso": ist_now.isoformat(),
        "meta": {
            "fifty_two_week_high": round(high_52w, 2),
            "fifty_two_week_low": round(low_52w, 2),
            "adr_20": adr_20,
            "adr_5": adr_5,
            "total_trading_days": len(enriched)
        },
        "latest": latest,
        "prev_day": prev_day,
        "history": enriched  # Chronological order
    }
    
    # Ensure directory exists
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
        
    print(f"Successfully saved {len(enriched)} records to {DATA_PATH}")
    if latest:
        print("\n--- LATEST NIFTY SUMMARY ---")
        print(f"Date:         {latest['date']} ({latest['day']})")
        print(f"Daily High:   {latest['high']}")
        print(f"Daily Low:    {latest['low']}")
        print(f"Daily Range:  {latest['range']} pts ({latest['range_pct']}%)")
        print(f"Open:         {latest['open']}")
        print(f"Close:        {latest['close']}")
        print(f"Net Change:   {latest['change']} pts ({latest['change_pct']}%)")
        print(f"Candle Type:  {latest['candle_type']}")
        print(f"20-Day ADR:   {adr_20} pts")
        print("----------------------------\n")


if __name__ == "__main__":
    main()
