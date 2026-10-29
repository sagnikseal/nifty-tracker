#!/usr/bin/env python3
"""
Fetch NIFTY 50 Daily Market Data and compute trading metrics.
Tracks Daily High, Daily Low, Range, CPR, Candlestick patterns, and Smart Money Concepts (SMC)
liquidity metrics (PDH/PDL sweeps, PWH/PWL, BOS, FVG, Premium/Discount, Next-Day Bias).
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


def compute_weekly_levels(raw_candles):
    """Computes Previous Week High (PWH) and Previous Week Low (PWL) for each date."""
    weeks = {}
    date_to_week = {}
    
    for c in raw_candles:
        dt = datetime.datetime.strptime(c['date'], "%Y-%m-%d")
        cal = dt.isocalendar()
        week_key = f"{cal[0]}-W{cal[1]:02d}"
        date_to_week[c['date']] = week_key
        
        if week_key not in weeks:
            weeks[week_key] = {"high": float(c['high']), "low": float(c['low']), "dates": []}
        else:
            weeks[week_key]["high"] = max(weeks[week_key]["high"], float(c['high']))
            weeks[week_key]["low"] = min(weeks[week_key]["low"], float(c['low']))
        weeks[week_key]["dates"].append(c['date'])
        
    sorted_week_keys = sorted(weeks.keys())
    week_prev_map = {}
    for i in range(1, len(sorted_week_keys)):
        curr_wk = sorted_week_keys[i]
        prev_wk = sorted_week_keys[i - 1]
        week_prev_map[curr_wk] = {
            "pwh": round(weeks[prev_wk]["high"], 2),
            "pwl": round(weeks[prev_wk]["low"], 2)
        }
        
    date_levels = {}
    for dt_str, wk in date_to_week.items():
        if wk in week_prev_map:
            date_levels[dt_str] = week_prev_map[wk]
        else:
            date_levels[dt_str] = {"pwh": None, "pwl": None}
            
    return date_levels


def compute_pivots_and_metrics(raw_candles, meta=None):
    """
    Computes daily range, change %, CPR, pivot points, inside/outside bars,
    SMC liquidity metrics (sweeps, BOS, Equilibrium/Zone), and candle momentum.
    """
    weekly_levels = compute_weekly_levels(raw_candles)
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
        
        # Candle Body & Wicks
        body = round(abs(cl - o), 2)
        body_pct = round((body / d_range * 100), 1) if d_range > 0 else 0.0
        upper_wick = round(h - max(o, cl), 2)
        lower_wick = round(min(o, cl) - l, 2)
        
        # Change from previous close
        prev_close = None
        prev_high = None
        prev_low = None
        change = 0.0
        change_pct = 0.0
        gap = 0.0
        gap_type = "Flat"
        is_inside_day = False
        is_outside_day = False
        
        # SMC Liquidity & Structure
        pdh_sweep = False  # Swept PDH but closed below -> Bearish Liquidity Grab
        pdl_sweep = False  # Swept PDL but closed above -> Bullish Liquidity Grab
        bullish_bos = False # Closed cleanly above PDH -> Break of Structure
        bearish_bos = False # Closed cleanly below PDL -> Break of Structure
        smc_structure = "Normal"
        
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
                
            # Smart Money Concepts: External Range Liquidity (ERL) sweeps
            if h > prev_high and cl < prev_high:
                pdh_sweep = True
                smc_structure = "BSL Sweep (Bearish Grab)"
            elif l < prev_low and cl > prev_low:
                pdl_sweep = True
                smc_structure = "SSL Sweep (Bullish Grab)"
            elif cl > prev_high:
                bullish_bos = True
                smc_structure = "Bullish BOS (Expansion)"
            elif cl < prev_low:
                bearish_bos = True
                smc_structure = "Bearish BOS (Expansion)"
            elif is_inside_day:
                smc_structure = "Inside Consolidation"
        else:
            change = round(cl - o, 2)
            change_pct = round((change / o) * 100, 2) if o else 0.0
            prev_close = o
            prev_high = h
            prev_low = l

        # Candle Classification
        is_expansion_bar = (body_pct >= 65 and d_range >= 130)
        is_pinbar = (upper_wick >= body * 1.8) or (lower_wick >= body * 1.8)
        
        if abs(cl - o) <= 15:
            candle_type = "Doji (Indecision)"
        elif cl > o:
            candle_type = "Strong Bullish Expansion" if is_expansion_bar else "Bullish"
        else:
            candle_type = "Strong Bearish Expansion" if is_expansion_bar else "Bearish"

        # Central Pivot Range (CPR) for this day's levels (based on High, Low, Close)
        pivot = round((h + l + cl) / 3.0, 2)
        bc = round((h + l) / 2.0, 2)
        tc = round((pivot - bc) + pivot, 2)
        cpr_width = round(abs(tc - bc), 2)
        cpr_type = "Narrow (Trend Potential)" if cpr_width <= 40 else ("Wide (Range Potential)" if cpr_width >= 90 else "Average")
        
        # Standard Classic Pivots
        r1 = round((2 * pivot) - l, 2)
        s1 = round((2 * pivot) - h, 2)
        r2 = round(pivot + (h - l), 2)
        s2 = round(pivot - (h - l), 2)
        
        # Weekly levels
        wk_info = weekly_levels.get(dt_str, {})
        pwh = wk_info.get("pwh")
        pwl = wk_info.get("pwl")
        
        # 20-Day Range Context & Equilibrium (Premium vs Discount Zone)
        past_20 = raw_candles[max(0, i - 19):i + 1]
        high_20 = max(float(x['high']) for x in past_20)
        low_20 = min(float(x['low']) for x in past_20)
        equilibrium = round((high_20 + low_20) / 2.0, 2)
        zone = "Premium" if cl >= equilibrium else "Discount"
        
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
            "prev_high": prev_high,
            "prev_low": prev_low,
            "change": change,
            "change_pct": change_pct,
            "gap": gap,
            "gap_type": gap_type,
            "body": body,
            "body_pct": body_pct,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "is_expansion_bar": is_expansion_bar,
            "is_pinbar": is_pinbar,
            "candle_type": candle_type,
            "is_inside_day": is_inside_day,
            "is_outside_day": is_outside_day,
            "pdh_sweep": pdh_sweep,
            "pdl_sweep": pdl_sweep,
            "bullish_bos": bullish_bos,
            "bearish_bos": bearish_bos,
            "smc_structure": smc_structure,
            "pwh": pwh,
            "pwl": pwl,
            "equilibrium": equilibrium,
            "zone": zone,
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


def generate_next_day_prediction(latest, prev, adr_20):
    """
    Applies Smart Money Concepts (SMC) algorithmic rules to generate next-day baseline prediction:
    1. Trend Inertia: Big green expansion bar -> continuation bias unless an event or sweep occurs.
    2. Liquidity Sweep: BSL sweep -> bearish mean-reversion; SSL sweep -> bullish mean-reversion.
    3. CPR Width: Narrow CPR -> High expansion probability; Wide CPR -> Rangebound expectation.
    4. Internal / External target identification.
    """
    if not latest:
        return None
        
    bias = "Neutral / Observational"
    confidence = "Moderate"
    rationale = []
    target_pool = f"PDH: {latest['high']}"
    secondary_target = f"PDL: {latest['low']}"
    cpr_expectation = "Average range expected"
    
    # 1. Evaluate CPR Width
    if latest['cpr_width'] <= 40:
        cpr_expectation = f"Narrow CPR ({latest['cpr_width']} pts) indicates high probability of a strong Trending Expansion Day."
    elif latest['cpr_width'] >= 90:
        cpr_expectation = f"Wide CPR ({latest['cpr_width']} pts) suggests Sideways / Rangebound price action between S1 and R1."

    # 2. Evaluate Liquidity Sweeps
    if latest['pdh_sweep']:
        bias = "Bearish Mean-Reversion / Reversal"
        confidence = "High"
        rationale.append("Buy-Side Liquidity (PDH) was swept, but price failed to accept above it and closed lower (Liquidity Run).")
        target_pool = f"Internal Liquidity / CPR: {latest['pivot']} or PDL: {latest['low']}"
    elif latest['pdl_sweep']:
        bias = "Bullish Mean-Reversion / Reversal"
        confidence = "High"
        rationale.append("Sell-Side Liquidity (PDL) was swept and rejected with strong buyer absorption (Turtle Soup).")
        target_pool = f"Internal Liquidity / CPR: {latest['pivot']} or PDH: {latest['high']}"
    elif latest['bullish_bos']:
        bias = "Bullish Trend Continuation"
        confidence = "High"
        rationale.append(f"Break of Structure (BOS): Daily close held cleanly above previous high ({latest['prev_high']}). Trend inertia remains dominant.")
        target_pool = f"Higher Buy-Side Liquidity Pool / R1: {latest['r1']}"
    elif latest['bearish_bos']:
        bias = "Bearish Trend Continuation"
        confidence = "High"
        rationale.append(f"Break of Structure (BOS): Daily close broke below previous low ({latest['prev_low']}). Institutional selling in control.")
        target_pool = f"Lower Sell-Side Liquidity Pool / S1: {latest['s1']}"
    elif latest['is_expansion_bar'] and latest['close'] > latest['open']:
        bias = "Bullish Continuation (Momentum Dominance)"
        confidence = "High"
        rationale.append(f"Big Green Candle with strong body ({latest['body_pct']}% of range). Buyers willing to carry overnight risk.")
        target_pool = f"Next External Buy-Side Liquidity: {latest['high']} (PDH)"
    elif latest['is_inside_day']:
        bias = "Range Expansion / Breakout Setup"
        confidence = "Moderate"
        rationale.append(f"Inside Bar compression day. Market built internal liquidity; watch for an opening sweep of {latest['high']} (PDH) or {latest['low']} (PDL).")
        target_pool = f"PDH: {latest['high']} or PDL: {latest['low']}"
    else:
        if latest['change'] >= 0:
            bias = "Mild Bullish Bias"
            rationale.append("Modest positive close. Bullish continuation holds unless an overnight event or gap invalidates.")
            target_pool = f"PDH: {latest['high']}"
        else:
            bias = "Mild Bearish Bias"
            rationale.append("Modest negative close. Sellers have slight upper hand; watch for support at PDL.")
            target_pool = f"PDL: {latest['low']}"

    return {
        "bias": bias,
        "confidence": confidence,
        "primary_liquidity_target": target_pool,
        "secondary_target": secondary_target,
        "cpr_structure": cpr_expectation,
        "expected_range_adr": f"{adr_20} pts",
        "rationale": " ".join(rationale),
        "event_rule": "Notice: Any unscheduled macro event or opening gap > 60 pts supersedes technical continuation."
    }


def main():
    print("=" * 60)
    print("Fetching NIFTY 50 Market Data with SMC Liquidity Engine...")
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
            
    sorted_raw = [raw_by_date[k] for k in sorted(raw_by_date.keys())]
    print(f"Total deduplicated trading days: {len(sorted_raw)}")
    
    enriched = compute_pivots_and_metrics(sorted_raw, meta)
    
    recent_20 = enriched[-20:] if len(enriched) >= 20 else enriched
    recent_5 = enriched[-5:] if len(enriched) >= 5 else enriched
    adr_20 = round(sum(d["range"] for d in recent_20) / len(recent_20), 2) if recent_20 else 0
    adr_5 = round(sum(d["range"] for d in recent_5) / len(recent_5), 2) if recent_5 else 0
    
    all_highs = [d["high"] for d in enriched]
    all_lows = [d["low"] for d in enriched]
    high_52w = max(all_highs) if all_highs else meta.get("fiftyTwoWeekHigh", 0)
    low_52w = min(all_lows) if all_lows else meta.get("fiftyTwoWeekLow", 0)
    
    latest = enriched[-1] if enriched else None
    prev_day = enriched[-2] if len(enriched) > 1 else None
    
    # Generate Next-Day Prediction using SMC principles
    next_day_prediction = generate_next_day_prediction(latest, prev_day, adr_20)
    
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
        "next_day_prediction": next_day_prediction,
        "history": enriched
    }
    
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
        
    print(f"Successfully saved {len(enriched)} records to {DATA_PATH}")
    if next_day_prediction:
        print("\n--- NEXT-DAY SMC PREDICTION ---")
        print(f"Bias:       {next_day_prediction['bias']} ({next_day_prediction['confidence']} Confidence)")
        print(f"Target:     {next_day_prediction['primary_liquidity_target']}")
        print(f"Rationale:  {next_day_prediction['rationale']}")
        print("-------------------------------\n")


if __name__ == "__main__":
    main()
