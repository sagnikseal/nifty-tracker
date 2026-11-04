"""Indian F&O statutory charges (NSE, as of Nov-2026). All rates configurable.

Sources: Budget-2026 STT hike effective 1-Apr-2026 (futures 0.05%, options 0.15% of
sell premium; NSE circular NSE/FATAX/73524), NSE uniform transaction charges from
1-Oct-2024, SEBI turnover fee Rs10/crore, stamp duty (Indian Stamp Act 2020 rates).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChargeRates:
    stt_option_sell: float = 0.0015      # on premium turnover, sell side
    stt_future_sell: float = 0.0005
    txn_option: float = 0.0003503        # NSE transaction charge on premium turnover
    txn_future: float = 0.0000173
    sebi_fee: float = 0.000001           # Rs 10 per crore
    stamp_option_buy: float = 0.00003
    stamp_future_buy: float = 0.00002
    gst: float = 0.18
    brokerage_per_order: float = 20.0


def order_charges(instrument: str, side: str, turnover: float, rates: ChargeRates) -> dict:
    """Charges for one executed order. instrument: 'option'|'future'; side: 'BUY'|'SELL';
    turnover = price * quantity (premium for options)."""
    opt = instrument == "option"
    brokerage = min(rates.brokerage_per_order, 0.0025 * turnover) if turnover > 0 else 0.0
    stt = (rates.stt_option_sell if opt else rates.stt_future_sell) * turnover if side == "SELL" else 0.0
    txn = (rates.txn_option if opt else rates.txn_future) * turnover
    sebi = rates.sebi_fee * turnover
    stamp = (rates.stamp_option_buy if opt else rates.stamp_future_buy) * turnover if side == "BUY" else 0.0
    gst = rates.gst * (brokerage + txn + sebi)
    total = brokerage + stt + txn + sebi + stamp + gst
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "txn": round(txn, 2),
        "sebi": round(sebi, 4),
        "stamp": round(stamp, 2),
        "gst": round(gst, 2),
        "total": round(total, 2),
    }
