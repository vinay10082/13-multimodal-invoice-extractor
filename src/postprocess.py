"""Normalization helpers applied to raw extracted field values."""

import re
from typing import Optional

from dateutil import parser as date_parser

_AMOUNT_CLEAN_RE = re.compile(r"[^\d.,\-]")


def normalize_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = date_parser.parse(value, fuzzy=True)
    except (ValueError, OverflowError):
        return value
    return parsed.date().isoformat()


def normalize_amount(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    cleaned = _AMOUNT_CLEAN_RE.sub("", str(value)).replace(",", "")
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def normalize_currency(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    symbol_map = {"$": "USD", "€": "EUR", "£": "GBP"}
    value = value.strip()
    return symbol_map.get(value, value.upper())
