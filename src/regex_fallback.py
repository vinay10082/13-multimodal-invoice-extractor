"""Deterministic regex heuristics for recovering critical fields when the
LayoutLM extractor reports low confidence."""

import re
from typing import Dict, Optional

FALLBACK_CONFIDENCE = 0.4

_PATTERNS: Dict[str, re.Pattern] = {
    "invoice_number": re.compile(
        r"invoice\s*(?:number|no\.?|num\.?|#)?\s*[:#\-]?\s*"
        r"(?=[A-Z0-9\-\/]*\d)([A-Z0-9][A-Z0-9\-\/]{2,19})",
        re.IGNORECASE,
    ),
    "po_number": re.compile(
        r"(?:p\.?\s*o\.?|purchase\s*order)\s*(?:number|no\.?|#)?\s*[:#\-]?\s*"
        r"(?=[A-Z0-9\-\/]*\d)([A-Z0-9][A-Z0-9\-\/]{2,19})",
        re.IGNORECASE,
    ),
    "invoice_date": re.compile(
        r"(?:invoice\s*date|date)\s*[:#]?\s*"
        r"([0-3]?\d[\/\-\.][01]?\d[\/\-\.]\d{2,4}|"
        r"[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    ),
    "due_date": re.compile(
        r"(?:due\s*date|payment\s*due)\s*[:#]?\s*"
        r"([0-3]?\d[\/\-\.][01]?\d[\/\-\.]\d{2,4}|"
        r"[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    ),
    "total_amount": re.compile(
        r"\b(?:grand\s*total|total\s*due|total\s*amount|amount\s*due|total)\b\s*[:#]?\s*"
        r"[$€£]?\s*([\d,]+\.\d{2})",
        re.IGNORECASE,
    ),
    "subtotal_amount": re.compile(
        r"\b(?:sub\s*-?\s*total)\b\s*[:#]?\s*[$€£]?\s*([\d,]+\.\d{2})", re.IGNORECASE
    ),
    "tax_amount": re.compile(
        r"(?:tax|vat|gst)\s*(?:\(\s*[\d.]+\s*%\s*\))?\s*[:#]?\s*[$€£]?\s*([\d,]+\.\d{2})",
        re.IGNORECASE,
    ),
    "currency": re.compile(r"(USD|EUR|GBP|INR|\$|€|£)"),
}

_CURRENCY_SYMBOL_MAP = {"$": "USD", "€": "EUR", "£": "GBP"}


def extract_field(field_name: str, text: str) -> Optional[Dict]:
    """Returns {'value': ..., 'confidence': ...} for the given field, or None."""
    pattern = _PATTERNS.get(field_name)
    if pattern is None:
        return None
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    if field_name == "currency":
        value = _CURRENCY_SYMBOL_MAP.get(value.upper(), value.upper())
    return {"value": value, "confidence": FALLBACK_CONFIDENCE}


def extract_all(text: str) -> Dict[str, Dict]:
    return {field: result for field in _PATTERNS if (result := extract_field(field, text)) is not None}
