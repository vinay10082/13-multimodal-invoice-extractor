"""Best-effort line-item table extraction from OCR word boxes.

Groups words into rows by vertical (y) proximity, then treats rows that end
in a currency-like number as candidate line items: leading text becomes the
description, the trailing numeric token becomes the amount.
"""

import re
from typing import Dict, List

from src.ocr_engine import Page
from src.postprocess import normalize_amount

_AMOUNT_RE = re.compile(r"^[$€£]?[\d,]+\.\d{2}$")
_ROW_Y_TOLERANCE = 8  # pixels


def _group_rows(page: Page) -> List[List]:
    words_sorted = sorted(page.words, key=lambda w: (w.box[1], w.box[0]))
    rows: List[List] = []
    for word in words_sorted:
        y_center = (word.box[1] + word.box[3]) / 2
        placed = False
        for row in rows:
            row_y_center = sum((w.box[1] + w.box[3]) / 2 for w in row) / len(row)
            if abs(y_center - row_y_center) <= _ROW_Y_TOLERANCE:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])
    for row in rows:
        row.sort(key=lambda w: w.box[0])
    return rows


def extract_line_items(page: Page) -> List[Dict]:
    rows = _group_rows(page)
    line_items: List[Dict] = []
    for row in rows:
        if len(row) < 2:
            continue
        trailing = row[-1].text
        if not _AMOUNT_RE.match(trailing):
            continue
        description = " ".join(w.text for w in row[:-1])
        amount = normalize_amount(trailing)
        if amount is None:
            continue
        line_items.append({"description": description, "amount": amount})
    return line_items
