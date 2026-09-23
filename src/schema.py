"""Output JSON schema assembly."""

from datetime import datetime, timezone
from typing import Dict, List, Optional

SCHEMA_VERSION = "1.0"


def build_result(
    source_file: str,
    fields: Dict[str, Dict],
    line_items: List[Dict],
    page_count: int,
) -> Dict:
    """Assembles the final structured JSON payload.

    `fields` maps field_name -> {"value", "confidence", "extracted_by"}.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_file,
        "page_count": page_count,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "fields": fields,
        "line_items": line_items,
    }


def empty_field(value: Optional[object] = None) -> Dict:
    return {"value": value, "confidence": 0.0, "extracted_by": "none"}
