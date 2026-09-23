"""Orchestrates OCR -> LayoutLM field extraction -> regex fallback -> schema output."""

import logging
from typing import Dict

from src import regex_fallback, schema
from src.config import Config
from src.field_extractor import FieldExtractor
from src.line_items import extract_line_items
from src.ocr_engine import OCREngine
from src.postprocess import normalize_amount, normalize_currency, normalize_date

logger = logging.getLogger(__name__)

_DATE_FIELDS = {"invoice_date", "due_date"}
_AMOUNT_FIELDS = {"subtotal_amount", "tax_amount", "total_amount"}
_CURRENCY_FIELDS = {"currency"}


def _normalize_value(field_name: str, value):
    if value is None:
        return None
    if field_name in _DATE_FIELDS:
        return normalize_date(value)
    if field_name in _AMOUNT_FIELDS:
        return normalize_amount(value)
    if field_name in _CURRENCY_FIELDS:
        return normalize_currency(value)
    return value.strip() if isinstance(value, str) else value


class InvoiceExtractionPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        logger.info("Loading OCR engine...")
        self.ocr_engine = OCREngine()
        logger.info("Loading field extraction model: %s", config.model_checkpoint_path)
        self.field_extractor = FieldExtractor(config.model_checkpoint_path, device=config.device)

    def process(self, file_path: str) -> Dict:
        pages = self.ocr_engine.process(file_path)
        if not pages:
            raise RuntimeError(f"No pages could be read from: {file_path}")

        full_text = "\n".join(page.text for page in pages)
        model_fields = self.field_extractor.extract_all(pages)
        regex_fields = regex_fallback.extract_all(full_text) if self.config.enable_regex_fallback else {}

        final_fields: Dict[str, Dict] = {}
        for field_name, result in model_fields.items():
            confidence = result.get("confidence", 0.0)
            value = result.get("value")
            extracted_by = "layoutlm" if value else "none"

            if confidence < self.config.ocr_min_confidence:
                fallback = regex_fields.get(field_name)
                if fallback is not None:
                    value = fallback["value"]
                    confidence = fallback["confidence"]
                    extracted_by = "regex"

            final_fields[field_name] = {
                "value": _normalize_value(field_name, value),
                "confidence": round(confidence, 4),
                "extracted_by": extracted_by,
            }

        best_page = max(pages, key=lambda p: len(p.words)) if pages else None
        line_items = extract_line_items(best_page) if best_page else []

        result = schema.build_result(
            source_file=file_path,
            fields=final_fields,
            line_items=line_items,
            page_count=len(pages),
        )
        logger.info("Extraction complete for %s", file_path)
        return result
