"""LayoutLM-based field extraction via document-question-answering.

Uses the word/box layout produced by the DocTR OCR engine (instead of the
model's built-in OCR) so both stages of the pipeline agree on the same text.
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.ocr_engine import Page

logger = logging.getLogger(__name__)

FIELD_QUESTIONS: Dict[str, str] = {
    "invoice_number": "What is the invoice number?",
    "invoice_date": "What is the invoice date?",
    "due_date": "What is the due date?",
    "vendor_name": "Who is the vendor or seller?",
    "bill_to": "Who is the customer or bill to?",
    "po_number": "What is the purchase order number?",
    "subtotal_amount": "What is the subtotal?",
    "tax_amount": "What is the tax amount?",
    "total_amount": "What is the total amount?",
    "currency": "What is the currency?",
}


class FieldExtractor:
    def __init__(self, model_checkpoint: str, device: str = "cpu") -> None:
        from transformers import pipeline

        device_arg = -1 if device == "cpu" else device
        self._pipeline = pipeline(
            "document-question-answering",
            model=model_checkpoint,
            device=device_arg,
        )

    @staticmethod
    def _word_boxes(page: Page) -> List[Tuple[str, List[int]]]:
        return [(w.text, list(w.box)) for w in page.words]

    def extract_field(self, page: Page, question: str) -> Optional[Dict]:
        if not page.words:
            return None
        word_boxes = self._word_boxes(page)
        try:
            answers = self._pipeline(
                image=page.image,
                question=question,
                word_boxes=word_boxes,
                top_k=1,
            )
        except Exception:  # noqa: BLE001 - model inference failure shouldn't crash the pipeline
            logger.exception("Field extraction failed for question: %s", question)
            return None

        if not answers:
            return None
        answer = answers[0] if isinstance(answers, list) else answers
        return {"value": answer.get("answer"), "confidence": float(answer.get("score", 0.0))}

    def extract_all(self, pages: List[Page]) -> Dict[str, Dict]:
        """Runs every field question against every page and keeps the best-scoring answer."""
        results: Dict[str, Dict] = {}
        for field_name, question in FIELD_QUESTIONS.items():
            best: Optional[Dict] = None
            for page in pages:
                candidate = self.extract_field(page, question)
                if candidate is None:
                    continue
                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate
            results[field_name] = best or {"value": None, "confidence": 0.0}
        return results
