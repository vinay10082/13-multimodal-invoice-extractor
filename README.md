# Invoice Field Extractor

## Description
A multimodal AI extraction pipeline for transforming unstructured invoice imagery into highly structured, validated JSON data.

## Architecture Overview
LayoutLMv3 transformer for primary spatial extraction, coupled with DocTR for OCR and a Regex heuristic fallback engine for critical field recovery.

### Extraction Engine Components
| Technology | Role in Architecture |
|---|---|
| **LayoutLMv3** | Context-aware, spatial semantic entity recognition (NER) combining text and vision. |
| **DocTR / PaddleOCR** | Hierarchical text extraction and bounding box detection directly from pixels. |
| **Regex Heuristics** | Deterministic extraction triggered automatically upon low AI confidence scores. |

## Prerequisites
* Python 3.11+
* `transformers`
* `torch`
* `doctr`
* CUDA-compatible GPU (recommended for inference speed).

## Environment Variables
* `MODEL_CHECKPOINT_PATH`
* `OCR_MIN_CONFIDENCE`
* `ENABLE_REGEX_FALLBACK`

## Quick Start & Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` (or edit the provided `.env`) to configure the model checkpoint, device, and confidence threshold.
3. Run the extractor against an invoice image or PDF:

```
python main.py path/to/invoice.pdf
python main.py path/to/invoice.png --output result.json --pretty
```

The pipeline runs DocTR OCR to extract words and bounding boxes, feeds them into a LayoutLM document-question-answering model to answer per-field questions (invoice number, dates, vendor, amounts, etc.), and falls back to deterministic regex heuristics for any field whose model confidence falls below `OCR_MIN_CONFIDENCE`. The result is written as JSON (to `OUTPUT_DIR` by default) and printed to stdout, with each field annotated with its value, confidence score, and extraction source (`layoutlm` or `regex`).
