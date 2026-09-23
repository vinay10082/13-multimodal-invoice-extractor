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
Submit a document image (PNG/PDF) to the pipeline to receive a strictly formatted JSON payload containing the extracted financial fields.

## Testing & CI
Evaluates extraction F1 scores against synthetic invoice datasets and validates the successful triggering of the regex fallback mechanism.
