"""DocTR-based OCR engine: pixel-level text and bounding box extraction."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Word:
    text: str
    confidence: float
    box: Tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel coordinates


@dataclass
class Page:
    image: Image.Image
    words: List[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


class OCREngine:
    """Wraps python-doctr to produce per-page words with pixel bounding boxes."""

    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self) -> None:
        from doctr.models import ocr_predictor

        self._predictor = ocr_predictor(pretrained=True)

    def process(self, file_path: str) -> List[Page]:
        from doctr.io import DocumentFile

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            doc_images = DocumentFile.from_pdf(str(path))
        elif path.suffix.lower() in self.SUPPORTED_IMAGE_EXTS:
            doc_images = DocumentFile.from_images(str(path))
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        result = self._predictor(doc_images)

        pages: List[Page] = []
        for raw_image, page_result in zip(doc_images, result.pages):
            pil_image = Image.fromarray(raw_image)
            width, height = pil_image.size
            words: List[Word] = []
            for block in page_result.blocks:
                for line in block.lines:
                    for word in line.words:
                        (x0, y0), (x1, y1) = word.geometry
                        words.append(
                            Word(
                                text=word.value,
                                confidence=float(word.confidence),
                                box=(
                                    int(x0 * width),
                                    int(y0 * height),
                                    int(x1 * width),
                                    int(y1 * height),
                                ),
                            )
                        )
            pages.append(Page(image=pil_image, words=words))

        logger.info("OCR extracted %d page(s), %d total word(s)", len(pages), sum(len(p.words) for p in pages))
        return pages
