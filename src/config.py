"""Central configuration loaded from environment variables / .env."""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    model_checkpoint_path: str
    device: str
    ocr_min_confidence: float
    enable_regex_fallback: bool
    output_dir: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model_checkpoint_path=os.getenv("MODEL_CHECKPOINT_PATH", "impira/layoutlm-invoices"),
            device=os.getenv("DEVICE", "cpu"),
            ocr_min_confidence=float(os.getenv("OCR_MIN_CONFIDENCE", "0.5")),
            enable_regex_fallback=_str_to_bool(os.getenv("ENABLE_REGEX_FALLBACK", "true")),
            output_dir=os.getenv("OUTPUT_DIR", "output"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
