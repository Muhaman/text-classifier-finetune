"""Training configuration and structured logging setup."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import structlog

DEFAULT_MAX_LEN = 64


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    dataset: str = "PolyAI/banking77"
    epochs: int = 3
    lr: float = 5e-5
    batch_size: int = 32
    max_len: int = DEFAULT_MAX_LEN
    sample: int | None = None
    output_dir: Path = Path("artifacts/model")
    seed: int = 42


def configure_logging() -> None:
    """JSON logs on stderr so stdout stays free for program output."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
