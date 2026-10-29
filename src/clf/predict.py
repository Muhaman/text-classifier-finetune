"""Inference wrapper around a saved fine-tuned model."""

import json
from pathlib import Path
from typing import Any, Self

import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clf.config import DEFAULT_MAX_LEN

LABEL_NAMES_FILE = "label_names.json"


class Prediction(BaseModel):
    label: str
    score: float


class Predictor:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        label_names: list[str],
        max_len: int = DEFAULT_MAX_LEN,
    ) -> None:
        self._model = model.eval()
        self._tokenizer = tokenizer
        self._label_names = label_names
        self._max_len = max_len

    @classmethod
    def from_dir(cls, path: Path | str) -> Self:
        directory = Path(path)
        label_names: list[str] = json.loads((directory / LABEL_NAMES_FILE).read_text())
        tokenizer = AutoTokenizer.from_pretrained(directory)
        model = AutoModelForSequenceClassification.from_pretrained(directory)
        return cls(model, tokenizer, label_names)

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_len,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self._model(**batch).logits
        scores, indices = torch.softmax(logits, dim=-1).max(dim=-1)
        return [
            Prediction(label=self._label_names[index], score=score)
            for index, score in zip(indices.tolist(), scores.tolist(), strict=True)
        ]
