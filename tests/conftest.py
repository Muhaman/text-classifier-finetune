import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, BertForSequenceClassification

TINY_MODEL = "prajjwal1/bert-tiny"
# bert-tiny has no model_type in config.json and ships only vocab.txt, so the Auto classes cannot
# resolve it: load the model as BERT and borrow the identical uncased-vocab tokenizer.
TOKENIZER = "distilbert-base-uncased"
LABELS = ["billing", "login", "shipping"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny BERT with a randomly initialised 3-label head, saved like a trained model."""
    path = tmp_path_factory.mktemp("tiny-model")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = BertForSequenceClassification.from_pretrained(TINY_MODEL, num_labels=len(LABELS))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "label_names.json").write_text(json.dumps(LABELS))
    return path
