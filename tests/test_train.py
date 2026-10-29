import json

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from clf import train as train_module
from clf.config import TrainConfig

NAMES = ["billing", "login", "shipping"]


def _split(rows_per_class: int) -> Dataset:
    texts = [f"{name} question {i}" for name in NAMES for i in range(rows_per_class)]
    labels = [label for label in range(len(NAMES)) for _ in range(rows_per_class)]
    features = Features({"text": Value("string"), "label": ClassLabel(names=NAMES)})
    return Dataset.from_dict({"text": texts, "label": labels}, features=features)


def test_given_tiny_dataset_when_training_then_model_and_results_are_written(
    monkeypatch, tmp_path, tiny_model_dir
):
    splits = DatasetDict({"train": _split(4), "test": _split(2)})
    monkeypatch.setattr(train_module, "load_data", lambda cfg: (splits, NAMES))
    cfg = TrainConfig(
        model_name=str(tiny_model_dir),
        epochs=1,
        batch_size=4,
        max_len=16,
        output_dir=tmp_path / "model",
    )

    model_dir = train_module.train(cfg, results_dir=tmp_path / "results")

    assert json.loads((model_dir / "label_names.json").read_text()) == NAMES
    assert (model_dir / "config.json").is_file()
    assert any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").is_file()
    metrics = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert set(metrics) == {
        "accuracy",
        "macro_f1",
        "epochs",
        "train_size",
        "test_size",
        "model_name",
        "trained_at",
    }
    assert metrics["train_size"] == 12
    assert metrics["test_size"] == 6
    assert metrics["epochs"] == 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert (tmp_path / "results" / "confusion_matrix.png").is_file()


def test_given_cli_args_when_parsing_then_sample_zero_means_full(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_module, "train", lambda cfg: captured.setdefault("cfg", cfg))

    train_module.main(["--epochs", "2", "--sample", "0"])
    assert captured["cfg"].epochs == 2
    assert captured["cfg"].sample is None

    captured.clear()
    train_module.main(["--sample", "4000"])
    assert captured["cfg"].sample == 4000
