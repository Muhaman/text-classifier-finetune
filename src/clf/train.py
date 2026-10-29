"""Fine-tune a sequence classifier, evaluate it on the test split and write artifacts."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from clf.config import TrainConfig, configure_logging
from clf.data import load_data
from clf.metrics import compute_metrics
from clf.predict import LABEL_NAMES_FILE

log = structlog.get_logger(__name__)

RESULTS_DIR = Path("results")


def train(cfg: TrainConfig, results_dir: Path = RESULTS_DIR) -> Path:
    """Train on the train split, evaluate once on the test split, return the model directory.

    The final-epoch model is kept: the dataset has no validation split, so nothing is
    selected on test data.
    """
    set_seed(cfg.seed)
    splits, label_names = load_data(cfg)
    log.info(
        "data_loaded",
        train_size=len(splits["train"]),
        test_size=len(splits["test"]),
        labels=len(label_names),
    )

    tokenizer: Any = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenized = splits.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=cfg.max_len),
        batched=True,
        remove_columns=["text"],
    ).rename_column("label", "labels")

    model: Any = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: index for index, name in enumerate(label_names)},
    )
    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        seed=cfg.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(cfg.output_dir)
    (cfg.output_dir / LABEL_NAMES_FILE).write_text(json.dumps(label_names))

    output = trainer.predict(tokenized["test"])
    scores = compute_metrics((output.predictions, output.label_ids))
    log.info("evaluated", **scores)

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        **scores,
        "epochs": cfg.epochs,
        "train_size": len(splits["train"]),
        "test_size": len(splits["test"]),
        "model_name": cfg.model_name,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _save_confusion_matrix(
        output.label_ids, output.predictions.argmax(axis=-1), results_dir / "confusion_matrix.png"
    )
    return cfg.output_dir


def _save_confusion_matrix(expected: Any, predicted: Any, path: Path) -> None:
    matrix = confusion_matrix(expected, predicted)
    figure = Figure(figsize=(12, 10))
    axes = figure.subplots()
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image)
    axes.set_xlabel("Predicted label index")
    axes.set_ylabel("True label index")
    axes.set_title("Confusion matrix (test split)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Fine-tune the intent classifier.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--sample", type=int, default=0, help="stratified subset size; 0 = full")
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    ns = parser.parse_args(argv)

    configure_logging()
    cfg = TrainConfig(
        model_name=ns.model_name,
        epochs=ns.epochs,
        sample=ns.sample or None,
        output_dir=ns.output_dir,
    )
    train(cfg)


if __name__ == "__main__":
    main()
