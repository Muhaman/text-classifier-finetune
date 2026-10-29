# Model card: Banking77 intent classifier

## Model

`distilbert-base-uncased` fine-tuned for 77-way sequence classification. Weights are not committed; they are produced by the `train` workflow and uploaded as a workflow artifact.

## Intended use

Routing short English customer-service messages from a retail bank to one of the 77 Banking77 intents. A demonstration of the fine-tune, evaluate, serve loop; not a production system.

## Data

[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77): 10,003 training and 3,080 test utterances across 77 intents.

## Training

AdamW via HF `Trainer`, learning rate 5e-5, batch size 32, max sequence length 64, 3 epochs by default, seed 42. CPU training on a GitHub Actions runner.

## Evaluation

Accuracy and macro-F1 on the test split, computed once after the final epoch; see `results/metrics.json` and `results/confusion_matrix.png` after a `train` run. `scripts/zero_shot_baseline.py` measures a zero-shot Claude baseline on a stratified test subset for comparison.

## Limitations

- English only; short single-turn messages only.
- Covers only the 77 Banking77 intents; inputs outside them still receive a label.
- Probabilities are uncalibrated.
- Inherits the biases and phrasing style of the crowd-sourced dataset.
