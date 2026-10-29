"""Dataset loading and stratified sampling."""

import random
from collections import defaultdict
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset

from clf.config import TrainConfig


def load_data(cfg: TrainConfig) -> tuple[DatasetDict, list[str]]:
    """Load the dataset splits and the ordered label names.

    When ``cfg.sample`` is set, every split is replaced by a stratified subset of that size.
    """
    splits = cast(DatasetDict, load_dataset(cfg.dataset))
    label_names = list(splits["train"].features["label"].names)
    if cfg.sample is not None:
        splits = DatasetDict(
            {name: stratified_sample(split, cfg.sample, cfg.seed) for name, split in splits.items()}
        )
    return splits, label_names


def stratified_sample(ds: Dataset, n: int, seed: int) -> Dataset:
    """Pick about ``n`` rows keeping class proportions and at least one row per class."""
    if n >= len(ds):
        return ds
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(ds["label"]):
        by_label[label].append(index)
    quotas = _quotas({label: len(indices) for label, indices in by_label.items()}, n)
    rng = random.Random(seed)
    picked: list[int] = []
    for label, indices in sorted(by_label.items()):
        rng.shuffle(indices)
        picked.extend(indices[: quotas[label]])
    return ds.select(sorted(picked))


def _quotas(sizes: dict[int, int], n: int) -> dict[int, int]:
    """Largest-remainder allocation of ``n`` rows across classes, minimum one each."""
    total = sum(sizes.values())
    exact = {label: n * size / total for label, size in sizes.items()}
    quotas = {label: max(1, int(value)) for label, value in exact.items()}
    by_fraction = sorted(exact, key=lambda label: exact[label] - int(exact[label]), reverse=True)
    for label in by_fraction[: max(n - sum(quotas.values()), 0)]:
        quotas[label] += 1
    return {label: min(quota, sizes[label]) for label, quota in quotas.items()}
