from collections import Counter

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from clf import data
from clf.config import TrainConfig

NAMES = ["billing", "login", "shipping"]


def _fake_split(per_class: int) -> Dataset:
    texts, labels = [], []
    for label in range(len(NAMES)):
        for i in range(per_class):
            texts.append(f"{NAMES[label]} example {i}")
            labels.append(label)
    features = Features({"text": Value("string"), "label": ClassLabel(names=NAMES)})
    return Dataset.from_dict({"text": texts, "label": labels}, features=features)


@pytest.fixture
def fake_splits() -> DatasetDict:
    return DatasetDict({"train": _fake_split(40), "test": _fake_split(20)})


def test_given_sample_when_loading_then_every_class_is_kept(monkeypatch, fake_splits):
    monkeypatch.setattr(data, "load_dataset", lambda name: fake_splits)

    splits, names = data.load_data(TrainConfig(sample=20))

    assert names == NAMES
    assert len(splits["train"]) == 20
    assert set(Counter(splits["train"]["label"])) == {0, 1, 2}
    assert set(Counter(splits["test"]["label"])) == {0, 1, 2}


def test_given_no_sample_when_loading_then_splits_are_untouched(monkeypatch, fake_splits):
    monkeypatch.setattr(data, "load_dataset", lambda name: fake_splits)

    splits, names = data.load_data(TrainConfig(sample=None))

    assert names == NAMES
    assert len(splits["train"]) == 120
    assert len(splits["test"]) == 60


def test_given_tiny_sample_when_sampling_then_each_class_keeps_at_least_one():
    sampled = data.stratified_sample(_fake_split(40), n=3, seed=1)

    assert sorted(sampled["label"]) == [0, 1, 2]


def test_given_same_seed_when_sampling_then_result_is_deterministic():
    first = data.stratified_sample(_fake_split(40), n=15, seed=7)
    second = data.stratified_sample(_fake_split(40), n=15, seed=7)

    assert first["text"] == second["text"]
