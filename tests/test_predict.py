import pytest

from clf.predict import Predictor

LABELS = ["billing", "login", "shipping"]


def test_given_saved_model_when_predicting_then_label_and_score_per_text(tiny_model_dir):
    predictor = Predictor.from_dir(tiny_model_dir)

    predictions = predictor.predict(["where is my parcel", "I forgot my password"])

    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction.label in LABELS
        assert 0.0 <= prediction.score <= 1.0


def test_given_no_texts_when_predicting_then_empty_list(tiny_model_dir):
    assert Predictor.from_dir(tiny_model_dir).predict([]) == []


def test_given_missing_dir_when_loading_then_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor.from_dir(tmp_path / "nope")
