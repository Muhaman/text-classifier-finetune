from types import SimpleNamespace

import httpx
import pytest
from anthropic import APIError
from pydantic import ValidationError

from clf import baseline
from clf.baseline import Intent, run_baseline

LABELS = ["card_arrival", "change_pin"]


def _response(label=None, stop_reason="end_turn", input_tokens=1000, output_tokens=100):
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=None if label is None else Intent(label=label),
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class StubMessages:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StubClient:
    def __init__(self, outcomes):
        self.messages = StubMessages(outcomes)
        self.beta = SimpleNamespace(messages=StubMessages(outcomes))


def _run(model, outcomes, expected=("card_arrival",)):
    client = StubClient(outcomes)
    examples = [(f"text {i}", label) for i, label in enumerate(expected)]
    return client, run_baseline(client, model, LABELS, examples)


def test_given_refusal_when_running_then_counts_as_error():
    _, result = _run("claude-haiku-5-5", [_response(stop_reason="refusal")])

    assert (result.total, result.correct, result.errors) == (1, 0, 1)


def test_given_truncated_output_when_running_then_counts_as_error_and_continues():
    with pytest.raises(ValidationError) as truncated:
        Intent.model_validate_json('{"label": "card_arr')
    outcomes = [truncated.value, _response("change_pin")]

    _, result = _run("claude-haiku-5-5", outcomes, expected=("card_arrival", "change_pin"))

    assert (result.total, result.correct, result.errors) == (2, 1, 1)


def test_given_max_tokens_stop_when_running_then_counts_as_error():
    _, result = _run("claude-haiku-5-5", [_response("card_arrival", stop_reason="max_tokens")])

    assert (result.correct, result.errors) == (0, 1)


def test_given_api_error_when_running_then_counts_as_error():
    error = APIError("boom", request=httpx.Request("POST", "https://example.test"), body=None)

    _, result = _run("claude-haiku-5-5", [error])

    assert (result.total, result.errors, result.latencies) == (1, 1, [])


def test_given_haiku_when_classifying_then_no_betas_or_fallbacks():
    client, _ = _run("claude-haiku-5-5", [_response("card_arrival")])

    (call,) = client.messages.calls
    assert "betas" not in call
    assert "fallbacks" not in call
    assert client.beta.messages.calls == []
    assert call["max_tokens"] == baseline.MAX_TOKENS
    assert call["output_config"] == {"effort": "low"}


def test_given_opus_when_classifying_then_fallbacks_are_sent():
    client, _ = _run("claude-opus-5-5", [_response("card_arrival")])

    (call,) = client.beta.messages.calls
    assert call["betas"] == [baseline.FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert client.messages.calls == []


def test_given_counters_when_summarising_then_accuracy_and_cost_are_derived():
    outcomes = [_response("card_arrival"), _response("card_arrival"), _response("change_pin")]

    _, result = _run(
        "claude-haiku-5-5", outcomes, expected=("card_arrival", "change_pin", "change_pin")
    )

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.cost("claude-haiku-5-5") == pytest.approx((3000 * 0.10 + 300 * 0.50) / 1e6)
    assert result.cost("some-other-model") is None
    assert result.mean_latency is not None
