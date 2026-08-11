import pytest
from canon.judge.base import Judge, Answer
from canon.judge.mock import MockJudge


def test_mock_judge_returns_scripted_choice():
    # The key must be a real substring of the question — this test previously
    # passed on the silent choices[0] fallback, not on its own script.
    j = MockJudge(script={"serve the mission": "yes"})
    a = j.ask("sys", "Does this serve the mission?", ("yes", "no"))
    assert isinstance(a, Answer) and a.choice == "yes"


def test_mock_judge_rejects_offscript_choice():
    j = MockJudge(script={"q": "maybe"})
    with pytest.raises(ValueError):
        j.ask("sys", "q", ("yes", "no"))   # "maybe" not in choices


def test_litellm_normalizes_provider_string():
    from canon.judge.litellm_judge import LiteLLMJudge
    j = LiteLLMJudge(model="openai:gpt-5.6-luna")
    assert j.model == "openai/gpt-5.6-luna"


def test_litellm_ask_parses_and_returns_answer(monkeypatch):
    from canon.judge.litellm_judge import LiteLLMJudge
    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **k: {
        "choices": [{"message": {"content": '{"choice": "yes", "evidence": "because X"}'}}]})
    a = LiteLLMJudge(model="openai:gpt-x").ask("sys", "q?", ("yes", "no"))
    assert a.choice == "yes" and a.evidence == "because X"


def test_litellm_ask_rejects_offchoices(monkeypatch):
    from canon.judge.litellm_judge import LiteLLMJudge
    from canon.errors import JudgeError
    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **k: {
        "choices": [{"message": {"content": '{"choice": "maybe", "evidence": "e"}'}}]})
    with pytest.raises(JudgeError):
        LiteLLMJudge(model="openai:gpt-x").ask("sys", "q?", ("yes", "no"))


def test_litellm_ask_unparseable_raises(monkeypatch):
    from canon.judge.litellm_judge import LiteLLMJudge
    from canon.errors import JudgeError
    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **k: {
        "choices": [{"message": {"content": "not json at all"}}]})
    with pytest.raises(JudgeError):
        LiteLLMJudge(model="openai:gpt-x").ask("sys", "q?", ("yes", "no"))


def test_litellm_normalizes_colon_before_a_slashed_model_path():
    """`provider:org/model` is a documented form — the colon still normalizes."""
    from canon.judge.litellm_judge import LiteLLMJudge
    j = LiteLLMJudge(model="together:deepseek-ai/DeepSeek-V4-Flash-0731")
    assert j.model == "together/deepseek-ai/DeepSeek-V4-Flash-0731"


def test_litellm_passes_through_already_slashed_model():
    from canon.judge.litellm_judge import LiteLLMJudge
    assert LiteLLMJudge(model="openai/gpt-5.6-luna").model == "openai/gpt-5.6-luna"


def test_litellm_leaves_a_colon_after_the_first_slash_untouched():
    """Fine-tune ids carry their own colon; only a provider prefix is normalized."""
    from canon.judge.litellm_judge import LiteLLMJudge
    assert LiteLLMJudge(model="openai/ft:gpt-x").model == "openai/ft:gpt-x"


def test_mock_judge_raises_when_no_script_entry_matches():
    """Silently answering choices[0] made suites vacuous without saying so."""
    from canon.errors import JudgeError
    j = MockJudge(script={"a question we never ask": "yes"})
    with pytest.raises(JudgeError, match="no script entry matching"):
        j.ask("sys", "some other question", ("n/a", "no", "yes"))


def test_mock_judge_honours_an_explicit_default_entry():
    j = MockJudge(script={"__default__": "no", "specific": "yes"})
    assert j.ask("sys", "a specific question", ("no", "yes")).choice == "yes"
    assert j.ask("sys", "anything else", ("no", "yes")).choice == "no"


def test_litellm_passes_request_timeout_and_temperature_through(monkeypatch):
    from canon.judge.litellm_judge import LiteLLMJudge
    import litellm
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": '{"choice": "yes", "evidence": "e"}'}}]}

    monkeypatch.setattr(litellm, "completion", fake_completion)
    LiteLLMJudge(model="openai:gpt-x", temperature=0.3, request_timeout=12.5).ask(
        "sys", "q?", ("yes", "no"))
    assert seen["temperature"] == 0.3
    assert seen["timeout"] == 12.5


def test_extract_json_from_a_json_fenced_code_block():
    from canon.judge.litellm_judge import _extract_json
    text = ('Sure, here is my answer:\n```json\n'
            '{"choice": "yes", "evidence": "because X"}\n```\nThanks!')
    assert _extract_json(text) == '{"choice": "yes", "evidence": "because X"}'


def test_extract_json_from_surrounding_prose():
    from canon.judge.litellm_judge import _extract_json
    text = 'The answer is {"choice": "yes", "evidence": "because X"} as explained above.'
    assert _extract_json(text) == '{"choice": "yes", "evidence": "because X"}'


def test_missing_evidence_key_defaults_to_empty_string(monkeypatch):
    from canon.judge.litellm_judge import LiteLLMJudge
    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **k: {
        "choices": [{"message": {"content": '{"choice": "yes"}'}}]})
    a = LiteLLMJudge(model="openai:gpt-x").ask("sys", "q?", ("yes", "no"))
    assert a.choice == "yes" and a.evidence == ""


def test_litellm_normalizes_a_bare_model_with_no_colon_unchanged():
    from canon.judge.litellm_judge import LiteLLMJudge
    assert LiteLLMJudge(model="gpt-4").model == "gpt-4"


def test_missing_provider_key_surfaces_as_judge_error_without_a_network_call(monkeypatch):
    """Env scrubbed of provider keys: a real LiteLLMJudge.ask must surface the
    auth failure as JudgeError, uniformly with any other provider failure —
    litellm.completion is monkeypatched to raise the auth error litellm
    itself raises, so no network call ever happens."""
    from canon.judge.litellm_judge import LiteLLMJudge
    from canon.errors import JudgeError
    import litellm
    import os

    for k in list(os.environ):
        if "API_KEY" in k:
            monkeypatch.delenv(k, raising=False)

    def raise_auth(**kwargs):
        raise litellm.exceptions.AuthenticationError(
            message="missing API key", llm_provider="openai", model="gpt-x")

    monkeypatch.setattr(litellm, "completion", raise_auth)
    with pytest.raises(JudgeError):
        LiteLLMJudge(model="openai:gpt-x").ask("sys", "q?", ("yes", "no"))
