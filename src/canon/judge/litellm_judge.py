from __future__ import annotations

import json

from canon.errors import JudgeError
from canon.judge.base import Answer, Judge


class LiteLLMJudge(Judge):
    def __init__(self, model: str, temperature: float | None = None, request_timeout: float = 60.0):
        # Accept both Canon's documented `provider:model` and LiteLLM's own
        # `provider/model`, so a string copied from either reads the same.
        self.model = _normalize_model(model)
        # Reasoning models reject any explicit temperature, so Canon sends one
        # only when a caller names it. Repeatability comes from the N-sample
        # majority vote, not from a pinned temperature.
        self.temperature = temperature
        # Bound each call so one stalled provider socket can't block a whole
        # `canon check` indefinitely (a slow/parked call surfaces as JudgeError).
        self.request_timeout = request_timeout

    def ask(self, system: str, question: str, choices: tuple[str, ...]) -> Answer:
        import litellm

        prompt = (
            f"{question}\n\nChoose exactly one of: {list(choices)}.\n"
            'Return ONLY JSON: {"choice": <one of the options>, '
            '"evidence": <one short sentence citing the artifact>}'
        )
        optional = {} if self.temperature is None else {"temperature": self.temperature}
        try:
            resp = litellm.completion(
                model=self.model,
                timeout=self.request_timeout,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **optional,
            )
            text = resp["choices"][0]["message"]["content"]
        except Exception as exc:  # network/provider/parse — surface uniformly
            raise JudgeError(f"judge call failed: {exc}") from exc
        try:
            data = json.loads(_extract_json(text))
            choice, evidence = data["choice"], data.get("evidence", "")
        except Exception as exc:
            raise JudgeError(f"judge returned unparseable answer: {text!r}") from exc
        if choice not in choices:
            raise JudgeError(f"judge chose {choice!r} not in {choices}")
        return Answer(choice=choice, evidence=str(evidence))


def _normalize_model(model: str) -> str:
    """`provider:model` -> `provider/model`, including `provider:org/model`.
    Only a colon before every slash is a provider separator; a later colon
    belongs to the model name itself (e.g. `openai/ft:gpt-x`) and is left
    alone. No provider aliasing happens here — that's LiteLLM's job."""
    if ":" not in model:
        return model
    slash = model.find("/")
    if slash != -1 and model.index(":") > slash:
        return model
    return model.replace(":", "/", 1)


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in judge output")
    return text[start : end + 1]
