from __future__ import annotations

from canon.judge.base import Answer, Judge
from canon.judge.mock import MockJudge

__all__ = ["Answer", "Judge", "MockJudge", "resolve_judge"]


def resolve_judge(judge: Judge | str | None) -> Judge:
    """Turn what a caller passed for `judge` into a Judge.
    None or a bare model string ("provider:model") is the documented shorthand
    for "use the default judge against this model", so every entry point spells
    the shorthand the same way."""
    if judge is not None and not isinstance(judge, str):
        return judge
    from canon.config import default_judge_model
    from canon.judge.litellm_judge import LiteLLMJudge

    return LiteLLMJudge(model=judge or default_judge_model())
