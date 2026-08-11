from __future__ import annotations

from collections.abc import Callable

from canon.errors import JudgeError
from canon.judge.base import Answer, Judge

DEFAULT_KEY = "__default__"


class MockJudge(Judge):
    """Deterministic judge for tests; `script` maps a question substring to a
    choice, or a callable(question, choices) -> choice. An unmatched question
    is an error, not a silent fallback to choices[0] — that used to disable
    facets unnoticed. Use a `"__default__"` entry for a real catch-all."""

    def __init__(self, script: dict[str, str] | Callable[[str, tuple[str, ...]], str]):
        self._script = script

    def ask(self, system: str, question: str, choices: tuple[str, ...]) -> Answer:
        if callable(self._script):
            choice = self._script(question, choices)
        else:
            choice = next(
                (v for k, v in self._script.items() if k != DEFAULT_KEY and k in question), None
            )
            if choice is None:
                choice = self._script.get(DEFAULT_KEY)
            if choice is None:
                raise JudgeError(f"MockJudge has no script entry matching {question!r}")
        if choice not in choices:
            raise ValueError(f"scripted choice {choice!r} not in {choices}")
        return Answer(choice=choice, evidence="mock")
