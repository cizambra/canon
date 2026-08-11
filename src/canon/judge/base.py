from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    choice: str
    evidence: str


class Judge(abc.ABC):
    @abc.abstractmethod
    def ask(self, system: str, question: str, choices: tuple[str, ...]) -> Answer:
        """Return one of `choices` plus one line of cited evidence."""
