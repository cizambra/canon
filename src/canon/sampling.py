from __future__ import annotations

from collections import Counter

from canon.judge.base import Judge
from canon.models import Constitution, QuestionResult
from canon.rubric import Question


def answer_question(question: Question, artifact: str, task: str,
                    constitution: Constitution, judge: Judge,
                    samples: int = 5) -> QuestionResult:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    system = (
        f"You judge whether a decision coheres with an organization's constitution.\n"
        f"MISSION: {constitution.mission}\n"
        f"PRINCIPLES:\n" + "\n".join(f"- {p}" for p in constitution.principles) +
        "\nAnswer the specific question about the decision, citing evidence from it."
    )
    q_text = f"Task: {task}\nDecision:\n{artifact}\n\nQuestion: {question.text}"
    answers = [judge.ask(system, q_text, question.choices) for _ in range(samples)]
    counts = Counter(a.choice for a in answers)
    choice, count = counts.most_common(1)[0]
    evidence = next(a.evidence for a in answers if a.choice == choice)
    confidence = count / samples
    if question.is_gate:
        # The rubric declares which answer means "violation" (Question.trips_on),
        # settled when it was loaded — nothing is inferred from choice order here.
        return QuestionResult(question.id, question.kind, None, 0.0, question.weight,
                              True, choice == question.trips_on, evidence, confidence,
                              subject=question.subject)
    if choice == "n/a":
        # The facet is not in play for this artifact, so score is None and
        # score_result excludes it from the weighted average, rather than
        # docking a terse-but-coherent artifact on a dimension it never had
        # occasion to exhibit.
        return QuestionResult(question.id, question.kind, None, 0.0, question.weight,
                              False, False, evidence, confidence,
                              subject=question.subject)
    score = float(question.scores.get(choice, 0))
    max_score = float(max(question.scores.values())) if question.scores else 0.0
    return QuestionResult(question.id, question.kind, score, max_score,
                          question.weight, False, False, evidence, confidence,
                          subject=question.subject)
