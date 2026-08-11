from __future__ import annotations

from canon.baseline import artifact_key
from canon.models import CoherenceResult, Constitution
from canon.rubric import Rubric, derive_questions, relevant_questions
from canon.sampling import answer_question
from canon.scoring import score_result


class CoherenceMetric:
    def __init__(self, constitution: Constitution, threshold: float = 0.85,
                 judge=None, samples: int = 5, rubric: Rubric | None = None):
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self.constitution = constitution
        self.threshold = threshold
        self.samples = samples
        self.rubric = rubric or Rubric.load_default()
        if judge is None or isinstance(judge, str):
            # A bare model string ("provider:model") is the documented shorthand
            # for "use the default judge against this model".
            from canon.judge.litellm_judge import LiteLLMJudge
            from canon.config import default_judge_model
            self.judge = LiteLLMJudge(model=judge or default_judge_model())
        else:
            self.judge = judge

    def score(self, artifact: str, task: str = "") -> CoherenceResult:
        questions = derive_questions(self.rubric, self.constitution)
        questions, excluded = relevant_questions(questions, artifact, task, self.judge,
                                                 report_excluded=True)
        results = [answer_question(q, artifact, task, self.constitution, self.judge,
                                   self.samples) for q in questions]
        return score_result(results, excluded_principles=excluded,
                            artifact_key=artifact_key(artifact))

    def assert_coheres(self, artifact: str, task: str = "") -> CoherenceResult:
        res = self.score(artifact, task)
        if res.gated or res.score < self.threshold:
            raise AssertionError(f"non-canon ({res.score}): " + "; ".join(res.reasons))
        return res
