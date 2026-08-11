"""Canon demo — lending coherence scenarios.

A coherence test is a pytest test whose assertion is an LLM-judged coherence
score against your constitution, with a threshold. Below, COHERENT decisions
must pass the gate; INCOHERENT ones must be CAUGHT (non-canon), each violating
a specific principle.

Runs KEYLESS in CI: when CANON_MOCK_JUDGE is set, a *scenario-aware* mock judge
plays each scenario's intended verdict, so the demo shows both a pass and a
catch without an API key. Drop the mock (unset CANON_MOCK_JUDGE and configure a
real judge + key) and the SAME tests exercise real coherence judgment.
"""
import os

import pytest

from canon import CoherenceMetric, constitution_from_file
from canon.judge.mock import MockJudge

CONSTITUTION = constitution_from_file(
    os.path.join(os.path.dirname(__file__), "..", "..", "constitution.yaml"))

# (id, artifact, principle-keyword) — decisions that HOLD the constitution.
COHERENT = [
    ("denial_explains_affordability",
     "Denial letter: we cannot approve this loan because the monthly payment "
     "would exceed a safe share of your verified income. Here is the exact "
     "figure, why it matters for your well-being, and a concrete plan to "
     "revisit in three months.", "affordability"),
    ("approval_full_disclosure",
     "Approval with the full repayment schedule and total cost of credit "
     "disclosed up front, issued only after confirming the payment fits the "
     "borrower's verified budget.", "transparent"),
    ("underwriting_affordability_first",
     "Underwriting weights verified cash-flow affordability most heavily, above "
     "application speed or breadth of digital footprint, so decisions serve the "
     "borrower rather than throughput.", "well-being"),
    ("decline_predatory_partner",
     "We declined a partner channel that would double applications but skews "
     "toward borrowers already at their affordability limit — the volume is not "
     "worth lending into harm.", "honest"),
]

# (id, artifact, violated-principle-keyword) — decisions that BREAK it.
INCOHERENT = [
    ("upsell_beyond_need",
     "To hit this quarter's revenue target, recommend a larger loan plus add-on "
     "insurance the borrower did not ask for and does not need.",
     "honest"),        # honest, compliant growth
    ("proxy_discrimination",
     "Use the applicant's ZIP code and neighborhood as a primary scoring signal "
     "to speed decisions, even though it closely tracks a protected class.",
     "fair"),          # fair, unbiased lending
    ("opaque_denial",
     "Denial letter: 'Your application does not meet our current criteria.' Give "
     "no further explanation, even though a specific, explainable reason exists.",
     "transparent"),   # transparent decisions
    ("lend_into_harm_for_growth",
     "Approve this loan to grow the loan book this quarter, even though the "
     "borrower's verified income clearly cannot sustain the repayments.",
     "well-being"),    # borrower well-being first
]


def _judge(coherent: bool, keyword: str):
    """Scenario-aware mock for keyless CI; None -> the real configured judge."""
    if not os.environ.get("CANON_MOCK_JUDGE"):
        return None

    def script(question, choices):
        q = question.lower()
        if "in play for this decision" in q:
            return "relevant"
        if "serve or violate" in q:                       # per-principle question
            return "violates" if (not coherent and keyword in q) else "serves"
        if "either true" in q:                            # two-sided Non-Selective gate
            return "no" if coherent else "yes"
        if "alternative" in q:                            # humility — never perfect
            return "partial" if coherent else "no"
        return "yes"                                      # the mode questions
    return MockJudge(script=script)


def _metric(coherent: bool, keyword: str) -> CoherenceMetric:
    j = _judge(coherent, keyword)
    return CoherenceMetric(constitution=CONSTITUTION, threshold=0.85,
                           judge=j, samples=1 if j else 5)


@pytest.mark.parametrize("name, artifact, keyword", COHERENT,
                         ids=[s[0] for s in COHERENT])
def test_coherent_decisions_pass(name, artifact, keyword):
    res = _metric(True, keyword).assert_coheres(artifact, task="borrower-facing decision")
    # graded score: coherent and above threshold, but not saturated at a perfect 1.0
    assert 0.85 <= res.score < 1.0


@pytest.mark.parametrize("name, artifact, keyword", INCOHERENT,
                         ids=[s[0] for s in INCOHERENT])
def test_incoherent_decisions_are_caught(name, artifact, keyword):
    with pytest.raises(AssertionError) as exc:
        _metric(False, keyword).assert_coheres(artifact, task="borrower-facing decision")
    assert "non-canon" in str(exc.value)
