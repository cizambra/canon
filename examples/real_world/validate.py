"""Real-world coherence validation — how Canon marks documented cases.

Each case pairs an organization's OWN stated values (its constitution) with a
documented decision or practice, and asks Canon's REAL judge whether the behavior
coheres with those stated values. Four are documented value-violations (should be
non-canon); two are documented value-affirming acts (controls; should be canon).

Unlike the keyless pytest demo, this uses a real LLM judge (via LiteLLM) and is
NOT part of CI — it needs a provider key and makes real calls. Run it yourself:

    export TOGETHER_AI_API_KEY=...        # or OPENAI_API_KEY, etc.
    python examples/real_world/validate.py

Sources for each case are listed in this directory's README.md.
"""

import os

from canon import CoherenceMetric
from canon.judge.litellm_judge import LiteLLMJudge
from canon.models import Constitution

# An inexpensive model is enough — these are clear, documented cases. Override
# with CANON_VALIDATE_MODEL (any LiteLLM model string).
MODEL = os.environ.get("CANON_VALIDATE_MODEL", "together_ai/deepseek-ai/DeepSeek-V4-Flash-0731")
THRESHOLD = 0.85
SAMPLES = 3

CASES = [
    (
        "Wells Fargo — fake accounts",
        "incoherent",
        "Satisfy our customers' financial needs and help them succeed financially.",
        [
            "Start with what the customer needs, not with what we want to sell them.",
            "Do what is right for the customer.",
            "Honesty, trust, and integrity in every interaction.",
        ],
        "To hit aggressive sales quotas the bank ran a volume-based sales model; "
        "employees opened roughly 1.5 million bank accounts and 500,000 credit-card "
        "accounts for customers without their consent or knowledge.",
    ),
    (
        "Enron — accounting fraud",
        "incoherent",
        "Create long-term value for shareholders while acting with integrity.",
        [
            "Integrity: work with customers and prospects openly, honestly, and sincerely.",
            "Communication: an obligation to communicate honestly.",
            "Respect: treat others as we would like to be treated.",
        ],
        "Executives used off-balance-sheet entities to hide billions in debt and "
        "inflate reported earnings, deceiving investors and employees about the "
        "company's true financial condition.",
    ),
    (
        "Volkswagen — Dieselgate",
        "incoherent",
        "Lead in sustainable, low-emission mobility and offer environmentally friendly vehicles.",
        [
            "Environmental responsibility — minimize emissions and harm.",
            "Honesty with customers and regulators.",
            "Comply with the law.",
        ],
        "The company installed defeat-device software that turned emission controls "
        "on only during regulatory tests; on the road the cars emitted up to 40 "
        "times the legal NOx limit, while being marketed as 'clean diesel'.",
    ),
    (
        "Purdue Pharma — OxyContin",
        "incoherent",
        "Improve patients' lives by relieving pain responsibly.",
        [
            "Patient well-being and safety come first.",
            "Honest, accurate communication about our products.",
            "Do no harm.",
        ],
        "The company aggressively marketed OxyContin as a non-addictive, long-lasting "
        "painkiller and told doctors fewer than 1% of users become addicted, while "
        "internal information already indicated widespread abuse and addiction.",
    ),
    (
        "Patagonia — Don't Buy This Jacket",
        "coherent",
        "We're in business to save our home planet.",
        [
            "Reduce unnecessary consumption and environmental harm.",
            "Integrity and authenticity over short-term sales.",
            "Quality and durability over volume.",
        ],
        "On Black Friday the company ran a full-page ad headlined 'Don't Buy This "
        "Jacket', urging customers not to buy what they don't need and detailing the "
        "environmental cost of its best-selling jacket.",
    ),
    (
        "CVS Health — dropping tobacco",
        "coherent",
        "Helping people on their path to better health.",
        [
            "Prioritize customer health over sales.",
            "What we sell must be consistent with our purpose.",
            "Do right by customers even at a cost.",
        ],
        "The company stopped selling all tobacco products across its ~7,700 stores, "
        "forgoing about $2 billion in annual revenue, because tobacco sales were "
        "inconsistent with its purpose of helping people get healthy.",
    ),
]


def main() -> None:
    judge = LiteLLMJudge(model=MODEL)
    print(f"judge: {MODEL}   threshold: {THRESHOLD}   samples: {SAMPLES}\n")
    print(f"{'CASE':34}{'EXPECT':12}{'SCORE':>7} {'GATED':>6}  OK   REASON")
    correct = 0
    for name, expected, mission, principles, artifact in CASES:
        con = Constitution(mission=mission, principles=tuple(principles))
        res = CoherenceMetric(
            constitution=con, threshold=THRESHOLD, judge=judge, samples=SAMPLES
        ).score(
            artifact,
            task=(
                "judge whether this documented behavior coheres with the "
                "organization's own stated values"
            ),
        )
        non_canon = res.gated or res.score < THRESHOLD
        ok = (non_canon and expected == "incoherent") or (
            (not non_canon) and expected == "coherent"
        )
        correct += ok
        reason = res.reasons[0][:60] if res.reasons else "(coherent — no failing facet)"
        print(
            f"{name:34}{expected:12}{res.score:>7.2f} {str(res.gated):>6}  "
            f"{'YES' if ok else 'NO':>3}  {reason}"
        )
    print(f"\nMarked correctly: {correct}/{len(CASES)}")


if __name__ == "__main__":
    main()
