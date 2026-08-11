# Canon example — lending coherence

A coherence test is a pytest test whose assertion is an LLM-judged coherence
score against your constitution, with a threshold. This demo shows the gate
working in **both** directions on a lending system.

`constitution.yaml` sets the mission ("help borrowers succeed, not just borrow")
and four principles (fair lending, borrower well-being, transparent decisions,
honest growth). `tests/coherence/test_lending.py` runs eight scenarios:

**Coherent decisions — must pass (`assert_coheres` succeeds):**

| scenario | why it holds |
|---|---|
| `denial_explains_affordability` | serves well-being + transparency |
| `approval_full_disclosure` | serves transparency + well-being |
| `underwriting_affordability_first` | serves well-being over throughput |
| `decline_predatory_partner` | serves honest growth over volume |

**Incoherent decisions — must be caught (non-canon), each breaking one principle:**

| scenario | principle it violates |
|---|---|
| `upsell_beyond_need` | honest, compliant growth |
| `proxy_discrimination` | fair, unbiased lending |
| `opaque_denial` | transparent decisions |
| `lend_into_harm_for_growth` | borrower well-being first |

## Run it (no API key)

The demo uses a **scenario-aware mock judge** when `CANON_MOCK_JUDGE` is set, so
it runs keyless in CI and demonstrates both a pass and a catch. Coherent
scenarios pass (score ≥ 0.85, never a saturated 1.0); incoherent ones trip the
Non-Selective gate and raise `AssertionError("non-canon …")` naming the
violated principle.

```bash
CANON_MOCK_JUDGE=yes pytest examples/tests/coherence/test_lending.py -q
```

Run the deploy gate over a suite (records, then checks, an accepted baseline):

```bash
cd examples
CANON_MOCK_JUDGE=yes canon accept --suite suite.yaml
CANON_MOCK_JUDGE=yes canon check  --suite suite.yaml   # PASS + pass-rate CI
```

## Real judging

Drop the mock and judge with a real model — the *same* tests then exercise real
coherence judgment (and a genuinely incoherent decision is caught on its
merits, not because a script said so):

```bash
canon set-judge --provider openai --model gpt-5.6-luna
export OPENAI_API_KEY=...
pytest examples/tests/coherence/test_lending.py -q
canon check --suite suite.yaml
```
