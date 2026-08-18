# Canon

A pytest-native **coherence-testing** framework: gate your deploys on whether your
multi-agent system still holds its **mission and principles**. If coherence
changed, something changed.

```python
from canon import CoherenceMetric, constitution_from_file

CONSTITUTION = constitution_from_file("constitution.yaml")


def test_holds_direction():
    artifact = my_agent.run("…")
    CoherenceMetric(constitution=CONSTITUTION, threshold=0.85).assert_coheres(artifact)
```

See `examples/` for a runnable demo. Install: `pip install canon-testing`.

## Concepts

A few terms used throughout Canon and in its failure messages:

- **Constitution** — the mission and principles you are testing against.
- **Rubric** — the questions a judge answers about an artifact. Canon ships one
  fixed rubric built on [Coherence Dynamics Theory
  (CDT)](https://adaptable-discipline.com/cdt), and uses it for every run — see
  [The rubric](#the-rubric).
- **Facet** — one rubric question's dimension of coherence (does it advance the
  mission, does it surface the tradeoff, and so on). A facet with no occasion to
  apply is answered `n/a` and excluded from the average rather than penalized.
- **Non-Selective principle** — coherence is not selective: an artifact that
  contradicts any part of the mission or principles fails outright, however well
  it scores elsewhere. Canon enforces it with the rubric's gate question, which
  hard-caps the score instead of letting a contradiction average out. Failure
  messages call this the **NSCP gate**.
- **Non-canon** — the verdict for an artifact that fails, either by tripping the
  Non-Selective gate or by scoring below your threshold. `assert_coheres` raises
  `AssertionError("non-canon …")` naming what failed.

## Configuring Canon

Canon looks for a `canon.yaml` (or a `[tool.canon]` table in `pyproject.toml`)
by walking up from your current directory. The first one it finds wins —
that's your project's config, however deep you're running the CLI from.

```yaml
# canon.yaml
judge_model: openai:gpt-5.6-luna
threshold: 0.85
constitution_path: constitution.yaml
baselines_dir: canon/baselines
tolerance: 0.02
```

Pass `--config path/to/canon.yaml` to any command to use a specific config
file instead of the walk-up search — useful for CI jobs or when you want to
point at a config that isn't in the current project tree. A `.yml` file works
the same as `.yaml`, and `--config pyproject.toml` reads its `[tool.canon]`
table, exactly as discovery would.

`constitution_path` and `baselines_dir` are relative to the config that
declares them, not to wherever you run the command. So `canon check` from a
subdirectory finds the same constitution and writes to the same baselines
directory as it does from the project root. Absolute paths are used as-is.
The exception is `--config`: naming a config from elsewhere doesn't move your
project, so its relative paths stay relative to the current directory. One
thing that does follow the named config is `.env` loading — key files are read
from the config file's own directory, so keep provider keys next to the config
they belong to (a missing key otherwise only surfaces later, when the judge is
actually called).

### `tolerance`

`tolerance` is how much a mean coherence score can drop between an accepted
baseline and a new run before `canon check` calls it a regression instead of
noise. Real judges aren't perfectly repeatable, so a tiny wobble — say
0.901 to 0.899 — shouldn't fail a check. The default is `0.02`: a drop has to
clear both the tolerance *and* the run-to-run statistical noise (a
significance test) before it's flagged. Raise it if your judge is noisier
than that; lower it if you want tighter drift detection.

### Secrets and `.env` files

Canon loads your provider API keys (and other environment variables) from
`.env` files next to your config, in this order — each later file overrides
the same key from an earlier one, but a real exported shell variable always
wins over any file:

1. `.env`
2. `.env.{APP_ENV}` (only if the `APP_ENV` environment variable is set — e.g.
   `.env.production`)
3. `.env.local`

This lets you keep shared defaults in `.env`, per-environment overrides in
`.env.production` / `.env.staging`, and your own untracked local key in
`.env.local`.

## The judge

Canon needs an LLM to answer the rubric's questions. You point it at one with
a `provider:model` string — either directly:

```python
CoherenceMetric(constitution=CONSTITUTION, judge="openai:gpt-5.6-luna")
```

or via `canon set-judge --provider openai --model gpt-5.6-luna`, which writes
`judge_model` into your project's `canon.yaml` (the one discovery would find —
not wherever you happen to be running the command from).

If your project is configured through `[tool.canon]` in `pyproject.toml`,
`set-judge` stops and tells you to set `judge_model` there instead. It only
manages `canon.yaml`, and writing one next to your `pyproject.toml` would take
over as the project config — reverting your threshold, tolerance and paths to
their defaults without saying so.

The model string can also carry a slash if the provider needs one, e.g.
`together:deepseek-ai/DeepSeek-V4-Flash-0731` — both `provider:model` and
`provider:org/model` are recognized and normalized the same way under the
hood (LiteLLM's own `provider/model` form).

Canon sends no `temperature` with a judge call. Reasoning models refuse an
explicit one outright, and what steadies a verdict here is the N-sample
majority vote, not a pinned temperature. If your model wants a specific one,
say so: `LiteLLMJudge(model=..., temperature=0.0)` passes it through.

## Baselines

`canon accept` records the current run as your baseline; `canon check` scores
a new run and compares it. The baseline file is JSON and includes, per
artifact: the overall score, whether it was gated, and (since the per-question
baseline format) a per-question breakdown — each rubric question's normalized
score and confidence. The per-question breakdown is what lets `canon check`
catch a single facet flipping solidly from "satisfied" to "violated," not just
the overall mean drifting.

### How a run is matched to its baseline

Suites get edited: artifacts are added, removed and reordered. So a baseline
records each artifact's *key* — a short digest of the artifact's own text —
and `canon check` compares an artifact against the record under its own key,
never against whatever sits at the same position in the list. An artifact with
no matching key is new and has nothing to regress from; a recorded artifact
missing from the run is reported as a note, not a failure.

Per-question rows are matched the same way. A rubric question derived from your
constitution also stores its *subject* — the principle text it was generated
from. Question ids are positional (`P1` is "the first principle"), so if you
reorder or delete a principle, `P1` now means something else: Canon notices the
subject changed, skips that comparison, and prints a note asking you to
re-accept rather than reporting a regression in a facet it never measured.

Older baselines recorded before per-question tracking still load and still
gate on the overall score — they just skip that facet-level check (Layer 1)
and fall back to the mean-drift comparison (Layer 2), with a note printed so
you know to re-accept.

## The rubric

Canon ships one rubric, built on CDT, and always uses it. You can run the tests
however you like; the questions they answer stay the same. That is deliberate.

A coherence score is only worth something if it means the same thing everywhere.
A project that picks its own questions is measuring something else, and its
number can't be set beside anyone else's — or beside its own from six months
ago. Worse, a rubric you can edit is a rubric that gets edited when a run comes
back badly: the yardstick bends toward the thing it is measuring, and the score
stops being evidence. Keeping it fixed and versioned keeps the measurement
honest and the bias out.

What *is* yours is your constitution. Canon derives a question per principle
from it, so every organization answers the same fixed facets — about its own
mission and its own principles. That is where specificity belongs.

So passing `rubric=` to `CoherenceMetric`, or putting a `rubric` key in
`canon.yaml`, raises with that explanation rather than being quietly accepted
or quietly ignored. When the packaged rubric's own version changes, `canon
check` refuses to compare against a baseline recorded under the old one and
asks you to re-accept, for the same reason: two rubrics, two yardsticks.

## Which direction is it serving?

A coherence score says how well an artifact reasons against one constitution.
It cannot say *which* constitution the artifact was reasoning against. That
matters when direction changes mid-run: an agent still holding the old copy
reasons beautifully against it and scores as high as an agent that moved,
because the heaviest rubric question — does this advance the mission — passes
whenever the mission itself didn't change.

`serves_direction` is the separate reading for that:

```python
from canon import serves_direction

verdict = serves_direction(artifact, current=NEW, superseded=OLD)
assert verdict.serves == "current", verdict.evidence
```

The judge makes a forced choice — `current`, `superseded`, `both` or
`neither` — over the same N-sample majority vote the rubric's questions use.
The `DirectionVerdict` you get back carries that choice, the whole vote
(`votes`, every choice including the ones nobody picked), its lopsidedness
(`confidence`, the same measure as a rubric question's), and the evidence the
winning sample cited. A split vote still returns a choice, so read
`confidence` before acting on one.

Direction is reported *beside* a coherence score and never folded into it: it
is not a rubric question, it moves no number, and `CoherenceMetric` takes no
second constitution. Two directions that state the same mission and principles
raise `ValueError` rather than returning a verdict that distinguished nothing.

## Calling the pieces directly

A few behaviours worth knowing if you use Canon's functions rather than its CLI:

- **`gate(results, baseline, threshold, tolerance=0.02)`** raises `ValueError`
  on an empty `results` list rather than returning a pass. A gate that succeeds
  because nothing was checked is the one answer it must never give. The
  `tolerance` argument is the same stability band described above, per call —
  the CLI passes your configured value through.
- **`relevant_questions(...)`** returns just the kept questions by default. With
  `report_excluded=True` it returns a `(kept, excluded)` tuple instead, where
  `excluded` is a tuple of the principles the judge ruled out of play — the
  shape changes, so unpack accordingly.
- **`wilson_ci(0, 0)`** returns `(0.0, 1.0)`. With no observations the honest
  interval is the whole range: you know nothing about the rate, rather than
  knowing it is zero.

## Testing without a judge

`MockJudge` answers rubric questions from a script instead of calling a real
model — handy for unit tests. Give it a dict mapping a substring of the
question to a choice, or a callable:

```python
from canon.judge.mock import MockJudge

judge = MockJudge(script={"serve the mission": "yes", "__default__": "no"})
```

An unmatched question raises rather than silently picking the first choice —
that used to let a mis-scripted test go green while quietly skipping whole
facets. If you genuinely want a catch-all answer for anything else, say so
explicitly with a `"__default__"` entry, as above.
