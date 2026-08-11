# Real-world coherence validation

Does Canon actually catch incoherence it *wasn't told about*? This directory
feeds six **documented** cases to Canon's **real** LLM judge — each pairing an
organization's own **stated values** with a **documented decision or practice** —
and asks whether the behavior coheres with those stated values. Four are real
value-violations; two are controls where the company acted *on* its values.

Run it (real judge, not part of CI — needs a provider key):

```bash
export TOGETHER_AI_API_KEY=...          # or OPENAI_API_KEY, etc.
python examples/real_world/validate.py
```

## Result

Judge: `together_ai/deepseek-ai/DeepSeek-V4-Flash-0731` (an inexpensive model —
these are clear cases), threshold `0.85`, 3 samples/question:

| Case | Expected | Score | Gated | Marked correctly |
|---|---|---|---|---|
| Wells Fargo — fake accounts | incoherent | **0.00** | ✅ | ✅ |
| Enron — accounting fraud | incoherent | **0.00** | ✅ | ✅ |
| Volkswagen — Dieselgate | incoherent | **0.00** | ✅ | ✅ |
| Purdue Pharma — OxyContin | incoherent | **0.00** | ✅ | ✅ |
| Patagonia — "Don't Buy This Jacket" | coherent | **0.90** | — | ✅ |
| CVS Health — dropping tobacco | coherent | **0.90** | — | ✅ |

**6/6.** Every documented value-violation is marked non-canon (gated, with a
localized reason like *"the defeat-device software increased on-road NOx"*), and
both controls — companies that acted *on* their stated values — pass. The judge
was never told the answer; it scored each behavior against that company's own
constitution. (LLM outputs vary run to run; exact scores may shift slightly, but
the violations gate and the controls pass.)

## Sources

- **Wells Fargo** — 2012 Vision & Values ("start with what the customer needs, not with what we want to sell them") vs. ~1.5M unauthorized accounts. [DOJ, $3B settlement](https://www.justice.gov/archives/opa/pr/wells-fargo-agrees-pay-3-billion-resolve-criminal-and-civil-investigations-sales-practices)
- **Enron** — 2000 code of ethics (integrity, communication, respect) vs. accounting fraud. [Forbes](https://www.forbes.com/sites/kensilverstein/2013/05/14/enron-ethics-and-todays-corporate-values/)
- **Volkswagen** — "clean diesel" marketing vs. defeat devices (~40× NOx). [FTC](https://www.ftc.gov/news-events/news/press-releases/2016/03/ftc-charges-volkswagen-deceived-consumers-its-clean-diesel-campaign)
- **Purdue Pharma** — "relieve pain responsibly" vs. downplaying OxyContin addiction. [Van Zee, *Am J Public Health*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2622774/)
- **Patagonia (control)** — "in business to save our home planet" + "Don't Buy This Jacket". [Patagonia](https://www.patagonia.com/stories/planet/activism/dont-buy-this-jacket-black-friday-and-the-new-york-times/story-18615.html)
- **CVS Health (control)** — "helping people on their path to better health" + ending ~$2B/yr tobacco sales. [CVS Health](https://www.cvshealth.com/news/community/cvs-health-research-institute-study-confirms-companys-tobacco-re.html)

## Note

Coherence is judged **relative to each organization's own stated values** — Canon
asks "did they act consistently with what *they* said they stand for," not against
an external morality. An earlier version of the rubric scored Patagonia 0.83 (a false near-miss)
because a short ad doesn't exhibit every facet; the **N/A** rule — a facet with no
occasion to apply is excluded from the average rather than penalized — fixed that,
and it now scores 0.90.
