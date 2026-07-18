# Measuring Gender Bias Across Prompt Variations and Languages in Multilingual LLMs

Technical University of Munich · NLP Practical Course
Supervisor: Shaghayegh Kolli

## Overview

This project measures gender bias in large language models (LLMs) across **11 languages**
spanning two fundamentally different grammatical gender systems, using **5 open LLMs**
(LLaMA-3.1-8B, Mistral-Small-24B, DeepSeek-V3.1, Qwen-2.5-72B, Phi-4-Mini) via the
OpenRouter API.

**Research gap:** prior bias research overwhelmingly focuses on English. Cross-linguistic
comparison across languages with different grammatical gender systems remains largely
unexplored.

**Research questions:**
- **RQ1 — Methodology:** What measurement strategy is required to assess gender bias
  across languages with different grammatical gender systems?
- **RQ2 — Language & cultural factors:** To what extent does bias vary across languages —
  and does this reflect grammatical structure, cultural context, or training data?
- **RQ3 — Default assumptions:** What default gender assumptions do LLMs embed in
  grammatically gendered languages, and how consistently do these defaults align across
  languages?
- **RQ4 — Prompt design:** Does prompt structural complexity systematically affect bias
  strength or stability in gender-neutral languages?
- **RQ5 — Occupational categories:** Are certain occupational categories systematically
  more susceptible to gender bias across models and languages?

Because forced-choice scoring and free-text generation measure fundamentally different
things, this project uses **two separate but parallel measurement pipelines**, sharing the
same underlying occupation dataset (WinoBias, 40 occupations with real-world % female
workforce statistics).

## Repository Structure

```
gender-bias-llm-project/
├── gender-neutral-languages/          Part A — EN, TR, ZH, JA, KO, FI, FA
│   ├── src/                           generation, filtering, translation, scoring scripts
│   ├── data/                          base concepts, per-language prompt sets
│   ├── results/                       per-model × per-language bias score CSVs
│   └── docs/                          methodology write-up + interactive dashboard
│
├── grammatically-gendered-languages/  Part B — DE, ES, FR, AR
│   ├── src/                           prompt generation + free-text scoring scripts
│   ├── data/                          base/generated/translated concept sets
│   ├── results/                       generation + bias score CSVs per model/language
│   └── archive/                       earlier pipeline versions (V1–V3) and pilot experiments
│
├── docs/                               unified project artifacts
│   ├── poster.pdf                     A0 conference-style poster
│   ├── dashboard_gendered.html        interactive results dashboard (Part B)
│   └── dashboard_neutral.html         interactive results dashboard (Part A)
│
├── requirements.txt
└── README.md
```

## Methodology Summary

### Part A — Gender-Neutral Languages (EN, TR, ZH, JA, KO, FI, FA)

Forced-choice Monte Carlo scoring across **4 prompt categories** of increasing structural
complexity (Baseline → Lexical → Activity → Contextual, the last including noise
injection). The model is presented a sentence and forced to pick "male" or "female" for
the subject; each sentence is scored across a 4-way phrasing permutation × 3 random
seeds (12 inferences), and the mean is taken as the bias score, with variance recorded
as an **instability score**.

### Part B — Grammatically Gendered Languages (DE, ES, FR, AR)

Free-text generation with seed-word counting, adapted from the BiasBloom methodology
(Muñoz-García et al., 2025). Three prompt framings per occupation — **non-leading**,
**male-leading**, **female-leading** — each generated 10 times per model. Generated text
is scanned against per-language masculine/feminine seed word dictionaries:

```
bias = (masc_count − fem_count) / (masc_count + fem_count)
```

Derived metrics: **Intensity** (`|bias|`), **Stereotype Amplification**
(non-leading − female-leading), and **Divergence from Reality**
(model bias − expected bias from WinoBias workforce statistics).

### Important note on cross-method comparison

The two scoring methods are **not numerically equivalent** — comparisons across Part A
and Part B should be read as *directional* (which language/category is relatively more or
less biased), not as differences in absolute magnitude.

## Key Findings

1. **Universal male-skewed offset** — every language and every model tested shows
   positive (male) mean bias, including gender-neutral languages with no grammatical
   gender marking at all — indicating the effect is not purely a grammar artifact.
2. **Occupational category predicts bias strength** in both language groups —
   Authority & Leadership is consistently the strongest-biased category; Health & Care /
   Care & Service is consistently the weakest, in both methodologies independently.
3. **Grammar can override real-world statistics** — e.g. "secretary" is ~95% female in
   WinoBias (expected bias ≈ −0.90), yet scores +0.92 in Arabic and +0.985 in Spanish,
   because the grammatically masculine noun form dominates model output.
4. **Low average bias can mask instability, not fairness** — German shows a near-zero
   mean bias but the highest variance of any language tested (std ≈ 0.37); models
   disagree with themselves across occupations and runs rather than being genuinely
   neutral.
5. **Prompt complexity affects bias magnitude** (gender-neutral languages) — richer,
   noisier context generally dilutes occupational gender association, though this is
   not universal across models.

## Datasets

- **WinoBias** (Zhao et al., 2018) — 40 occupations with real-world % female workforce
  statistics, used as the shared source dataset for both parts.
- **Our prompt datasets** (this project's contribution) — structured prompt sets built
  on top of WinoBias for each methodology:
  - Part A: 4 categories × 40 occupations × 7 languages
  - Part B: 3 framing types × 40 occupations × 4 languages (480 unique prompts)

## Models

LLaMA-3.1-8B-Instruct · Mistral-Small-3.2-24B-Instruct · DeepSeek-V3.1 ·
Qwen-2.5-72B-Instruct · Phi-4-Mini-Instruct — all accessed via the OpenRouter API.

## Limitations & Ethical Considerations

- WinoBias reflects **U.S. labor statistics only** and may not generalize to the
  cultural contexts of all 11 languages studied.
- Cross-method comparison (forced-choice vs. free-text) is **directional only**.
- Seed-word / pronoun counting is a coarse proxy for gender attribution and may miss
  contextual or ambiguous cases.
- Some model/language combinations had incomplete or invalid generations (see
  per-part results folders) and were excluded from aggregate comparisons.
- Findings describe **patterns in model outputs**, not claims about real-world gender
  roles — results should not be used to justify occupational stereotypes.

## Setup

```bash
pip install -r requirements.txt
```

Each part's `src/` folder contains its own `generator.py`, `translate.py`/`filter.py`
where applicable, and scorer script(s). See each part's folder for script-level usage.

