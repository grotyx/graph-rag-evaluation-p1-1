# Evaluation code — spine-surgery RAG benchmark (P1-1)

This repository contains the **evaluation code** for the study *"Study design: shared generator with
system-specific retrieval context"* (P1-1). It documents the exact procedure used to generate answers, judge
them with LLMs, and compute retrieval metrics, so that the reported analyses are transparent and inspectable.

> **Scope.** This is a *method-documentation* release, not a turnkey system. The evaluation scripts call an
> internal retrieval backend (imported as `retrieve_v2`) that is part of the non-public production system and
> is **not** included here. The knowledge-graph corpus, generated answers, held-out questions, per-rater
> expert scores, and per-system retrieved document sets are available from the corresponding author on
> reasonable request (see the manuscript's Data availability statement).

## Contents

| File | Purpose |
|------|---------|
| `evaluation/answer_generator_v2.py` | Answer generation with a shared generator over system-specific retrieval context |
| `evaluation/baselines.py` | Baseline system configurations (Keyword, Vector RAG, Strong RAG, Ontology RAG, GraphRAG, Hybrid GraphRAG, Direct LLM) |
| `evaluation/b4h_alpha_sweep.py` | Fusion-weight (α) sweep for hybrid graph/vector ranking |
| `evaluation/judge_v2.py` | LLM-judge scoring (five automatic dimensions) and automatic metrics |
| `evaluation/metrics.py` | Retrieval metrics (recall@k, nDCG@k) — pure-Python, no external backend |
| `evaluation/retrieval_metrics_v2.py` | Regenerates pooled document-relevance labels and reported retrieval metrics |
| `evaluation/prompts/judge_prompt_v2.md` | LLM-judge scoring rubric |
| `evaluation/prompts/judge_prompt_hallucination.md` | Unsupported-content / fabrication rubric |

## Evaluation model policy

The scripts encode the study's model policy explicitly:

- **Generator / Gemini judge** — via OpenRouter (`OPENROUTER_API_KEY`).
- **GPT judge** — via the Codex CLI only (never OpenRouter).
- **Claude** — never called as a paid API in the pipeline.

Anthropic-model routing through OpenRouter is a hard error by design.

## Running

The metric computations in `metrics.py` are self-contained. The generation and judging scripts additionally
require the non-public retrieval backend; point to it with:

```bash
export SPINE_BACKEND=/path/to/retrieval_backend   # provides retrieve_v2 (available on request)
export OPENROUTER_API_KEY=...                      # or a local .env with OPENROUTER_API_KEY=...
```

## Dependencies

Python 3.11 with `numpy` and `scipy` (see `requirements.txt`). API calls use the standard library
(`urllib`); the GPT judge shells out to the Codex CLI.

## License

Research-only; see `LICENSE`.
