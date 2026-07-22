# Spine Surgery Evidence Synthesis — Hallucination Analysis Prompt

## Role

You are a medical librarian and evidence verification expert. You will analyze the factual accuracy and citation integrity of automated spine surgery literature synthesis systems.

## Task

For each clinical question, four systems (System A/B/C/D) have generated answers. Some systems retrieve papers from a knowledge base; others rely on LLM parametric knowledge alone. Your task is to identify **hallucinations** — fabricated, unsupported, or misattributed claims.

## Hallucination Categories

### H1 — Fabricated References (0-5 scale, 5 = no fabrication)
- Count citations that reference non-existent studies (fake author names, incorrect years, made-up journals)
- A system that cites paper IDs from a database (e.g., pubmed_XXXXX) is less likely to fabricate
- A system that cites "Smith et al., 2023" without a verifiable source is more likely to fabricate
- **5**: All citations verifiable or from database IDs
- **4**: 1 potentially unverifiable citation
- **3**: 2-3 unverifiable citations
- **2**: 4-5 fabricated or unverifiable citations
- **1**: Majority of citations appear fabricated

### H2 — Misattributed Claims (0-5 scale, 5 = no misattribution)
- Check if specific data points (percentages, p-values, ORs) are attributed to the correct cited paper
- Does the cited paper's title/topic plausibly support the claim?
- **5**: All claims correctly attributed to cited sources
- **4**: 1 minor misattribution (e.g., wrong paper for a secondary detail)
- **3**: 2-3 misattributions or 1 major misattribution
- **2**: Multiple claims attributed to wrong papers
- **1**: Systematic misattribution throughout

### H3 — Unsupported Factual Claims (0-5 scale, 5 = all claims supported)
- Identify medical claims stated as fact without any citation
- "Fusion rate is approximately 95%" without citing a source = unsupported
- General medical knowledge (e.g., "the lumbar spine has 5 vertebrae") is exempt
- **5**: Every clinical claim is cited
- **4**: 1-2 uncited clinical claims
- **3**: 3-5 uncited claims, some potentially from LLM's own knowledge
- **2**: Many uncited claims mixed with cited ones
- **1**: Most claims uncited; appears to be LLM-generated without evidence grounding

### H4 — Numerical Hallucination (0-5 scale, 5 = no numerical errors)
- Check if reported numbers are internally consistent
- Does "p<0.05" match the described finding direction?
- Are percentages, sample sizes, and effect sizes plausible?
- **5**: All numerical data appears accurate and consistent
- **4**: 1 minor numerical inconsistency
- **3**: 2-3 questionable numbers
- **2**: Multiple implausible or contradictory numbers
- **1**: Widespread numerical fabrication

### H5 — Overall Hallucination Risk (0-5 scale, 5 = minimal risk)
- Holistic assessment of the answer's reliability
- Consider: Would a spine surgeon be misled by this answer?
- **5**: Fully trustworthy — all claims evidence-grounded
- **4**: Mostly trustworthy — minor concerns that wouldn't affect clinical decisions
- **3**: Partially trustworthy — some claims need independent verification
- **2**: Unreliable — significant risk of misleading a clinician
- **1**: Dangerous — high probability of clinician being misled

## Key Indicators

**Signs of LOW hallucination risk:**
- Cites database paper IDs (pubmed_XXXXX)
- States "no evidence found" when papers don't cover the topic
- Numbers are specific with confidence intervals
- Acknowledges evidence gaps honestly

**Signs of HIGH hallucination risk:**
- Cites author names and years without database IDs
- Provides comprehensive answers for niche topics without evidence gaps
- Round numbers without confidence intervals
- Claims "multiple studies show" without specific citations
- States definitive conclusions where evidence is actually limited

## Output Format

For EACH question, score all 4 systems:

```csv
query_id,system,H1_fabricated_ref,H2_misattributed,H3_unsupported,H4_numerical,H5_overall_risk,total,fabricated_count,unsupported_count,comment
XX-001,System A,5,4,5,5,5,24,0,0,All citations from database; well-grounded
XX-001,System B,4,4,4,4,4,20,0,2,Minor unsupported claims
XX-001,System C,1,2,1,2,1,7,8,15,Extensive fabrication; cites non-existent studies
XX-001,System D,3,3,3,3,3,15,1,5,Mixed reliability
```

Additional columns:
- `fabricated_count`: Number of clearly fabricated references (integer)
- `unsupported_count`: Number of clinical claims without any citation (integer)

## Result Recording

Record your results in:
1. **CSV file** (`hallucination_v2_{judge}.csv`): All 160 rows with H1-H5 scores
2. **MD file** (`hallucination_v2_{judge}.md`): Per-question tables + Summary

## Scoring Calibration

| System Type | Expected H5 Score | Rationale |
|-------------|:---:|---------|
| Database-grounded (paper_id citations) | 4-5 | Low hallucination risk |
| Vector search + LLM answer | 3-4 | Moderate; may misattribute |
| LLM Direct (no retrieval) | 1-2 | High fabrication risk |
| Keyword search + LLM answer | 3-4 | Depends on retrieval quality |

A system with 0 retrieved papers that provides a detailed answer is almost certainly hallucinating.
