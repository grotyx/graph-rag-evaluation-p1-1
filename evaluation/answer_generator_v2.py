"""v2 eval answer generation — B1 / B2 / B_sRAG / B4 retrieval + one shared generator.

npj re-test stack. Runs entirely on the v2 reified-Evidence graph (bolt 7688) via retrieve_v2.
INVARIANT (methods requirement): all systems use the SAME generation model + prompt; they differ
ONLY in the retrieved evidence they are given. Generation backbone = deepseek-v4-pro (OpenRouter).

POLICY: NO Anthropic API anywhere (no generation, no HyDE, no rerank). OpenAI embeddings + deepseek
(OpenRouter) only — the same allowances as the ingest path.

Systems:
  B1  keyword    — Neo4j fulltext over Chunk.text (PubMed-like keyword search)
  B2  vector     — chunk vector top-k (graph OFF)
  B_sRAG strong  — vector + HyDE + deepseek rerank + MMR diversity (graph OFF) ← the real comparator
  B4  graphrag   — retrieve_v2: resolve → reified Evidence (quote+paper_id+evidence_level) → ranked; semantic fallback
  B3  llm_direct — no retrieval

Usage:
  python evaluation/answer_generator_v2.py --smoke "does TLIF improve VAS compared to PLIF?"
"""
import os, sys, json, re, time, math, urllib.request, urllib.error

REPO = "."
PROTO = f"{REPO}/docs/redesign_v2/prototypes"
sys.path.insert(0, PROTO)
import retrieve_v2 as R          # v2 retrieval (bolt 7688) + _run + _qvec + parse_question + q1/q2/prognostic

# ── deepseek-v4-pro generator via OpenRouter (no Anthropic fallback) ──────────────────────────────
GEN_MODEL = os.environ.get("V2_EVAL_GEN_MODEL", "deepseek/deepseek-v4-pro")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

def _or_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for line in open(f"{REPO}/.env"):
        if line.startswith("OPENROUTER_API_KEY="):
            k = line.split("=", 1)[1].strip().strip('"')
            os.environ["OPENROUTER_API_KEY"] = k
            return k
    raise RuntimeError("OPENROUTER_API_KEY not found (.env)")

def deepseek(system, user, max_tokens=2200, temperature=0.0, model=None, retries=4):
    """One OpenRouter chat call. deepseek-v4-pro by default. NEVER routes to Anthropic."""
    model = model or GEN_MODEL
    if "anthropic" in model or "claude" in model:
        raise RuntimeError(f"policy: Anthropic model '{model}' forbidden in eval path")
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    hdr = {"Authorization": f"Bearer {_or_key()}", "Content-Type": "application/json",
           "X-Title": "Spine GraphRAG v2 eval"}
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(OR_URL, data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            msg = (d.get("choices") or [{}])[0].get("message", {})
            content = (msg.get("content") or "").strip()   # NEVER fall back to msg['reasoning'] — that is the
            if content:                                    # chain-of-thought, not the answer (would leak as output)
                return content
            last = f"empty content (finish={((d.get('choices') or [{}])[0]).get('finish_reason')})"
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"OpenRouter failed after {retries}: {last}")

# ── shared answer prompt (identical for B1/B2/B_sRAG/B4; B3 is context-free) ──────────────────────
GEN_SYSTEM = (
    "You are a spine-surgery evidence assistant. Answer the clinical question USING ONLY the numbered "
    "evidence provided. After each factual claim, cite the supporting study by its bracketed identifier "
    "exactly as shown in the evidence (for example [pubmed_26654337]); never write the literal word "
    "'paper_id'. If the evidence is insufficient or absent, say so explicitly — never invent studies, "
    "numbers, or citations. Output ONLY the final answer prose — do not show your reasoning or restate the "
    "question. Be concise and clinically precise.")
B3_SYSTEM = (
    "You are a spine-surgery clinical assistant. Answer the question from your own knowledge. Where you "
    "cite evidence, give the study identifier if you can. Be concise and clinically precise.")

_LEAK = re.compile(r"^\s*(we\s+(?:are|must|need|will|have)|let\s+(?:me|us)|the\s+question|okay|first,|"
                   r"i\s+(?:need|will|must|am)|the\s+user|analyz|to\s+answer\b|based\s+on\s+the\s+question)", re.I)

def _gen(question, context):
    user = f"Clinical question:\n{question}\n\nEvidence:\n{context}\n\nAnswer (cite [paper_id] per claim):"
    txt = deepseek(GEN_SYSTEM, user)
    if _LEAK.match(txt or ""):   # reasoning/prompt-echo leaked → retry once, forcing a direct opening
        txt = deepseek(GEN_SYSTEM + " CRITICAL: your first sentence MUST be the clinical answer itself — do "
                       "not write 'we are asked', 'let me', or any preamble/reasoning.",
                       user + "\n\n(Answer directly. No preamble.)")
    return txt

CITE_RE = re.compile(r"\[([a-zA-Z0-9_:\-]+)\]")
_CITE_NOISE = {"paper_id", "pmid", "id", "citation", "ref", "reference"}
def _citations(text):
    return sorted({c for c in CITE_RE.findall(text or "") if c.lower() not in _CITE_NOISE})

# ── retrieval helpers ─────────────────────────────────────────────────────────────────────────────
CY_VEC_FULL = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $qvec) YIELD node AS c, score
MATCH (c)-[:FROM_PAPER]->(p:Paper)
RETURN c.chunk_id AS chunk_id, c.text AS text, score AS score,
       p.paper_id AS paper_id, p.title AS title, p.evidence_level AS evidence_level, p.year AS year
ORDER BY score DESC
"""
def _vec(query, k):
    return R._run(CY_VEC_FULL, k=k, qvec=R._qvec(query))

_FT_READY = False
def _ensure_fulltext():
    global _FT_READY
    if _FT_READY:
        return
    R._run("CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
    _FT_READY = True

def _kw_query(question):
    # content terms only (drop scaffolding); Lucene OR
    toks = [w for w in re.findall(r"[A-Za-z0-9\-]{3,}", question.lower()) if w not in R._SCAFFOLD]
    return " OR ".join(dict.fromkeys(toks)) or question

CY_FT = """
CALL db.index.fulltext.queryNodes('chunk_fulltext', $q) YIELD node AS c, score
MATCH (c)-[:FROM_PAPER]->(p:Paper)
RETURN c.chunk_id AS chunk_id, c.text AS text, score AS score,
       p.paper_id AS paper_id, p.title AS title, p.evidence_level AS evidence_level, p.year AS year
ORDER BY score DESC LIMIT $k
"""

def _chunk_ctx(rows, max_chars=600):
    return "\n".join(
        f"[{r['paper_id']}] (level {r.get('evidence_level') or '?'}, {r.get('year') or '?'}): "
        f"{(r['text'] or '')[:max_chars]}" for r in rows)

# ── B1 keyword ──────────────────────────────────────────────────────────────────────────────────
def retrieve_b1(question, k=6):
    _ensure_fulltext()
    rows = R._run(CY_FT, q=_kw_query(question), k=k)
    return rows, _chunk_ctx(rows)

# ── B2 vector-only ────────────────────────────────────────────────────────────────────────────────
def retrieve_b2(question, k=6):
    rows = _vec(question, k)
    return rows, _chunk_ctx(rows)

# ── B_sRAG strong RAG (HyDE + rerank + MMR), graph OFF ────────────────────────────────────────────
HYDE_SYSTEM = "You are a spine surgeon. Write a short, factual paragraph that would answer the question. No citations."
def _hyde(question):
    try:
        return deepseek(HYDE_SYSTEM, question, max_tokens=220)
    except Exception:
        return question

def _rerank(question, rows):
    """deepseek relevance score 0-1 per chunk; robust to parse failure (keep vector order)."""
    if not rows:
        return rows
    items = "\n".join(f"{i}: {(r['text'] or '')[:300]}" for i, r in enumerate(rows))
    try:
        out = deepseek(
            "Score each numbered passage 0.0-1.0 for relevance to the question. Return ONLY JSON "
            '{"scores":{"0":0.x,...}}.',
            f"Question: {question}\n\nPassages:\n{items}", max_tokens=400)
        sc = json.loads(re.search(r"\{.*\}", out, re.S).group())["scores"]
        for i, r in enumerate(rows):
            r["_rr"] = float(sc.get(str(i), 0))
    except Exception:
        for i, r in enumerate(rows):
            r["_rr"] = 1.0 - i * 0.01
    return sorted(rows, key=lambda r: -r.get("_rr", 0))

def _mmr(rows, k, lam=0.7):
    """Greedy MMR on chunk embeddings for diversity. Falls back to rerank order if embeddings missing."""
    pool = [r for r in rows if r.get("_emb")]
    if not pool:
        return rows[:k]
    chosen = []
    cand = pool[:]
    def cos(a, b):
        s = sum(x * y for x, y in zip(a, b)); na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return s / (na * nb + 1e-9)
    while cand and len(chosen) < k:
        best, bi = None, None
        for i, r in enumerate(cand):
            rel = r.get("_rr", 0)
            div = max((cos(r["_emb"], c["_emb"]) for c in chosen), default=0)
            mmr = lam * rel - (1 - lam) * div
            if best is None or mmr > best:
                best, bi = mmr, i
        chosen.append(cand.pop(bi))
    return chosen

def retrieve_srag(question, k=6, pool=40):
    rows = _vec(_hyde(question), pool)          # HyDE-driven dense retrieval
    rows = _rerank(question, rows)              # deepseek rerank
    for r in rows[:min(len(rows), 20)]:         # embed top for MMR diversity
        r["_emb"] = R._qvec((r["text"] or "")[:400])
    rows = _mmr(rows[:20], k)
    return rows, _chunk_ctx(rows)

# ── B4 GraphRAG (reified Evidence; semantic fallback) ─────────────────────────────────────────────
def _ev_ctx(records):
    lines = []
    for r in records:
        q = str(r.get("source_quote") or "").strip()
        lines.append(f"[{r['paper_id']}] (level {r.get('evidence_level') or '?'}, "
                     f"dir={r.get('direction_norm') or '?'}, effect={r.get('effect_size') or '?'}): {q}")
    return "\n".join(lines)

def retrieve_b4(question, k=12):
    # "A" upgrade: query ALL resolved intervention×outcome pairs (multi-outcome PICO), with IS_A rollup ON
    # (poolable ontology children), and fold in pooled meta estimates — so the graph surfaces the evidence
    # it actually holds instead of a single narrow pair. Gives the graph its best shot.
    p = R.parse_question(question)
    qt = p["query_type"]
    ivs, outs = p["interventions"], p["outcomes"]
    recs, seen = [], set()
    def add(rlist):
        for r in (rlist or []):
            if not isinstance(r, dict):
                continue
            key = (r.get("paper_id"), str(r.get("source_quote") or "")[:48])
            if r.get("paper_id") and key not in seen:
                seen.add(key); recs.append(r)
    if qt == "compare" and len(ivs) >= 2 and outs:
        for o in outs:
            res = R.q2_compare(ivs[0], ivs[1], o)
            for arm in (res.get("arms", {}) or {}).values():
                add((arm or {}).get("top_evidence", []))
    elif qt == "prognostic" and p.get("predictor") and outs:
        for o in outs:
            res = R.prognostic_evidence(p["predictor"], o)
            add(res.get("ranked_evidence", []) if isinstance(res, dict) else [])
    elif ivs and outs:
        for iv in ivs:
            for o in outs:
                res = R.q1_intervention_outcome(iv, o, rollup=True)   # IS_A poolable rollup ON
                add(res.get("ranked_evidence", []))
                for pe in (res.get("pooled_evidence") or []):
                    add([pe])
    recs = sorted(recs, key=lambda r: -(r.get("score") or 0))[:k]
    if recs:
        return {"mode": qt, "records": recs, "parse": p}, _ev_ctx(recs)
    # honest graph-miss → semantic fallback over chunks (still v2, labeled)
    rows = _vec(question, k=6)
    return {"mode": "semantic_fallback", "records": rows, "parse": p}, _chunk_ctx(rows)

# ── B_sRAG_O = strong RAG + ONTOLOGY QUERY EXPANSION (no graph evidence injection) ────────────────
# Isolates the ontology's value on the VECTOR path: resolve the question's concepts, broaden the retrieval
# query with their canonical name + curated aliases + IS_A neighbours (parent/child terms), then run the
# same HyDE+vector+rerank+MMR. NO structured-evidence injection (that diluted answers in B4h). Graph is
# used ONLY as a controlled-vocabulary/ontology thesaurus for query expansion.
_VOCAB = None
def _vocab():
    global _VOCAB
    if _VOCAB is None:
        vf = os.environ.get("VOCAB_FILE") or f"{REPO}/docs/redesign_v2/reference_vocab_v2.json"
        V = json.load(open(vf)); _VOCAB = {}
        for et in ("interventions", "outcomes", "pathologies", "anatomy"):
            for x in V.get(et, []):
                _VOCAB[x["reference_id"]] = (et, x.get("name", ""), x.get("aliases", []) or [])
    return _VOCAB

_ONTO_LABEL = {"interventions": "Intervention", "outcomes": "Outcome", "pathologies": "Pathology", "anatomy": "Anatomy"}
def _onto_terms(question):
    p = R.parse_question(question)
    voc = _vocab(); terms = set()
    ids = [("interventions", i) for i in p["interventions"]] + [("outcomes", o) for o in p["outcomes"]]
    if p.get("predictor"):
        ids.append((None, p["predictor"]))
    for et, rid in ids:
        v = voc.get(rid)
        if not v:
            continue
        et2, name, aliases = v
        if name:
            terms.add(name)
        terms.update(aliases[:4])
        lbl = _ONTO_LABEL.get(et2, "Intervention")
        try:
            for r in R._run(f"MATCH (c:{lbl} {{reference_id:$rid}})-[:IS_A]-(n) RETURN n.name AS name LIMIT 4", rid=rid):
                if r.get("name"):
                    terms.add(r["name"])
        except Exception:
            pass
    return [t for t in terms if t]

def retrieve_srag_onto(question, k=6, pool=40):
    terms = _onto_terms(question)
    exp = question + ((" (related terms: " + ", ".join(terms[:12]) + ")") if terms else "")
    rows = _vec(_hyde(exp), pool)            # HyDE + dense retrieval on the ontology-EXPANDED query
    rows = _rerank(question, rows)           # rerank against the ORIGINAL question
    for r in rows[:min(len(rows), 20)]:
        r["_emb"] = R._qvec((r["text"] or "")[:400])
    rows = _mmr(rows[:20], k)
    return rows, _chunk_ctx(rows)

# ── B4h GraphRAG-HYBRID (strong-RAG base + graph structured evidence + GRADE) — the FAIR graph system ─
# The plain B4 was graph-OR-vector (threw vector away when the graph was sparse → low recall). A proper
# graph-enhanced RAG is a SUPERSET of strong RAG: same HyDE+vector+rerank+MMR base, ENRICHED with the
# reified structured Evidence (direction, effect, evidence_level, provenance quote). Retrieved papers =
# graph evidence papers (on-target, GRADE-carrying) FIRST, then the strong-RAG vector papers, deduped.
RRF_K = 60
def _dedup(rows):
    seen, out = set(), []
    for r in rows:
        p = r.get("paper_id")
        if p and p not in seen:
            seen.add(p); out.append(r)
    return out

def retrieve_b4h(question, k=10, alpha=None):
    # alpha = graph WEIGHT in the fusion (0 = pure vector, 1 = pure graph). Reciprocal-Rank-Fusion blends
    # the strong-RAG vector ranking and the graph structured-evidence ranking so BOTH the graph's coverage
    # AND vector's early precision count — no naive graph-first that buries relevant vector hits.
    alpha = float(os.environ.get("V2_B4H_GRAPH_W", "0.5")) if alpha is None else alpha
    srag_rows, _ = retrieve_srag(question, k=k)          # strong-RAG base (broad recall)
    gmeta, _ = retrieve_b4(question, k=8)                # graph structured evidence (or semantic fallback)
    grecs = gmeta.get("records", []) if isinstance(gmeta, dict) else []
    struct = grecs if (isinstance(gmeta, dict) and gmeta.get("mode") != "semantic_fallback") else []
    gl, vl = _dedup(struct), _dedup(srag_rows)
    grank = {r["paper_id"]: i for i, r in enumerate(gl)}
    vrank = {r["paper_id"]: i for i, r in enumerate(vl)}
    info = {}
    for r in gl:                                         # prefer the graph record (carries provenance/level)
        info[r["paper_id"]] = r
    for r in vl:
        info.setdefault(r["paper_id"], r)
    def rrf(p):
        s = 0.0
        if p in grank: s += alpha / (RRF_K + grank[p])
        if p in vrank: s += (1 - alpha) / (RRF_K + vrank[p])
        return s
    ordered = sorted(info.keys(), key=lambda p: -rrf(p))
    records = [info[p] for p in ordered]
    ev_ctx = _ev_ctx(gl[:8])                             # structured evidence block (provenance + GRADE)
    ch_ctx = _chunk_ctx(vl[:k])
    ctx = ((f"STRUCTURED GRAPH EVIDENCE (per-paper direction/effect + OCEBM level + provenance):\n{ev_ctx}\n\n"
            if ev_ctx.strip() else "")
           + (f"SUPPORTING LITERATURE CHUNKS:\n{ch_ctx}" if ch_ctx.strip() else ""))
    return {"mode": "hybrid_rrf", "records": records, "graph_n": len(gl), "alpha": alpha}, ctx


# ── top-level answer per system ───────────────────────────────────────────────────────────────────
def answer(system, question):
    system = system.lower()
    if system == "b3":
        txt = deepseek(B3_SYSTEM, question)
        return {"system": "B3", "question": question, "answer": txt,
                "citations": _citations(txt), "retrieved": []}
    if system == "b1":
        rows, ctx = retrieve_b1(question)
    elif system == "b2":
        rows, ctx = retrieve_b2(question)
    elif system in ("b_srag", "srag", "bsrag"):
        rows, ctx = retrieve_srag(question)
    elif system in ("b_srag_o", "srag_o"):
        rows, ctx = retrieve_srag_onto(question)
    elif system == "b4":
        meta, ctx = retrieve_b4(question); rows = meta
    elif system in ("b4h", "b4_hybrid"):
        meta, ctx = retrieve_b4h(question); rows = meta
    else:
        raise ValueError(f"unknown system {system}")
    txt = _gen(question, ctx) if ctx.strip() else "Insufficient evidence retrieved to answer."
    retrieved = (rows["records"] if isinstance(rows, dict) else rows)
    return {"system": system.upper(), "question": question, "answer": txt,
            "citations": _citations(txt), "context_chars": len(ctx),
            "n_retrieved": len(retrieved),
            "retrieved": [{k: v for k, v in r.items() if not k.startswith("_") and k != "text"
                           and k != "embedding"} for r in retrieved]}

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--smoke":
        q = sys.argv[2]
        for sysname in ("b1", "b2", "b_srag", "b4", "b3"):
            print(f"\n{'='*70}\n{sysname.upper()}\n{'='*70}")
            try:
                a = answer(sysname, q)
                print(f"n_retrieved={a.get('n_retrieved')} citations={a['citations']}")
                print(a["answer"][:700])
            except Exception as e:
                print(f"ERROR: {e}")
    else:
        print(__doc__)
