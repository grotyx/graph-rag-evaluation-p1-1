"""Phase-A LLM-as-judge harness (npj re-test). Scores B*_v2 answers on a 5-dimension rubric,
adds two automatic metrics, aggregates, and reports inter-judge agreement.

Judges (policy-compliant, all DISTINCT from the deepseek generator → no self-preference):
  - gemini : OpenRouter  (google/gemini-* ; user-approved OpenRouter route)
  - gpt    : Codex CLI    (GPT only via Codex CLI — never OpenRouter)
  - claude : IN-SESSION   (harness Agent; NEVER the paid Anthropic API) — via export/import bundle

Automatic (no LLM): hallucination_rate (cited paper_id absent from the v2 graph) + mean evidence_level
of cited papers. NO Anthropic API anywhere in this file.

Flow:
  1. python judge_v2.py score --answers DIR --judge gemini            # OpenRouter judge, writes scores
  2. python judge_v2.py score --answers DIR --judge gpt               # Codex CLI judge
  3. python judge_v2.py export-claude --answers DIR --out tasks.json  # in-session Claude scores these
     python judge_v2.py import-claude --scores claude_scores.json --answers DIR
  4. python judge_v2.py aggregate --answers DIR                       # combine + inter-judge agreement
"""
import os, sys, json, re, argparse, subprocess, urllib.request, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./docs/redesign_v2/prototypes")

DIMS = ["faithfulness", "citation_fidelity", "answer_relevancy", "completeness", "clinical_correctness"]
RUBRIC = (
    "You are grading an answer to a spine-surgery clinical question. Score each dimension 0.0-1.0:\n"
    "- faithfulness: every claim is supported by the answer's own cited evidence (no unsupported assertions).\n"
    "- citation_fidelity: the [paper_id] citations actually match the claims they are attached to.\n"
    "- answer_relevancy: the answer addresses the specific question asked.\n"
    "- completeness: key evidence/considerations for this question are not omitted.\n"
    "- clinical_correctness: the answer is medically accurate and safe.\n"
    'Return ONLY JSON: {"faithfulness":0.0,"citation_fidelity":0.0,"answer_relevancy":0.0,'
    '"completeness":0.0,"clinical_correctness":0.0,"note":"<one line>"}')

def _prompt(ans):
    return (f"{RUBRIC}\n\nQUESTION:\n{ans['question']}\n\nANSWER:\n{ans['answer']}\n\n"
            f"CITATIONS IN ANSWER: {ans.get('citations')}\n\nJSON:")

def _parse_scores(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except Exception:
        return None
    out = {k: float(d.get(k, 0) or 0) for k in DIMS}
    out["note"] = str(d.get("note", ""))[:200]
    return out

# ── OpenRouter judge (Gemini). GPT via OpenRouter is FORBIDDEN by policy. ──────────────────────────
def _or_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        for line in open("./.env"):
            if line.startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"'); os.environ["OPENROUTER_API_KEY"] = k; break
    return k

def openrouter_judge(ans, model=None):
    # gemini-2.5-flash: strong-enough judge, cheap, minimal reasoning bloat (2.5-pro burns the token
    # budget on hidden reasoning and truncates the JSON). Override via V2_JUDGE_GEMINI_MODEL.
    model = model or os.environ.get("V2_JUDGE_GEMINI_MODEL", "google/gemini-2.5-flash")
    if "anthropic" in model or "claude" in model or model.startswith("openai/") or "gpt" in model.lower():
        raise RuntimeError(f"policy: '{model}' not allowed via OpenRouter (Claude=in-session, GPT=Codex CLI)")
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 1500,
                       "messages": [{"role": "user", "content": _prompt(ans)}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {_or_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    msg = (d.get("choices") or [{}])[0].get("message", {})
    return _parse_scores(msg.get("content") or msg.get("reasoning") or "")

# ── GPT judge via Codex CLI (never OpenRouter) ────────────────────────────────────────────────────
def codex_judge(ans):
    try:
        p = subprocess.run(["codex", "exec", _prompt(ans)], capture_output=True, text=True, timeout=180)
        return _parse_scores(p.stdout)
    except FileNotFoundError:
        raise RuntimeError("codex CLI not found — run `codex login` / install; GPT judge needs Codex CLI")

# ── automatic metrics (no LLM) ────────────────────────────────────────────────────────────────────
_PAPER_CACHE = None
def _corpus_papers():
    """Set of corpus paper_ids for the non-indexed-citation check. Returns None if Neo4j is unreachable
    (the check is then skipped, not fatal — a DB blip must never crash a long judge run)."""
    global _PAPER_CACHE
    if _PAPER_CACHE is None:
        try:
            import retrieve_v2 as R
            rows = R._run("MATCH (p:Paper) RETURN p.paper_id AS pid")
            _PAPER_CACHE = {r["pid"] for r in rows}
        except Exception as e:
            print(f"[auto_metrics] corpus lookup unavailable ({e}); skipping non-indexed check")
            return None
    return _PAPER_CACHE

def auto_metrics(ans):
    cites = ans.get("citations") or []
    papers = _corpus_papers()
    if papers is None:
        return {"n_citations": len(cites), "n_non_indexed": None, "non_indexed_citations": [],
                "hallucination_rate": None, "mean_evidence_rank": None, "corpus_check": "unavailable"}
    non_indexed = [c for c in cites if c not in papers]
    # evidence_level of cited (indexed) papers, if present in retrieved records
    lv = {r.get("paper_id"): r.get("evidence_level") for r in (ans.get("retrieved") or []) if isinstance(r, dict)}
    RANK = {"1a": 1, "1b": 2, "2a": 3, "2b": 4, "3": 5, "4": 6, "5": 7}
    ranks = [RANK[lv[c]] for c in cites if lv.get(c) in RANK]
    return {"n_citations": len(cites), "n_non_indexed": len(non_indexed),
            "non_indexed_citations": non_indexed,
            "hallucination_rate": round(len(non_indexed) / len(cites), 3) if cites else 0.0,
            "mean_evidence_rank": round(statistics.mean(ranks), 2) if ranks else None}

# ── scoring driver ────────────────────────────────────────────────────────────────────────────────
def _answer_files(adir):
    return [os.path.join(adir, f) for f in os.listdir(adir) if f.endswith(".json") and not f.startswith("_")]

def score(adir, judge, model=None, limit=None):
    backend = {"gemini": lambda a: openrouter_judge(a, model), "gpt": codex_judge}[judge]
    for fp in _answer_files(adir):
        blob = json.load(open(fp))
        answers = blob["answers"][:limit] if limit else blob["answers"]
        for a in answers:
            a.setdefault("auto", auto_metrics(a))
            a.setdefault("judges", {})
            prev = a["judges"].get(judge)
            if isinstance(prev, dict) and "error" not in prev:   # keep only a valid prior score
                continue
            try:
                a["judges"][judge] = backend(a)
            except Exception as e:
                a["judges"][judge] = {"error": str(e)}
            time.sleep(0.2)
        json.dump(blob, open(fp, "w"), ensure_ascii=False, indent=1)
        print(f"[{judge}] scored {fp} ({len(answers)} answers)")

# ── Claude in-session export/import ───────────────────────────────────────────────────────────────
def export_claude(adir, out):
    tasks = []
    for fp in _answer_files(adir):
        blob = json.load(open(fp))
        for a in blob["answers"]:
            tasks.append({"key": f"{blob['system']}::{a.get('id')}", "prompt": _prompt(a)})
    json.dump({"rubric_dims": DIMS, "tasks": tasks}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"exported {len(tasks)} Claude judge tasks → {out}  (score in-session, save as claude_scores.json: {{key: {{dims...}}}})")

def import_claude(scores_path, adir):
    scores = json.load(open(scores_path))
    for fp in _answer_files(adir):
        blob = json.load(open(fp))
        for a in blob["answers"]:
            k = f"{blob['system']}::{a.get('id')}"
            if k in scores:
                a.setdefault("judges", {})["claude"] = scores[k]
        json.dump(blob, open(fp, "w"), ensure_ascii=False, indent=1)
    print(f"imported claude scores into {adir}")

# ── aggregation + inter-judge agreement ───────────────────────────────────────────────────────────
def aggregate(adir):
    per_system = {}
    judge_pairs = {}
    for fp in _answer_files(adir):
        blob = json.load(open(fp)); sysn = blob["system"]
        rows = []
        for a in blob["answers"]:
            js = {j: s for j, s in (a.get("judges") or {}).items() if isinstance(s, dict) and "error" not in s}
            if js:
                mean_dim = {d: statistics.mean([s[d] for s in js.values() if d in s]) for d in DIMS}
                rows.append({"id": a.get("id"), "dims": mean_dim,
                             "total": round(statistics.mean(list(mean_dim.values())), 3),
                             "auto": a.get("auto", {})})
                # collect for pairwise agreement (per-dim overall score)
                for j, s in js.items():
                    judge_pairs.setdefault(j, []).append(statistics.mean([s[d] for d in DIMS if d in s]))
        if rows:
            per_system[sysn] = {
                "n": len(rows),
                "mean_total": round(statistics.mean(r["total"] for r in rows), 3),
                "dims": {d: round(statistics.mean(r["dims"][d] for r in rows), 3) for d in DIMS},
                "mean_hallucination": round(statistics.mean(r["auto"].get("hallucination_rate", 0) for r in rows), 3),
            }
    print(json.dumps({"per_system": per_system,
                      "judges_scored": {j: len(v) for j, v in judge_pairs.items()}}, indent=1, ensure_ascii=False))
    return per_system

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--answers", required=True); s.add_argument("--judge", required=True, choices=["gemini", "gpt"]); s.add_argument("--model"); s.add_argument("--limit", type=int)
    e = sub.add_parser("export-claude"); e.add_argument("--answers", required=True); e.add_argument("--out", default="evaluation/results/claude_judge_tasks.json")
    i = sub.add_parser("import-claude"); i.add_argument("--scores", required=True); i.add_argument("--answers", required=True)
    g = sub.add_parser("aggregate"); g.add_argument("--answers", required=True)
    a = ap.parse_args()
    if a.cmd == "score": score(a.answers, a.judge, a.model, a.limit)
    elif a.cmd == "export-claude": export_claude(a.answers, a.out)
    elif a.cmd == "import-claude": import_claude(a.scores, a.answers)
    elif a.cmd == "aggregate": aggregate(a.answers)
