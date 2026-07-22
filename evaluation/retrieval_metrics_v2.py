"""Retrieval-level evaluation (npj Track A, Phase 4) — pooled-relevance recall@k / precision@k / nDCG.

No pre-existing gold sets → TREC-style POOLING: pool the papers retrieved by B1/B2/B_sRAG/B4 per question,
judge each pooled paper's relevance once (batched per question, gemini via OpenRouter), then score each
system's ranked retrieval against the pooled-relevant set. Answers whether the graph retrieves the RIGHT
evidence even when final answer-quality ties. NO Anthropic API.

Usage: python evaluation/retrieval_metrics_v2.py --answers evaluation/results/answers_v2_subset32 --k 5,10
"""
import os, sys, json, re, argparse, math, urllib.request, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./docs/redesign_v2/prototypes")
SYSTEMS = ["b1", "b2", "b_srag", "b_srag_o", "b4", "b4h"]   # b3 has no retrieval; absent files are skipped

def _or_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        for line in open("./.env"):
            if line.startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"'); break
    return k

def _ranked_papers(ans, k=12):
    """Dedup paper_ids in retrieval rank order from an answer's retrieved list."""
    seen, out = set(), []
    for r in (ans.get("retrieved") or []):
        pid = r.get("paper_id")
        if pid and pid not in seen:
            seen.add(pid); out.append(pid)
        if len(out) >= k:
            break
    return out

_PINFO = {}
def _paper_info(pids):
    import retrieve_v2 as R
    todo = [p for p in pids if p not in _PINFO]
    if todo:
        rows = R._run("MATCH (p:Paper) WHERE p.paper_id IN $ids "
                      "OPTIONAL MATCH (p)<-[:FROM_PAPER]-(c:Chunk) "
                      "WITH p, collect(c.text)[0] AS snip "
                      "RETURN p.paper_id AS pid, p.title AS title, left(coalesce(snip,''),300) AS snip", ids=todo)
        for r in rows:
            _PINFO[r["pid"]] = (r.get("title") or r["pid"], r.get("snip") or "")
        for p in todo:
            _PINFO.setdefault(p, (p, ""))
    return {p: _PINFO[p] for p in pids}

def judge_relevance(question, pool, model=None):
    """One batched call: rate each pooled paper 1 (relevant to answering the question) or 0."""
    model = model or os.environ.get("V2_JUDGE_GEMINI_MODEL", "google/gemini-2.5-flash")
    info = _paper_info(pool)
    listing = "\n".join(f"[{p}] {info[p][0]} :: {info[p][1]}" for p in pool)
    prompt = ("For the clinical question, mark each paper 1 if it is RELEVANT to answering it (same "
              "intervention/comparison/outcome/population), else 0. Return ONLY JSON {\"<paper_id>\":0or1,...}.\n\n"
              f"QUESTION: {question}\n\nPAPERS:\n{listing}\n\nJSON:")
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 1200,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {_or_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    msg = (d.get("choices") or [{}])[0].get("message", {})
    txt = msg.get("content") or msg.get("reasoning") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    rel = {}
    if m:
        try:
            raw = json.loads(m.group())
            rel = {p: int(bool(int(raw.get(p, 0)))) for p in pool}
        except Exception:
            pass
    return {p: rel.get(p, 0) for p in pool}

def _ndcg(ranked, relset, k):
    dcg = sum((1 if ranked[i] in relset else 0) / math.log2(i + 2) for i in range(min(k, len(ranked))))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relset))))
    return dcg / idcg if idcg else 0.0

def run(answers_dir, ks):
    systems = [s for s in SYSTEMS if os.path.exists(f"{answers_dir}/{s}.json")]   # skip absent systems
    ansmap = {s: {a["id"]: a for a in json.load(open(f"{answers_dir}/{s}.json"))["answers"]} for s in systems}
    qids = sorted(set.intersection(*[set(m) for m in ansmap.values()]))
    per_sys = {s: {f"recall@{k}": [] for k in ks} for s in systems}
    for s in systems:
        for k in ks:
            per_sys[s][f"precision@{k}"] = []; per_sys[s][f"ndcg@{k}"] = []
    pooled_rel_counts = []
    for qid in qids:
        q = ansmap[systems[0]][qid]["question"]
        ranked = {s: _ranked_papers(ansmap[s][qid]) for s in systems}
        pool = list(dict.fromkeys(p for s in systems for p in ranked[s]))
        if not pool:
            continue
        rel = judge_relevance(q, pool)
        relset = {p for p in pool if rel[p]}
        pooled_rel_counts.append(len(relset))
        for s in systems:
            for k in ks:
                topk = ranked[s][:k]
                hit = sum(1 for p in topk if p in relset)
                per_sys[s][f"recall@{k}"].append(hit / len(relset) if relset else 0.0)
                per_sys[s][f"precision@{k}"].append(hit / k)
                per_sys[s][f"ndcg@{k}"].append(_ndcg(ranked[s], relset, k))
    print(f"questions={len(qids)}  mean pooled-relevant/q={statistics.mean(pooled_rel_counts):.1f}")
    hdr = ["system"] + [f"R@{k}" for k in ks] + [f"P@{k}" for k in ks] + [f"nDCG@{k}" for k in ks]
    print("  ".join(f"{h:>8}" for h in hdr))
    out = {}
    for s in systems:
        row = [s]
        for k in ks: row.append(round(statistics.mean(per_sys[s][f"recall@{k}"]), 3))
        for k in ks: row.append(round(statistics.mean(per_sys[s][f"precision@{k}"]), 3))
        for k in ks: row.append(round(statistics.mean(per_sys[s][f"ndcg@{k}"]), 3))
        out[s] = dict(zip(hdr[1:], row[1:]))
        print("  ".join(f"{str(v):>8}" for v in row))
    json.dump({"questions": len(qids), "per_system": out}, open(f"{answers_dir}/_retrieval_metrics.json", "w"), indent=1)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--k", default="5,10")
    a = ap.parse_args()
    run(a.answers, [int(x) for x in a.k.split(",")])
