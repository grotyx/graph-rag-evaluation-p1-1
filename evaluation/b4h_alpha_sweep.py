"""B4h fusion-weight (alpha) sweep — find the graph% that best fuses graph+vector retrieval.

RRF only reorders the SAME union of papers, so one retrieval pass + one relevance-judgment pass suffices;
we then score recall@k / nDCG@k for several alpha (0=pure vector … 1=pure graph). Answers whether fixing
the ranking (vs naive graph-first) lets the hybrid beat vector, and at what graph weight. NO Anthropic API.
"""
import os, sys, json, math, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./docs/redesign_v2/prototypes")
import answer_generator_v2 as G
import retrieval_metrics_v2 as RM

RRF_K = 60
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
KS = [5, 10]

def _lists(question, k=10):
    """The graph and vector ranked paper lists that feed the fusion (dedup, rank order)."""
    srag_rows, _ = G.retrieve_srag(question, k=k)
    gmeta, _ = G.retrieve_b4(question, k=8)
    grecs = gmeta.get("records", []) if isinstance(gmeta, dict) else []
    struct = grecs if (isinstance(gmeta, dict) and gmeta.get("mode") != "semantic_fallback") else []
    gl = [r["paper_id"] for r in G._dedup(struct)]
    vl = [r["paper_id"] for r in G._dedup(srag_rows)]
    return gl, vl

def _rrf_order(gl, vl, alpha):
    gr = {p: i for i, p in enumerate(gl)}; vr = {p: i for i, p in enumerate(vl)}
    def s(p):
        x = 0.0
        if p in gr: x += alpha / (RRF_K + gr[p])
        if p in vr: x += (1 - alpha) / (RRF_K + vr[p])
        return x
    return sorted(set(gl) | set(vr), key=lambda p: -s(p))

def _ndcg(ranked, relset, k):
    dcg = sum((1 if ranked[i] in relset else 0) / math.log2(i + 2) for i in range(min(k, len(ranked))))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relset))))
    return dcg / idcg if idcg else 0.0

def run(qfile):
    qs = json.load(open(qfile))["questions"]
    acc = {a: {f"R@{k}": [] for k in KS} for a in ALPHAS}
    for a in ALPHAS:
        for k in KS: acc[a][f"nDCG@{k}"] = []
    for q in qs:
        gl, vl = _lists(q["question"])
        pool = list(dict.fromkeys(gl + vl))
        if not pool:
            continue
        rel = RM.judge_relevance(q["question"], pool)
        relset = {p for p in pool if rel[p]}
        if not relset:
            continue
        for a in ALPHAS:
            order = _rrf_order(gl, vl, a)
            for k in KS:
                hit = sum(1 for p in order[:k] if p in relset)
                acc[a][f"R@{k}"].append(hit / len(relset))
                acc[a][f"nDCG@{k}"].append(_ndcg(order, relset, k))
    print("alpha(graph%)  " + "  ".join(f"{m:>8}" for m in [f"R@{k}" for k in KS] + [f"nDCG@{k}" for k in KS]))
    out = {}
    for a in ALPHAS:
        row = [round(statistics.mean(acc[a][f"R@{k}"]), 3) for k in KS] + [round(statistics.mean(acc[a][f"nDCG@{k}"]), 3) for k in KS]
        out[a] = row
        print(f"  {a:<11}  " + "  ".join(f"{v:>8}" for v in row))
    json.dump(out, open("./evaluation/results/answers_v2_subset32/_alpha_sweep.json", "w"), indent=1)

if __name__ == "__main__":
    run(sys.argv[1])
