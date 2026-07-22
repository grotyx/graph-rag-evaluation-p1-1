"""Evaluation metrics for retrieval performance.

Metrics:
  - Precision@K, Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)
  - ELA (Evidence Level Accuracy) — average OCEBM level of returned papers
  - Top5_EL — average evidence level of top-5 results
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# OCEBM Evidence Level → numeric score (higher = better evidence)
EVIDENCE_LEVEL_SCORE = {
    "1a": 7,  # Systematic review / Meta-analysis
    "1b": 6,  # RCT
    "2a": 5,  # Cohort study
    "2b": 4,  # Case-control study
    "3": 3,   # Case series
    "4": 2,   # Expert opinion
    "5": 1,   # Unknown / narrative
}


@dataclass
class RetrievalMetrics:
    """Metrics for a single query evaluation."""

    query_id: str
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    ela: float = 0.0          # Evidence Level Accuracy (all results)
    top5_el: float = 0.0      # Average evidence level of top-5
    num_results: int = 0
    num_relevant: int = 0


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all queries."""

    baseline_name: str
    num_queries: int = 0
    mean_precision_at_5: float = 0.0
    mean_precision_at_10: float = 0.0
    mean_precision_at_20: float = 0.0
    mean_recall_at_5: float = 0.0
    mean_recall_at_10: float = 0.0
    mean_recall_at_20: float = 0.0
    mean_ndcg_at_10: float = 0.0
    mean_mrr: float = 0.0
    mean_ela: float = 0.0
    mean_top5_el: float = 0.0
    per_query: list[RetrievalMetrics] = field(default_factory=list)
    per_domain: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "baseline": self.baseline_name,
            "num_queries": self.num_queries,
            "P@5": round(self.mean_precision_at_5, 4),
            "P@10": round(self.mean_precision_at_10, 4),
            "P@20": round(self.mean_precision_at_20, 4),
            "R@5": round(self.mean_recall_at_5, 4),
            "R@10": round(self.mean_recall_at_10, 4),
            "R@20": round(self.mean_recall_at_20, 4),
            "NDCG@10": round(self.mean_ndcg_at_10, 4),
            "MRR": round(self.mean_mrr, 4),
            "ELA": round(self.mean_ela, 4),
            "Top5_EL": round(self.mean_top5_el, 4),
            "per_domain": self.per_domain,
        }


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant documents found in top-K."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def dcg_at_k(
    retrieved: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Discounted Cumulative Gain at K.

    Args:
        retrieved: Ordered list of document IDs.
        relevance_scores: doc_id → relevance score (e.g., 0/1 binary or graded).
        k: Cutoff rank.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(
    retrieved: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Computes NDCG by dividing DCG by the ideal DCG (perfect ranking).
    """
    dcg = dcg_at_k(retrieved, relevance_scores, k)

    # Ideal ranking: sort by relevance descending
    ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = dcg_at_k(ideal_order, relevance_scores, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(
    retrieved: list[str],
    relevant: set[str],
) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evidence_level_score(evidence_level: Optional[str]) -> float:
    """Convert evidence level string to numeric score."""
    if not evidence_level:
        return EVIDENCE_LEVEL_SCORE.get("5", 1)
    level = evidence_level.strip().lower()
    return EVIDENCE_LEVEL_SCORE.get(level, 1)


def evidence_level_accuracy(
    evidence_levels: list[Optional[str]],
) -> float:
    """Average evidence level score of returned results (0-7 scale)."""
    if not evidence_levels:
        return 0.0
    scores = [evidence_level_score(el) for el in evidence_levels]
    return sum(scores) / len(scores)


def evaluate_single_query(
    query_id: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    evidence_levels: list[Optional[str]],
    relevance_scores: Optional[dict[str, float]] = None,
) -> RetrievalMetrics:
    """Evaluate retrieval performance for a single query.

    Args:
        query_id: Unique query identifier.
        retrieved_ids: Ordered list of retrieved document (paper) IDs.
        relevant_ids: Set of relevant document IDs (gold standard).
        evidence_levels: Evidence levels of retrieved documents (same order).
        relevance_scores: Optional graded relevance scores for NDCG.
            If None, uses binary relevance (1 if in relevant_ids, else 0).
    """
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

    return RetrievalMetrics(
        query_id=query_id,
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_10=precision_at_k(retrieved_ids, relevant_ids, 10),
        precision_at_20=precision_at_k(retrieved_ids, relevant_ids, 20),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
        recall_at_20=recall_at_k(retrieved_ids, relevant_ids, 20),
        ndcg_at_10=ndcg_at_k(retrieved_ids, relevance_scores, 10),
        mrr=mrr(retrieved_ids, relevant_ids),
        ela=evidence_level_accuracy(evidence_levels),
        top5_el=evidence_level_accuracy(evidence_levels[:5]),
        num_results=len(retrieved_ids),
        num_relevant=len(relevant_ids),
    )


def aggregate_metrics(
    baseline_name: str,
    per_query_metrics: list[RetrievalMetrics],
    query_domains: Optional[dict[str, str]] = None,
) -> AggregateMetrics:
    """Aggregate per-query metrics into overall and per-domain averages.

    Args:
        baseline_name: Name of the baseline (B1, B2, B3, B4).
        per_query_metrics: List of per-query metrics.
        query_domains: Optional mapping of query_id → domain for per-domain breakdown.
    """
    n = len(per_query_metrics)
    if n == 0:
        return AggregateMetrics(baseline_name=baseline_name)

    agg = AggregateMetrics(
        baseline_name=baseline_name,
        num_queries=n,
        mean_precision_at_5=sum(m.precision_at_5 for m in per_query_metrics) / n,
        mean_precision_at_10=sum(m.precision_at_10 for m in per_query_metrics) / n,
        mean_precision_at_20=sum(m.precision_at_20 for m in per_query_metrics) / n,
        mean_recall_at_5=sum(m.recall_at_5 for m in per_query_metrics) / n,
        mean_recall_at_10=sum(m.recall_at_10 for m in per_query_metrics) / n,
        mean_recall_at_20=sum(m.recall_at_20 for m in per_query_metrics) / n,
        mean_ndcg_at_10=sum(m.ndcg_at_10 for m in per_query_metrics) / n,
        mean_mrr=sum(m.mrr for m in per_query_metrics) / n,
        mean_ela=sum(m.ela for m in per_query_metrics) / n,
        mean_top5_el=sum(m.top5_el for m in per_query_metrics) / n,
        per_query=per_query_metrics,
    )

    # Per-domain breakdown
    if query_domains:
        domain_metrics: dict[str, list[RetrievalMetrics]] = {}
        for m in per_query_metrics:
            domain = query_domains.get(m.query_id, "unknown")
            domain_metrics.setdefault(domain, []).append(m)

        for domain, metrics_list in domain_metrics.items():
            dn = len(metrics_list)
            agg.per_domain[domain] = {
                "num_queries": dn,
                "P@10": round(sum(m.precision_at_10 for m in metrics_list) / dn, 4),
                "R@10": round(sum(m.recall_at_10 for m in metrics_list) / dn, 4),
                "NDCG@10": round(sum(m.ndcg_at_10 for m in metrics_list) / dn, 4),
                "MRR": round(sum(m.mrr for m in metrics_list) / dn, 4),
                "ELA": round(sum(m.ela for m in metrics_list) / dn, 4),
            }

    return agg
