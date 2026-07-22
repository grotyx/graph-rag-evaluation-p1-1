"""Baseline search modes for evaluation.

Four baseline search modes for comparing retrieval performance:
  B1: Keyword search (Neo4j fulltext index)
  B2: Vector-only search (embedding similarity, no authority/graph weighting)
  B3: LLM direct (ask Claude without Knowledge Graph)
  B4: GraphRAG (full hybrid system — current production mode)
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a baseline search."""

    paper_id: str
    title: str
    score: float
    evidence_level: Optional[str] = None
    year: Optional[int] = None
    study_design: Optional[str] = None
    chunk_text: Optional[str] = None


class BaselineSearch(ABC):
    """Abstract base class for baseline search modes."""

    name: str

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[BaselineResult]:
        """Execute search and return ranked results."""
        ...

    async def close(self) -> None:
        """Cleanup resources."""
        pass


class KeywordSearch(BaselineSearch):
    """B1: Keyword search using Neo4j fulltext index.

    Simulates PubMed-style keyword search. No semantic understanding,
    no evidence ranking — pure text matching.
    """

    name = "B1_Keyword"

    def __init__(self, neo4j_client: Any):
        self.client = neo4j_client

    async def search(self, query: str, top_k: int = 20) -> list[BaselineResult]:
        # Escape Lucene special characters for fulltext search
        escaped = _escape_lucene(query)

        cypher = """
        CALL db.index.fulltext.queryNodes('paper_text_search', $query)
        YIELD node, score
        WHERE node:Paper
        RETURN node.paper_id AS paper_id,
               node.title AS title,
               score,
               node.evidence_level AS evidence_level,
               node.publication_year AS year,
               node.study_design AS study_design
        ORDER BY score DESC
        LIMIT $top_k
        """
        try:
            rows = await self.client.run_query(
                cypher, {"query": escaped, "top_k": top_k}
            )
        except Exception:
            # Fallback: CONTAINS search with extracted keywords
            logger.warning("Fulltext index unavailable, falling back to CONTAINS")
            keywords = _extract_keywords(query)
            all_rows = []
            for kw in keywords[:5]:
                cypher = """
                MATCH (p:Paper)
                WHERE toLower(p.title) CONTAINS toLower($keyword)
                RETURN p.paper_id AS paper_id,
                       p.title AS title,
                       1.0 AS score,
                       p.evidence_level AS evidence_level,
                       p.year AS year,
                       p.study_design AS study_design
                LIMIT $limit
                """
                kw_rows = await self.client.run_query(
                    cypher, {"keyword": kw, "limit": top_k}
                )
                all_rows.extend(kw_rows)
            # Dedup
            seen_ids = set()
            rows = []
            for r in all_rows:
                pid = r.get("paper_id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    rows.append(r)
            rows = rows[:top_k]

        return [
            BaselineResult(
                paper_id=r["paper_id"],
                title=r.get("title", ""),
                score=r.get("score", 0.0),
                evidence_level=r.get("evidence_level"),
                year=r.get("year"),
                study_design=r.get("study_design"),
            )
            for r in rows
            if r.get("paper_id")
        ]


class VectorOnlySearch(BaselineSearch):
    """B2: Vector-only search using embedding similarity.

    Uses the same embedding model as GraphRAG but without
    authority weighting, graph relevance, or IS_A expansion.
    Pure semantic similarity ranking.
    """

    name = "B2_VectorOnly"

    def __init__(self, neo4j_client: Any, embedding_client: Any):
        self.client = neo4j_client
        self.embedding_client = embedding_client

    async def search(self, query: str, top_k: int = 20) -> list[BaselineResult]:
        # Generate embedding
        embedding = await self.embedding_client.embed(query)

        # Pure vector search — no graph filters, no authority scoring
        rows = await self.client.vector_search_chunks(
            embedding=embedding,
            top_k=top_k * 2,  # Over-retrieve to deduplicate by paper
            min_score=0.3,
        )

        # Deduplicate by paper_id, keep highest-scoring chunk per paper
        seen: dict[str, BaselineResult] = {}
        for r in rows:
            pid = r.get("paper_id", "")
            if not pid:
                continue
            score = r.get("score", 0.0)
            if pid not in seen or score > seen[pid].score:
                seen[pid] = BaselineResult(
                    paper_id=pid,
                    title=r.get("paper_title", ""),
                    score=score,
                    evidence_level=r.get("evidence_level"),
                    year=r.get("paper_year"),
                    chunk_text=r.get("content", "")[:200],
                )

        results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return results[:top_k]


class LLMDirectSearch(BaselineSearch):
    """B3: LLM direct — ask Claude without Knowledge Graph.

    Simulates the approach used in Kartal 2025, Najjar 2025, etc.
    Sends the clinical question directly to Claude and parses the response
    for referenced papers. Then matches against the database.

    This baseline measures the added value of the Knowledge Graph.
    """

    name = "B3_LLMDirect"

    def __init__(self, llm_client: Any, neo4j_client: Any):
        self.llm = llm_client
        self.client = neo4j_client

    async def search(self, query: str, top_k: int = 20) -> list[BaselineResult]:
        system_prompt = (
            "You are a spine surgery expert. Answer the clinical question below "
            "based on your medical knowledge. For each point, cite specific studies "
            "by first author and year (e.g., 'Smith 2023'). Be specific about "
            "study designs (RCT, cohort, meta-analysis) and evidence levels. "
            "List the most relevant studies with their key findings."
        )

        response = await self.llm.generate(
            prompt=query,
            system_prompt=system_prompt,
        )

        # Extract author-year citations from LLM response
        citations = _extract_citations(response)

        # Match citations against Neo4j database
        results: list[BaselineResult] = []
        for i, (author, year) in enumerate(citations[:top_k * 2]):
            matches = await self._find_paper(author, year)
            for m in matches:
                if m["paper_id"] not in {r.paper_id for r in results}:
                    results.append(BaselineResult(
                        paper_id=m["paper_id"],
                        title=m.get("title", ""),
                        score=1.0 - (i * 0.02),  # Rank by mention order
                        evidence_level=m.get("evidence_level"),
                        year=m.get("publication_year"),
                        study_design=m.get("study_design"),
                    ))

        # If few citations matched, also do a keyword fallback
        if len(results) < 5:
            logger.info("B3: Few citation matches (%d), adding keyword fallback", len(results))
            existing_ids = {r.paper_id for r in results}
            kw_results = await self._keyword_fallback(query, top_k=10)
            for kr in kw_results:
                if kr.paper_id not in existing_ids:
                    kr.score = 0.3  # Lower score for fallback
                    results.append(kr)

        return results[:top_k]

    async def _find_paper(self, author: str, year: Optional[int]) -> list[dict]:
        """Find paper in Neo4j by author name and year."""
        cypher = """
        MATCH (p:Paper)
        WHERE toLower(p.authors) CONTAINS toLower($author)
        """
        params: dict[str, Any] = {"author": author}
        if year:
            cypher += " AND p.publication_year = $year"
            params["year"] = year
        cypher += " RETURN p.paper_id AS paper_id, p.title AS title, "
        cypher += "p.evidence_level AS evidence_level, p.publication_year AS publication_year, "
        cypher += "p.study_design AS study_design LIMIT 3"
        try:
            return await self.client.run_query(cypher, params)
        except Exception as e:
            logger.warning("B3: Paper lookup failed for %s %s: %s", author, year, e)
            return []

    async def _keyword_fallback(self, query: str, top_k: int = 10) -> list[BaselineResult]:
        """Keyword search fallback when citation matching fails."""
        cypher = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($query)
           OR toLower(p.abstract) CONTAINS toLower($query)
        RETURN p.paper_id AS paper_id, p.title AS title,
               p.evidence_level AS evidence_level,
               p.publication_year AS publication_year,
               p.study_design AS study_design
        LIMIT $top_k
        """
        try:
            rows = await self.client.run_query(cypher, {"query": query, "top_k": top_k})
            return [
                BaselineResult(
                    paper_id=r["paper_id"],
                    title=r.get("title", ""),
                    score=0.3,
                    evidence_level=r.get("evidence_level"),
                    year=r.get("publication_year"),
                )
                for r in rows if r.get("paper_id")
            ]
        except Exception:
            return []


class GraphRAGSearch(BaselineSearch):
    """B4: Full GraphRAG — current production system.

    Uses the complete hybrid search pipeline:
    - 3-way ranking (semantic 0.4 + authority 0.3 + graph relevance 0.3)
    - IS_A ontology expansion
    - SNOMED-CT entity matching
    - Evidence level weighting
    """

    name = "B4_GraphRAG"

    def __init__(self, neo4j_client: Any, embedding_client: Any):
        self.client = neo4j_client
        self.embedding_client = embedding_client

    async def search(self, query: str, top_k: int = 20) -> list[BaselineResult]:
        from solver.hybrid_ranker import HybridRanker

        embedding = await self.embedding_client.embed(query)
        ranker = HybridRanker(neo4j_client=self.client)

        hybrid_results = await ranker.search(
            query=query,
            query_embedding=embedding,
            top_k=top_k * 2,  # Over-retrieve for dedup
        )

        # Deduplicate by paper_id
        seen: dict[str, BaselineResult] = {}
        for hr in hybrid_results:
            pid = hr.paper_id or ""
            if not pid:
                continue
            if pid not in seen or hr.final_score > seen[pid].score:
                seen[pid] = BaselineResult(
                    paper_id=pid,
                    title=hr.title or "",
                    score=hr.final_score,
                    evidence_level=hr.evidence_level,
                    year=hr.year,
                    chunk_text=(hr.content or "")[:200],
                )

        results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ============================================================================
# Helper functions
# ============================================================================

def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful medical keywords from a clinical question."""
    stop_words = {
        "what", "is", "the", "are", "for", "in", "of", "and", "or", "a", "an",
        "to", "from", "with", "by", "on", "at", "how", "does", "do", "which",
        "that", "this", "than", "versus", "vs", "between", "after", "before",
        "current", "evidence", "best", "role", "impact", "effect", "based",
        "compared", "comparing", "outcomes", "results", "regarding", "terms",
        "clinical", "surgical", "treatment", "management", "regarding",
    }
    words = re.findall(r"[A-Za-z0-9-]+", query)
    keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return keywords


def _escape_lucene(query: str) -> str:
    """Escape Lucene special characters for fulltext search."""
    special = r'+-&|!(){}[]^"~*?:\/'
    escaped = []
    for ch in query:
        if ch in special:
            escaped.append(f"\\{ch}")
        else:
            escaped.append(ch)
    return "".join(escaped)


def _extract_citations(text: str) -> list[tuple[str, Optional[int]]]:
    """Extract author-year citations from LLM response text.

    Handles patterns like:
      - Smith et al. (2023)
      - Smith 2023
      - (Smith, 2023)
      - Smith et al., 2023
    """
    patterns = [
        r"([A-Z][a-z]+)\s+et\s+al\.?\s*\(?(\d{4})\)?",
        r"([A-Z][a-z]+)\s+and\s+[A-Z][a-z]+\s*\(?(\d{4})\)?",
        r"\(([A-Z][a-z]+),?\s*(\d{4})\)",
        r"([A-Z][a-z]+)\s+(\d{4})",
    ]
    citations: list[tuple[str, Optional[int]]] = []
    seen: set[tuple[str, int]] = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            author = match.group(1)
            year = int(match.group(2))
            if 1990 <= year <= 2030 and (author, year) not in seen:
                seen.add((author, year))
                citations.append((author, year))

    return citations
