"""
ResearchAgent – orchestrates the full PaperTrail research pipeline.

Flow:
  question
    → PlanningChain   (structured search plan)
    → PaperRetriever  (semantic search)
    → Reranker        (re-score chunks)
    → SynthesisChain  (LLM-based answer)
    → CritiqueChain   (quality critique)
    → Evaluation      (faithfulness + coverage)
    → ResearchReport
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from papertrail.chains.critique import critique
from papertrail.chains.planning import plan_research
from papertrail.chains.synthesis import synthesize
from papertrail.evaluation.coverage import score_coverage
from papertrail.evaluation.faithfulness import score_faithfulness
from papertrail.memory.paper_memory import PaperMemory
from papertrail.memory.trend_memory import TrendMemory
from papertrail.retrieval.reranker import rerank
from papertrail.retrieval.retrievers import PaperRetriever
from papertrail.schemas.schema import ResearchReport, SearchResult


class ResearchAgent:
    """
    High-level research agent.

    Parameters
    ----------
    index_name : str
        Name of the FAISS index to use (default: "papers").
    retrieve_k : int
        How many chunks to retrieve before reranking (default: 20).
    rerank_k : int
        How many chunks to keep after reranking (default: 6).
    """

    def __init__(
        self,
        index_name: str = "papers",
        retrieve_k: int = 20,
        rerank_k: int = 6,
    ) -> None:
        self.retriever = PaperRetriever(index_name=index_name)
        self.memory = PaperMemory()
        self.trends = TrendMemory()
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def research(self, question: str) -> ResearchReport:
        """
        Run the full research pipeline for *question*.

        Returns a ResearchReport with synthesis, critique, and eval scores.
        """
        # 1. Plan
        plan = plan_research(question)

        # 2. Retrieve  (use all plan queries, deduplicate results)
        all_results: List[SearchResult] = []
        seen_keys: set = set()
        queries = [question] + (plan.queries or [])
        for q in queries[:4]:                         # limit API/compute cost
            for r in self.retriever.retrieve(q, k=self.retrieve_k):
                key = (r.paper_id, r.chunk_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(r)

        # 3. Rerank
        top_results = rerank(question, all_results, top_k=self.rerank_k)

        # 4. Synthesize
        synth = synthesize(question, top_results)

        # 5. Critique
        crit = critique(synth, question, top_results)

        # 6. Evaluate
        faith = score_faithfulness(synth, top_results)
        cov = score_coverage(question, top_results)

        return ResearchReport(
            question=question,
            plan=plan,
            synthesis=synth,
            critique=crit,
            sources=top_results,
            faithfulness_score=round(faith, 3),
            coverage_score=round(cov, 3),
            metadata={
                "retrieve_k": self.retrieve_k,
                "rerank_k": self.rerank_k,
                "total_candidates": len(all_results),
            },
            created_at=datetime.utcnow(),
        )

    # ──────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Lightweight Q&A – synthesis only (no critique / eval)."""
        results = rerank(
            question,
            self.retriever.retrieve(question, k=self.retrieve_k),
            top_k=self.rerank_k,
        )
        return synthesize(question, results)

    def is_ready(self) -> bool:
        """True when there are indexed papers to query."""
        return not self.retriever.is_empty()

    def stats(self) -> dict:
        return self.retriever.stats()
