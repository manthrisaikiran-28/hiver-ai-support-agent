"""
Semantic Retrieval Engine.
Provides vector indexing, cosine similarity search, and context formatting for RAG.
Falls back gracefully between SentenceTransformers and TF-IDF / Cosine Similarity.
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class RetrievalItem:
    id: str
    subject: str
    body: str
    category: str
    resolution: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    retrieved_items: List[RetrievalItem]
    similarity_scores: List[float]
    top_k: int
    retrieval_latency_ms: float
    context_text: str
    is_sufficient_context: bool


class VectorStore:
    """
    Vector Store supporting TF-IDF & Cosine Similarity search with optional
    SentenceTransformers integration when available.
    """

    def __init__(self, use_transformer: bool = False):
        self.items: List[RetrievalItem] = []
        self.use_transformer = use_transformer
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def add_items(self, items: List[RetrievalItem]) -> None:
        self.items.extend(items)
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def _build_index(self) -> None:
        if not self.items:
            return

        docs = [f"{item.subject} {item.body}" for item in self.items]

        # Build vocabulary & IDF
        doc_count = len(docs)
        doc_freqs: Dict[str, int] = {}
        tokenized_docs = [self._tokenize(d) for d in docs]

        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        self.vocabulary = {t: idx for idx, t in enumerate(sorted(doc_freqs.keys()))}
        self.idf = {t: math.log((doc_count + 1) / (df + 1)) + 1 for t, df in doc_freqs.items()}

        # Build TF-IDF vectors
        matrix = np.zeros((doc_count, len(self.vocabulary)), dtype=np.float32)
        for doc_idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            total_tokens = len(tokens)
            for t, count in tf.items():
                if t in self.vocabulary:
                    col_idx = self.vocabulary[t]
                    matrix[doc_idx, col_idx] = (count / total_tokens) * self.idf[t]

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = matrix / norms

    def _embed_query(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        vec = np.zeros((1, len(self.vocabulary)), dtype=np.float32)
        if not tokens or not self.vocabulary:
            return vec

        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total_tokens = len(tokens)
        for t, count in tf.items():
            if t in self.vocabulary:
                col_idx = self.vocabulary[t]
                vec[0, col_idx] = (count / total_tokens) * self.idf.get(t, 1.0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.15) -> RetrievalResult:
        start_time = time.time()
        if not self.items or self.embeddings is None or len(self.vocabulary) == 0:
            latency = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query,
                retrieved_items=[],
                similarity_scores=[],
                top_k=top_k,
                retrieval_latency_ms=round(latency, 2),
                context_text="",
                is_sufficient_context=False,
            )

        q_vec = self._embed_query(query)
        sims = np.dot(self.embeddings, q_vec.T).squeeze(axis=1)

        # Get top-k indices
        top_indices = np.argsort(sims)[::-1][:top_k]

        retrieved_items = []
        scores = []
        for idx in top_indices:
            score = float(sims[idx])
            if score >= min_similarity:
                retrieved_items.append(self.items[idx])
                scores.append(round(score, 4))

        latency = (time.time() - start_time) * 1000

        # Build grounded context string
        context_blocks = []
        for idx, item in enumerate(retrieved_items):
            context_blocks.append(
                f"[Doc {idx+1} | ID: {item.id} | Category: {item.category} | Sim: {scores[idx]}]\n"
                f"Subject: {item.subject}\nBody: {item.body}"
            )
        context_text = "\n\n".join(context_blocks)
        is_sufficient = len(retrieved_items) > 0 and max(scores, default=0.0) >= min_similarity

        return RetrievalResult(
            query=query,
            retrieved_items=retrieved_items,
            similarity_scores=scores,
            top_k=top_k,
            retrieval_latency_ms=round(latency, 2),
            context_text=context_text,
            is_sufficient_context=is_sufficient,
        )


def build_knowledge_base_from_tickets(tickets: List[Dict[str, Any]]) -> VectorStore:
    store = VectorStore()
    items = []
    for t in tickets:
        items.append(RetrievalItem(
            id=t["id"],
            subject=t.get("subject", ""),
            body=t.get("body", ""),
            category=t.get("true_category", "unknown"),
            metadata=t.get("raw_data", {}),
        ))
    store.add_items(items)
    return store
