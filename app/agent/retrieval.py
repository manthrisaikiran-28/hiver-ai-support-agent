"""
Semantic Retrieval Engine.
Provides vector indexing, cosine similarity search, and context formatting for RAG.
Supports two retrieval modes:
  - Version A: TF-IDF Cosine Similarity Baseline
  - Version B: Dense Embedding Vector Retrieval (via SentenceTransformers when available)

Includes evaluation-safe leave-one-out retrieval (exclude_ticket_id) to prevent golden-set leakage.
"""

from __future__ import annotations
import math
import re
import time
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
    retrieved_ids: List[str]
    similarity_scores: List[float]
    retrieval_top_similarity: float
    top_k: int
    retrieval_latency_ms: float
    context_text: str
    is_sufficient_context: bool
    retrieval_method: str
    excluded_source_id: Optional[str] = None


class VectorStore:
    """
    Vector Store supporting TF-IDF & Cosine Similarity search with optional
    SentenceTransformers integration when available.
    """

    def __init__(self, use_transformer: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.items: List[RetrievalItem] = []
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.method_name = f"SentenceTransformer ({model_name})" if use_transformer else "TF-IDF Baseline"

    def add_items(self, items: List[RetrievalItem]) -> None:
        self.items = list(items)
        if self.use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[Warning] SentenceTransformer not available ({e}). Using TF-IDF fallback.")
                self.model = None
                self.use_transformer = False
                self.method_name = "TF-IDF Baseline (Fallback)"
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def _build_index(self) -> None:
        if not self.items:
            return

        docs = [f"{item.subject} {item.body}" for item in self.items]
        if self.use_transformer and self.model is not None:
            vecs = self.model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            self.embeddings = vecs.astype(np.float32)
            return

        doc_count = len(docs)
        tokenized_docs = [self._tokenize(d) for d in docs]
        doc_freqs: Dict[str, int] = {}

        for tokens in tokenized_docs:
            for t in set(tokens):
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        self.vocabulary = {t: idx for idx, t in enumerate(sorted(doc_freqs.keys()))}
        self.idf = {t: math.log((doc_count + 1) / (df + 1)) + 1.0 for t, df in doc_freqs.items()}

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

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = matrix / norms

    def _embed_query(self, query: str) -> np.ndarray:
        if self.use_transformer and self.model is not None:
            return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

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

    def search(self, query: str, top_k: int = 3,
               min_similarity: float = 0.15,
               exclude_ticket_id: Optional[str] = None) -> RetrievalResult:
        start_time = time.time()

        if not self.items or self.embeddings is None:
            latency = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query,
                retrieved_items=[],
                retrieved_ids=[],
                similarity_scores=[],
                retrieval_top_similarity=0.0,
                top_k=top_k,
                retrieval_latency_ms=round(latency, 2),
                context_text="",
                is_sufficient_context=False,
                retrieval_method=self.method_name,
                excluded_source_id=exclude_ticket_id,
            )

        # Leave-one-out filter
        valid_indices = [idx for idx, item in enumerate(self.items) if item.id != exclude_ticket_id]

        if not valid_indices:
            latency = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query,
                retrieved_items=[],
                retrieved_ids=[],
                similarity_scores=[],
                retrieval_top_similarity=0.0,
                top_k=top_k,
                retrieval_latency_ms=round(latency, 2),
                context_text="",
                is_sufficient_context=False,
                retrieval_method=self.method_name,
                excluded_source_id=exclude_ticket_id,
            )

        sub_embeddings = self.embeddings[valid_indices]
        q_vec = self._embed_query(query)
        sims = np.dot(sub_embeddings, q_vec.T).squeeze(axis=1)

        top_sub_indices = np.argsort(sims)[::-1][:top_k]

        retrieved_items = []
        retrieved_ids = []
        scores = []

        for sub_idx in top_sub_indices:
            score = float(sims[sub_idx])
            original_idx = valid_indices[sub_idx]
            if score >= min_similarity:
                item = self.items[original_idx]
                retrieved_items.append(item)
                retrieved_ids.append(item.id)
                scores.append(round(score, 4))

        top_sim = max(scores, default=0.0)
        latency = (time.time() - start_time) * 1000

        context_blocks = []
        for item, score in zip(retrieved_items, scores):
            context_blocks.append(
                f"[Doc | ID: {item.id} | Category: {item.category} | Sim: {score:.4f}]\n"
                f"Subject: {item.subject}\nBody: {item.body}"
            )
        context_text = "\n\n".join(context_blocks)
        is_sufficient = len(retrieved_items) > 0 and top_sim >= min_similarity

        return RetrievalResult(
            query=query,
            retrieved_items=retrieved_items,
            retrieved_ids=retrieved_ids,
            similarity_scores=scores,
            retrieval_top_similarity=top_sim,
            top_k=top_k,
            retrieval_latency_ms=round(latency, 2),
            context_text=context_text,
            is_sufficient_context=is_sufficient,
            retrieval_method=self.method_name,
            excluded_source_id=exclude_ticket_id,
        )


def build_knowledge_base_from_tickets(tickets: List[Dict[str, Any]],
                                       use_sentence_transformers: bool = False) -> VectorStore:
    store = VectorStore(use_transformer=use_sentence_transformers)
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


TFIDFVectorStore = VectorStore
SentenceTransformerVectorStore = VectorStore
