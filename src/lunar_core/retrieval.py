"""
FAISS Vector Indexing & Similarity Retrieval Engine
Indexes LunaDNA 256-D descriptors, executes nearest-neighbor queries,
and measures Top-1, Top-5, Top-10 precision and query latencies.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import faiss
from pathlib import Path


class FAISSRetrievalEngine:
    """Manages dense vector indexing and candidate retrieval for lunar reference map matching."""

    def __init__(self, dimension: int = 256, use_gpu: bool = False):
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index = faiss.IndexFlatIP(dimension) # Inner Product on unit normalized vectors = Cosine Similarity
        self.doc_ids: List[str] = []

    def build_index(self, vectors: np.ndarray, doc_ids: List[str]):
        """Populate FAISS index with LunaDNA database vectors."""
        assert vectors.shape[1] == self.dimension, f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
        # Ensure L2 normalized float32
        faiss.normalize_L2(vectors)
        self.index.reset()
        self.index.add(vectors.astype(np.float32))
        self.doc_ids = list(doc_ids)

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[str], np.ndarray, float]:
        """Query index and return top_k matching document IDs, similarity scores, and execution latency (ms)."""
        t0 = time.perf_counter()
        q = query_vector.copy().astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        matched_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(self.doc_ids):
                matched_ids.append(self.doc_ids[idx])
            else:
                matched_ids.append("UNKNOWN")

        return matched_ids, scores[0], latency_ms

    def evaluate_retrieval(
        self,
        test_queries: np.ndarray,
        ground_truth_ids: List[str],
        top_k_eval: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """Compute Top-1, Top-5, Top-10 retrieval accuracy against known paired tiles."""
        metrics = {f"top_{k}_accuracy": 0.0 for k in top_k_eval}
        total = len(ground_truth_ids)
        if total == 0:
            return metrics

        max_k = max(top_k_eval)
        correct_counts = {k: 0 for k in top_k_eval}

        for i in range(total):
            matched_ids, _, _ = self.query(test_queries[i], top_k=max_k)
            gt = ground_truth_ids[i]
            for k in top_k_eval:
                if gt in matched_ids[:k]:
                    correct_counts[k] += 1

        for k in top_k_eval:
            metrics[f"top_{k}_accuracy"] = round((correct_counts[k] / total) * 100.0, 2)

        return metrics

    def save_index(self, index_path: str):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: str, doc_ids: List[str]):
        self.index = faiss.read_index(index_path)
        self.doc_ids = list(doc_ids)
