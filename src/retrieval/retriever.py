from typing import List, Dict, Optional
import torch
import numpy as np
from dataclasses import dataclass
from ..memory.memory_node import MemoryNode


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    node: MemoryNode
    score: float
    relevance: float
    distance: Optional[float] = None


class MemoryRetriever:
    """Core retrieval system for hierarchical memory."""

    def __init__(
            self,
            similarity_threshold: float = 0.7,
            max_results: int = 5,
            spatial_weight: float = 0.3
    ):
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.spatial_weight = spatial_weight

    def retrieve(
            self,
            query_embedding: torch.Tensor,
            memory_nodes: List[MemoryNode],
            position: Optional[np.ndarray] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant memories."""
        results = []

        for node in memory_nodes:
            # Compute semantic similarity
            similarity = self._compute_similarity(query_embedding, node.embedding)

            # Compute spatial score if position provided
            spatial_score = self._compute_spatial_score(position, node) if position is not None else 1.0

            # Combine scores
            score = similarity * (1 - self.spatial_weight) + spatial_score * self.spatial_weight

            if score > self.similarity_threshold:
                results.append(RetrievalResult(
                    node=node,
                    score=score,
                    relevance=similarity,
                    distance=spatial_score
                ))

        # Sort and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:self.max_results]

    def _compute_similarity(
            self,
            query_embedding: torch.Tensor,
            node_embedding: torch.Tensor
    ) -> float:
        """Compute semantic similarity."""
        return torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            node_embedding.unsqueeze(0)
        ).item()

    def _compute_spatial_score(
            self,
            position: np.ndarray,
            node: MemoryNode
    ) -> float:
        """Compute spatial relevance score."""
        distance = np.linalg.norm(position - node.spatial_info.position)
        return 1.0 / (1.0 + distance)