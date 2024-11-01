from typing import Dict, List, Optional
import torch
import numpy as np


class RelevanceScorer:
    """Computes relevance scores for memory retrieval."""

    def __init__(
            self,
            semantic_weight: float = 0.6,
            temporal_weight: float = 0.2,
            spatial_weight: float = 0.2
    ):
        self.weights = {
            'semantic': semantic_weight,
            'temporal': temporal_weight,
            'spatial': spatial_weight
        }

    def compute_score(
            self,
            query_embedding: torch.Tensor,
            memory_embedding: torch.Tensor,
            memory_position: Optional[np.ndarray] = None,
            query_position: Optional[np.ndarray] = None,
            timestamp: Optional[float] = None
    ) -> float:
        """Compute combined relevance score."""
        scores = {}

        # Semantic similarity
        scores['semantic'] = self._semantic_score(
            query_embedding,
            memory_embedding
        )

        # Spatial relevance if positions provided
        if memory_position is not None and query_position is not None:
            scores['spatial'] = self._spatial_score(
                memory_position,
                query_position
            )

        # Temporal relevance if timestamp provided
        if timestamp is not None:
            scores['temporal'] = self._temporal_score(timestamp)

        # Combine scores
        total_score = sum(
            scores[k] * self.weights[k]
            for k in scores
        )

        return total_score

    def _semantic_score(
            self,
            query: torch.Tensor,
            memory: torch.Tensor
    ) -> float:
        """Compute semantic similarity."""
        return torch.cosine_similarity(
            query.unsqueeze(0),
            memory.unsqueeze(0)
        ).item()

    def _spatial_score(
            self,
            pos1: np.ndarray,
            pos2: np.ndarray
    ) -> float:
        """Compute spatial relevance."""
        distance = np.linalg.norm(pos1 - pos2)
        return 1.0 / (1.0 + distance)

    def _temporal_score(
            self,
            timestamp: float
    ) -> float:
        """Compute temporal relevance."""
        age = time.time() - timestamp
        return np.exp(-age / 3600)  # Decay over hours