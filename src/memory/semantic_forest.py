from typing import List, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
from .memory_node import MemoryNode
import torch


@dataclass
class SpatialInfo:
    """Spatial information for memory nodes."""
    position: np.ndarray  # 3D position
    orientation: np.ndarray  # Quaternion
    scale: float  # Spatial scale of the node
    bounds: Optional[np.ndarray] = None  # Bounding box if applicable


class SemanticForest:
    """
    Hierarchical memory structure for storing agent experiences.
    Organizes memories in a forest of trees based on spatial and semantic relationships.
    """

    def __init__(
            self,
            max_nodes: int = 10000,
            embedding_dim: int = 256,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.max_nodes = max_nodes
        self.embedding_dim = embedding_dim
        self.device = device

        # Initialize forest structure
        self.roots: List[MemoryNode] = []
        self.node_count = 0

        # Spatial index for quick retrieval
        self.spatial_index = {}  # Dictionary mapping spatial regions to nodes

    def add_observation(
            self,
            embedding: torch.Tensor,
            spatial_info: SpatialInfo,
            metadata: Dict,
            parent: Optional[MemoryNode] = None
    ) -> MemoryNode:
        """Add new observation to the forest."""
        # Create new node
        node = MemoryNode(
            id=self.node_count,
            embedding=embedding,
            spatial_info=spatial_info,
            metadata=metadata
        )
        self.node_count += 1

        if parent:
            parent.add_child(node)
        else:
            # Try to find existing root to attach to, or create new root
            best_root = self._find_best_root(spatial_info)
            if best_root:
                best_root.add_child(node)
            else:
                self.roots.append(node)

        # Update spatial index
        self._update_spatial_index(node)

        return node

    def query(
            self,
            embedding: torch.Tensor,
            spatial_hint: Optional[np.ndarray] = None,
            k: int = 5
    ) -> List[MemoryNode]:
        """
        Query the forest for relevant nodes.

        Args:
            embedding: Query embedding
            spatial_hint: Optional spatial location to bias search
            k: Number of nodes to retrieve

        Returns:
            List of most relevant nodes
        """
        candidates = self._get_spatial_candidates(spatial_hint) if spatial_hint else None

        if candidates:
            # Score only spatially relevant nodes
            scores = torch.stack([
                self._compute_relevance(node, embedding)
                for node in candidates
            ])
        else:
            # Score all root nodes and their children
            scores = []
            nodes = []
            for root in self.roots:
                for node in root.traverse():
                    scores.append(self._compute_relevance(node, embedding))
                    nodes.append(node)
            scores = torch.stack(scores)

        # Get top-k nodes
        top_k_idx = torch.topk(scores, min(k, len(scores))).indices
        return [nodes[i] for i in top_k_idx]

    def _find_best_root(
            self,
            spatial_info: SpatialInfo
    ) -> Optional[MemoryNode]:
        """Find best existing root node to attach new observation."""
        if not self.roots:
            return None

        # Compute distances to existing roots
        distances = [
            np.linalg.norm(
                root.spatial_info.position - spatial_info.position
            )
            for root in self.roots
        ]

        best_idx = np.argmin(distances)
        if distances[best_idx] < spatial_info.scale:
            return self.roots[best_idx]
        return None

    def _update_spatial_index(self, node: MemoryNode):
        """Update spatial index with new node."""
        pos = tuple(np.round(node.spatial_info.position / node.spatial_info.scale))
        if pos not in self.spatial_index:
            self.spatial_index[pos] = []
        self.spatial_index[pos].append(node)

    def _get_spatial_candidates(
            self,
            position: np.ndarray,
            radius: float = 1.0
    ) -> List[MemoryNode]:
        """Get nodes within spatial radius of position."""
        candidates = []
        pos_key = tuple(np.round(position))

        # Check neighboring cells in spatial index
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    key = (pos_key[0] + dx, pos_key[1] + dy, pos_key[2] + dz)
                    if key in self.spatial_index:
                        candidates.extend(self.spatial_index[key])

        return candidates

    def _compute_relevance(
            self,
            node: MemoryNode,
            query_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute relevance score between node and query."""
        return torch.cosine_similarity(
            node.embedding.unsqueeze(0),
            query_embedding.unsqueeze(0)
        )

    def prune(self, max_age: Optional[float] = None):
        """Prune old or irrelevant nodes."""
        if max_age:
            current_time = time.time()
            for root in self.roots:
                root.prune_old_nodes(current_time - max_age)