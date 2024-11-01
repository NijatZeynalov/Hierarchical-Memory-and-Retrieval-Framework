import torch
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from .semantic_forest import SemanticForest, SpatialInfo
from ..language.encoder import TextEncoder


class MemoryManager:
    """
    High-level interface for managing hierarchical memory.
    Handles memory operations, maintenance, and integration with other components.
    """

    def __init__(
            self,
            embedding_dim: int = 256,
            max_nodes: int = 10000,
            text_encoder: Optional[TextEncoder] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.forest = SemanticForest(
            max_nodes=max_nodes,
            embedding_dim=embedding_dim,
            device=device
        )
        self.text_encoder = text_encoder
        self.device = device

        # Cache for recent queries
        self.query_cache = {}

    def add_observation(
            self,
            observation: Dict,
            position: np.ndarray,
            metadata: Optional[Dict] = None
    ) -> int:
        """
        Process and store new observation.

        Args:
            observation: Dict containing observation data
            position: 3D position where observation was made
            metadata: Additional metadata for the observation

        Returns:
            ID of created memory node
        """
        # Create embedding from observation
        embedding = self._process_observation(observation)

        # Create spatial info
        spatial_info = SpatialInfo(
            position=position,
            orientation=observation.get('orientation', np.zeros(4)),
            scale=observation.get('scale', 1.0)
        )

        # Add to forest
        node = self.forest.add_observation(
            embedding=embedding,
            spatial_info=spatial_info,
            metadata=metadata or {}
        )

        return node.id

    def query_memory(
            self,
            query: Union[str, torch.Tensor],
            position: Optional[np.ndarray] = None,
            k: int = 5
    ) -> List[Dict]:
        """
        Query memory for relevant information.

        Args:
            query: Text query or embedding
            position: Optional position hint
            k: Number of results to return

        Returns:
            List of relevant memories with metadata
        """
        # Convert query to embedding if needed
        if isinstance(query, str) and self.text_encoder:
            query_embedding = self.text_encoder.encode(query)
        else:
            query_embedding = query

        # Query forest
        relevant_nodes = self.forest.query(
            embedding=query_embedding,
            spatial_hint=position,
            k=k
        )

        # Format results
        results = []
        for node in relevant_nodes:
            results.append({
                'id': node.id,
                'position': node.spatial_info.position,
                'metadata': node.metadata,
                'similarity': torch.cosine_similarity(
                    node.embedding.unsqueeze(0),
                    query_embedding.unsqueeze(0)
                ).item()
            })

        return results

    def maintain_memory(self, max_age: Optional[float] = None):
        """Perform memory maintenance."""
        # Prune old nodes
        if max_age:
            self.forest.prune(max_age)

        # Clear query cache
        self.query_cache.clear()

    def get_spatial_coverage(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get spatial bounds of all memories."""
        if not self.forest.roots:
            return np.zeros(3), np.zeros(3)

        positions = []
        for root in self.forest.roots:
            bounds = root.get_spatial_bounds()
            positions.extend([bounds[0], bounds[1]])

        positions = np.stack(positions)
        return np.min(positions, axis=0), np.max(positions, axis=0)

    def _process_observation(
            self,
            observation: Dict
    ) -> torch.Tensor:
        """Convert observation to embedding."""
        if 'embedding' in observation:
            return observation['embedding']

        # Process different observation types
        if 'image' in observation and self.text_encoder:
            return self.text_encoder.encode_image(observation['image'])
        elif 'text' in observation and self.text_encoder:
            return self.text_encoder.encode(observation['text'])
        else:
            raise ValueError("Unsupported observation format")