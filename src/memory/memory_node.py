import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class MemoryNode:
    """
    Node in the semantic forest, representing a single memory unit.
    Can represent different granularities (object, room, area).
    """

    id: int
    embedding: torch.Tensor
    spatial_info: 'SpatialInfo'  # Forward reference
    metadata: Dict
    children: List['MemoryNode'] = None
    parent: Optional['MemoryNode'] = None
    creation_time: float = None

    def __post_init__(self):
        self.children = []
        self.creation_time = time.time()

    def add_child(self, child: 'MemoryNode'):
        """Add child node."""
        self.children.append(child)
        child.parent = self

    def remove_child(self, child: 'MemoryNode'):
        """Remove child node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def get_descendants(self) -> List['MemoryNode']:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def traverse(self) -> List['MemoryNode']:
        """Traverse node and all descendants."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.traverse())
        return nodes

    def prune_old_nodes(self, max_age: float):
        """Remove nodes older than max_age seconds."""
        current_time = time.time()
        self.children = [
            child for child in self.children
            if (current_time - child.creation_time) <= max_age
        ]
        for child in self.children:
            child.prune_old_nodes(max_age)

    def merge_with(self, other: 'MemoryNode'):
        """Merge another node into this one."""
        # Average embeddings
        self.embedding = (self.embedding + other.embedding) / 2

        # Merge metadata
        self.metadata.update(other.metadata)

        # Adopt children
        for child in other.children:
            self.add_child(child)

    def get_spatial_bounds(self) -> np.ndarray:
        """Get bounding box including all descendants."""
        positions = [self.spatial_info.position]
        for desc in self.get_descendants():
            positions.append(desc.spatial_info.position)
        positions = np.stack(positions)

        return np.array([
            np.min(positions, axis=0),
            np.max(positions, axis=0)
        ])