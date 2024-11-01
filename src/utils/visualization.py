import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from ..memory.memory_node import MemoryNode


class MemoryVisualizer:
    """Simple visualization tools for memory structure."""

    def plot_memory_graph(
            self,
            nodes: List[MemoryNode],
            ax: Optional[plt.Axes] = None
    ):
        """Plot memory nodes and connections."""
        if ax is None:
            _, ax = plt.subplots()

        positions = [node.spatial_info.position for node in nodes]
        positions = np.array(positions)

        # Plot nodes
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', alpha=0.6)

        # Plot connections
        for node in nodes:
            if node.children:
                for child in node.children:
                    child_pos = child.spatial_info.position
                    ax.plot(
                        [node.spatial_info.position[0], child_pos[0]],
                        [node.spatial_info.position[1], child_pos[1]],
                        'k-', alpha=0.3
                    )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return ax

    def plot_path(
            self,
            path: List[np.ndarray],
            ax: Optional[plt.Axes] = None
    ):
        """Plot navigation path."""
        if ax is None:
            _, ax = plt.subplots()

        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'r--', linewidth=2)
        ax.scatter(path[0, 0], path[0, 1], c='g', label='Start')
        ax.scatter(path[-1, 0], path[-1, 1], c='r', label='Goal')
        ax.legend()
        return ax