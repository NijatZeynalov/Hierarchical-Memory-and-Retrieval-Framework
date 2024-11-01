import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Bounds:
    """3D bounding box."""
    min_point: np.ndarray
    max_point: np.ndarray

def compute_distance(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """Compute Euclidean distance between points."""
    return np.linalg.norm(point1 - point2)

def get_bounds(points: List[np.ndarray]) -> Bounds:
    """Get bounding box for set of points."""
    points_array = np.array(points)
    return Bounds(
        min_point=np.min(points_array, axis=0),
        max_point=np.max(points_array, axis=0)
    )

def is_inside_bounds(
    point: np.ndarray,
    bounds: Bounds,
    margin: float = 0.0
) -> bool:
    """Check if point is inside bounds."""
    return all(
        bounds.min_point[i] - margin <= point[i] <= bounds.max_point[i] + margin
        for i in range(len(point))
    )

def normalize_orientation(
    quaternion: np.ndarray
) -> np.ndarray:
    """Normalize quaternion orientation."""
    return quaternion / np.linalg.norm(quaternion)