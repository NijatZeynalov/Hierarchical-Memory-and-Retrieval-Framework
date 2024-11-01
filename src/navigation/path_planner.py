from typing import List, Optional, Tuple
import numpy as np


class PathPlanner:
    """Simple path planner for agent navigation."""

    def __init__(
            self,
            step_size: float = 0.5,
            collision_threshold: float = 0.3
    ):
        self.step_size = step_size
        self.collision_threshold = collision_threshold

    def plan(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            obstacles: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """Generate collision-free path."""
        path = [start]
        current = start

        while np.linalg.norm(current - goal) > self.step_size:
            # Get next position
            direction = self._get_direction(current, goal, obstacles)
            next_pos = current + direction * self.step_size

            # Check if valid
            if obstacles and self._check_collision(next_pos, obstacles):
                break

            path.append(next_pos)
            current = next_pos

        return path

    def _get_direction(
            self,
            current: np.ndarray,
            goal: np.ndarray,
            obstacles: Optional[List[np.ndarray]]
    ) -> np.ndarray:
        """Get safe direction towards goal."""
        direction = goal - current
        return direction / np.linalg.norm(direction)

    def _check_collision(
            self,
            position: np.ndarray,
            obstacles: List[np.ndarray]
    ) -> bool:
        """Check for collisions with obstacles."""
        return any(
            np.linalg.norm(position - obs) < self.collision_threshold
            for obs in obstacles
        )