from typing import List, Dict
import numpy as np


class ActionGenerator:
    """Converts path plan into robot actions."""

    def __init__(self, action_step: float = 0.5):
        self.action_step = action_step

    def generate_actions(
            self,
            path: List[np.ndarray],
            current_pose: np.ndarray
    ) -> List[Dict]:
        """Convert path to robot actions."""
        actions = []
        current = current_pose

        for target in path[1:]:  # Skip start position
            # Get direction
            direction = target - current[:3]

            # Add rotation action if needed
            angle = np.arctan2(direction[1], direction[0])
            if abs(angle - current[3]) > 0.1:
                actions.append({
                    'type': 'rotate',
                    'params': {'angle': angle}
                })

            # Add movement action
            actions.append({
                'type': 'move',
                'params': {'position': target}
            })

            current = np.array([*target, angle])

        return actions