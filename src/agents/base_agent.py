from typing import Dict, List, Optional
import numpy as np
from ..memory.memory_manager import MemoryManager
from dataclasses import dataclass


@dataclass
class AgentState:
    position: np.ndarray
    orientation: np.ndarray
    status: str
    memory_id: Optional[int] = None


class BaseAgent:
    """Base class for embodied agents with memory capabilities."""

    def __init__(
            self,
            memory_manager: MemoryManager,
            observation_types: List[str] = ["visual", "spatial"]
    ):
        self.memory = memory_manager
        self.observation_types = observation_types
        self.state = AgentState(
            position=np.zeros(3),
            orientation=np.zeros(4),
            status="idle"
        )

    def observe(self) -> Dict:
        """Get current observation. Override in subclasses."""
        raise NotImplementedError

    def act(self, action: Dict) -> bool:
        """Execute action. Override in subclasses."""
        raise NotImplementedError

    def update_memory(self, observation: Dict) -> int:
        """Store new observation in memory."""
        return self.memory.add_observation(
            observation=observation,
            position=self.state.position,
            metadata={"agent_state": self.state}
        )

    def query_memory(self, query: str) -> List[Dict]:
        """Query memory for relevant information."""
        return self.memory.query_memory(
            query=query,
            position=self.state.position
        )

    def reset(self):
        """Reset agent state."""
        self.state = AgentState(
            position=np.zeros(3),
            orientation=np.zeros(4),
            status="idle"
        )