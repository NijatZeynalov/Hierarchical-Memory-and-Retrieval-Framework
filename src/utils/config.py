from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path
import yaml
import torch


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    embedding_dim: int = 256
    max_nodes: int = 10000
    similarity_threshold: float = 0.7
    spatial_weight: float = 0.3
    temporal_weight: float = 0.2


@dataclass
class NavigationConfig:
    """Navigation system configuration."""
    safety_margin: float = 0.5
    max_planning_steps: int = 100
    step_size: float = 0.2
    collision_threshold: float = 0.3


@dataclass
class AgentConfig:
    """Agent configuration."""
    platform_type: str = "mobile"  # mobile, drone, quadruped
    sensor_types: List[str] = None
    max_velocity: float = 1.0
    max_acceleration: float = 0.5

    def __post_init__(self):
        if self.sensor_types is None:
            self.sensor_types = ["camera", "lidar"]


@dataclass
class Config:
    """Main configuration class."""

    def __init__(
            self,
            config_path: Optional[str] = None,
            **kwargs
    ):
        # Default configurations
        self.memory = MemoryConfig()
        self.navigation = NavigationConfig()
        self.agent = AgentConfig()

        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # System settings
        self.debug_mode = False
        self.log_level = "INFO"
        self.save_path = "outputs"

        # Load from file if provided
        if config_path:
            self.load_config(config_path)

        # Override with kwargs
        self.update(**kwargs)

    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Update configurations
        if 'memory' in config_dict:
            self.memory = MemoryConfig(**config_dict['memory'])
        if 'navigation' in config_dict:
            self.navigation = NavigationConfig(**config_dict['navigation'])
        if 'agent' in config_dict:
            self.agent = AgentConfig(**config_dict['agent'])

        # Update system settings
        for key, value in config_dict.get('system', {}).items():
            if hasattr(self, key):
                setattr(self, key, value)

    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    def save(self, save_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'memory': self.memory.__dict__,
            'navigation': self.navigation.__dict__,
            'agent': self.agent.__dict__,
            'system': {
                'device': self.device,
                'debug_mode': self.debug_mode,
                'log_level': self.log_level,
                'save_path': self.save_path
            }
        }

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @property
    def config_dict(self) -> Dict:
        """Get configuration as dictionary."""
        return {
            'memory': self.memory.__dict__,
            'navigation': self.navigation.__dict__,
            'agent': self.agent.__dict__,
            'device': self.device,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level,
            'save_path': self.save_path
        }