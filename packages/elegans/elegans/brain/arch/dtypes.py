"""Define the types of brains used in the elegans project."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class BrainType(Enum):
    """Different types of brains.

    Naming convention: {Architecture}_{Algorithm}
    - MLP prefix = classical multi-layer perceptron
    - Spiking prefix = spiking neural network
    """

    # New canonical names
    HYBRID_CLASSICAL = "hybridclassical"
    MLP_REINFORCE = "mlpreinforce"
    MLP_DQN = "mlpdqn"
    MLP_PPO = "mlpppo"
    SPIKING_REINFORCE = "spikingreinforce"

    # Deprecated aliases (kept for backward compatibility)
    MLP = "mlp"
    PPO = "ppo"
    SPIKING = "spiking"


class DeviceType(Enum):
    """Different types of devices for running processing for brains.

    - CPU: Central Processing Unit
    - GPU: Graphics Processing Unit
    """

    CPU = "cpu"
    GPU = "gpu"


class BrainConfig(BaseModel):
    """Configuration for the brain architecture."""

    seed: int | None = None  # Random seed for reproducibility


BRAIN_TYPES = Literal[
    BrainType.HYBRID_CLASSICAL,
    BrainType.MLP_REINFORCE,
    BrainType.MLP_DQN,
    BrainType.MLP_PPO,
    BrainType.SPIKING_REINFORCE,
    # Deprecated aliases
    BrainType.MLP,
    BrainType.PPO,
    BrainType.SPIKING,
]
CLASSICAL_BRAIN_TYPES: set[BrainType] = {
    BrainType.HYBRID_CLASSICAL,
    BrainType.MLP_REINFORCE,
    BrainType.MLP_DQN,
    BrainType.MLP_PPO,
    BrainType.MLP,
    BrainType.PPO,
}
SPIKING_BRAIN_TYPES: set[BrainType] = {
    BrainType.SPIKING_REINFORCE,
    BrainType.SPIKING,
}

# Map deprecated names to canonical names
BRAIN_NAME_ALIASES: dict[str, str] = {
    "mlp": "mlpreinforce",
    "ppo": "mlpppo",
    "spiking": "spikingreinforce",
}

# Defaults
DEFAULT_BRAIN_TYPE = BrainType.MLP_REINFORCE
