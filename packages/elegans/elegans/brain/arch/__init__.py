"""Module for brain architectures."""

from ._brain import Brain, BrainData, BrainParams, ClassicalBrain
from .hybridclassical import HybridClassicalBrain, HybridClassicalBrainConfig
from .mlpdqn import MLPDQNBrain, MLPDQNBrainConfig
from .mlpppo import MLPPPOBrain, MLPPPOBrainConfig, PPOBrain, PPOBrainConfig
from .mlpreinforce import MLPBrain, MLPBrainConfig, MLPReinforceBrain, MLPReinforceBrainConfig
from .spikingreinforce import (
    SpikingBrain,
    SpikingBrainConfig,
    SpikingReinforceBrain,
    SpikingReinforceBrainConfig,
)

__all__ = [
    "Brain",
    "BrainData",
    "BrainParams",
    "ClassicalBrain",
    "HybridClassicalBrain",
    "HybridClassicalBrainConfig",
    "MLPBrain",
    "MLPBrainConfig",
    "MLPDQNBrain",
    "MLPDQNBrainConfig",
    "MLPPPOBrain",
    "MLPPPOBrainConfig",
    "MLPReinforceBrain",
    "MLPReinforceBrainConfig",
    "PPOBrain",
    "PPOBrainConfig",
    "SpikingBrain",
    "SpikingBrainConfig",
    "SpikingReinforceBrain",
    "SpikingReinforceBrainConfig",
]
