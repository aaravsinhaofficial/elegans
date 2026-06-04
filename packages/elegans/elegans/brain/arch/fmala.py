import numpy as np
import torch
from torch import nn

from elegans.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from elegans.brain.arch import BrainData, BrainParams, ClassicalBrain


DIRECTION_MAP = {
    Direction.UP: np.pi / 2,
    Direction.DOWN: -np.pi / 2,
    Direction.LEFT: np.pi,
    Direction.RIGHT: 0.0,
}

def relative_angle(heading, angle):
    return (angle - heading + np.pi) % (2 * np.pi) - np.pi

class FMalaBrainConfig(BrainConfig):
    pass

class FMalaBrain(ClassicalBrain):
    def __init__(self, config: FMalaBrainConfig, input_dim: int,
                 num_actions: int, device: DeviceType = DeviceType.CPU,
                 action_set: list[Action] = DEFAULT_ACTIONS) -> None:
        super().__init__()

        # Initialize seeding for reproducibility
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)  # Set global numpy/torch seeds
        logger.info(f"FMalaBrain using seed: {self.seed}")

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """
        Preprocess BrainParams to cast them into a uniform set of parameters
        for a circular distribution over new directions.

        Parameters
        ----------
            params: BrainParams containing interoceptive state, gradient
                    strength, and direction.

        Returns
        -------
            (mu, kappa): Parameters for a von Mises proposal over a new
                         direction and a Forward MALA acceptance ratio
        """
        heading_angle = DIRECTION_MAP.get(
            params.agent_direction or Direction.UP, np.pi / 2
        )

        # Satiety serves as a temperature, not an inverse-temperature/precision.
        motive = torch.polar(params.food_gradient_strength / params.satiety,
                             params.food_gradient_direction)

        # torch.polar gives me complex numbers equivalent to allocentric vectors
        health_temperature = params.health / params.max_health
        motive = motive - torch.polar(
            params.predator_gradient_strength * health_temperature,
            params.predator_gradient_direction
        )

        # Too hot (cool down) or too cold (warm up)?
        thermal_strain = params.temperature - params.cultivation_temperature
        motive = motive - torch.polar(
            thermal_strain * params.temperature_gradient_strength,
            params.temperature_gradient_direction
        )

        # After adding up in Euclidean, allocentric space, convert back to
        # egocentric polar coordinates.
        strength = torch.hypot(motive.real, motive.imag)
        angle = torch.atan2(motive.imag, motive.real)
        egocentric_angle = relative_angle(heading_angle, angle)

        return (egocentric_angle, 1 / strength)
