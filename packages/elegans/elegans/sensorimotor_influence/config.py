"""Validated configuration for toy sensorimotor-influence experiments.

The configuration layer deliberately contains no simulation logic.  It fixes the
meaning of each experimental condition so estimators, behavioral experiments,
controls, and analyses can share one immutable description of a run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CouplingMode(StrEnum):
    """How a phase's sensory dynamics are driven by action."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    YOKED = "yoked"
    REVERSED = "reversed"


class ExperimentMode(StrEnum):
    """Source of the agent's commanded action."""

    ESTIMATOR = "estimator"
    GATED = "gated"


class VarianceMode(StrEnum):
    """Treatment of predictive variance in the Gaussian learners."""

    ADAPTIVE = "adaptive"
    FIXED = "fixed"


class _FrozenConfig(BaseModel):
    """Common strict, immutable behavior for experiment configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class PhaseConfig(_FrozenConfig):
    """One contiguous segment of an action-coupling schedule.

    ``gain_scale`` is a nonnegative magnitude.  ``REVERSED`` supplies the minus
    sign, while ``YOKED`` applies the same magnitude to an independent action.
    This keeps connected and yoked sensory variance matched by construction.
    """

    name: str = Field(min_length=1)
    steps: int = Field(default=1_500, gt=0, strict=True)
    coupling: CouplingMode = CouplingMode.CONNECTED
    gain_scale: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    noise_scale: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)

    @field_validator("name")
    @classmethod
    def _name_must_not_be_blank(cls, value: str) -> str:
        name = value.strip()
        if not name:
            message = "name must contain a non-whitespace character"
            raise ValueError(message)
        return name

    @property
    def own_action_multiplier(self) -> float:
        """Signed multiplier applied to the agent's own action."""
        if self.coupling is CouplingMode.CONNECTED:
            return self.gain_scale
        if self.coupling is CouplingMode.REVERSED:
            return -self.gain_scale
        return 0.0

    @property
    def yoked_action_multiplier(self) -> float:
        """Multiplier applied to an independently sampled matched action."""
        return self.gain_scale if self.coupling is CouplingMode.YOKED else 0.0


class EnvironmentConfig(_FrozenConfig):
    """Parameters of the scalar linear-Gaussian toy environment."""

    rho: float = Field(default=0.8, gt=-1.0, lt=1.0, allow_inf_nan=False)
    coupling_gain: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    process_noise_std: float = Field(default=0.2, gt=0.0, allow_inf_nan=False)
    action_std: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    initial_state: float = Field(default=0.0, allow_inf_nan=False)


class LearnerConfig(_FrozenConfig):
    """Online linear-Gaussian predictor learning parameters."""

    mean_learning_rate: float = Field(default=0.01, gt=0.0, le=1.0, allow_inf_nan=False)
    variance_mode: VarianceMode = VarianceMode.ADAPTIVE
    variance_learning_rate: float = Field(default=0.01, gt=0.0, le=1.0, allow_inf_nan=False)
    initial_variance: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    fixed_variance: float = Field(default=0.04, gt=0.0, allow_inf_nan=False)
    min_variance: float = Field(default=1e-4, gt=0.0, allow_inf_nan=False)
    max_variance: float = Field(default=100.0, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _variance_bounds_are_consistent(self) -> Self:
        if self.min_variance >= self.max_variance:
            message = "min_variance must be smaller than max_variance"
            raise ValueError(message)
        for name, value in (
            ("initial_variance", self.initial_variance),
            ("fixed_variance", self.fixed_variance),
        ):
            if not self.min_variance <= value <= self.max_variance:
                message = f"{name} must lie within [min_variance, max_variance]"
                raise ValueError(message)
        return self


class EstimatorConfig(_FrozenConfig):
    """Slow evidence accumulator for action-conditioned predictive advantage."""

    smoothing_rate: float = Field(default=0.02, gt=0.0, le=1.0, allow_inf_nan=False)
    evidence_clip: float = Field(default=5.0, gt=0.0, allow_inf_nan=False)
    initial_influence: float = Field(default=0.0, allow_inf_nan=False)


class GateConfig(_FrozenConfig):
    """Fixed homeostatic controller, vigor gate, and persistent probe settings."""

    enabled: bool = False
    controller_gain: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    target_state: float = Field(default=1.0, allow_inf_nan=False)
    min_vigor: float = Field(default=0.05, ge=0.0, le=1.0, allow_inf_nan=False)
    logistic_gain: float = Field(default=8.0, gt=0.0, allow_inf_nan=False)
    influence_threshold: float = Field(default=0.1, allow_inf_nan=False)
    calibration_steps: int = Field(default=300, ge=0, strict=True)
    action_clip: float = Field(default=3.0, gt=0.0, allow_inf_nan=False)
    probes_enabled: bool = True
    probe_probability: float = Field(default=0.03, ge=0.0, le=1.0, allow_inf_nan=False)
    probe_amplitude: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)


def disconnected_schedule(steps: int = 1_500) -> tuple[PhaseConfig, ...]:
    """Return the canonical connected-disconnected-connected schedule."""
    return (
        PhaseConfig(name="connected_initial", steps=steps),
        PhaseConfig(name="disconnected", steps=steps, coupling=CouplingMode.DISCONNECTED),
        PhaseConfig(name="connected_restored", steps=steps),
    )


def yoked_schedule(steps: int = 1_500) -> tuple[PhaseConfig, ...]:
    """Return the matched-variance connected-yoked-connected schedule."""
    return (
        PhaseConfig(name="connected_initial", steps=steps),
        PhaseConfig(name="yoked", steps=steps, coupling=CouplingMode.YOKED),
        PhaseConfig(name="connected_restored", steps=steps),
    )


class StudyConfig(_FrozenConfig):
    """Complete reproducible specification for a multi-seed study."""

    phases: tuple[PhaseConfig, ...] = Field(default_factory=disconnected_schedule, min_length=1)
    mode: ExperimentMode = ExperimentMode.ESTIMATOR
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    learner: LearnerConfig = Field(default_factory=LearnerConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    n_seeds: int = Field(default=30, gt=0, strict=True)
    base_seed: int = Field(default=20_250_101, ge=0, strict=True)
    bootstrap_replicates: int = Field(default=10_000, gt=0, strict=True)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0, allow_inf_nan=False)
    phase_metric_burn_in: int = Field(default=250, ge=0, strict=True)

    @model_validator(mode="after")
    def _phase_names_are_unique(self) -> Self:
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            message = "phase names must be unique"
            raise ValueError(message)
        return self

    @property
    def total_steps(self) -> int:
        """Total number of transitions in one run."""
        return sum(phase.steps for phase in self.phases)

    @property
    def seeds(self) -> tuple[int, ...]:
        """Deterministic seed sequence used for independent runs."""
        return tuple(range(self.base_seed, self.base_seed + self.n_seeds))

    @property
    def phase_boundaries(self) -> tuple[tuple[str, int, int], ...]:
        """Half-open ``(name, start, stop)`` intervals for every phase."""
        boundaries: list[tuple[str, int, int]] = []
        start = 0
        for phase in self.phases:
            stop = start + phase.steps
            boundaries.append((phase.name, start, stop))
            start = stop
        return tuple(boundaries)


__all__ = [
    "CouplingMode",
    "EnvironmentConfig",
    "EstimatorConfig",
    "ExperimentMode",
    "GateConfig",
    "LearnerConfig",
    "PhaseConfig",
    "StudyConfig",
    "VarianceMode",
    "disconnected_schedule",
    "yoked_schedule",
]
