"""Linear-Gaussian cable world and online sensorimotor-influence experiment."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from .config import CouplingMode, ExperimentMode, PhaseConfig, StudyConfig
from .predictors import OnlineLinearGaussian, predictor_features

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
BehaviorPolicy = Literal["learned", "ungated", "oracle"]
NEGATIVE_PROBE_PROBABILITY = 0.5


@dataclass(slots=True)
class SimulationTrace:
    """Complete transition-level record from one online run."""

    seed: int
    experiment_mode: str
    phase_names: tuple[str, ...]
    phase_couplings: tuple[str, ...]
    time: IntArray
    phase_index: IntArray
    state: FloatArray
    action: FloatArray
    yoked_action: FloatArray
    applied_drive: FloatArray
    process_noise: FloatArray
    coupling_coefficient: FloatArray
    controller_command: FloatArray
    vigor: FloatArray
    is_probe: BoolArray
    is_calibration: BoolArray
    true_information: FloatArray
    prediction_blind: FloatArray
    prediction_aware: FloatArray
    error_blind: FloatArray
    error_aware: FloatArray
    squared_error_blind: FloatArray
    squared_error_aware: FloatArray
    loss_blind: FloatArray
    loss_aware: FloatArray
    loss_dummy: FloatArray
    loss_shuffled: FloatArray
    loss_wrong_lag: FloatArray
    variance_blind: FloatArray
    variance_aware: FloatArray
    learned_action_weight: FloatArray
    evidence: FloatArray
    clipped_evidence: FloatArray
    influence: FloatArray
    influence_dummy: FloatArray
    influence_shuffled: FloatArray
    influence_wrong_lag: FloatArray

    def save(self, path: Path) -> None:
        """Save the trace as a compressed, non-pickle NumPy archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        values: dict[str, np.ndarray] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, tuple):
                values[field.name] = np.asarray(value, dtype=np.str_)
            else:
                values[field.name] = np.asarray(value)
        np.savez_compressed(path, **values)  # pyright: ignore[reportArgumentType]


def analytic_conditional_information(
    action_gain: float,
    action_std: float,
    process_noise_std: float,
) -> float:
    """Return ``I(A_t; X_(t+1) | X_t)`` in nats for the toy system."""
    if action_std < 0.0:
        message = "action_std must be non-negative"
        raise ValueError(message)
    if process_noise_std <= 0.0:
        message = "process_noise_std must be positive"
        raise ValueError(message)
    signal_to_noise = (action_gain * action_std / process_noise_std) ** 2
    return 0.5 * float(np.log1p(signal_to_noise))


def run_simulation(  # noqa: PLR0915
    config: StudyConfig,
    seed: int,
    *,
    behavior_policy: BehaviorPolicy = "learned",
    probe_probability: float | None = None,
    probe_probabilities_by_phase: Sequence[float] | None = None,
) -> SimulationTrace:
    """Run one prequential experiment with independent random-number streams.

    Random streams are keyed by ``(seed, stream id)``.  Consequently, changing
    the coupling schedule preserves the agent-action, yoked-action, process-noise,
    dummy-input, and probe draws used by a paired condition.
    """
    if behavior_policy not in {"learned", "ungated", "oracle"}:
        message = f"unknown behavior_policy: {behavior_policy}"
        raise ValueError(message)
    phase_probe_probabilities = _resolve_probe_probabilities(
        config,
        probe_probability,
        probe_probabilities_by_phase,
    )

    total_steps = config.total_steps
    environment = config.environment
    estimator = config.estimator
    gate = config.gate

    agent_rng = _rng(seed, 0)
    yoked_rng = _rng(seed, 1)
    noise_rng = _rng(seed, 2)
    dummy_rng = _rng(seed, 3)
    shuffle_rng = _rng(seed, 4)
    probe_rng = _rng(seed, 5)

    external_actions = agent_rng.normal(0.0, environment.action_std, total_steps)
    yoked_actions = yoked_rng.normal(0.0, environment.action_std, total_steps)
    standard_noise = noise_rng.normal(0.0, 1.0, total_steps)
    dummy_inputs = dummy_rng.normal(0.0, environment.action_std, total_steps)
    probe_draws = probe_rng.random(total_steps)
    probe_signs = np.where(
        probe_rng.random(total_steps) < NEGATIVE_PROBE_PROBABILITY,
        -1.0,
        1.0,
    )

    phase_index, step_phases = _expand_phases(config.phases)
    shuffled_inputs = np.empty_like(external_actions)
    for phase_number in range(len(config.phases)):
        mask = phase_index == phase_number
        shuffled_inputs[mask] = shuffle_rng.permutation(external_actions[mask])
    time = np.arange(total_steps, dtype=np.int64)
    state = np.empty(total_steps + 1, dtype=np.float64)
    state[0] = environment.initial_state

    float_names = (
        "action",
        "yoked_action",
        "applied_drive",
        "process_noise",
        "coupling_coefficient",
        "controller_command",
        "vigor",
        "true_information",
        "prediction_blind",
        "prediction_aware",
        "error_blind",
        "error_aware",
        "squared_error_blind",
        "squared_error_aware",
        "loss_blind",
        "loss_aware",
        "loss_dummy",
        "loss_shuffled",
        "loss_wrong_lag",
        "variance_blind",
        "variance_aware",
        "learned_action_weight",
        "evidence",
        "clipped_evidence",
        "influence",
        "influence_dummy",
        "influence_shuffled",
        "influence_wrong_lag",
    )
    data = {name: np.empty(total_steps, dtype=np.float64) for name in float_names}
    is_probe = np.zeros(total_steps, dtype=np.bool_)
    is_calibration = np.zeros(total_steps, dtype=np.bool_)

    predictors = {
        "blind": OnlineLinearGaussian(3, config.learner),
        "aware": OnlineLinearGaussian(3, config.learner),
        "dummy": OnlineLinearGaussian(3, config.learner),
        "shuffled": OnlineLinearGaussian(3, config.learner),
        "wrong_lag": OnlineLinearGaussian(3, config.learner),
    }
    influence = float(estimator.initial_influence)
    null_influences = {name: float(estimator.initial_influence) for name in predictors}
    previous_action = 0.0

    for step, phase in enumerate(step_phases):
        current_state = float(state[step])
        controller_command = gate.controller_gain * (gate.target_state - current_state)
        current_vigor = _select_vigor(
            influence,
            phase.coupling,
            config,
            behavior_policy,
        )

        if config.mode is ExperimentMode.ESTIMATOR:
            current_action = float(external_actions[step])
            current_vigor = 1.0
        elif step < gate.calibration_steps:
            current_action = float(external_actions[step])
            current_vigor = 1.0
            is_calibration[step] = True
            is_probe[step] = True
        else:
            phase_probe_probability = phase_probe_probabilities[int(phase_index[step])]
            probe_now = gate.probes_enabled and probe_draws[step] < phase_probe_probability
            if probe_now:
                current_action = float(gate.probe_amplitude * probe_signs[step])
                is_probe[step] = True
            else:
                current_action = float(current_vigor * controller_command)
            current_action = float(
                np.clip(current_action, -gate.action_clip, gate.action_clip),
            )

        action_source, own_action_gain = _action_source(
            phase,
            current_action,
            float(yoked_actions[step]),
            environment.coupling_gain,
        )
        additive_drive = environment.coupling_gain * phase.gain_scale * action_source
        process_noise = environment.process_noise_std * phase.noise_scale * standard_noise[step]
        next_state = environment.rho * current_state + additive_drive + process_noise
        state[step + 1] = next_state

        auxiliary_inputs = {
            "blind": 0.0,
            "aware": current_action,
            "dummy": float(dummy_inputs[step]),
            "shuffled": float(shuffled_inputs[step]),
            "wrong_lag": previous_action,
        }
        features = {
            name: predictor_features(current_state, auxiliary)
            for name, auxiliary in auxiliary_inputs.items()
        }
        predictions = {
            name: predictor.score(features[name], float(next_state))
            for name, predictor in predictors.items()
        }

        raw_evidence = (
            predictions["blind"].negative_log_likelihood
            - predictions["aware"].negative_log_likelihood
        )
        clipped_evidence = float(
            np.clip(raw_evidence, -estimator.evidence_clip, estimator.evidence_clip),
        )
        influence = _smooth(
            influence,
            clipped_evidence,
            estimator.smoothing_rate,
        )
        for name in ("dummy", "shuffled", "wrong_lag"):
            null_evidence = (
                predictions["blind"].negative_log_likelihood
                - predictions[name].negative_log_likelihood
            )
            null_influences[name] = _smooth(
                null_influences[name],
                float(
                    np.clip(
                        null_evidence,
                        -estimator.evidence_clip,
                        estimator.evidence_clip,
                    ),
                ),
                estimator.smoothing_rate,
            )

        # The current transition becomes training data only after all scores exist.
        for name, predictor in predictors.items():
            predictor.update(features[name], predictions[name])

        data["action"][step] = current_action
        data["yoked_action"][step] = yoked_actions[step]
        data["applied_drive"][step] = additive_drive
        data["process_noise"][step] = process_noise
        data["coupling_coefficient"][step] = own_action_gain * phase.gain_scale
        data["controller_command"][step] = controller_command
        data["vigor"][step] = current_vigor
        data["true_information"][step] = _true_information(phase, config)
        data["prediction_blind"][step] = predictions["blind"].mean
        data["prediction_aware"][step] = predictions["aware"].mean
        data["error_blind"][step] = predictions["blind"].error
        data["error_aware"][step] = predictions["aware"].error
        data["squared_error_blind"][step] = predictions["blind"].squared_error
        data["squared_error_aware"][step] = predictions["aware"].squared_error
        data["loss_blind"][step] = predictions["blind"].negative_log_likelihood
        data["loss_aware"][step] = predictions["aware"].negative_log_likelihood
        data["loss_dummy"][step] = predictions["dummy"].negative_log_likelihood
        data["loss_shuffled"][step] = predictions["shuffled"].negative_log_likelihood
        data["loss_wrong_lag"][step] = predictions["wrong_lag"].negative_log_likelihood
        data["variance_blind"][step] = predictions["blind"].variance
        data["variance_aware"][step] = predictions["aware"].variance
        data["learned_action_weight"][step] = predictors["aware"].weights[2]
        data["evidence"][step] = raw_evidence
        data["clipped_evidence"][step] = clipped_evidence
        data["influence"][step] = influence
        data["influence_dummy"][step] = null_influences["dummy"]
        data["influence_shuffled"][step] = null_influences["shuffled"]
        data["influence_wrong_lag"][step] = null_influences["wrong_lag"]
        previous_action = current_action

    return SimulationTrace(
        seed=seed,
        experiment_mode=config.mode.value,
        phase_names=tuple(phase.name for phase in config.phases),
        phase_couplings=tuple(phase.coupling.value for phase in config.phases),
        time=time,
        phase_index=phase_index,
        state=state,
        is_probe=is_probe,
        is_calibration=is_calibration,
        **data,
    )


def _rng(seed: int, stream_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream_id]))


def _resolve_probe_probabilities(
    config: StudyConfig,
    probe_probability: float | None,
    probe_probabilities_by_phase: Sequence[float] | None,
) -> tuple[float, ...]:
    if probe_probability is not None and probe_probabilities_by_phase is not None:
        message = "specify either probe_probability or probe_probabilities_by_phase, not both"
        raise ValueError(message)
    if probe_probabilities_by_phase is None:
        probability = (
            config.gate.probe_probability if probe_probability is None else probe_probability
        )
        probabilities = (float(probability),) * len(config.phases)
    else:
        if len(probe_probabilities_by_phase) != len(config.phases):
            message = "probe_probabilities_by_phase must match the number of phases"
            raise ValueError(message)
        probabilities = tuple(float(value) for value in probe_probabilities_by_phase)
    if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        message = "every probe probability must lie in [0, 1]"
        raise ValueError(message)
    return probabilities


def _expand_phases(phases: tuple[PhaseConfig, ...]) -> tuple[IntArray, list[PhaseConfig]]:
    indexes: list[int] = []
    expanded: list[PhaseConfig] = []
    for index, phase in enumerate(phases):
        indexes.extend([index] * phase.steps)
        expanded.extend([phase] * phase.steps)
    return np.asarray(indexes, dtype=np.int64), expanded


def _smooth(previous: float, observation: float, rate: float) -> float:
    return float((1.0 - rate) * previous + rate * observation)


def _logistic(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)


def _select_vigor(
    influence: float,
    coupling: CouplingMode,
    config: StudyConfig,
    behavior_policy: BehaviorPolicy,
) -> float:
    gate = config.gate
    if config.mode is ExperimentMode.ESTIMATOR or not gate.enabled:
        return 1.0
    if behavior_policy == "ungated":
        return 1.0
    if behavior_policy == "oracle":
        return 1.0 if coupling in {CouplingMode.CONNECTED, CouplingMode.REVERSED} else 0.0
    active_fraction = _logistic(
        gate.logistic_gain * (influence - gate.influence_threshold),
    )
    return float(gate.min_vigor + (1.0 - gate.min_vigor) * active_fraction)


def _action_source(
    phase: PhaseConfig,
    agent_action: float,
    yoked_action: float,
    coupling_gain: float,
) -> tuple[float, float]:
    if phase.coupling is CouplingMode.CONNECTED:
        return agent_action, coupling_gain
    if phase.coupling is CouplingMode.DISCONNECTED:
        return 0.0, 0.0
    if phase.coupling is CouplingMode.YOKED:
        return yoked_action, 0.0
    if phase.coupling is CouplingMode.REVERSED:
        return -agent_action, -coupling_gain
    message = f"unhandled coupling: {phase.coupling}"
    raise AssertionError(message)


def _true_information(phase: PhaseConfig, config: StudyConfig) -> float:
    if config.mode is not ExperimentMode.ESTIMATOR:
        return float("nan")
    if phase.coupling not in {CouplingMode.CONNECTED, CouplingMode.REVERSED}:
        return 0.0
    environment = config.environment
    return analytic_conditional_information(
        action_gain=environment.coupling_gain * phase.gain_scale,
        action_std=environment.action_std,
        process_noise_std=environment.process_noise_std * phase.noise_scale,
    )
