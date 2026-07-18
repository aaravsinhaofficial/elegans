"""Seed-level metrics and uncertainty summaries for the toy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .simulation import SimulationTrace

FloatArray = NDArray[np.float64]
LiteralDirection = Literal["above", "below"]
LiteralSeries = Literal["influence", "vigor"]
NUM_SCHEDULE_PHASES = 3
MIN_CORRELATION_SAMPLES = 2
NUMERICAL_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Point estimate and percentile-bootstrap confidence interval."""

    estimate: float
    low: float
    high: float
    n: int


def rolling_mean(values: FloatArray, window: int) -> FloatArray:
    """Return a trailing mean with a shortened window at the left edge."""
    if window < 1:
        message = "window must be positive"
        raise ValueError(message)
    array = np.asarray(values, dtype=np.float64)
    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    indexes = np.arange(array.size)
    starts = np.maximum(0, indexes - window + 1)
    counts = indexes - starts + 1
    return (cumulative[indexes + 1] - cumulative[starts]) / counts


def phase_mask(trace: SimulationTrace, phase: int, burn_in: int = 0) -> NDArray[np.bool_]:
    """Select one phase after dropping its first ``burn_in`` transitions."""
    indexes = np.flatnonzero(trace.phase_index == phase)
    if indexes.size == 0:
        message = f"trace has no phase index {phase}"
        raise ValueError(message)
    if burn_in >= indexes.size:
        message = "burn_in must be shorter than the selected phase"
        raise ValueError(message)
    mask = trace.phase_index == phase
    if burn_in:
        mask[indexes[:burn_in]] = False
    return mask


def phase_metric_rows(
    traces: Sequence[SimulationTrace],
    *,
    burn_in: int,
    target_state: float = 1.0,
    condition: str,
    variance_floor: float = 1e-4,
) -> list[dict[str, float | int | str]]:
    """Reduce transition traces to one independent row per seed and phase."""
    rows: list[dict[str, float | int | str]] = []
    for trace in traces:
        for phase in range(len(trace.phase_names)):
            mask = phase_mask(trace, phase, burn_in)
            states_now = trace.state[:-1][mask]
            states_next = trace.state[1:][mask]
            actions = trace.action[mask]
            rows.append(
                {
                    "condition": condition,
                    "seed": trace.seed,
                    "phase_index": phase,
                    "phase_name": trace.phase_names[phase],
                    "coupling": trace.phase_couplings[phase],
                    "n_transitions": int(np.count_nonzero(mask)),
                    "mean_influence": float(np.mean(trace.influence[mask])),
                    "mean_true_information": _finite_mean(trace.true_information[mask]),
                    "mean_blind_nll": float(np.mean(trace.loss_blind[mask])),
                    "median_blind_nll": float(np.median(trace.loss_blind[mask])),
                    "mean_aware_nll": float(np.mean(trace.loss_aware[mask])),
                    "median_aware_nll": float(np.median(trace.loss_aware[mask])),
                    "mean_loss_advantage": float(
                        np.mean(trace.loss_blind[mask] - trace.loss_aware[mask]),
                    ),
                    "median_loss_advantage": float(np.median(trace.evidence[mask])),
                    "mean_dummy_advantage": float(
                        np.mean(trace.loss_blind[mask] - trace.loss_dummy[mask]),
                    ),
                    "mean_shuffled_advantage": float(
                        np.mean(trace.loss_blind[mask] - trace.loss_shuffled[mask]),
                    ),
                    "mean_wrong_lag_advantage": float(
                        np.mean(trace.loss_blind[mask] - trace.loss_wrong_lag[mask]),
                    ),
                    "mean_dummy_influence": float(np.mean(trace.influence_dummy[mask])),
                    "mean_shuffled_influence": float(
                        np.mean(trace.influence_shuffled[mask]),
                    ),
                    "mean_wrong_lag_influence": float(
                        np.mean(trace.influence_wrong_lag[mask]),
                    ),
                    "mean_blind_squared_error": float(
                        np.mean(trace.squared_error_blind[mask]),
                    ),
                    "mean_aware_squared_error": float(
                        np.mean(trace.squared_error_aware[mask]),
                    ),
                    "evidence_clipped_fraction": float(
                        np.mean(
                            ~np.isclose(
                                trace.evidence[mask],
                                trace.clipped_evidence[mask],
                                rtol=0.0,
                                atol=1e-12,
                            ),
                        ),
                    ),
                    "variance_floor_fraction": float(
                        np.mean(
                            (trace.variance_blind[mask] <= variance_floor * 1.001)
                            | (trace.variance_aware[mask] <= variance_floor * 1.001),
                        ),
                    ),
                    "sensory_variance": float(np.var(states_next, ddof=1)),
                    "action_variance": float(np.var(actions, ddof=1)),
                    "action_energy": float(np.sum(actions * actions)),
                    "mean_action_energy": float(np.mean(actions * actions)),
                    "probe_action_energy": float(
                        np.sum(actions[trace.is_probe[mask]] ** 2),
                    ),
                    "nonprobe_action_energy": float(
                        np.sum(actions[~trace.is_probe[mask]] ** 2),
                    ),
                    "mean_vigor": float(np.mean(trace.vigor[mask])),
                    "mean_homeostatic_squared_error": float(
                        np.mean((states_next - target_state) ** 2),
                    ),
                    "probe_fraction": float(np.mean(trace.is_probe[mask])),
                    "action_outcome_correlation": _correlation(actions, states_next),
                    "partial_action_outcome_correlation": partial_correlation(
                        actions,
                        states_next,
                        states_now,
                    ),
                },
            )
    return rows


def partial_correlation(actions: FloatArray, outcomes: FloatArray, states: FloatArray) -> float:
    """Correlation of action and outcome residuals after linear state control."""
    actions_array = np.asarray(actions, dtype=np.float64)
    outcomes_array = np.asarray(outcomes, dtype=np.float64)
    states_array = np.asarray(states, dtype=np.float64)
    if not (
        actions_array.shape == outcomes_array.shape == states_array.shape
        and actions_array.ndim == 1
    ):
        message = "actions, outcomes, and states must be same-length vectors"
        raise ValueError(message)
    design = np.column_stack((states_array, np.ones(states_array.size)))
    action_fit, *_ = np.linalg.lstsq(design, actions_array, rcond=None)
    outcome_fit, *_ = np.linalg.lstsq(design, outcomes_array, rcond=None)
    action_residual = actions_array - design @ action_fit
    outcome_residual = outcomes_array - design @ outcome_fit
    return _correlation(action_residual, outcome_residual)


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 91_731,
) -> ConfidenceInterval:
    """Percentile-bootstrap interval, resampling independent seeds."""
    data = _finite_array(values)
    if replicates < 1:
        message = "replicates must be positive"
        raise ValueError(message)
    if not 0.0 < confidence_level < 1.0:
        message = "confidence_level must lie in (0, 1)"
        raise ValueError(message)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, data.size, size=(replicates, data.size))
    bootstrap_means = np.mean(data[indexes], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return ConfidenceInterval(
        estimate=float(np.mean(data)),
        low=float(low),
        high=float(high),
        n=int(data.size),
    )


def bootstrap_paired_difference(
    first: Iterable[float],
    second: Iterable[float],
    *,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 91_731,
) -> ConfidenceInterval:
    """Bootstrap the mean paired difference ``first - second``."""
    first_array = np.asarray(tuple(first), dtype=np.float64)
    second_array = np.asarray(tuple(second), dtype=np.float64)
    if first_array.shape != second_array.shape:
        message = "paired samples must have the same shape"
        raise ValueError(message)
    return bootstrap_mean_interval(
        first_array - second_array,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
    )


def roc_auc(positive: Iterable[float], negative: Iterable[float]) -> float:
    """Compute the probability that a positive score exceeds a negative score."""
    positive_array = _finite_array(positive)
    negative_array = _finite_array(negative)
    combined = np.concatenate((positive_array, negative_array))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    start = 0
    while start < combined.size:
        stop = start + 1
        while stop < combined.size and combined[order[stop]] == combined[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(np.sum(ranks[: positive_array.size]))
    mann_whitney = positive_rank_sum - 0.5 * positive_array.size * (positive_array.size + 1)
    return mann_whitney / (positive_array.size * negative_array.size)


def sustained_crossing_latency(  # noqa: PLR0913
    values: FloatArray,
    *,
    start: int,
    stop: int,
    threshold: float,
    direction: LiteralDirection,
    consecutive: int = 20,
) -> float:
    """Online confirmation time of the first sustained crossing, else NaN.

    A run that begins at sample ``i`` is only knowable after observing sample
    ``i + consecutive - 1``.  Returning the confirmation sample keeps latency
    causal rather than backdating it to the beginning of the qualifying run.
    """
    if not 0 <= start < stop <= len(values):
        message = "invalid [start, stop) interval"
        raise ValueError(message)
    if consecutive < 1:
        message = "consecutive must be positive"
        raise ValueError(message)
    segment = np.asarray(values[start:stop], dtype=np.float64)
    if direction == "above":
        criterion = segment >= threshold
    elif direction == "below":
        criterion = segment <= threshold
    else:
        message = "direction must be 'above' or 'below'"
        raise ValueError(message)
    if criterion.size < consecutive:
        return float("nan")
    runs = np.convolve(criterion.astype(np.int64), np.ones(consecutive, dtype=np.int64), "valid")
    hits = np.flatnonzero(runs == consecutive)
    return float(hits[0] + consecutive - 1) if hits.size else float("nan")


def transition_latencies(
    trace: SimulationTrace,
    *,
    series: LiteralSeries = "influence",
    off_threshold: float | None = None,
    on_threshold: float | None = None,
    consecutive: int = 10,
) -> tuple[float, float, float, float]:
    """Return detection/recovery latency and the frozen off/on thresholds."""
    if len(trace.phase_names) != NUM_SCHEDULE_PHASES:
        message = "transition latency requires a three-phase schedule"
        raise ValueError(message)
    values = np.asarray(getattr(trace, series), dtype=np.float64)
    first_end = int(np.flatnonzero(trace.phase_index == 0)[-1]) + 1
    middle_end = int(np.flatnonzero(trace.phase_index == 1)[-1]) + 1
    baseline_start = max(0, first_end - min(300, first_end // 2))
    baseline = float(np.mean(values[baseline_start:first_end]))
    if off_threshold is None:
        off_threshold = 0.25 * baseline if series == "influence" else 0.25
    if on_threshold is None:
        on_threshold = 0.75 * baseline if series == "influence" else 0.75
    detection = sustained_crossing_latency(
        values,
        start=first_end,
        stop=middle_end,
        threshold=off_threshold,
        direction="below",
        consecutive=consecutive,
    )
    recovery = sustained_crossing_latency(
        values,
        start=middle_end,
        stop=len(values),
        threshold=on_threshold,
        direction="above",
        consecutive=consecutive,
    )
    return detection, recovery, off_threshold, on_threshold


def state_classification_error_rates(
    trace: SimulationTrace,
    *,
    off_threshold: float,
    on_threshold: float,
    burn_in: int,
) -> tuple[float, float]:
    """Return false-active and false-passive fractions under hysteresis."""
    if off_threshold >= on_threshold:
        message = "off_threshold must be below on_threshold"
        raise ValueError(message)
    active = True
    classifications = np.empty(trace.influence.size, dtype=np.bool_)
    for index, value in enumerate(trace.influence):
        if value >= on_threshold:
            active = True
        elif value <= off_threshold:
            active = False
        classifications[index] = active
    connected_mask = phase_mask(trace, 0, burn_in) | phase_mask(trace, 2, burn_in)
    disconnected_mask = phase_mask(trace, 1, burn_in)
    false_active = float(np.mean(classifications[disconnected_mask]))
    false_passive = float(np.mean(~classifications[connected_mask]))
    return false_active, false_passive


def _correlation(first: FloatArray, second: FloatArray) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.size < MIN_CORRELATION_SAMPLES:
        return float("nan")
    if np.std(first_array) <= NUMERICAL_EPSILON or np.std(second_array) <= NUMERICAL_EPSILON:
        return float("nan")
    return float(np.corrcoef(first_array, second_array)[0, 1])


def _finite_array(values: Iterable[float]) -> FloatArray:
    data = np.asarray(tuple(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        message = "at least one finite value is required"
        raise ValueError(message)
    return data


def _finite_mean(values: FloatArray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")
