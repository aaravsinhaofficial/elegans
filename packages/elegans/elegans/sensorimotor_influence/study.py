"""Multi-seed validation study for reward-free sensorimotor influence."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    ConfidenceInterval,
    bootstrap_mean_interval,
    bootstrap_paired_difference,
    phase_mask,
    phase_metric_rows,
    roc_auc,
    state_classification_error_rates,
    transition_latencies,
)
from .config import (
    CouplingMode,
    EnvironmentConfig,
    EstimatorConfig,
    ExperimentMode,
    GateConfig,
    LearnerConfig,
    PhaseConfig,
    StudyConfig,
    disconnected_schedule,
    yoked_schedule,
)
from .simulation import SimulationTrace, run_simulation

ProgressCallback = Callable[[str], None]
MINIMUM_SEEDS = 2
MINIMUM_PHASE_STEPS = 100
NUMERICAL_EPSILON = 1e-12
RECOVERY_PROBE_PROBABILITIES = (0.0, 0.03, 0.05)
ROBUSTNESS_GAINS = (0.1, 0.25, 0.5, 0.75, 1.0)
ROBUSTNESS_NOISES = (0.1, 0.2, 0.4, 0.8, 1.2)
LATENCY_CONSECUTIVE = 10
INFLUENCE_OFF_FRACTION = 0.25
INFLUENCE_ON_FRACTION = 0.75
VIGOR_OFF_THRESHOLD = 0.25
VIGOR_ON_THRESHOLD = 0.75
H7_MINIMUM_AUC_CORRELATION = 0.3
H7_MAXIMUM_LATENCY_CORRELATION = -0.2
REVERSAL_EQUIVALENCE_MARGIN = 0.163
IDENTIFIABILITY_NULL_MARGIN = 0.1


@dataclass(slots=True)
class StudyResults:
    """In-memory traces needed for plots plus all reduced result tables."""

    estimator_disconnected: list[SimulationTrace]
    estimator_yoked: list[SimulationTrace]
    estimator_reversed: list[SimulationTrace]
    behavior_by_probe: dict[float, list[SimulationTrace]]
    phase_rows: list[dict[str, float | int | str]]
    latency_rows: list[dict[str, float | int | str]]
    robustness_rows: list[dict[str, float | int]]
    inference_rows: list[dict[str, float | int | str]]
    hypothesis_results: dict[str, dict[str, Any]]
    configs: dict[str, StudyConfig]


def run_full_study(  # noqa: PLR0913
    *,
    output_directory: Path,
    n_seeds: int = 30,
    phase_steps: int = 1_500,
    base_seed: int = 751_000,
    bootstrap_replicates: int = 10_000,
    robustness_seeds: int | None = None,
    progress: ProgressCallback | None = None,
) -> StudyResults:
    """Run estimator, yoking, behavioral, control, and robustness experiments."""
    if n_seeds < MINIMUM_SEEDS:
        message = "n_seeds must be at least two for seed-level uncertainty"
        raise ValueError(message)
    if phase_steps < MINIMUM_PHASE_STEPS:
        message = "phase_steps must be at least 100"
        raise ValueError(message)
    robustness_n = n_seeds if robustness_seeds is None else robustness_seeds
    if robustness_n < MINIMUM_SEEDS:
        message = "robustness_seeds must be at least two"
        raise ValueError(message)

    output_directory.mkdir(parents=True, exist_ok=True)
    estimator_disconnected_config, estimator_yoked_config, behavior_config = _study_configs(
        n_seeds=n_seeds,
        phase_steps=phase_steps,
        base_seed=base_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    _notify(progress, "running estimator-only cable schedule")
    estimator_disconnected = _run_replicates(estimator_disconnected_config)
    _notify(progress, "running matched-variance yoked schedule")
    estimator_yoked = _run_replicates(estimator_yoked_config)

    # Match every arm through the entire initial connected phase.  The zero arm
    # then withdraws probes at disconnection, while the other arms retain them.
    # This isolates rediscovery after reconnection from pre-switch estimator decay.
    probe_probabilities = RECOVERY_PROBE_PROBABILITIES
    behavior_by_probe: dict[float, list[SimulationTrace]] = {}
    for probability in probe_probabilities:
        phase_probe_probabilities = (
            behavior_config.gate.probe_probability,
            probability,
            probability,
        )
        _notify(
            progress,
            f"running vigor gate with post-initial probe probability {probability:.2f}",
        )
        behavior_by_probe[probability] = _run_replicates(
            behavior_config,
            probe_probabilities_by_phase=phase_probe_probabilities,
        )

    phase_rows: list[dict[str, float | int | str]] = []
    phase_rows.extend(
        phase_metric_rows(
            estimator_disconnected,
            burn_in=estimator_disconnected_config.phase_metric_burn_in,
            condition="estimator_disconnected",
        ),
    )
    phase_rows.extend(
        phase_metric_rows(
            estimator_yoked,
            burn_in=estimator_yoked_config.phase_metric_burn_in,
            condition="estimator_yoked",
        ),
    )
    for probability, traces in behavior_by_probe.items():
        phase_rows.extend(
            phase_metric_rows(
                traces,
                burn_in=behavior_config.phase_metric_burn_in,
                target_state=behavior_config.gate.target_state,
                condition=f"behavior_probe_{probability:.2f}",
            ),
        )

    _notify(progress, "running reversed, deterministic, yoked-behavior, and oracle controls")
    control_rows, control_extras, reversed_traces, control_configs = _run_controls(
        estimator_disconnected_config,
        behavior_config,
    )
    phase_rows.extend(control_rows)

    latency_rows = _latency_rows(
        estimator_disconnected,
        estimator_yoked,
        behavior_by_probe,
        estimator_disconnected_config,
        estimator_yoked_config,
        behavior_config,
    )
    _notify(progress, "running gain-by-noise robustness grid")
    robustness_config = estimator_yoked_config.model_copy(
        update={
            "n_seeds": robustness_n,
            "base_seed": base_seed + 10_000,
        },
    )
    robustness_rows = _run_robustness_grid(
        robustness_config,
        progress=progress,
    )

    inference_rows, hypothesis_results = _summarize_hypotheses(
        phase_rows=phase_rows,
        latency_rows=latency_rows,
        robustness_rows=robustness_rows,
        control_extras=control_extras,
        bootstrap_replicates=bootstrap_replicates,
        phase_steps=phase_steps,
    )
    configs = {
        "estimator_disconnected": estimator_disconnected_config,
        "estimator_yoked": estimator_yoked_config,
        "behavior": behavior_config,
        "robustness_template": robustness_config,
        **control_configs,
    }
    results = StudyResults(
        estimator_disconnected=estimator_disconnected,
        estimator_yoked=estimator_yoked,
        estimator_reversed=reversed_traces,
        behavior_by_probe=behavior_by_probe,
        phase_rows=phase_rows,
        latency_rows=latency_rows,
        robustness_rows=robustness_rows,
        inference_rows=inference_rows,
        hypothesis_results=hypothesis_results,
        configs=configs,
    )
    write_study_results(results, output_directory)
    return results


def write_study_results(results: StudyResults, output_directory: Path) -> None:
    """Persist reduced results, configurations, and representative raw traces."""
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "phase_metrics.csv", results.phase_rows)
    _write_csv(output_directory / "latencies.csv", results.latency_rows)
    _write_csv(output_directory / "robustness.csv", results.robustness_rows)
    _write_csv(output_directory / "inference.csv", results.inference_rows)
    _write_csv(
        output_directory / "control_summary.csv",
        _control_summary_rows(results.phase_rows),
    )
    configuration = {
        name: config.model_dump(mode="json") for name, config in results.configs.items()
    }
    (output_directory / "configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    canonical_configuration = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    configuration_hash = hashlib.sha256(canonical_configuration.encode()).hexdigest()
    execution_protocol = _execution_protocol(results)
    canonical_protocol = json.dumps(
        {"configuration": configuration, "execution_protocol": execution_protocol},
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest = {
        "configuration_sha256": configuration_hash,
        "protocol_sha256": hashlib.sha256(canonical_protocol.encode()).hexdigest(),
        "source_sha256": _source_sha256(),
        "execution_protocol": execution_protocol,
        "test_seeds": {name: list(config.seeds) for name, config in results.configs.items()},
        "independent_unit": "seed",
        "bootstrap_unit": "seed",
        "software_versions": {
            "python": platform.python_version(),
            "numpy": _dependency_version("numpy"),
            "pydantic": _dependency_version("pydantic"),
            "matplotlib": _dependency_version("matplotlib"),
        },
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "hypothesis_results.json").write_text(
        json.dumps(_json_safe(results.hypothesis_results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_directory = output_directory / "representative_traces"
    results.estimator_disconnected[0].save(raw_directory / "estimator_disconnected.npz")
    results.estimator_yoked[0].save(raw_directory / "estimator_yoked.npz")
    results.estimator_reversed[0].save(raw_directory / "estimator_reversed.npz")
    for probability, traces in results.behavior_by_probe.items():
        label = f"{probability:.2f}".replace(".", "p")
        traces[0].save(raw_directory / f"behavior_probe_{label}.npz")


def _execution_protocol(results: StudyResults) -> dict[str, Any]:
    behavior = results.configs["behavior"]
    initial_probe_probability = behavior.gate.probe_probability
    return {
        "random_stream_pairing": {
            "enabled": True,
            "stream_ids": {
                "focal_action": 0,
                "yoked_action": 1,
                "process_noise": 2,
                "dummy_input": 3,
                "within_phase_shuffle": 4,
                "probe_occurrence_and_sign": 5,
            },
        },
        "behavior_probe_probabilities_by_phase": {
            f"post_initial_{probability:.2f}": [
                initial_probe_probability,
                probability,
                probability,
            ]
            for probability in RECOVERY_PROBE_PROBABILITIES
        },
        "robustness_grid": {
            "gains": list(ROBUSTNESS_GAINS),
            "noise_standard_deviations": list(ROBUSTNESS_NOISES),
        },
        "latency": {
            "reported_sample": "online confirmation sample",
            "consecutive_samples": LATENCY_CONSECUTIVE,
            "influence_off_fraction_of_analytic_reference": INFLUENCE_OFF_FRACTION,
            "influence_on_fraction_of_analytic_reference": INFLUENCE_ON_FRACTION,
            "vigor_off_threshold": VIGOR_OFF_THRESHOLD,
            "vigor_on_threshold": VIGOR_ON_THRESHOLD,
            "censor_value": "phase horizon",
        },
        "hypothesis_decision_constants": {
            "h7_minimum_auc_correlation": H7_MINIMUM_AUC_CORRELATION,
            "h7_maximum_latency_correlation": H7_MAXIMUM_LATENCY_CORRELATION,
            "reversal_equivalence_margin_nats": REVERSAL_EQUIVALENCE_MARGIN,
            "identifiability_null_margin_nats": IDENTIFIABILITY_NULL_MARGIN,
        },
        "analysis": {
            "phase_summary_window": "final 50% of each phase",
            "bootstrap_replicates": behavior.bootstrap_replicates,
            "confidence_level": behavior.confidence_level,
            "bootstrap_unit": "seed",
        },
    }


def _source_sha256() -> str:
    digest = hashlib.sha256()
    source_directory = Path(__file__).resolve().parent
    for path in sorted(source_directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _study_configs(
    *,
    n_seeds: int,
    phase_steps: int,
    base_seed: int,
    bootstrap_replicates: int,
) -> tuple[StudyConfig, StudyConfig, StudyConfig]:
    burn_in = phase_steps // 2
    learner = LearnerConfig(
        mean_learning_rate=0.015,
        variance_learning_rate=0.01,
        initial_variance=1.0,
    )
    estimator = EstimatorConfig(
        smoothing_rate=0.01,
        evidence_clip=8.0,
    )
    common: dict[str, Any] = {
        "environment": EnvironmentConfig(),
        "learner": learner,
        "estimator": estimator,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "phase_metric_burn_in": burn_in,
    }
    disconnected = StudyConfig(phases=disconnected_schedule(phase_steps), **common)
    yoked = StudyConfig(phases=yoked_schedule(phase_steps), **common)
    behavior = StudyConfig(
        phases=disconnected_schedule(phase_steps),
        mode=ExperimentMode.GATED,
        gate=GateConfig(
            enabled=True,
            controller_gain=0.5,
            target_state=1.0,
            min_vigor=0.01,
            logistic_gain=30.0,
            influence_threshold=0.06,
            calibration_steps=min(300, phase_steps // 3),
            action_clip=3.0,
            probes_enabled=True,
            probe_probability=0.03,
            probe_amplitude=1.0,
        ),
        **common,
    )
    return disconnected, yoked, behavior


def _run_replicates(
    config: StudyConfig,
    *,
    behavior_policy: str = "learned",
    probe_probability: float | None = None,
    probe_probabilities_by_phase: Sequence[float] | None = None,
) -> list[SimulationTrace]:
    return [
        run_simulation(
            config,
            seed,
            behavior_policy=behavior_policy,  # type: ignore[arg-type]
            probe_probability=probe_probability,
            probe_probabilities_by_phase=probe_probabilities_by_phase,
        )
        for seed in config.seeds
    ]


def _run_controls(
    estimator_config: StudyConfig,
    behavior_config: StudyConfig,
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, list[float]],
    list[SimulationTrace],
    dict[str, StudyConfig],
]:
    phase_steps = estimator_config.phases[0].steps
    reversed_config = estimator_config.model_copy(
        update={
            "phases": (
                PhaseConfig(name="connected_initial", steps=phase_steps),
                PhaseConfig(
                    name="reversed",
                    steps=phase_steps,
                    coupling=CouplingMode.REVERSED,
                ),
                PhaseConfig(name="connected_restored", steps=phase_steps),
            ),
            "base_seed": estimator_config.base_seed + 1_000,
        },
    )
    reversed_traces = _run_replicates(reversed_config)

    no_randomization_gate = behavior_config.gate.model_copy(
        update={
            "calibration_steps": 0,
            "probes_enabled": False,
            "probe_probability": 0.0,
        },
    )
    deterministic_config = behavior_config.model_copy(
        update={
            "phases": (
                PhaseConfig(name="connected_1", steps=phase_steps),
                PhaseConfig(name="connected_2", steps=phase_steps),
                PhaseConfig(name="connected_3", steps=phase_steps),
            ),
            "gate": no_randomization_gate,
            "base_seed": behavior_config.base_seed + 2_000,
        },
    )
    deterministic_traces = _run_replicates(
        deterministic_config,
        behavior_policy="ungated",
        probe_probability=0.0,
    )

    yoked_behavior_config = behavior_config.model_copy(
        update={
            "phases": yoked_schedule(phase_steps),
            "base_seed": behavior_config.base_seed + 3_000,
        },
    )
    yoked_behavior_traces = _run_replicates(yoked_behavior_config)

    ungated_config = behavior_config.model_copy(
        update={
            "gate": behavior_config.gate.model_copy(
                update={"probes_enabled": False, "probe_probability": 0.0},
            ),
        },
    )
    ungated_traces = _run_replicates(ungated_config, behavior_policy="ungated")
    oracle_config = behavior_config.model_copy(
        update={
            "gate": behavior_config.gate.model_copy(
                update={"probes_enabled": False, "probe_probability": 0.0},
            ),
        },
    )
    oracle_traces = _run_replicates(
        oracle_config,
        behavior_policy="oracle",
        probe_probability=0.0,
    )

    rows: list[dict[str, float | int | str]] = []
    for label, traces, config in (
        ("estimator_reversed", reversed_traces, reversed_config),
        ("deterministic_policy", deterministic_traces, deterministic_config),
        ("behavior_yoked", yoked_behavior_traces, yoked_behavior_config),
        ("behavior_ungated", ungated_traces, ungated_config),
        ("behavior_oracle", oracle_traces, oracle_config),
    ):
        rows.extend(
            phase_metric_rows(
                traces,
                burn_in=config.phase_metric_burn_in,
                target_state=config.gate.target_state,
                condition=label,
            ),
        )

    extras: dict[str, list[float]] = {
        "reversal_early_mse": [],
        "connected_tail_mse": [],
        "reversal_tail_influence": [],
        "reversal_connected_tail_influence": [],
    }
    for trace in reversed_traces:
        middle = np.flatnonzero(trace.phase_index == 1)
        first = np.flatnonzero(trace.phase_index == 0)
        early = middle[: min(50, middle.size)]
        tail = middle[-min(300, middle.size // 2) :]
        first_tail = first[-min(300, first.size // 2) :]
        extras["reversal_early_mse"].append(float(np.mean(trace.squared_error_aware[early])))
        extras["connected_tail_mse"].append(
            float(np.mean(trace.squared_error_aware[first_tail])),
        )
        extras["reversal_tail_influence"].append(float(np.mean(trace.influence[tail])))
        extras["reversal_connected_tail_influence"].append(
            float(np.mean(trace.influence[first_tail])),
        )
    control_configs = {
        "estimator_reversed": reversed_config,
        "deterministic_policy": deterministic_config,
        "behavior_yoked": yoked_behavior_config,
        "behavior_ungated": ungated_config,
        "behavior_oracle": oracle_config,
    }
    return rows, extras, reversed_traces, control_configs


def _latency_rows(  # noqa: PLR0913
    estimator_disconnected: Sequence[SimulationTrace],
    estimator_yoked: Sequence[SimulationTrace],
    behavior_by_probe: dict[float, list[SimulationTrace]],
    estimator_disconnected_config: StudyConfig,
    estimator_yoked_config: StudyConfig,
    behavior_config: StudyConfig,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for condition, traces, config in (
        (
            "estimator_disconnected",
            estimator_disconnected,
            estimator_disconnected_config,
        ),
        ("estimator_yoked", estimator_yoked, estimator_yoked_config),
    ):
        reference = float(traces[0].true_information[0])
        off_threshold = INFLUENCE_OFF_FRACTION * reference
        on_threshold = INFLUENCE_ON_FRACTION * reference
        phase_length = config.phases[2].steps
        for trace in traces:
            detection, recovery, _, _ = transition_latencies(
                trace,
                series="influence",
                off_threshold=off_threshold,
                on_threshold=on_threshold,
                consecutive=LATENCY_CONSECUTIVE,
            )
            false_active, false_passive = state_classification_error_rates(
                trace,
                off_threshold=off_threshold,
                on_threshold=on_threshold,
                burn_in=config.phase_metric_burn_in,
            )
            rows.append(
                {
                    "condition": condition,
                    "series": "influence",
                    "probe_probability": "",
                    "seed": trace.seed,
                    "detection_latency": detection,
                    "recovery_latency": recovery,
                    "recovery_latency_censored": (
                        recovery if np.isfinite(recovery) else float(phase_length)
                    ),
                    "recovery_failed": int(not np.isfinite(recovery)),
                    "off_threshold": off_threshold,
                    "on_threshold": on_threshold,
                    "false_active_fraction": false_active,
                    "false_passive_fraction": false_passive,
                },
            )

    phase_length = behavior_config.phases[2].steps
    for probability, traces in behavior_by_probe.items():
        for trace in traces:
            detection, recovery, off_threshold, on_threshold = transition_latencies(
                trace,
                series="vigor",
                off_threshold=VIGOR_OFF_THRESHOLD,
                on_threshold=VIGOR_ON_THRESHOLD,
                consecutive=LATENCY_CONSECUTIVE,
            )
            rows.append(
                {
                    "condition": f"post_initial_probe_{probability:.2f}",
                    "series": "vigor",
                    "probe_probability": probability,
                    "seed": trace.seed,
                    "detection_latency": detection,
                    "recovery_latency": recovery,
                    "recovery_latency_censored": (
                        recovery if np.isfinite(recovery) else float(phase_length)
                    ),
                    "recovery_failed": int(not np.isfinite(recovery)),
                    "off_threshold": off_threshold,
                    "on_threshold": on_threshold,
                    "false_active_fraction": "",
                    "false_passive_fraction": "",
                },
            )
    return rows


def _run_robustness_grid(
    base_config: StudyConfig,
    *,
    progress: ProgressCallback | None,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for noise in ROBUSTNESS_NOISES:
        for gain in ROBUSTNESS_GAINS:
            config = base_config.model_copy(
                update={
                    "environment": base_config.environment.model_copy(
                        update={
                            "coupling_gain": gain,
                            "process_noise_std": noise,
                        },
                    ),
                },
            )
            connected_scores: list[float] = []
            yoked_scores: list[float] = []
            connected_loss_advantages: list[float] = []
            yoked_loss_advantages: list[float] = []
            connected_aware_mse: list[float] = []
            yoked_aware_mse: list[float] = []
            seed_aucs: list[float] = []
            clipped_fractions: list[float] = []
            analytic_targets: list[float] = []
            detection_latencies: list[float] = []
            recovery_latencies: list[float] = []
            for seed in config.seeds:
                trace = run_simulation(config, seed)
                first = phase_mask(trace, 0, config.phase_metric_burn_in)
                middle = phase_mask(trace, 1, config.phase_metric_burn_in)
                connected_scores.append(float(np.mean(trace.influence[first])))
                yoked_scores.append(float(np.mean(trace.influence[middle])))
                connected_loss_advantages.append(float(np.mean(trace.evidence[first])))
                yoked_loss_advantages.append(float(np.mean(trace.evidence[middle])))
                connected_aware_mse.append(
                    float(np.mean(trace.squared_error_aware[first])),
                )
                yoked_aware_mse.append(float(np.mean(trace.squared_error_aware[middle])))
                seed_aucs.append(roc_auc(trace.influence[first], trace.influence[middle]))
                clipped_fractions.append(
                    float(
                        np.mean(
                            ~np.isclose(
                                trace.evidence[first],
                                trace.clipped_evidence[first],
                                rtol=0.0,
                                atol=1e-12,
                            ),
                        ),
                    ),
                )
                analytic_targets.append(float(np.mean(trace.true_information[first])))
                target = analytic_targets[-1]
                detection, recovery, _, _ = transition_latencies(
                    trace,
                    off_threshold=0.25 * target,
                    on_threshold=0.75 * target,
                    consecutive=10,
                )
                phase_length = config.phases[1].steps
                detection_latencies.append(
                    detection if np.isfinite(detection) else float(phase_length),
                )
                recovery_latencies.append(
                    recovery if np.isfinite(recovery) else float(phase_length),
                )
            target = float(np.mean(analytic_targets))
            mean_connected = float(np.mean(connected_scores))
            mean_loss_advantage = float(np.mean(connected_loss_advantages))
            rows.append(
                {
                    "gain": gain,
                    "noise_std": noise,
                    "analytic_information": target,
                    "mean_connected_influence": mean_connected,
                    "mean_yoked_influence": float(np.mean(yoked_scores)),
                    "mean_connected_loss_advantage": mean_loss_advantage,
                    "mean_yoked_loss_advantage": float(
                        np.mean(yoked_loss_advantages),
                    ),
                    "mean_connected_aware_mse": float(np.mean(connected_aware_mse)),
                    "mean_yoked_aware_mse": float(np.mean(yoked_aware_mse)),
                    "mean_clipped_fraction": float(np.mean(clipped_fractions)),
                    "absolute_calibration_error": abs(mean_loss_advantage - target),
                    "relative_calibration": (
                        mean_loss_advantage / target if target > NUMERICAL_EPSILON else float("nan")
                    ),
                    "roc_auc": float(np.mean(seed_aucs)),
                    "median_detection_latency": float(np.median(detection_latencies)),
                    "median_recovery_latency": float(np.median(recovery_latencies)),
                    "detection_censor_fraction": float(
                        np.mean(np.asarray(detection_latencies) >= config.phases[1].steps),
                    ),
                    "recovery_censor_fraction": float(
                        np.mean(np.asarray(recovery_latencies) >= config.phases[2].steps),
                    ),
                    "n_seeds": config.n_seeds,
                },
            )
            _notify(progress, f"robustness gain={gain:.2f}, noise={noise:.2f}")
    return rows


def _summarize_hypotheses(  # noqa: PLR0913, PLR0915
    *,
    phase_rows: Sequence[dict[str, float | int | str]],
    latency_rows: Sequence[dict[str, float | int | str]],
    robustness_rows: Sequence[dict[str, float | int]],
    control_extras: dict[str, list[float]],
    bootstrap_replicates: int,
    phase_steps: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, Any]]]:
    inference: list[dict[str, float | int | str]] = []

    def values(condition: str, phase_index: int, metric: str) -> list[float]:
        selected = [
            float(row[metric])
            for row in phase_rows
            if row["condition"] == condition and row["phase_index"] == phase_index
        ]
        if not selected:
            message = f"no rows for {condition}, phase {phase_index}, {metric}"
            raise ValueError(message)
        return selected

    def add_interval(name: str, interval: ConfidenceInterval, interpretation: str) -> None:
        inference.append(
            {
                "comparison": name,
                "estimate": interval.estimate,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "n_seeds": interval.n,
                "interpretation": interpretation,
            },
        )

    bootstrap_options = {"replicates": bootstrap_replicates, "confidence_level": 0.95}
    connected_yoked = bootstrap_paired_difference(
        values("estimator_yoked", 0, "mean_influence"),
        values("estimator_yoked", 1, "mean_influence"),
        **bootstrap_options,
    )
    add_interval("H1_connected_minus_yoked_influence", connected_yoked, "positive supports H1")
    analytic_calibration_error = bootstrap_paired_difference(
        values("estimator_yoked", 0, "mean_loss_advantage"),
        values("estimator_yoked", 0, "mean_true_information"),
        **bootstrap_options,
    )
    add_interval(
        "analytic_calibration_error",
        analytic_calibration_error,
        "unclipped loss advantage minus analytic information",
    )
    variance_difference = bootstrap_paired_difference(
        values("estimator_yoked", 0, "sensory_variance"),
        values("estimator_yoked", 1, "sensory_variance"),
        **bootstrap_options,
    )
    add_interval(
        "yoked_connected_minus_yoked_sensory_variance",
        variance_difference,
        "interval near zero validates variance matching",
    )
    disconnection_drop = bootstrap_paired_difference(
        values("estimator_disconnected", 0, "mean_influence"),
        values("estimator_disconnected", 1, "mean_influence"),
        **bootstrap_options,
    )
    add_interval("H2_disconnection_drop", disconnection_drop, "positive supports H2")
    reconnection_rise = bootstrap_paired_difference(
        values("estimator_disconnected", 2, "mean_influence"),
        values("estimator_disconnected", 1, "mean_influence"),
        **bootstrap_options,
    )
    add_interval("H2_reconnection_rise", reconnection_rise, "positive supports H2")
    energy_saving = bootstrap_paired_difference(
        values("behavior_ungated", 1, "mean_action_energy"),
        values("behavior_probe_0.03", 1, "mean_action_energy"),
        **bootstrap_options,
    )
    add_interval("H3_ungated_minus_gated_energy", energy_saving, "positive supports H3")
    shuffle_gap = bootstrap_mean_interval(
        values("estimator_yoked", 0, "mean_shuffled_advantage"),
        **bootstrap_options,
    )
    add_interval("H6_shuffled_action_advantage", shuffle_gap, "zero expected under H6")
    reversed_positive = bootstrap_mean_interval(
        control_extras["reversal_tail_influence"],
        **bootstrap_options,
    )
    add_interval("H8_reversed_tail_influence", reversed_positive, "positive supports H8")
    reversed_equivalence = bootstrap_paired_difference(
        control_extras["reversal_tail_influence"],
        control_extras["reversal_connected_tail_influence"],
        **bootstrap_options,
    )
    add_interval(
        "H8_reversed_minus_connected_tail_influence",
        reversed_equivalence,
        "interval within equivalence margin supports equal influence",
    )

    latency_by_probability: dict[float, list[float]] = {}
    failures_by_probability: dict[float, float] = {}
    for probability in RECOVERY_PROBE_PROBABILITIES:
        selected = [
            row
            for row in latency_rows
            if row["series"] == "vigor" and np.isclose(float(row["probe_probability"]), probability)
        ]
        latency_by_probability[probability] = [
            float(row["recovery_latency_censored"]) for row in selected
        ]
        failures_by_probability[probability] = float(
            np.mean([float(row["recovery_failed"]) for row in selected]),
        )
    probe_gain = bootstrap_paired_difference(
        latency_by_probability[0.0],
        latency_by_probability[0.03],
        **bootstrap_options,
    )
    add_interval(
        "H4_withdrawn_minus_probe_recovery_latency",
        probe_gain,
        "positive supports H4",
    )

    high_noise_connected_mse = values(
        "estimator_yoked",
        0,
        "mean_aware_squared_error",
    )
    low_noise_yoked_mse = values("estimator_yoked", 1, "mean_aware_squared_error")
    # Within this matched condition raw error can separate phases; its lack of
    # specificity is assessed in the robustness grid and reversal transient.
    raw_error_difference = bootstrap_paired_difference(
        low_noise_yoked_mse,
        high_noise_connected_mse,
        **bootstrap_options,
    )
    add_interval(
        "H5_yoked_minus_connected_raw_error",
        raw_error_difference,
        "positive here, but raw error is noise- and reversal-sensitive",
    )

    robustness_auc = np.asarray([float(row["roc_auc"]) for row in robustness_rows])
    robustness_snr = np.asarray(
        [(float(row["gain"]) / float(row["noise_std"])) ** 2 for row in robustness_rows],
    )
    latency = np.asarray(
        [float(row["median_detection_latency"]) for row in robustness_rows],
    )
    snr_auc_correlation = _safe_correlation(np.log1p(robustness_snr), robustness_auc)
    snr_latency_correlation = _safe_correlation(np.log1p(robustness_snr), latency)
    analytic_grid = np.asarray(
        [float(row["analytic_information"]) for row in robustness_rows],
    )
    estimated_grid = np.asarray(
        [float(row["mean_connected_loss_advantage"]) for row in robustness_rows],
    )
    calibration_slope, calibration_intercept = np.polyfit(analytic_grid, estimated_grid, 1)
    calibration_rmse = float(np.sqrt(np.mean((estimated_grid - analytic_grid) ** 2)))

    high_noise_connected = min(
        robustness_rows,
        key=lambda row: abs(float(row["gain"]) - 0.1) + abs(float(row["noise_std"]) - 1.2),
    )
    low_noise_yoked = min(
        robustness_rows,
        key=lambda row: abs(float(row["gain"]) - 1.0) + abs(float(row["noise_std"]) - 0.1),
    )
    raw_error_reversal_spike = bootstrap_paired_difference(
        control_extras["reversal_early_mse"],
        control_extras["connected_tail_mse"],
        **bootstrap_options,
    )

    deterministic_influence = float(
        np.mean(values("deterministic_policy", 2, "mean_influence")),
    )
    aware_connected = values("estimator_yoked", 0, "mean_loss_advantage")
    shuffled_connected = values("estimator_yoked", 0, "mean_shuffled_advantage")
    hypothesis_results: dict[str, dict[str, Any]] = {
        "H1": {
            "claim": "Influence is higher when coupled than yoked with matched sensory variance.",
            "supported": connected_yoked.low > 0.0,
            "paired_difference": connected_yoked.estimate,
            "ci": [connected_yoked.low, connected_yoked.high],
            "variance_difference_ci": [variance_difference.low, variance_difference.high],
            "unclipped_calibration_error": analytic_calibration_error.estimate,
        },
        "H2": {
            "claim": "Influence falls after disconnection and rises after reconnection.",
            "supported": disconnection_drop.low > 0.0 and reconnection_rise.low > 0.0,
            "drop": disconnection_drop.estimate,
            "rise": reconnection_rise.estimate,
        },
        "H3": {
            "claim": "The learned gate saves action energy while disconnected.",
            "supported": energy_saving.low > 0.0,
            "energy_saved_per_step": energy_saving.estimate,
        },
        "H4": {
            "claim": "Persistent fixed-amplitude probes improve reconnection recovery.",
            "supported": probe_gain.low > 0.0
            or failures_by_probability[0.03] < failures_by_probability[0.0],
            "latency_improvement": probe_gain.estimate,
            "failure_fraction_probes_withdrawn": failures_by_probability[0.0],
            "failure_fraction_probe_0.03": failures_by_probability[0.03],
        },
        "H5": {
            "claim": "Raw forward error is a non-specific surprise baseline.",
            "supported": float(high_noise_connected["mean_connected_aware_mse"])
            > float(low_noise_yoked["mean_yoked_aware_mse"])
            and raw_error_reversal_spike.low > 0.0,
            "qualification": (
                "It separates the matched yoked phase in this parameterization, but changes "
                "with irreducible noise and spikes under still-controllable reversal."
            ),
            "matched_yoked_error_difference": raw_error_difference.estimate,
            "high_noise_connected_mse": float(
                high_noise_connected["mean_connected_aware_mse"],
            ),
            "low_noise_yoked_mse": float(low_noise_yoked["mean_yoked_aware_mse"]),
            "reversal_mse_spike": raw_error_reversal_spike.estimate,
        },
        "H6": {
            "claim": "Shuffling the command removes the action-aware advantage.",
            "supported": abs(float(np.mean(shuffled_connected)))
            < 0.2 * abs(float(np.mean(aware_connected))),
            "shuffled_advantage": float(np.mean(shuffled_connected)),
            "real_action_advantage": float(np.mean(aware_connected)),
        },
        "H7": {
            "claim": "Detection improves with action-to-noise ratio.",
            "supported": snr_auc_correlation > H7_MINIMUM_AUC_CORRELATION
            and snr_latency_correlation < H7_MAXIMUM_LATENCY_CORRELATION,
            "snr_auc_correlation": snr_auc_correlation,
            "snr_detection_latency_correlation": snr_latency_correlation,
        },
        "H8": {
            "claim": "Reversal retains influence after adaptation despite initial mismatch.",
            "supported": reversed_positive.low > 0.0
            and reversed_equivalence.low > -REVERSAL_EQUIVALENCE_MARGIN
            and reversed_equivalence.high < REVERSAL_EQUIVALENCE_MARGIN,
            "reversed_tail_influence": reversed_positive.estimate,
            "mean_initial_reversal_squared_error": float(
                np.mean(control_extras["reversal_early_mse"]),
            ),
            "reversed_minus_connected": reversed_equivalence.estimate,
        },
        "identifiability_control": {
            "claim": "A deterministic action given state does not identify conditional influence.",
            "mean_tail_influence": deterministic_influence,
            "supported": abs(deterministic_influence) < IDENTIFIABILITY_NULL_MARGIN,
        },
        "study_design": {
            "phase_steps": phase_steps,
            "independent_unit": "seed",
            "time_steps_treated_as_independent": False,
            "steady_state_window": "final 50% of each phase",
            "bootstrap_replicates": bootstrap_replicates,
            "calibration_slope": float(calibration_slope),
            "calibration_intercept": float(calibration_intercept),
            "calibration_rmse": calibration_rmse,
        },
    }
    return inference, hypothesis_results


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        message = f"cannot write empty table: {path}"
        raise ValueError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _control_summary_rows(
    phase_rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    selections = (
        ("real_action_connected", "estimator_yoked", 0),
        ("cable_disconnected", "estimator_disconnected", 1),
        ("matched_yoked", "estimator_yoked", 1),
        ("reversed_mapping", "estimator_reversed", 1),
        ("deterministic_policy", "deterministic_policy", 2),
        ("learned_gate_disconnected", "behavior_probe_0.03", 1),
        ("ungated_disconnected", "behavior_ungated", 1),
        ("oracle_disconnected", "behavior_oracle", 1),
    )
    metrics = (
        "mean_influence",
        "mean_loss_advantage",
        "mean_dummy_advantage",
        "mean_shuffled_advantage",
        "mean_wrong_lag_advantage",
        "mean_aware_squared_error",
        "action_outcome_correlation",
        "partial_action_outcome_correlation",
        "sensory_variance",
        "mean_action_energy",
        "mean_vigor",
        "mean_homeostatic_squared_error",
        "evidence_clipped_fraction",
    )
    summaries: list[dict[str, float | int | str]] = []
    for label, condition, phase_index in selections:
        selected = [
            row
            for row in phase_rows
            if row["condition"] == condition and row["phase_index"] == phase_index
        ]
        summary: dict[str, float | int | str] = {
            "control": label,
            "source_condition": condition,
            "phase_index": phase_index,
            "n_seeds": len(selected),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[metric] = float(np.mean(finite)) if finite.size else float("nan")
            summary[f"{metric}_seed_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
        summaries.append(summary)
    return summaries


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= NUMERICAL_EPSILON or np.std(second) <= NUMERICAL_EPSILON:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _json_safe(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
