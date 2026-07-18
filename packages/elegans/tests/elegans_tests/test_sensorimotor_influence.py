"""Tests for the isolated online sensorimotor-influence proof of concept."""

from __future__ import annotations

from math import log, pi

import numpy as np
import pytest
from elegans.sensorimotor_influence.analysis import (
    phase_mask,
    roc_auc,
    sustained_crossing_latency,
    transition_latencies,
)
from elegans.sensorimotor_influence.config import (
    CouplingMode,
    EstimatorConfig,
    ExperimentMode,
    GateConfig,
    LearnerConfig,
    PhaseConfig,
    StudyConfig,
    disconnected_schedule,
    yoked_schedule,
)
from elegans.sensorimotor_influence.predictors import (
    OnlineLinearGaussian,
    predictor_features,
)
from elegans.sensorimotor_influence.simulation import (
    analytic_conditional_information,
    run_simulation,
)


def test_analytic_conditional_information_has_expected_symmetries() -> None:
    """The analytic target is zero at no gain and invariant to gain sign."""
    expected = 0.5 * log(1.0 + 1.0 / 0.04)
    assert analytic_conditional_information(1.0, 1.0, 0.2) == pytest.approx(expected)
    assert analytic_conditional_information(-1.0, 1.0, 0.2) == pytest.approx(expected)
    assert analytic_conditional_information(0.0, 1.0, 0.2) == 0.0
    assert analytic_conditional_information(1.0, 1.0, 0.4) < expected


def test_predictor_scores_before_updating_current_transition() -> None:
    """A learner must report its frozen pre-transition prediction and loss."""
    learner = OnlineLinearGaussian(3, LearnerConfig(mean_learning_rate=0.1))
    features = predictor_features(0.5, 1.0)
    prediction = learner.step(features, target=1.0)

    assert prediction.mean == 0.0
    assert prediction.error == 1.0
    assert prediction.negative_log_likelihood == pytest.approx(0.5 * (log(2.0 * pi) + 1.0))
    assert np.linalg.norm(learner.weights) > 0.0


def test_paired_conditions_reuse_actions_and_noise_but_change_world_drive() -> None:
    """Paired conditions share exogenous draws but differ in applied drive."""
    disconnected = StudyConfig(phases=disconnected_schedule(300), n_seeds=1)
    yoked = disconnected.model_copy(update={"phases": yoked_schedule(300)})
    disconnected_trace = run_simulation(disconnected, seed=812)
    yoked_trace = run_simulation(yoked, seed=812)

    np.testing.assert_array_equal(disconnected_trace.action, yoked_trace.action)
    np.testing.assert_array_equal(disconnected_trace.process_noise, yoked_trace.process_noise)
    np.testing.assert_array_equal(
        disconnected_trace.state[:301],
        yoked_trace.state[:301],
    )
    assert not np.array_equal(
        disconnected_trace.applied_drive[300:600],
        yoked_trace.applied_drive[300:600],
    )


def test_estimator_separates_coupled_from_yoked_with_matched_variability() -> None:
    """Real commands, but not shuffled commands, explain matched sensory change."""
    phase_steps = 800
    burn_in = phase_steps // 2
    config = StudyConfig(
        phases=yoked_schedule(phase_steps),
        learner=LearnerConfig(mean_learning_rate=0.015),
        estimator=EstimatorConfig(smoothing_rate=0.01, evidence_clip=8.0),
        n_seeds=4,
        base_seed=4_200,
        phase_metric_burn_in=burn_in,
    )
    traces = [run_simulation(config, seed) for seed in config.seeds]
    connected = np.asarray(
        [np.mean(trace.influence[phase_mask(trace, 0, burn_in)]) for trace in traces],
    )
    yoked = np.asarray(
        [np.mean(trace.influence[phase_mask(trace, 1, burn_in)]) for trace in traces],
    )
    connected_variance = np.asarray(
        [np.var(trace.state[1:][phase_mask(trace, 0, burn_in)]) for trace in traces],
    )
    yoked_variance = np.asarray(
        [np.var(trace.state[1:][phase_mask(trace, 1, burn_in)]) for trace in traces],
    )

    assert np.mean(connected - yoked) > 0.8
    assert abs(np.mean(yoked)) < 0.1
    assert 0.7 < np.mean(yoked_variance) / np.mean(connected_variance) < 1.3
    real_advantage = np.mean(
        [
            np.mean(
                trace.loss_blind[phase_mask(trace, 0, burn_in)]
                - trace.loss_aware[phase_mask(trace, 0, burn_in)],
            )
            for trace in traces
        ],
    )
    shuffled_advantage = np.mean(
        [
            np.mean(
                trace.loss_blind[phase_mask(trace, 0, burn_in)]
                - trace.loss_shuffled[phase_mask(trace, 0, burn_in)],
            )
            for trace in traces
        ],
    )
    assert abs(shuffled_advantage) < 0.1 * real_advantage


def test_fixed_amplitude_probes_restore_vigor_after_reconnection() -> None:
    """Ongoing probes recover vigor after an initially matched arm withdraws them."""
    phase_steps = 800
    config = StudyConfig(
        phases=disconnected_schedule(phase_steps),
        mode=ExperimentMode.GATED,
        learner=LearnerConfig(mean_learning_rate=0.015),
        estimator=EstimatorConfig(smoothing_rate=0.01, evidence_clip=8.0),
        gate=GateConfig(
            enabled=True,
            min_vigor=0.01,
            logistic_gain=30.0,
            influence_threshold=0.06,
            calibration_steps=250,
            probe_probability=0.03,
        ),
        n_seeds=4,
        base_seed=9_100,
        phase_metric_burn_in=phase_steps // 2,
    )
    withdrawn_recovery: list[float] = []
    probe_recovery: list[float] = []
    withdrawn_final_vigor: list[float] = []
    probe_final_vigor: list[float] = []
    for seed in config.seeds:
        withdrawn = run_simulation(
            config,
            seed,
            probe_probabilities_by_phase=(0.03, 0.0, 0.0),
        )
        probing = run_simulation(
            config,
            seed,
            probe_probabilities_by_phase=(0.03, 0.03, 0.03),
        )
        np.testing.assert_array_equal(
            withdrawn.action[:phase_steps],
            probing.action[:phase_steps],
        )
        withdrawn_recovery.append(
            transition_latencies(
                withdrawn,
                series="vigor",
                off_threshold=0.25,
                on_threshold=0.75,
            )[1],
        )
        probe_recovery.append(
            transition_latencies(
                probing,
                series="vigor",
                off_threshold=0.25,
                on_threshold=0.75,
            )[1],
        )
        restored = phase_mask(probing, 2, phase_steps // 2)
        withdrawn_final_vigor.append(float(np.mean(withdrawn.vigor[restored])))
        probe_final_vigor.append(float(np.mean(probing.vigor[restored])))

    assert all(not np.isfinite(latency) for latency in withdrawn_recovery)
    assert all(np.isfinite(latency) for latency in probe_recovery)
    assert np.mean(probe_final_vigor) > 0.9
    assert np.mean(withdrawn_final_vigor) < 0.3


def test_reversal_retains_influence_after_model_adapts() -> None:
    """The learner adapts its action sign and recovers positive influence."""
    phase_steps = 1_000
    config = StudyConfig(
        phases=(
            PhaseConfig(name="connected", steps=phase_steps),
            PhaseConfig(name="reversed", steps=phase_steps, coupling=CouplingMode.REVERSED),
            PhaseConfig(name="restored", steps=phase_steps),
        ),
        learner=LearnerConfig(mean_learning_rate=0.015),
        estimator=EstimatorConfig(smoothing_rate=0.01, evidence_clip=8.0),
        n_seeds=1,
    )
    trace = run_simulation(config, seed=733)
    reversed_tail = phase_mask(trace, 1, 700)

    assert np.mean(trace.influence[reversed_tail]) > 1.0
    assert np.mean(trace.learned_action_weight[reversed_tail]) < -0.8


def test_roc_auc_handles_ties() -> None:
    """The rank implementation gives half credit to tied scores."""
    assert roc_auc([1.0, 2.0], [0.0, 1.0]) == pytest.approx(0.875)


def test_sustained_crossing_latency_reports_online_confirmation_time() -> None:
    """A sustained crossing is detected only after its full run is observed."""
    values = np.asarray([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)

    latency = sustained_crossing_latency(
        values,
        start=0,
        stop=len(values),
        threshold=0.5,
        direction="above",
        consecutive=3,
    )

    assert latency == 3.0
