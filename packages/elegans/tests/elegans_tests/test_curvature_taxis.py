"""Tests for the minimal continuous-curvature taxis demonstration."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from elegans.curvature_taxis import (
    CurvatureTaxisConfig,
    CurvatureTaxisEnvironment,
    TerminationReason,
    bilateral_curvature,
    run_heading_sweep,
    run_no_steering,
    run_taxis,
    save_demo_artifacts,
    summarize_heading_sweep,
)
from matplotlib import image as mpimg


def comparison_config() -> CurvatureTaxisConfig:
    """Return the small heading offset that makes steering necessary."""
    return replace(CurvatureTaxisConfig(), initial_heading_degrees=30.0)


def test_reset_observation_contains_only_bilateral_concentrations():
    """Reset returns exactly the two local concentrations and configured state."""
    environment = CurvatureTaxisEnvironment()

    observation = environment.reset()

    assert observation.shape == (2,)
    assert np.all((observation > 0.0) & (observation <= 1.0))
    assert environment.position == pytest.approx((2.0, 3.0))
    assert environment.heading == pytest.approx(np.deg2rad(20.0))


@pytest.mark.parametrize(
    ("curvature", "expected_heading_sign", "expected_y_sign"),
    [
        (0.0, 0, 0),
        (1.0, 1, 1),
        (-1.0, -1, -1),
    ],
)
def test_fixed_curvature_has_straight_left_right_semantics(
    curvature,
    expected_heading_sign,
    expected_y_sign,
):
    """Zero is straight, positive bends left, and negative bends right."""
    config = replace(
        CurvatureTaxisConfig(),
        start=(5.0, 5.0),
        source=(9.0, 9.0),
        initial_heading_degrees=0.0,
        target_radius=0.1,
    )
    environment = CurvatureTaxisEnvironment(config)
    environment.reset()

    for _ in range(20):
        _observation, _reward, terminated, _info = environment.step(curvature)
        assert not terminated

    heading_sign = int(np.sign(environment.heading))
    y_sign = int(np.sign(environment.position[1] - config.start[1]))
    assert heading_sign == expected_heading_sign
    assert y_sign == expected_y_sign
    assert environment.position[0] > config.start[0]


def test_bilateral_controller_turns_toward_stronger_sensor():
    """The controller's curvature sign follows the stronger sensor."""
    assert bilateral_curvature(np.array([0.6, 0.4])) > 0.0
    assert bilateral_curvature(np.array([0.4, 0.6])) < 0.0
    assert bilateral_curvature(np.array([0.5, 0.5])) == pytest.approx(0.0)


def test_step_uses_curvature_as_sole_action_and_has_no_reward():
    """Step applies the kinematic equation and does not introduce reward."""
    environment = CurvatureTaxisEnvironment(comparison_config())
    old_heading = environment.heading

    observation, reward, terminated, info = environment.step(1.25)

    assert observation.shape == (2,)
    assert reward == 0.0
    assert not terminated
    assert info["curvature"] == pytest.approx(1.25)
    assert environment.heading - old_heading == pytest.approx(
        environment.config.speed * 1.25 * environment.config.dt,
    )


def test_episode_log_aligns_observation_action_next_observation():
    """Every transition is aligned to its surrounding state snapshots."""
    trace = run_taxis(comparison_config())

    assert len(trace.snapshots) == len(trace.transitions) + 1
    for index, transition in enumerate(trace.transitions):
        current = trace.snapshots[index]
        following = trace.snapshots[index + 1]
        assert transition.observation == pytest.approx(
            (current.left_concentration, current.right_concentration),
        )
        assert transition.next_observation == pytest.approx(
            (following.left_concentration, following.right_concentration),
        )


def test_taxis_reaches_source_while_zero_curvature_baseline_misses():
    """Bilateral steering succeeds from a heading where straight motion misses."""
    config = comparison_config()

    taxis = run_taxis(config)
    baseline = run_no_steering(config)

    assert taxis.success
    assert taxis.termination_reason is TerminationReason.TARGET_REACHED
    assert taxis.distances[-1] < config.target_radius
    assert taxis.distances[-1] < taxis.distances[0]
    assert taxis.center_concentrations[-1] > taxis.center_concentrations[0]
    assert np.any(np.abs(taxis.curvatures) > 0.0)

    assert not baseline.success
    assert baseline.termination_reason is TerminationReason.ARENA_EXITED
    assert np.min(baseline.distances) > config.target_radius


def test_twenty_degree_requested_heading_is_almost_the_direct_bearing():
    """The exact requested geometry explains why its straight baseline succeeds."""
    config = CurvatureTaxisConfig()
    direct_bearing = np.rad2deg(
        np.arctan2(
            config.source[1] - config.start[1],
            config.source[0] - config.start[0],
        ),
    )

    assert direct_bearing == pytest.approx(19.9831, abs=1e-4)
    assert run_no_steering(config).success


def test_heading_sweep_outperforms_no_steering():
    """Taxis remains substantially more robust across initial headings."""
    rows = run_heading_sweep(comparison_config(), count=20)
    summary = summarize_heading_sweep(rows)

    assert summary["taxis_successes"] >= 18
    assert summary["baseline_successes"] <= 2
    assert summary["taxis_success_rate"] > summary["baseline_success_rate"]


def test_demo_artifacts_are_complete_and_readable(tmp_path):
    """The runnable demo writes complete, decodable analysis artifacts."""
    summary = save_demo_artifacts(tmp_path, comparison_config(), heading_count=4)

    expected_names = {
        "curvature_taxis_demo.png",
        "taxis_trace.csv",
        "no_steering_trace.csv",
        "heading_sweep.csv",
        "summary.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    assert mpimg.imread(tmp_path / "curvature_taxis_demo.png").size > 0

    with (tmp_path / "taxis_trace.csv").open(encoding="utf-8", newline="") as csv_file:
        first_row = next(csv.DictReader(csv_file))
    assert {
        "left_concentration",
        "right_concentration",
        "curvature",
        "next_left_concentration",
        "next_right_concentration",
    } <= first_row.keys()

    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["taxis"]["success"]
    assert not saved_summary["no_steering"]["success"]
    assert summary["heading_sweep"]["heading_count"] == 4
