"""Minimal continuous-curvature taxis environment and hand-written controller.

This module is deliberately separate from the project's learned, discrete grid agent.  It proves
one small substrate: a forward-moving point agent can steer toward a smooth source using only the
difference between two local concentration sensors.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path

POSITION_DIMENSIONS = 2

type Observation = NDArray[np.float64]
type Controller = Callable[[Observation], float]


class StepInfo(TypedDict):
    """Diagnostic values returned by :meth:`CurvatureTaxisEnvironment.step`."""

    time: float
    position: tuple[float, float]
    heading: float
    left_concentration: float
    right_concentration: float
    center_concentration: float
    curvature: float
    distance_to_source: float
    success: bool
    termination_reason: str


class HeadingSweepSummary(TypedDict):
    """Aggregate success and distance metrics for a heading sweep."""

    heading_count: int
    taxis_successes: int
    baseline_successes: int
    taxis_success_rate: float
    baseline_success_rate: float
    median_taxis_time_to_target: float | None
    median_taxis_final_distance: float
    median_baseline_final_distance: float


class TerminationReason(StrEnum):
    """Reasons a curvature-taxis episode can end."""

    TARGET_REACHED = "target_reached"
    ARENA_EXITED = "arena_exited"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True, slots=True)
class CurvatureTaxisConfig:
    """Fixed parameters for the smallest curvature-navigation experiment."""

    arena_size: float = 10.0
    source: tuple[float, float] = (7.5, 5.0)
    start: tuple[float, float] = (2.0, 3.0)
    initial_heading_degrees: float = 20.0
    sigma: float = 2.5
    speed: float = 0.3
    dt: float = 0.05
    sensor_forward_offset: float = 0.20
    sensor_lateral_offset: float = 0.12
    max_curvature: float = 4.0
    controller_gain: float = 2.0
    target_radius: float = 0.5
    max_duration: float = 120.0

    def __post_init__(self) -> None:
        """Reject invalid geometry and dynamics before a rollout begins."""
        positive_values = {
            "arena_size": self.arena_size,
            "sigma": self.sigma,
            "speed": self.speed,
            "dt": self.dt,
            "sensor_forward_offset": self.sensor_forward_offset,
            "sensor_lateral_offset": self.sensor_lateral_offset,
            "max_curvature": self.max_curvature,
            "controller_gain": self.controller_gain,
            "target_radius": self.target_radius,
            "max_duration": self.max_duration,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0.0:
                message = f"{name} must be finite and greater than zero"
                raise ValueError(message)

        if not np.isfinite(self.initial_heading_degrees):
            message = "initial_heading_degrees must be finite"
            raise ValueError(message)

        for name, point in (("source", self.source), ("start", self.start)):
            if len(point) != POSITION_DIMENSIONS or not all(
                np.isfinite(coordinate) for coordinate in point
            ):
                message = f"{name} must contain two finite coordinates"
                raise ValueError(message)
            if not all(0.0 <= coordinate <= self.arena_size for coordinate in point):
                message = f"{name} must lie inside the arena"
                raise ValueError(message)


@dataclass(frozen=True, slots=True)
class TaxisSnapshot:
    """All state and sensory values recorded at one instant."""

    time: float
    position: tuple[float, float]
    heading: float
    left_concentration: float
    right_concentration: float
    center_concentration: float
    distance_to_source: float


@dataclass(frozen=True, slots=True)
class TaxisTransition:
    """One ``(observation, curvature, next_observation)`` transition."""

    observation: tuple[float, float]
    curvature: float
    next_observation: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TaxisTrace:
    """A complete episode, including aligned state and transition logs."""

    snapshots: tuple[TaxisSnapshot, ...]
    transitions: tuple[TaxisTransition, ...]
    success: bool
    termination_reason: TerminationReason

    @property
    def times(self) -> NDArray[np.float64]:
        """Return snapshot times as an array."""
        return np.fromiter((snapshot.time for snapshot in self.snapshots), dtype=np.float64)

    @property
    def positions(self) -> NDArray[np.float64]:
        """Return the ``(x, y)`` trajectory."""
        return np.asarray([snapshot.position for snapshot in self.snapshots], dtype=np.float64)

    @property
    def headings(self) -> NDArray[np.float64]:
        """Return headings in radians."""
        return np.fromiter((snapshot.heading for snapshot in self.snapshots), dtype=np.float64)

    @property
    def left_concentrations(self) -> NDArray[np.float64]:
        """Return left-sensor concentrations."""
        return np.fromiter(
            (snapshot.left_concentration for snapshot in self.snapshots),
            dtype=np.float64,
        )

    @property
    def right_concentrations(self) -> NDArray[np.float64]:
        """Return right-sensor concentrations."""
        return np.fromiter(
            (snapshot.right_concentration for snapshot in self.snapshots),
            dtype=np.float64,
        )

    @property
    def center_concentrations(self) -> NDArray[np.float64]:
        """Return concentrations experienced at the agent center."""
        return np.fromiter(
            (snapshot.center_concentration for snapshot in self.snapshots),
            dtype=np.float64,
        )

    @property
    def distances(self) -> NDArray[np.float64]:
        """Return distances from the source."""
        return np.fromiter(
            (snapshot.distance_to_source for snapshot in self.snapshots),
            dtype=np.float64,
        )

    @property
    def curvatures(self) -> NDArray[np.float64]:
        """Return one curvature command per transition."""
        return np.fromiter(
            (transition.curvature for transition in self.transitions),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class HeadingSweepRow:
    """Controller and no-steering outcomes for one initial heading."""

    heading_degrees: float
    taxis_success: bool
    taxis_time: float
    taxis_final_distance: float
    baseline_success: bool
    baseline_time: float
    baseline_final_distance: float


class CurvatureTaxisEnvironment:
    """A deterministic 2D Gaussian field with continuous curvature dynamics."""

    def __init__(self, config: CurvatureTaxisConfig | None = None) -> None:
        """Create the environment and initialize it at the configured start state."""
        self.config = config or CurvatureTaxisConfig()
        self._source = np.asarray(self.config.source, dtype=np.float64)
        self._position = np.empty(2, dtype=np.float64)
        self._heading = 0.0
        self._time = 0.0
        self._terminated = False
        self._termination_reason: TerminationReason | None = None
        self.reset()

    @property
    def position(self) -> NDArray[np.float64]:
        """Return a copy of the current position."""
        return self._position.copy()

    @property
    def heading(self) -> float:
        """Return the current heading in radians."""
        return self._heading

    @property
    def time(self) -> float:
        """Return elapsed simulated time."""
        return self._time

    def reset(self) -> Observation:
        """Restore the configured state and return ``[left, right]`` concentration."""
        self._position = np.asarray(self.config.start, dtype=np.float64)
        self._heading = float(np.deg2rad(self.config.initial_heading_degrees))
        self._time = 0.0
        self._terminated = False
        self._termination_reason = None
        return self.observe()

    def concentration(self, position: Sequence[float] | NDArray[np.float64]) -> float:
        """Evaluate the Gaussian source concentration at a 2D position."""
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (POSITION_DIMENSIONS,):
            message = "position must have shape (2,)"
            raise ValueError(message)
        displacement = point - self._source
        squared_distance = float(displacement @ displacement)
        return float(np.exp(-squared_distance / (2.0 * self.config.sigma**2)))

    def sensor_positions(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return left and right sensor positions in world coordinates."""
        heading = np.array([np.cos(self._heading), np.sin(self._heading)], dtype=np.float64)
        left_normal = np.array([-heading[1], heading[0]], dtype=np.float64)
        forward = self.config.sensor_forward_offset * heading
        lateral = self.config.sensor_lateral_offset * left_normal
        return self._position + forward + lateral, self._position + forward - lateral

    def observe(self) -> Observation:
        """Return only the two local sensor concentrations available to the controller."""
        left_position, right_position = self.sensor_positions()
        return np.array(
            [self.concentration(left_position), self.concentration(right_position)],
            dtype=np.float64,
        )

    def snapshot(self) -> TaxisSnapshot:
        """Capture the complete state needed for analysis and plotting."""
        observation = self.observe()
        distance = float(np.linalg.norm(self._position - self._source))
        return TaxisSnapshot(
            time=self._time,
            position=(float(self._position[0]), float(self._position[1])),
            heading=self._heading,
            left_concentration=float(observation[0]),
            right_concentration=float(observation[1]),
            center_concentration=self.concentration(self._position),
            distance_to_source=distance,
        )

    def step(self, curvature: float) -> tuple[Observation, float, bool, StepInfo]:
        """Advance once using curvature as the sole action.

        Positive curvature turns counterclockwise (left), negative curvature turns clockwise
        (right), and zero curvature preserves the heading.  Reward is always zero because this
        hand-written-controller demo does not learn.
        """
        if self._terminated:
            message = "step() called after the episode terminated; call reset() first"
            raise RuntimeError(message)

        command = float(curvature)
        if not np.isfinite(command):
            message = "curvature must be finite"
            raise ValueError(message)
        command = float(np.clip(command, -self.config.max_curvature, self.config.max_curvature))

        self._heading += self.config.speed * command * self.config.dt
        direction = np.array(
            [np.cos(self._heading), np.sin(self._heading)],
            dtype=np.float64,
        )
        self._position += self.config.speed * direction * self.config.dt
        self._time += self.config.dt

        snapshot = self.snapshot()
        if snapshot.distance_to_source < self.config.target_radius:
            self._termination_reason = TerminationReason.TARGET_REACHED
        elif not self._inside_arena():
            self._termination_reason = TerminationReason.ARENA_EXITED
        elif self._time >= self.config.max_duration:
            self._termination_reason = TerminationReason.TIME_LIMIT

        self._terminated = self._termination_reason is not None
        observation = np.array(
            [snapshot.left_concentration, snapshot.right_concentration],
            dtype=np.float64,
        )
        info: StepInfo = {
            "time": snapshot.time,
            "position": snapshot.position,
            "heading": snapshot.heading,
            "left_concentration": snapshot.left_concentration,
            "right_concentration": snapshot.right_concentration,
            "center_concentration": snapshot.center_concentration,
            "curvature": command,
            "distance_to_source": snapshot.distance_to_source,
            "success": self._termination_reason is TerminationReason.TARGET_REACHED,
            "termination_reason": self._termination_reason.value
            if self._termination_reason is not None
            else "",
        }
        return observation, 0.0, self._terminated, info

    def _inside_arena(self) -> bool:
        return bool(np.all((self._position >= 0.0) & (self._position <= self.config.arena_size)))


def bilateral_curvature(
    observation: Observation,
    *,
    max_curvature: float = 4.0,
    gain: float = 2.0,
) -> float:
    """Map normalized left-right concentration difference to signed curvature."""
    sensor_values = np.asarray(observation, dtype=np.float64)
    if sensor_values.shape != (POSITION_DIMENSIONS,) or not np.all(np.isfinite(sensor_values)):
        message = "observation must contain two finite sensor values"
        raise ValueError(message)
    if max_curvature <= 0.0 or gain <= 0.0:
        message = "max_curvature and gain must be greater than zero"
        raise ValueError(message)

    left, right = sensor_values
    sensor_error = (left - right) / (left + right + 1e-8)
    return float(max_curvature * np.tanh(gain * sensor_error))


def no_steering(_observation: Observation) -> float:
    """Return the zero-curvature baseline action."""
    return 0.0


def run_episode(config: CurvatureTaxisConfig, controller: Controller) -> TaxisTrace:
    """Run one deterministic episode and retain every state and transition."""
    environment = CurvatureTaxisEnvironment(config)
    observation = environment.reset()
    snapshots = [environment.snapshot()]
    transitions: list[TaxisTransition] = []

    while True:
        requested_curvature = controller(observation.copy())
        next_observation, _reward, terminated, info = environment.step(requested_curvature)
        applied_curvature = float(info["curvature"])
        transitions.append(
            TaxisTransition(
                observation=(float(observation[0]), float(observation[1])),
                curvature=applied_curvature,
                next_observation=(float(next_observation[0]), float(next_observation[1])),
            ),
        )
        snapshots.append(environment.snapshot())
        observation = next_observation
        if terminated:
            reason = TerminationReason(str(info["termination_reason"]))
            return TaxisTrace(
                snapshots=tuple(snapshots),
                transitions=tuple(transitions),
                success=reason is TerminationReason.TARGET_REACHED,
                termination_reason=reason,
            )


def run_taxis(config: CurvatureTaxisConfig) -> TaxisTrace:
    """Run the bilateral hand-written taxis controller."""
    controller = partial(
        bilateral_curvature,
        max_curvature=config.max_curvature,
        gain=config.controller_gain,
    )
    return run_episode(config, controller)


def run_no_steering(config: CurvatureTaxisConfig) -> TaxisTrace:
    """Run an otherwise identical zero-curvature baseline."""
    return run_episode(config, no_steering)


def run_heading_sweep(
    config: CurvatureTaxisConfig,
    count: int = 20,
) -> tuple[HeadingSweepRow, ...]:
    """Compare taxis and no steering across evenly spaced initial headings."""
    if count < 1:
        message = "count must be at least one"
        raise ValueError(message)

    rows: list[HeadingSweepRow] = []
    for heading_degrees in np.linspace(0.0, 360.0, count, endpoint=False):
        heading_config = replace(config, initial_heading_degrees=float(heading_degrees))
        taxis = run_taxis(heading_config)
        baseline = run_no_steering(heading_config)
        rows.append(
            HeadingSweepRow(
                heading_degrees=float(heading_degrees),
                taxis_success=taxis.success,
                taxis_time=taxis.snapshots[-1].time,
                taxis_final_distance=taxis.snapshots[-1].distance_to_source,
                baseline_success=baseline.success,
                baseline_time=baseline.snapshots[-1].time,
                baseline_final_distance=baseline.snapshots[-1].distance_to_source,
            ),
        )
    return tuple(rows)


def summarize_heading_sweep(rows: Sequence[HeadingSweepRow]) -> HeadingSweepSummary:
    """Return compact success and final-distance statistics for a heading sweep."""
    if not rows:
        message = "rows must not be empty"
        raise ValueError(message)

    successful_times = [row.taxis_time for row in rows if row.taxis_success]
    return {
        "heading_count": len(rows),
        "taxis_successes": sum(row.taxis_success for row in rows),
        "baseline_successes": sum(row.baseline_success for row in rows),
        "taxis_success_rate": float(np.mean([row.taxis_success for row in rows])),
        "baseline_success_rate": float(np.mean([row.baseline_success for row in rows])),
        "median_taxis_time_to_target": float(np.median(successful_times))
        if successful_times
        else None,
        "median_taxis_final_distance": float(
            np.median([row.taxis_final_distance for row in rows]),
        ),
        "median_baseline_final_distance": float(
            np.median([row.baseline_final_distance for row in rows]),
        ),
    }


def save_demo_artifacts(
    output_dir: Path,
    config: CurvatureTaxisConfig,
    *,
    heading_count: int = 20,
) -> dict[str, Any]:
    """Run the comparison and write its figure, traces, sweep, and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    taxis = run_taxis(config)
    baseline = run_no_steering(config)
    sweep = run_heading_sweep(config, heading_count)

    figure_path = output_dir / "curvature_taxis_demo.png"
    taxis_csv_path = output_dir / "taxis_trace.csv"
    baseline_csv_path = output_dir / "no_steering_trace.csv"
    sweep_csv_path = output_dir / "heading_sweep.csv"
    summary_path = output_dir / "summary.json"

    _plot_demo(figure_path, config, taxis, baseline)
    _write_trace_csv(taxis_csv_path, taxis)
    _write_trace_csv(baseline_csv_path, baseline)
    _write_sweep_csv(sweep_csv_path, sweep)

    summary: dict[str, Any] = {
        "config": asdict(config),
        "source_bearing_degrees": float(
            np.rad2deg(
                np.arctan2(
                    config.source[1] - config.start[1],
                    config.source[0] - config.start[0],
                ),
            ),
        ),
        "taxis": _trace_summary(taxis),
        "no_steering": _trace_summary(baseline),
        "heading_sweep": summarize_heading_sweep(sweep),
        "artifacts": {
            "figure": str(figure_path),
            "taxis_trace": str(taxis_csv_path),
            "no_steering_trace": str(baseline_csv_path),
            "heading_sweep": str(sweep_csv_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _trace_summary(trace: TaxisTrace) -> dict[str, float | bool | str]:
    initial = trace.snapshots[0]
    final = trace.snapshots[-1]
    return {
        "success": trace.success,
        "termination_reason": trace.termination_reason.value,
        "duration": final.time,
        "steps": len(trace.transitions),
        "initial_distance": initial.distance_to_source,
        "final_distance": final.distance_to_source,
        "initial_center_concentration": initial.center_concentration,
        "final_center_concentration": final.center_concentration,
    }


def _write_trace_csv(path: Path, trace: TaxisTrace) -> None:
    fieldnames = [
        "time",
        "x",
        "y",
        "heading",
        "left_concentration",
        "right_concentration",
        "center_concentration",
        "curvature",
        "distance_to_source",
        "next_time",
        "next_x",
        "next_y",
        "next_heading",
        "next_left_concentration",
        "next_right_concentration",
        "next_center_concentration",
        "next_distance_to_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, transition in enumerate(trace.transitions):
            current = trace.snapshots[index]
            following = trace.snapshots[index + 1]
            writer.writerow(
                {
                    "time": current.time,
                    "x": current.position[0],
                    "y": current.position[1],
                    "heading": current.heading,
                    "left_concentration": transition.observation[0],
                    "right_concentration": transition.observation[1],
                    "center_concentration": current.center_concentration,
                    "curvature": transition.curvature,
                    "distance_to_source": current.distance_to_source,
                    "next_time": following.time,
                    "next_x": following.position[0],
                    "next_y": following.position[1],
                    "next_heading": following.heading,
                    "next_left_concentration": transition.next_observation[0],
                    "next_right_concentration": transition.next_observation[1],
                    "next_center_concentration": following.center_concentration,
                    "next_distance_to_source": following.distance_to_source,
                },
            )


def _write_sweep_csv(path: Path, rows: Sequence[HeadingSweepRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _plot_demo(
    path: Path,
    config: CurvatureTaxisConfig,
    taxis: TaxisTrace,
    baseline: TaxisTrace,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    coordinates = np.linspace(0.0, config.arena_size, 240)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    squared_distance = (grid_x - config.source[0]) ** 2 + (grid_y - config.source[1]) ** 2
    field = np.exp(-squared_distance / (2.0 * config.sigma**2))

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    trajectory_axis, distance_axis, concentration_axis, curvature_axis = axes.ravel()

    heatmap = trajectory_axis.contourf(grid_x, grid_y, field, levels=30, cmap="viridis")
    figure.colorbar(heatmap, ax=trajectory_axis, label="Odor concentration")
    trajectory_axis.plot(
        baseline.positions[:, 0],
        baseline.positions[:, 1],
        color="white",
        linestyle="--",
        linewidth=2.0,
        label="No steering",
    )
    trajectory_axis.plot(
        taxis.positions[:, 0],
        taxis.positions[:, 1],
        color="#ff6b35",
        linewidth=3.0,
        label="Bilateral taxis",
    )
    trajectory_axis.scatter(*config.start, color="cyan", edgecolor="black", s=80, label="Start")
    trajectory_axis.scatter(
        *config.source,
        marker="*",
        color="yellow",
        edgecolor="black",
        s=260,
        label="Source",
    )
    target = Circle(
        config.source,
        config.target_radius,
        fill=False,
        color="yellow",
        linewidth=2,
    )
    trajectory_axis.add_patch(target)
    trajectory_axis.scatter(
        *taxis.snapshots[-1].position,
        marker="x",
        color="#ff6b35",
        s=90,
        linewidth=3,
        label="Taxis final",
    )
    trajectory_axis.set(
        xlim=(0.0, config.arena_size),
        ylim=(0.0, config.arena_size),
        xlabel="x",
        ylabel="y",
        title="Trajectory over the odor field",
        aspect="equal",
    )
    trajectory_axis.legend(loc="upper left")

    distance_axis.plot(taxis.times, taxis.distances, color="#e4572e", label="Bilateral taxis")
    distance_axis.plot(
        baseline.times,
        baseline.distances,
        color="#4c78a8",
        linestyle="--",
        label="No steering",
    )
    distance_axis.axhline(
        config.target_radius,
        color="black",
        linestyle=":",
        label="Target radius",
    )
    distance_axis.set(
        xlabel="Time (s)",
        ylabel="Distance",
        title="Distance to source",
    )
    distance_axis.grid(alpha=0.25)
    distance_axis.legend()

    concentration_axis.plot(
        taxis.times,
        taxis.center_concentrations,
        color="#54a24b",
        label="Bilateral taxis",
    )
    concentration_axis.plot(
        baseline.times,
        baseline.center_concentrations,
        color="#4c78a8",
        linestyle="--",
        label="No steering",
    )
    concentration_axis.set(
        xlabel="Time (s)",
        ylabel="Concentration",
        title="Experienced concentration",
    )
    concentration_axis.grid(alpha=0.25)
    concentration_axis.legend()

    curvature_axis.plot(
        taxis.times[:-1],
        taxis.curvatures,
        color="#b279a2",
        label="Bilateral taxis",
    )
    curvature_axis.axhline(0.0, color="black", linewidth=1.0)
    curvature_axis.set(
        xlabel="Time (s)",
        ylabel="Curvature",
        title="Curvature command",
    )
    curvature_axis.grid(alpha=0.25)
    curvature_axis.legend()

    success_label = "reached" if taxis.success else "missed"
    figure.suptitle(
        f"Curvature taxis ({config.initial_heading_degrees:g}° start): controller {success_label} "
        f"the source",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


__all__ = [
    "Controller",
    "CurvatureTaxisConfig",
    "CurvatureTaxisEnvironment",
    "HeadingSweepRow",
    "HeadingSweepSummary",
    "Observation",
    "TaxisSnapshot",
    "TaxisTrace",
    "TaxisTransition",
    "TerminationReason",
    "bilateral_curvature",
    "no_steering",
    "run_episode",
    "run_heading_sweep",
    "run_no_steering",
    "run_taxis",
    "save_demo_artifacts",
    "summarize_heading_sweep",
]
