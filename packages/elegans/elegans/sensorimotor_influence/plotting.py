"""Publication-style figures for the sensorimotor-influence toy study."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .analysis import phase_mask, rolling_mean

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .simulation import SimulationTrace
    from .study import StudyResults

FloatArray = NDArray[np.float64]

BLIND_COLOR = "#6A51A3"
AWARE_COLOR = "#0072B2"
INFLUENCE_COLOR = "#009E73"
VIGOR_COLOR = "#D55E00"
ACTION_COLOR = "#CC79A7"
NEUTRAL_COLOR = "#4D4D4D"
MIDDLE_SHADE = "#D9D9D9"


def make_all_figures(results: StudyResults, output_directory: Path) -> list[Path]:
    """Create the four primary figures and the reversal diagnostic."""
    output_directory.mkdir(parents=True, exist_ok=True)
    default_probe = results.configs["behavior"].gate.probe_probability
    stems = [
        output_directory / "figure1_cable_vigor_trace",
        output_directory / "figure2_yoked_distributions",
        output_directory / "figure3_probe_recovery",
        output_directory / "figure4_gain_noise_robustness",
        output_directory / "figure5_reversed_mapping",
    ]
    with plt.rc_context(_plot_style()):
        plot_cable_vigor_trace(results.behavior_by_probe[default_probe], stems[0])
        plot_yoked_distributions(results, stems[1])
        plot_probe_recovery(results, stems[2])
        plot_robustness_heatmaps(results, stems[3])
        plot_reversed_mapping(results.estimator_reversed, stems[4])
    return [path.with_suffix(".png") for path in stems]


def plot_reversed_mapping(traces: Sequence[SimulationTrace], output_stem: Path) -> None:
    """Separate transient expected-feedback failure from steady influence."""
    if not traces:
        message = "at least one trace is required"
        raise ValueError(message)
    trace = traces[0]
    window = max(20, len(trace.time) // 150)
    figure, axes = plt.subplots(4, 1, figsize=(10.5, 7.6), sharex=True, constrained_layout=True)
    axes[0].step(
        trace.time,
        trace.coupling_coefficient,
        where="post",
        color=NEUTRAL_COLOR,
    )
    axes[0].set_ylabel("Own-action\ngain")
    axes[0].set_yticks([-1.0, 0.0, 1.0])
    _plot_ensemble(
        axes[1],
        traces,
        lambda item: rolling_mean(item.squared_error_aware, window),
        AWARE_COLOR,
        "action-aware squared error",
    )
    axes[1].set_ylabel("Forward-model\nsquared error")
    _plot_ensemble(
        axes[2],
        traces,
        lambda item: rolling_mean(item.evidence, window),
        NEUTRAL_COLOR,
        "unclipped d",
        linestyle="--",
    )
    _plot_ensemble(
        axes[2],
        traces,
        lambda item: item.influence,
        INFLUENCE_COLOR,
        "influence q",
    )
    axes[2].axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    axes[2].set_ylabel("Predictive\nadvantage (nats)")
    axes[2].legend(frameon=False, ncol=2, loc="upper right")
    _plot_ensemble(
        axes[3],
        traces,
        lambda item: item.learned_action_weight,
        ACTION_COLOR,
        "learned action coefficient",
    )
    axes[3].axhline(1.0, color=NEUTRAL_COLOR, linestyle=":", linewidth=0.8)
    axes[3].axhline(-1.0, color=NEUTRAL_COLOR, linestyle=":", linewidth=0.8)
    axes[3].set_ylabel("Learned action\ncoefficient")
    axes[3].set_xlabel("Transition")
    _decorate_phases(axes, trace)
    figure.suptitle(
        "Reversal causes transient mismatch but preserves steady action influence",
        fontweight="bold",
    )
    _save_figure(figure, output_stem)


def plot_cable_vigor_trace(traces: Sequence[SimulationTrace], output_stem: Path) -> None:
    """Plot coupling, predictor losses, influence, vigor, and action magnitude."""
    if not traces:
        message = "at least one trace is required"
        raise ValueError(message)
    trace = traces[0]
    steps = trace.time
    window = max(20, len(steps) // 150)
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(11.0, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.65, 1.3, 1.15, 1.0, 1.0]},
        constrained_layout=True,
    )

    axes[0].step(steps, trace.coupling_coefficient, where="post", color=NEUTRAL_COLOR)
    axes[0].set_ylabel("Own-action\ngain")
    axes[0].set_yticks([0.0, 1.0])

    _plot_ensemble(
        axes[1],
        traces,
        lambda item: rolling_mean(item.loss_blind, window),
        BLIND_COLOR,
        "action-blind NLL",
    )
    _plot_ensemble(
        axes[1],
        traces,
        lambda item: rolling_mean(item.loss_aware, window),
        AWARE_COLOR,
        "action-aware NLL",
    )
    axes[1].set_ylabel("Pre-update loss\n(nats)")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    _plot_ensemble(
        axes[2],
        traces,
        lambda item: rolling_mean(item.evidence, window),
        NEUTRAL_COLOR,
        "unclipped loss advantage d",
        linestyle="--",
    )
    _plot_ensemble(
        axes[2],
        traces,
        lambda item: item.influence,
        INFLUENCE_COLOR,
        "estimated influence",
    )
    axes[2].axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("Smoothed loss\nadvantage q (nats)")
    axes[2].legend(frameon=False, ncol=2, loc="upper right")

    _plot_ensemble(
        axes[3],
        traces,
        lambda item: rolling_mean(item.vigor, window),
        VIGOR_COLOR,
        "vigor",
    )
    axes[3].set_ylabel("Vigor m")
    axes[3].set_ylim(-0.03, 1.05)

    _plot_ensemble(
        axes[4],
        traces,
        lambda item: np.sqrt(rolling_mean(item.action**2, window)),
        ACTION_COLOR,
        "action RMS",
    )
    axes[4].set_ylabel("Action RMS")
    axes[4].set_xlabel("Transition")

    _decorate_phases(axes, trace)
    figure.suptitle(
        "Learned action-outcome contingency reduces and restores behavioral vigor",
        fontweight="bold",
    )
    _save_figure(figure, output_stem)


def plot_yoked_distributions(results: StudyResults, output_stem: Path) -> None:
    """Compare phase-level influence and sensory variance across controls."""
    burn_in = results.configs["estimator_yoked"].phase_metric_burn_in
    connected = [
        float(np.mean(trace.influence[phase_mask(trace, 0, burn_in)]))
        for trace in results.estimator_yoked
    ]
    disconnected = [
        float(np.mean(trace.influence[phase_mask(trace, 1, burn_in)]))
        for trace in results.estimator_disconnected
    ]
    yoked = [
        float(np.mean(trace.influence[phase_mask(trace, 1, burn_in)]))
        for trace in results.estimator_yoked
    ]
    connected_variance = [
        float(np.var(trace.state[1:][phase_mask(trace, 0, burn_in)], ddof=1))
        for trace in results.estimator_yoked
    ]
    disconnected_variance = [
        float(np.var(trace.state[1:][phase_mask(trace, 1, burn_in)], ddof=1))
        for trace in results.estimator_disconnected
    ]
    yoked_variance = [
        float(np.var(trace.state[1:][phase_mask(trace, 1, burn_in)], ddof=1))
        for trace in results.estimator_yoked
    ]
    labels = ("Connected", "Disconnected", "Yoked")
    colors = (INFLUENCE_COLOR, "#999999", "#E69F00")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    _distribution_panel(
        axes[0],
        [connected, disconnected, yoked],
        labels,
        colors,
        ylabel="Phase-mean influence q (nats)",
    )
    target = float(
        np.mean(
            results.estimator_yoked[0].true_information[
                phase_mask(results.estimator_yoked[0], 0, burn_in)
            ],
        ),
    )
    axes[0].axhline(
        target,
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"analytic target = {target:.2f}",
    )
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("Action-specific predictive advantage")

    _distribution_panel(
        axes[1],
        [connected_variance, disconnected_variance, yoked_variance],
        labels,
        colors,
        ylabel="Sensory-state variance",
    )
    axes[1].set_title("Yoking preserves sensory variability")
    figure.suptitle(
        "Matched yoking removes contingency, not environmental activity",
        fontweight="bold",
    )
    _save_figure(figure, output_stem)


def plot_probe_recovery(results: StudyResults, output_stem: Path) -> None:
    """Plot paired, right-censored recovery latency across probe rates."""
    probabilities = (0.0, 0.03, 0.05)
    labels = ("Probes withdrawn\nafter phase 1", "p = 0.03", "p = 0.05")
    phase_length = results.configs["behavior"].phases[2].steps
    rows_by_probability = {
        probability: sorted(
            [
                row
                for row in results.latency_rows
                if row["series"] == "vigor"
                and np.isclose(float(row["probe_probability"]), probability)
            ],
            key=lambda row: int(row["seed"]),
        )
        for probability in probabilities
    }
    latency = np.asarray(
        [
            [float(row["recovery_latency_censored"]) for row in rows_by_probability[p]]
            for p in probabilities
        ],
        dtype=np.float64,
    ).T
    failures = np.asarray(
        [
            [bool(int(row["recovery_failed"])) for row in rows_by_probability[p]]
            for p in probabilities
        ],
        dtype=np.bool_,
    ).T

    figure, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    x = np.arange(len(probabilities), dtype=np.float64)
    for seed_index in range(latency.shape[0]):
        axis.plot(x, latency[seed_index], color="#BDBDBD", linewidth=0.7, alpha=0.55)
        for condition_index in range(len(probabilities)):
            marker = "^" if failures[seed_index, condition_index] else "o"
            axis.scatter(
                x[condition_index],
                latency[seed_index, condition_index],
                color=INFLUENCE_COLOR,
                marker=marker,
                s=20,
                alpha=0.7,
                zorder=3,
            )
    medians = np.median(latency, axis=0)
    axis.scatter(x, medians, color=NEUTRAL_COLOR, marker="_", s=500, linewidth=3, zorder=4)
    for index, _probability in enumerate(probabilities):
        failed = int(np.sum(failures[:, index]))
        axis.text(
            x[index],
            phase_length * 1.04,
            f"{failed}/{latency.shape[0]} censored",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.axhline(
        phase_length,
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=0.9,
        label="reconnection phase ended",
    )
    axis.set_xticks(x, labels)
    # Leave room for the first censored-count label so it does not collide with
    # the y-axis tick labels in either the raster or vector export.
    axis.set_xlim(-0.35, 2.35)
    axis.set_ylabel("Recovery latency after reconnection (steps)")
    axis.set_ylim(0.0, phase_length * 1.18)
    axis.set_title("Fixed-amplitude probes prevent self-sealing passivity", fontweight="bold")
    axis.legend(frameon=False, loc="upper right")
    _save_figure(figure, output_stem)


def plot_robustness_heatmaps(results: StudyResults, output_stem: Path) -> None:
    """Plot analytic, estimated, error, and discrimination robustness maps."""
    gains = sorted({float(row["gain"]) for row in results.robustness_rows})
    noises = sorted({float(row["noise_std"]) for row in results.robustness_rows})
    analytic = _grid(results.robustness_rows, gains, noises, "analytic_information")
    estimated = _grid(
        results.robustness_rows,
        gains,
        noises,
        "mean_connected_loss_advantage",
    )
    calibration_error = _grid(
        results.robustness_rows,
        gains,
        noises,
        "absolute_calibration_error",
    )
    auc = _grid(results.robustness_rows, gains, noises, "roc_auc")

    figure, axes_grid = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    axes = list(axes_grid.flat)
    panels = (
        (analytic, "Analytic conditional information", "nats", "viridis", 0.0, None),
        (estimated, "Mean unclipped loss advantage", "nats", "viridis", 0.0, None),
        (calibration_error, "Absolute calibration error", "nats", "magma", 0.0, None),
        (auc, "Coupled-vs-yoked ROC-AUC", "seed-level ROC-AUC", "cividis", 0.5, 1.0),
    )
    for axis, (values, title, colorbar_label, cmap, minimum, maximum) in zip(
        axes,
        panels,
        strict=True,
    ):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            vmin=minimum,
            vmax=maximum,
            cmap=cmap,
        )
        _label_heatmap(axis, values, gains, noises, formatter=lambda value: f"{value:.2f}")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.82, label=colorbar_label)
        axis.set_xlabel("Absolute action gain")
        axis.set_ylabel("Process-noise standard deviation")
    figure.suptitle(
        "Analytic calibration and discrimination across gain and noise",
        fontweight="bold",
    )
    _save_figure(figure, output_stem)


def _plot_ensemble(  # noqa: PLR0913
    axis: Axes,
    traces: Sequence[SimulationTrace],
    extractor: Callable[[SimulationTrace], FloatArray],
    color: str,
    label: str,
    *,
    linestyle: str = "-",
) -> None:
    values = np.stack([extractor(trace) for trace in traces])
    mean = np.mean(values, axis=0)
    if values.shape[0] > 1:
        half_width = 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        axis.fill_between(
            traces[0].time,
            mean - half_width,
            mean + half_width,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
    axis.plot(
        traces[0].time,
        mean,
        color=color,
        linewidth=1.35,
        linestyle=linestyle,
        label=label,
    )


def _decorate_phases(axes: Sequence[Axes], trace: SimulationTrace) -> None:
    boundaries = [0] + [
        int(np.flatnonzero(trace.phase_index == phase)[-1]) + 1
        for phase in range(len(trace.phase_names))
    ]
    for axis in axes:
        axis.axvspan(boundaries[1], boundaries[2], color=MIDDLE_SHADE, alpha=0.24, zorder=-5)
        for boundary in boundaries[1:-1]:
            axis.axvline(boundary, color=NEUTRAL_COLOR, linewidth=0.8, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.5)
    label_axis = axes[0]
    y_top = label_axis.get_ylim()[1]
    for index, name in enumerate(trace.phase_names):
        center = 0.5 * (boundaries[index] + boundaries[index + 1])
        label_axis.text(
            center,
            y_top,
            name.replace("_", " "),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _distribution_panel(
    axis: Axes,
    data: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    ylabel: str,
) -> None:
    boxes = axis.boxplot(
        data,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": NEUTRAL_COLOR, "linewidth": 1.6},
        whiskerprops={"color": NEUTRAL_COLOR},
        capprops={"color": NEUTRAL_COLOR},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(4_291)
    for index, (values, color) in enumerate(zip(data, colors, strict=True), start=1):
        jitter = rng.normal(0.0, 0.045, len(values))
        axis.scatter(
            index + jitter,
            values,
            s=18,
            color=color,
            edgecolor="none",
            alpha=0.75,
            zorder=3,
        )
    axis.set_xticks(np.arange(1, len(labels) + 1), labels)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.5)


def _grid(
    rows: Sequence[dict[str, float | int]],
    gains: Sequence[float],
    noises: Sequence[float],
    metric: str,
) -> FloatArray:
    lookup = {(float(row["gain"]), float(row["noise_std"])): float(row[metric]) for row in rows}
    return np.asarray(
        [[lookup[(gain, noise)] for gain in gains] for noise in noises],
        dtype=np.float64,
    )


def _label_heatmap(
    axis: Axes,
    values: FloatArray,
    gains: Sequence[float],
    noises: Sequence[float],
    *,
    formatter: Callable[[float], str],
) -> None:
    axis.set_xticks(np.arange(len(gains)), [f"{gain:g}" for gain in gains])
    axis.set_yticks(np.arange(len(noises)), [f"{noise:g}" for noise in noises])
    midpoint = 0.5 * (float(np.nanmin(values)) + float(np.nanmax(values)))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            text_color = "white" if value < midpoint else "black"
            axis.text(
                column,
                row,
                formatter(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )


def _save_figure(figure: Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_style() -> mpl.RcParams:
    return mpl.RcParams(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.linewidth": 0.8,
        },
    )
