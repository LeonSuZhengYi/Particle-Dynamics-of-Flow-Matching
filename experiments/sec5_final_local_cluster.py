from pathlib import Path

from output_paths import output_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch

from sec4 import (
    closed_path,
    distance_to_filled_polygon,
    outward_normal,
    sample_uniform_in_polygon,
)


OUT_PATH = output_path("cluster_z.png")
DIST_OUT_PATH = output_path("dist_cluster_z.png")

SEED = 20260514
T_START = 0.40
T0 = 0.75
T_END = 0.998
DT = 0.05
BACKWARD_DT = DT
N_PER_CLUSTER = 500
NEIGHBORHOOD_R = 0.025
N_TRAJECTORIES = 5
SHOW_DISCRETE_NODES = True
FLOW_AXIS_FONT_SIZE = 24
FLOW_TRAJECTORY_AXIS_LABEL_FONT_SIZE = 20
FLOW_TRAJECTORY_TICK_FONT_SIZE = 18
FLOW_CLUSTER_LABEL_FONT_SIZE = 24
FLOW_LABEL_FONT_SIZE = 21
FLOW_INFO_FONT_SIZE = 18
FLOW_LEGEND_FONT_SIZE = 18
DIST_TITLE_FONT_SIZE = 20
DIST_AXIS_LABEL_FONT_SIZE = 20
DIST_TICK_FONT_SIZE = 18
DIST_LEGEND_FONT_SIZE = 21
DIST_ANNOTATION_FONT_SIZE = 18

# Direct inputs for the five displayed trajectories at the initial time T_START.
# Tweak these coordinates to adjust the five x^j_{0.4} positions.
INITIAL_POINTS_T_START = np.array(
    [
        [1.43009576, 0.75021864],
        [2.87208128, 0.88736968],
        [3.68386128, 2.26294420],
        [2.51657135, 3.51145371],
        [0.70763986, 2.64791343],
    ]
)


def angle_wrap(theta):
    return np.angle(np.exp(1j * theta))


def rotate(points, degrees):
    angle = np.deg2rad(degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return points @ rotation.T


def make_nonconvex_boundary(center, scale=(1.0, 0.78), rotation=0.0, phase=0.0, n=560):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dimple_a = np.exp(-(angle_wrap(theta - phase) ** 2) / (2.0 * 0.34**2))
    dimple_b = np.exp(-(angle_wrap(theta - phase - 2.05) ** 2) / (2.0 * 0.30**2))
    dimple_c = np.exp(-(angle_wrap(theta - phase + 2.15) ** 2) / (2.0 * 0.28**2))
    radius = (
        1.0
        - 0.47 * dimple_a
        - 0.29 * dimple_b
        - 0.23 * dimple_c
        + 0.16 * np.sin(3.0 * theta + 0.7 * phase)
        + 0.09 * np.cos(5.0 * theta - 0.4)
    )
    local = np.column_stack([scale[0] * radius * np.cos(theta), scale[1] * radius * np.sin(theta)])
    return rotate(local, rotation) + np.asarray(center)


def make_cluster_boundaries():
    specs = [
        ((4.25, 3.85), (0.62, 0.45), -22.0, 0.15),
        ((6.28, 3.85), (0.65, 0.46), 18.0, 1.00),
        ((6.92, 5.76), (0.63, 0.45), -32.0, -0.65),
        ((5.26, 6.88), (0.61, 0.44), 33.0, 2.10),
        ((3.60, 5.76), (0.63, 0.45), 16.0, 2.85),
    ]
    return [
        make_nonconvex_boundary(center, scale=scale, rotation=rotation, phase=phase)
        for center, scale, rotation, phase in specs
    ]


def sample_clusters(boundaries):
    samples = []
    labels = []
    for cluster_id, boundary in enumerate(boundaries):
        cluster_samples = sample_uniform_in_polygon(
            boundary,
            N_PER_CLUSTER,
            seed=SEED + 101 * cluster_id,
        )
        samples.append(cluster_samples)
        labels.append(np.full(len(cluster_samples), cluster_id, dtype=int))
    return np.vstack(samples), np.concatenate(labels)


def posterior_mean_and_cluster_mass(x, t, samples, labels, n_clusters):
    residual = x[None, :] - t * samples
    log_weights = -np.sum(residual * residual, axis=1) / (2.0 * (1.0 - t) ** 2)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weight_sum = np.sum(weights)
    mean = np.sum(weights[:, None] * samples, axis=0) / weight_sum
    masses = np.bincount(labels, weights=weights, minlength=n_clusters) / weight_sum
    return mean, masses


def make_initial_condition(boundaries, cluster_id):
    boundary = boundaries[cluster_id]
    center = boundary.mean(axis=0)
    global_center = np.mean([item.mean(axis=0) for item in boundaries], axis=0)
    # Pick the outward-facing side of the target cluster, then start just
    # outside the scaled cluster. This mimics a final-stage restart point.
    direction = center - global_center
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1.0, -0.35])
    direction = direction / np.linalg.norm(direction)
    score = (boundary - center) @ direction
    idx = int(np.argmax(score))
    normal = outward_normal(boundary, idx)
    offset_factors = [1.05, 0.46, 0.66, 0.78, 0.36]
    offset = offset_factors[cluster_id % len(offset_factors)] * NEIGHBORHOOD_R
    return T0 * (boundary[idx] + offset * normal)


def flow_velocity(x, t, samples, labels, n_clusters):
    mean, masses = posterior_mean_and_cluster_mass(x, t, samples, labels, n_clusters)
    return (mean - x) / (1.0 - t), masses


def run_forward_flow(samples, labels, boundaries, x0, t0=T0, t_end=T_END, dt=DT):
    n_clusters = len(boundaries)
    ts = [t0]
    xs = [x0.copy()]

    t = t0
    x = x0.copy()
    while t < t_end - 1e-12:
        step = min(dt, t_end - t)
        velocity, _ = flow_velocity(x, t, samples, labels, n_clusters)
        x = x + step * velocity
        t += step
        ts.append(t)
        xs.append(x.copy())

    return np.asarray(ts), np.asarray(xs)


def run_backward_flow(samples, labels, boundaries, x0, t0=T0, t_start=T_START, dt=BACKWARD_DT):
    n_clusters = len(boundaries)
    ts = [t0]
    xs = [x0.copy()]

    t = t0
    x = x0.copy()
    while t > t_start + 1e-12:
        step = min(dt, t - t_start)
        velocity, _ = flow_velocity(x, t, samples, labels, n_clusters)
        x = x - step * velocity
        t -= step
        ts.append(t)
        xs.append(x.copy())

    return np.asarray(ts[::-1]), np.asarray(xs[::-1])


def cluster_masses_along_path(ts, xs, samples, labels, n_clusters):
    masses = []
    for t, x in zip(ts, xs):
        _, cluster_mass = posterior_mean_and_cluster_mass(x, t, samples, labels, n_clusters)
        masses.append(cluster_mass)
    return np.asarray(masses)


def cluster_distances_to_neighborhood(ts, xs, boundaries, radius=NEIGHBORHOOD_R):
    distances = np.zeros((len(ts), len(boundaries)))
    raw_distances = np.zeros_like(distances)
    for i, (t, x) in enumerate(zip(ts, xs)):
        for j, boundary in enumerate(boundaries):
            raw = distance_to_filled_polygon(x, t * boundary)
            raw_distances[i, j] = raw
            distances[i, j] = max(raw - t * radius, 0.0)
    return distances, raw_distances


def first_entry_indices(neighborhood_distances):
    entries = []
    for j in range(neighborhood_distances.shape[1]):
        inside = neighborhood_distances[:, j] <= 1e-10
        entries.append(int(np.argmax(inside)) if np.any(inside) else None)
    return entries


def polygon_patch(points, **kwargs):
    return PathPatch(closed_path(points), joinstyle="round", **kwargs)


def add_cluster_neighborhood_contour(
    ax,
    boundary,
    radius,
    color,
    scale=1.0,
    linestyle="--",
    alpha=0.12,
    line_alpha=0.58,
    zorder=1,
):
    scaled = scale * boundary
    pad = max(scale * radius + 0.12, 0.18)
    lo = scaled.min(axis=0) - pad
    hi = scaled.max(axis=0) + pad
    xs = np.linspace(lo[0], hi[0], 110)
    ys = np.linspace(lo[1], hi[1], 110)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    values = np.asarray([distance_to_filled_polygon(point, scaled) for point in grid])
    values = values.reshape(xx.shape)
    level = scale * radius
    ax.contourf(xx, yy, values, levels=[0.0, level], colors=[color], alpha=alpha, zorder=zorder)
    ax.contour(
        xx,
        yy,
        values,
        levels=[level],
        colors=[color],
        linestyles=linestyle,
        linewidths=1.55,
        alpha=line_alpha,
        zorder=zorder + 1,
    )


def plot_scaled_cluster_boundary(ax, boundary, scale, color, zorder=12):
    scaled = scale * boundary
    ax.plot(
        np.r_[scaled[:, 0], scaled[0, 0]],
        np.r_[scaled[:, 1], scaled[0, 1]],
        color=color,
        linestyle="--",
        linewidth=1.55,
        alpha=0.34,
        zorder=zorder,
    )


def add_coordinate_axes(ax):
    axis_color = "#111111"
    ax.annotate("", xy=(8.75, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.1, "color": axis_color}, zorder=0)
    ax.annotate("", xy=(0, 8.45), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.1, "color": axis_color}, zorder=0)
    ax.text(
        -0.03,
        -0.18,
        "O",
        color=axis_color,
        fontsize=FLOW_AXIS_FONT_SIZE,
        ha="center",
        va="top",
    )
    ax.text(
        8.67,
        0.13,
        r"$x_1$",
        color=axis_color,
        fontsize=FLOW_AXIS_FONT_SIZE,
        ha="right",
        va="bottom",
    )
    ax.text(
        0.13,
        8.38,
        r"$x_2$",
        color=axis_color,
        fontsize=FLOW_AXIS_FONT_SIZE,
        ha="left",
        va="top",
    )


def add_colored_trajectory(ax, xs, dominant_clusters, palette):
    segments = np.stack([xs[:-1], xs[1:]], axis=1)
    colors = [palette[int(cluster_id)] for cluster_id in dominant_clusters]
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=2.8,
        alpha=0.91,
        zorder=20,
    )
    collection.set_capstyle("round")
    ax.add_collection(collection)


def summarize_run(run):
    distances = run["neighborhood_distances"]
    attracted_cluster = int(np.argmin(distances[-1]))
    entries = first_entry_indices(distances)
    entry_idx = entries[attracted_cluster]
    if entry_idx is None:
        entry_idx = int(np.argmin(np.abs(run["ts"] - T_START)))
    return attracted_cluster, entry_idx, entries


def plot_experiment(boundaries, samples, runs):
    palette = ["#287c62", "#b56c1f", "#6f5bb8", "#c93d45", "#2f6fa3"]
    summaries = [summarize_run(run) for run in runs]

    fig, ax = plt.subplots(figsize=(9.0, 7.8))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    for j, boundary in enumerate(boundaries):
        ax.add_patch(
            polygon_patch(
                boundary,
                facecolor=palette[j],
                edgecolor="none",
                linewidth=0.0,
                alpha=0.050,
                zorder=4,
            )
        )
        ax.scatter(
            samples[j * N_PER_CLUSTER : (j + 1) * N_PER_CLUSTER, 0],
            samples[j * N_PER_CLUSTER : (j + 1) * N_PER_CLUSTER, 1],
            s=2.6,
            color=palette[j],
            alpha=0.24,
            linewidths=0,
            zorder=3,
        )
        center = boundary.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            rf"$\Omega_{j+1}$",
            color=palette[j],
            fontsize=FLOW_CLUSTER_LABEL_FONT_SIZE,
            ha="center",
            va="center",
            zorder=8,
            path_effects=[pe.withStroke(linewidth=1.4, foreground="white", alpha=0.72)],
        )
        add_cluster_neighborhood_contour(
            ax,
            boundary,
            NEIGHBORHOOD_R,
            palette[j],
            scale=T_START,
            linestyle="--",
            alpha=0.006,
            line_alpha=0.16,
            zorder=2,
        )
        add_cluster_neighborhood_contour(
            ax,
            boundary,
            NEIGHBORHOOD_R,
            palette[j],
            scale=T_END,
            linestyle="--",
            alpha=0.028,
            line_alpha=0.66,
            zorder=6,
        )

    for run_id, run in enumerate(runs):
        ts = run["ts"]
        xs = run["xs"]
        attracted_cluster, entry_idx, _ = summaries[run_id]
        dominant_for_segments = np.full(len(xs) - 1, attracted_cluster, dtype=int)

        add_cluster_neighborhood_contour(
            ax,
            boundaries[attracted_cluster],
            NEIGHBORHOOD_R,
            palette[attracted_cluster],
            scale=ts[entry_idx],
            linestyle="--",
            alpha=0.010,
            line_alpha=0.28,
            zorder=10,
        )

        add_colored_trajectory(ax, xs, dominant_for_segments, palette)
        if SHOW_DISCRETE_NODES:
            ax.scatter(
                xs[1:-1, 0],
                xs[1:-1, 1],
                s=24,
                marker="o",
                facecolor=palette[attracted_cluster],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.96,
                zorder=24,
            )
        ax.scatter(
            [xs[entry_idx, 0]],
            [xs[entry_idx, 1]],
            s=168,
            marker="*",
            facecolor=palette[attracted_cluster],
            edgecolor="#111111",
            linewidth=1.1,
            zorder=30,
        )
        ax.scatter(
            [xs[0, 0]],
            [xs[0, 1]],
            s=118,
            facecolor="#111111",
            edgecolor="white",
            linewidth=1.2,
            zorder=28,
        )
        ax.scatter(
            [xs[-1, 0]],
            [xs[-1, 1]],
            s=70,
            marker="s",
            facecolor=palette[attracted_cluster],
            edgecolor="white",
            linewidth=1.4,
            zorder=29,
        )
        ax.text(
            xs[0, 0],
            xs[0, 1] - 0.10,
            rf"$z_0^{{j={run_id+1}}}$",
            color="#111111",
            fontsize=FLOW_LABEL_FONT_SIZE,
            ha="center",
            va="top",
            zorder=31,
        )

    legend_x = 0.045
    legend_y = 0.905
    legend_entries = [
        (0.16, rf"$t={T_START:.1f}$ neighborhood"),
        (0.34, r"first-entry neighborhood"),
        (0.66, r"final neighborhood"),
    ]
    for i, (alpha, text) in enumerate(legend_entries):
        y = legend_y - i * 0.052
        for color_id, color in enumerate(palette):
            y0 = y + (color_id - 2) * 0.0026
            ax.plot(
                [legend_x, legend_x + 0.062],
                [y0, y0],
                transform=ax.transAxes,
                color=color,
                linestyle=(0, (4.0, 2.2)),
                linewidth=1.35,
                alpha=alpha,
                zorder=41,
            )
        ax.text(
            legend_x + 0.078,
            y,
            text,
            transform=ax.transAxes,
            fontsize=FLOW_LEGEND_FONT_SIZE,
            color="#222222",
            ha="left",
            va="center",
            zorder=41,
        )

    ax.set_xlabel(r"$x_1$", fontsize=FLOW_TRAJECTORY_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(r"$x_2$", fontsize=FLOW_TRAJECTORY_AXIS_LABEL_FONT_SIZE)
    ax.grid(True, color="#dedede", linewidth=0.8, alpha=0.52)
    ax.tick_params(axis="both", labelsize=FLOW_TRAJECTORY_TICK_FONT_SIZE)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.25, 8.85)
    ax.set_ylim(-0.35, 8.55)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=240)
    plt.close(fig)

    plot_distance_figure(runs, summaries, palette)
    return summaries


def plot_distance_figure(runs, summaries, palette):
    fig, ax = plt.subplots(figsize=(8.2, 7.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for run_id, run in enumerate(runs):
        ts = run["ts"]
        neighborhood_distances = run["neighborhood_distances"]
        attracted_cluster, entry_idx, _ = summaries[run_id]
        color = palette[attracted_cluster]
        ax.plot(
            ts,
            neighborhood_distances[:, attracted_cluster],
            color=color,
            linewidth=2.5,
            alpha=0.92,
            marker="o" if SHOW_DISCRETE_NODES else None,
            markersize=4.2 if SHOW_DISCRETE_NODES else 0.0,
            markerfacecolor=color if SHOW_DISCRETE_NODES else None,
            markeredgecolor=color if SHOW_DISCRETE_NODES else None,
            markeredgewidth=0.0 if SHOW_DISCRETE_NODES else 0.0,
            label=rf"$j={run_id+1}$",
        )
        ax.scatter(
            [ts[entry_idx]],
            [neighborhood_distances[entry_idx, attracted_cluster]],
            s=96,
            marker="*",
            facecolor=color,
            edgecolor="#111111",
            linewidth=1.0,
            zorder=8,
        )

    ax.axhline(
        0.0,
        color="#111111",
        linestyle="-",
        linewidth=1.2,
        alpha=0.36,
    )
    ax.set_xlabel(r"$t_i$", fontsize=DIST_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("distance", fontsize=DIST_AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=DIST_TICK_FONT_SIZE)
    ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.62)
    ax.set_xlim(T_START - 0.01, T_END + 0.01)
    top = max(
        np.max(run["neighborhood_distances"][:, summary[0]])
        for run, summary in zip(runs, summaries)
    )
    top = max(top, 1e-8)
    ax.set_ylim(bottom=-0.045 * top, top=1.12 * top)
    ax.legend(frameon=False, fontsize=DIST_LEGEND_FONT_SIZE, loc="upper right")

    fig.tight_layout()
    fig.savefig(DIST_OUT_PATH, dpi=240)
    plt.close(fig)


def main():
    np.random.seed(SEED)
    boundaries = make_cluster_boundaries()
    samples, labels = sample_clusters(boundaries)
    runs = []
    initial_points = np.asarray(INITIAL_POINTS_T_START, dtype=float)
    if initial_points.ndim != 2 or initial_points.shape[1] != 2:
        raise ValueError("INITIAL_POINTS_T_START must have shape (n_points, 2).")
    if len(initial_points) < N_TRAJECTORIES:
        raise ValueError("INITIAL_POINTS_T_START must provide at least N_TRAJECTORIES points.")
    for cluster_id, x_start in enumerate(initial_points[: min(N_TRAJECTORIES, len(boundaries))]):
        ts, xs = run_forward_flow(
            samples,
            labels,
            boundaries,
            x_start,
            t0=T_START,
        )
        cluster_masses = cluster_masses_along_path(
            ts,
            xs,
            samples,
            labels,
            len(boundaries),
        )
        neighborhood_distances, raw_distances = cluster_distances_to_neighborhood(ts, xs, boundaries)
        runs.append(
            {
                "target_cluster": cluster_id,
                "ts": ts,
                "xs": xs,
                "cluster_masses": cluster_masses,
                "neighborhood_distances": neighborhood_distances,
                "raw_distances": raw_distances,
            }
        )
    summaries = plot_experiment(boundaries, samples, runs)

    print(f"saved {OUT_PATH}")
    print(f"saved {DIST_OUT_PATH}")
    for run_id, (attracted_cluster, entry_idx, _) in enumerate(summaries):
        run = runs[run_id]
        print(f"trajectory {run_id + 1}: terminal attracted cluster Omega_{attracted_cluster + 1}")
        print(
            f"  first entered t_i B_r(Omega_{attracted_cluster + 1}) "
            f"at index {entry_idx}, t={run['ts'][entry_idx]:.6f}"
        )
        print(f"  terminal neighborhood distances: {np.round(run['neighborhood_distances'][-1], 6).tolist()}")
        print(f"  terminal posterior masses: {np.round(run['cluster_masses'][-1], 6).tolist()}")


if __name__ == "__main__":
    main()
