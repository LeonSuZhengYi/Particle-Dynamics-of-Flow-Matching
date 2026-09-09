from pathlib import Path

from output_paths import output_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from sec4 import (
    T0,
    add_filled_boundary,
    make_bean_boundary,
    plot_closed_polyline,
    project_to_filled_polygon,
)


OUT_PATH = output_path("sec4_projection_ode_discrete.png")
DIST_OUT_PATH = output_path("sec4_projection_ode_discrete_distance.png")
T_END = 1.0
DT = 0.1


def make_initial_condition(boundary):
    starts = boundary
    ends = np.roll(boundary, -1, axis=0)
    midpoints = 0.5 * (starts + ends)
    score = midpoints[:, 0] - 1.45 * midpoints[:, 1]
    idx = np.argmax(score)

    ybar = midpoints[idx]
    tangent = ends[idx] - starts[idx]
    tangent = tangent / np.linalg.norm(tangent)
    candidates = [
        np.array([tangent[1], -tangent[0]]),
        np.array([-tangent[1], tangent[0]]),
    ]
    center = boundary.mean(axis=0)
    normal = max(candidates, key=lambda candidate: np.dot(candidate, ybar - center))
    x0 = T0 * (ybar + 2.35 * normal)
    return x0, ybar


def run_projection_euler(boundary, x0, dt=DT, t0=T0, t_end=T_END):
    ts = [t0]
    xs = [x0.copy()]
    projections = []
    yps = []

    t = t0
    x = x0.copy()
    while True:
        projection = project_to_filled_polygon(x, t * boundary)
        yp = projection / t
        projections.append(projection.copy())
        yps.append(yp.copy())

        if t >= t_end - 1e-12:
            break

        step = min(dt, t_end - t)
        x = x + step * (yp - x) / (1.0 - t)
        t += step
        ts.append(t)
        xs.append(x.copy())

    return np.asarray(ts), np.asarray(xs), np.asarray(projections), np.asarray(yps)


def add_polyline_segments(ax, points, color, linewidth=3.0, zorder=10):
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=0.9,
        zorder=zorder,
    )
    collection.set_capstyle("round")
    ax.add_collection(collection)
    return collection


def add_sparse_arrows(ax, points, color, every=2):
    for i in range(0, len(points) - 1, every):
        start = points[i]
        end = points[i + 1]
        direction = end - start
        p0 = start + 0.52 * direction
        p1 = start + 0.72 * direction
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.6,
                "mutation_scale": 13,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=14,
        )


def add_coordinate_axes(ax):
    axis_color = "#111111"
    ax.annotate(
        "",
        xy=(5.55, 0),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "lw": 2.1, "color": axis_color},
        zorder=0,
    )
    ax.annotate(
        "",
        xy=(0, 5.55),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "lw": 2.1, "color": axis_color},
        zorder=0,
    )
    ax.plot([0, 5.25], [0, 5.25], "--", color=axis_color, linewidth=1.4, alpha=0.45)
    ax.text(-0.03, -0.19, "O", color=axis_color, fontsize=15, ha="center", va="top")
    ax.text(5.48, 0.12, "X", color=axis_color, fontsize=15, ha="right", va="bottom")
    ax.text(0.12, 5.48, "Y", color=axis_color, fontsize=15, ha="left", va="top")


def add_projection_line_legend(ax, colors, times):
    x0, y0 = 0.07, 0.855
    line_height = 0.045
    entries = [
        ("#2f7f42", "-", r"$D$"),
       # ("#d48b21", ":", r"fixed data-space ray through $\bar y$"),
        ("#707070", ":", r"$x_i$ to $\mathrm{Proj}_{t_iD}(x_i)$"),
    ]
    for color, time in zip(colors, times):
        entries.append((color, "--", rf"$t_iD$, $t_i={time:.1f}$"))

    for i, (color, linestyle, text) in enumerate(entries):
        y = y0 - i * line_height
        ax.plot(
            [x0, x0 + 0.043],
            [y, y],
            transform=ax.transAxes,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            solid_capstyle="round",
            zorder=30,
        )
        ax.text(
            x0 + 0.056,
            y,
            text,
            transform=ax.transAxes,
            fontsize=10.7,
            color="#222222",
            ha="left",
            va="center",
            zorder=30,
        )


def add_projection_marker_legend(ax, colors, times):
    x0, y0 = 0.675, 0.335
    line_height = 0.047
    entries = [
        ("step", "#2f71b8", "o", r"$x_i$ to $x_{i+1}$"),
        ("marker", "#2f71b8", "o", r"Euler vertex $x_i$"),
    ]
    for color, time in zip(colors, times):
        entries.append(
            ("marker", color, "D", rf"$t_i y^p_{{t_i}}(x_i)$, $t_i={time:.1f}$")
        )

    for i, (kind, color, marker, text) in enumerate(entries):
        y = y0 - i * line_height
        if kind == "step":
            ax.plot(
                [x0, x0 + 0.050],
                [y, y],
                transform=ax.transAxes,
                color=color,
                linewidth=2.5,
                marker=marker,
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.5,
                markeredgecolor=color,
                solid_capstyle="round",
                zorder=30,
            )
        else:
            ax.scatter(
                [x0 + 0.025],
                [y],
                transform=ax.transAxes,
                s=62,
                marker=marker,
                facecolor="white" if marker == "o" else color,
                edgecolor=color,
                linewidth=1.7,
                zorder=30,
            )
        ax.text(
            x0 + 0.062,
            y,
            text,
            transform=ax.transAxes,
            fontsize=10.8,
            color="#222222",
            ha="left",
            va="center",
            zorder=30,
        )


def plot_projection_trajectory(ts, xs, scaled_projections, ybar, boundary, max_y_error):
    fig, ax = plt.subplots(figsize=(8.7, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    green = "#2f7f42"
    blue = "#2f71b8"
    orange = "#d48b21"
    grey = "#707070"
    black = "#111111"
    snapshot_colors = ["#d28b21", "#73a942", "#2f71b8", "#8e63b7"]
    snapshot_indices = [0, 2, 4, len(ts) - 1]
    snapshot_indices = sorted(set(idx for idx in snapshot_indices if idx < len(ts)))
    snapshot_times = [ts[idx] for idx in snapshot_indices]

    add_coordinate_axes(ax)

    add_filled_boundary(
        ax,
        boundary,
        facecolor="#ffffff",
        edgecolor=green,
        alpha=0.98,
        linewidth=2.2,
        label=None,
    )
    ax.fill(boundary[:, 0], boundary[:, 1], color=green, alpha=0.045)

    for idx, color in zip(snapshot_indices, snapshot_colors):
        t = ts[idx]
        scaled_boundary = t * boundary
        add_filled_boundary(
            ax,
            scaled_boundary,
            facecolor=color,
            edgecolor="none",
            alpha=0.030,
            linewidth=0.0,
            label=None,
        )
        plot_closed_polyline(
            ax,
            scaled_boundary,
            color=color,
            linewidth=2.1 if idx in (0, len(ts) - 1) else 1.7,
            alpha=0.82,
            linestyle="--",
            zorder=3,
        )

    ax.plot(
        [0.0, ybar[0]],
        [0.0, ybar[1]],
        linestyle=":",
        linewidth=2.4,
        color=orange,
        alpha=0.85,
        zorder=3,
    )
    ax.scatter(
        [ybar[0]],
        [ybar[1]],
        s=92,
        marker="D",
        facecolor=orange,
        edgecolor="white",
        linewidth=1.4,
        zorder=13,
    )

    for idx in snapshot_indices:
        x = xs[idx]
        proj = scaled_projections[idx]
        ax.plot(
            [x[0], proj[0]],
            [x[1], proj[1]],
            linestyle=":",
            color=grey,
            linewidth=1.1,
            alpha=0.56,
            zorder=4,
        )

    ax.plot(
        xs[:, 0],
        xs[:, 1],
        color="#2f71b8",
        linewidth=1.1,
        alpha=0.35,
        linestyle="--",
        zorder=8,
    )
    add_polyline_segments(ax, xs, blue, linewidth=3.2, zorder=10)
    add_sparse_arrows(ax, xs, blue, every=2)

    ax.scatter(
        xs[:, 0],
        xs[:, 1],
        s=72,
        facecolor="white",
        edgecolor=blue,
        linewidth=2.0,
        zorder=15,
    )
    for idx, color in zip(snapshot_indices, snapshot_colors):
        ax.scatter(
            [scaled_projections[idx, 0]],
            [scaled_projections[idx, 1]],
            s=118,
            marker="D",
            facecolor=color,
            edgecolor="white",
            linewidth=1.3,
            zorder=17,
        )

    ax.text(
        xs[0, 0] - 0.0,
        xs[0, 1] - 0.0,
        r"$x_{t_0}$",
        color=black,
        fontsize=15,
        ha="right",
        va="top",
    )
    ax.text(
        xs[-1, 0] + 0.08,
        xs[-1, 1] - 0.1,
        r"$x_1$",
        color=blue,
        fontsize=15,
        ha="left",
        va="bottom",
    )
    ax.text(
        4.15 + 0.18,
        4.15 - 0.20,
        r"$D$",
        color=black,
        fontsize=15,
        ha="left",
        va="top",
    )
    ax.text(
        T0 * 4.15 -0.2,
        T0 * 4.15 + 0.00,
        r"$t_0D$",
        color= orange,
        fontsize=13,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.07,
        0.905,
        rf"$\Delta t=0.1$; $t_0 = 0.3$",
        transform=ax.transAxes,
        color=black,
        fontsize=14.5,
        ha="left",
        va="top",
    )
    add_projection_line_legend(
        ax,
        snapshot_colors[: len(snapshot_indices)],
        snapshot_times,
    )
    add_projection_marker_legend(
        ax,
        snapshot_colors[: len(snapshot_indices)],
        snapshot_times,
    )

    ax.set_xlim(-0.20, 5.60)
    ax.set_ylim(-0.30, 5.60)
    ax.axis("off")

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_PATH, dpi=240)
    plt.close(fig)


def plot_projection_distance(ts, xs, scaled_projections):
    fig, dist_ax = plt.subplots(figsize=(7.8, 8.2))
    fig.patch.set_facecolor("white")
    dist_ax.set_facecolor("white")

    blue = "#2f71b8"
    distances = np.linalg.norm(xs - scaled_projections, axis=1)
    theoretical = distances[0] * (1.0 - ts) / (1.0 - ts[0])
    dist_ax.plot(
        ts,
        theoretical,
        color="#9a9a9a",
        linestyle="--",
        linewidth=2.0,
        label=r"exact linear rate",
    )
    dist_ax.plot(
        ts,
        distances,
        color=blue,
        linewidth=2.2,
        marker="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label=r"$\mathrm{dist}(x_i,t_iD)$",
    )
    dist_ax.set_title("Discrete distance decay", fontsize=14, pad=9)
    dist_ax.set_xlabel(r"$t_i$", fontsize=13)
    dist_ax.set_ylabel(r"$\mathrm{dist}(x_i,t_iD)$", fontsize=13)
    dist_ax.tick_params(axis="both", labelsize=11)
    dist_ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.62)
    dist_ax.legend(frameon=False, fontsize=11.5, loc="upper right")
    dist_ax.set_xlim(ts[0] - 0.015, ts[-1] + 0.015)
    dist_ax.set_ylim(bottom=-0.015 * distances.max(), top=1.10 * distances.max())

    fig.tight_layout()
    fig.savefig(DIST_OUT_PATH, dpi=240)
    plt.close(fig)


def main():
    boundary = make_bean_boundary()
    x0, ybar = make_initial_condition(boundary)
    ts, xs, scaled_projections, yps = run_projection_euler(boundary, x0)
    max_y_error = np.max(np.linalg.norm(yps - ybar[None, :], axis=1))
    plot_projection_trajectory(ts, xs, scaled_projections, ybar, boundary, max_y_error)
    plot_projection_distance(ts, xs, scaled_projections)
    print(f"saved {OUT_PATH}")
    print(f"saved {DIST_OUT_PATH}")
    print(f"max projection-coordinate error: {max_y_error:.6e}")


if __name__ == "__main__":
    main()
