from pathlib import Path

from output_paths import output_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

from sec4 import (
    T0,
    add_filled_boundary,
    distance_to_filled_polygon,
    distances_to_scaled_convex_hull,
    plot_closed_polyline,
    run_unconditional_flow_ode,
    sample_uniform_in_polygon,
)


OUT_STEM = "sec5_flow_ode_discrete_absorption"
DIST_OUT_STEM = "sec5_flow_ode_discrete_absorption_distance"
DISCRETE_T_END = 1.0
CONTINUOUS_T_END = 0.998
N_DATA_POINTS = 500
DATA_SEED = 7
FLOW_LEGEND_FONT_SIZE = 17.0
FLOW_TITLE_FONT_SIZE = 18.0
FLOW_AXIS_FONT_SIZE = 24.0
FLOW_TRAJECTORY_AXIS_LABEL_FONT_SIZE = 20.0
FLOW_TRAJECTORY_TICK_FONT_SIZE = 18.0
FLOW_LABEL_FONT_SIZE = 21.0
FLOW_TIME_LABEL_FONT_SIZE = 17.0
DIST_TITLE_FONT_SIZE = 18.0
DIST_AXIS_LABEL_FONT_SIZE = 20.0
DIST_TICK_FONT_SIZE = 18.0
DIST_LEGEND_FONT_SIZE = 21.0
RUN_CONFIGS = (
    {
        "dt": 0.10,
        "suffix": "dt0100",
        "t_end": DISCRETE_T_END,
        "show_nodes": True,
        "use_entry_color": True,
    },
    {
        "dt": 0.05,
        "suffix": "dt0050",
        "out_name": "convexhull_z.png",
        "dist_out_name": "dist_conv_z.png",
        "t_end": DISCRETE_T_END,
        "show_nodes": True,
        "use_entry_color": True,
    },
    {
        "dt": 0.025,
        "suffix": "dt0025",
        "t_end": DISCRETE_T_END,
        "show_nodes": True,
        "use_entry_color": True,
    },
    {
        "dt": 0.002,
        "suffix": "continuous_dt0002",
        "t_end": CONTINUOUS_T_END,
        "show_nodes": False,
        "use_entry_color": False,
    },
)
CRESCENT_CENTER = np.array([4.15, 4.15])
CRESCENT_AXIS = np.array([1.0, -1.45])
CRESCENT_AXIS = CRESCENT_AXIS / np.linalg.norm(CRESCENT_AXIS)
CRESCENT_NORMAL = np.array([-CRESCENT_AXIS[1], CRESCENT_AXIS[0]])


def make_absorption_boundary(n=720):
    radius = 0.80
    gap_deg = 75
    width = 0.10
    gap = np.deg2rad(gap_deg)

    n_outer = n // 2
    n_inner = n - n_outer
    outer_theta = np.linspace(gap / 2.0, 2.0 * np.pi - gap / 2.0, n_outer)
    inner_theta = np.linspace(2.0 * np.pi - gap / 2.0, gap / 2.0, n_inner)
    outer_radius = radius + width / 2.0
    inner_radius = radius - width / 2.0

    outer_arc = np.column_stack(
        [outer_radius * np.cos(outer_theta), outer_radius * np.sin(outer_theta)]
    )
    inner_arc = np.column_stack(
        [
            inner_radius * np.cos(inner_theta),
            inner_radius * np.sin(inner_theta),
        ]
    )
    local = np.vstack([outer_arc, inner_arc])
    rotation = np.vstack([CRESCENT_AXIS, CRESCENT_NORMAL])
    return local @ rotation + CRESCENT_CENTER


def make_discrete_data(boundary):
    return sample_uniform_in_polygon(boundary, N_DATA_POINTS, seed=DATA_SEED)


def make_initial_condition(boundary):
    return T0 * (CRESCENT_CENTER + 1.60 * CRESCENT_AXIS)


def choose_flow_run(samples, hull_vertices, x0):
    fallback = None
    for config in RUN_CONFIGS:
        dt = config["dt"]
        ts, xs, inside, means = run_unconditional_flow_ode(
            samples,
            hull_vertices,
            x0,
            dt=dt,
            t_end=config["t_end"],
        )
        fallback = (dt, ts, xs, inside, means)
        if np.any(inside):
            first_inside = int(np.argmax(inside))
            if len(ts) - first_inside >= 4:
                return fallback
    return fallback


def add_colored_segments(ax, points, inside, first_inside, has_entry, use_entry_color=True):
    pre_color = "#cf3e3e"
    post_color = "#2f7f42"
    entry_color = "#d28b21"

    for i in range(len(points) - 1):
        if (not has_entry) or i + 1 < first_inside or (not use_entry_color and i < first_inside):
            color = pre_color
            linewidth = 2.7
            alpha = 0.88
        elif use_entry_color and i + 1 == first_inside:
            color = entry_color
            linewidth = 3.4
            alpha = 0.96
        else:
            color = post_color
            linewidth = 3.0
            alpha = 0.92

        segment = np.asarray([[points[i], points[i + 1]]])
        collection = LineCollection(
            segment,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            zorder=10,
        )
        collection.set_capstyle("round")
        ax.add_collection(collection)


def add_sparse_arrows(ax, points, first_inside, has_entry, use_entry_color=True):
    for i in range(0, len(points) - 1, 2):
        if (not has_entry) or i + 1 < first_inside:
            continue
        if use_entry_color and i + 1 == first_inside:
            color = "#d28b21"
        else:
            color = "#2f7f42"
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
                "lw": 1.45,
                "mutation_scale": 12,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=14,
        )


def absorption_snapshot_indices(ts, first_inside, max_snapshots=4):
    if first_inside is None:
        return []
    post = np.arange(first_inside, len(ts))
    count = min(max_snapshots, len(post))
    if count <= 0:
        return []
    positions = np.linspace(0, len(post) - 1, count, dtype=int)
    selected = []
    for pos in positions:
        idx = int(post[pos])
        if idx not in selected:
            selected.append(idx)
    return selected


def displayed_node_indices(ts, first_inside, snapshot_indices):
    if first_inside is None:
        return np.arange(len(ts), dtype=int), np.array([], dtype=int)

    outside_indices = np.arange(first_inside, dtype=int)
    snapshot_set = set(int(idx) for idx in snapshot_indices)
    absorbed_indices = np.asarray(
        [idx for idx in range(first_inside + 1, len(ts)) if idx in snapshot_set],
        dtype=int,
    )
    return outside_indices, absorbed_indices


def add_time_labels(ax, ts, xs, indices, color="#111111", skip_last=True):
    for idx in indices:
        if skip_last and idx == len(ts) - 1:
            continue
        ax.text(
            xs[idx, 0] + 0.055,
            xs[idx, 1] - 0.050,
            rf"$z_{{{idx}}}$",
            color=color,
            fontsize=FLOW_TIME_LABEL_FONT_SIZE,
            ha="left",
            va="top",
            zorder=22,
        )


def add_coordinate_axes(ax):
    axis_color = "#111111"
    ax.annotate(
        "",
        xy=(5.85, 0),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "lw": 2.2, "color": axis_color},
        zorder=0,
    )
    ax.annotate(
        "",
        xy=(0, 5.85),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "lw": 2.2, "color": axis_color},
        zorder=0,
    )
    ax.plot([0, 5.55], [0, 5.55], "--", color=axis_color, linewidth=1.45, alpha=0.48)
    ax.text(
        -0.03,
        -0.20,
        "O",
        color=axis_color,
        fontsize=FLOW_AXIS_FONT_SIZE,
        ha="center",
        va="top",
    )
    ax.text(
        5.78,
        0.13,
        r"$x_1$",
        color=axis_color,
        fontsize=FLOW_AXIS_FONT_SIZE,
        ha="right",
        va="bottom",
    )
    ax.text(
        0.13,
        5.78,
        r"$x_2$",
        color=axis_color,
        fontsize=FLOW_AXIS_FONT_SIZE,
        ha="left",
        va="top",
    )


def add_flow_line_legend(ax, snapshot_times, snapshot_colors):
    x0, y0 = 0.04, 0.910
    line_height = 0.047
    entries = [
        ("marker", "#2f7f42", "o", r"data points $D$"),
        ("marker", "#a21a31", "o", r"initial $t_0D$"),
        ("line", "#1c6d4a", ":", r"$\mathrm{Conv}(D)$"),
    ]
    for time, color in zip(snapshot_times, snapshot_colors):
        entries.append(("line", color, "--", rf"$t_i\mathrm{{Conv}}(D)$ at $t_i={time:.3f}$"))

    for i, (kind, color, style, text) in enumerate(entries):
        y = y0 - i * line_height
        if kind == "marker":
            ax.scatter(
                [x0 + 0.0215],
                [y],
                transform=ax.transAxes,
                s=34,
                marker=style,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.9,
                zorder=30,
            )
        else:
            ax.plot(
                [x0, x0 + 0.043],
                [y, y],
                transform=ax.transAxes,
                color=color,
                linestyle=style,
                linewidth=2.6,
                solid_capstyle="round",
                zorder=30,
            )
        ax.text(
            x0 + 0.056,
            y,
            text,
            transform=ax.transAxes,
            fontsize=FLOW_LEGEND_FONT_SIZE,
            color="#111111",
            ha="left",
            va="center",
            zorder=30,
        )


def add_flow_marker_legend(
    ax,
    use_entry_color=True,
    include_step_legend=True,
    include_outside=True,
):
    x0, y0 = 0.640, 0.320
    line_height = 0.053
    entries = []
    if include_step_legend:
        entries.append(("step", "#cf3e3e", "o", "before entry"))
    if include_step_legend and use_entry_color:
        entries.append(("step", "#d28b21", "o", "entry step"))
    entries.append(("step", "#2f7f42", "o", "after entry"))
    for i, (kind, color, marker, text) in enumerate(entries):
        y = y0 - i * line_height
        if kind == "step":
            ax.plot(
                [x0, x0 + 0.050],
                [y, y],
                transform=ax.transAxes,
                color=color,
                linewidth=2.6,
                marker=marker,
                markersize=3.8,
                markerfacecolor=color,
                markeredgewidth=1.25,
                markeredgecolor="white",
                solid_capstyle="round",
                zorder=30,
            )
        else:
            ax.scatter(
                [x0 + 0.025],
                [y],
                transform=ax.transAxes,
                s=78 if marker == "*" else 42,
                marker=marker,
                facecolor=color,
                edgecolor="#111111" if marker == "*" else "white",
                linewidth=1.25,
                zorder=30,
            )
        ax.text(
            x0 + 0.062,
            y,
            text,
            transform=ax.transAxes,
            fontsize=FLOW_LEGEND_FONT_SIZE,
            color="#111111",
            ha="left",
            va="center",
            zorder=30,
        )


def plot_flow_trajectory(
    ts,
    xs,
    inside,
    hull_vertices,
    data_points,
    dt,
    out_path,
    show_nodes=True,
    use_entry_color=True,
):
    has_entry = bool(np.any(inside))
    first_inside = int(np.argmax(inside)) if has_entry else None

    fig, ax = plt.subplots(figsize=(8.7, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    green = "#2f7f42"
    red = "#cf3e3e"
    entry_color = "#d28b21"
    black = "#111111"
    hull_color = "#496a9a"

    ax.scatter(
        data_points[:, 0],
        data_points[:, 1],
        s=9,
        facecolor=green,
        edgecolor="white",
        linewidth=0.16,
        alpha=0.76,
        zorder=5,
    )
    ax.scatter(
        T0 * data_points[:, 0],
        T0 * data_points[:, 1],
        s=7,
        facecolor=red,
        edgecolor="none",
        alpha=0.34,
        zorder=3,
    )
    ax.text(
        T0 * 4.15 + 0.34,
        T0 * 4.15 + 0.26,
        r"$t_0D$",
        color=black,
        fontsize=FLOW_LABEL_FONT_SIZE,
        ha="left",
        va="center",
    )
    plot_closed_polyline(
        ax,
        hull_vertices,
        color=hull_color,
        linewidth=1.5,
        linestyle=":",
        alpha=0.62,
        zorder=2,
    )
    ax.text(
        hull_vertices[:, 0].max() + 0.16,
        hull_vertices[:, 1].max() - 0.10,
        r"$D$",
        color=black,
        fontsize=FLOW_LABEL_FONT_SIZE + 1.5,
        ha="left",
        va="top",
    )

    snapshot_colors = (
        ["#d28b21", "#73a942", "#2f7f42", "#1c6e52"]
        if use_entry_color
        else ["#d28b21", "#2f7f42", "#1c6e52", "#496a9a"]
    )
    snapshot_indices = absorption_snapshot_indices(ts, first_inside)
    used_snapshot_colors = snapshot_colors[: len(snapshot_indices)]
    snapshot_times = []
    for count, idx in enumerate(snapshot_indices):
        color = snapshot_colors[count % len(snapshot_colors)]
        snapshot_times.append(ts[idx])
        scaled_hull = ts[idx] * hull_vertices
        add_filled_boundary(
            ax,
            scaled_hull,
            facecolor=color,
            edgecolor="none",
            alpha=0.045,
            linewidth=0.0,
            label=None,
        )
        plot_closed_polyline(
            ax,
            scaled_hull,
            color=color,
            linewidth=2.1 if idx == first_inside else 1.8,
            linestyle="--",
            alpha=0.88,
            zorder=4,
        )

    add_colored_segments(ax, xs, inside, first_inside, has_entry, use_entry_color)
    if show_nodes:
        add_sparse_arrows(ax, xs, first_inside, has_entry, use_entry_color)

    if show_nodes:
        outside_indices, absorbed_indices = displayed_node_indices(
            ts,
            first_inside,
            snapshot_indices,
        )
        ax.scatter(
            xs[outside_indices, 0],
            xs[outside_indices, 1],
            s=24,
            facecolor=red,
            edgecolor=red,
            linewidth=0.0,
            zorder=15,
        )
        if len(absorbed_indices) > 0:
            ax.scatter(
                xs[absorbed_indices, 0],
                xs[absorbed_indices, 1],
                s=28,
                facecolor=green,
                edgecolor="white",
                linewidth=1.0,
                zorder=15,
            )
            add_time_labels(ax, ts, xs, absorbed_indices)
    elif len(snapshot_indices) > 0:
        ax.scatter(
            xs[snapshot_indices, 0],
            xs[snapshot_indices, 1],
            s=28,
            facecolor=green,
            edgecolor=green,
            linewidth=0.0,
            zorder=15,
        )
        labeled_snapshot_indices = [
            idx for idx in snapshot_indices if (not has_entry) or idx != first_inside
        ]
        add_time_labels(ax, ts, xs, labeled_snapshot_indices)
    if has_entry and (show_nodes or not use_entry_color):
        ax.scatter(
            [xs[first_inside, 0]],
            [xs[first_inside, 1]],
            s=230,
            marker="*",
            facecolor=entry_color,
            edgecolor=black,
            linewidth=1.3,
            zorder=18,
        )
        add_time_labels(ax, ts, xs, [first_inside])

    ax.text(
        xs[0, 0] - 0.0,
        xs[0, 1] - 0.0,
        r"$z_0$",
        color=black,
        fontsize=FLOW_LABEL_FONT_SIZE,
        ha="right",
        va="top",
    )
    ax.text(
        xs[-1, 0] + 0.0,
        xs[-1, 1] - 0.055,
        rf"$z_{{{len(ts)-1}}}$",
        color=black,
        fontsize=FLOW_LABEL_FONT_SIZE - 0.5,
        ha="left",
        va="top",
    )
    add_flow_line_legend(ax, snapshot_times, used_snapshot_colors)
    if show_nodes:
        add_flow_marker_legend(ax, use_entry_color)
    elif not use_entry_color:
        add_flow_marker_legend(
            ax,
            use_entry_color=True,
            include_step_legend=False,
            include_outside=False,
        )

    ax.set_xlabel(r"$x_1$", fontsize=FLOW_TRAJECTORY_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(r"$x_2$", fontsize=FLOW_TRAJECTORY_AXIS_LABEL_FONT_SIZE)
    ax.grid(True, color="#dedede", linewidth=0.8, alpha=0.52)
    ax.tick_params(axis="both", labelsize=FLOW_TRAJECTORY_TICK_FONT_SIZE)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.30, 5.90)
    ax.set_ylim(-0.45, 5.90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_distance_figure(ts, xs, inside, hull_vertices, out_path, show_nodes=True, use_entry_color=True):
    has_entry = bool(np.any(inside))
    first_inside = int(np.argmax(inside)) if has_entry else None

    fig, dist_ax = plt.subplots(figsize=(7.8, 8.2))
    fig.patch.set_facecolor("white")
    dist_ax.set_facecolor("white")

    red = "#cf3e3e"
    green = "#2f7f42"
    entry_color = "#d28b21"
    black = "#111111"
    distances = distances_to_scaled_convex_hull(ts, xs, hull_vertices)
    distances[inside] = 0.0
    outside = ~inside
    for i in range(len(ts) - 1):
        if (not has_entry) or i + 1 < first_inside or (not use_entry_color and i < first_inside):
            color = red
            linewidth = 1.8
            alpha = 0.78
        elif use_entry_color and i + 1 == first_inside:
            color = entry_color
            linewidth = 2.2
            alpha = 0.92
        else:
            color = green
            linewidth = 2.0
            alpha = 0.84
        dist_ax.plot(
            ts[i : i + 2],
            distances[i : i + 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )
    if show_nodes:
        dist_ax.scatter(
            ts[outside],
            distances[outside],
            s=22,
            facecolor=red,
            edgecolor=red,
            linewidth=0.0,
            label="outside",
            zorder=4,
        )
        dist_ax.scatter(
            ts[inside],
            distances[inside],
            s=24,
            facecolor=green,
            edgecolor="white",
            linewidth=0.8,
            label="absorbed",
            zorder=5,
        )
    if has_entry:
        dist_ax.axvline(
            ts[first_inside],
            color=entry_color,
            linestyle="--",
            linewidth=1.7,
            alpha=0.88,
        )
        dist_ax.scatter(
            [ts[first_inside]],
            [distances[first_inside]],
            s=160,
            marker="*",
            facecolor=entry_color,
            edgecolor=black,
            linewidth=1.1,
            zorder=6,
            label="first entry",
        )
    dist_ax.axhline(0.0, color=black, linewidth=1.1, alpha=0.55)
    dist_ax.set_xlabel(r"$t_i$", fontsize=DIST_AXIS_LABEL_FONT_SIZE)
    dist_ax.set_ylabel(
        r"$\mathrm{dist}(z_i,t_i\mathrm{Conv}(D))$",
        fontsize=DIST_AXIS_LABEL_FONT_SIZE,
    )
    dist_ax.tick_params(axis="both", labelsize=DIST_TICK_FONT_SIZE)
    dist_ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.62)
    dist_ax.legend(frameon=False, fontsize=DIST_LEGEND_FONT_SIZE, loc="upper right")
    dist_ax.set_xlim(ts[0] - 0.015, ts[-1] + 0.015)
    top = max(distances.max(), 1e-8)
    dist_ax.set_ylim(bottom=-0.06 * top, top=1.12 * top)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    boundary = make_absorption_boundary()
    data_points = make_discrete_data(boundary)
    hull = ConvexHull(data_points)
    hull_vertices = data_points[hull.vertices]
    x0 = make_initial_condition(boundary)

    for config in RUN_CONFIGS:
        dt = config["dt"]
        suffix = config["suffix"]
        t_end = config["t_end"]
        out_path = output_path(
            config.get("out_name", f"{OUT_STEM}_{suffix}.png")
        )
        dist_out_path = output_path(
            config.get("dist_out_name", f"{DIST_OUT_STEM}_{suffix}.png")
        )
        ts, xs, inside, _ = run_unconditional_flow_ode(
            data_points,
            hull_vertices,
            x0,
            dt=dt,
            t_end=t_end,
        )

        plot_flow_trajectory(
            ts,
            xs,
            inside,
            hull_vertices,
            data_points,
            dt,
            out_path,
            show_nodes=config["show_nodes"],
            use_entry_color=config["use_entry_color"],
        )
        plot_distance_figure(
            ts,
            xs,
            inside,
            hull_vertices,
            dist_out_path,
            show_nodes=config["show_nodes"],
            use_entry_color=config["use_entry_color"],
        )
        print(f"saved {out_path}")
        print(f"saved {dist_out_path}")
        if np.any(inside):
            first_inside = int(np.argmax(inside))
            print(
                f"dt={dt:.3f}: first entered t_i Conv(D) "
                f"at index {first_inside}, t={ts[first_inside]:.6f}"
            )
            print(f"  post-entry inside flags: {inside[first_inside:].astype(int).tolist()}")
        else:
            terminal_distance = distance_to_filled_polygon(xs[-1], ts[-1] * hull_vertices)
            print(
                f"dt={dt:.3f}: no entry before t_end={t_end:.3f}; "
                f"terminal distance={terminal_distance:.6e}"
            )


if __name__ == "__main__":
    main()
