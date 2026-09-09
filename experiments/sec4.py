from pathlib import Path

from output_paths import output_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch
from scipy.spatial import ConvexHull


T0 = 0.3
T_END = 0.985
PROJECTION_DT = 5e-4
FLOW_DT = 1e-3
N_UNIFORM_SAMPLES = 5000
OUT_PATH = output_path("sec5_uncond_flow_convex_hull_d.png")
DIST_OUT_PATH = output_path("sec5_distance_to_convex_hull.png")


def make_bean_boundary(n=720):
    """A smooth closed non-convex bean-like set centered near the line y=x."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    wrapped = np.angle(np.exp(1j * theta))

    # Round bean with a local dimple on one side, so the set is non-convex
    # without looking spiky.
    dimple = 0.50 * np.exp(-(wrapped**2) / (2.0 * 0.58**2))
    r = 1.0 - dimple + 0.10 * np.sin(theta - 0.35)
    x = 0.95 * r * np.cos(theta)
    y = 1.05 * r * np.sin(theta)

    angle = np.deg2rad(18.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    points = np.column_stack([x, y]) @ rotation.T

    center = np.array([4.15, 4.15])
    return points + center


def closed_path(points):
    vertices = np.vstack([points, points[0]])
    codes = np.full(len(vertices), MplPath.LINETO)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    return MplPath(vertices, codes)


def add_filled_boundary(ax, points, facecolor, edgecolor, alpha, linewidth, label):
    patch = PathPatch(
        closed_path(points),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        joinstyle="round",
    )
    ax.add_patch(patch)
    return patch


def add_point(ax, xy, label=None, text_offset=(0.0, 0.0), color="#5b42b3", size=58):
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=size * 4.2,
        color="#b9a7dc",
        alpha=0.62,
        edgecolors="none",
        zorder=6,
    )
    ax.scatter([xy[0]], [xy[1]], s=size, color=color, zorder=7)
    if label is not None:
        ax.annotate(
            label,
            xy,
            xytext=(xy[0] + text_offset[0], xy[1] + text_offset[1]),
            fontsize=22,
            color=color,
            ha="center",
            va="center",
        )


def outward_normal(points, idx):
    tangent = points[(idx + 3) % len(points)] - points[idx - 3]
    tangent = tangent / np.linalg.norm(tangent)
    candidates = [np.array([tangent[1], -tangent[0]]), np.array([-tangent[1], tangent[0]])]
    center = points.mean(axis=0)
    return max(candidates, key=lambda normal: np.dot(normal, points[idx] - center))


def project_to_filled_polygon(point, polygon):
    if closed_path(polygon).contains_point(point, radius=1e-12):
        return point.copy()

    segments_start = polygon
    segments_end = np.roll(polygon, -1, axis=0)
    segment_vec = segments_end - segments_start
    rel = point - segments_start
    denom = np.sum(segment_vec * segment_vec, axis=1)
    alpha = np.clip(np.sum(rel * segment_vec, axis=1) / denom, 0.0, 1.0)
    candidates = segments_start + alpha[:, None] * segment_vec
    distances = np.linalg.norm(candidates - point, axis=1)
    return candidates[np.argmin(distances)]


def run_projection_ode(boundary, x0, dt=PROJECTION_DT, t0=T0, t_end=T_END):
    ts = [t0]
    xs = [x0.copy()]
    t = t0
    x = x0.copy()

    while t < t_end - 1e-12:
        step = min(dt, t_end - t)
        scaled_boundary = t * boundary
        projection = project_to_filled_polygon(x, scaled_boundary)
        y_p = projection / t
        x = x + step * (y_p - x) / (1.0 - t)
        t += step
        ts.append(t)
        xs.append(x.copy())

    return np.asarray(ts), np.asarray(xs)


def sample_uniform_in_polygon(polygon, n_samples, seed=7):
    rng = np.random.default_rng(seed)
    path = closed_path(polygon)
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    samples = []

    while sum(len(chunk) for chunk in samples) < n_samples:
        batch_size = max(5000, 2 * (n_samples - sum(len(chunk) for chunk in samples)))
        candidates = rng.uniform(lo, hi, size=(batch_size, 2))
        inside = path.contains_points(candidates)
        samples.append(candidates[inside])

    return np.vstack(samples)[:n_samples]


def posterior_mean_uniform(x, t, samples):
    residual = x[None, :] - t * samples
    log_weights = -np.sum(residual * residual, axis=1) / (2.0 * (1.0 - t) ** 2)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return np.sum(weights[:, None] * samples, axis=0) / np.sum(weights)


def point_in_convex_polygon(point, hull_vertices):
    return closed_path(hull_vertices).contains_point(point, radius=1e-10)


def distance_to_filled_polygon(point, polygon):
    projection = project_to_filled_polygon(point, polygon)
    return np.linalg.norm(point - projection)


def distances_to_scaled_convex_hull(ts, path, hull_vertices):
    return np.asarray(
        [
            distance_to_filled_polygon(x, t * hull_vertices)
            for t, x in zip(ts, path)
        ]
    )


def run_unconditional_flow_ode(samples, hull_vertices, x0, dt=FLOW_DT, t0=T0, t_end=T_END):
    ts = [t0]
    xs = [x0.copy()]
    inside_convex_hull = [point_in_convex_polygon(x0, t0 * hull_vertices)]
    posterior_means = []
    t = t0
    x = x0.copy()

    while t < t_end - 1e-12:
        step = min(dt, t_end - t)
        mean = posterior_mean_uniform(x, t, samples)
        posterior_means.append(mean)
        x = x + step * (mean - x) / (1.0 - t)
        t += step
        ts.append(t)
        xs.append(x.copy())
        inside_convex_hull.append(point_in_convex_polygon(x, t * hull_vertices))

    return np.asarray(ts), np.asarray(xs), np.asarray(inside_convex_hull), np.asarray(posterior_means)


def sparse_hull_times(ts, inside, max_snapshots=4):
    if not np.any(inside):
        return []

    first_inside = int(np.argmax(inside))
    inside_indices = np.flatnonzero(inside)
    candidate_positions = np.linspace(
        0,
        len(inside_indices) - 1,
        min(max_snapshots, len(inside_indices)),
        dtype=int,
    )
    selected = [int(inside_indices[pos]) for pos in candidate_positions]
    if selected[0] != first_inside:
        selected[0] = first_inside
    return selected


def plot_closed_polyline(ax, points, **kwargs):
    ax.plot(
        np.r_[points[:, 0], points[0, 0]],
        np.r_[points[:, 1], points[0, 1]],
        **kwargs,
    )


def add_info_panel(ax, snapshot_times, snapshot_colors):
    x0, y0 = 0.035, 0.935
    line_height = 0.043
    panel = FancyBboxPatch(
        (0.018, y0 - 0.305),
        0.40,
        0.32,
        boxstyle="round,pad=0.012,rounding_size=0.008",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=0.82,
        zorder=20,
    )
    ax.add_patch(panel)

    lines = [
        ("#2f7f42", "-", r"$D$ and $t_0D$, $t_0=0.3$"),
        ("#2f71b8", "-", "Projection ODE"),
        ("#cf3e3e", "-", "Unconditional flow ODE"),
    ]
    for time, color in zip(snapshot_times, snapshot_colors):
        lines.append((color, "--", rf"$t\mathrm{{Conv}}(D)$ at $t={time:.3f}$"))

    for i, (color, line_style, text) in enumerate(lines):
        y = y0 - i * line_height - 0.012
        ax.plot(
            [x0, x0 + 0.045],
            [y, y],
            transform=ax.transAxes,
            color=color,
            linestyle=line_style,
            linewidth=3.0 if line_style == "-" else 2.4,
            solid_capstyle="round",
            zorder=21,
        )
        ax.text(
            x0 + 0.058,
            y0 - i * line_height,
            text,
            transform=ax.transAxes,
            fontsize=18,
            color=color,
            ha="left",
            va="top",
            zorder=21,
        )


def add_xt_marker_panel(ax, snapshot_times, snapshot_colors):
    x0, y0 = 0.72, 0.34
    line_height = 0.046
    panel = FancyBboxPatch(
        (x0 - 0.025, y0 - len(snapshot_times) * line_height - 0.018),
        0.265,
        len(snapshot_times) * line_height + 0.042,
        boxstyle="round,pad=0.012,rounding_size=0.008",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=0.76,
        zorder=20,
    )
    ax.add_patch(panel)

    for i, (time, color) in enumerate(zip(snapshot_times, snapshot_colors)):
        y = y0 - i * line_height
        suffix = " (first entry)" if i == 0 else ""
        ax.scatter(
            [x0],
            [y],
            s=112 if i == 0 else 92,
            marker="*" if i == 0 else "o",
            facecolor=color,
            edgecolor="#111111" if i == 0 else "white",
            linewidth=1.5,
            transform=ax.transAxes,
            zorder=21,
        )
        ax.text(
            x0 + 0.024,
            y,
            rf"$x_t$",
            transform=ax.transAxes,
            fontsize=18,
            color="#222222",
            ha="left",
            va="center",
            zorder=21,
        )


def plot_distance_curves(
    projection_ts,
    projection_distances,
    flow_ts,
    flow_distances,
    projection_color,
    flow_color,
    first_entry_t,
):
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.set_facecolor("white")

    ax.plot(
        projection_ts,
        projection_distances,
        color=projection_color,
        linewidth=2.8,
        label=r"projection ODE",
    )
    ax.plot(
        flow_ts,
        flow_distances,
        color=flow_color,
        linewidth=2.8,
        label=r"flow ODE",
    )
    ax.axvline(
        first_entry_t,
        color="#111111",
        linestyle="--",
        linewidth=1.6,
        alpha=0.72,
    )
    ax.set_xlabel(r"$t$", fontsize=20)
    ax.set_ylabel("distance", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, color="#d6d6d6", linewidth=0.8, alpha=0.55)
    ax.legend(frameon=False, fontsize=18, loc="upper right")
    ax.set_xlim(T0, T_END)
    ax.set_ylim(bottom=-0.02 * max(projection_distances), top=1.08 * max(projection_distances))
    fig.tight_layout()
    fig.savefig(DIST_OUT_PATH, dpi=220)
    plt.close(fig)


def main():
    boundary = make_bean_boundary()
    scaled_boundary = T0 * boundary
    uniform_samples = sample_uniform_in_polygon(boundary, N_UNIFORM_SAMPLES)
    hull = ConvexHull(boundary)
    hull_vertices = boundary[hull.vertices]

    # Choose a smooth boundary point near the lower-right side of D. Its scaled
    # copy is the projection point on t0D.
    score = boundary[:, 0] - 1.45 * boundary[:, 1]
    idx = np.argmax(score)
    y_p_t0 = boundary[idx]
    q_t0 = T0 * y_p_t0
    normal = outward_normal(boundary, idx)
    x_t0 = T0 * (y_p_t0 + 2.35 * normal)
    projection_ts, projection_path = run_projection_ode(boundary, x_t0)
    flow_ts, flow_path, inside_hull, _ = run_unconditional_flow_ode(
        uniform_samples, hull_vertices, x_t0
    )
    hull_snapshot_indices = sparse_hull_times(flow_ts, inside_hull)
    projection_distances = np.asarray(
        [
            distance_to_filled_polygon(x, t * boundary)
            for t, x in zip(projection_ts, projection_path)
        ]
    )
    flow_distances = distances_to_scaled_convex_hull(flow_ts, flow_path, hull_vertices)
    flow_distances[inside_hull] = 0.0
    first_inside = int(np.argmax(inside_hull)) if np.any(inside_hull) else None

    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    axis_color = "#111111"
    green = "#2f7f42"
    base_point_color = "#111111"
    projection_color = "#2f71b8"
    flow_color = "#cf3e3e"
    hull_snapshot_colors = ["#d49a27", "#73a942", "#2f71b8", "#8e63b7"]
    flow_end_color = hull_snapshot_colors[-1]

    ax.annotate("", xy=(6.35, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.3, "color": axis_color})
    ax.annotate("", xy=(0, 6.35), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 2.3, "color": axis_color})
    ax.plot([0, 6], [0, 6], "--", color=axis_color, linewidth=1.7, alpha=0.65)
    ax.text(-0.03, -0.24, "O", color=base_point_color, fontsize=24, ha="center", va="top", zorder=30)
    ax.text(6.27, 0.16, "X", color=base_point_color, fontsize=24, ha="right", va="bottom", zorder=30)
    ax.text(0.16, 6.24, "Y", color=base_point_color, fontsize=24, ha="left", va="top", zorder=30)

    add_filled_boundary(
        ax,
        boundary,
        facecolor="#ffffff",
        edgecolor=green,
        alpha=0.98,
        linewidth=2.5,
        label=r"$D$",
    )
    add_filled_boundary(
        ax,
        scaled_boundary,
        facecolor="#ffffff",
        edgecolor=green,
        alpha=0.98,
        linewidth=2.5,
        label=rf"$t_0D,\ t_0={T0}$",
    )

    ax.fill(boundary[:, 0], boundary[:, 1], color="#2f7f42", alpha=0.045)
    ax.fill(scaled_boundary[:, 0], scaled_boundary[:, 1], color="#2f7f42", alpha=0.045)

    used_snapshot_colors = []
    snapshot_times = []
    for count, idx_snapshot in enumerate(hull_snapshot_indices):
        t = flow_ts[idx_snapshot]
        scaled_hull = t * hull_vertices
        color = hull_snapshot_colors[count % len(hull_snapshot_colors)]
        used_snapshot_colors.append(color)
        snapshot_times.append(t)
        add_filled_boundary(
            ax,
            scaled_hull,
            facecolor=color,
            edgecolor="none",
            alpha=0.035,
            linewidth=0.0,
            label=None,
        )
        plot_closed_polyline(
            ax,
            scaled_hull,
            linestyle="--",
            color=color,
            linewidth=2.0,
            alpha=0.78,
            label=None,
            zorder=4,
        )
        if count == 0:
            ax.scatter(
                [flow_path[idx_snapshot, 0]],
                [flow_path[idx_snapshot, 1]],
                s=260,
                marker="*",
                facecolor=color,
                edgecolor="#111111",
                linewidth=1.4,
                zorder=12,
            )
        else:
            ax.scatter(
                [flow_path[idx_snapshot, 0]],
                [flow_path[idx_snapshot, 1]],
                s=130,
                facecolor=color,
                edgecolor="white",
                linewidth=1.7,
                zorder=10,
            )

    ax.plot(
        projection_path[:, 0],
        projection_path[:, 1],
        color=projection_color,
        linewidth=2.5,
        alpha=0.75,
        solid_capstyle="round",
        label=None,
        zorder=5,
    )
    ax.plot(
        flow_path[:, 0],
        flow_path[:, 1],
        color=flow_color,
        linewidth=3.0,
        solid_capstyle="round",
        label=None,
        zorder=6,
    )
    ax.scatter(flow_path[::80, 0], flow_path[::80, 1], s=14, color=flow_color, zorder=7)
    ax.scatter(
        [projection_path[-1, 0]],
        [projection_path[-1, 1]],
        s=118,
        color=projection_color,
        edgecolor="white",
        linewidth=1.5,
        zorder=11,
    )
    ax.scatter(
        [flow_path[-1, 0]],
        [flow_path[-1, 1]],
        s=118,
        color=flow_end_color,
        edgecolor="white",
        linewidth=1.5,
        zorder=11,
    )

    add_point(ax, np.array([0.0, 0.0]), color=base_point_color, size=58)
    add_point(ax, x_t0, color=base_point_color, size=50)
    ax.text(
        x_t0[0] - 0.1,
        x_t0[1] - 0.1,
        r"$x_{\mathrm{start}}$",
        color=base_point_color,
        fontsize=22,
        ha="right",
        va="top",
    )
    ax.text(
        T0 * 4.15 + 0.23,
        T0 * 4.15 + 0.03,
        r"scaled $D$",
        color=green,
        fontsize=22,
        ha="left",
        va="bottom",
    )
    ax.text(
        4.15 + 0.18,
        4.15 - 0.22,
        r"$D$",
        color=green,
        fontsize=24,
        ha="left",
        va="top",
    )
    ax.text(
        projection_path[-1, 0] + 0.1,
        projection_path[-1, 1] + 0.1,
        "projection end",
        color=projection_color,
        fontsize=18,
        ha="left",
        va="top",
    )
    ax.text(
        flow_path[-1, 0] + 1.05,
        flow_path[-1, 1] + 0.05,
        "flow end",
        color=flow_end_color,
        fontsize=18,
        ha="right",
        va="bottom",
    )
    ax.set_xlim(-0.35, 6.35)
    ax.set_ylim(-0.55, 6.35)
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_PATH, dpi=220)
    plt.close(fig)
    if first_inside is not None:
        plot_distance_curves(
            projection_ts,
            projection_distances,
            flow_ts,
            flow_distances,
            projection_color,
            flow_color,
            flow_ts[first_inside],
        )
    print(f"saved {OUT_PATH}")
    print(f"saved {DIST_OUT_PATH}")
    if first_inside is not None:
        print(f"first entered t*Conv(D) at t={flow_ts[first_inside]:.6f}")
    else:
        print("flow trajectory did not enter t*Conv(D) before T_END")


if __name__ == "__main__":
    main()
