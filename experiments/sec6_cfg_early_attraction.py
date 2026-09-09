from pathlib import Path

from output_paths import output_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from sec5_final_local_cluster import (
    SEED,
    make_cluster_boundaries,
    sample_clusters,
)


OUT_PATH = output_path("sec6_cfg_z.png")
GAP_OUT_TEMPLATE = "sec6_cfg_prediction_gap_fit_w{tag}_z.png"
DIST_OUT_TEMPLATE = "sec6_cfg_early_distance_w{tag}.png"

TARGET_CLUSTER = 4  # Omega_5 in zero-based indexing.
X0 = np.array([-1.0, 0.0])
T0 = 0.0
T_END = 0.998
DT = 0.001
GUIDANCE_SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]
CURVE_SCALES = [1.5, 3.0]
EARLY_RADIUS = 0.05
EARLY_T_END = 0.30
GAP_FIT_T_MIN = 0.35
GAP_FIT_T_MAX = 0.85
COLORS = {
    1.0: "#3b0f4f",
    1.5: "#b23a8f",
    2.0: "#0f9a9a",
    2.5: "#168a45",
    3.0: "#f28e1c",
}
CLUSTER_PALETTE = ["#287c62", "#b56c1f", "#6f5bb8", "#c93d45", "#2f6fa3"]
HIGHLIGHT_TIMES = [0.1, 0.2, 0.3]
MAIN_TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16.5
EARLY_DISTANCE_LEGEND_FONT_SIZE = 20
GAP_AXIS_LABEL_FONT_SIZE = 23
GAP_TICK_FONT_SIZE = 20
GAP_LEGEND_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 18
SMALL_LABEL_FONT_SIZE = 18
CLUSTER_LABEL_FONT_SIZE = 19
TARGET_CLUSTER_LABEL_FONT_SIZE = 21


def stable_posterior_mean(x, t, samples):
    residual = x[None, :] - t * samples
    log_weights = -np.sum(residual * residual, axis=1) / (2.0 * (1.0 - t) ** 2)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return np.sum(weights[:, None] * samples, axis=0) / np.sum(weights)


def posterior_means(x, t, samples, conditional_samples):
    mean_uncond = stable_posterior_mean(x, t, samples)
    mean_cond = stable_posterior_mean(x, t, conditional_samples)
    return mean_uncond, mean_cond


def cfg_velocity_and_gap(x, t, samples, conditional_samples, w):
    mean_uncond, mean_cond = posterior_means(x, t, samples, conditional_samples)
    mean_cfg = mean_uncond + w * (mean_cond - mean_uncond)
    velocity = (mean_cfg - x) / (1.0 - t)
    prediction_gap = np.linalg.norm((mean_cond - mean_uncond) / (1.0 - t))
    return velocity, prediction_gap


def run_cfg_flow(samples, conditional_samples, w):
    ts = [T0]
    xs = [X0.copy()]
    gaps = []

    t = T0
    x = X0.copy()
    while t < T_END - 1e-12:
        velocity, gap = cfg_velocity_and_gap(x, t, samples, conditional_samples, w)
        gaps.append(gap)
        step = min(DT, T_END - t)
        x = x + step * velocity
        t += step
        ts.append(t)
        xs.append(x.copy())

    _, final_gap = cfg_velocity_and_gap(x, min(t, T_END), samples, conditional_samples, w)
    gaps.append(final_gap)
    return np.asarray(ts), np.asarray(xs), np.asarray(gaps)


def plot_trajectories(boundaries, samples, labels, runs, mean_uncond, mean_cond):
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")

    for cluster_id, boundary in enumerate(boundaries):
        is_target = cluster_id == TARGET_CLUSTER
        color = CLUSTER_PALETTE[cluster_id]
        alpha = 0.18 if not is_target else 0.34
        ax.scatter(
            samples[labels == cluster_id, 0],
            samples[labels == cluster_id, 1],
            s=5.0 if is_target else 4.0,
            color=color,
            alpha=alpha,
            linewidths=0,
            zorder=2 if not is_target else 3,
        )
        ax.plot(
            np.r_[boundary[:, 0], boundary[0, 0]],
            np.r_[boundary[:, 1], boundary[0, 1]],
            color=color,
            linewidth=1.1 if is_target else 0.9,
            alpha=0.28 if is_target else 0.20,
            zorder=1,
        )
        center = boundary.mean(axis=0)
        label_pos = center.copy()
        if is_target:
            label_pos += np.array([0.32, 0.23])
        ax.text(
            label_pos[0],
            label_pos[1],
            rf"$\Omega_{cluster_id + 1}$",
            color=color,
            fontsize=TARGET_CLUSTER_LABEL_FONT_SIZE if is_target else CLUSTER_LABEL_FONT_SIZE,
            ha="center",
            va="center",
            alpha=0.98 if is_target else 0.86,
            zorder=4,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white", alpha=0.86)],
        )

    t_line = np.linspace(0.0, 1.0, 200)
    label_offsets = {
        1.0: np.array([0.10, -0.22]),
        1.5: np.array([0.10, 0.02]),
        2.0: np.array([0.10, 0.13]),
        2.5: np.array([0.10, 0.21]),
        3.0: np.array([0.10, 0.29]),
    }
    farthest_attractor = mean_uncond + max(GUIDANCE_SCALES) * (mean_cond - mean_uncond)
    ax.plot(
        [mean_uncond[0], farthest_attractor[0]],
        [mean_uncond[1], farthest_attractor[1]],
        color="#174cff",
        linestyle=(0, (4.2, 3.0)),
        linewidth=1.5,
        alpha=0.36,
        zorder=4,
    )
    for w, run in runs.items():
        color = COLORS[w]
        ts = run["ts"]
        xs = run["xs"]
        attractor = mean_uncond + w * (mean_cond - mean_uncond)
        attractor_path = t_line[:, None] * attractor[None, :]
        ax.plot(
            attractor_path[:, 0],
            attractor_path[:, 1],
            color=color,
            linestyle=(0, (5.0, 2.4)),
            linewidth=2.0,
            alpha=0.32,
            zorder=5,
        )
        ax.scatter(
            [attractor[0]],
            [attractor[1]],
            s=55,
            marker="D",
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=13,
        )
        ax.plot(
            xs[:, 0],
            xs[:, 1],
            color=color,
            linewidth=2.7,
            alpha=0.96,
            label=rf"$w={w:.1f}$",
            zorder=10,
        )
        ax.scatter(
            xs[::140, 0],
            xs[::140, 1],
            s=14,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.95,
            zorder=11,
        )
        if w in CURVE_SCALES:
            for time_id, mark_t in enumerate(HIGHLIGHT_TIMES):
                idx = int(np.argmin(np.abs(ts - mark_t)))
                actual_point = xs[idx]
                mean_point = mark_t * attractor
                ax.scatter(
                    [actual_point[0]],
                    [actual_point[1]],
                    s=34,
                    marker="o",
                    facecolor=color,
                    edgecolor="#111111",
                    linewidth=0.65,
                    alpha=0.68,
                    zorder=18,
                )
                ax.scatter(
                    [mean_point[0]],
                    [mean_point[1]],
                    s=42,
                    marker="s",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.35,
                    alpha=0.68,
                    zorder=18,
                )

    ax.scatter(
        [X0[0]],
        [X0[1]],
        s=110,
        marker="o",
        facecolor="#111111",
        edgecolor="white",
        linewidth=1.2,
        label=r"$z_0=(-1,0)$",
        zorder=15,
    )
    ax.scatter(
        [mean_cond[0]],
        [mean_cond[1]],
        s=250,
        marker="*",
        facecolor="#ff2d20",
        edgecolor="white",
        linewidth=1.2,
        label=r"$E_{\mathrm{cond}}$",
        zorder=16,
    )
    ax.scatter(
        [mean_uncond[0]],
        [mean_uncond[1]],
        s=145,
        marker="x",
        color="#174cff",
        linewidth=2.6,
        label=r"$E_{\mathrm{uncond}}$",
        zorder=16,
    )

    ax.set_xlabel(r"$x_1$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(r"$x_2$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.grid(True, color="#dedede", linewidth=0.8, alpha=0.52)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.set_xlim(-2.00, 7.85)
    ax.set_ylim(-0.75, 8.05)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        legend_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=240)
    plt.close(fig)


def output_tag(w):
    return f"{int(round(10 * w)):02d}"


def fit_gap_bound(ts, gaps):
    mask = (
        (ts >= GAP_FIT_T_MIN)
        & (ts <= GAP_FIT_T_MAX)
        & (gaps > 1e-10)
        & (ts < 1.0)
    )
    fit_ts = ts[mask]
    fit_gaps = gaps[mask]
    if len(fit_ts) < 3:
        raise RuntimeError("Not enough positive tail points to fit the prediction-gap bound.")
    design_x = 1.0 / (1.0 - fit_ts) ** 2
    response_y = np.log(fit_gaps * (1.0 - fit_ts))
    slope, intercept = np.polyfit(design_x, response_y, 1)
    c1 = float(np.exp(intercept))
    c2 = float(max(-slope, 0.0))
    fitted_y = slope * design_x + intercept
    ss_res = float(np.sum((response_y - fitted_y) ** 2))
    ss_tot = float(np.sum((response_y - np.mean(response_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return c1, c2, r2, fit_ts


def gap_bound_curve(ts, c1, c2):
    safe = np.maximum(1.0 - ts, 1e-12)
    exponent = -c2 / (safe * safe)
    return c1 / safe * np.exp(np.clip(exponent, -745.0, 50.0))


def plot_prediction_gap_fit(w, run):
    fig, ax = plt.subplots(figsize=(8.4, 6.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ts = run["ts"]
    gaps = run["gaps"]
    c1, c2, r2, fit_ts = fit_gap_bound(ts, gaps)
    fit_grid = np.linspace(fit_ts[0], min(T_END, 0.98), 500)
    fit_curve = gap_bound_curve(fit_grid, c1, c2)
    color = COLORS[w]

    ax.plot(
        ts,
        gaps,
        color=color,
        linewidth=2.7,
        alpha=0.98,
        label=rf"measured gap, $w={w:.1f}$",
    )
    ax.plot(
        fit_grid,
        fit_curve,
        color="#111111",
        linestyle=(0, (5.0, 2.6)),
        linewidth=2.1,
        alpha=0.82,
        label="fitted asymptotic form",
    )
    ax.axvspan(fit_ts[0], fit_ts[-1], color=color, alpha=0.075, linewidth=0)
    ax.set_xlabel(r"$t$", fontsize=GAP_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(
        r"$\|v_c(t_i,z_i)-v_{\emptyset}(t_i,z_i)\|$",
        fontsize=GAP_AXIS_LABEL_FONT_SIZE,
    )
    ax.set_xlim(-0.01, 1.005)
    top = float(np.max(gaps))
    ax.set_ylim(-0.03 * top, 1.08 * top)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.60)
    ax.tick_params(axis="both", labelsize=GAP_TICK_FONT_SIZE)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=GAP_LEGEND_FONT_SIZE)
    fig.tight_layout()
    out_path = output_path(GAP_OUT_TEMPLATE.format(tag=output_tag(w)))
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path, c1, c2, r2


def early_ball_distances(ts, xs, extra_mean, radius):
    center_path = ts[:, None] * extra_mean[None, :]
    return np.maximum(np.linalg.norm(xs - center_path, axis=1) - ts * radius, 0.0)


def plot_early_distance(w, run, mean_uncond, mean_cond):
    ts = run["ts"]
    xs = run["xs"]
    extra_mean = mean_uncond + w * (mean_cond - mean_uncond)
    distances = early_ball_distances(ts, xs, extra_mean, EARLY_RADIUS)
    early_mask = ts <= EARLY_T_END
    color = COLORS[w]

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        ts[early_mask],
        distances[early_mask],
        color=color,
        linewidth=2.8,
        label="measured distance",
    )
    ax.plot(
        ts[early_mask],
        np.linalg.norm(X0) * (1.0 - ts[early_mask]),
        color="#111111",
        linestyle=(0, (5.0, 2.6)),
        linewidth=2.0,
        alpha=0.75,
        label="linear reference",
    )
    ax.scatter(
        ts[early_mask][::35],
        distances[early_mask][::35],
        s=24,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    ax.set_xlabel(r"$t$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("distance", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlim(-0.005, EARLY_T_END + 0.005)
    top = float(np.max(distances[early_mask]))
    ax.set_ylim(0.0, 1.06 * top)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.62)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=EARLY_DISTANCE_LEGEND_FONT_SIZE)
    fig.tight_layout()
    out_path = output_path(DIST_OUT_TEMPLATE.format(tag=output_tag(w)))
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def main():
    np.random.seed(SEED)
    boundaries = make_cluster_boundaries()
    samples, labels = sample_clusters(boundaries)
    conditional_samples = samples[labels == TARGET_CLUSTER]
    mean_uncond = samples.mean(axis=0)
    mean_cond = conditional_samples.mean(axis=0)

    runs = {}
    for w in GUIDANCE_SCALES:
        ts, xs, gaps = run_cfg_flow(samples, conditional_samples, w)
        runs[w] = {"ts": ts, "xs": xs, "gaps": gaps}

    plot_trajectories(boundaries, samples, labels, runs, mean_uncond, mean_cond)
    gap_outputs = {}
    distance_outputs = {}
    for w in CURVE_SCALES:
        gap_outputs[w] = plot_prediction_gap_fit(w, runs[w])
        distance_outputs[w] = plot_early_distance(w, runs[w], mean_uncond, mean_cond)

    print(f"saved {OUT_PATH}")
    for w, (path, c1, c2, r2) in gap_outputs.items():
        print(f"saved {path} (w={w:.1f}, C1={c1:.6f}, C2={c2:.6f}, R2={r2:.6f})")
    for w, path in distance_outputs.items():
        print(f"saved {path} (w={w:.1f}, r={EARLY_RADIUS:.3f}, early_t_end={EARLY_T_END:.3f})")
    print(f"E_uncond={np.round(mean_uncond, 6).tolist()}")
    print(f"E_cond(Omega_5)={np.round(mean_cond, 6).tolist()}")
    for w, run in runs.items():
        attractor = mean_uncond + w * (mean_cond - mean_uncond)
        print(
            f"w={w:.1f}: extrapolated mean={np.round(attractor, 6).tolist()}, "
            f"terminal x={np.round(run['xs'][-1], 6).tolist()}, "
            f"terminal gap={run['gaps'][-1]:.6e}"
        )


if __name__ == "__main__":
    main()
