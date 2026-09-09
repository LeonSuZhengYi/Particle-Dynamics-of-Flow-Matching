from pathlib import Path

from output_paths import output_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sec4 import distance_to_filled_polygon
from sec5_final_local_cluster import make_cluster_boundaries, sample_clusters
from sec6_cfg_early_attraction import (
    COLORS,
    TARGET_CLUSTER,
    run_cfg_flow,
)


OUT_PATH = output_path("sec6_cfg_final_cluster_distance_r005.png")

RADIUS = 0.05
PLOT_T_START = 0.40
GUIDANCE_SCALES = [1.0, 1.5, 3.0]
TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 21
ANNOTATION_FONT_SIZE = 18


def distance_to_scaled_cluster_neighborhood(t, x, boundary, radius):
    if t <= 1e-12:
        return float(np.linalg.norm(x))
    raw_distance = distance_to_filled_polygon(x, t * boundary)
    return max(raw_distance - t * radius, 0.0)


def distances_along_path(ts, xs, boundary, radius):
    return np.asarray(
        [
            distance_to_scaled_cluster_neighborhood(t, x, boundary, radius)
            for t, x in zip(ts, xs)
        ]
    )


def first_entry_index(distances):
    inside = np.where(distances <= 1e-10)[0]
    return int(inside[0]) if len(inside) else None


def plot_distance_curves(runs):
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for w, run in runs.items():
        color = COLORS[w]
        ts = run["ts"]
        distances = run["distances"]
        mask = ts >= PLOT_T_START
        ax.plot(
            ts[mask],
            distances[mask],
            color=color,
            linewidth=2.65,
            alpha=0.96,
            label=rf"$w={w:.1f}$",
        )
        entry_idx = run["entry_idx"]
        if entry_idx is not None and ts[entry_idx] >= PLOT_T_START:
            ax.scatter(
                ts[entry_idx],
                distances[entry_idx],
                marker="*",
                s=265,
                color=color,
                edgecolor="#111111",
                linewidth=0.85,
                zorder=5,
            )

    ax.axhline(0.0, color="#111111", linewidth=1.15, alpha=0.42)
    ax.set_xlabel(r"$t$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(r"$\mathrm{distance}$", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlim(PLOT_T_START - 0.01, 1.005)
    top = max(
        float(np.max(run["distances"][run["ts"] >= PLOT_T_START]))
        for run in runs.values()
    )
    ax.set_ylim(-0.045 * top, 1.12 * top)
    ax.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.62)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=240)
    plt.close(fig)


def main():
    boundaries = make_cluster_boundaries()
    samples, labels = sample_clusters(boundaries)
    conditional_samples = samples[labels == TARGET_CLUSTER]
    target_boundary = boundaries[TARGET_CLUSTER]

    runs = {}
    for w in GUIDANCE_SCALES:
        ts, xs, _ = run_cfg_flow(samples, conditional_samples, w)
        distances = distances_along_path(ts, xs, target_boundary, RADIUS)
        runs[w] = {
            "ts": ts,
            "xs": xs,
            "distances": distances,
            "entry_idx": first_entry_index(distances),
        }

    plot_distance_curves(runs)
    print(f"saved {OUT_PATH}")
    for w, run in runs.items():
        entry_idx = run["entry_idx"]
        if entry_idx is None:
            print(
                f"w={w:.1f}: no entry; terminal neighborhood distance={run['distances'][-1]:.6e}"
            )
        else:
            print(
                f"w={w:.1f}: first entry at t={run['ts'][entry_idx]:.6f}; "
                f"terminal neighborhood distance={run['distances'][-1]:.6e}"
            )


if __name__ == "__main__":
    main()
