"""Shared geometry and flow utilities for the 2D synthetic experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


T0 = 0.3
T_END = 0.985
FLOW_DT = 1e-3


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


def outward_normal(points, idx):
    tangent = points[(idx + 3) % len(points)] - points[idx - 3]
    tangent = tangent / np.linalg.norm(tangent)
    candidates = [
        np.array([tangent[1], -tangent[0]]),
        np.array([-tangent[1], tangent[0]]),
    ]
    center = points.mean(axis=0)
    return max(candidates, key=lambda normal: np.dot(normal, points[idx] - center))


def project_to_filled_polygon(point, polygon):
    """Return the Euclidean projection onto a filled polygon."""
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


def sample_uniform_in_polygon(polygon, n_samples, seed=7):
    rng = np.random.default_rng(seed)
    path = closed_path(polygon)
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    samples = []

    while sum(len(chunk) for chunk in samples) < n_samples:
        remaining = n_samples - sum(len(chunk) for chunk in samples)
        batch_size = max(5000, 2 * remaining)
        candidates = rng.uniform(lo, hi, size=(batch_size, 2))
        samples.append(candidates[path.contains_points(candidates)])

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
        [distance_to_filled_polygon(x, t * hull_vertices) for t, x in zip(ts, path)]
    )


def run_unconditional_flow_ode(
    samples,
    hull_vertices,
    x0,
    dt=FLOW_DT,
    t0=T0,
    t_end=T_END,
):
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

    return (
        np.asarray(ts),
        np.asarray(xs),
        np.asarray(inside_convex_hull),
        np.asarray(posterior_means),
    )


def plot_closed_polyline(ax, points, **kwargs):
    ax.plot(
        np.r_[points[:, 0], points[0, 0]],
        np.r_[points[:, 1], points[0, 1]],
        **kwargs,
    )
