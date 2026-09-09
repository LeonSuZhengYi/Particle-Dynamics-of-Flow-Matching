# Particle Dynamics of Flow Matching and Classifier-Free Guidance from a Stagewise Geometry Perspective

Two-dimensional synthetic experiments for the paper by **Jian-Feng Cai, Zhengyi Su, and Chao Wang**.

[Paper (arXiv:2609.06947)](https://arxiv.org/abs/2609.06947)

This repository contains the synthetic experiments corresponding to **Figures 5–9 of arXiv v1**, plus two supplementary geometry examples. Data are generated locally from fixed random seeds; no dataset downloads, pretrained models, or GPU are required. The ImageNet, MNIST, and CIFAR experiments and the conceptual diagrams in Figures 1–4 are outside this release.

## Preview

| Convex-hull absorption | Local-cluster absorption | CFG trajectories |
| --- | --- | --- |
| ![Convex hull](figures/convexhull_z.png) | ![Local clusters](figures/cluster_z.png) | ![CFG](figures/sec6_cfg_z.png) |

## Quick start

Use Python 3.9 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

The default command runs all four paper experiment scripts. Figures, diagnostic logs, and a run manifest recording package versions and runtimes are written to `outputs/`. Existing output files with the same names are overwritten. Generated files are excluded from Git.

Run one experiment or change the output directory:

```bash
python reproduce.py --experiment local-clusters
python reproduce.py --experiment cfg --output-dir results/cfg
python reproduce.py --experiment all
```

`all` also runs the supplementary projection and bean-shaped flow examples. Each script can be run directly, for example `python experiments/sec5_final_local_cluster.py`.

## Paper figures

| Figure (v1) | Experiment | Output files |
| --- | --- | --- |
| 5: convex-hull attraction and absorption | `convex-hull` | `convexhull_z.png`, `dist_conv_z.png` |
| 6: final local-cluster attraction and absorption | `local-clusters` | `cluster_z.png`, `dist_cluster_z.png` |
| 7: CFG trajectories and final distance | `cfg`, `cfg-final` | `sec6_cfg_z.png`, `sec6_cfg_final_cluster_distance_r005.png` |
| 8: early CFG attraction, w = 1.5 and 3.0 | `cfg` | `sec6_cfg_early_distance_w15.png`, `sec6_cfg_early_distance_w30.png` |
| 9: fitted prediction-gap decay | `cfg` | `sec6_cfg_prediction_gap_fit_w15_z.png`, `sec6_cfg_prediction_gap_fit_w30_z.png` |

The convex-hull script also generates comparisons at step sizes 0.1 and 0.025, and a fine-step approximation at 0.002. The panel images are saved separately rather than assembled into the paper's two-column layout.

## Experiment settings and code

The numerical formulas, seeds, initial conditions, and plotting settings were preserved from the original research scripts. Output paths have been centralized so generated figures are separate from source code. Script filenames retain their historical section numbers; use the table above for the current paper figure mapping.

- `experiments/sec4.py`: shared polygon geometry, sampling, posterior means, and Euler integration; also a supplementary bean-shaped example.
- `experiments/sec4_projection_ode_discrete.py`: supplementary projection-flow example.
- `experiments/sec5_flow_ode_discrete_absorption.py`: 500 synthetic points, seed 7; Euler step sizes 0.1, 0.05, 0.025, and 0.002. Figure 5 uses 0.05.
- `experiments/sec5_final_local_cluster.py`: five nonconvex clusters, 500 samples per cluster, seed 20260514; step size 0.05, neighborhood radius 0.025.
- `experiments/sec6_cfg_early_attraction.py`: guidance scales 1, 1.5, 2, 2.5, 3; step size 0.001; target is cluster 5. The gap fit uses t in [0.35, 0.85].
- `experiments/sec6_cfg_final_cluster_distance.py`: guidance scales 1, 1.5, 3; neighborhood radius 0.05.

Adjust the named constants near the top of the relevant script to explore different settings. CFG uses `mean_uncond + w * (mean_cond - mean_uncond)`, so w = 1 is the conditional flow. The continuous-flow plots are numerical Euler approximations, not exact ODE solutions. Prediction-gap curves are empirical fits, not independent estimates of the theorem's constants.

## Validation

All six experiment entry points were run successfully on macOS with Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, and Matplotlib 3.9.4, producing 20 PNG files. All ten panels listed for Figures 5–9 matched the original locally saved figure images pixel-for-pixel in this environment. Rendering may vary across package versions and operating systems. Use `requirements-tested.txt` to install the exact dependency versions used for this check.

## Citation

```bibtex
@misc{cai2026particle,
  title={Particle Dynamics of Flow Matching and Classifier-Free Guidance from a Stagewise Geometry Perspective},
  author={Jian-Feng Cai and Zhengyi Su and Chao Wang},
  year={2026},
  eprint={2609.06947},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2609.06947}
}
```
