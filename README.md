# Supplementary 2D Experiments

This repository provides the implementation and reproducibility details for the two-dimensional synthetic experiments in

> **Particle Dynamics of Flow Matching and Classifier-Free Guidance from a Stagewise Geometry Perspective**
>
> Jian-Feng Cai, Zhengyi Su, and Chao Wang
>
> [arXiv:2609.06947](https://arxiv.org/abs/2609.06947)

The experiments illustrate the stagewise geometric behavior proved in the paper: attraction to the scaled data convex hull, final-stage attraction to local cluster neighborhoods, and the effect of classifier-free guidance (CFG) on early and late trajectory geometry. They reproduce Figures 5–9 in arXiv v1.

## Scope

All examples use synthetic data in two dimensions. The data are generated at runtime from fixed random seeds, so no datasets, checkpoints, or GPUs are required. This repository does not include the conceptual illustrations in Figures 1–4 or the image experiments in Figures 10–13.

| Paper figure | Geometric phenomenon | Command |
| --- | --- | --- |
| Figure 5 | Convex-hull attraction and absorption | `python reproduce.py --experiment convex-hull` |
| Figure 6 | Final local-cluster attraction and absorption | `python reproduce.py --experiment local-clusters` |
| Figure 7 | CFG trajectory geometry and final cluster distance | `python reproduce.py --experiment cfg` and `--experiment cfg-final` |
| Figure 8 | Early attraction to the extrapolated CFG mean | `python reproduce.py --experiment cfg` |
| Figure 9 | Final-stage prediction-gap decay | `python reproduce.py --experiment cfg` |

| Convex-hull absorption | Local-cluster absorption | CFG trajectories |
| --- | --- | --- |
| ![Convex-hull experiment](figures/convexhull_z.png) | ![Local-cluster experiment](figures/cluster_z.png) | ![CFG experiment](figures/sec6_cfg_z.png) |

## Experiment details

- **Figure 5:** a crescent-shaped dataset illustrates attraction to and absorption by the moving convex hull. Several Euler step sizes are included for comparison.
- **Figure 6:** five separated nonconvex clusters illustrate final-stage local-cluster attraction and absorption.
- **Figures 7–9:** the fifth cluster is used as the conditional target to illustrate CFG trajectories, early-stage attraction, final-cluster distance, and prediction-gap decay at several guidance scales.

The synthetic datasets, random seeds, numerical settings, and plotting parameters are defined near the beginning of each experiment script.

## Reproduction

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/LeonSuZhengYi/Particle-Dynamics-of-Flow-Matching.git
cd Particle-Dynamics-of-Flow-Matching
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

The final command runs all experiments. To run one part or choose another output directory:

```bash
python reproduce.py --experiment local-clusters
python reproduce.py --experiment cfg --output-dir results/cfg
```

Generated figures, console logs, and `run-manifest.json` are written to `outputs/` by default. The manifest records package versions, exit codes, and runtimes. Generated files are excluded from version control. Each experiment script can also be executed directly.

The standard dependencies are listed in `requirements.txt`. `requirements-tested.txt` records the exact environment used for the reference validation.

## Output-to-figure map

| Paper figure | Output file |
| --- | --- |
| Figure 5, left | `convexhull_z.png` |
| Figure 5, right | `dist_conv_z.png` |
| Figure 6, left | `cluster_z.png` |
| Figure 6, right | `dist_cluster_z.png` |
| Figure 7, left | `sec6_cfg_z.png` |
| Figure 7, right | `sec6_cfg_final_cluster_distance_r005.png` |
| Figure 8, left/right | `sec6_cfg_early_distance_w15.png`, `sec6_cfg_early_distance_w30.png` |
| Figure 9, left/right | `sec6_cfg_prediction_gap_fit_w15_z.png`, `sec6_cfg_prediction_gap_fit_w30_z.png` |

The scripts save the panels separately; the paper combines them in its typeset two-column layout. The convex-hull script additionally saves the other step-size comparisons.

## Code organization

```text
experiments/
  sec4.py                                  shared geometry and flow utilities
  sec5_flow_ode_discrete_absorption.py     convex-hull experiment
  sec5_final_local_cluster.py              local-cluster experiment
  sec6_cfg_early_attraction.py             CFG trajectories, early distance, and gap fit
  sec6_cfg_final_cluster_distance.py       CFG final-cluster distance
  output_paths.py                          common output-directory handling
reproduce.py                               experiment runner and run manifest
```

The historical section-based filenames are retained to preserve the connection with the research code. Named constants near the beginning of each script contain the seeds, time intervals, step sizes, neighborhood radii, and plotting settings.

## Validation

All four experiment entry points were rerun on macOS with Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, and Matplotlib 3.9.4. They produced the ten panels listed above, plus the convex-hull step-size comparisons. The ten paper panels matched the original locally saved figures pixel for pixel in this environment. Small rendering differences may occur with other library versions or operating systems.

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
