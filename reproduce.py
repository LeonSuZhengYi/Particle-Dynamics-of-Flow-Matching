"""Run the two-dimensional synthetic experiments with the original settings."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from importlib.metadata import version

EXPERIMENTS = {
    'convex-hull': 'sec5_flow_ode_discrete_absorption.py',
    'local-clusters': 'sec5_final_local_cluster.py',
    'cfg': 'sec6_cfg_early_attraction.py',
    'cfg-final': 'sec6_cfg_final_cluster_distance.py',
    'projection': 'sec4_projection_ode_discrete.py',
    'bean-flow': 'sec4.py',
}
PAPER = ['convex-hull', 'local-clusters', 'cfg', 'cfg-final']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['paper', 'all', *EXPERIMENTS], default='paper')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = PAPER if args.experiment == 'paper' else list(EXPERIMENTS) if args.experiment == 'all' else [args.experiment]
    env = dict(os.environ, JMLR_OUTPUT_DIR=str(output), MPLBACKEND='Agg')
    report = {'python': platform.python_version(), 'packages': {p: version(p) for p in ['numpy', 'scipy', 'matplotlib']}, 'experiments': {}}
    for name in selected:
        print(f'Running {name}...', flush=True)
        start = time.monotonic()
        script = Path(__file__).resolve().parent / 'experiments' / EXPERIMENTS[name]
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(script)], env=env, stdout=log, stderr=subprocess.STDOUT)
        report['experiments'][name] = {'exit_code': result.returncode, 'seconds': round(time.monotonic() - start, 2)}
        (output / 'run-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(f'{name} failed; see {output / (name + ".log")}')
        print(f'Finished {name}.', flush=True)
    print(f'Figures and logs: {output}')


if __name__ == '__main__':
    main()
