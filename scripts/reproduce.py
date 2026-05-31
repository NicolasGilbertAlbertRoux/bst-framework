from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    "experiments/01_closure/canonical_closure_law_test.py",
    "experiments/02_projection/latent_aliasing_scan.py",
    "experiments/03_memory/Rlambda_generative_memory_test.py",
    "experiments/04_arbitration/arbitration_as_branch_selector_test.py",
    "experiments/05_zeta/zeta_final_falsifiability_test.py",
    "experiments/06_geometry/generative_curvature_activation_test.py",
]

for i, exp in enumerate(EXPERIMENTS, start=1):
    print(f"[{i}/{len(EXPERIMENTS)}] {Path(exp).name}")
    subprocess.run([sys.executable, str(ROOT / exp)], check=False)