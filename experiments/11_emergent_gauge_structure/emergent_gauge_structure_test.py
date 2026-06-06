#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Emergent Gauge Structure Test
===================================

Purpose
-------
Test whether latent perturbations can change internal closure coordinates
while leaving reconstructed observables approximately invariant.

Gauge-like behavior is identified when:

    latent_change >> observable_change

This supports the interpretation of gauge structure as reconstruction
redundancy: distinct latent descriptions producing nearly identical
observable states.

Outputs
-------
results/research_final/emergent_gauge_structure_test/
    emergent_gauge_structure_summary.csv
    emergent_gauge_structure_trials.csv
    emergent_gauge_ratio.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/research_final/emergent_gauge_structure_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
N = 512
LATENT_DIM = 8
OBS_DIM = 3
N_TRIALS = 400
PERTURBATION_SCALE = 0.15
EPS = 1e-12


def make_projection(rng):
    """
    Construct a rank-deficient observable projection.

    Null directions of this projection represent candidate gauge directions.
    """
    projection = rng.normal(size=(OBS_DIM, LATENT_DIM))
    return projection


def reconstruct_observable(z, projection):
    return projection @ z


def null_space_basis(matrix, tol=1e-10):
    _, s, vh = np.linalg.svd(matrix)
    rank = int(np.sum(s > tol))
    return vh[rank:].T


def normalized_norm(x):
    return float(np.linalg.norm(x))


def main():
    print("\n=== BST EMERGENT GAUGE STRUCTURE TEST ===")

    rng = np.random.default_rng(SEED)

    projection = make_projection(rng)
    gauge_basis = null_space_basis(projection)

    if gauge_basis.shape[1] == 0:
        raise RuntimeError("No null-space directions found; projection is full rank.")

    rows = []

    for trial in range(N_TRIALS):
        z = rng.normal(size=LATENT_DIM)

        # Gauge-like perturbation: lives in approximate projection null space.
        coeffs = rng.normal(size=gauge_basis.shape[1])
        delta_gauge = gauge_basis @ coeffs
        delta_gauge *= PERTURBATION_SCALE / (np.linalg.norm(delta_gauge) + EPS)

        # Generic perturbation: arbitrary latent direction.
        delta_generic = rng.normal(size=LATENT_DIM)
        delta_generic *= PERTURBATION_SCALE / (np.linalg.norm(delta_generic) + EPS)

        o = reconstruct_observable(z, projection)

        o_gauge = reconstruct_observable(z + delta_gauge, projection)
        o_generic = reconstruct_observable(z + delta_generic, projection)

        latent_gauge_change = normalized_norm(delta_gauge)
        latent_generic_change = normalized_norm(delta_generic)

        observable_gauge_change = normalized_norm(o_gauge - o)
        observable_generic_change = normalized_norm(o_generic - o)

        gauge_ratio = observable_gauge_change / (latent_gauge_change + EPS)
        generic_ratio = observable_generic_change / (latent_generic_change + EPS)

        rows.append({
            "trial": trial,
            "latent_gauge_change": latent_gauge_change,
            "observable_gauge_change": observable_gauge_change,
            "gauge_ratio": gauge_ratio,
            "latent_generic_change": latent_generic_change,
            "observable_generic_change": observable_generic_change,
            "generic_ratio": generic_ratio,
            "suppression_factor": generic_ratio / (gauge_ratio + EPS),
        })

    df = pd.DataFrame(rows)

    mean_gauge_ratio = float(df["gauge_ratio"].mean())
    mean_generic_ratio = float(df["generic_ratio"].mean())
    mean_suppression = float(df["suppression_factor"].mean())
    median_suppression = float(df["suppression_factor"].median())

    if mean_gauge_ratio < 1e-8:
        verdict = "exact_gauge_null_directions"
    elif mean_suppression > 100:
        verdict = "strong_emergent_gauge_redundancy"
    elif mean_suppression > 10:
        verdict = "moderate_emergent_gauge_redundancy"
    elif mean_suppression > 3:
        verdict = "weak_emergent_gauge_redundancy"
    else:
        verdict = "no_gauge_redundancy"

    summary = pd.DataFrame([{
        "latent_dim": LATENT_DIM,
        "observable_dim": OBS_DIM,
        "nullity": int(gauge_basis.shape[1]),
        "num_trials": N_TRIALS,
        "mean_gauge_ratio": mean_gauge_ratio,
        "mean_generic_ratio": mean_generic_ratio,
        "mean_suppression_factor": mean_suppression,
        "median_suppression_factor": median_suppression,
        "verdict": verdict,
    }])

    trials_path = OUT / "emergent_gauge_structure_trials.csv"
    summary_path = OUT / "emergent_gauge_structure_summary.csv"

    df.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(df["gauge_ratio"], bins=40, alpha=0.7, label="gauge-like perturbations")
    plt.hist(df["generic_ratio"], bins=40, alpha=0.7, label="generic perturbations")
    plt.xlabel("observable change / latent change")
    plt.ylabel("count")
    plt.title("Gauge redundancy diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "emergent_gauge_ratio.png", dpi=220)
    plt.close()

    print(summary.to_string(index=False))
    print(f"\n[OK] wrote {trials_path}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {OUT / 'emergent_gauge_ratio.png'}")
    print("[DONE] emergent gauge structure test complete")


if __name__ == "__main__":
    main()