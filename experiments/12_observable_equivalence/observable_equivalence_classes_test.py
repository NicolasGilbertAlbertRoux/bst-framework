#!/usr/bin/env python3

"""
BST — Observable Equivalence Classes Test

Theorem T18 precursor:
States differing only along gauge-null directions
must belong to the same observable equivalence class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


OUTDIR = (
    Path("results")
    / "research_final"
    / "observable_equivalence_classes_test"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

LATENT_DIM = 8
OBS_DIM = 3

NUM_BASE_STATES = 100
NUM_GAUGE_SAMPLES = 25

OBS_TOL = 1e-10

# --------------------------------------------------
# LATENT -> OBSERVABLE MAP
# --------------------------------------------------

A = np.random.randn(OBS_DIM, LATENT_DIM)

# rank = observable dimension
u, s, vh = np.linalg.svd(A, full_matrices=False)
A = u @ np.diag(s) @ vh

rank = np.linalg.matrix_rank(A)
nullity = LATENT_DIM - rank

# --------------------------------------------------
# NULL SPACE
# --------------------------------------------------

u_full, s_full, vh_full = np.linalg.svd(A, full_matrices=True)

null_basis = vh_full[rank:].T

# --------------------------------------------------
# GENERATE EQUIVALENCE CLASSES
# --------------------------------------------------

records = []

class_id = 0

for _ in range(NUM_BASE_STATES):

    z0 = np.random.randn(LATENT_DIM)

    x0 = A @ z0

    for _ in range(NUM_GAUGE_SAMPLES):

        coeffs = np.random.randn(nullity)

        dz = null_basis @ coeffs

        z1 = z0 + dz

        x1 = A @ z1

        observable_error = np.linalg.norm(x1 - x0)

        records.append(
            {
                "class_id": class_id,
                "observable_error": observable_error,
                "latent_shift_norm": np.linalg.norm(dz),
            }
        )

    class_id += 1

df = pd.DataFrame(records)

# --------------------------------------------------
# CLASS VALIDATION
# --------------------------------------------------

valid_ratio = (
    df["observable_error"] < OBS_TOL
).mean()

summary = pd.DataFrame(
    [
        {
            "latent_dim": LATENT_DIM,
            "observable_dim": OBS_DIM,
            "rank": rank,
            "nullity": nullity,
            "num_classes": NUM_BASE_STATES,
            "samples_per_class": NUM_GAUGE_SAMPLES,
            "mean_observable_error":
                df["observable_error"].mean(),
            "max_observable_error":
                df["observable_error"].max(),
            "valid_ratio":
                valid_ratio,
            "verdict":
                (
                    "observable_equivalence_confirmed"
                    if valid_ratio > 0.999
                    else "not_confirmed"
                ),
        }
    ]
)

# --------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------

latent_cloud = []

for _ in range(300):

    z0 = np.random.randn(LATENT_DIM)

    coeffs = np.random.randn(nullity)

    dz = null_basis @ coeffs

    latent_cloud.append(z0 + dz)

latent_cloud = np.asarray(latent_cloud)

pca = PCA(n_components=2)
proj = pca.fit_transform(latent_cloud)

plt.figure(figsize=(6, 5))
plt.scatter(
    proj[:, 0],
    proj[:, 1],
    s=10,
    alpha=0.7,
)
plt.title("Observable Equivalence Classes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(
    OUTDIR / "observable_equivalence_classes.png",
    dpi=200,
)
plt.close()

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

df.to_csv(
    OUTDIR / "observable_equivalence_trials.csv",
    index=False,
)

summary.to_csv(
    OUTDIR / "observable_equivalence_summary.csv",
    index=False,
)

print("\n=== BST OBSERVABLE EQUIVALENCE CLASSES TEST ===\n")
print(summary.to_string(index=False))

print(
    "\n[OK] wrote "
    "results/research_final/"
    "observable_equivalence_classes_test/"
    "observable_equivalence_trials.csv"
)

print(
    "[OK] wrote "
    "results/research_final/"
    "observable_equivalence_classes_test/"
    "observable_equivalence_summary.csv"
)

print(
    "[OK] wrote "
    "results/research_final/"
    "observable_equivalence_classes_test/"
    "observable_equivalence_classes.png"
)

print(
    "[DONE] observable equivalence classes test complete"
)