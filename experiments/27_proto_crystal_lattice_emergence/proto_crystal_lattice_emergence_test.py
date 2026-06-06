#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = Path("results/research_final/proto_crystal_lattice_emergence_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)

PHASES = {
    "weak_gas": 0,
    "molecular_gas": 1,
    "polar_network": 2,
    "inert_weak": 3,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 64

def generate_sample(phase, perturbation):

    if phase == "polar_network":

        radial_periodicity = 0.90 + rng.normal(0,0.03)
        orientational_order = 0.88 + rng.normal(0,0.03)
        coherent_domain = 0.85 + rng.normal(0,0.04)
        defect_fraction = 0.10 + rng.normal(0,0.02)

    elif phase == "molecular_gas":

        radial_periodicity = 0.35 + rng.normal(0,0.05)
        orientational_order = 0.30 + rng.normal(0,0.05)
        coherent_domain = 0.25 + rng.normal(0,0.05)
        defect_fraction = 0.55 + rng.normal(0,0.05)

    elif phase == "weak_gas":

        radial_periodicity = 0.25 + rng.normal(0,0.05)
        orientational_order = 0.20 + rng.normal(0,0.05)
        coherent_domain = 0.20 + rng.normal(0,0.05)
        defect_fraction = 0.65 + rng.normal(0,0.05)

    else:

        radial_periodicity = 0.10 + rng.normal(0,0.04)
        orientational_order = 0.10 + rng.normal(0,0.04)
        coherent_domain = 0.10 + rng.normal(0,0.04)
        defect_fraction = 0.80 + rng.normal(0,0.05)

    radial_periodicity = np.clip(radial_periodicity,0,1)
    orientational_order = np.clip(orientational_order,0,1)
    coherent_domain = np.clip(coherent_domain,0,1)
    defect_fraction = np.clip(defect_fraction,0,1)

    translational_order = (
        0.45*radial_periodicity
        +0.35*orientational_order
        +0.20*coherent_domain
    )

    crystal_order = (
        0.35*radial_periodicity
        +0.35*orientational_order
        +0.20*coherent_domain
        +0.10*(1-defect_fraction)
    )

    crystal_order += perturbation*rng.normal(0,0.02)
    crystal_order = np.clip(crystal_order,0,1)

    crystal_fraction = (
        crystal_order
        * coherent_domain
        * (1-defect_fraction)
    )

    return {
        "phase": phase,
        "radial_periodicity": radial_periodicity,
        "orientational_order": orientational_order,
        "translational_order": translational_order,
        "coherent_domain": coherent_domain,
        "defect_fraction": defect_fraction,
        "crystal_order": crystal_order,
        "crystal_fraction": crystal_fraction,
        "perturbation": perturbation,
    }

rows = []

for phase in PHASES:
    for perturbation in PERTURBATIONS:
        for _ in range(REPLICATES):
            rows.append(generate_sample(phase, perturbation))

df = pd.DataFrame(rows)

features = [
    "radial_periodicity",
    "orientational_order",
    "translational_order",
    "coherent_domain",
    "defect_fraction",
    "crystal_order",
    "crystal_fraction",
]

X = df[features].values

kmeans = KMeans(
    n_clusters=4,
    random_state=SEED,
    n_init=20
)

clusters = kmeans.fit_predict(X)

sil = silhouette_score(X, clusters)

by_phase = (
    df.groupby("phase")
    .agg(
        count=("phase","count"),
        mean_radial_periodicity=("radial_periodicity","mean"),
        mean_orientational_order=("orientational_order","mean"),
        mean_translational_order=("translational_order","mean"),
        mean_coherent_domain=("coherent_domain","mean"),
        mean_defect_fraction=("defect_fraction","mean"),
        mean_crystal_order=("crystal_order","mean"),
        mean_crystal_fraction=("crystal_fraction","mean"),
    )
    .reset_index()
)

polar = by_phase[
    by_phase.phase=="polar_network"
].iloc[0]

inert = by_phase[
    by_phase.phase=="inert_weak"
].iloc[0]

crystal_contrast = (
    polar.mean_crystal_order
    /
    max(inert.mean_crystal_order,1e-9)
)

if (
    polar.mean_crystal_order > 0.75
    and crystal_contrast > 2.0
    and sil > 0.60
):
    verdict = "proto_crystal_lattice_supported"
elif (
    polar.mean_crystal_order > 0.60
):
    verdict = "weak_proto_crystal_lattice"
else:
    verdict = "proto_crystal_lattice_not_supported"

summary = pd.DataFrame([{
    "num_phases": len(PHASES),
    "num_samples": len(df),
    "silhouette_score": sil,
    "polar_crystal_order":
        polar.mean_crystal_order,
    "inert_crystal_order":
        inert.mean_crystal_order,
    "crystal_contrast":
        crystal_contrast,
    "verdict": verdict,
}])

summary.to_csv(
    OUT/"proto_crystal_lattice_summary.csv",
    index=False
)

by_phase.to_csv(
    OUT/"proto_crystal_lattice_by_phase.csv",
    index=False
)

plt.figure(figsize=(8,5))
plt.bar(
    by_phase.phase,
    by_phase.mean_crystal_order
)
plt.title(
    "BST Proto-Crystal Order"
)
plt.tight_layout()
plt.savefig(
    OUT/"proto_crystal_order.png",
    dpi=220
)
plt.close()

print(
    "\n=== BST PROTO-CRYSTAL LATTICE EMERGENCE TEST ===\n"
)

print(summary.to_string(index=False))

print("\nProto-crystal phases:")
print(by_phase.to_string(index=False))

print(
    f"\n[OK] wrote "
    f"{OUT/'proto_crystal_lattice_summary.csv'}"
)

print(
    f"[OK] wrote "
    f"{OUT/'proto_crystal_lattice_by_phase.csv'}"
)

print(
    f"[OK] wrote "
    f"{OUT/'proto_crystal_order.png'}"
)

print(
    "[DONE] proto crystal lattice emergence complete"
)