#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Proto-Condensed Matter Network Test

Tests whether stable proto-molecules can assemble into extended
proto-material networks.

Hypothesis:
    proto-atoms
        -> proto-molecules
        -> stable molecular graph
        -> condensed proto-matter phases

This is not a chemistry engine. It tests whether the BST descriptors already
used for proto-periodicity and proto-bonding can support higher-order,
network-level matter organization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

OUT = Path("results/research_final/proto_condensed_matter_network_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

MOLECULE_TYPES = {
    "H2":  dict(bond_stability=0.978, polarity=0.00, valence_ports=0, mass=2,  phase_hint="gas"),
    "O2":  dict(bond_stability=0.978, polarity=0.00, valence_ports=0, mass=32, phase_hint="gas"),
    "N2":  dict(bond_stability=0.978, polarity=0.00, valence_ports=0, mass=28, phase_hint="gas"),
    "F2":  dict(bond_stability=0.978, polarity=0.00, valence_ports=0, mass=38, phase_hint="gas"),
    "H2O": dict(bond_stability=0.950, polarity=0.31, valence_ports=2, mass=18, phase_hint="polar_network"),
    "CO2": dict(bond_stability=0.957, polarity=0.22, valence_ports=0, mass=44, phase_hint="linear_gas"),
    "CH4": dict(bond_stability=0.973, polarity=0.09, valence_ports=0, mass=16, phase_hint="molecular_gas"),
    "NH3": dict(bond_stability=0.962, polarity=0.21, valence_ports=1, mass=17, phase_hint="polar_network"),
    "Ne2": dict(bond_stability=0.294, polarity=0.00, valence_ports=0, mass=40, phase_hint="inert_weak"),
}

PHASES = {
    "weak_gas": 0,
    "molecular_gas": 1,
    "polar_network": 2,
    "inert_weak": 3,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 64
NETWORK_SIZE = 48


def sample_molecule_type(phase):
    if phase == "weak_gas":
        pool = ["H2", "O2", "N2", "F2", "CO2"]
        p = np.array([0.20, 0.22, 0.22, 0.12, 0.24])
    elif phase == "molecular_gas":
        pool = ["CH4", "CO2", "H2", "N2"]
        p = np.array([0.45, 0.20, 0.20, 0.15])
    elif phase == "polar_network":
        pool = ["H2O", "NH3"]
        p = np.array([0.70, 0.30])
    elif phase == "inert_weak":
        pool = ["Ne2"]
        p = np.array([1.0])
    else:
        raise ValueError(phase)
    return str(rng.choice(pool, p=p / p.sum()))


def interaction_strength(a, b, perturbation):
    A = MOLECULE_TYPES[a]
    B = MOLECULE_TYPES[b]

    stability = np.sqrt(A["bond_stability"] * B["bond_stability"])
    polarity_coupling = A["polarity"] * B["polarity"]
    port_coupling = min(A["valence_ports"], B["valence_ports"]) / 2.0
    mass_dispersion = np.sqrt(A["mass"] * B["mass"]) / 50.0

    weak_vdw = 0.08 * mass_dispersion
    polar_bridge = 0.55 * polarity_coupling
    network_bridge = 0.30 * port_coupling

    inert_penalty = 0.18 if (a == "Ne2" or b == "Ne2") else 1.0

    strength = (
        0.25 * stability
        + weak_vdw
        + polar_bridge
        + network_bridge
    ) * inert_penalty

    return float(max(0.0, strength + perturbation * rng.normal(0, 0.015)))


def generate_network(network_id, target_phase, perturbation):
    molecules = [sample_molecule_type(target_phase) for _ in range(NETWORK_SIZE)]

    W = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=float)

    for i in range(NETWORK_SIZE):
        for j in range(i + 1, NETWORK_SIZE):
            strength = interaction_strength(molecules[i], molecules[j], perturbation)

            threshold = {
                "weak_gas": 0.26,
                "molecular_gas": 0.24,
                "polar_network": 0.30,
                "inert_weak": 0.18,
            }[target_phase]

            probability = 1.0 / (1.0 + np.exp(-18.0 * (strength - threshold)))

            if rng.random() < probability:
                W[i, j] = W[j, i] = strength

    edge_count = int(np.sum(W > 0) // 2)
    possible = NETWORK_SIZE * (NETWORK_SIZE - 1) / 2
    density = edge_count / possible

    degrees = np.sum(W > 0, axis=1)
    weighted_degree = W.sum(axis=1)

    if edge_count:
        mean_edge_weight = float(W[W > 0].mean())
        total_weight = float(W.sum() / 2)
    else:
        mean_edge_weight = 0.0
        total_weight = 0.0

    # Simple triangle and clustering estimates.
    A = (W > 0).astype(int)
    triangles = int(np.trace(A @ A @ A) // 6)

    clustering_vals = []
    for i in range(NETWORK_SIZE):
        neigh = np.where(A[i] > 0)[0]
        k = len(neigh)
        if k < 2:
            clustering_vals.append(0.0)
        else:
            sub = A[np.ix_(neigh, neigh)]
            local_edges = sub.sum() / 2
            clustering_vals.append(local_edges / (k * (k - 1) / 2))

    clustering = float(np.mean(clustering_vals))

    # Components.
    seen = set()
    comp_sizes = []
    for i in range(NETWORK_SIZE):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in np.where(A[u] > 0)[0]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comp_sizes.append(size)

    largest_component_ratio = max(comp_sizes) / NETWORK_SIZE
    num_components = len(comp_sizes)

    recurrence = (
        0.35 * mean_edge_weight
        + 0.25 * clustering
        + 0.25 * largest_component_ratio
        + 0.15 * np.exp(-num_components / NETWORK_SIZE)
    )

    condensed_stability = (
        0.30 * largest_component_ratio
        + 0.25 * mean_edge_weight
        + 0.20 * clustering
        + 0.15 * density
        + 0.10 * recurrence
    )

    return {
        "network_id": network_id,
        "target_phase": target_phase,
        "phase_id": PHASES[target_phase],
        "perturbation": perturbation,
        "edge_count": edge_count,
        "density": density,
        "mean_degree": float(degrees.mean()),
        "mean_weighted_degree": float(weighted_degree.mean()),
        "mean_edge_weight": mean_edge_weight,
        "total_weight": total_weight,
        "triangles": triangles,
        "clustering": clustering,
        "largest_component_ratio": largest_component_ratio,
        "num_components": num_components,
        "recurrence": recurrence,
        "condensed_stability": condensed_stability,
        "mean_molecular_stability": float(np.mean([MOLECULE_TYPES[m]["bond_stability"] for m in molecules])),
        "mean_polarity": float(np.mean([MOLECULE_TYPES[m]["polarity"] for m in molecules])),
        "mean_valence_ports": float(np.mean([MOLECULE_TYPES[m]["valence_ports"] for m in molecules])),
        "mean_mass": float(np.mean([MOLECULE_TYPES[m]["mass"] for m in molecules])),
    }


rows = []
network_id = 0

for perturbation in PERTURBATIONS:
    for phase in PHASES:
        for _ in range(REPLICATES):
            rows.append(generate_network(network_id, phase, perturbation))
            network_id += 1

df = pd.DataFrame(rows)

feature_cols = [
    "edge_count",
    "density",
    "mean_degree",
    "mean_weighted_degree",
    "mean_edge_weight",
    "total_weight",
    "triangles",
    "clustering",
    "largest_component_ratio",
    "num_components",
    "recurrence",
    "condensed_stability",
    "mean_molecular_stability",
    "mean_polarity",
    "mean_valence_ports",
    "mean_mass",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

kmeans = KMeans(n_clusters=len(PHASES), random_state=SEED, n_init=20)
clusters = kmeans.fit_predict(Xn)

try:
    sil = float(silhouette_score(Xn, df["target_phase"]))
except Exception:
    sil = np.nan

ari = float(adjusted_rand_score(df["phase_id"], clusters))

by_phase = df.groupby("target_phase").agg(
    count=("target_phase", "count"),
    mean_density=("density", "mean"),
    mean_edge_weight=("mean_edge_weight", "mean"),
    mean_clustering=("clustering", "mean"),
    mean_largest_component=("largest_component_ratio", "mean"),
    mean_components=("num_components", "mean"),
    mean_recurrence=("recurrence", "mean"),
    mean_condensed_stability=("condensed_stability", "mean"),
    mean_polarity=("mean_polarity", "mean"),
    mean_valence_ports=("mean_valence_ports", "mean"),
).reset_index()

polar = df[df["target_phase"] == "polar_network"]
inert = df[df["target_phase"] == "inert_weak"]

polar_stable_ratio = float(np.mean(polar["condensed_stability"] >= 0.45))
inert_stable_ratio = float(np.mean(inert["condensed_stability"] >= 0.45))

mean_polar_stability = float(polar["condensed_stability"].mean())
mean_inert_stability = float(inert["condensed_stability"].mean())
stability_contrast = mean_polar_stability / (mean_inert_stability + EPS)

mean_condensed_stability = float(df["condensed_stability"].mean())
mean_recurrence = float(df["recurrence"].mean())

if (
    polar_stable_ratio >= 0.90
    and inert_stable_ratio <= 0.10
    and stability_contrast >= 2.0
    and sil >= 0.60
):
    verdict = "proto_condensed_matter_network_supported"
elif polar_stable_ratio >= 0.75 and stability_contrast >= 1.5:
    verdict = "weak_proto_condensed_matter_network"
else:
    verdict = "proto_condensed_matter_network_not_supported"
    
summary = pd.DataFrame([{
    "num_phases": len(PHASES),
    "num_networks": len(df),
    "network_size": NETWORK_SIZE,
    "polar_stable_ratio": polar_stable_ratio,
    "inert_stable_ratio": inert_stable_ratio,
    "mean_polar_stability": mean_polar_stability,
    "mean_inert_stability": mean_inert_stability,
    "stability_contrast": stability_contrast,
    "mean_condensed_stability": mean_condensed_stability,
    "mean_recurrence": mean_recurrence,
    "phase_ARI": ari,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "proto_condensed_matter_networks.csv", index=False)
summary.to_csv(OUT / "proto_condensed_matter_summary.csv", index=False)
by_phase.to_csv(OUT / "proto_condensed_matter_by_phase.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(by_phase["target_phase"], by_phase["mean_condensed_stability"])
plt.axhline(0.45, linestyle="--")
plt.ylabel("Condensed stability")
plt.title("BST proto-condensed matter stability by phase")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT / "proto_condensed_matter_stability.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(by_phase["target_phase"], by_phase["mean_largest_component"])
plt.ylabel("Largest component ratio")
plt.title("BST proto-condensed matter connectivity")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT / "proto_condensed_matter_connectivity.png", dpi=220)
plt.close()

print("\n=== BST PROTO-CONDENSED MATTER NETWORK TEST ===\n")
print(summary.to_string(index=False))

print("\nProto-condensed matter by phase:")
print(by_phase.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'proto_condensed_matter_networks.csv'}")
print(f"[OK] wrote {OUT / 'proto_condensed_matter_summary.csv'}")
print(f"[OK] wrote {OUT / 'proto_condensed_matter_by_phase.csv'}")
print(f"[OK] wrote {OUT / 'proto_condensed_matter_stability.png'}")
print(f"[OK] wrote {OUT / 'proto_condensed_matter_connectivity.png'}")
print("[DONE] proto-condensed matter network test complete")