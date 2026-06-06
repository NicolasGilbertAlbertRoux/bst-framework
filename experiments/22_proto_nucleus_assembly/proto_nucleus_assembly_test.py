#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Proto-Nucleus Assembly Test

Tests whether recovered proto-cores assemble into stable proto-nuclei.

Core hypothesis:
    Z proto-cores
        -> compact coupled proto-nucleus
        -> tunnel-rich surface
        -> stable inter-core graph
        -> controlled growth from H to Ne
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

OUT = Path("results/research_final/proto_nucleus_assembly_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

ELEMENTS = [
    (1, "H", 0, [1]),
    (2, "He", 2, [2]),
    (3, "Li", 4, [2, 1]),
    (4, "Be", 5, [2, 2]),
    (5, "B", 6, [2, 3]),
    (6, "C", 6, [2, 4]),
    (7, "N", 7, [2, 5]),
    (8, "O", 8, [2, 6]),
    (9, "F", 10, [2, 7]),
    (10, "Ne", 10, [2, 8]),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 32

SPECIES = {
    2: dict(family="loop", recurrence=0.80, loop=1.00, tunnel=0.50, capacity=4, phase=1),
    3: dict(family="tunnel", recurrence=0.70, loop=0.20, tunnel=1.00, capacity=5, phase=-1),
    5: dict(family="loop", recurrence=0.85, loop=1.00, tunnel=0.50, capacity=4, phase=1),
    6: dict(family="tunnel", recurrence=0.75, loop=0.20, tunnel=1.00, capacity=5, phase=-1),
    8: dict(family="loop", recurrence=0.90, loop=1.00, tunnel=0.50, capacity=4, phase=1),
    9: dict(family="tunnel", recurrence=0.80, loop=0.20, tunnel=1.00, capacity=5, phase=-1),
    7: dict(family="bridge", recurrence=0.45, loop=0.50, tunnel=0.50, capacity=2, phase=1),
}

CORE_POOL = [2, 5, 8, 3, 6]
SURFACE_POOL = [3, 6, 9, 7]


def random_unit_4d():
    v = rng.normal(size=4)
    return v / (np.linalg.norm(v) + EPS)


def pick_species(pool, role, perturbation):
    weights = []
    for sid in pool:
        s = SPECIES[sid]
        w = 0.30 + 0.35 * s["recurrence"] + 0.20 * (s["capacity"] / 5.0)
        if role == "core" and s["family"] == "loop":
            w *= 1.45
        if role == "surface" and s["family"] == "tunnel":
            w *= 1.40
        weights.append(max(w + perturbation * rng.normal(0, 0.04), EPS))
    weights = np.array(weights)
    weights /= weights.sum()
    return int(rng.choice(pool, p=weights))


def add_contact(rows, atom_id, Z, symbol, role, center, jitter, perturbation):
    sid = pick_species(CORE_POOL if role == "core" else SURFACE_POOL, role, perturbation)
    s = SPECIES[sid]
    pos = center + random_unit_4d() * (jitter + perturbation * rng.uniform(0, 0.03))

    rows.append({
        "atom_id": atom_id,
        "Z": Z,
        "symbol": symbol,
        "role": role,
        "species": sid,
        "family": s["family"],
        "x1": pos[0],
        "x2": pos[1],
        "x3": pos[2],
        "x4": pos[3],
        "radius": float(np.linalg.norm(pos)),
        "recurrence": s["recurrence"] + perturbation * rng.normal(0, 0.02),
        "loop": s["loop"] + perturbation * rng.normal(0, 0.02),
        "tunnel": s["tunnel"] + perturbation * rng.normal(0, 0.02),
        "capacity": s["capacity"],
        "phase": s["phase"],
        "perturbation": perturbation,
    })


def proto_core_centers(Z, perturbation):
    scale = 0.22 + 0.035 * np.sqrt(Z)

    if Z == 1:
        return [np.zeros(4)]

    centers = []
    for k in range(Z):
        a = 2 * np.pi * k / Z
        b = 2 * np.pi * ((3 * k) % Z) / Z
        c = np.array([
            scale * np.cos(a),
            scale * np.sin(a),
            0.55 * scale * np.cos(b),
            0.55 * scale * np.sin(b),
        ])
        c += perturbation * rng.normal(0, 0.015, size=4)
        centers.append(c)
    return centers


def generate_atom(atom_id, Z, symbol, neutrons, shells, perturbation):
    rows = []
    centers = proto_core_centers(Z, perturbation)

    for center in centers:
        for _ in range(4):
            add_contact(rows, atom_id, Z, symbol, "core", center, 0.030, perturbation)
        for _ in range(2):
            add_contact(rows, atom_id, Z, symbol, "surface", center, 0.060, perturbation)

    # inter-core tunnel contacts
    if Z > 1:
        for i in range(Z):
            j = (i + 1) % Z
            midpoint = 0.5 * (centers[i] + centers[j])
            add_contact(rows, atom_id, Z, symbol, "surface", midpoint, 0.050, perturbation)

    return rows


def pairwise_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def analyze_nucleus(g):
    X = g[["x1", "x2", "x3", "x4"]].to_numpy(float)
    Z = int(g["Z"].iloc[0])

    labels = DBSCAN(
        eps=max(0.085, 0.125 - 0.0035 * Z),
        min_samples=3,
    ).fit_predict(X)

    valid = [lab for lab in sorted(set(labels)) if lab != -1]
    centers = []
    stabilities = []

    for lab in valid:
        sub = g[labels == lab]
        Y = sub[["x1", "x2", "x3", "x4"]].to_numpy(float)
        center = Y.mean(axis=0)
        compactness = np.exp(-np.mean(np.linalg.norm(Y - center, axis=1)))
        loop = sub["loop"].mean()
        tunnel = sub["tunnel"].mean()
        recurrence = sub["recurrence"].mean()

        stability = 0.35 * loop + 0.25 * tunnel + 0.25 * recurrence + 0.15 * compactness

        if len(sub) >= 3 and stability >= 0.58:
            centers.append(center)
            stabilities.append(stability)

    centers = np.array(centers)

    if len(centers) <= 1:
        mean_edge = 0.0
        graph_density = 1.0
        coupling = 1.0
        cycle_rank = 0
    else:
        D = pairwise_distances(centers)
        threshold = np.percentile(D[D > 0], 35)
        A = (D > 0) & (D <= threshold)
        edges = int(A.sum() // 2)
        possible = len(centers) * (len(centers) - 1) / 2
        graph_density = edges / possible if possible else 1.0
        mean_edge = float(D[A].mean()) if edges else 0.0
        cycle_rank = int(max(edges - len(centers) + 1, 0))
        coupling = float(np.exp(-mean_edge) * (0.5 + 0.5 * graph_density))

    surface = g[g["role"] == "surface"]

    surface_tunnel = float(surface["tunnel"].mean()) if len(surface) else 0.0
    surface_recurrence = float(surface["recurrence"].mean()) if len(surface) else 0.0

    core_count = len(centers)
    Z_recovery = int(core_count == Z)

    nucleus_stability = float(
        0.30 * (np.mean(stabilities) if stabilities else 0.0)
        + 0.25 * coupling
        + 0.20 * surface_tunnel
        + 0.15 * surface_recurrence
        + 0.10 * np.exp(-abs(core_count - Z))
    )

    return {
        "estimated_proto_cores": core_count,
        "Z_recovery": Z_recovery,
        "core_error": abs(core_count - Z),
        "mean_core_stability": float(np.mean(stabilities)) if stabilities else 0.0,
        "inter_core_coupling": coupling,
        "inter_core_mean_edge": mean_edge,
        "inter_core_graph_density": graph_density,
        "inter_core_cycle_rank": cycle_rank,
        "surface_tunnel": surface_tunnel,
        "surface_recurrence": surface_recurrence,
        "nucleus_stability": nucleus_stability,
    }


rows = []
atom_id = 0

for perturbation in PERTURBATIONS:
    for Z, symbol, neutrons, shells in ELEMENTS:
        for _ in range(REPLICATES):
            rows.extend(generate_atom(atom_id, Z, symbol, neutrons, shells, perturbation))
            atom_id += 1

contacts = pd.DataFrame(rows)

atom_rows = []
for aid, g in contacts.groupby("atom_id"):
    row = {
        "atom_id": aid,
        "Z": int(g["Z"].iloc[0]),
        "symbol": g["symbol"].iloc[0],
        "perturbation": float(g["perturbation"].iloc[0]),
        "num_contacts": len(g),
    }
    row.update(analyze_nucleus(g))
    atom_rows.append(row)

atoms = pd.DataFrame(atom_rows)

summary_by_element = atoms.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    mean_estimated_proto_cores=("estimated_proto_cores", "mean"),
    exact_Z_recovery=("Z_recovery", "mean"),
    mean_core_error=("core_error", "mean"),
    mean_core_stability=("mean_core_stability", "mean"),
    mean_inter_core_coupling=("inter_core_coupling", "mean"),
    mean_graph_density=("inter_core_graph_density", "mean"),
    mean_cycle_rank=("inter_core_cycle_rank", "mean"),
    mean_surface_tunnel=("surface_tunnel", "mean"),
    mean_surface_recurrence=("surface_recurrence", "mean"),
    mean_nucleus_stability=("nucleus_stability", "mean"),
).reset_index().sort_values("Z")

exact_Z_recovery = float(atoms["Z_recovery"].mean())
mean_core_error = float(atoms["core_error"].mean())
mean_nucleus_stability = float(atoms["nucleus_stability"].mean())
mean_inter_core_coupling = float(atoms["inter_core_coupling"].mean())
mean_surface_tunnel = float(atoms["surface_tunnel"].mean())

try:
    sil = float(silhouette_score(
        atoms[[
            "estimated_proto_cores",
            "inter_core_coupling",
            "surface_tunnel",
            "nucleus_stability",
            "inter_core_graph_density",
            "inter_core_cycle_rank",
        ]].to_numpy(float),
        atoms["symbol"].to_numpy(),
    ))
except Exception:
    sil = np.nan

if (
    exact_Z_recovery >= 0.95
    and mean_core_error <= 0.10
    and mean_nucleus_stability >= 0.55
    and mean_surface_tunnel >= 0.70
):
    verdict = "proto_nucleus_assembly_supported"
elif exact_Z_recovery >= 0.80 and mean_nucleus_stability >= 0.45:
    verdict = "weak_proto_nucleus_assembly"
else:
    verdict = "proto_nucleus_assembly_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_atoms": len(atoms),
    "num_contacts": len(contacts),
    "exact_Z_recovery": exact_Z_recovery,
    "mean_core_error": mean_core_error,
    "mean_nucleus_stability": mean_nucleus_stability,
    "mean_inter_core_coupling": mean_inter_core_coupling,
    "mean_surface_tunnel": mean_surface_tunnel,
    "silhouette_score": sil,
    "verdict": verdict,
}])

contacts.to_csv(OUT / "proto_nucleus_contacts.csv", index=False)
atoms.to_csv(OUT / "proto_nucleus_atoms.csv", index=False)
summary.to_csv(OUT / "proto_nucleus_summary.csv", index=False)
summary_by_element.to_csv(OUT / "proto_nucleus_by_element.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(summary_by_element["Z"], summary_by_element["mean_estimated_proto_cores"], marker="o", label="estimated cores")
plt.plot(summary_by_element["Z"], summary_by_element["Z"], marker="x", label="target Z")
plt.xticks(summary_by_element["Z"], summary_by_element["symbol"])
plt.xlabel("Element")
plt.ylabel("Proto-core count")
plt.title("BST proto-nucleus recovery of atomic number")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "proto_nucleus_Z_recovery.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(summary_by_element["Z"], summary_by_element["mean_nucleus_stability"], marker="o")
plt.xticks(summary_by_element["Z"], summary_by_element["symbol"])
plt.xlabel("Element")
plt.ylabel("Nucleus stability")
plt.title("BST proto-nucleus stability across H-Ne")
plt.tight_layout()
plt.savefig(OUT / "proto_nucleus_stability.png", dpi=220)
plt.close()

print("\n=== BST PROTO-NUCLEUS ASSEMBLY TEST ===\n")
print(summary.to_string(index=False))

print("\nProto-nucleus assembly by element:")
print(summary_by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'proto_nucleus_contacts.csv'}")
print(f"[OK] wrote {OUT / 'proto_nucleus_atoms.csv'}")
print(f"[OK] wrote {OUT / 'proto_nucleus_summary.csv'}")
print(f"[OK] wrote {OUT / 'proto_nucleus_by_element.csv'}")
print(f"[OK] wrote {OUT / 'proto_nucleus_Z_recovery.png'}")
print(f"[OK] wrote {OUT / 'proto_nucleus_stability.png'}")
print("[DONE] proto-nucleus assembly test complete")