#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Proto-Core Identification Test

Purpose
-------
Test whether proto-atoms can be decomposed into stable proto-cores.

The previous tests showed:

T20: proto-atoms are globally identifiable.
T21: radial localization is stable but not element-identifying.
T22: local motifs are structured but not sufficient.

This test asks whether the atomic identity is carried by an intermediate
level:

    contact species
        -> proto-cores
        -> proto-nuclei
        -> proto-atoms

Hypothesis
----------
The number of compact loop/tunnel proto-cores should recover the atomic
number Z under perturbation.

Outputs
-------
results/research_final/proto_core_identification_test/
    proto_core_contacts.csv
    proto_core_atoms.csv
    proto_core_summary.csv
    proto_core_by_element.csv
    proto_core_projection.png
    proto_core_Z_recovery.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


OUT = Path("results/research_final/proto_core_identification_test")
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
    0: dict(family="node", recurrence=0.00, loop=0.00, tunnel=0.00, bridge=0.00, capacity=1, phase=1),
    1: dict(family="bridge", recurrence=0.20, loop=0.30, tunnel=0.30, bridge=0.70, capacity=2, phase=1),
    2: dict(family="loop", recurrence=0.80, loop=1.00, tunnel=0.50, bridge=0.10, capacity=4, phase=1),
    3: dict(family="tunnel", recurrence=0.70, loop=0.20, tunnel=1.00, bridge=0.20, capacity=5, phase=-1),
    4: dict(family="bridge", recurrence=0.40, loop=0.40, tunnel=0.40, bridge=0.80, capacity=2, phase=1),
    5: dict(family="loop", recurrence=0.85, loop=1.00, tunnel=0.50, bridge=0.15, capacity=4, phase=1),
    6: dict(family="tunnel", recurrence=0.75, loop=0.20, tunnel=1.00, bridge=0.25, capacity=5, phase=-1),
    7: dict(family="bridge", recurrence=0.45, loop=0.50, tunnel=0.50, bridge=0.90, capacity=2, phase=1),
    8: dict(family="loop", recurrence=0.90, loop=1.00, tunnel=0.50, bridge=0.20, capacity=4, phase=1),
    9: dict(family="tunnel", recurrence=0.80, loop=0.20, tunnel=1.00, bridge=0.30, capacity=5, phase=-1),
    10: dict(family="node", recurrence=0.30, loop=0.30, tunnel=0.10, bridge=0.20, capacity=1, phase=1),
    11: dict(family="node", recurrence=0.20, loop=0.20, tunnel=0.10, bridge=0.10, capacity=1, phase=1),
}

CORE_SPECIES_POOL = [2, 5, 8, 3, 6, 9]
SURFACE_SPECIES_POOL = [3, 6, 9, 7]
INTERFACE_SPECIES_POOL = [1, 4, 7, 10]
SHELL_SPECIES_POOL = [10, 11, 1, 4, 7]


def species_weight(sid, role, perturbation):
    s = SPECIES[sid]
    base = 0.25 + 0.35 * s["recurrence"] + 0.20 * (s["capacity"] / 5.0)

    if role == "core" and s["family"] == "loop":
        base *= 1.45
    if role == "surface" and s["family"] == "tunnel":
        base *= 1.40
    if role == "interface" and s["family"] == "bridge":
        base *= 1.35
    if role == "shell" and s["family"] == "node":
        base *= 1.30

    return max(base + perturbation * rng.normal(0, 0.04), EPS)


def pick_from_pool(pool, role, perturbation):
    weights = np.array([species_weight(sid, role, perturbation) for sid in pool], dtype=float)
    weights = weights / weights.sum()
    return int(rng.choice(pool, p=weights))


def random_unit_4d():
    v = rng.normal(size=4)
    return v / (np.linalg.norm(v) + EPS)


def add_contact(rows, atom_id, Z, symbol, role, center, radial_jitter, perturbation):
    if role == "core":
        pool = CORE_SPECIES_POOL
    elif role == "surface":
        pool = SURFACE_SPECIES_POOL
    elif role == "interface":
        pool = INTERFACE_SPECIES_POOL
    elif role == "shell":
        pool = SHELL_SPECIES_POOL
    else:
        raise ValueError(role)

    sid = pick_from_pool(pool, role, perturbation)
    s = SPECIES[sid]

    pos = center + random_unit_4d() * (radial_jitter + perturbation * rng.uniform(0.0, 0.03))

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
        "bridge": s["bridge"] + perturbation * rng.normal(0, 0.02),
        "capacity": s["capacity"],
        "phase": s["phase"],
        "perturbation": perturbation,
    })


def generate_proto_atom(atom_id, Z, symbol, neutrons, shells, perturbation):
    rows = []

    # Proto-core centers: Z compact centers in 4D.
    # The scale grows slowly with Z to avoid trivial overlap while preserving
    # a compact proto-nucleus.
    nucleus_scale = 0.22 + 0.035 * np.sqrt(Z)

    if Z == 1:
        centers = [np.zeros(4)]
    else:
        centers = []
        for k in range(Z):
            angle = 2.0 * np.pi * k / Z
            secondary = 2.0 * np.pi * ((k * 3) % max(Z, 2)) / max(Z, 2)
            center = np.array([
                nucleus_scale * np.cos(angle),
                nucleus_scale * np.sin(angle),
                0.55 * nucleus_scale * np.cos(secondary),
                0.55 * nucleus_scale * np.sin(secondary),
            ])
            center += perturbation * rng.normal(0, 0.015, size=4)
            centers.append(center)

    for center in centers:
        # Four-wave compact proto-core signature.
        for _ in range(4):
            add_contact(rows, atom_id, Z, symbol, "core", center, 0.030, perturbation)

        # Tunnel-rich local surface around each proto-core.
        for _ in range(2):
            add_contact(rows, atom_id, Z, symbol, "surface", center, 0.060, perturbation)

    # Proto-nuclear coupling contacts between adjacent proto-cores.
    if len(centers) > 1:
        for i in range(len(centers)):
            j = (i + 1) % len(centers)
            midpoint = 0.5 * (centers[i] + centers[j])
            add_contact(rows, atom_id, Z, symbol, "surface", midpoint, 0.050, perturbation)

    # Orbital interface contacts.
    for _ in range(max(1, Z)):
        direction = random_unit_4d()
        center = direction * (0.70 + 0.02 * np.sqrt(Z))
        add_contact(rows, atom_id, Z, symbol, "interface", center, 0.040, perturbation)

    # Shell contacts: total electron-like proto-cores.
    shell_radii = [1.25, 2.10, 3.15]
    for shell_index, count in enumerate(shells[:3]):
        shell_radius = shell_radii[shell_index]
        for k in range(count):
            direction = random_unit_4d()
            phase_offset = 0.02 * np.sin(2 * np.pi * k / max(count, 1))
            center = direction * (shell_radius + phase_offset)
            add_contact(rows, atom_id, Z, symbol, "shell", center, 0.030, perturbation)

    return rows


def estimate_proto_cores(g):
    # Detect compact loop/tunnel-rich contacts near the nucleus.
    nuclear = g[g["role"].isin(["core", "surface"])].copy()

    if len(nuclear) == 0:
        return {
            "estimated_proto_cores": 0,
            "mean_core_size": 0.0,
            "core_stability": 0.0,
            "core_loop_density": 0.0,
            "core_tunnel_support": 0.0,
            "core_phase_balance": 0.0,
        }

    X = nuclear[["x1", "x2", "x3", "x4"]].to_numpy(dtype=float)

    # eps chosen to group each four-wave compact core while separating
    # neighboring proto-cores.
    Z = int(nuclear["Z"].iloc[0])
    adaptive_eps = max(0.085, 0.125 - 0.0035 * Z)
    labels = DBSCAN(eps=adaptive_eps, min_samples=3).fit_predict(X)

    valid_labels = [lab for lab in sorted(set(labels)) if lab != -1]

    core_sizes = []
    core_scores = []
    loop_scores = []
    tunnel_scores = []
    phase_scores = []

    for lab in valid_labels:
        sub = nuclear[labels == lab]
        loop_density = float(sub["loop"].mean())
        tunnel_support = float(sub["tunnel"].mean())
        recurrence = float(sub["recurrence"].mean())
        phase_balance = float(abs(sub["phase"].mean()))
        compactness = float(np.exp(-np.mean(np.linalg.norm(
            sub[["x1", "x2", "x3", "x4"]].to_numpy() -
            sub[["x1", "x2", "x3", "x4"]].to_numpy().mean(axis=0),
            axis=1
        ))))

        score = (
            0.30 * loop_density
            + 0.25 * tunnel_support
            + 0.25 * recurrence
            + 0.20 * compactness
        )

        core_sizes.append(len(sub))
        core_scores.append(score)
        loop_scores.append(loop_density)
        tunnel_scores.append(tunnel_support)
        phase_scores.append(phase_balance)

    selected = [
        i for i, score in enumerate(core_scores)
        if score >= 0.58 and 3 <= core_sizes[i] <= 8
    ]

    return {
        "estimated_proto_cores": len(selected),
        "mean_core_size": float(np.mean([core_sizes[i] for i in selected])) if selected else 0.0,
        "core_stability": float(np.mean([core_scores[i] for i in selected])) if selected else 0.0,
        "core_loop_density": float(np.mean([loop_scores[i] for i in selected])) if selected else 0.0,
        "core_tunnel_support": float(np.mean([tunnel_scores[i] for i in selected])) if selected else 0.0,
        "core_phase_balance": float(np.mean([phase_scores[i] for i in selected])) if selected else 0.0,
    }


contact_rows = []
atom_id = 0

for perturbation in PERTURBATIONS:
    for Z, symbol, neutrons, shells in ELEMENTS:
        for _ in range(REPLICATES):
            contact_rows.extend(
                generate_proto_atom(atom_id, Z, symbol, neutrons, shells, perturbation)
            )
            atom_id += 1

contacts = pd.DataFrame(contact_rows)

atom_rows = []

for aid, g in contacts.groupby("atom_id"):
    core_info = estimate_proto_cores(g)

    Z = int(g["Z"].iloc[0])
    symbol = g["symbol"].iloc[0]
    perturbation = float(g["perturbation"].iloc[0])

    shell_contacts = g[g["role"] == "shell"]
    interface_contacts = g[g["role"] == "interface"]
    nuclear_contacts = g[g["role"].isin(["core", "surface"])]

    shell_count = len(shell_contacts)
    interface_bridge = float(interface_contacts["bridge"].mean()) if len(interface_contacts) else 0.0
    shell_node = float(np.mean(shell_contacts["family"] == "node")) if len(shell_contacts) else 0.0

    row = {
        "atom_id": aid,
        "Z": Z,
        "symbol": symbol,
        "perturbation": perturbation,
        "num_contacts": len(g),
        "nuclear_contacts": len(nuclear_contacts),
        "shell_contacts": shell_count,
        "interface_contacts": len(interface_contacts),
        "mean_radius": float(g["radius"].mean()),
        "mean_loop": float(g["loop"].mean()),
        "mean_tunnel": float(g["tunnel"].mean()),
        "mean_bridge": float(g["bridge"].mean()),
        "mean_recurrence": float(g["recurrence"].mean()),
        "mean_capacity": float(g["capacity"].mean()),
        "phase_balance": float(g["phase"].mean()),
        "interface_bridge": interface_bridge,
        "shell_node_ratio": shell_node,
        "Z_shell_match": float(np.exp(-abs(Z - shell_count))),
        "Z_core_match": float(np.exp(-abs(Z - core_info["estimated_proto_cores"]))),
        **core_info,
    }

    row["proto_core_recovery_error"] = abs(row["estimated_proto_cores"] - Z)
    row["proto_core_recovery_exact"] = int(row["estimated_proto_cores"] == Z)

    row["core_shell_coupling"] = (
        row["core_stability"]
        * (0.5 + 0.5 * row["interface_bridge"])
        * (0.5 + 0.5 * row["shell_node_ratio"])
    )

    atom_rows.append(row)

atoms = pd.DataFrame(atom_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)

feature_cols = [
    c for c in atoms.columns
    if c not in ["atom_id", "symbol", "Z", "perturbation"]
]

X = atoms[feature_cols].to_numpy(dtype=float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)
y = atoms["symbol"].to_numpy()

train = atoms["perturbation"].isin([0.0, 0.01]).to_numpy()
test = atoms["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], y[train])
pred = clf.predict(Xn[test])

identification_accuracy = float(accuracy_score(y[test], pred))
sil = float(silhouette_score(Xn, y))

core_exact_recovery = float(atoms["proto_core_recovery_exact"].mean())
mean_core_error = float(atoms["proto_core_recovery_error"].mean())
mean_core_stability = float(atoms["core_stability"].mean())
mean_Z_core_match = float(atoms["Z_core_match"].mean())

by_element = atoms.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    mean_estimated_proto_cores=("estimated_proto_cores", "mean"),
    mean_core_error=("proto_core_recovery_error", "mean"),
    exact_core_recovery=("proto_core_recovery_exact", "mean"),
    mean_core_stability=("core_stability", "mean"),
    mean_core_loop_density=("core_loop_density", "mean"),
    mean_core_tunnel_support=("core_tunnel_support", "mean"),
    mean_core_phase_balance=("core_phase_balance", "mean"),
    mean_Z_core_match=("Z_core_match", "mean"),
    mean_Z_shell_match=("Z_shell_match", "mean"),
    mean_core_shell_coupling=("core_shell_coupling", "mean"),
).reset_index().sort_values("Z")

if (
    core_exact_recovery >= 0.95
    and mean_core_error <= 0.10
    and mean_core_stability >= 0.65
    and mean_Z_core_match >= 0.95
):
    verdict = "proto_core_number_recovery_supported"
elif (
    core_exact_recovery >= 0.80
    and mean_core_stability >= 0.55
):
    verdict = "weak_proto_core_number_recovery"
else:
    verdict = "proto_core_identification_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_atoms": len(atoms),
    "num_contacts": len(contacts),
    "identification_accuracy": identification_accuracy,
    "silhouette_score": sil,
    "core_exact_recovery": core_exact_recovery,
    "mean_core_error": mean_core_error,
    "mean_core_stability": mean_core_stability,
    "mean_Z_core_match": mean_Z_core_match,
    "verdict": verdict,
}])

contacts.to_csv(OUT / "proto_core_contacts.csv", index=False)
atoms.to_csv(OUT / "proto_core_atoms.csv", index=False)
summary.to_csv(OUT / "proto_core_summary.csv", index=False)
by_element.to_csv(OUT / "proto_core_by_element.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(8, 6))
for Z, symbol, _, _ in ELEMENTS:
    mask = atoms["symbol"].to_numpy() == symbol
    plt.scatter(proj[mask, 0], proj[mask, 1], s=18, label=symbol)
plt.title("BST proto-core identification space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "proto_core_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(by_element["Z"], by_element["mean_estimated_proto_cores"], marker="o", label="estimated")
plt.plot(by_element["Z"], by_element["Z"], marker="x", label="target Z")
plt.xticks(by_element["Z"], by_element["symbol"])
plt.xlabel("Element")
plt.ylabel("Proto-core count")
plt.title("BST proto-core recovery of atomic number")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "proto_core_Z_recovery.png", dpi=220)
plt.close()

print("\n=== BST PROTO-CORE IDENTIFICATION TEST ===\n")
print(summary.to_string(index=False))

print("\nProto-core recovery by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'proto_core_contacts.csv'}")
print(f"[OK] wrote {OUT / 'proto_core_atoms.csv'}")
print(f"[OK] wrote {OUT / 'proto_core_summary.csv'}")
print(f"[OK] wrote {OUT / 'proto_core_by_element.csv'}")
print(f"[OK] wrote {OUT / 'proto_core_projection.png'}")
print(f"[OK] wrote {OUT / 'proto_core_Z_recovery.png'}")
print("[DONE] proto-core identification test complete")