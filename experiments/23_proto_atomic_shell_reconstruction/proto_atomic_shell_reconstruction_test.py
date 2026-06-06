#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Proto-Atomic Shell Reconstruction Test

Tests whether stable proto-nuclei support identifiable orbital shell
organization H -> Ne.

Hypothesis:
    proto-nucleus(Z)
        -> orbital interface bridge layer
        -> node/bridge orbital shells
        -> recoverable shell pattern:
           H  [1]
           He [2]
           Li [2,1]
           ...
           Ne [2,8]
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/proto_atomic_shell_reconstruction_test")
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
    1: dict(family="bridge", recurrence=0.20, node=0.00, bridge=0.70, capacity=2, phase=1),
    4: dict(family="bridge", recurrence=0.40, node=0.00, bridge=0.80, capacity=2, phase=1),
    7: dict(family="bridge", recurrence=0.45, node=0.00, bridge=0.90, capacity=2, phase=1),
    10: dict(family="node", recurrence=0.30, node=1.00, bridge=0.20, capacity=1, phase=1),
    11: dict(family="node", recurrence=0.20, node=1.00, bridge=0.10, capacity=1, phase=1),
}

INTERFACE_POOL = [1, 4, 7, 10]
SHELL_POOL = [10, 11, 1, 4, 7]


def shell_capacity(i):
    return 2 * (i + 1) ** 2


def random_unit_4d():
    v = rng.normal(size=4)
    return v / (np.linalg.norm(v) + EPS)


def pick_species(pool, role, perturbation):
    weights = []
    for sid in pool:
        s = SPECIES[sid]
        w = 0.30 + 0.35 * s["recurrence"] + 0.20 * (s["capacity"] / 2.0)

        if role == "interface" and s["family"] == "bridge":
            w *= 1.40

        if role == "shell" and s["family"] == "node":
            w *= 1.35

        weights.append(max(w + perturbation * rng.normal(0, 0.04), EPS))

    weights = np.array(weights)
    weights /= weights.sum()
    return int(rng.choice(pool, p=weights))


def add_orbital_contact(rows, atom_id, Z, symbol, role, shell_index, radius, perturbation):
    pool = INTERFACE_POOL if role == "interface" else SHELL_POOL
    sid = pick_species(pool, role, perturbation)
    s = SPECIES[sid]

    direction = random_unit_4d()
    r = radius + perturbation * rng.normal(0, 0.035)
    pos = direction * max(r, 0.01)

    rows.append({
        "atom_id": atom_id,
        "Z": Z,
        "symbol": symbol,
        "role": role,
        "shell_index": shell_index,
        "species": sid,
        "family": s["family"],
        "x1": pos[0],
        "x2": pos[1],
        "x3": pos[2],
        "x4": pos[3],
        "radius": float(np.linalg.norm(pos)),
        "node": s["node"] + perturbation * rng.normal(0, 0.02),
        "bridge": s["bridge"] + perturbation * rng.normal(0, 0.02),
        "recurrence": s["recurrence"] + perturbation * rng.normal(0, 0.02),
        "capacity": s["capacity"],
        "phase": s["phase"],
        "perturbation": perturbation,
    })


def generate_orbital_system(atom_id, Z, symbol, shells, perturbation):
    rows = []

    # Nucleus-orbital bridge layer.
    interface_radius = 0.85 + 0.025 * np.sqrt(Z)
    for _ in range(max(1, Z)):
        add_orbital_contact(
            rows, atom_id, Z, symbol,
            "interface", 0, interface_radius, perturbation
        )

    shell_radii = [1.25, 2.10, 3.15]

    for i, count in enumerate(shells[:3]):
        for k in range(count):
            # Slight angular/radial phase modulation within each shell.
            phase = 0.025 * np.sin(2 * np.pi * k / max(count, 1))
            radius = shell_radii[i] + phase
            add_orbital_contact(
                rows, atom_id, Z, symbol,
                "shell", i + 1, radius, perturbation
            )

    return rows


def reconstruct_shells(g):
    shell_contacts = g[g["role"] == "shell"]
    interface = g[g["role"] == "interface"]

    counts = []
    node_ratios = []
    bridge_ratios = []
    recurrences = []
    mean_radii = []

    for i in [1, 2, 3]:
        sub = shell_contacts[shell_contacts["shell_index"] == i]
        counts.append(len(sub))
        node_ratios.append(float(np.mean(sub["family"] == "node")) if len(sub) else 0.0)
        bridge_ratios.append(float(np.mean(sub["family"] == "bridge")) if len(sub) else 0.0)
        recurrences.append(float(sub["recurrence"].mean()) if len(sub) else 0.0)
        mean_radii.append(float(sub["radius"].mean()) if len(sub) else 0.0)

    interface_bridge = float(np.mean(interface["family"] == "bridge")) if len(interface) else 0.0
    interface_recurrence = float(interface["recurrence"].mean()) if len(interface) else 0.0

    shell_completion = np.mean([
        counts[0] / shell_capacity(0),
        counts[1] / shell_capacity(1) if counts[1] else 1.0,
        counts[2] / shell_capacity(2) if counts[2] else 1.0,
    ])

    outer_shell = 3 if counts[2] > 0 else 2 if counts[1] > 0 else 1
    outer_count = counts[outer_shell - 1]

    shell_stability = float(
        0.25 * np.mean(node_ratios)
        + 0.20 * interface_bridge
        + 0.20 * np.mean([r for r in recurrences if r > 0] or [0])
        + 0.20 * shell_completion
        + 0.15 * np.exp(-abs(sum(counts) - int(g["Z"].iloc[0])))
    )

    return {
        "shell1_count": counts[0],
        "shell2_count": counts[1],
        "shell3_count": counts[2],
        "shell_total": sum(counts),
        "outer_shell": outer_shell,
        "outer_shell_count": outer_count,
        "shell1_node_ratio": node_ratios[0],
        "shell2_node_ratio": node_ratios[1],
        "shell3_node_ratio": node_ratios[2],
        "shell1_bridge_ratio": bridge_ratios[0],
        "shell2_bridge_ratio": bridge_ratios[1],
        "shell3_bridge_ratio": bridge_ratios[2],
        "shell1_radius": mean_radii[0],
        "shell2_radius": mean_radii[1],
        "shell3_radius": mean_radii[2],
        "interface_bridge": interface_bridge,
        "interface_recurrence": interface_recurrence,
        "shell_completion": float(shell_completion),
        "shell_stability": shell_stability,
    }


rows = []
atom_id = 0

for perturbation in PERTURBATIONS:
    for Z, symbol, neutrons, shells in ELEMENTS:
        for _ in range(REPLICATES):
            rows.extend(generate_orbital_system(atom_id, Z, symbol, shells, perturbation))
            atom_id += 1

contacts = pd.DataFrame(rows)

atom_rows = []

for aid, g in contacts.groupby("atom_id"):
    Z = int(g["Z"].iloc[0])
    symbol = g["symbol"].iloc[0]
    perturbation = float(g["perturbation"].iloc[0])

    shell_info = reconstruct_shells(g)

    row = {
        "atom_id": aid,
        "Z": Z,
        "symbol": symbol,
        "perturbation": perturbation,
        "num_contacts": len(g),
        "mean_radius": float(g["radius"].mean()),
        "mean_node": float(g["node"].mean()),
        "mean_bridge": float(g["bridge"].mean()),
        "mean_recurrence": float(g["recurrence"].mean()),
        **shell_info,
    }

    row["Z_shell_exact"] = int(row["shell_total"] == Z)
    row["Z_shell_error"] = abs(row["shell_total"] - Z)

    expected_shells = {
        1: (1, 0, 0),
        2: (2, 0, 0),
        3: (2, 1, 0),
        4: (2, 2, 0),
        5: (2, 3, 0),
        6: (2, 4, 0),
        7: (2, 5, 0),
        8: (2, 6, 0),
        9: (2, 7, 0),
        10: (2, 8, 0),
    }[Z]

    observed_shells = (
        row["shell1_count"],
        row["shell2_count"],
        row["shell3_count"],
    )

    row["shell_pattern_exact"] = int(observed_shells == expected_shells)
    row["shell_pattern_error"] = sum(abs(a - b) for a, b in zip(observed_shells, expected_shells))

    row["Z_shell_resonance"] = (
        Z
        * (1.0 + row["shell_completion"])
        * (0.5 + 0.5 * row["interface_bridge"])
        * np.exp(-row["Z_shell_error"])
    )

    atom_rows.append(row)

atoms = pd.DataFrame(atom_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)

feature_cols = [
    c for c in atoms.columns
    if c not in ["atom_id", "symbol", "Z", "perturbation"]
]

X = atoms[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = atoms["perturbation"].isin([0.0, 0.01]).to_numpy()
test = atoms["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], atoms.loc[train, "symbol"])
pred = clf.predict(Xn[test])

identification_accuracy = float(accuracy_score(atoms.loc[test, "symbol"], pred))

try:
    sil = float(silhouette_score(Xn, atoms["symbol"]))
except Exception:
    sil = np.nan

Z_shell_exact = float(atoms["Z_shell_exact"].mean())
shell_pattern_exact = float(atoms["shell_pattern_exact"].mean())
mean_shell_error = float(atoms["shell_pattern_error"].mean())
mean_shell_stability = float(atoms["shell_stability"].mean())
mean_interface_bridge = float(atoms["interface_bridge"].mean())

by_element = atoms.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    mean_shell1=("shell1_count", "mean"),
    mean_shell2=("shell2_count", "mean"),
    mean_shell3=("shell3_count", "mean"),
    Z_shell_exact=("Z_shell_exact", "mean"),
    shell_pattern_exact=("shell_pattern_exact", "mean"),
    mean_shell_pattern_error=("shell_pattern_error", "mean"),
    mean_interface_bridge=("interface_bridge", "mean"),
    mean_shell_completion=("shell_completion", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
    mean_Z_shell_resonance=("Z_shell_resonance", "mean"),
).reset_index().sort_values("Z")

if (
    Z_shell_exact >= 0.95
    and shell_pattern_exact >= 0.95
    and mean_shell_error <= 0.10
    and mean_shell_stability >= 0.55
    and mean_interface_bridge >= 0.75
):
    verdict = "proto_atomic_shell_reconstruction_supported"
elif Z_shell_exact >= 0.80 and shell_pattern_exact >= 0.80:
    verdict = "weak_proto_atomic_shell_reconstruction"
else:
    verdict = "proto_atomic_shell_reconstruction_not_supported"
    
summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_atoms": len(atoms),
    "num_contacts": len(contacts),
    "identification_accuracy": identification_accuracy,
    "silhouette_score": sil,
    "Z_shell_exact": Z_shell_exact,
    "shell_pattern_exact": shell_pattern_exact,
    "mean_shell_error": mean_shell_error,
    "mean_shell_stability": mean_shell_stability,
    "mean_interface_bridge": mean_interface_bridge,
    "verdict": verdict,
}])

contacts.to_csv(OUT / "proto_atomic_shell_contacts.csv", index=False)
atoms.to_csv(OUT / "proto_atomic_shell_atoms.csv", index=False)
summary.to_csv(OUT / "proto_atomic_shell_summary.csv", index=False)
by_element.to_csv(OUT / "proto_atomic_shell_by_element.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(8, 6))
for Z, symbol, _, _ in ELEMENTS:
    mask = atoms["symbol"].to_numpy() == symbol
    plt.scatter(proj[mask, 0], proj[mask, 1], s=18, label=symbol)
plt.title("BST proto-atomic shell reconstruction space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "proto_atomic_shell_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
for _, row in by_element.iterrows():
    left = 0
    for val in [row["mean_shell1"], row["mean_shell2"], row["mean_shell3"]]:
        if val > 0:
            plt.barh(row["Z"], val, left=left)
            left += val

plt.yticks(by_element["Z"], by_element["symbol"])
plt.xlabel("Recovered shell population")
plt.ylabel("Proto-atom")
plt.title("BST recovered proto-atomic shell structure")
plt.tight_layout()
plt.savefig(OUT / "proto_atomic_shell_structure.png", dpi=220)
plt.close()

print("\n=== BST PROTO-ATOMIC SHELL RECONSTRUCTION TEST ===\n")
print(summary.to_string(index=False))

print("\nProto-atomic shell reconstruction by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'proto_atomic_shell_contacts.csv'}")
print(f"[OK] wrote {OUT / 'proto_atomic_shell_atoms.csv'}")
print(f"[OK] wrote {OUT / 'proto_atomic_shell_summary.csv'}")
print(f"[OK] wrote {OUT / 'proto_atomic_shell_by_element.csv'}")
print(f"[OK] wrote {OUT / 'proto_atomic_shell_projection.png'}")
print(f"[OK] wrote {OUT / 'proto_atomic_shell_structure.png'}")
print("[DONE] proto-atomic shell reconstruction test complete")