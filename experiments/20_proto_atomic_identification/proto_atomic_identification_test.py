#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

OUT = Path("results/research_final/proto_atomic_identification_test")
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
    0: {"family": "node", "periodicity": 0.00, "recurrence": 0.00, "loop": 0.00, "tunnel": 0.00, "capacity": 1, "phase": 1},
    1: {"family": "bridge", "periodicity": 0.25, "recurrence": 0.20, "loop": 0.30, "tunnel": 0.30, "capacity": 2, "phase": 1},
    2: {"family": "loop", "periodicity": 0.75, "recurrence": 0.80, "loop": 1.00, "tunnel": 0.50, "capacity": 4, "phase": 1},
    3: {"family": "tunnel", "periodicity": 0.50, "recurrence": 0.70, "loop": 0.20, "tunnel": 1.00, "capacity": 5, "phase": -1},
    4: {"family": "bridge", "periodicity": 0.40, "recurrence": 0.40, "loop": 0.40, "tunnel": 0.40, "capacity": 2, "phase": 1},
    5: {"family": "loop", "periodicity": 0.80, "recurrence": 0.85, "loop": 1.00, "tunnel": 0.50, "capacity": 4, "phase": 1},
    6: {"family": "tunnel", "periodicity": 0.60, "recurrence": 0.75, "loop": 0.20, "tunnel": 1.00, "capacity": 5, "phase": -1},
    7: {"family": "bridge", "periodicity": 0.45, "recurrence": 0.45, "loop": 0.50, "tunnel": 0.50, "capacity": 2, "phase": 1},
    8: {"family": "loop", "periodicity": 0.85, "recurrence": 0.90, "loop": 1.00, "tunnel": 0.50, "capacity": 4, "phase": 1},
    9: {"family": "tunnel", "periodicity": 0.65, "recurrence": 0.80, "loop": 0.20, "tunnel": 1.00, "capacity": 5, "phase": -1},
    10: {"family": "node", "periodicity": 0.30, "recurrence": 0.30, "loop": 0.30, "tunnel": 0.10, "capacity": 1, "phase": 1},
    11: {"family": "node", "periodicity": 0.20, "recurrence": 0.20, "loop": 0.20, "tunnel": 0.10, "capacity": 1, "phase": 1},
}

PROTO_CORE_TEMPLATES = {
    "proton_like":  [2, 5, 8, 3],
    "neutron_like": [5, 8, 6, 9],
    "electron_like": [10, 11, 1, 4],
}


def species_vector(sid):
    s = SPECIES[sid]
    return np.array([
        1.0 if s["family"] == "node" else 0.0,
        1.0 if s["family"] == "bridge" else 0.0,
        1.0 if s["family"] == "loop" else 0.0,
        1.0 if s["family"] == "tunnel" else 0.0,
        s["periodicity"],
        s["recurrence"],
        s["loop"],
        s["tunnel"],
        s["capacity"] / 5.0,
        s["phase"],
    ], dtype=float)


def proto_core(kind, perturbation):
    template = PROTO_CORE_TEMPLATES[kind]
    vectors = np.array([species_vector(s) for s in template])
    mean = vectors.mean(axis=0)

    if kind == "proton_like":
        charge = 1.0
        mass = 1.0
        radius = 0.28
        localization = 0.92
    elif kind == "neutron_like":
        charge = 0.0
        mass = 1.08
        radius = 0.31
        localization = 0.90
    else:
        charge = -1.0
        mass = 0.055
        radius = 0.75
        localization = 0.35

    return {
        "charge": charge + perturbation * rng.normal(0, 0.05),
        "mass": mass + perturbation * rng.normal(0, 0.03),
        "radius": radius + perturbation * rng.normal(0, 0.02),
        "localization": localization + perturbation * rng.normal(0, 0.03),
        "node": mean[0] + perturbation * rng.normal(0, 0.02),
        "bridge": mean[1] + perturbation * rng.normal(0, 0.02),
        "loop": mean[6] + perturbation * rng.normal(0, 0.02),
        "tunnel": mean[7] + perturbation * rng.normal(0, 0.02),
        "periodicity": mean[4] + perturbation * rng.normal(0, 0.02),
        "recurrence": mean[5] + perturbation * rng.normal(0, 0.02),
        "capacity": mean[8] + perturbation * rng.normal(0, 0.02),
        "phase": mean[9] + perturbation * rng.normal(0, 0.02),
        "closure": 1.0 / (1.0 + np.mean(np.linalg.norm(vectors - mean, axis=1))),
    }


def generate_atom(Z, symbol, N, shells, perturbation, replicate):
    protons = [proto_core("proton_like", perturbation) for _ in range(Z)]
    neutrons = [proto_core("neutron_like", perturbation) for _ in range(N)]
    electrons = [proto_core("electron_like", perturbation) for _ in range(sum(shells))]

    nucleus = protons + neutrons

    nuclear_charge = sum(c["charge"] for c in protons)
    electronic_charge = sum(c["charge"] for c in electrons)
    net_charge = nuclear_charge + electronic_charge

    nuclear_mass = sum(c["mass"] for c in nucleus)
    orbital_mass = sum(c["mass"] for c in electrons)

    nuclear_loop = np.mean([c["loop"] for c in nucleus])
    nuclear_tunnel = np.mean([c["tunnel"] for c in nucleus])
    nuclear_recurrence = np.mean([c["recurrence"] for c in nucleus])
    nuclear_closure = np.mean([c["closure"] for c in nucleus])

    orbital_bridge = np.mean([c["bridge"] for c in electrons])
    orbital_node = np.mean([c["node"] for c in electrons])
    orbital_recurrence = np.mean([c["recurrence"] for c in electrons])
    orbital_radius = np.mean([c["radius"] for c in electrons])

    shell_1 = shells[0] if len(shells) > 0 else 0
    shell_2 = shells[1] if len(shells) > 1 else 0
    shell_3 = shells[2] if len(shells) > 2 else 0

    shell_completion = np.mean([
        shell_1 / 2.0,
        shell_2 / 8.0 if len(shells) > 1 else 1.0,
        shell_3 / 18.0 if len(shells) > 2 else 1.0,
    ])

    nucleus_orbital_coupling = np.exp(-abs(nuclear_recurrence - orbital_recurrence)) * (0.5 + 0.5 * orbital_bridge)

    proto_atomic_stability = (
        0.22 * nuclear_recurrence
        + 0.18 * nuclear_closure
        + 0.18 * shell_completion
        + 0.17 * nucleus_orbital_coupling
        + 0.15 * np.exp(-abs(net_charge))
        + 0.10 * nuclear_loop
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "neutrons": N,
        "replicate": replicate,
        "perturbation": perturbation,
        "electron_count": sum(shells),
        "shell_1": shell_1,
        "shell_2": shell_2,
        "shell_3": shell_3,
        "nuclear_charge": nuclear_charge,
        "electronic_charge": electronic_charge,
        "net_charge": net_charge,
        "nuclear_mass": nuclear_mass,
        "orbital_mass": orbital_mass,
        "nuclear_loop": nuclear_loop,
        "nuclear_tunnel": nuclear_tunnel,
        "nuclear_recurrence": nuclear_recurrence,
        "nuclear_closure": nuclear_closure,
        "orbital_bridge": orbital_bridge,
        "orbital_node": orbital_node,
        "orbital_recurrence": orbital_recurrence,
        "orbital_radius": orbital_radius,
        "shell_completion": shell_completion,
        "nucleus_orbital_coupling": nucleus_orbital_coupling,
        "proto_atomic_stability": proto_atomic_stability,
    }


rows = []

for perturbation in PERTURBATIONS:
    for Z, symbol, N, shells in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(generate_atom(Z, symbol, N, shells, perturbation, r))

atoms = pd.DataFrame(rows)

feature_cols = [
    "nuclear_charge",
    "electron_count",
    "shell_1",
    "shell_2",
    "shell_3",
    "net_charge",
    "nuclear_mass",
    "orbital_mass",
    "nuclear_loop",
    "nuclear_tunnel",
    "nuclear_recurrence",
    "nuclear_closure",
    "orbital_bridge",
    "orbital_node",
    "orbital_recurrence",
    "orbital_radius",
    "shell_completion",
    "nucleus_orbital_coupling",
    "proto_atomic_stability",
]

X = atoms[feature_cols].to_numpy(dtype=float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)
y = atoms["symbol"].to_numpy()

train = atoms["perturbation"].isin([0.0, 0.01]).to_numpy()
test = atoms["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], y[train])
pred = clf.predict(Xn[test])

identification_accuracy = accuracy_score(y[test], pred)
sil = silhouette_score(Xn, y)

stable_ratio = float(np.mean(atoms["proto_atomic_stability"] >= 0.64))
neutral_ratio = float(np.mean(np.abs(atoms["net_charge"]) <= 0.25))

mean_proto_atomic_stability = float(atoms["proto_atomic_stability"].mean())
mean_nucleus_orbital_coupling = float(atoms["nucleus_orbital_coupling"].mean())

by_element = atoms.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    mean_net_charge=("net_charge", "mean"),
    mean_nuclear_mass=("nuclear_mass", "mean"),
    mean_shell_completion=("shell_completion", "mean"),
    mean_nucleus_orbital_coupling=("nucleus_orbital_coupling", "mean"),
    mean_proto_atomic_stability=("proto_atomic_stability", "mean"),
).reset_index().sort_values("Z")

if (
    identification_accuracy >= 0.90
    and stable_ratio >= 0.95
    and neutral_ratio >= 0.95
    and mean_proto_atomic_stability >= 0.65
    and mean_nucleus_orbital_coupling >= 0.40
):
    verdict = "proto_atomic_identification_supported"
elif (
    identification_accuracy >= 0.80
    and stable_ratio >= 0.80
):
    verdict = "weak_proto_atomic_identification"
else:
    verdict = "proto_atomic_identification_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_candidates": len(atoms),
    "identification_accuracy": identification_accuracy,
    "silhouette_score": sil,
    "stable_ratio": stable_ratio,
    "neutral_ratio": neutral_ratio,
    "mean_proto_atomic_stability": mean_proto_atomic_stability,
    "mean_nucleus_orbital_coupling": mean_nucleus_orbital_coupling,
    "verdict": verdict,
}])

atoms.to_csv(OUT / "proto_atomic_candidates.csv", index=False)
summary.to_csv(OUT / "proto_atomic_summary.csv", index=False)
by_element.to_csv(OUT / "proto_atomic_by_element.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(8, 6))
for Z, symbol, _, _ in ELEMENTS:
    mask = atoms["symbol"].to_numpy() == symbol
    plt.scatter(proj[mask, 0], proj[mask, 1], s=18, label=symbol)

plt.title("BST proto-atomic identification space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "proto_atomic_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
for Z, symbol, _, shells in ELEMENTS:
    left = 0
    for count in shells:
        plt.barh(Z, count, left=left)
        left += count

plt.yticks([e[0] for e in ELEMENTS], [e[1] for e in ELEMENTS])
plt.xlabel("Orbital proto-core count")
plt.ylabel("Identified proto-atom")
plt.title("BST proto-atomic shell organization")
plt.tight_layout()
plt.savefig(OUT / "proto_atomic_shells.png", dpi=220)
plt.close()

print("\n=== BST PROTO-ATOMIC IDENTIFICATION TEST ===\n")
print(summary.to_string(index=False))

print("\nProto-atomic element summary:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'proto_atomic_candidates.csv'}")
print(f"[OK] wrote {OUT / 'proto_atomic_summary.csv'}")
print(f"[OK] wrote {OUT / 'proto_atomic_by_element.csv'}")
print(f"[OK] wrote {OUT / 'proto_atomic_projection.png'}")
print(f"[OK] wrote {OUT / 'proto_atomic_shells.png'}")
print("[DONE] proto-atomic identification test complete")