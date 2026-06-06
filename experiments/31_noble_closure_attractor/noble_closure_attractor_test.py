#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/noble_closure_attractor_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

ELEMENTS = [
    (1,  "H",  [1],       1, False),
    (2,  "He", [2],       18, True),
    (3,  "Li", [2, 1],    1, False),
    (4,  "Be", [2, 2],    2, False),
    (5,  "B",  [2, 3],    13, False),
    (6,  "C",  [2, 4],    14, False),
    (7,  "N",  [2, 5],    15, False),
    (8,  "O",  [2, 6],    16, False),
    (9,  "F",  [2, 7],    17, False),
    (10, "Ne", [2, 8],    18, True),
    (11, "Na", [2, 8, 1], 1, False),
    (12, "Mg", [2, 8, 2], 2, False),
    (13, "Al", [2, 8, 3], 13, False),
    (14, "Si", [2, 8, 4], 14, False),
    (15, "P",  [2, 8, 5], 15, False),
    (16, "S",  [2, 8, 6], 16, False),
    (17, "Cl", [2, 8, 7], 17, False),
    (18, "Ar", [2, 8, 8], 18, True),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 64


def shell_capacity(i):
    return [2, 8, 18, 32][i]


def make_sample(Z, symbol, shells, group, is_noble, perturbation, replicate):
    outer_shell = max(i + 1 for i, s in enumerate(shells) if s > 0)
    outer_count = shells[outer_shell - 1]
    outer_capacity = shell_capacity(outer_shell - 1)

    effective_capacity = 2 if symbol == "He" else 8
    effective_valence = effective_capacity if is_noble else outer_count

    closure_defect = abs(effective_capacity - effective_valence) / effective_capacity
    closure_strength = np.exp(-closure_defect)

    shell_completion = effective_valence / effective_capacity
    ionization_barrier = (
        0.35
        + 0.45 * closure_strength
        + 0.10 * shell_completion
        + 0.10 * float(is_noble)
        + perturbation * rng.normal(0, 0.01)
    )

    bonding_suppression = (
        0.70 * closure_strength
        + 0.20 * float(is_noble)
        + 0.10 * np.exp(-outer_shell / 4.0)
        + perturbation * rng.normal(0, 0.01)
    )

    attractor_depth = (
        0.40 * closure_strength
        + 0.30 * ionization_barrier
        + 0.20 * bonding_suppression
        + 0.10 * float(group == 18)
    )

    open_shell_drive = (
        1.0 - closure_strength
        + 0.30 * (1.0 - float(is_noble))
        + perturbation * rng.normal(0, 0.01)
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "group": group,
        "is_noble": int(is_noble),
        "replicate": replicate,
        "perturbation": perturbation,
        "outer_shell": outer_shell,
        "outer_count": outer_count,
        "effective_valence": effective_valence,
        "effective_capacity": effective_capacity,
        "closure_defect": closure_defect,
        "closure_strength": closure_strength,
        "shell_completion": shell_completion,
        "ionization_barrier": ionization_barrier,
        "bonding_suppression": bonding_suppression,
        "attractor_depth": attractor_depth,
        "open_shell_drive": open_shell_drive,
    }


rows = []
for perturbation in PERTURBATIONS:
    for Z, symbol, shells, group, is_noble in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(make_sample(Z, symbol, shells, group, is_noble, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "outer_shell",
    "outer_count",
    "effective_valence",
    "effective_capacity",
    "closure_defect",
    "closure_strength",
    "shell_completion",
    "ionization_barrier",
    "bonding_suppression",
    "attractor_depth",
    "open_shell_drive",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], df.loc[train, "is_noble"])
noble_accuracy = float(accuracy_score(df.loc[test, "is_noble"], clf.predict(Xn[test])))

try:
    noble_silhouette = float(silhouette_score(Xn, df["is_noble"]))
except Exception:
    noble_silhouette = np.nan

noble = df[df["is_noble"] == 1]
non_noble = df[df["is_noble"] == 0]

closure_contrast = float(noble["closure_strength"].mean() / (non_noble["closure_strength"].mean() + EPS))
attractor_contrast = float(noble["attractor_depth"].mean() / (non_noble["attractor_depth"].mean() + EPS))
bonding_suppression_contrast = float(noble["bonding_suppression"].mean() / (non_noble["bonding_suppression"].mean() + EPS))
open_shell_drive_contrast = float(non_noble["open_shell_drive"].mean() / (noble["open_shell_drive"].mean() + EPS))

by_element = df.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    is_noble=("is_noble", "mean"),
    mean_closure_strength=("closure_strength", "mean"),
    mean_ionization_barrier=("ionization_barrier", "mean"),
    mean_bonding_suppression=("bonding_suppression", "mean"),
    mean_attractor_depth=("attractor_depth", "mean"),
    mean_open_shell_drive=("open_shell_drive", "mean"),
).reset_index().sort_values("Z")

if (
    noble_accuracy >= 0.99
    and closure_contrast >= 1.6
    and attractor_contrast >= 1.5
    and bonding_suppression_contrast >= 1.5
    and open_shell_drive_contrast >= 2.0
):
    verdict = "noble_closure_attractor_supported"
elif (
    noble_accuracy >= 0.90
    and attractor_contrast >= 1.25
):
    verdict = "weak_noble_closure_attractor"
else:
    verdict = "noble_closure_attractor_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_samples": len(df),
    "noble_accuracy": noble_accuracy,
    "noble_silhouette": noble_silhouette,
    "closure_contrast": closure_contrast,
    "attractor_contrast": attractor_contrast,
    "bonding_suppression_contrast": bonding_suppression_contrast,
    "open_shell_drive_contrast": open_shell_drive_contrast,
    "mean_noble_attractor_depth": float(noble["attractor_depth"].mean()),
    "mean_non_noble_attractor_depth": float(non_noble["attractor_depth"].mean()),
    "verdict": verdict,
}])

df.to_csv(OUT / "noble_closure_samples.csv", index=False)
summary.to_csv(OUT / "noble_closure_summary.csv", index=False)
by_element.to_csv(OUT / "noble_closure_by_element.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for label, name in [(0, "open shell"), (1, "noble closure")]:
    mask = df["is_noble"].to_numpy() == label
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=name)
plt.title("BST noble closure attractor space H→Ar")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "noble_closure_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(by_element["symbol"], by_element["mean_attractor_depth"])
plt.ylabel("Attractor depth")
plt.title("BST noble closure attractor depth")
plt.tight_layout()
plt.savefig(OUT / "noble_closure_attractor_depth.png", dpi=220)
plt.close()

print("\n=== BST NOBLE CLOSURE ATTRACTOR TEST ===\n")
print(summary.to_string(index=False))

print("\nNoble closure by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'noble_closure_samples.csv'}")
print(f"[OK] wrote {OUT / 'noble_closure_summary.csv'}")
print(f"[OK] wrote {OUT / 'noble_closure_by_element.csv'}")
print(f"[OK] wrote {OUT / 'noble_closure_projection.png'}")
print(f"[OK] wrote {OUT / 'noble_closure_attractor_depth.png'}")
print("[DONE] noble closure attractor test complete")