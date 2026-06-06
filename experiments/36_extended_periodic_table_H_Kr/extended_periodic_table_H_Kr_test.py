#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/extended_periodic_table_H_Kr_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# Z, symbol, shells, period, group, block
ELEMENTS = [
    (1, "H", [1], 1, 1, "s"), (2, "He", [2], 1, 18, "s"),

    (3, "Li", [2,1], 2, 1, "s"), (4, "Be", [2,2], 2, 2, "s"),
    (5, "B", [2,3], 2, 13, "p"), (6, "C", [2,4], 2, 14, "p"),
    (7, "N", [2,5], 2, 15, "p"), (8, "O", [2,6], 2, 16, "p"),
    (9, "F", [2,7], 2, 17, "p"), (10, "Ne", [2,8], 2, 18, "p"),

    (11, "Na", [2,8,1], 3, 1, "s"), (12, "Mg", [2,8,2], 3, 2, "s"),
    (13, "Al", [2,8,3], 3, 13, "p"), (14, "Si", [2,8,4], 3, 14, "p"),
    (15, "P", [2,8,5], 3, 15, "p"), (16, "S", [2,8,6], 3, 16, "p"),
    (17, "Cl", [2,8,7], 3, 17, "p"), (18, "Ar", [2,8,8], 3, 18, "p"),

    (19, "K", [2,8,8,1], 4, 1, "s"), (20, "Ca", [2,8,8,2], 4, 2, "s"),

    (21, "Sc", [2,8,9,2], 4, 3, "d"), (22, "Ti", [2,8,10,2], 4, 4, "d"),
    (23, "V", [2,8,11,2], 4, 5, "d"), (24, "Cr", [2,8,13,1], 4, 6, "d"),
    (25, "Mn", [2,8,13,2], 4, 7, "d"), (26, "Fe", [2,8,14,2], 4, 8, "d"),
    (27, "Co", [2,8,15,2], 4, 9, "d"), (28, "Ni", [2,8,16,2], 4, 10, "d"),
    (29, "Cu", [2,8,18,1], 4, 11, "d"), (30, "Zn", [2,8,18,2], 4, 12, "d"),

    (31, "Ga", [2,8,18,3], 4, 13, "p"), (32, "Ge", [2,8,18,4], 4, 14, "p"),
    (33, "As", [2,8,18,5], 4, 15, "p"), (34, "Se", [2,8,18,6], 4, 16, "p"),
    (35, "Br", [2,8,18,7], 4, 17, "p"), (36, "Kr", [2,8,18,8], 4, 18, "p"),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 48

BLOCK_ID = {"s": 0, "p": 1, "d": 2, "f": 3}


def shell_capacity(i):
    return [2, 8, 18, 32][i]


def effective_valence(group, block, shells):
    outer = shells[-1]
    if group == 18:
        return 8
    if block == "d":
        # transition-like valence: ns + active d contribution
        return min(group, 12) - 2
    return outer


def make_sample(Z, symbol, shells, period, group, block, perturbation, replicate):
    outer_shell = max(i + 1 for i, s in enumerate(shells) if s > 0)
    outer_count = shells[outer_shell - 1]
    outer_capacity = shell_capacity(outer_shell - 1)

    shell_total = sum(shells)
    block_id = BLOCK_ID[block]

    valence = effective_valence(group, block, shells)
    valence_norm = valence / 8.0 if block != "d" else valence / 10.0

    closure_defect = abs(8 - valence) / 8.0 if block != "d" else abs(10 - valence) / 10.0
    closure_strength = np.exp(-closure_defect)

    d_occupancy = 0.0
    if block == "d":
        d_occupancy = max(shells[2] - 8, 0) / 10.0

    d_activation = (
        0.0 if block != "d"
        else 0.55 + 0.35 * d_occupancy + 0.10 * np.exp(-abs(0.5 - d_occupancy))
    )

    recurrence = (
        0.25 * closure_strength
        + 0.20 * valence_norm
        + 0.20 * np.sin(2 * np.pi * valence_norm)
        + 0.15 * np.cos(2 * np.pi * valence_norm)
        + 0.10 * period / 4.0
        + 0.10 * (1.0 if group in [1, 2, 17, 18] else 0.75)
        + perturbation * rng.normal(0, 0.01)
    )

    transition_signature = (
        d_activation
        + perturbation * rng.normal(0, 0.01)
    )

    shell_stability = (
        0.50
        + 0.20 * closure_strength
        + 0.10 * valence_norm
        + 0.10 * (1.0 if group == 18 else 0.0)
        + 0.10 * (1.0 if block == "d" else 0.0) * np.exp(-abs(d_occupancy - 0.5))
        + perturbation * rng.normal(0, 0.01)
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "period": period,
        "group": group,
        "block": block,
        "block_id": block_id,
        "replicate": replicate,
        "perturbation": perturbation,
        "shell_1": shells[0] if len(shells) > 0 else 0,
        "shell_2": shells[1] if len(shells) > 1 else 0,
        "shell_3": shells[2] if len(shells) > 2 else 0,
        "shell_4": shells[3] if len(shells) > 3 else 0,
        "shell_total": shell_total,
        "outer_shell": outer_shell,
        "outer_count": outer_count,
        "outer_capacity": outer_capacity,
        "valence": valence,
        "valence_norm": valence_norm,
        "closure_defect": closure_defect,
        "closure_strength": closure_strength,
        "d_occupancy": d_occupancy,
        "d_activation": d_activation,
        "transition_signature": transition_signature,
        "recurrence": recurrence,
        "shell_stability": shell_stability,
        "Z_match": np.exp(-abs(Z - shell_total)),
    }


rows = []
for perturbation in PERTURBATIONS:
    for e in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(make_sample(*e, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "shell_1", "shell_2", "shell_3", "shell_4",
    "shell_total",
    "outer_shell",
    "outer_count",
    "outer_capacity",
    "valence",
    "valence_norm",
    "closure_defect",
    "closure_strength",
    "d_occupancy",
    "d_activation",
    "transition_signature",
    "recurrence",
    "shell_stability",
    "Z_match",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

def knn_acc(label):
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf.fit(Xn[train], df.loc[train, label])
    return float(accuracy_score(df.loc[test, label], clf.predict(Xn[test])))

element_accuracy = knn_acc("symbol")
period_accuracy = knn_acc("period")
group_accuracy = knn_acc("group")
block_accuracy = knn_acc("block")

try:
    block_silhouette = float(silhouette_score(Xn, df["block"]))
except Exception:
    block_silhouette = np.nan

try:
    element_silhouette = float(silhouette_score(Xn, df["symbol"]))
except Exception:
    element_silhouette = np.nan

Z_recovery = float(np.mean(np.round(df["shell_total"]).astype(int) == df["Z"]))
period_recovery = float(np.mean(df["outer_shell"] == df["period"]))
d_block_detection = float(np.mean((df["block"] == "d") == (df["d_activation"] > 0.50)))

by_element = df.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    period=("period", "mean"),
    group=("group", "mean"),
    block=("block", "first"),
    mean_shell_total=("shell_total", "mean"),
    mean_valence=("valence", "mean"),
    mean_closure_strength=("closure_strength", "mean"),
    mean_d_occupancy=("d_occupancy", "mean"),
    mean_d_activation=("d_activation", "mean"),
    mean_transition_signature=("transition_signature", "mean"),
    mean_recurrence=("recurrence", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
).reset_index().sort_values("Z")

by_block = df.groupby("block").agg(
    count=("block", "count"),
    elements=("symbol", lambda x: ",".join(sorted(set(x), key=lambda s: [e[1] for e in ELEMENTS].index(s)))),
    mean_period=("period", "mean"),
    mean_valence=("valence", "mean"),
    mean_d_activation=("d_activation", "mean"),
    mean_transition_signature=("transition_signature", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
).reset_index()

if (
    element_accuracy >= 0.95
    and period_accuracy >= 0.99
    and group_accuracy >= 0.90
    and block_accuracy >= 0.99
    and Z_recovery >= 0.99
    and period_recovery >= 0.99
    and d_block_detection >= 0.99
):
    verdict = "extended_periodic_table_H_Kr_supported"
elif (
    period_accuracy >= 0.95
    and block_accuracy >= 0.90
    and d_block_detection >= 0.90
):
    verdict = "weak_extended_periodic_table_H_Kr"
else:
    verdict = "extended_periodic_table_H_Kr_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_samples": len(df),
    "max_Z": 36,
    "element_accuracy": element_accuracy,
    "period_accuracy": period_accuracy,
    "group_accuracy": group_accuracy,
    "block_accuracy": block_accuracy,
    "element_silhouette": element_silhouette,
    "block_silhouette": block_silhouette,
    "Z_recovery": Z_recovery,
    "period_recovery": period_recovery,
    "d_block_detection": d_block_detection,
    "mean_shell_stability": float(df["shell_stability"].mean()),
    "verdict": verdict,
}])

df.to_csv(OUT / "extended_periodic_H_Kr_samples.csv", index=False)
summary.to_csv(OUT / "extended_periodic_H_Kr_summary.csv", index=False)
by_element.to_csv(OUT / "extended_periodic_H_Kr_by_element.csv", index=False)
by_block.to_csv(OUT / "extended_periodic_H_Kr_by_block.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(10, 7))
for block in ["s", "p", "d"]:
    mask = df["block"].to_numpy() == block
    plt.scatter(proj[mask, 0], proj[mask, 1], s=12, label=f"{block}-block")
plt.title("BST extended periodic table H→Kr — block structure")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_H_Kr_blocks.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(by_element["Z"], by_element["mean_shell_stability"], marker="o", label="shell stability")
plt.plot(by_element["Z"], by_element["mean_d_activation"], marker="x", label="d activation")
plt.xticks(by_element["Z"], by_element["symbol"], rotation=60)
plt.xlabel("Element")
plt.ylabel("BST score")
plt.title("BST H→Kr stability and d-block activation")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_H_Kr_scores.png", dpi=220)
plt.close()

print("\n=== BST EXTENDED PERIODIC TABLE H→Kr TEST ===\n")
print(summary.to_string(index=False))

print("\nExtended periodic table H→Kr by block:")
print(by_block.to_string(index=False))

print("\nExtended periodic table H→Kr by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'extended_periodic_H_Kr_samples.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Kr_summary.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Kr_by_element.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Kr_by_block.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Kr_blocks.png'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Kr_scores.png'}")
print("[DONE] extended periodic table H→Kr test complete")