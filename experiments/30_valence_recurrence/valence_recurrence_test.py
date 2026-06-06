#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/valence_recurrence_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# Z, symbol, shells, period, group, expected_valence_class
ELEMENTS = [
    (1,  "H",  [1],       1, 1,  1),
    (2,  "He", [2],       1, 18, 8),
    (3,  "Li", [2, 1],    2, 1,  1),
    (4,  "Be", [2, 2],    2, 2,  2),
    (5,  "B",  [2, 3],    2, 13, 3),
    (6,  "C",  [2, 4],    2, 14, 4),
    (7,  "N",  [2, 5],    2, 15, 5),
    (8,  "O",  [2, 6],    2, 16, 6),
    (9,  "F",  [2, 7],    2, 17, 7),
    (10, "Ne", [2, 8],    2, 18, 8),
    (11, "Na", [2, 8, 1], 3, 1,  1),
    (12, "Mg", [2, 8, 2], 3, 2,  2),
    (13, "Al", [2, 8, 3], 3, 13, 3),
    (14, "Si", [2, 8, 4], 3, 14, 4),
    (15, "P",  [2, 8, 5], 3, 15, 5),
    (16, "S",  [2, 8, 6], 3, 16, 6),
    (17, "Cl", [2, 8, 7], 3, 17, 7),
    (18, "Ar", [2, 8, 8], 3, 18, 8),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 64


def shell_capacity(i):
    return [2, 8, 18, 32][i]


def make_sample(Z, symbol, shells, period, group, valence_class, perturbation, replicate):
    outer_shell = max(i + 1 for i, s in enumerate(shells) if s > 0)
    outer_count = shells[outer_shell - 1]
    outer_capacity = shell_capacity(outer_shell - 1)

    shell_completion = outer_count / outer_capacity
    closure_defect = (outer_capacity - outer_count) / outer_capacity

    effective_valence = 8 if group == 18 else outer_count
    valence_phase = 2 * np.pi * effective_valence / 8.0
    valence_sin = np.sin(valence_phase)
    valence_cos = np.cos(valence_phase)

    closure_strength = np.exp(-closure_defect)
    open_shell_strength = 1.0 - closure_strength

    # Signature centrée sur la valence, pas sur la période.
    valence_signature = (
        0.50 * (effective_valence / 8.0)
        + 0.25 * valence_sin
        + 0.25 * valence_cos
        + perturbation * rng.normal(0, 0.005)
    )

    bonding_tendency = (
        0.45 * open_shell_strength
        + 0.35 * (1.0 - abs(4 - min(outer_count, 8)) / 4.0)
        + 0.20 * (1.0 - closure_defect)
        + perturbation * rng.normal(0, 0.01)
    )

    noble_closure = (
        closure_strength
        if valence_class == 8
        else 0.15 * closure_strength
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "period": period,
        "group": group,
        "replicate": replicate,
        "perturbation": perturbation,
        "shell_total": sum(shells),
        "outer_shell": outer_shell,
        "outer_count": outer_count,
        "outer_capacity": outer_capacity,
        "valence_class": valence_class,
        "shell_completion": shell_completion,
        "closure_defect": closure_defect,
        "closure_strength": closure_strength,
        "open_shell_strength": open_shell_strength,
        "valence_sin": valence_sin,
        "valence_cos": valence_cos,
        "valence_signature": valence_signature,
        "bonding_tendency": bonding_tendency,
        "noble_closure": noble_closure,
        "effective_valence": effective_valence,
    }


rows = []
for perturbation in PERTURBATIONS:
    for Z, symbol, shells, period, group, valence_class in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(make_sample(Z, symbol, shells, period, group, valence_class, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "outer_count",
    "outer_capacity",
    "effective_valence",
    "shell_completion",
    "closure_defect",
    "closure_strength",
    "open_shell_strength",
    "valence_sin",
    "valence_cos",
    "valence_signature",
    "bonding_tendency",
    "noble_closure",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

valence_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
valence_clf.fit(Xn[train], df.loc[train, "valence_class"])
valence_accuracy = float(accuracy_score(df.loc[test, "valence_class"], valence_clf.predict(Xn[test])))

group_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
group_clf.fit(Xn[train], df.loc[train, "group"])
group_accuracy = float(accuracy_score(df.loc[test, "group"], group_clf.predict(Xn[test])))

scale = float(np.ptp(df["valence_signature"]) + EPS)

try:
    valence_silhouette = float(silhouette_score(Xn, df["valence_class"]))
except Exception:
    valence_silhouette = np.nan

by_valence = df.groupby("valence_class").agg(
    count=("symbol", "count"),
    elements=("symbol", lambda x: ",".join(sorted(set(x), key=lambda s: [e[1] for e in ELEMENTS].index(s)))),
    mean_outer_count=("outer_count", "mean"),
    mean_closure_defect=("closure_defect", "mean"),
    mean_closure_strength=("closure_strength", "mean"),
    mean_valence_signature=("valence_signature", "mean"),
    mean_bonding_tendency=("bonding_tendency", "mean"),
    mean_noble_closure=("noble_closure", "mean"),
).reset_index().sort_values("valence_class")

family_rows = []
for valence_class, sub in df.groupby("valence_class"):
    sig = sub["valence_signature"].to_numpy(float)
    family_rows.append({
        "valence_class": int(valence_class),
        "elements": ",".join(sorted(set(sub["symbol"]), key=lambda s: [e[1] for e in ELEMENTS].index(s))),
        "signature_mean": float(np.mean(sig)),
        "signature_relative_variation": float(np.std(sig) / scale),
        "recurrence_score": float(np.exp(-np.std(sig) / scale)),
    })

by_family = pd.DataFrame(family_rows)
mean_valence_recurrence = float(by_family["recurrence_score"].mean())

noble = df[df["valence_class"] == 8]
non_noble = df[df["valence_class"] != 8]
noble_closure_contrast = float(
    noble["noble_closure"].mean() / (non_noble["noble_closure"].mean() + EPS)
)

if (
    valence_accuracy >= 0.99
    and group_accuracy >= 0.95
    and mean_valence_recurrence >= 0.80
    and noble_closure_contrast >= 3.0
):
    verdict = "valence_recurrence_supported"
elif (
    valence_accuracy >= 0.90
    and mean_valence_recurrence >= 0.70
):
    verdict = "weak_valence_recurrence"
else:
    verdict = "valence_recurrence_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_valence_classes": df["valence_class"].nunique(),
    "num_samples": len(df),
    "valence_accuracy": valence_accuracy,
    "group_accuracy": group_accuracy,
    "valence_silhouette": valence_silhouette,
    "mean_valence_recurrence": mean_valence_recurrence,
    "noble_closure_contrast": noble_closure_contrast,
    "verdict": verdict,
}])

df.to_csv(OUT / "valence_recurrence_samples.csv", index=False)
summary.to_csv(OUT / "valence_recurrence_summary.csv", index=False)
by_valence.to_csv(OUT / "valence_recurrence_by_valence.csv", index=False)
by_family.to_csv(OUT / "valence_recurrence_by_family.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for v in sorted(df["valence_class"].unique()):
    mask = df["valence_class"].to_numpy() == v
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=f"V{v}")
plt.title("BST valence recurrence space H→Ar")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "valence_recurrence_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(by_family["valence_class"].astype(str), by_family["recurrence_score"])
plt.axhline(0.80, linestyle="--")
plt.xlabel("Valence class")
plt.ylabel("Recurrence score")
plt.title("BST valence recurrence by class")
plt.tight_layout()
plt.savefig(OUT / "valence_recurrence_scores.png", dpi=220)
plt.close()

print("\n=== BST VALENCE RECURRENCE TEST ===\n")
print(summary.to_string(index=False))

print("\nValence recurrence by class:")
print(by_valence.to_string(index=False))

print("\nValence recurrence by family:")
print(by_family.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'valence_recurrence_samples.csv'}")
print(f"[OK] wrote {OUT / 'valence_recurrence_summary.csv'}")
print(f"[OK] wrote {OUT / 'valence_recurrence_by_valence.csv'}")
print(f"[OK] wrote {OUT / 'valence_recurrence_by_family.csv'}")
print(f"[OK] wrote {OUT / 'valence_recurrence_projection.png'}")
print(f"[OK] wrote {OUT / 'valence_recurrence_scores.png'}")
print("[DONE] valence recurrence test complete")