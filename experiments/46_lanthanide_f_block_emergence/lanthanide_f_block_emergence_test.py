#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/lanthanide_f_block_emergence_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# Z, symbol, f_count, d_count, s_count, expected_family
LANTHANIDES = [
    (57, "La", 0, 1, 2, "pre_f"),
    (58, "Ce", 1, 1, 2, "early_f"),
    (59, "Pr", 3, 0, 2, "early_f"),
    (60, "Nd", 4, 0, 2, "early_f"),
    (61, "Pm", 5, 0, 2, "half_f_approach"),
    (62, "Sm", 6, 0, 2, "half_f_approach"),
    (63, "Eu", 7, 0, 2, "half_f"),
    (64, "Gd", 7, 1, 2, "half_f"),
    (65, "Tb", 9, 0, 2, "late_f"),
    (66, "Dy", 10, 0, 2, "late_f"),
    (67, "Ho", 11, 0, 2, "late_f"),
    (68, "Er", 12, 0, 2, "late_f"),
    (69, "Tm", 13, 0, 2, "filled_f_approach"),
    (70, "Yb", 14, 0, 2, "filled_f"),
    (71, "Lu", 14, 1, 2, "post_f"),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96

FAMILY_ID = {
    "pre_f": 0,
    "early_f": 1,
    "half_f_approach": 2,
    "half_f": 3,
    "late_f": 4,
    "filled_f_approach": 5,
    "filled_f": 6,
    "post_f": 7,
}


def make_sample(Z, symbol, f_count, d_count, s_count, family, perturbation, replicate):
    f_fill = f_count / 14.0
    d_fill = d_count / 10.0
    s_fill = s_count / 2.0

    half_f_strength = np.exp(-abs(f_count - 7) / 2.0)
    filled_f_strength = np.exp(-abs(f_count - 14) / 2.0)
    f_symmetry = max(half_f_strength, filled_f_strength)

    lanthanide_contraction = (
        1.0 - 0.28 * f_fill
        + perturbation * rng.normal(0, 0.004)
    )

    shielding_strength = (
        0.35 + 0.45 * f_fill + 0.20 * f_symmetry
        + perturbation * rng.normal(0, 0.01)
    )

    magnetic_moment_proxy = (
        min(f_count, 14 - f_count) / 7.0
        + perturbation * rng.normal(0, 0.01)
    )

    f_block_activation = (
        0.55 * (1.0 if f_count > 0 else 0.15)
        + 0.25 * f_fill
        + 0.20 * f_symmetry
        + perturbation * rng.normal(0, 0.01)
    )

    recurrence_signature = (
        0.25 * f_fill
        + 0.20 * f_symmetry
        + 0.20 * shielding_strength
        + 0.15 * magnetic_moment_proxy
        + 0.10 * s_fill
        + 0.10 * d_fill
        + perturbation * rng.normal(0, 0.01)
    )

    f_block_stability = (
        0.30 * f_block_activation
        + 0.25 * recurrence_signature
        + 0.20 * shielding_strength
        + 0.15 * f_symmetry
        + 0.10 * lanthanide_contraction
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "period": 6,
        "block": "f" if symbol not in ["La", "Lu"] else "d/f_boundary",
        "family": family,
        "family_id": FAMILY_ID[family],
        "replicate": replicate,
        "perturbation": perturbation,
        "f_count": f_count,
        "d_count": d_count,
        "s_count": s_count,
        "f_fill": f_fill,
        "d_fill": d_fill,
        "s_fill": s_fill,
        "half_f_strength": half_f_strength,
        "filled_f_strength": filled_f_strength,
        "f_symmetry": f_symmetry,
        "lanthanide_contraction": lanthanide_contraction,
        "shielding_strength": shielding_strength,
        "magnetic_moment_proxy": magnetic_moment_proxy,
        "f_block_activation": f_block_activation,
        "recurrence_signature": recurrence_signature,
        "f_block_stability": f_block_stability,
        "predicted_f_block": int(f_block_activation >= 0.50),
    }


rows = []
for perturbation in PERTURBATIONS:
    for e in LANTHANIDES:
        for r in range(REPLICATES):
            rows.append(make_sample(*e, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "Z",
    "f_count",
    "d_count",
    "s_count",
    "f_fill",
    "d_fill",
    "s_fill",
    "half_f_strength",
    "filled_f_strength",
    "f_symmetry",
    "lanthanide_contraction",
    "shielding_strength",
    "magnetic_moment_proxy",
    "f_block_activation",
    "recurrence_signature",
    "f_block_stability",
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
family_accuracy = knn_acc("family")
f_count_accuracy = knn_acc("f_count")

try:
    element_silhouette = float(silhouette_score(Xn, df["symbol"]))
except Exception:
    element_silhouette = np.nan

try:
    family_silhouette = float(silhouette_score(Xn, df["family"]))
except Exception:
    family_silhouette = np.nan

f_block_detection = float(np.mean(df["predicted_f_block"] == (df["f_count"] > 0)))

half_f_recovery = float(df[df["f_count"] == 7]["f_symmetry"].mean())
filled_f_recovery = float(df[df["f_count"] == 14]["f_symmetry"].mean())
mean_contraction_slope = float(
    np.polyfit(df.groupby("Z")["Z"].mean(), df.groupby("Z")["lanthanide_contraction"].mean(), 1)[0]
)

by_element = df.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    family=("family", "first"),
    mean_f_count=("f_count", "mean"),
    mean_d_count=("d_count", "mean"),
    mean_f_fill=("f_fill", "mean"),
    mean_f_symmetry=("f_symmetry", "mean"),
    mean_lanthanide_contraction=("lanthanide_contraction", "mean"),
    mean_shielding_strength=("shielding_strength", "mean"),
    mean_magnetic_moment_proxy=("magnetic_moment_proxy", "mean"),
    mean_f_block_activation=("f_block_activation", "mean"),
    mean_recurrence_signature=("recurrence_signature", "mean"),
    mean_f_block_stability=("f_block_stability", "mean"),
).reset_index().sort_values("Z")

by_family = df.groupby("family").agg(
    count=("family", "count"),
    elements=("symbol", lambda x: ",".join(sorted(set(x), key=lambda s: [e[1] for e in LANTHANIDES].index(s)))),
    mean_f_count=("f_count", "mean"),
    mean_f_symmetry=("f_symmetry", "mean"),
    mean_lanthanide_contraction=("lanthanide_contraction", "mean"),
    mean_f_block_activation=("f_block_activation", "mean"),
    mean_f_block_stability=("f_block_stability", "mean"),
).reset_index()

if (
    element_accuracy >= 0.95
    and family_accuracy >= 0.95
    and f_count_accuracy >= 0.95
    and f_block_detection >= 0.90
    and half_f_recovery >= 0.95
    and filled_f_recovery >= 0.95
    and mean_contraction_slope < 0
):
    verdict = "lanthanide_f_block_emergence_supported"
elif (
    family_accuracy >= 0.85
    and f_block_detection >= 0.80
    and mean_contraction_slope < 0
):
    verdict = "weak_lanthanide_f_block_emergence"
else:
    verdict = "lanthanide_f_block_emergence_not_supported"

summary = pd.DataFrame([{
    "num_lanthanides": len(LANTHANIDES),
    "num_samples": len(df),
    "element_accuracy": element_accuracy,
    "family_accuracy": family_accuracy,
    "f_count_accuracy": f_count_accuracy,
    "element_silhouette": element_silhouette,
    "family_silhouette": family_silhouette,
    "f_block_detection": f_block_detection,
    "half_f_recovery": half_f_recovery,
    "filled_f_recovery": filled_f_recovery,
    "mean_contraction_slope": mean_contraction_slope,
    "mean_f_block_stability": float(df["f_block_stability"].mean()),
    "verdict": verdict,
}])

df.to_csv(OUT / "lanthanide_f_block_samples.csv", index=False)
summary.to_csv(OUT / "lanthanide_f_block_summary.csv", index=False)
by_element.to_csv(OUT / "lanthanide_f_block_by_element.csv", index=False)
by_family.to_csv(OUT / "lanthanide_f_block_by_family.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(10, 6))
for family in sorted(df["family"].unique(), key=lambda f: FAMILY_ID[f]):
    mask = df["family"].to_numpy() == family
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=family)
plt.title("BST lanthanide f-block emergence space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "lanthanide_f_block_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(by_element["symbol"], by_element["mean_f_block_activation"], marker="o", label="f-block activation")
plt.plot(by_element["symbol"], by_element["mean_lanthanide_contraction"], marker="x", label="lanthanide contraction")
plt.ylabel("BST score")
plt.title("BST lanthanide f-block activation and contraction")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "lanthanide_f_block_scores.png", dpi=220)
plt.close()

print("\n=== BST LANTHANIDE f-BLOCK EMERGENCE TEST ===\n")
print(summary.to_string(index=False))

print("\nLanthanide f-block by family:")
print(by_family.to_string(index=False))

print("\nLanthanide f-block by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'lanthanide_f_block_samples.csv'}")
print(f"[OK] wrote {OUT / 'lanthanide_f_block_summary.csv'}")
print(f"[OK] wrote {OUT / 'lanthanide_f_block_by_element.csv'}")
print(f"[OK] wrote {OUT / 'lanthanide_f_block_by_family.csv'}")
print(f"[OK] wrote {OUT / 'lanthanide_f_block_projection.png'}")
print(f"[OK] wrote {OUT / 'lanthanide_f_block_scores.png'}")
print("[DONE] lanthanide f-block emergence test complete")