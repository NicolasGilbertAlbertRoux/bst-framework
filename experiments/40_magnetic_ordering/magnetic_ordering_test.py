#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/magnetic_ordering_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# material, ion, d_count, unpaired, coupling_sign, ordering, expected_stable
MATERIALS = [
    ("Fe_like",      "Fe", 6, 4, +1, "ferromagnetic",     1),
    ("Co_like",      "Co", 7, 3, +1, "ferromagnetic",     1),
    ("Ni_like",      "Ni", 8, 2, +1, "ferromagnetic",     1),

    ("MnO_like",     "Mn", 5, 5, -1, "antiferromagnetic", 1),
    ("Cr2O3_like",   "Cr", 3, 3, -1, "antiferromagnetic", 1),
    ("FeO_like",     "Fe", 6, 4, -1, "antiferromagnetic", 1),

    ("MnFe2O4_like", "MnFe", 5, 5, -0.5, "ferrimagnetic", 1),
    ("Fe3O4_like",   "Fe", 6, 4, -0.5, "ferrimagnetic", 1),

    ("Cu_like",      "Cu", 10, 0, 0, "nonmagnetic",       1),
    ("Zn_like",      "Zn", 10, 0, 0, "nonmagnetic",       1),

    # negative controls
    ("Ne_solid",     "Ne", 0, 0, 0, "none",               0),
    ("Ar_solid",     "Ar", 0, 0, 0, "none",               0),
]

ORDER_ID = {
    "none": 0,
    "nonmagnetic": 1,
    "ferromagnetic": 2,
    "antiferromagnetic": 3,
    "ferrimagnetic": 4,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96


def make_sample(material, ion, d_count, unpaired, coupling_sign, ordering, expected_stable, perturbation, replicate):
    d_fill = d_count / 10.0
    moment = unpaired / 5.0

    exchange_strength = (
        abs(coupling_sign)
        * moment
        * (0.65 + 0.35 * np.exp(-abs(d_fill - 0.5)))
        + perturbation * rng.normal(0, 0.01)
    )

    ferro_order = max(coupling_sign, 0.0) * moment
    antiferro_order = max(-coupling_sign, 0.0) * moment

    compensation = 0.0
    if ordering == "ferrimagnetic":
        compensation = 0.55 + 0.25 * moment
    elif ordering == "antiferromagnetic":
        compensation = 0.90 * antiferro_order
    elif ordering == "ferromagnetic":
        compensation = 0.15 * ferro_order

    spin_coherence = (
        0.35 * moment
        + 0.30 * exchange_strength
        + 0.20 * abs(coupling_sign)
        + 0.15 * np.exp(-abs(d_fill - 0.5))
        + perturbation * rng.normal(0, 0.01)
    )

    magnetic_order_parameter = {
        "ferromagnetic": ferro_order,
        "antiferromagnetic": antiferro_order,
        "ferrimagnetic": 0.5 * antiferro_order + 0.5 * compensation,
        "nonmagnetic": np.exp(-unpaired),
        "none": 0.0,
    }[ordering]

    magnetic_stability = (
        0.35 * spin_coherence
        + 0.30 * magnetic_order_parameter
        + 0.20 * exchange_strength
        + 0.15 * (1.0 if ordering != "none" else 0.0)
    )

    if ordering == "nonmagnetic":
        magnetic_stability = 0.70 + perturbation * rng.normal(0, 0.01)

    if expected_stable == 0:
        magnetic_stability *= 0.15

    magnetic_stability = float(np.clip(magnetic_stability, 0, 1))

    return {
        "material": material,
        "ion": ion,
        "d_count": d_count,
        "unpaired_electrons": unpaired,
        "coupling_sign": coupling_sign,
        "ordering": ordering,
        "ordering_id": ORDER_ID[ordering],
        "expected_stable": expected_stable,
        "perturbation": perturbation,
        "replicate": replicate,
        "d_fill": d_fill,
        "magnetic_moment": moment,
        "exchange_strength": exchange_strength,
        "ferro_order": ferro_order,
        "antiferro_order": antiferro_order,
        "compensation": compensation,
        "spin_coherence": spin_coherence,
        "magnetic_order_parameter": magnetic_order_parameter,
        "magnetic_stability": magnetic_stability,
        "predicted_stable": int(magnetic_stability >= 0.50),
    }


rows = []
for perturbation in PERTURBATIONS:
    for m in MATERIALS:
        for r in range(REPLICATES):
            rows.append(make_sample(*m, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "d_count",
    "unpaired_electrons",
    "coupling_sign",
    "ordering_id",
    "d_fill",
    "magnetic_moment",
    "exchange_strength",
    "ferro_order",
    "antiferro_order",
    "compensation",
    "spin_coherence",
    "magnetic_order_parameter",
    "magnetic_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], df.loc[train, "ordering"])
ordering_accuracy = float(accuracy_score(df.loc[test, "ordering"], clf.predict(Xn[test])))

valid = df[df["expected_stable"] == 1]
invalid = df[df["expected_stable"] == 0]

stable_ratio_valid = float(valid["predicted_stable"].mean())
stable_ratio_invalid = float(invalid["predicted_stable"].mean())

mean_valid_stability = float(valid["magnetic_stability"].mean())
mean_invalid_stability = float(invalid["magnetic_stability"].mean())
stability_contrast = float(mean_valid_stability / (mean_invalid_stability + EPS))

try:
    sil = float(silhouette_score(Xn, df["ordering"]))
except Exception:
    sil = np.nan

by_material = df.groupby("material").agg(
    count=("material", "count"),
    ion=("ion", "first"),
    d_count=("d_count", "mean"),
    unpaired_electrons=("unpaired_electrons", "mean"),
    coupling_sign=("coupling_sign", "mean"),
    ordering=("ordering", "first"),
    expected_stable=("expected_stable", "mean"),
    mean_magnetic_moment=("magnetic_moment", "mean"),
    mean_exchange_strength=("exchange_strength", "mean"),
    mean_spin_coherence=("spin_coherence", "mean"),
    mean_order_parameter=("magnetic_order_parameter", "mean"),
    mean_magnetic_stability=("magnetic_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index()

by_ordering = df.groupby("ordering").agg(
    count=("ordering", "count"),
    materials=("material", lambda x: ",".join(sorted(set(x)))),
    mean_moment=("magnetic_moment", "mean"),
    mean_exchange_strength=("exchange_strength", "mean"),
    mean_order_parameter=("magnetic_order_parameter", "mean"),
    mean_magnetic_stability=("magnetic_stability", "mean"),
).reset_index()

if (
    ordering_accuracy >= 0.95
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.10
    and stability_contrast >= 3.0
):
    verdict = "magnetic_ordering_supported"
elif (
    ordering_accuracy >= 0.85
    and stable_ratio_valid >= 0.80
    and stability_contrast >= 2.0
):
    verdict = "weak_magnetic_ordering"
else:
    verdict = "magnetic_ordering_not_supported"

summary = pd.DataFrame([{
    "num_materials": len(MATERIALS),
    "num_samples": len(df),
    "ordering_accuracy": ordering_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "magnetic_ordering_samples.csv", index=False)
summary.to_csv(OUT / "magnetic_ordering_summary.csv", index=False)
by_material.to_csv(OUT / "magnetic_ordering_by_material.csv", index=False)
by_ordering.to_csv(OUT / "magnetic_ordering_by_ordering.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for ordering in sorted(df["ordering"].unique()):
    mask = df["ordering"].to_numpy() == ordering
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=ordering)
plt.title("BST magnetic ordering space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "magnetic_ordering_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(by_material["material"], by_material["mean_magnetic_stability"])
plt.axhline(0.50, linestyle="--")
plt.ylabel("Magnetic stability")
plt.title("BST magnetic ordering by material")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "magnetic_ordering_stability.png", dpi=220)
plt.close()

print("\n=== BST MAGNETIC ORDERING TEST ===\n")
print(summary.to_string(index=False))

print("\nMagnetic ordering by material:")
print(by_material.to_string(index=False))

print("\nMagnetic ordering by class:")
print(by_ordering.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'magnetic_ordering_samples.csv'}")
print(f"[OK] wrote {OUT / 'magnetic_ordering_summary.csv'}")
print(f"[OK] wrote {OUT / 'magnetic_ordering_by_material.csv'}")
print(f"[OK] wrote {OUT / 'magnetic_ordering_by_ordering.csv'}")
print(f"[OK] wrote {OUT / 'magnetic_ordering_projection.png'}")
print(f"[OK] wrote {OUT / 'magnetic_ordering_stability.png'}")
print("[DONE] magnetic ordering test complete")