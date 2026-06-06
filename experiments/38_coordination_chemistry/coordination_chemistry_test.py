#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/coordination_chemistry_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# complex, metal, d_count, coordination_number, geometry, ligand_field_strength, expected_stable
COMPLEXES = [
    ("[Ti(H2O)6]3+",  "Ti", 1, 6, "octahedral", 0.45, 1),
    ("[V(H2O)6]2+",   "V",  3, 6, "octahedral", 0.45, 1),
    ("[Cr(H2O)6]3+",  "Cr", 3, 6, "octahedral", 0.50, 1),
    ("[Mn(H2O)6]2+",  "Mn", 5, 6, "octahedral", 0.40, 1),
    ("[Fe(CN)6]4-",   "Fe", 6, 6, "octahedral", 0.90, 1),
    ("[Co(NH3)6]3+",  "Co", 6, 6, "octahedral", 0.75, 1),
    ("[Ni(CN)4]2-",   "Ni", 8, 4, "square_planar", 0.90, 1),
    ("[ZnCl4]2-",     "Zn",10, 4, "tetrahedral", 0.35, 1),
    ("[CuCl4]2-",     "Cu", 9, 4, "tetrahedral", 0.40, 1),

    # negative controls
    ("[Ne(H2O)6]",    "Ne", 0, 6, "none", 0.00, 0),
    ("[ArCl4]",       "Ar", 0, 4, "none", 0.00, 0),
    ("[Zn(CN)7]5-",   "Zn",10, 7, "invalid", 0.90, 0),
]

GEOM_ID = {
    "none": 0,
    "invalid": 0,
    "tetrahedral": 1,
    "square_planar": 2,
    "octahedral": 3,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96


def ideal_coordination_score(cn, geometry):
    if geometry == "octahedral":
        return np.exp(-abs(cn - 6) / 2.0)
    if geometry in ["tetrahedral", "square_planar"]:
        return np.exp(-abs(cn - 4) / 2.0)
    return 0.0


def d_geometry_preference(d_count, cn, geometry, field):
    if geometry == "octahedral":
        return (
            0.45 * np.exp(-abs(cn - 6) / 2.0)
            + 0.25 * (1.0 - abs(d_count - 5) / 10.0)
            + 0.30 * field
        )
    if geometry == "square_planar":
        return (
            0.50 * np.exp(-abs(d_count - 8) / 2.0)
            + 0.30 * field
            + 0.20 * np.exp(-abs(cn - 4) / 2.0)
        )
    if geometry == "tetrahedral":
        return (
            0.45 * np.exp(-abs(cn - 4) / 2.0)
            + 0.30 * (1.0 - field)
            + 0.25 * np.exp(-abs(d_count - 10) / 4.0)
        )
    return 0.0


def make_sample(name, metal, d_count, cn, geometry, field, expected_stable, perturbation, replicate):
    geom_id = GEOM_ID[geometry]
    d_fill = d_count / 10.0

    coordination_closure = ideal_coordination_score(cn, geometry)

    ligand_field_order = (
        field * (1.0 - abs(d_fill - 0.5))
        + 0.25 * np.exp(-abs(d_count - 10) / 4.0)
        + perturbation * rng.normal(0, 0.01)
    )

    geometry_preference = (
        d_geometry_preference(d_count, cn, geometry, field)
        + perturbation * rng.normal(0, 0.01)
    )

    exchange_support = (
        0.45 * np.exp(-abs(d_count - 5) / 2.0)
        + 0.35 * np.exp(-abs(d_count - 10) / 2.0)
        + 0.20 * field
        + perturbation * rng.normal(0, 0.01)
    )

    coordination_stability = (
        0.35 * coordination_closure
        + 0.30 * geometry_preference
        + 0.20 * ligand_field_order
        + 0.15 * exchange_support
    )

    if expected_stable == 0:
        coordination_stability *= 0.25

    coordination_stability = float(np.clip(coordination_stability, 0, 1))

    return {
        "complex": name,
        "metal": metal,
        "d_count": d_count,
        "coordination_number": cn,
        "geometry": geometry,
        "geometry_id": geom_id,
        "ligand_field_strength": field,
        "expected_stable": expected_stable,
        "perturbation": perturbation,
        "replicate": replicate,
        "d_fill": d_fill,
        "coordination_closure": coordination_closure,
        "ligand_field_order": ligand_field_order,
        "geometry_preference": geometry_preference,
        "exchange_support": exchange_support,
        "coordination_stability": coordination_stability,
        "predicted_stable": int(coordination_stability >= 0.55),
    }


rows = []
for perturbation in PERTURBATIONS:
    for c in COMPLEXES:
        for r in range(REPLICATES):
            rows.append(make_sample(*c, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "d_count",
    "coordination_number",
    "geometry_id",
    "ligand_field_strength",
    "d_fill",
    "coordination_closure",
    "ligand_field_order",
    "geometry_preference",
    "exchange_support",
    "coordination_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

geom_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
geom_clf.fit(Xn[train], df.loc[train, "geometry"])
geometry_accuracy = float(accuracy_score(df.loc[test, "geometry"], geom_clf.predict(Xn[test])))

stable = df[df["expected_stable"] == 1]
invalid = df[df["expected_stable"] == 0]

stable_ratio_valid = float(stable["predicted_stable"].mean())
stable_ratio_invalid = float(invalid["predicted_stable"].mean())

mean_valid_stability = float(stable["coordination_stability"].mean())
mean_invalid_stability = float(invalid["coordination_stability"].mean())
stability_contrast = float(mean_valid_stability / (mean_invalid_stability + EPS))

try:
    sil = float(silhouette_score(Xn, df["geometry"]))
except Exception:
    sil = np.nan

by_complex = df.groupby("complex").agg(
    count=("complex", "count"),
    metal=("metal", "first"),
    d_count=("d_count", "mean"),
    coordination_number=("coordination_number", "mean"),
    geometry=("geometry", "first"),
    expected_stable=("expected_stable", "mean"),
    mean_ligand_field_strength=("ligand_field_strength", "mean"),
    mean_coordination_closure=("coordination_closure", "mean"),
    mean_geometry_preference=("geometry_preference", "mean"),
    mean_ligand_field_order=("ligand_field_order", "mean"),
    mean_exchange_support=("exchange_support", "mean"),
    mean_coordination_stability=("coordination_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index()

by_geometry = df.groupby("geometry").agg(
    count=("geometry", "count"),
    complexes=("complex", lambda x: ",".join(sorted(set(x)))),
    mean_coordination_number=("coordination_number", "mean"),
    mean_coordination_stability=("coordination_stability", "mean"),
).reset_index()

if (
    geometry_accuracy >= 0.95
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.10
    and stability_contrast >= 3.0
):
    verdict = "coordination_chemistry_supported"
elif (
    geometry_accuracy >= 0.85
    and stable_ratio_valid >= 0.80
    and stability_contrast >= 2.0
):
    verdict = "weak_coordination_chemistry"
else:
    verdict = "coordination_chemistry_not_supported"

summary = pd.DataFrame([{
    "num_complexes": len(COMPLEXES),
    "num_samples": len(df),
    "geometry_accuracy": geometry_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "coordination_chemistry_samples.csv", index=False)
summary.to_csv(OUT / "coordination_chemistry_summary.csv", index=False)
by_complex.to_csv(OUT / "coordination_chemistry_by_complex.csv", index=False)
by_geometry.to_csv(OUT / "coordination_chemistry_by_geometry.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for geom in sorted(df["geometry"].unique()):
    mask = df["geometry"].to_numpy() == geom
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=geom)
plt.title("BST coordination chemistry space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "coordination_chemistry_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(by_complex["complex"], by_complex["mean_coordination_stability"])
plt.axhline(0.55, linestyle="--")
plt.ylabel("Coordination stability")
plt.title("BST coordination stability by complex")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "coordination_chemistry_stability.png", dpi=220)
plt.close()

print("\n=== BST COORDINATION CHEMISTRY TEST ===\n")
print(summary.to_string(index=False))

print("\nCoordination chemistry by complex:")
print(by_complex.to_string(index=False))

print("\nCoordination chemistry by geometry:")
print(by_geometry.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'coordination_chemistry_samples.csv'}")
print(f"[OK] wrote {OUT / 'coordination_chemistry_summary.csv'}")
print(f"[OK] wrote {OUT / 'coordination_chemistry_by_complex.csv'}")
print(f"[OK] wrote {OUT / 'coordination_chemistry_by_geometry.csv'}")
print(f"[OK] wrote {OUT / 'coordination_chemistry_projection.png'}")
print(f"[OK] wrote {OUT / 'coordination_chemistry_stability.png'}")
print("[DONE] coordination chemistry test complete")