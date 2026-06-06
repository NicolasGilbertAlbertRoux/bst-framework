#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/molecular_geometry_reconstruction_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# formula, central, bonded_atoms, lone_pairs, expected_geometry, ideal_angle
MOLECULES = [
    ("CO2",  "C", 2, 0, "linear",      180.0),
    ("BeCl2","Be",2, 0, "linear",      180.0),
    ("BF3",  "B", 3, 0, "trigonal",    120.0),
    ("CH4",  "C", 4, 0, "tetrahedral", 109.5),
    ("NH3",  "N", 3, 1, "pyramidal",   107.0),
    ("H2O",  "O", 2, 2, "bent",        104.5),
    ("SO2",  "S", 2, 1, "bent",        119.0),
    ("PCl3", "P", 3, 1, "pyramidal",   100.0),

    # negative / weak geometry controls
    ("Ne2",  "Ne",0, 0, "none",          0.0),
    ("Ar2",  "Ar",0, 0, "none",          0.0),
]

GEOMETRY_ID = {
    "none": 0,
    "linear": 1,
    "trigonal": 2,
    "tetrahedral": 3,
    "pyramidal": 4,
    "bent": 5,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96


def expected_features(bonded_atoms, lone_pairs, ideal_angle):
    steric_number = bonded_atoms + lone_pairs

    if steric_number == 0:
        angular_order = 0.0
        geometry_closure = 0.0
        lone_pair_distortion = 0.0
    else:
        angular_order = ideal_angle / 180.0
        geometry_closure = np.exp(-abs(steric_number - round(steric_number)) / 4.0)
        lone_pair_distortion = lone_pairs / max(steric_number, 1)

    tetrahedrality = np.exp(-abs(ideal_angle - 109.5) / 40.0) if steric_number > 0 else 0.0
    linearity = np.exp(-abs(ideal_angle - 180.0) / 30.0) if steric_number > 0 else 0.0
    planarity = np.exp(-abs(ideal_angle - 120.0) / 30.0) if steric_number > 0 else 0.0
    bend_signature = lone_pair_distortion * (1.0 - linearity)

    return steric_number, angular_order, geometry_closure, lone_pair_distortion, tetrahedrality, linearity, planarity, bend_signature


def make_sample(formula, central, bonded_atoms, lone_pairs, geometry, ideal_angle, perturbation, replicate):
    (
        steric_number,
        angular_order,
        geometry_closure,
        lone_pair_distortion,
        tetrahedrality,
        linearity,
        planarity,
        bend_signature,
    ) = expected_features(bonded_atoms, lone_pairs, ideal_angle)

    angle = ideal_angle + perturbation * rng.normal(0, 3.0)

    angular_stability = np.exp(-abs(angle - ideal_angle) / 30.0)

    geometry_stability = (
        0.30 * geometry_closure
        + 0.25 * angular_stability
        + 0.20 * (1.0 - min(lone_pair_distortion, 1.0))
        + 0.15 * max(linearity, planarity, tetrahedrality)
        + 0.10 * (1.0 if geometry != "none" else 0.0)
        + perturbation * rng.normal(0, 0.01)
    )

    if geometry == "none":
        geometry_stability *= 0.15

    geometry_stability = float(np.clip(geometry_stability, 0, 1))

    return {
        "formula": formula,
        "central": central,
        "geometry": geometry,
        "geometry_id": GEOMETRY_ID[geometry],
        "perturbation": perturbation,
        "replicate": replicate,
        "bonded_atoms": bonded_atoms,
        "lone_pairs": lone_pairs,
        "steric_number": steric_number,
        "ideal_angle": ideal_angle,
        "measured_angle": angle,
        "angular_order": angular_order,
        "geometry_closure": geometry_closure,
        "lone_pair_distortion": lone_pair_distortion,
        "tetrahedrality": tetrahedrality,
        "linearity": linearity,
        "planarity": planarity,
        "bend_signature": bend_signature,
        "angular_stability": angular_stability,
        "geometry_stability": geometry_stability,
    }


rows = []
for perturbation in PERTURBATIONS:
    for mol in MOLECULES:
        for r in range(REPLICATES):
            rows.append(make_sample(*mol, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "bonded_atoms",
    "lone_pairs",
    "steric_number",
    "measured_angle",
    "angular_order",
    "geometry_closure",
    "lone_pair_distortion",
    "tetrahedrality",
    "linearity",
    "planarity",
    "bend_signature",
    "angular_stability",
    "geometry_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], df.loc[train, "geometry"])
pred = clf.predict(Xn[test])

geometry_accuracy = float(accuracy_score(df.loc[test, "geometry"], pred))

try:
    sil = float(silhouette_score(Xn, df["geometry"]))
except Exception:
    sil = np.nan

valid = df[df["geometry"] != "none"]
invalid = df[df["geometry"] == "none"]

valid_geometry_ratio = float(np.mean(valid["geometry_stability"] >= 0.65))
invalid_geometry_ratio = float(np.mean(invalid["geometry_stability"] >= 0.65))

mean_valid_stability = float(valid["geometry_stability"].mean())
mean_invalid_stability = float(invalid["geometry_stability"].mean())
stability_contrast = float(mean_valid_stability / (mean_invalid_stability + EPS))

by_formula = df.groupby("formula").agg(
    count=("formula", "count"),
    geometry=("geometry", "first"),
    mean_bonded_atoms=("bonded_atoms", "mean"),
    mean_lone_pairs=("lone_pairs", "mean"),
    mean_steric_number=("steric_number", "mean"),
    mean_measured_angle=("measured_angle", "mean"),
    mean_geometry_closure=("geometry_closure", "mean"),
    mean_lone_pair_distortion=("lone_pair_distortion", "mean"),
    mean_geometry_stability=("geometry_stability", "mean"),
).reset_index()

by_geometry = df.groupby("geometry").agg(
    count=("geometry", "count"),
    formulas=("formula", lambda x: ",".join(sorted(set(x)))),
    mean_steric_number=("steric_number", "mean"),
    mean_angle=("measured_angle", "mean"),
    mean_lone_pair_distortion=("lone_pair_distortion", "mean"),
    mean_geometry_stability=("geometry_stability", "mean"),
).reset_index()

if (
    geometry_accuracy >= 0.95
    and valid_geometry_ratio >= 0.90
    and invalid_geometry_ratio <= 0.10
    and stability_contrast >= 3.0
):
    verdict = "molecular_geometry_reconstruction_supported"
elif (
    geometry_accuracy >= 0.85
    and stability_contrast >= 2.0
):
    verdict = "weak_molecular_geometry_reconstruction"
else:
    verdict = "molecular_geometry_reconstruction_not_supported"

summary = pd.DataFrame([{
    "num_molecules": len(MOLECULES),
    "num_samples": len(df),
    "geometry_accuracy": geometry_accuracy,
    "silhouette_score": sil,
    "valid_geometry_ratio": valid_geometry_ratio,
    "invalid_geometry_ratio": invalid_geometry_ratio,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "verdict": verdict,
}])

df.to_csv(OUT / "molecular_geometry_samples.csv", index=False)
summary.to_csv(OUT / "molecular_geometry_summary.csv", index=False)
by_formula.to_csv(OUT / "molecular_geometry_by_formula.csv", index=False)
by_geometry.to_csv(OUT / "molecular_geometry_by_geometry.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for geom in sorted(df["geometry"].unique()):
    mask = df["geometry"].to_numpy() == geom
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=geom)
plt.title("BST molecular geometry reconstruction space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "molecular_geometry_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(by_formula["formula"], by_formula["mean_geometry_stability"])
plt.axhline(0.65, linestyle="--")
plt.ylabel("Geometry stability")
plt.title("BST molecular geometry stability by formula")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT / "molecular_geometry_stability.png", dpi=220)
plt.close()

print("\n=== BST MOLECULAR GEOMETRY RECONSTRUCTION TEST ===\n")
print(summary.to_string(index=False))

print("\nMolecular geometry by formula:")
print(by_formula.to_string(index=False))

print("\nMolecular geometry by class:")
print(by_geometry.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'molecular_geometry_samples.csv'}")
print(f"[OK] wrote {OUT / 'molecular_geometry_summary.csv'}")
print(f"[OK] wrote {OUT / 'molecular_geometry_by_formula.csv'}")
print(f"[OK] wrote {OUT / 'molecular_geometry_by_geometry.csv'}")
print(f"[OK] wrote {OUT / 'molecular_geometry_projection.png'}")
print(f"[OK] wrote {OUT / 'molecular_geometry_stability.png'}")
print("[DONE] molecular geometry reconstruction test complete")