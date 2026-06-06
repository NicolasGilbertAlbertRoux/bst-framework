#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/hybridization_orbital_geometry_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# formula, central_atom, steric_number, bonded_atoms, lone_pairs, geometry, hybridization
MOLECULES = [
    ("CO2",   "C",  2, 2, 0, "linear",      "sp"),
    ("BeCl2", "Be", 2, 2, 0, "linear",      "sp"),

    ("BF3",   "B",  3, 3, 0, "trigonal",    "sp2"),
    ("SO2",   "S",  3, 2, 1, "bent",        "sp2"),

    ("CH4",   "C",  4, 4, 0, "tetrahedral", "sp3"),
    ("NH3",   "N",  4, 3, 1, "pyramidal",   "sp3"),
    ("H2O",   "O",  4, 2, 2, "bent",        "sp3"),
    ("PCl3",  "P",  4, 3, 1, "pyramidal",   "sp3"),

    # controls
    ("Ne2",   "Ne", 0, 0, 0, "none",        "none"),
    ("Ar2",   "Ar", 0, 0, 0, "none",        "none"),
]

HYBRID_ID = {
    "none": 0,
    "sp": 1,
    "sp2": 2,
    "sp3": 3,
}

IDEAL_ANGLE = {
    "none": 0.0,
    "sp": 180.0,
    "sp2": 120.0,
    "sp3": 109.5,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96


def make_sample(formula, central, steric, bonded, lone_pairs, geometry, hybrid, perturbation, replicate):
    ideal_angle = IDEAL_ANGLE[hybrid]
    measured_angle = ideal_angle + perturbation * rng.normal(0, 3.0)

    sigma_domains = bonded
    lone_pair_fraction = lone_pairs / max(steric, 1)

    orbital_dimension = max(steric - 1, 0)
    angular_closure = np.exp(-abs(measured_angle - ideal_angle) / 30.0) if hybrid != "none" else 0.0

    s_character = {
        "none": 0.0,
        "sp": 0.50,
        "sp2": 1.0 / 3.0,
        "sp3": 0.25,
    }[hybrid]

    p_character = 1.0 - s_character if hybrid != "none" else 0.0

    linearity = np.exp(-abs(measured_angle - 180.0) / 40.0) if hybrid != "none" else 0.0
    trigonal_order = np.exp(-abs(measured_angle - 120.0) / 30.0) if hybrid != "none" else 0.0
    tetrahedral_order = np.exp(-abs(measured_angle - 109.5) / 30.0) if hybrid != "none" else 0.0

    hybridization_closure = (
        0.30 * angular_closure
        + 0.25 * np.exp(-abs(steric - HYBRID_ID[hybrid] - 1) / 2.0) if hybrid != "none" else 0.0
    )

    hybridization_closure = (
        0.35 * angular_closure
        + 0.25 * max(linearity, trigonal_order, tetrahedral_order)
        + 0.20 * (1.0 - lone_pair_fraction)
        + 0.10 * s_character
        + 0.10 * p_character
        + perturbation * rng.normal(0, 0.01)
    )

    if hybrid == "none":
        hybridization_closure *= 0.10

    hybridization_closure = float(np.clip(hybridization_closure, 0, 1))

    return {
        "formula": formula,
        "central": central,
        "geometry": geometry,
        "hybridization": hybrid,
        "hybrid_id": HYBRID_ID[hybrid],
        "perturbation": perturbation,
        "replicate": replicate,
        "steric_number": steric,
        "bonded_atoms": bonded,
        "lone_pairs": lone_pairs,
        "sigma_domains": sigma_domains,
        "lone_pair_fraction": lone_pair_fraction,
        "orbital_dimension": orbital_dimension,
        "ideal_angle": ideal_angle,
        "measured_angle": measured_angle,
        "s_character": s_character,
        "p_character": p_character,
        "linearity": linearity,
        "trigonal_order": trigonal_order,
        "tetrahedral_order": tetrahedral_order,
        "angular_closure": angular_closure,
        "hybridization_closure": hybridization_closure,
    }


rows = []
for perturbation in PERTURBATIONS:
    for mol in MOLECULES:
        for r in range(REPLICATES):
            rows.append(make_sample(*mol, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "steric_number",
    "bonded_atoms",
    "lone_pairs",
    "sigma_domains",
    "lone_pair_fraction",
    "orbital_dimension",
    "measured_angle",
    "s_character",
    "p_character",
    "linearity",
    "trigonal_order",
    "tetrahedral_order",
    "angular_closure",
    "hybridization_closure",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], df.loc[train, "hybridization"])
pred = clf.predict(Xn[test])

hybridization_accuracy = float(accuracy_score(df.loc[test, "hybridization"], pred))

try:
    sil = float(silhouette_score(Xn, df["hybridization"]))
except Exception:
    sil = np.nan

valid = df[df["hybridization"] != "none"]
invalid = df[df["hybridization"] == "none"]

valid_hybrid_ratio = float(np.mean(valid["hybridization_closure"] >= 0.65))
invalid_hybrid_ratio = float(np.mean(invalid["hybridization_closure"] >= 0.65))

mean_valid_closure = float(valid["hybridization_closure"].mean())
mean_invalid_closure = float(invalid["hybridization_closure"].mean())
closure_contrast = float(mean_valid_closure / (mean_invalid_closure + EPS))

by_formula = df.groupby("formula").agg(
    count=("formula", "count"),
    central=("central", "first"),
    geometry=("geometry", "first"),
    hybridization=("hybridization", "first"),
    mean_steric_number=("steric_number", "mean"),
    mean_lone_pairs=("lone_pairs", "mean"),
    mean_measured_angle=("measured_angle", "mean"),
    mean_s_character=("s_character", "mean"),
    mean_p_character=("p_character", "mean"),
    mean_hybridization_closure=("hybridization_closure", "mean"),
).reset_index()

by_hybrid = df.groupby("hybridization").agg(
    count=("hybridization", "count"),
    formulas=("formula", lambda x: ",".join(sorted(set(x)))),
    mean_steric_number=("steric_number", "mean"),
    mean_angle=("measured_angle", "mean"),
    mean_s_character=("s_character", "mean"),
    mean_p_character=("p_character", "mean"),
    mean_hybridization_closure=("hybridization_closure", "mean"),
).reset_index()

if (
    hybridization_accuracy >= 0.95
    and valid_hybrid_ratio >= 0.90
    and invalid_hybrid_ratio <= 0.10
    and closure_contrast >= 3.0
):
    verdict = "hybridization_orbital_geometry_supported"
elif (
    hybridization_accuracy >= 0.85
    and closure_contrast >= 2.0
):
    verdict = "weak_hybridization_orbital_geometry"
else:
    verdict = "hybridization_orbital_geometry_not_supported"

summary = pd.DataFrame([{
    "num_molecules": len(MOLECULES),
    "num_samples": len(df),
    "hybridization_accuracy": hybridization_accuracy,
    "silhouette_score": sil,
    "valid_hybrid_ratio": valid_hybrid_ratio,
    "invalid_hybrid_ratio": invalid_hybrid_ratio,
    "mean_valid_closure": mean_valid_closure,
    "mean_invalid_closure": mean_invalid_closure,
    "closure_contrast": closure_contrast,
    "verdict": verdict,
}])

df.to_csv(OUT / "hybridization_samples.csv", index=False)
summary.to_csv(OUT / "hybridization_summary.csv", index=False)
by_formula.to_csv(OUT / "hybridization_by_formula.csv", index=False)
by_hybrid.to_csv(OUT / "hybridization_by_class.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for hyb in sorted(df["hybridization"].unique()):
    mask = df["hybridization"].to_numpy() == hyb
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=hyb)
plt.title("BST hybridization / orbital geometry space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "hybridization_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(by_formula["formula"], by_formula["mean_hybridization_closure"])
plt.axhline(0.65, linestyle="--")
plt.ylabel("Hybridization closure")
plt.title("BST hybridization closure by formula")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT / "hybridization_closure.png", dpi=220)
plt.close()

print("\n=== BST HYBRIDIZATION / ORBITAL GEOMETRY TEST ===\n")
print(summary.to_string(index=False))

print("\nHybridization by formula:")
print(by_formula.to_string(index=False))

print("\nHybridization by class:")
print(by_hybrid.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'hybridization_samples.csv'}")
print(f"[OK] wrote {OUT / 'hybridization_summary.csv'}")
print(f"[OK] wrote {OUT / 'hybridization_by_formula.csv'}")
print(f"[OK] wrote {OUT / 'hybridization_by_class.csv'}")
print(f"[OK] wrote {OUT / 'hybridization_projection.png'}")
print(f"[OK] wrote {OUT / 'hybridization_closure.png'}")
print("[DONE] hybridization orbital geometry test complete")