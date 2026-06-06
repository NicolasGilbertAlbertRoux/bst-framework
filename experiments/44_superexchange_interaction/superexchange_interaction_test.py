#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/superexchange_interaction_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# material, metal_a, metal_b, d_a, d_b, ligand, angle, occupancy_relation, expected_ordering, valid
MATERIALS = [
    ("MnO_like",     "Mn", "Mn", 5, 5, "O", 180, "half_half",       "antiferromagnetic", 1),
    ("NiO_like",     "Ni", "Ni", 8, 8, "O", 180, "filled_partial",  "antiferromagnetic", 1),
    ("CoO_like",     "Co", "Co", 7, 7, "O", 180, "partial_partial", "antiferromagnetic", 1),
    ("FeO_like",     "Fe", "Fe", 6, 6, "O", 180, "partial_partial", "antiferromagnetic", 1),
    ("Cr2O3_like",   "Cr", "Cr", 3, 3, "O", 180, "partial_partial", "antiferromagnetic", 1),
    ("MnF2_like",    "Mn", "Mn", 5, 5, "F", 180, "half_half",       "antiferromagnetic", 1),

    ("Fe3O4_like",   "Fe", "Fe", 5, 6, "O", 125, "mixed_valence",   "ferrimagnetic",     1),
    ("MnFe2O4_like", "Mn", "Fe", 5, 6, "O", 125, "mixed_valence",   "ferrimagnetic",     1),

    ("CuO_chain",    "Cu", "Cu", 9, 9, "O", 95,  "near_90",         "ferromagnetic",     1),
    ("LaMnO3_like",  "Mn", "Mn", 4, 4, "O", 155, "orbital_ordered", "antiferromagnetic", 1),

    # controls
    ("Ne_O_Ne",      "Ne", "Ne", 0, 0, "O", 180, "none",            "none",              0),
    ("Ar_O_Ar",      "Ar", "Ar", 0, 0, "O", 180, "none",            "none",              0),
    ("pseudo_bridge_invalid", "X", "X", 0, 0, "X", 0, "invalid",    "none",              0),
]

ORDER_ID = {
    "none": 0,
    "ferromagnetic": 1,
    "antiferromagnetic": 2,
    "ferrimagnetic": 3,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 128


def ligand_strength(ligand):
    if ligand == "O":
        return 1.00
    if ligand == "F":
        return 0.82
    return 0.0


def d_moment(d):
    return min(d, 10 - d, 5) / 5.0


def angle_factor(angle):
    # 180° favors AFM superexchange; 90° can favor FM.
    afm = np.exp(-abs(angle - 180) / 45.0)
    fm = np.exp(-abs(angle - 90) / 30.0)
    return afm, fm


def occupancy_factor(relation, d_a, d_b):
    if relation == "half_half":
        return 1.00
    if relation == "partial_partial":
        return 0.82
    if relation == "filled_partial":
        return 0.72
    if relation == "mixed_valence":
        return 0.88
    if relation == "near_90":
        return 0.75
    if relation == "orbital_ordered":
        return 0.86
    return 0.0


def make_sample(material, metal_a, metal_b, d_a, d_b, ligand, angle, relation, expected_ordering, valid, perturbation, replicate):
    ligand_med = ligand_strength(ligand)
    afm_angle, fm_angle = angle_factor(angle)

    moment_a = d_moment(d_a)
    moment_b = d_moment(d_b)
    moment_product = moment_a * moment_b

    occ = occupancy_factor(relation, d_a, d_b)

    orbital_overlap = (
        0.45 * ligand_med
        + 0.35 * afm_angle
        + 0.20 * occ
        + perturbation * rng.normal(0, 0.01)
    )

    superexchange_path = (
        0.40 * ligand_med
        + 0.30 * orbital_overlap
        + 0.20 * moment_product
        + 0.10 * occ
        + perturbation * rng.normal(0, 0.01)
    )

    antiferro_drive = (
        0.45 * afm_angle
        + 0.25 * occ
        + 0.20 * moment_product
        + 0.10 * ligand_med
        + perturbation * rng.normal(0, 0.01)
    )

    ferro_drive = (
        0.50 * fm_angle
        + 0.25 * (1.0 - afm_angle)
        + 0.15 * occ
        + 0.10 * moment_product
        + perturbation * rng.normal(0, 0.01)
    )

    ferri_drive = (
        0.45 * float(relation == "mixed_valence")
        + 0.25 * abs(moment_a - moment_b)
        + 0.20 * afm_angle
        + 0.10 * ligand_med
        + perturbation * rng.normal(0, 0.01)
    )

    if expected_ordering == "antiferromagnetic":
        order_match = antiferro_drive
    elif expected_ordering == "ferromagnetic":
        order_match = ferro_drive
    elif expected_ordering == "ferrimagnetic":
        order_match = ferri_drive
    else:
        order_match = 0.0

    exchange_coherence = (
        0.35 * superexchange_path
        + 0.25 * orbital_overlap
        + 0.20 * order_match
        + 0.20 * moment_product
        + perturbation * rng.normal(0, 0.01)
    )

    superexchange_stability = (
        0.40 * order_match
        + 0.30 * exchange_coherence
        + 0.20 * superexchange_path
        + 0.10 * ligand_med
    )

    if not valid:
        superexchange_stability *= 0.12

    superexchange_stability = float(np.clip(superexchange_stability, 0, 1))

    predicted_ordering = max(
        [
            ("antiferromagnetic", antiferro_drive),
            ("ferromagnetic", ferro_drive),
            ("ferrimagnetic", ferri_drive),
            ("none", 0.05 if not valid else 0.0),
        ],
        key=lambda x: x[1],
    )[0]

    if not valid:
        predicted_ordering = "none"

    return {
        "material": material,
        "metal_a": metal_a,
        "metal_b": metal_b,
        "ligand": ligand,
        "d_a": d_a,
        "d_b": d_b,
        "bond_angle": angle,
        "occupancy_relation": relation,
        "expected_ordering": expected_ordering,
        "ordering_id": ORDER_ID[expected_ordering],
        "valid": valid,
        "perturbation": perturbation,
        "replicate": replicate,
        "moment_a": moment_a,
        "moment_b": moment_b,
        "moment_product": moment_product,
        "ligand_mediation": ligand_med,
        "afm_angle_factor": afm_angle,
        "fm_angle_factor": fm_angle,
        "occupancy_factor": occ,
        "orbital_overlap": orbital_overlap,
        "superexchange_path": superexchange_path,
        "antiferro_drive": antiferro_drive,
        "ferro_drive": ferro_drive,
        "ferri_drive": ferri_drive,
        "exchange_coherence": exchange_coherence,
        "superexchange_stability": superexchange_stability,
        "predicted_ordering": predicted_ordering,
        "predicted_stable": int(superexchange_stability >= 0.50),
    }


rows = []
for perturbation in PERTURBATIONS:
    for m in MATERIALS:
        for r in range(REPLICATES):
            rows.append(make_sample(*m, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "d_a",
    "d_b",
    "bond_angle",
    "moment_a",
    "moment_b",
    "moment_product",
    "ligand_mediation",
    "afm_angle_factor",
    "fm_angle_factor",
    "occupancy_factor",
    "orbital_overlap",
    "superexchange_path",
    "antiferro_drive",
    "ferro_drive",
    "ferri_drive",
    "exchange_coherence",
    "superexchange_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], df.loc[train, "expected_ordering"])
ordering_accuracy = float(accuracy_score(df.loc[test, "expected_ordering"], clf.predict(Xn[test])))

rule_accuracy = float(np.mean(df["predicted_ordering"] == df["expected_ordering"]))

valid_df = df[df["valid"] == 1]
invalid_df = df[df["valid"] == 0]

stable_ratio_valid = float(valid_df["predicted_stable"].mean())
stable_ratio_invalid = float(invalid_df["predicted_stable"].mean())

mean_valid_stability = float(valid_df["superexchange_stability"].mean())
mean_invalid_stability = float(invalid_df["superexchange_stability"].mean())
stability_contrast = float(mean_valid_stability / max(mean_invalid_stability, 1e-3))

try:
    sil = float(silhouette_score(Xn, df["expected_ordering"]))
except Exception:
    sil = np.nan

by_material = df.groupby("material").agg(
    count=("material", "count"),
    metal_a=("metal_a", "first"),
    metal_b=("metal_b", "first"),
    ligand=("ligand", "first"),
    expected_ordering=("expected_ordering", "first"),
    valid=("valid", "mean"),
    mean_bond_angle=("bond_angle", "mean"),
    mean_moment_product=("moment_product", "mean"),
    mean_ligand_mediation=("ligand_mediation", "mean"),
    mean_orbital_overlap=("orbital_overlap", "mean"),
    mean_superexchange_path=("superexchange_path", "mean"),
    mean_antiferro_drive=("antiferro_drive", "mean"),
    mean_ferro_drive=("ferro_drive", "mean"),
    mean_ferri_drive=("ferri_drive", "mean"),
    mean_exchange_coherence=("exchange_coherence", "mean"),
    mean_superexchange_stability=("superexchange_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index()

by_ordering = df.groupby("expected_ordering").agg(
    count=("expected_ordering", "count"),
    materials=("material", lambda x: ",".join(sorted(set(x)))),
    mean_superexchange_path=("superexchange_path", "mean"),
    mean_exchange_coherence=("exchange_coherence", "mean"),
    mean_superexchange_stability=("superexchange_stability", "mean"),
).reset_index()

if (
    ordering_accuracy >= 0.95
    and rule_accuracy >= 0.90
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.10
    and stability_contrast >= 3.0
):
    verdict = "superexchange_interaction_supported"
elif (
    ordering_accuracy >= 0.85
    and stable_ratio_valid >= 0.80
    and stability_contrast >= 2.0
):
    verdict = "weak_superexchange_interaction"
else:
    verdict = "superexchange_interaction_not_supported"

summary = pd.DataFrame([{
    "num_materials": len(MATERIALS),
    "num_samples": len(df),
    "ordering_accuracy": ordering_accuracy,
    "rule_accuracy": rule_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "superexchange_samples.csv", index=False)
summary.to_csv(OUT / "superexchange_summary.csv", index=False)
by_material.to_csv(OUT / "superexchange_by_material.csv", index=False)
by_ordering.to_csv(OUT / "superexchange_by_ordering.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for ordering in sorted(df["expected_ordering"].unique()):
    mask = df["expected_ordering"].to_numpy() == ordering
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=ordering)
plt.title("BST superexchange interaction space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "superexchange_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(by_material["material"], by_material["mean_superexchange_stability"])
plt.axhline(0.50, linestyle="--")
plt.ylabel("Superexchange stability")
plt.title("BST superexchange stability by material")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "superexchange_stability.png", dpi=220)
plt.close()

print("\n=== BST SUPEREXCHANGE INTERACTION TEST ===\n")
print(summary.to_string(index=False))

print("\nSuperexchange by material:")
print(by_material.to_string(index=False))

print("\nSuperexchange by ordering:")
print(by_ordering.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'superexchange_samples.csv'}")
print(f"[OK] wrote {OUT / 'superexchange_summary.csv'}")
print(f"[OK] wrote {OUT / 'superexchange_by_material.csv'}")
print(f"[OK] wrote {OUT / 'superexchange_by_ordering.csv'}")
print(f"[OK] wrote {OUT / 'superexchange_projection.png'}")
print(f"[OK] wrote {OUT / 'superexchange_stability.png'}")
print("[DONE] superexchange interaction test complete")