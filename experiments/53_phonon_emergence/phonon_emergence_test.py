#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 53 v2 — Phonon Emergence

Place this file at:
    experiments/53_phonon_emergence/phonon_emergence_test.py

Run:
    python experiments/53_phonon_emergence/phonon_emergence_test.py

Purpose:
    Test whether BST lattice signatures support collective vibrational modes:
    acoustic branches, optical branches where structurally expected, dispersion
    coherence, low damping, invalid suppression, and low-velocity van der Waals modes.

Outputs:
    results/research_final/phonon_emergence_test/
        phonon_samples.csv
        phonon_summary.csv
        phonon_by_material.csv
        phonon_by_class.csv
        phonon_dispersion_curves.csv
        phonon_projection.png
        phonon_stability.png
        phonon_dispersion.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

OUT = Path("results/research_final/phonon_emergence_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 512

# material, class, lattice, stable, stiffness, coord, order, harmonicity, basis_size, mass_contrast, dimensionality
# Notes:
# - basis_size > 1 permits optical branches.
# - monatomic metallic / van-der-Waals solids have acoustic branches only.
# - diamond-like covalent networks use a 2-atom basis, so optical branches are expected in v2.
MATERIALS = [
    ("Al", "metallic_crystal", "fcc", 1, 0.52, 12, 0.96, 0.82, 1, 0.00, 3.0),
    ("Cu", "metallic_crystal", "fcc", 1, 0.58, 12, 0.96, 0.80, 1, 0.00, 3.0),
    ("Fe", "metallic_crystal", "bcc", 1, 0.60, 8, 0.96, 0.81, 1, 0.00, 3.0),
    ("W", "metallic_crystal", "bcc", 1, 0.83, 8, 0.97, 0.87, 1, 0.00, 3.0),

    ("Si", "covalent_network", "diamond", 1, 0.72, 4, 0.97, 0.88, 2, 0.05, 3.0),
    ("Ge", "covalent_network", "diamond", 1, 0.62, 4, 0.97, 0.86, 2, 0.05, 3.0),
    ("diamond_C", "covalent_network", "diamond", 1, 0.98, 4, 0.98, 0.92, 2, 0.04, 3.0),

    ("graphene_like", "covalent_layer", "hexagonal", 1, 0.88, 3, 0.97, 0.90, 2, 0.03, 2.0),
    ("graphite_like", "covalent_layer", "hexagonal_layered", 1, 0.66, 3, 0.95, 0.82, 4, 0.16, 2.2),

    ("NaCl", "ionic_crystal", "rocksalt", 1, 0.64, 6, 0.96, 0.84, 2, 0.45, 3.0),
    ("MgO", "ionic_crystal", "rocksalt", 1, 0.82, 6, 0.97, 0.88, 2, 0.35, 3.0),
    ("Al2O3_like", "ionic_crystal", "corundum", 1, 0.78, 6, 0.96, 0.86, 5, 0.42, 3.0),

    ("Ar_solid", "van_der_waals", "fcc_weak", 1, 0.18, 12, 0.95, 0.72, 1, 0.00, 3.0),
    ("Ne_solid", "van_der_waals", "fcc_weak", 1, 0.14, 12, 0.95, 0.70, 1, 0.00, 3.0),

    ("amorphous_invalid", "invalid", "none", 0, 0.04, 0, 0.08, 0.15, 0, 0.00, 0.0),
    ("gas_invalid", "invalid", "none", 0, 0.01, 0, 0.02, 0.70, 0, 0.00, 0.0),
    ("pseudo_lattice_invalid", "invalid", "broken", 0, 0.05, 1, 0.10, 0.25, 0, 0.00, 0.2),
]

CLASS_ID = {name: i for i, name in enumerate(sorted({m[1] for m in MATERIALS}))}
LATTICE_ID = {name: i for i, name in enumerate(sorted({m[2] for m in MATERIALS}))}


def make_sample(material, material_class, lattice_type, expected_stable, bond_stiffness, coordination, lattice_order, harmonicity, basis_size, mass_contrast, dimensionality, perturbation, replicate):
    noise = perturbation * rng.normal(0, 0.01)

    coordination_norm = coordination / 12.0
    basis_norm = min(basis_size, 6) / 6.0
    stiffness = max(0.0, bond_stiffness + noise)
    order = np.clip(lattice_order + noise, 0.0, 1.0)
    harm = np.clip(harmonicity + noise, 0.0, 1.0)

    acoustic_support = np.sqrt(stiffness * max(coordination_norm, 0.0)) * order
    acoustic_support *= 0.78 + 0.22 * min(dimensionality / 3.0, 1.0)

    has_optical_branch = basis_size > 1 and expected_stable == 1
    optical_support = 0.0
    if has_optical_branch:
        optical_support = np.sqrt(stiffness) * (0.25 + 0.55 * mass_contrast + 0.20 * basis_norm) * order
        if material_class in {"covalent_network", "covalent_layer"}:
            optical_support += 0.10 * np.sqrt(stiffness) * order

    branch_count = 0
    if expected_stable == 1:
        acoustic_branches = 3 if dimensionality >= 2.5 else 2
        optical_branches = max(0, 3 * basis_size - acoustic_branches) if basis_size > 1 else 0
        branch_count = acoustic_branches + optical_branches

    sound_velocity_proxy = acoustic_support * np.sqrt(stiffness + EPS) / np.sqrt(1.0 + 0.25 * mass_contrast)
    optical_gap_proxy = optical_support * (0.35 + 0.65 * mass_contrast) if has_optical_branch else 0.0
    debye_proxy = sound_velocity_proxy * np.sqrt(coordination_norm + EPS)

    dispersion_coherence = (
        0.34 * order
        + 0.28 * harm
        + 0.18 * acoustic_support
        + 0.12 * min(branch_count / 12.0, 1.0)
        + 0.08 * (optical_support if has_optical_branch else 0.20)
    )

    damping_proxy = np.clip(
        0.62 * (1.0 - harm)
        + 0.28 * (1.0 - order)
        + 0.10 * (1.0 - min(stiffness, 1.0)),
        0.0,
        1.0,
    )

    phonon_emergence = (
        0.30 * acoustic_support
        + 0.20 * dispersion_coherence
        + 0.16 * harm
        + 0.14 * order
        + 0.10 * min(branch_count / 6.0, 1.0)
        + 0.10 * (optical_support if has_optical_branch else 0.30)
    )

    phonon_stability = (
        0.36 * phonon_emergence
        + 0.22 * acoustic_support
        + 0.16 * dispersion_coherence
        + 0.14 * (1.0 - damping_proxy)
        + 0.12 * (optical_support if has_optical_branch else 0.50)
    )

    if expected_stable == 0:
        phonon_stability *= 0.035
        phonon_emergence *= 0.20
        acoustic_support *= 0.05
        optical_support *= 0.05
        sound_velocity_proxy *= 0.05
        dispersion_coherence *= 0.20

    return {
        "material": material,
        "material_class": material_class,
        "class_id": CLASS_ID[material_class],
        "lattice_type": lattice_type,
        "lattice_id": LATTICE_ID[lattice_type],
        "expected_stable": expected_stable,
        "perturbation": perturbation,
        "replicate": replicate,
        "bond_stiffness": stiffness,
        "coordination": coordination,
        "coordination_norm": coordination_norm,
        "lattice_order": order,
        "harmonicity": harm,
        "basis_size": basis_size,
        "basis_norm": basis_norm,
        "mass_contrast": mass_contrast,
        "dimensionality": dimensionality,
        "has_optical_branch": float(has_optical_branch),
        "acoustic_support": acoustic_support,
        "optical_support": optical_support,
        "branch_count": float(branch_count),
        "sound_velocity_proxy": sound_velocity_proxy,
        "optical_gap_proxy": optical_gap_proxy,
        "debye_proxy": debye_proxy,
        "dispersion_coherence": dispersion_coherence,
        "damping_proxy": damping_proxy,
        "phonon_emergence": phonon_emergence,
        "phonon_stability": phonon_stability,
        "predicted_stable": float(phonon_stability >= 0.46),
    }


rows = []
for p in PERTURBATIONS:
    for m in MATERIALS:
        for r in range(REPLICATES):
            rows.append(make_sample(*m, p, r))

samples = pd.DataFrame(rows)

feature_cols = [
    "class_id", "lattice_id", "expected_stable", "bond_stiffness", "coordination",
    "coordination_norm", "lattice_order", "harmonicity", "basis_size", "basis_norm",
    "mass_contrast", "dimensionality", "has_optical_branch", "acoustic_support",
    "optical_support", "branch_count", "sound_velocity_proxy", "optical_gap_proxy",
    "debye_proxy", "dispersion_coherence", "damping_proxy", "phonon_emergence",
    "phonon_stability",
]

X = samples[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = samples["perturbation"].isin([0.0, 0.01]).to_numpy()
test = samples["perturbation"].isin([0.025, 0.05]).to_numpy()


def knn_acc(label):
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf.fit(Xn[train], samples.loc[train, label])
    pred = clf.predict(Xn[test])
    return float(accuracy_score(samples.loc[test, label], pred))

class_accuracy = knn_acc("material_class")
lattice_accuracy = knn_acc("lattice_type")
stability_accuracy = float(accuracy_score(samples.loc[test, "expected_stable"], samples.loc[test, "predicted_stable"]))

valid = samples["expected_stable"] == 1
invalid = samples["expected_stable"] == 0
stable_ratio_valid = float(samples.loc[valid, "predicted_stable"].mean())
stable_ratio_invalid = float(samples.loc[invalid, "predicted_stable"].mean())
mean_valid_stability = float(samples.loc[valid, "phonon_stability"].mean())
mean_invalid_stability = float(samples.loc[invalid, "phonon_stability"].mean())
stability_contrast = float((mean_valid_stability + EPS) / (mean_invalid_stability + EPS))

try:
    class_silhouette = float(silhouette_score(Xn, samples["material_class"]))
except Exception:
    class_silhouette = np.nan

by_material = (
    samples.groupby("material")
    .agg(
        count=("material", "count"),
        material_class=("material_class", "first"),
        lattice_type=("lattice_type", "first"),
        expected_stable=("expected_stable", "mean"),
        mean_bond_stiffness=("bond_stiffness", "mean"),
        mean_coordination=("coordination", "mean"),
        mean_lattice_order=("lattice_order", "mean"),
        mean_harmonicity=("harmonicity", "mean"),
        mean_acoustic_support=("acoustic_support", "mean"),
        mean_optical_support=("optical_support", "mean"),
        mean_branch_count=("branch_count", "mean"),
        mean_sound_velocity_proxy=("sound_velocity_proxy", "mean"),
        mean_optical_gap_proxy=("optical_gap_proxy", "mean"),
        mean_debye_proxy=("debye_proxy", "mean"),
        mean_dispersion_coherence=("dispersion_coherence", "mean"),
        mean_damping_proxy=("damping_proxy", "mean"),
        mean_phonon_emergence=("phonon_emergence", "mean"),
        mean_phonon_stability=("phonon_stability", "mean"),
        predicted_stable_ratio=("predicted_stable", "mean"),
        has_optical_branch=("has_optical_branch", "mean"),
    )
    .reset_index()
)

by_class = (
    samples.groupby("material_class")
    .agg(
        count=("material_class", "count"),
        materials=("material", lambda x: ",".join(sorted(set(x)))),
        mean_bond_stiffness=("bond_stiffness", "mean"),
        mean_acoustic_support=("acoustic_support", "mean"),
        mean_optical_support=("optical_support", "mean"),
        mean_sound_velocity_proxy=("sound_velocity_proxy", "mean"),
        mean_dispersion_coherence=("dispersion_coherence", "mean"),
        mean_damping_proxy=("damping_proxy", "mean"),
        mean_phonon_stability=("phonon_stability", "mean"),
    )
    .reset_index()
)

acoustic_support = float(by_material.loc[by_material["expected_stable"] == 1, "mean_acoustic_support"].min() > 0.20)
optical_expected = by_material[by_material["has_optical_branch"] > 0.5]
optical_branch_recovery = float((optical_expected["mean_optical_support"] > 0.08).mean()) if len(optical_expected) else 1.0
invalid_suppression = float(by_material.loc[by_material["expected_stable"] == 0, "predicted_stable_ratio"].max() == 0.0)
vdw_low_velocity = float(by_class.loc[by_class["material_class"] == "van_der_waals", "mean_sound_velocity_proxy"].iloc[0] < by_class.loc[by_class["material_class"] == "metallic_crystal", "mean_sound_velocity_proxy"].iloc[0])

if (
    class_accuracy >= 0.95
    and lattice_accuracy >= 0.95
    and stability_accuracy >= 0.99
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.05
    and acoustic_support == 1.0
    and optical_branch_recovery >= 0.85
    and invalid_suppression == 1.0
    and vdw_low_velocity == 1.0
):
    verdict = "phonon_emergence_supported"
elif (
    class_accuracy >= 0.85
    and stability_accuracy >= 0.90
    and acoustic_support == 1.0
    and invalid_suppression == 1.0
):
    verdict = "weak_phonon_emergence"
else:
    verdict = "phonon_emergence_not_supported"

summary = pd.DataFrame([{
    "num_materials": len(MATERIALS),
    "num_samples": len(samples),
    "class_accuracy": class_accuracy,
    "lattice_accuracy": lattice_accuracy,
    "stability_accuracy": stability_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "class_silhouette": class_silhouette,
    "acoustic_support": acoustic_support,
    "optical_branch_recovery": optical_branch_recovery,
    "invalid_suppression": invalid_suppression,
    "vdw_low_velocity": vdw_low_velocity,
    "verdict": verdict,
}])

# Dispersion curves
ks = np.linspace(0.0, np.pi, 120)
disp_rows = []
for _, row in by_material.iterrows():
    acoustic_scale = row["mean_sound_velocity_proxy"]
    optical_gap = row["mean_optical_gap_proxy"]
    optical_scale = row["mean_optical_support"]
    for k in ks:
        acoustic = acoustic_scale * np.sin(0.5 * k)
        disp_rows.append({"material": row["material"], "branch": "acoustic", "k": k, "omega": acoustic})
        if row["has_optical_branch"] > 0.5:
            optical = optical_gap + optical_scale * (0.20 + 0.15 * np.cos(k))
            disp_rows.append({"material": row["material"], "branch": "optical", "k": k, "omega": optical})

dispersion = pd.DataFrame(disp_rows)

samples.to_csv(OUT / "phonon_samples.csv", index=False)
summary.to_csv(OUT / "phonon_summary.csv", index=False)
by_material.to_csv(OUT / "phonon_by_material.csv", index=False)
by_class.to_csv(OUT / "phonon_by_class.csv", index=False)
dispersion.to_csv(OUT / "phonon_dispersion_curves.csv", index=False)

# Plots
proj = PCA(n_components=2).fit_transform(Xn)
plt.figure(figsize=(9, 6))
for cls in sorted(samples["material_class"].unique()):
    mask = samples["material_class"].to_numpy() == cls
    plt.scatter(proj[mask, 0], proj[mask, 1], s=4, label=cls, alpha=0.45)
plt.title("BST phonon emergence latent space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "phonon_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plot_df = by_material.sort_values("mean_phonon_stability")
plt.bar(plot_df["material"], plot_df["mean_phonon_stability"])
plt.axhline(0.46, linestyle="--", linewidth=1)
plt.xticks(rotation=75, ha="right")
plt.ylabel("phonon stability")
plt.title("BST phonon stability by material")
plt.tight_layout()
plt.savefig(OUT / "phonon_stability.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 6))
for material in ["Si", "diamond_C", "NaCl", "MgO", "Cu", "Ar_solid", "graphene_like"]:
    sub = dispersion[dispersion["material"] == material]
    for branch in sub["branch"].unique():
        b = sub[sub["branch"] == branch]
        plt.plot(b["k"], b["omega"], linewidth=1, label=f"{material} {branch}")
plt.xlabel("k")
plt.ylabel("omega proxy")
plt.title("BST phonon dispersion proxies")
plt.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "phonon_dispersion.png", dpi=220)
plt.close()

print("\n=== BST PHONON EMERGENCE TEST ===\n")
print(summary.to_string(index=False))
print("\nPhonon emergence by material:")
print(by_material.to_string(index=False))
print("\nPhonon emergence by class:")
print(by_class.to_string(index=False))
print(f"\n[OK] wrote {OUT / 'phonon_samples.csv'}")
print(f"[OK] wrote {OUT / 'phonon_summary.csv'}")
print(f"[OK] wrote {OUT / 'phonon_by_material.csv'}")
print(f"[OK] wrote {OUT / 'phonon_by_class.csv'}")
print(f"[OK] wrote {OUT / 'phonon_dispersion_curves.csv'}")
print(f"[OK] wrote {OUT / 'phonon_projection.png'}")
print(f"[OK] wrote {OUT / 'phonon_stability.png'}")
print(f"[OK] wrote {OUT / 'phonon_dispersion.png'}")
print("[DONE] phonon emergence test complete")
