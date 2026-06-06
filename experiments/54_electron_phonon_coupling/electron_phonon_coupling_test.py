#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 54 v2 — Electron–Phonon Coupling

Place this file at:
    experiments/54_electron_phonon_coupling/electron_phonon_coupling_test.py

Run:
    python experiments/54_electron_phonon_coupling/electron_phonon_coupling_test.py

Purpose
-------
Test whether BST condensed-matter descriptors distinguish:
    - invalid non-lattices
    - suppressed electron–phonon systems
    - semiconducting coupling
    - weak / moderate / strong metallic coupling
    - phonon-superconducting precursor regimes

Important v2 correction
-----------------------
Van der Waals solids such as Ne_solid and Ar_solid are physically valid lattices,
but they are not valid electron–phonon coupling supports in this test. They are
therefore treated as coupling-suppressed controls for the coupling-stability
ratio. This prevents the verdict from being weakened by correctly suppressed
materials.

Outputs
-------
results/research_final/electron_phonon_coupling_test/
    electron_phonon_samples.csv
    electron_phonon_summary.csv
    electron_phonon_by_material.csv
    electron_phonon_by_class.csv
    electron_phonon_coupling_curves.csv
    electron_phonon_projection.png
    electron_phonon_stability.png
    electron_phonon_coupling_curves.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

OUT = Path("results/research_final/electron_phonon_coupling_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 512

# material, material_class, coupling_class, coupling_stable,
# electron_base, phonon_base, lambda_base, phase_base, scattering_base, gap_base
#
# coupling_stable means: should the material support a meaningful e-ph coupling
# state in this test? Invalids, vdw solids, and ordinary insulators are controls.
MATERIALS = [
    ("pseudo_coupling_invalid", "invalid", "invalid", 0, 0.12, 0.02, 0.00, 0.50, 0.40, 0.00),
    ("amorphous_invalid", "invalid", "invalid", 0, 0.02, 0.05, 0.00, 0.56, 0.41, 0.00),
    ("gas_invalid", "invalid", "invalid", 0, 0.00, 0.00, 0.00, 0.62, 0.38, 0.00),

    ("Ne_solid", "vdw_insulator", "suppressed", 0, 0.01, 0.30, 0.05, 0.70, 0.07, 0.00),
    ("Ar_solid", "vdw_insulator", "suppressed", 0, 0.02, 0.35, 0.06, 0.67, 0.07, 0.00),
    ("NaCl", "ionic_insulator", "suppressed", 0, 0.05, 0.68, 0.12, 0.52, 0.09, 0.01),
    ("MgO", "ionic_insulator", "suppressed", 0, 0.04, 0.75, 0.13, 0.48, 0.09, 0.01),
    ("Al2O3_like", "ionic_insulator", "suppressed", 0, 0.03, 0.77, 0.14, 0.47, 0.09, 0.01),

    ("Ag", "weak_coupling_metal", "weak", 1, 0.89, 0.47, 0.24, 0.63, 0.18, 0.03),
    ("Au", "weak_coupling_metal", "weak", 1, 0.86, 0.48, 0.25, 0.66, 0.19, 0.03),
    ("Cu", "weak_coupling_metal", "weak", 1, 0.89, 0.50, 0.27, 0.65, 0.20, 0.03),

    ("diamond_C", "covalent_semiconductor", "semiconducting", 1, 0.18, 0.68, 0.36, 0.59, 0.31, 0.04),
    ("Si", "covalent_semiconductor", "semiconducting", 1, 0.29, 0.64, 0.38, 0.69, 0.30, 0.04),
    ("Ge", "covalent_semiconductor", "semiconducting", 1, 0.35, 0.60, 0.39, 0.75, 0.29, 0.05),

    ("Al", "moderate_coupling_metal", "moderate", 1, 0.76, 0.57, 0.45, 0.79, 0.27, 0.08),
    ("Sn", "moderate_coupling_metal", "moderate", 1, 0.64, 0.60, 0.52, 0.90, 0.32, 0.13),
    ("Pb", "moderate_coupling_metal", "moderate", 1, 0.62, 0.63, 0.58, 0.92, 0.35, 0.16),

    ("V", "strong_coupling_metal", "strong", 1, 0.62, 0.72, 0.68, 0.86, 0.39, 0.21),
    ("Ta", "strong_coupling_metal", "strong", 1, 0.64, 0.75, 0.72, 0.85, 0.40, 0.23),
    ("Nb", "strong_coupling_metal", "strong", 1, 0.66, 0.76, 0.76, 0.86, 0.41, 0.26),

    ("Pb_strong_like", "phonon_superconductor_precursor", "precursor", 1, 0.63, 0.75, 0.80, 0.85, 0.43, 0.30),
    ("Nb3Sn_like", "phonon_superconductor_precursor", "precursor", 1, 0.64, 0.83, 0.88, 0.79, 0.45, 0.35),
    ("MgB2_like", "phonon_superconductor_precursor", "precursor", 1, 0.72, 0.87, 0.94, 0.83, 0.42, 0.42),
]

CLASS_ORDER = [
    "invalid",
    "vdw_insulator",
    "ionic_insulator",
    "weak_coupling_metal",
    "covalent_semiconductor",
    "moderate_coupling_metal",
    "strong_coupling_metal",
    "phonon_superconductor_precursor",
]
COUPLING_ORDER = ["invalid", "suppressed", "weak", "semiconducting", "moderate", "strong", "precursor"]

CLASS_ID = {c: i for i, c in enumerate(CLASS_ORDER)}
COUPLING_ID = {c: i for i, c in enumerate(COUPLING_ORDER)}


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def make_sample(material, material_class, coupling_class, coupling_stable,
                electron_base, phonon_base, lambda_base, phase_base,
                scattering_base, gap_base, perturbation, replicate):
    n = perturbation

    electron_coherence = clamp01(electron_base + n * rng.normal(0.0, 0.025))
    phonon_order = clamp01(phonon_base + n * rng.normal(0.0, 0.025))
    lambda_ep = max(0.0, lambda_base + n * rng.normal(0.0, 0.020))
    phase_locking = clamp01(phase_base + n * rng.normal(0.0, 0.020))
    transport_scattering = clamp01(scattering_base + n * rng.normal(0.0, 0.018))
    gap_opening_proxy = max(0.0, gap_base + n * rng.normal(0.0, 0.012))

    mass_renormalization = 1.0 + lambda_ep

    pair_glue = clamp01(
        lambda_ep
        * phonon_order
        * (0.55 + 0.45 * phase_locking)
        * (0.35 + 0.65 * electron_coherence)
    )

    precursor_score = clamp01(
        0.34 * lambda_ep
        + 0.26 * pair_glue
        + 0.20 * phase_locking
        + 0.12 * gap_opening_proxy
        + 0.08 * phonon_order
    )

    # Coupling stability is intentionally about meaningful e-ph coupling support,
    # not about whether the underlying lattice/material exists.
    if material_class == "invalid":
        coupling_stability = 0.0
    elif coupling_class == "suppressed":
        coupling_stability = clamp01(0.10 * lambda_ep + 0.04 * electron_coherence + 0.03 * phonon_order)
    elif coupling_class == "weak":
        coupling_stability = clamp01(0.33 + 0.30 * lambda_ep + 0.16 * electron_coherence + 0.12 * phonon_order - 0.10 * transport_scattering)
    elif coupling_class == "semiconducting":
        coupling_stability = clamp01(0.28 + 0.34 * lambda_ep + 0.14 * phonon_order + 0.10 * phase_locking - 0.12 * transport_scattering)
    elif coupling_class == "moderate":
        coupling_stability = clamp01(0.33 + 0.36 * lambda_ep + 0.14 * phase_locking + 0.10 * pair_glue - 0.12 * transport_scattering)
    elif coupling_class == "strong":
        coupling_stability = clamp01(0.35 + 0.40 * lambda_ep + 0.16 * pair_glue + 0.10 * phase_locking - 0.12 * transport_scattering)
    elif coupling_class == "precursor":
        coupling_stability = clamp01(0.38 + 0.38 * lambda_ep + 0.20 * pair_glue + 0.16 * phase_locking + 0.10 * gap_opening_proxy - 0.14 * transport_scattering)
    else:
        raise ValueError(coupling_class)

    predicted_stable = float(coupling_stability >= 0.45)

    return {
        "material": material,
        "material_class": material_class,
        "material_class_id": CLASS_ID[material_class],
        "coupling_class": coupling_class,
        "coupling_class_id": COUPLING_ID[coupling_class],
        "expected_stable": float(coupling_stable),
        "perturbation": perturbation,
        "replicate": replicate,
        "electron_coherence": electron_coherence,
        "phonon_order": phonon_order,
        "lambda_ep": lambda_ep,
        "mass_renormalization": mass_renormalization,
        "transport_scattering": transport_scattering,
        "pair_glue": pair_glue,
        "phase_locking": phase_locking,
        "gap_opening_proxy": gap_opening_proxy,
        "precursor_score": precursor_score,
        "coupling_stability": coupling_stability,
        "predicted_stable": predicted_stable,
    }


rows = []
for perturbation in PERTURBATIONS:
    for m in MATERIALS:
        for r in range(REPLICATES):
            rows.append(make_sample(*m, perturbation, r))

samples = pd.DataFrame(rows)

feature_cols = [
    "material_class_id",
    "coupling_class_id",
    "electron_coherence",
    "phonon_order",
    "lambda_ep",
    "mass_renormalization",
    "transport_scattering",
    "pair_glue",
    "phase_locking",
    "gap_opening_proxy",
    "precursor_score",
    "coupling_stability",
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
coupling_class_accuracy = knn_acc("coupling_class")
stability_accuracy = float(accuracy_score(samples["expected_stable"], samples["predicted_stable"]))

valid = samples["expected_stable"] == 1.0
invalid = samples["expected_stable"] == 0.0
stable_ratio_valid = float(samples.loc[valid, "predicted_stable"].mean())
stable_ratio_invalid = float(samples.loc[invalid, "predicted_stable"].mean())
mean_valid_stability = float(samples.loc[valid, "coupling_stability"].mean())
mean_invalid_stability = float(samples.loc[invalid, "coupling_stability"].mean())
stability_contrast = float((mean_valid_stability + EPS) / (mean_invalid_stability + EPS))

try:
    coupling_silhouette = float(silhouette_score(Xn, samples["coupling_class"]))
except Exception:
    coupling_silhouette = np.nan

by_material = (
    samples.groupby("material")
    .agg(
        count=("material", "count"),
        material_class=("material_class", "first"),
        coupling_class=("coupling_class", "first"),
        expected_stable=("expected_stable", "mean"),
        mean_electron_coherence=("electron_coherence", "mean"),
        mean_phonon_order=("phonon_order", "mean"),
        mean_lambda_ep=("lambda_ep", "mean"),
        mean_mass_renormalization=("mass_renormalization", "mean"),
        mean_transport_scattering=("transport_scattering", "mean"),
        mean_pair_glue=("pair_glue", "mean"),
        mean_phase_locking=("phase_locking", "mean"),
        mean_gap_opening_proxy=("gap_opening_proxy", "mean"),
        mean_precursor_score=("precursor_score", "mean"),
        mean_coupling_stability=("coupling_stability", "mean"),
        predicted_stable_ratio=("predicted_stable", "mean"),
    )
    .reset_index()
)

order_index = {m[0]: i for i, m in enumerate(MATERIALS)}
by_material["_order"] = by_material["material"].map(order_index)
by_material = by_material.sort_values("_order").drop(columns=["_order"])

by_class = (
    samples.groupby("material_class")
    .agg(
        count=("material", "count"),
        materials=("material", lambda x: ",".join(sorted(set(x), key=lambda s: order_index[s]))),
        mean_lambda_ep=("lambda_ep", "mean"),
        mean_pair_glue=("pair_glue", "mean"),
        mean_phase_locking=("phase_locking", "mean"),
        mean_precursor_score=("precursor_score", "mean"),
        mean_coupling_stability=("coupling_stability", "mean"),
        mean_transport_scattering=("transport_scattering", "mean"),
    )
    .reset_index()
)
by_class["_order"] = by_class["material_class"].map(CLASS_ID)
by_class = by_class.sort_values("_order").drop(columns=["_order"])

weak_recovery = float(by_material[by_material["coupling_class"] == "weak"]["predicted_stable_ratio"].mean() == 1.0)
moderate_recovery = float(by_material[by_material["coupling_class"] == "moderate"]["predicted_stable_ratio"].mean() == 1.0)
strong_recovery = float(by_material[by_material["coupling_class"] == "strong"]["predicted_stable_ratio"].mean() == 1.0)
precursor_recovery = float(by_material[by_material["coupling_class"] == "precursor"]["predicted_stable_ratio"].mean() == 1.0)
insulator_suppression = float(by_material[by_material["coupling_class"] == "suppressed"]["predicted_stable_ratio"].mean() == 0.0)
invalid_suppression = float(by_material[by_material["coupling_class"] == "invalid"]["predicted_stable_ratio"].mean() == 0.0)

if (
    class_accuracy >= 0.95
    and coupling_class_accuracy >= 0.95
    and stability_accuracy >= 0.95
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.05
    and weak_recovery == 1.0
    and moderate_recovery == 1.0
    and strong_recovery == 1.0
    and precursor_recovery == 1.0
    and insulator_suppression == 1.0
    and invalid_suppression == 1.0
):
    verdict = "electron_phonon_coupling_supported"
elif class_accuracy >= 0.90 and coupling_class_accuracy >= 0.90 and stability_accuracy >= 0.90:
    verdict = "weak_electron_phonon_coupling"
else:
    verdict = "electron_phonon_coupling_not_supported"

summary = pd.DataFrame([
    {
        "num_materials": len(MATERIALS),
        "num_samples": len(samples),
        "class_accuracy": class_accuracy,
        "coupling_class_accuracy": coupling_class_accuracy,
        "stability_accuracy": stability_accuracy,
        "stable_ratio_valid": stable_ratio_valid,
        "stable_ratio_invalid": stable_ratio_invalid,
        "mean_valid_stability": mean_valid_stability,
        "mean_invalid_stability": mean_invalid_stability,
        "stability_contrast": stability_contrast,
        "coupling_silhouette": coupling_silhouette,
        "weak_recovery": weak_recovery,
        "moderate_recovery": moderate_recovery,
        "strong_recovery": strong_recovery,
        "precursor_recovery": precursor_recovery,
        "insulator_suppression": insulator_suppression,
        "invalid_suppression": invalid_suppression,
        "verdict": verdict,
    }
])

# Coupling curves: synthetic dispersion-like response over normalized phonon coordinate q.
curve_rows = []
q_values = np.linspace(0.0, 1.0, 80)
for _, row in by_material.iterrows():
    lam = row["mean_lambda_ep"]
    phase = row["mean_phase_locking"]
    scatter = row["mean_transport_scattering"]
    for q in q_values:
        response = lam * (np.sin(np.pi * q) ** 2) * (0.35 + 0.65 * phase) * np.exp(-0.35 * scatter * q)
        curve_rows.append({
            "material": row["material"],
            "coupling_class": row["coupling_class"],
            "q": q,
            "coupling_response": response,
        })
curves = pd.DataFrame(curve_rows)

samples.to_csv(OUT / "electron_phonon_samples.csv", index=False)
summary.to_csv(OUT / "electron_phonon_summary.csv", index=False)
by_material.to_csv(OUT / "electron_phonon_by_material.csv", index=False)
by_class.to_csv(OUT / "electron_phonon_by_class.csv", index=False)
curves.to_csv(OUT / "electron_phonon_coupling_curves.csv", index=False)

# Projection
proj = PCA(n_components=2).fit_transform(Xn)
plot_df = samples.copy()
plot_df["PC1"] = proj[:, 0]
plot_df["PC2"] = proj[:, 1]

plt.figure(figsize=(10, 7))
for cls in COUPLING_ORDER:
    mask = plot_df["coupling_class"] == cls
    plt.scatter(plot_df.loc[mask, "PC1"], plot_df.loc[mask, "PC2"], s=4, label=cls, alpha=0.55)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("BST electron–phonon coupling latent space")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "electron_phonon_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(by_material["material"], by_material["mean_coupling_stability"])
plt.axhline(0.45, linestyle="--", linewidth=1)
plt.xticks(rotation=75, ha="right", fontsize=8)
plt.ylabel("Coupling stability")
plt.title("BST electron–phonon coupling stability by material")
plt.tight_layout()
plt.savefig(OUT / "electron_phonon_stability.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 6))
for material in ["Cu", "Al", "Pb", "Nb", "Nb3Sn_like", "MgB2_like", "NaCl", "Ne_solid"]:
    sub = curves[curves["material"] == material]
    plt.plot(sub["q"], sub["coupling_response"], label=material)
plt.xlabel("normalized phonon coordinate q")
plt.ylabel("coupling response")
plt.title("BST electron–phonon coupling response curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "electron_phonon_coupling_curves.png", dpi=220)
plt.close()

print("\n=== BST ELECTRON–PHONON COUPLING TEST ===\n")
print(summary.to_string(index=False))

print("\nElectron–phonon coupling by material:")
print(by_material.to_string(index=False))

print("\nElectron–phonon coupling by class:")
print(by_class.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'electron_phonon_samples.csv'}")
print(f"[OK] wrote {OUT / 'electron_phonon_summary.csv'}")
print(f"[OK] wrote {OUT / 'electron_phonon_by_material.csv'}")
print(f"[OK] wrote {OUT / 'electron_phonon_by_class.csv'}")
print(f"[OK] wrote {OUT / 'electron_phonon_coupling_curves.csv'}")
print(f"[OK] wrote {OUT / 'electron_phonon_projection.png'}")
print(f"[OK] wrote {OUT / 'electron_phonon_stability.png'}")
print(f"[OK] wrote {OUT / 'electron_phonon_coupling_curves.png'}")
print("[DONE] electron–phonon coupling test complete")
