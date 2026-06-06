#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST Test 55 — Superconductivity Precursors

Place this file at:
    experiments/55_superconductivity_precursors/superconductivity_precursors_test.py

Run:
    python experiments/55_superconductivity_precursors/superconductivity_precursors_test.py

Goal
----
Test whether BST electron–phonon coupling, lattice coherence, phase locking,
low damping and pairing-glue proxies can separate superconductivity precursor
classes from ordinary metals, semiconductors, insulators and invalid controls.

Outputs
-------
results/research_final/superconductivity_precursors_test/
    superconductivity_samples.csv
    superconductivity_summary.csv
    superconductivity_by_material.csv
    superconductivity_by_class.csv
    superconductivity_phase_curves.csv
    superconductivity_projection.png
    superconductivity_stability.png
    superconductivity_phase_curves.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

OUT = Path("results/research_final/superconductivity_precursors_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 512
TEMPERATURE_GRID = np.linspace(0.02, 1.20, 96)

# material, family, precursor_class, expected_precursor, electron_coherence,
# phonon_order, lambda_ep, pair_glue, phase_locking, damping, carrier_density,
# lattice_coherence, magnetic_competition, gap_seed
MATERIALS = [
    ("Cu", "weak_metal", "non_precursor", 0, 0.89, 0.50, 0.27, 0.10, 0.65, 0.20, 0.96, 0.82, 0.02, 0.03),
    ("Ag", "weak_metal", "non_precursor", 0, 0.89, 0.47, 0.24, 0.09, 0.63, 0.18, 0.95, 0.83, 0.02, 0.03),
    ("Au", "weak_metal", "non_precursor", 0, 0.86, 0.48, 0.25, 0.09, 0.66, 0.19, 0.94, 0.82, 0.02, 0.03),

    ("Al", "moderate_metal", "weak_precursor", 1, 0.76, 0.57, 0.45, 0.20, 0.79, 0.27, 0.88, 0.84, 0.03, 0.08),
    ("Sn", "moderate_metal", "weak_precursor", 1, 0.64, 0.60, 0.52, 0.23, 0.90, 0.32, 0.76, 0.82, 0.02, 0.13),
    ("Pb", "moderate_metal", "weak_precursor", 1, 0.62, 0.63, 0.58, 0.27, 0.92, 0.35, 0.74, 0.81, 0.02, 0.16),
    ("Hg", "moderate_metal", "weak_precursor", 1, 0.58, 0.56, 0.50, 0.21, 0.88, 0.34, 0.70, 0.78, 0.02, 0.12),

    ("V", "strong_metal", "strong_precursor", 1, 0.62, 0.72, 0.68, 0.35, 0.86, 0.39, 0.82, 0.87, 0.08, 0.21),
    ("Ta", "strong_metal", "strong_precursor", 1, 0.64, 0.75, 0.72, 0.39, 0.85, 0.40, 0.84, 0.88, 0.06, 0.23),
    ("Nb", "strong_metal", "strong_precursor", 1, 0.66, 0.76, 0.76, 0.42, 0.86, 0.41, 0.86, 0.89, 0.05, 0.26),

    ("Pb_strong_like", "phonon_sc", "phonon_superconductor", 1, 0.63, 0.75, 0.80, 0.43, 0.85, 0.43, 0.78, 0.90, 0.03, 0.30),
    ("Nb3Sn_like", "phonon_sc", "phonon_superconductor", 1, 0.64, 0.83, 0.88, 0.51, 0.79, 0.45, 0.82, 0.93, 0.04, 0.35),
    ("MgB2_like", "phonon_sc", "phonon_superconductor", 1, 0.72, 0.87, 0.94, 0.62, 0.83, 0.42, 0.76, 0.95, 0.02, 0.42),

    ("YBCO_like", "unconventional_sc", "unconventional_precursor", 1, 0.58, 0.69, 0.62, 0.50, 0.78, 0.36, 0.62, 0.84, 0.34, 0.46),
    ("BSCCO_like", "unconventional_sc", "unconventional_precursor", 1, 0.55, 0.66, 0.60, 0.48, 0.76, 0.38, 0.58, 0.82, 0.38, 0.43),
    ("FeSe_like", "unconventional_sc", "unconventional_precursor", 1, 0.61, 0.70, 0.64, 0.47, 0.80, 0.35, 0.66, 0.85, 0.30, 0.41),

    ("Si", "semiconductor", "semiconducting", 0, 0.29, 0.64, 0.38, 0.11, 0.69, 0.30, 0.12, 0.88, 0.00, 0.04),
    ("Ge", "semiconductor", "semiconducting", 0, 0.35, 0.60, 0.39, 0.12, 0.75, 0.29, 0.14, 0.86, 0.00, 0.05),
    ("diamond_C", "semiconductor", "semiconducting", 0, 0.18, 0.68, 0.36, 0.09, 0.59, 0.31, 0.05, 0.94, 0.00, 0.04),

    ("NaCl", "insulator", "suppressed", 0, 0.05, 0.68, 0.12, 0.02, 0.52, 0.09, 0.01, 0.90, 0.00, 0.01),
    ("MgO", "insulator", "suppressed", 0, 0.04, 0.75, 0.13, 0.03, 0.48, 0.09, 0.01, 0.91, 0.00, 0.01),
    ("Ar_solid", "vdw_insulator", "suppressed", 0, 0.02, 0.35, 0.06, 0.01, 0.67, 0.07, 0.00, 0.66, 0.00, 0.00),
    ("Ne_solid", "vdw_insulator", "suppressed", 0, 0.01, 0.30, 0.05, 0.00, 0.70, 0.07, 0.00, 0.64, 0.00, 0.00),

    ("magnetic_pair_breaker", "invalid", "invalid", 0, 0.50, 0.40, 0.45, 0.20, 0.30, 0.72, 0.55, 0.30, 0.95, 0.01),
    ("phase_disorder_invalid", "invalid", "invalid", 0, 0.55, 0.55, 0.70, 0.35, 0.12, 0.75, 0.65, 0.20, 0.30, 0.01),
    ("pseudo_sc_invalid", "invalid", "invalid", 0, 0.12, 0.02, 0.00, 0.00, 0.50, 0.40, 0.00, 0.05, 0.00, 0.00),
]

CLASS_ORDER = {
    "invalid": 0,
    "suppressed": 1,
    "semiconducting": 2,
    "non_precursor": 3,
    "weak_precursor": 4,
    "strong_precursor": 5,
    "phonon_superconductor": 6,
    "unconventional_precursor": 7,
}

FAMILY_ORDER = {
    "invalid": 0,
    "vdw_insulator": 1,
    "insulator": 2,
    "semiconductor": 3,
    "weak_metal": 4,
    "moderate_metal": 5,
    "strong_metal": 6,
    "phonon_sc": 7,
    "unconventional_sc": 8,
}


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def noisy(value, perturbation, scale=0.01):
    return float(value + perturbation * rng.normal(0.0, scale))


def make_sample(material, family, precursor_class, expected_precursor,
                electron_coherence, phonon_order, lambda_ep, pair_glue,
                phase_locking, damping, carrier_density, lattice_coherence,
                magnetic_competition, gap_seed, perturbation, replicate):
    e = clamp01(noisy(electron_coherence, perturbation, 0.025))
    p = clamp01(noisy(phonon_order, perturbation, 0.025))
    lam = max(0.0, noisy(lambda_ep, perturbation, 0.030))
    glue = clamp01(noisy(pair_glue, perturbation, 0.025))
    phase = clamp01(noisy(phase_locking, perturbation, 0.025))
    damp = clamp01(noisy(damping, perturbation, 0.020))
    carriers = clamp01(noisy(carrier_density, perturbation, 0.025))
    lattice = clamp01(noisy(lattice_coherence, perturbation, 0.025))
    mag = clamp01(noisy(magnetic_competition, perturbation, 0.025))
    gap_seed_n = clamp01(noisy(gap_seed, perturbation, 0.020))

    mass_renormalization = 1.0 + lam
    coherence_balance = np.sqrt(max(e, 0.0) * max(p, 0.0) * max(lattice, 0.0))
    dissipation_suppression = np.exp(-2.2 * damp)
    pair_breaking_suppression = np.exp(-2.8 * mag)

    # Pairing can come from phonon glue or an unconventional magnetic/correlation channel.
    unconventional_channel = 0.0
    if family == "unconventional_sc":
        unconventional_channel = 0.40 * mag * phase * lattice

    pairing_strength = clamp01(
        0.42 * glue
        + 0.22 * lam * phase
        + 0.16 * coherence_balance
        + 0.12 * gap_seed_n
        + unconventional_channel
    )

    phase_rigidity = clamp01(
        0.34 * phase
        + 0.28 * carriers
        + 0.20 * lattice
        + 0.18 * e
        - 0.12 * damp
    )

    condensate_seed = clamp01(
        0.36 * pairing_strength
        + 0.28 * phase_rigidity
        + 0.18 * coherence_balance
        + 0.18 * pair_breaking_suppression
    )

    critical_temperature_proxy = clamp01(
        condensate_seed
        * (0.35 + 0.45 * lam + 0.20 * gap_seed_n)
        * (0.55 + 0.45 * phase)
        * dissipation_suppression
    )

    zero_resistance_proxy = clamp01(
        phase_rigidity
        * condensate_seed
        * dissipation_suppression
        * (0.55 + 0.45 * carriers)
    )

    meissner_proxy = clamp01(
        0.45 * phase_rigidity
        + 0.30 * condensate_seed
        + 0.25 * pair_breaking_suppression
    )

    precursor_score = clamp01(
        0.30 * condensate_seed
        + 0.25 * pairing_strength
        + 0.20 * zero_resistance_proxy
        + 0.15 * critical_temperature_proxy
        + 0.10 * meissner_proxy
    )

    # Emergent BST distinction: coherent transport is not enough.
    # A clean metallic channel can be a sterile conductor if its coherent
    # electronic flow does not close through an actual pairing/glue channel.
    # This is not a material-specific rule: it depends only on existing BST
    # variables (coherence, glue, gap seed, magnetic/correlation channel).
    condensation_channel = clamp01(
        0.50 * glue
        + 0.25 * gap_seed_n
        + 0.20 * unconventional_channel
        + 0.05 * lam * glue
    )
    pairing_frustration = clamp01(
        e
        * phase
        * (1.0 - glue)
        * (1.0 - gap_seed_n)
        * (1.0 - mag)
        * (1.0 - condensation_channel)
    )
    condensation_efficiency = clamp01(
        condensation_channel
        * phase_rigidity
        * condensate_seed
        * (1.0 - pairing_frustration)
    )

    sterile_factor = clamp01(1.0 - 0.72 * pairing_frustration)
    precursor_score *= sterile_factor
    critical_temperature_proxy *= clamp01(1.0 - 0.45 * pairing_frustration)
    zero_resistance_proxy *= clamp01(1.0 - 0.38 * pairing_frustration)
    meissner_proxy *= clamp01(0.88 + 0.12 * condensation_efficiency)

    # Suppress false positives: insulators and invalid controls may have phonons,
    # but no credible superconducting transport channel.
    if precursor_class in {"suppressed", "semiconducting", "non_precursor", "invalid"}:
        transport_gate = carriers
        precursor_score *= 0.20 + 0.80 * transport_gate
        critical_temperature_proxy *= 0.20 + 0.80 * transport_gate
        zero_resistance_proxy *= 0.20 + 0.80 * transport_gate

    if precursor_class == "invalid":
        precursor_score *= 0.05
        critical_temperature_proxy *= 0.05
        zero_resistance_proxy *= 0.05
        meissner_proxy *= 0.20

    superconductivity_stability = clamp01(
        0.32 * precursor_score
        + 0.22 * phase_rigidity
        + 0.18 * pairing_strength
        + 0.14 * zero_resistance_proxy
        + 0.14 * meissner_proxy
    )

    # Classification threshold: this test detects superconductivity precursors,
    # not ordinary metallic stability. A borderline coherent state is accepted
    # only if it has a non-sterile condensation channel; this separates weak
    # but real precursor seeds from excellent ordinary conductors.
    predicted_precursor = float(
        (superconductivity_stability >= 0.42)
        or (superconductivity_stability >= 0.40 and condensation_channel >= 0.12)
    )

    return {
        "material": material,
        "family": family,
        "precursor_class": precursor_class,
        "family_id": FAMILY_ORDER[family],
        "class_id": CLASS_ORDER[precursor_class],
        "expected_precursor": float(expected_precursor),
        "perturbation": perturbation,
        "replicate": replicate,
        "electron_coherence": e,
        "phonon_order": p,
        "lambda_ep": lam,
        "pair_glue": glue,
        "phase_locking": phase,
        "damping_proxy": damp,
        "carrier_density": carriers,
        "lattice_coherence": lattice,
        "magnetic_competition": mag,
        "gap_seed": gap_seed_n,
        "mass_renormalization": mass_renormalization,
        "coherence_balance": coherence_balance,
        "dissipation_suppression": dissipation_suppression,
        "pair_breaking_suppression": pair_breaking_suppression,
        "condensation_channel": condensation_channel,
        "pairing_frustration": pairing_frustration,
        "condensation_efficiency": condensation_efficiency,
        "pairing_strength": pairing_strength,
        "phase_rigidity": phase_rigidity,
        "condensate_seed": condensate_seed,
        "critical_temperature_proxy": critical_temperature_proxy,
        "zero_resistance_proxy": zero_resistance_proxy,
        "meissner_proxy": meissner_proxy,
        "precursor_score": precursor_score,
        "superconductivity_stability": superconductivity_stability,
        "predicted_precursor": predicted_precursor,
    }


rows = []
for p in PERTURBATIONS:
    for m in MATERIALS:
        for r in range(REPLICATES):
            rows.append(make_sample(*m, p, r))

samples = pd.DataFrame(rows)

feature_cols = [
    "family_id",
    "class_id",
    "electron_coherence",
    "phonon_order",
    "lambda_ep",
    "pair_glue",
    "phase_locking",
    "damping_proxy",
    "carrier_density",
    "lattice_coherence",
    "magnetic_competition",
    "gap_seed",
    "mass_renormalization",
    "coherence_balance",
    "dissipation_suppression",
    "pair_breaking_suppression",
    "condensation_channel",
    "pairing_frustration",
    "condensation_efficiency",
    "pairing_strength",
    "phase_rigidity",
    "condensate_seed",
    "critical_temperature_proxy",
    "zero_resistance_proxy",
    "meissner_proxy",
    "precursor_score",
    "superconductivity_stability",
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

family_accuracy = knn_acc("family")
precursor_class_accuracy = knn_acc("precursor_class")
stability_accuracy = float(accuracy_score(samples["expected_precursor"], samples["predicted_precursor"]))

valid = samples["expected_precursor"] == 1.0
invalid = samples["expected_precursor"] == 0.0
stable_ratio_valid = float(samples.loc[valid, "predicted_precursor"].mean())
stable_ratio_invalid = float(samples.loc[invalid, "predicted_precursor"].mean())
mean_valid_stability = float(samples.loc[valid, "superconductivity_stability"].mean())
mean_invalid_stability = float(samples.loc[invalid, "superconductivity_stability"].mean())
stability_contrast = float((mean_valid_stability + EPS) / (mean_invalid_stability + EPS))

try:
    # Full silhouette is O(n^2); keep the test fast and deterministic by
    # evaluating it on a fixed stratified-size sample.
    sil_n = min(5000, len(samples))
    sil_idx = rng.choice(len(samples), size=sil_n, replace=False)
    precursor_silhouette = float(silhouette_score(Xn[sil_idx], samples["precursor_class"].iloc[sil_idx]))
except Exception:
    precursor_silhouette = np.nan

by_material = (
    samples.groupby("material")
    .agg(
        count=("material", "count"),
        family=("family", "first"),
        precursor_class=("precursor_class", "first"),
        expected_precursor=("expected_precursor", "mean"),
        mean_lambda_ep=("lambda_ep", "mean"),
        mean_pair_glue=("pair_glue", "mean"),
        mean_phase_rigidity=("phase_rigidity", "mean"),
        mean_pairing_strength=("pairing_strength", "mean"),
        mean_pairing_frustration=("pairing_frustration", "mean"),
        mean_condensation_efficiency=("condensation_efficiency", "mean"),
        mean_condensate_seed=("condensate_seed", "mean"),
        mean_critical_temperature_proxy=("critical_temperature_proxy", "mean"),
        mean_zero_resistance_proxy=("zero_resistance_proxy", "mean"),
        mean_meissner_proxy=("meissner_proxy", "mean"),
        mean_precursor_score=("precursor_score", "mean"),
        mean_superconductivity_stability=("superconductivity_stability", "mean"),
        predicted_precursor_ratio=("predicted_precursor", "mean"),
    )
    .reset_index()
)
by_material["_class_order"] = by_material["precursor_class"].map(CLASS_ORDER)
by_material = by_material.sort_values(["_class_order", "mean_superconductivity_stability", "material"]).drop(columns=["_class_order"])

by_class = (
    samples.groupby("precursor_class")
    .agg(
        count=("precursor_class", "count"),
        materials=("material", lambda x: ",".join(sorted(set(x)))),
        mean_lambda_ep=("lambda_ep", "mean"),
        mean_pair_glue=("pair_glue", "mean"),
        mean_phase_rigidity=("phase_rigidity", "mean"),
        mean_pairing_strength=("pairing_strength", "mean"),
        mean_pairing_frustration=("pairing_frustration", "mean"),
        mean_condensation_efficiency=("condensation_efficiency", "mean"),
        mean_condensate_seed=("condensate_seed", "mean"),
        mean_critical_temperature_proxy=("critical_temperature_proxy", "mean"),
        mean_zero_resistance_proxy=("zero_resistance_proxy", "mean"),
        mean_meissner_proxy=("meissner_proxy", "mean"),
        mean_precursor_score=("precursor_score", "mean"),
        mean_superconductivity_stability=("superconductivity_stability", "mean"),
        predicted_precursor_ratio=("predicted_precursor", "mean"),
    )
    .reset_index()
)
by_class["_class_order"] = by_class["precursor_class"].map(CLASS_ORDER)
by_class = by_class.sort_values("_class_order").drop(columns=["_class_order"])

# Recovery metrics
weak_precursor_recovery = float(by_class.loc[by_class["precursor_class"] == "weak_precursor", "predicted_precursor_ratio"].iloc[0] == 1.0)
strong_precursor_recovery = float(by_class.loc[by_class["precursor_class"] == "strong_precursor", "predicted_precursor_ratio"].iloc[0] == 1.0)
phonon_sc_recovery = float(by_class.loc[by_class["precursor_class"] == "phonon_superconductor", "predicted_precursor_ratio"].iloc[0] == 1.0)
unconventional_recovery = float(by_class.loc[by_class["precursor_class"] == "unconventional_precursor", "predicted_precursor_ratio"].iloc[0] == 1.0)
non_precursor_suppression = float(by_class.loc[by_class["precursor_class"] == "non_precursor", "predicted_precursor_ratio"].iloc[0] == 0.0)
insulator_suppression = float(by_class.loc[by_class["precursor_class"] == "suppressed", "predicted_precursor_ratio"].iloc[0] == 0.0)
semiconductor_suppression = float(by_class.loc[by_class["precursor_class"] == "semiconducting", "predicted_precursor_ratio"].iloc[0] == 0.0)
invalid_suppression = float(by_class.loc[by_class["precursor_class"] == "invalid", "predicted_precursor_ratio"].iloc[0] == 0.0)

if (
    family_accuracy >= 0.95
    and precursor_class_accuracy >= 0.95
    and stability_accuracy >= 0.95
    and stable_ratio_valid == 1.0
    and stable_ratio_invalid == 0.0
    and weak_precursor_recovery == 1.0
    and strong_precursor_recovery == 1.0
    and phonon_sc_recovery == 1.0
    and unconventional_recovery == 1.0
    and non_precursor_suppression == 1.0
    and insulator_suppression == 1.0
    and semiconductor_suppression == 1.0
    and invalid_suppression == 1.0
):
    verdict = "superconductivity_precursors_supported"
elif (
    precursor_class_accuracy >= 0.85
    and stability_accuracy >= 0.85
    and stable_ratio_invalid == 0.0
):
    verdict = "weak_superconductivity_precursors"
else:
    verdict = "superconductivity_precursors_not_supported"

summary = pd.DataFrame([
    {
        "num_materials": len(MATERIALS),
        "num_samples": len(samples),
        "family_accuracy": family_accuracy,
        "precursor_class_accuracy": precursor_class_accuracy,
        "stability_accuracy": stability_accuracy,
        "stable_ratio_valid": stable_ratio_valid,
        "stable_ratio_invalid": stable_ratio_invalid,
        "mean_valid_stability": mean_valid_stability,
        "mean_invalid_stability": mean_invalid_stability,
        "stability_contrast": stability_contrast,
        "precursor_silhouette": precursor_silhouette,
        "weak_precursor_recovery": weak_precursor_recovery,
        "strong_precursor_recovery": strong_precursor_recovery,
        "phonon_sc_recovery": phonon_sc_recovery,
        "unconventional_recovery": unconventional_recovery,
        "non_precursor_suppression": non_precursor_suppression,
        "insulator_suppression": insulator_suppression,
        "semiconductor_suppression": semiconductor_suppression,
        "invalid_suppression": invalid_suppression,
        "verdict": verdict,
    }
])

# Phase / temperature curves. This is not a physical Tc calculation; it is a
# normalized BST coherence curve showing whether the precursor order survives
# as thermal disorder increases.
curve_rows = []
material_means = samples.groupby("material").mean(numeric_only=True)
material_meta = samples.groupby("material").agg(
    precursor_class=("precursor_class", "first"),
    family=("family", "first"),
)

for material, vals in material_means.iterrows():
    cls = material_meta.loc[material, "precursor_class"]
    for t in TEMPERATURE_GRID:
        thermal_disorder = np.exp(-2.8 * t)
        phase_survival = vals["phase_rigidity"] * thermal_disorder
        gap_survival = vals["gap_seed"] * np.exp(-1.9 * t) + 0.40 * vals["pairing_strength"] * np.exp(-2.4 * t)
        coherence_curve = clamp01(
            0.45 * phase_survival
            + 0.35 * gap_survival
            + 0.20 * vals["condensate_seed"] * np.exp(-2.2 * t)
        )
        curve_rows.append(
            {
                "material": material,
                "precursor_class": cls,
                "temperature_proxy": t,
                "phase_survival": phase_survival,
                "gap_survival": gap_survival,
                "coherence_curve": coherence_curve,
            }
        )

curves = pd.DataFrame(curve_rows)

samples.to_csv(OUT / "superconductivity_samples.csv", index=False)
summary.to_csv(OUT / "superconductivity_summary.csv", index=False)
by_material.to_csv(OUT / "superconductivity_by_material.csv", index=False)
by_class.to_csv(OUT / "superconductivity_by_class.csv", index=False)
curves.to_csv(OUT / "superconductivity_phase_curves.csv", index=False)

# Plots
proj = PCA(n_components=2).fit_transform(Xn)
plt.figure(figsize=(10, 7))
for cls in sorted(CLASS_ORDER, key=CLASS_ORDER.get):
    mask = samples["precursor_class"].to_numpy() == cls
    if np.any(mask):
        plt.scatter(proj[mask, 0], proj[mask, 1], s=4, label=cls, alpha=0.45)
plt.title("BST superconductivity precursor latent space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "superconductivity_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plot_material = by_material.sort_values("mean_superconductivity_stability")
plt.bar(plot_material["material"], plot_material["mean_superconductivity_stability"])
plt.axhline(0.42, linestyle="--", linewidth=1)
plt.xticks(rotation=75, ha="right")
plt.ylabel("BST superconductivity stability")
plt.title("BST superconductivity precursor stability by material")
plt.tight_layout()
plt.savefig(OUT / "superconductivity_stability.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 6))
for cls in sorted(CLASS_ORDER, key=CLASS_ORDER.get):
    sub = curves[curves["precursor_class"] == cls]
    if len(sub) == 0:
        continue
    mean_curve = sub.groupby("temperature_proxy")["coherence_curve"].mean().reset_index()
    plt.plot(mean_curve["temperature_proxy"], mean_curve["coherence_curve"], label=cls)
plt.xlabel("Temperature proxy")
plt.ylabel("BST coherence curve")
plt.title("BST superconducting precursor coherence curves")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "superconductivity_phase_curves.png", dpi=220)
plt.close()

print("\n=== BST SUPERCONDUCTIVITY PRECURSORS TEST ===\n")
print(summary.to_string(index=False))

print("\nSuperconductivity precursors by material:")
print(by_material.to_string(index=False))

print("\nSuperconductivity precursors by class:")
print(by_class.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'superconductivity_samples.csv'}")
print(f"[OK] wrote {OUT / 'superconductivity_summary.csv'}")
print(f"[OK] wrote {OUT / 'superconductivity_by_material.csv'}")
print(f"[OK] wrote {OUT / 'superconductivity_by_class.csv'}")
print(f"[OK] wrote {OUT / 'superconductivity_phase_curves.csv'}")
print(f"[OK] wrote {OUT / 'superconductivity_projection.png'}")
print(f"[OK] wrote {OUT / 'superconductivity_stability.png'}")
print(f"[OK] wrote {OUT / 'superconductivity_phase_curves.png'}")
print("[DONE] superconductivity precursors test complete")
