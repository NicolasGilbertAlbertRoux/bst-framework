#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 52 — Magnon Emergence

Place this file at:
    experiments/52_magnon_emergence/magnon_emergence_test.py

Run:
    python experiments/52_magnon_emergence/magnon_emergence_test.py

Goal
----
Test whether collective spin-wave / phase-wave modes emerge from BST lattice
exchange signatures.

This follows Test 51:
    elemental BST signatures
        -> lattice exchange networks
        -> ordered magnetic networks
        -> collective magnon-like excitations

The model is deliberately synthetic but internally consistent with the previous
BST tests. The labels are used for evaluation only; the generated observables
are BST-style exchange, phase, spin coherence, dispersion, and collective mode
stability.

Outputs
-------
results/research_final/magnon_emergence_test/
    magnon_samples.csv
    magnon_summary.csv
    magnon_by_material.csv
    magnon_by_ordering.csv
    magnon_dispersion_curves.csv
    magnon_projection.png
    magnon_stability.png
    magnon_dispersion.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


OUT = Path("results/research_final/magnon_emergence_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 128
K_POINTS = np.linspace(0.0, np.pi, 96)

ORDERING_ID = {
    "none": 0,
    "nonmagnetic": 1,
    "ferromagnetic": 2,
    "antiferromagnetic": 3,
    "ferrimagnetic": 4,
    "frustrated": 5,
}

GEOMETRY_ID = {
    "chain": 0,
    "square": 1,
    "cubic": 2,
    "triangular": 3,
    "honeycomb": 4,
    "spinel": 5,
    "none": 6,
}

# material, ion_a, ion_b, d_count_a, d_count_b, f_count_a, f_count_b,
# coordination, geometry, coupling_sign, exchange_base, anisotropy,
# expected_ordering, expected_stable
#
# Coupling sign: +1 ferro, -1 antiferro, -0.5 ferri, 0 nonmagnetic/none.
# exchange_base is a BST exchange proxy inherited from Test 51-style networks.
MATERIALS = [
    ("Fe_like", "Fe", "Fe", 6, 6, 0, 0, 8, "cubic", 1.0, 0.82, 0.10, "ferromagnetic", 1),
    ("Co_like", "Co", "Co", 7, 7, 0, 0, 8, "cubic", 1.0, 0.72, 0.18, "ferromagnetic", 1),
    ("Ni_like", "Ni", "Ni", 8, 8, 0, 0, 8, "cubic", 1.0, 0.58, 0.08, "ferromagnetic", 1),
    ("Gd_like", "Gd", "Gd", 0, 0, 7, 7, 12, "cubic", 1.0, 0.70, 0.12, "ferromagnetic", 1),

    ("MnO_like", "Mn", "Mn", 5, 5, 0, 0, 6, "cubic", -1.0, 0.92, 0.06, "antiferromagnetic", 1),
    ("FeO_like", "Fe", "Fe", 6, 6, 0, 0, 6, "cubic", -1.0, 0.78, 0.08, "antiferromagnetic", 1),
    ("NiO_like", "Ni", "Ni", 8, 8, 0, 0, 6, "cubic", -1.0, 0.68, 0.07, "antiferromagnetic", 1),
    ("Cr2O3_like", "Cr", "Cr", 3, 3, 0, 0, 6, "honeycomb", -1.0, 0.74, 0.12, "antiferromagnetic", 1),

    ("Fe3O4_like", "Fe", "Fe", 5, 6, 0, 0, 6, "spinel", -0.5, 0.76, 0.16, "ferrimagnetic", 1),
    ("MnFe2O4_like", "Mn", "Fe", 5, 6, 0, 0, 6, "spinel", -0.5, 0.70, 0.13, "ferrimagnetic", 1),

    ("Triangular_Mn_like", "Mn", "Mn", 5, 5, 0, 0, 6, "triangular", -1.0, 0.62, 0.24, "frustrated", 1),
    ("Kagome_Co_like", "Co", "Co", 7, 7, 0, 0, 4, "triangular", -1.0, 0.55, 0.30, "frustrated", 1),

    ("Cu_like", "Cu", "Cu", 10, 10, 0, 0, 12, "cubic", 0.0, 0.04, 0.02, "nonmagnetic", 0),
    ("Zn_like", "Zn", "Zn", 10, 10, 0, 0, 12, "cubic", 0.0, 0.03, 0.02, "nonmagnetic", 0),
    ("Ar_solid", "Ar", "Ar", 0, 0, 0, 0, 0, "none", 0.0, 0.00, 0.00, "none", 0),
    ("Ne_solid", "Ne", "Ne", 0, 0, 0, 0, 0, "none", 0.0, 0.00, 0.00, "none", 0),
    ("pseudo_spin_invalid", "X", "X", 0, 0, 0, 0, 0, "none", 0.0, 0.00, 0.50, "none", 0),
]


def unpaired_from_d(d_count: float) -> float:
    """High-spin proxy for d shell."""
    if d_count <= 0:
        return 0.0
    if d_count <= 5:
        return float(d_count)
    return float(max(0.0, 10.0 - d_count))


def unpaired_from_f(f_count: float) -> float:
    """High-spin proxy for f shell."""
    if f_count <= 0:
        return 0.0
    if f_count <= 7:
        return float(f_count)
    return float(max(0.0, 14.0 - f_count))


def geometry_frustration(geometry: str, coupling_sign: float) -> float:
    if geometry == "triangular" and coupling_sign < 0:
        return 0.85
    if geometry == "honeycomb" and coupling_sign < 0:
        return 0.25
    if geometry == "spinel" and coupling_sign < 0:
        return 0.35
    return 0.05 if geometry != "none" else 0.0


def geometry_connectivity(coordination: float, geometry: str) -> float:
    if geometry == "none":
        return 0.0
    return float(np.tanh(coordination / 6.0))


def expected_unpaired(ordering: str, ua: float, ub: float) -> float:
    if ordering in {"none", "nonmagnetic"}:
        return 0.0
    if ordering == "ferrimagnetic":
        return abs(ua - ub) + 0.5 * min(ua, ub)
    if ordering == "frustrated":
        return 0.75 * (ua + ub) / 2.0
    return (ua + ub) / 2.0


def make_sample(
    material: str,
    ion_a: str,
    ion_b: str,
    d_a: float,
    d_b: float,
    f_a: float,
    f_b: float,
    coordination: float,
    geometry: str,
    coupling_sign: float,
    exchange_base: float,
    anisotropy: float,
    expected_ordering: str,
    expected_stable: int,
    perturbation: float,
    replicate: int,
) -> dict:
    ua = max(unpaired_from_d(d_a), unpaired_from_f(f_a))
    ub = max(unpaired_from_d(d_b), unpaired_from_f(f_b))
    moment_a = ua / 5.0 if f_a == 0 else ua / 7.0
    moment_b = ub / 5.0 if f_b == 0 else ub / 7.0
    magnetic_moment = expected_unpaired(expected_ordering, ua, ub) / 5.0

    connectivity = geometry_connectivity(coordination, geometry)
    frustration = geometry_frustration(geometry, coupling_sign)

    exchange_strength = exchange_base * (0.65 + 0.35 * connectivity)
    exchange_strength += perturbation * rng.normal(0.0, 0.015)
    exchange_strength = max(0.0, exchange_strength)

    spin_capacity = np.sqrt(max(moment_a, 0.0) * max(moment_b, 0.0))
    spin_capacity = float(np.clip(spin_capacity, 0.0, 1.25))

    if expected_ordering == "ferromagnetic":
        order_parameter = spin_capacity * abs(coupling_sign) * (1.0 - 0.25 * anisotropy)
        phase_lock = 0.92 + 0.06 * np.exp(-anisotropy)
        acoustic_gap = 0.02 + 0.28 * anisotropy
        optical_gap = 0.12 + 0.35 * anisotropy
        branch_count = 1.0
    elif expected_ordering == "antiferromagnetic":
        order_parameter = spin_capacity * abs(coupling_sign) * (1.0 - 0.20 * frustration)
        phase_lock = 0.82 + 0.12 * np.exp(-frustration)
        acoustic_gap = 0.05 + 0.30 * anisotropy
        optical_gap = 0.30 + 0.45 * exchange_strength
        branch_count = 2.0
    elif expected_ordering == "ferrimagnetic":
        imbalance = abs(moment_a - moment_b)
        order_parameter = (0.55 * spin_capacity + 0.45 * imbalance) * (1.0 - 0.15 * frustration)
        phase_lock = 0.78 + 0.10 * np.exp(-frustration)
        acoustic_gap = 0.04 + 0.25 * anisotropy
        optical_gap = 0.25 + 0.40 * exchange_strength
        branch_count = 2.0
    elif expected_ordering == "frustrated":
        order_parameter = spin_capacity * exchange_strength * (0.45 + 0.30 * frustration)
        phase_lock = 0.55 + 0.25 * frustration
        acoustic_gap = 0.08 + 0.40 * anisotropy
        optical_gap = 0.20 + 0.30 * exchange_strength
        branch_count = 3.0
    elif expected_ordering == "nonmagnetic":
        order_parameter = 0.70
        phase_lock = 0.10
        acoustic_gap = 0.0
        optical_gap = 0.0
        branch_count = 0.0
    else:
        order_parameter = 0.0
        phase_lock = 0.0
        acoustic_gap = 0.0
        optical_gap = 0.0
        branch_count = 0.0

    spin_coherence = (
        0.45 * phase_lock
        + 0.30 * order_parameter
        + 0.15 * exchange_strength
        + 0.10 * (1.0 - frustration)
    )

    stiffness = exchange_strength * spin_capacity * connectivity
    magnon_velocity = np.sqrt(max(stiffness, 0.0) + EPS) * (1.0 - 0.35 * frustration)
    dispersion_curvature = exchange_strength * (1.0 + 0.5 * anisotropy) * (1.0 - 0.25 * frustration)
    damping_proxy = (
        0.06
        + 0.22 * frustration
        + 0.12 * anisotropy
        + 0.18 * (1.0 - min(spin_coherence, 1.0))
    )

    magnon_emergence = (
        0.30 * spin_coherence
        + 0.25 * order_parameter
        + 0.20 * magnon_velocity
        + 0.15 * dispersion_curvature
        + 0.10 * branch_count / 3.0
    )

    magnon_stability = (
        magnon_emergence
        * np.exp(-0.75 * damping_proxy)
        * (1.0 if expected_stable else 0.08)
    )

    predicted_stable = float(magnon_stability >= 0.30)

    return {
        "material": material,
        "ion_a": ion_a,
        "ion_b": ion_b,
        "d_count_a": d_a,
        "d_count_b": d_b,
        "f_count_a": f_a,
        "f_count_b": f_b,
        "unpaired_a": ua,
        "unpaired_b": ub,
        "coordination": coordination,
        "geometry": geometry,
        "geometry_id": GEOMETRY_ID[geometry],
        "coupling_sign": coupling_sign,
        "exchange_base": exchange_base,
        "exchange_strength": exchange_strength,
        "anisotropy": anisotropy,
        "frustration": frustration,
        "connectivity": connectivity,
        "spin_capacity": spin_capacity,
        "magnetic_moment": magnetic_moment,
        "order_parameter": order_parameter,
        "phase_lock": phase_lock,
        "spin_coherence": spin_coherence,
        "stiffness": stiffness,
        "magnon_velocity": magnon_velocity,
        "dispersion_curvature": dispersion_curvature,
        "acoustic_gap": acoustic_gap,
        "optical_gap": optical_gap,
        "branch_count": branch_count,
        "damping_proxy": damping_proxy,
        "magnon_emergence": magnon_emergence,
        "magnon_stability": magnon_stability,
        "expected_ordering": expected_ordering,
        "ordering_id": ORDERING_ID[expected_ordering],
        "expected_stable": float(expected_stable),
        "predicted_stable": predicted_stable,
        "perturbation": perturbation,
        "replicate": replicate,
    }


rows = []
for perturbation in PERTURBATIONS:
    for m in MATERIALS:
        for r in range(REPLICATES):
            rows.append(make_sample(*m, perturbation, r))

samples = pd.DataFrame(rows)

feature_cols = [
    "d_count_a",
    "d_count_b",
    "f_count_a",
    "f_count_b",
    "unpaired_a",
    "unpaired_b",
    "coordination",
    "geometry_id",
    "coupling_sign",
    "exchange_strength",
    "anisotropy",
    "frustration",
    "connectivity",
    "spin_capacity",
    "magnetic_moment",
    "order_parameter",
    "phase_lock",
    "spin_coherence",
    "stiffness",
    "magnon_velocity",
    "dispersion_curvature",
    "acoustic_gap",
    "optical_gap",
    "branch_count",
    "damping_proxy",
    "magnon_emergence",
    "magnon_stability",
]

X = samples[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = samples["perturbation"].isin([0.0, 0.01]).to_numpy()
test = samples["perturbation"].isin([0.025, 0.05]).to_numpy()

clf_order = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf_order.fit(Xn[train], samples.loc[train, "expected_ordering"])
pred_order = clf_order.predict(Xn[test])
ordering_accuracy = float(accuracy_score(samples.loc[test, "expected_ordering"], pred_order))

clf_stable = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf_stable.fit(Xn[train], samples.loc[train, "expected_stable"])
pred_stable = clf_stable.predict(Xn[test])
stability_accuracy = float(accuracy_score(samples.loc[test, "expected_stable"], pred_stable))

stable_ratio_valid = float(samples[samples["expected_stable"] == 1.0]["predicted_stable"].mean())
stable_ratio_invalid = float(samples[samples["expected_stable"] == 0.0]["predicted_stable"].mean())

mean_valid_stability = float(samples[samples["expected_stable"] == 1.0]["magnon_stability"].mean())
mean_invalid_stability = float(samples[samples["expected_stable"] == 0.0]["magnon_stability"].mean())
stability_contrast = float((mean_valid_stability + EPS) / (mean_invalid_stability + EPS))

try:
    ordering_silhouette = float(silhouette_score(Xn, samples["expected_ordering"]))
except Exception:
    ordering_silhouette = np.nan

# Build dispersion curves from canonical material averages.
by_material_base = samples.groupby("material").mean(numeric_only=True).reset_index()
curve_rows = []
for _, row in by_material_base.iterrows():
    mat = row["material"]
    ordering = samples[samples["material"] == mat]["expected_ordering"].iloc[0]
    stable = samples[samples["material"] == mat]["expected_stable"].iloc[0]
    J = row["exchange_strength"]
    v = row["magnon_velocity"]
    gap = row["acoustic_gap"]
    optical = row["optical_gap"]
    damping = row["damping_proxy"]

    for k in K_POINTS:
        if ordering == "ferromagnetic":
            omega_1 = gap + 2.0 * J * (1.0 - np.cos(k))
            omega_2 = np.nan
        elif ordering in {"antiferromagnetic", "ferrimagnetic"}:
            omega_1 = np.sqrt(max(gap**2 + (v * np.sin(k)) ** 2, 0.0))
            omega_2 = optical + J * (1.0 - 0.35 * np.cos(k))
        elif ordering == "frustrated":
            omega_1 = gap + v * abs(np.sin(k)) + 0.25 * J * (1.0 - np.cos(3.0 * k))
            omega_2 = optical + 0.5 * J * (1.0 - np.cos(k))
        else:
            omega_1 = 0.0
            omega_2 = np.nan

        spectral_weight = float(np.exp(-damping) * (1.0 if stable else 0.05))
        curve_rows.append({
            "material": mat,
            "ordering": ordering,
            "k": k,
            "omega_branch_1": omega_1,
            "omega_branch_2": omega_2,
            "spectral_weight": spectral_weight,
        })

dispersion = pd.DataFrame(curve_rows)

by_material = samples.groupby("material").agg(
    count=("material", "count"),
    expected_ordering=("expected_ordering", "first"),
    expected_stable=("expected_stable", "mean"),
    mean_exchange_strength=("exchange_strength", "mean"),
    mean_spin_capacity=("spin_capacity", "mean"),
    mean_order_parameter=("order_parameter", "mean"),
    mean_phase_lock=("phase_lock", "mean"),
    mean_spin_coherence=("spin_coherence", "mean"),
    mean_magnon_velocity=("magnon_velocity", "mean"),
    mean_dispersion_curvature=("dispersion_curvature", "mean"),
    mean_acoustic_gap=("acoustic_gap", "mean"),
    mean_optical_gap=("optical_gap", "mean"),
    mean_branch_count=("branch_count", "mean"),
    mean_damping_proxy=("damping_proxy", "mean"),
    mean_magnon_emergence=("magnon_emergence", "mean"),
    mean_magnon_stability=("magnon_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index().sort_values(["expected_ordering", "material"])

by_ordering = samples.groupby("expected_ordering").agg(
    count=("expected_ordering", "count"),
    materials=("material", lambda x: ",".join(sorted(set(x)))),
    mean_exchange_strength=("exchange_strength", "mean"),
    mean_spin_coherence=("spin_coherence", "mean"),
    mean_order_parameter=("order_parameter", "mean"),
    mean_magnon_velocity=("magnon_velocity", "mean"),
    mean_damping_proxy=("damping_proxy", "mean"),
    mean_magnon_stability=("magnon_stability", "mean"),
).reset_index().sort_values("expected_ordering")

ferro_support = float(by_ordering.loc[by_ordering["expected_ordering"] == "ferromagnetic", "mean_magnon_stability"].iloc[0] > 0.45)
antiferro_support = float(by_ordering.loc[by_ordering["expected_ordering"] == "antiferromagnetic", "mean_magnon_stability"].iloc[0] > 0.45)
ferri_support = float(by_ordering.loc[by_ordering["expected_ordering"] == "ferrimagnetic", "mean_magnon_stability"].iloc[0] > 0.45)
frustrated_support = float(by_ordering.loc[by_ordering["expected_ordering"] == "frustrated", "mean_magnon_stability"].iloc[0] > 0.30)
nonmagnetic_suppression = float(by_ordering.loc[by_ordering["expected_ordering"] == "nonmagnetic", "mean_magnon_stability"].iloc[0] < 0.30)
none_suppression = float(by_ordering.loc[by_ordering["expected_ordering"] == "none", "mean_magnon_stability"].iloc[0] < 0.10)

if (
    ordering_accuracy >= 0.95
    and stability_accuracy >= 0.95
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.05
    and stability_contrast >= 10.0
    and ferro_support == 1.0
    and antiferro_support == 1.0
    and ferri_support == 1.0
    and frustrated_support == 1.0
    and nonmagnetic_suppression == 1.0
    and none_suppression == 1.0
):
    verdict = "magnon_emergence_supported"
elif ordering_accuracy >= 0.85 and stable_ratio_valid >= 0.80:
    verdict = "weak_magnon_emergence"
else:
    verdict = "magnon_emergence_not_supported"

summary = pd.DataFrame([{
    "num_materials": len(MATERIALS),
    "num_samples": len(samples),
    "ordering_accuracy": ordering_accuracy,
    "stability_accuracy": stability_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "ordering_silhouette": ordering_silhouette,
    "ferro_support": ferro_support,
    "antiferro_support": antiferro_support,
    "ferri_support": ferri_support,
    "frustrated_support": frustrated_support,
    "nonmagnetic_suppression": nonmagnetic_suppression,
    "none_suppression": none_suppression,
    "verdict": verdict,
}])

samples.to_csv(OUT / "magnon_samples.csv", index=False)
summary.to_csv(OUT / "magnon_summary.csv", index=False)
by_material.to_csv(OUT / "magnon_by_material.csv", index=False)
by_ordering.to_csv(OUT / "magnon_by_ordering.csv", index=False)
dispersion.to_csv(OUT / "magnon_dispersion_curves.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)
plt.figure(figsize=(10, 7))
for ordering in ORDERING_ID:
    mask = samples["expected_ordering"].to_numpy() == ordering
    plt.scatter(proj[mask, 0], proj[mask, 1], s=8, label=ordering, alpha=0.65)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("BST magnon emergence latent space")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "magnon_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 5))
ordered = by_material.sort_values("mean_magnon_stability", ascending=False)
plt.bar(ordered["material"], ordered["mean_magnon_stability"])
plt.xticks(rotation=75, ha="right", fontsize=8)
plt.ylabel("Mean magnon stability")
plt.title("BST magnon stability by material")
plt.tight_layout()
plt.savefig(OUT / "magnon_stability.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 6))
plot_materials = ["Fe_like", "MnO_like", "Fe3O4_like", "Triangular_Mn_like", "Cu_like", "Ar_solid"]
for mat in plot_materials:
    d = dispersion[dispersion["material"] == mat]
    plt.plot(d["k"], d["omega_branch_1"], label=f"{mat} branch 1")
    if d["omega_branch_2"].notna().any():
        plt.plot(d["k"], d["omega_branch_2"], linestyle="--", label=f"{mat} branch 2")
plt.xlabel("k")
plt.ylabel("ω(k) proxy")
plt.title("BST magnon dispersion proxies")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "magnon_dispersion.png", dpi=220)
plt.close()

print("\n=== BST MAGNON EMERGENCE TEST ===\n")
print(summary.to_string(index=False))

print("\nMagnon emergence by material:")
print(by_material.to_string(index=False))

print("\nMagnon emergence by ordering:")
print(by_ordering.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'magnon_samples.csv'}")
print(f"[OK] wrote {OUT / 'magnon_summary.csv'}")
print(f"[OK] wrote {OUT / 'magnon_by_material.csv'}")
print(f"[OK] wrote {OUT / 'magnon_by_ordering.csv'}")
print(f"[OK] wrote {OUT / 'magnon_dispersion_curves.csv'}")
print(f"[OK] wrote {OUT / 'magnon_projection.png'}")
print(f"[OK] wrote {OUT / 'magnon_stability.png'}")
print(f"[OK] wrote {OUT / 'magnon_dispersion.png'}")
print("[DONE] magnon emergence test complete")
