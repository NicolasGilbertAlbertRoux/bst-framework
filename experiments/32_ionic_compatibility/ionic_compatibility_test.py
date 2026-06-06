#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

OUT = Path("results/research_final/ionic_compatibility_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

ELEMENTS = {
    "H":  dict(Z=1,  valence=1, group=1,  period=1, en=2.20),
    "Li": dict(Z=3,  valence=1, group=1,  period=2, en=0.98),
    "Na": dict(Z=11, valence=1, group=1,  period=3, en=0.93),
    "Be": dict(Z=4,  valence=2, group=2,  period=2, en=1.57),
    "Mg": dict(Z=12, valence=2, group=2,  period=3, en=1.31),
    "O":  dict(Z=8,  valence=6, group=16, period=2, en=3.44),
    "F":  dict(Z=9,  valence=7, group=17, period=2, en=3.98),
    "Cl": dict(Z=17, valence=7, group=17, period=3, en=3.16),
    "Ne": dict(Z=10, valence=8, group=18, period=2, en=0.00),
    "Ar": dict(Z=18, valence=8, group=18, period=3, en=0.00),
}

# formula, cation, anion, cation_charge, anion_charge, expected_stable
PAIRS = [
    ("LiF",   "Li", "F",  +1, -1, 1),
    ("LiCl",  "Li", "Cl", +1, -1, 1),
    ("NaF",   "Na", "F",  +1, -1, 1),
    ("NaCl",  "Na", "Cl", +1, -1, 1),
    ("BeO",   "Be", "O",  +2, -2, 1),
    ("MgO",   "Mg", "O",  +2, -2, 1),
    ("MgF2",  "Mg", "F",  +2, -1, 1),
    ("BeCl2", "Be", "Cl", +2, -1, 1),

    # negative controls
    ("NeAr",  "Ne", "Ar", 0, 0, 0),
    ("NaLi",  "Na", "Li", +1, +1, 0),
    ("FCl",   "F",  "Cl", -1, -1, 0),
    ("HH",    "H",  "H",  +1, +1, 0),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96


def shell_need_for_noble(valence):
    if valence == 8:
        return 0
    if valence <= 4:
        return valence
    return 8 - valence


def make_sample(formula, cation, anion, c_charge, a_charge, expected_stable, perturbation, replicate):
    C = ELEMENTS[cation]
    A = ELEMENTS[anion]

    charge_balance = np.exp(-abs(c_charge + a_charge))
    en_gap = abs(A["en"] - C["en"])

    cation_closure = np.exp(-abs(C["valence"] - abs(c_charge)) / 8.0)
    anion_closure = np.exp(-abs((8 - A["valence"]) - abs(a_charge)) / 8.0)

    noble_target_match = 0.5 * (cation_closure + anion_closure)

    coulomb_lock = (
        max(c_charge, 0)
        * max(-a_charge, 0)
        / 4.0
    )

    ionic_drive = (
        0.40 * min(en_gap / 3.0, 1.0)
        + 0.30 * charge_balance
        + 0.20 * noble_target_match
        + 0.10 * coulomb_lock
        + perturbation * rng.normal(0, 0.01)
    )

    lattice_closure = (
        0.35 * charge_balance
        + 0.30 * noble_target_match
        + 0.20 * coulomb_lock
        + 0.15 * min(en_gap / 3.0, 1.0)
        + perturbation * rng.normal(0, 0.01)
    )

    incompatible_charge = int(c_charge * a_charge > 0 or (c_charge == 0 and a_charge == 0))

    ionic_stability = (
        0.35 * ionic_drive
        + 0.35 * lattice_closure
        + 0.20 * charge_balance
        + 0.10 * noble_target_match
    )

    if incompatible_charge:
        ionic_stability *= 0.25

    ionic_stability = float(np.clip(ionic_stability, 0, 1))

    return {
        "formula": formula,
        "cation": cation,
        "anion": anion,
        "expected_stable": expected_stable,
        "perturbation": perturbation,
        "replicate": replicate,
        "cation_charge": c_charge,
        "anion_charge": a_charge,
        "charge_balance": charge_balance,
        "electronegativity_gap": en_gap,
        "cation_closure": cation_closure,
        "anion_closure": anion_closure,
        "noble_target_match": noble_target_match,
        "coulomb_lock": coulomb_lock,
        "ionic_drive": ionic_drive,
        "lattice_closure": lattice_closure,
        "ionic_stability": ionic_stability,
        "predicted_stable": int(ionic_stability >= 0.55),
    }


rows = []
for perturbation in PERTURBATIONS:
    for pair in PAIRS:
        for r in range(REPLICATES):
            rows.append(make_sample(*pair, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "cation_charge",
    "anion_charge",
    "charge_balance",
    "electronegativity_gap",
    "cation_closure",
    "anion_closure",
    "noble_target_match",
    "coulomb_lock",
    "ionic_drive",
    "lattice_closure",
    "ionic_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

valid = df[df["expected_stable"] == 1]
invalid = df[df["expected_stable"] == 0]

stable_ratio_valid = float(valid["predicted_stable"].mean())
stable_ratio_invalid = float(invalid["predicted_stable"].mean())

mean_valid_stability = float(valid["ionic_stability"].mean())
mean_invalid_stability = float(invalid["ionic_stability"].mean())
stability_contrast = float(mean_valid_stability / (mean_invalid_stability + EPS))

try:
    sil = float(silhouette_score(Xn, df["expected_stable"]))
except Exception:
    sil = np.nan

by_formula = df.groupby("formula").agg(
    count=("formula", "count"),
    expected_stable=("expected_stable", "mean"),
    mean_charge_balance=("charge_balance", "mean"),
    mean_electronegativity_gap=("electronegativity_gap", "mean"),
    mean_noble_target_match=("noble_target_match", "mean"),
    mean_coulomb_lock=("coulomb_lock", "mean"),
    mean_ionic_drive=("ionic_drive", "mean"),
    mean_lattice_closure=("lattice_closure", "mean"),
    mean_ionic_stability=("ionic_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index()

if (
    stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.10
    and stability_contrast >= 2.0
    and mean_valid_stability >= 0.55
):
    verdict = "ionic_compatibility_supported"
elif (
    stable_ratio_valid >= 0.80
    and stability_contrast >= 1.5
):
    verdict = "weak_ionic_compatibility"
else:
    verdict = "ionic_compatibility_not_supported"

summary = pd.DataFrame([{
    "num_pairs": len(PAIRS),
    "num_samples": len(df),
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "ionic_compatibility_samples.csv", index=False)
summary.to_csv(OUT / "ionic_compatibility_summary.csv", index=False)
by_formula.to_csv(OUT / "ionic_compatibility_by_formula.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(8, 6))
for label, name in [(0, "invalid"), (1, "ionic-compatible")]:
    mask = df["expected_stable"].to_numpy() == label
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=name)
plt.title("BST ionic compatibility space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "ionic_compatibility_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(by_formula["formula"], by_formula["mean_ionic_stability"])
plt.axhline(0.55, linestyle="--")
plt.ylabel("Ionic stability")
plt.title("BST ionic compatibility by formula")
plt.tight_layout()
plt.savefig(OUT / "ionic_compatibility_stability.png", dpi=220)
plt.close()

print("\n=== BST IONIC COMPATIBILITY TEST ===\n")
print(summary.to_string(index=False))

print("\nIonic compatibility by formula:")
print(by_formula.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'ionic_compatibility_samples.csv'}")
print(f"[OK] wrote {OUT / 'ionic_compatibility_summary.csv'}")
print(f"[OK] wrote {OUT / 'ionic_compatibility_by_formula.csv'}")
print(f"[OK] wrote {OUT / 'ionic_compatibility_projection.png'}")
print(f"[OK] wrote {OUT / 'ionic_compatibility_stability.png'}")
print("[DONE] ionic compatibility test complete")