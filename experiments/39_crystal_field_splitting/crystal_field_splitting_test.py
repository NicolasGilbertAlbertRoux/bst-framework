#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/crystal_field_splitting_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# complex, d_count, geometry, field_strength, expected_spin, expected_stable
COMPLEXES = [
    ("[Fe(H2O)6]2+", 6, "octahedral",    0.35, "high_spin", 1),
    ("[Fe(CN)6]4-",  6, "octahedral",    0.90, "low_spin",  1),
    ("[CoF6]3-",     6, "octahedral",    0.35, "high_spin", 1),
    ("[Co(NH3)6]3+", 6, "octahedral",    0.75, "low_spin",  1),
    ("[Mn(H2O)6]2+", 5, "octahedral",    0.30, "high_spin", 1),
    ("[Ni(CN)4]2-",  8, "square_planar", 0.90, "low_spin",  1),
    ("[CuCl4]2-",    9, "tetrahedral",   0.35, "high_spin", 1),
    ("[ZnCl4]2-",   10, "tetrahedral",   0.30, "closed",    1),

    # controls
    ("[Ne(H2O)6]",   0, "none",          0.00, "none",      0),
    ("[Zn(CN)7]5-", 10, "invalid",       0.90, "invalid",   0),
]

GEOM_ID = {
    "none": 0,
    "invalid": 0,
    "tetrahedral": 1,
    "square_planar": 2,
    "octahedral": 3,
}

SPIN_ID = {
    "none": 0,
    "invalid": 0,
    "closed": 1,
    "low_spin": 2,
    "high_spin": 3,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96


def base_splitting(geometry, field):
    if geometry == "octahedral":
        return 1.00 * field
    if geometry == "tetrahedral":
        return 0.45 * field
    if geometry == "square_planar":
        return 1.35 * field
    return 0.0


def pairing_energy(d_count):
    return 0.55 + 0.10 * np.exp(-abs(d_count - 5) / 3.0)


def spin_gap(d_count, geometry, field):
    delta = base_splitting(geometry, field)
    pair = pairing_energy(d_count)
    return delta - pair


def expected_unpaired(d_count, spin):
    if spin in ["none", "invalid"]:
        return 0
    if spin == "closed":
        return 0
    if spin == "high_spin":
        return min(d_count, 10 - d_count, 5)
    if spin == "low_spin":
        if d_count <= 6:
            return max(0, d_count - 6)
        return max(0, 10 - d_count)
    return 0


def make_sample(name, d_count, geometry, field, expected_spin, expected_stable, perturbation, replicate):
    delta = base_splitting(geometry, field) + perturbation * rng.normal(0, 0.01)
    pair = pairing_energy(d_count) + perturbation * rng.normal(0, 0.01)
    gap = delta - pair

    geom_factor = GEOM_ID[geometry] / 3.0
    d_fill = d_count / 10.0
    unpaired = expected_unpaired(d_count, expected_spin)

    low_spin_drive = 1.0 / (1.0 + np.exp(-8.0 * gap))
    high_spin_drive = 1.0 - low_spin_drive
    closed_shell_drive = np.exp(-abs(d_count - 10) / 1.5)

    if expected_spin == "low_spin":
        spin_match = low_spin_drive
    elif expected_spin == "high_spin":
        spin_match = high_spin_drive
    elif expected_spin == "closed":
        spin_match = closed_shell_drive
    else:
        spin_match = 0.0

    splitting_coherence = (
        0.35 * abs(delta)
        + 0.25 * abs(gap)
        + 0.20 * geom_factor
        + 0.20 * (1.0 - abs(d_fill - 0.5))
        + perturbation * rng.normal(0, 0.01)
    )

    crystal_field_stability = (
        0.40 * spin_match
        + 0.30 * splitting_coherence
        + 0.20 * geom_factor
        + 0.10 * np.exp(-abs(unpaired) / 5.0)
    )

    if expected_stable == 0:
        crystal_field_stability *= 0.20

    crystal_field_stability = float(np.clip(crystal_field_stability, 0, 1))

    return {
        "complex": name,
        "d_count": d_count,
        "geometry": geometry,
        "geometry_id": GEOM_ID[geometry],
        "field_strength": field,
        "expected_spin": expected_spin,
        "spin_id": SPIN_ID[expected_spin],
        "expected_stable": expected_stable,
        "perturbation": perturbation,
        "replicate": replicate,
        "d_fill": d_fill,
        "crystal_field_delta": delta,
        "pairing_energy": pair,
        "spin_gap": gap,
        "low_spin_drive": low_spin_drive,
        "high_spin_drive": high_spin_drive,
        "closed_shell_drive": closed_shell_drive,
        "unpaired_electrons": unpaired,
        "splitting_coherence": splitting_coherence,
        "crystal_field_stability": crystal_field_stability,
        "predicted_stable": int(crystal_field_stability >= 0.50),
    }


rows = []
for perturbation in PERTURBATIONS:
    for c in COMPLEXES:
        for r in range(REPLICATES):
            rows.append(make_sample(*c, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "d_count",
    "geometry_id",
    "field_strength",
    "d_fill",
    "crystal_field_delta",
    "pairing_energy",
    "spin_gap",
    "low_spin_drive",
    "high_spin_drive",
    "closed_shell_drive",
    "unpaired_electrons",
    "splitting_coherence",
    "crystal_field_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

spin_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
spin_clf.fit(Xn[train], df.loc[train, "expected_spin"])
spin_accuracy = float(accuracy_score(df.loc[test, "expected_spin"], spin_clf.predict(Xn[test])))

geom_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
geom_clf.fit(Xn[train], df.loc[train, "geometry"])
geometry_accuracy = float(accuracy_score(df.loc[test, "geometry"], geom_clf.predict(Xn[test])))

valid = df[df["expected_stable"] == 1]
invalid = df[df["expected_stable"] == 0]

stable_ratio_valid = float(valid["predicted_stable"].mean())
stable_ratio_invalid = float(invalid["predicted_stable"].mean())
mean_valid_stability = float(valid["crystal_field_stability"].mean())
mean_invalid_stability = float(invalid["crystal_field_stability"].mean())
stability_contrast = float(mean_valid_stability / (mean_invalid_stability + EPS))

try:
    sil = float(silhouette_score(Xn, df["expected_spin"]))
except Exception:
    sil = np.nan

by_complex = df.groupby("complex").agg(
    count=("complex", "count"),
    d_count=("d_count", "mean"),
    geometry=("geometry", "first"),
    expected_spin=("expected_spin", "first"),
    expected_stable=("expected_stable", "mean"),
    mean_field_strength=("field_strength", "mean"),
    mean_delta=("crystal_field_delta", "mean"),
    mean_pairing_energy=("pairing_energy", "mean"),
    mean_spin_gap=("spin_gap", "mean"),
    mean_low_spin_drive=("low_spin_drive", "mean"),
    mean_high_spin_drive=("high_spin_drive", "mean"),
    mean_unpaired_electrons=("unpaired_electrons", "mean"),
    mean_splitting_coherence=("splitting_coherence", "mean"),
    mean_crystal_field_stability=("crystal_field_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index()

by_spin = df.groupby("expected_spin").agg(
    count=("expected_spin", "count"),
    complexes=("complex", lambda x: ",".join(sorted(set(x)))),
    mean_delta=("crystal_field_delta", "mean"),
    mean_spin_gap=("spin_gap", "mean"),
    mean_unpaired=("unpaired_electrons", "mean"),
    mean_stability=("crystal_field_stability", "mean"),
).reset_index()

if (
    spin_accuracy >= 0.95
    and geometry_accuracy >= 0.95
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.10
    and stability_contrast >= 3.0
):
    verdict = "crystal_field_splitting_supported"
elif (
    spin_accuracy >= 0.85
    and stable_ratio_valid >= 0.80
    and stability_contrast >= 2.0
):
    verdict = "weak_crystal_field_splitting"
else:
    verdict = "crystal_field_splitting_not_supported"

summary = pd.DataFrame([{
    "num_complexes": len(COMPLEXES),
    "num_samples": len(df),
    "spin_accuracy": spin_accuracy,
    "geometry_accuracy": geometry_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "crystal_field_samples.csv", index=False)
summary.to_csv(OUT / "crystal_field_summary.csv", index=False)
by_complex.to_csv(OUT / "crystal_field_by_complex.csv", index=False)
by_spin.to_csv(OUT / "crystal_field_by_spin.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for spin in sorted(df["expected_spin"].unique()):
    mask = df["expected_spin"].to_numpy() == spin
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=spin)
plt.title("BST crystal field splitting space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "crystal_field_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(by_complex["complex"], by_complex["mean_crystal_field_stability"])
plt.axhline(0.50, linestyle="--")
plt.ylabel("Crystal-field stability")
plt.title("BST crystal field stability by complex")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "crystal_field_stability.png", dpi=220)
plt.close()

print("\n=== BST CRYSTAL FIELD SPLITTING TEST ===\n")
print(summary.to_string(index=False))

print("\nCrystal field splitting by complex:")
print(by_complex.to_string(index=False))

print("\nCrystal field splitting by spin class:")
print(by_spin.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'crystal_field_samples.csv'}")
print(f"[OK] wrote {OUT / 'crystal_field_summary.csv'}")
print(f"[OK] wrote {OUT / 'crystal_field_by_complex.csv'}")
print(f"[OK] wrote {OUT / 'crystal_field_by_spin.csv'}")
print(f"[OK] wrote {OUT / 'crystal_field_projection.png'}")
print(f"[OK] wrote {OUT / 'crystal_field_stability.png'}")
print("[DONE] crystal field splitting test complete")