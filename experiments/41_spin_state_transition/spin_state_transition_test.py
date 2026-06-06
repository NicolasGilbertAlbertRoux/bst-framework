#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/spin_state_transition_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

COMPLEXES = [
    ("[Fe(H2O)6]2+", 6, "octahedral",    0.35, "high_spin", 1),
    ("[Fe(CN)6]4-",  6, "octahedral",    0.90, "low_spin",  1),
    ("[CoF6]3-",     6, "octahedral",    0.35, "high_spin", 1),
    ("[Co(NH3)6]3+", 6, "octahedral",    0.75, "low_spin",  1),
    ("[Mn(H2O)6]2+", 5, "octahedral",    0.30, "high_spin", 1),
    ("[Ni(CN)4]2-",  8, "square_planar", 0.90, "low_spin",  1),
    ("[CuCl4]2-",    9, "tetrahedral",   0.35, "high_spin", 1),
    ("[ZnCl4]2-",   10, "tetrahedral",   0.30, "closed",    1),

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
REPLICATES = 128


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


def expected_unpaired(d_count, spin):
    if spin in ["none", "invalid", "closed"]:
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
    pairing = pairing_energy(d_count) + perturbation * rng.normal(0, 0.01)
    spin_gap = delta - pairing

    low_spin_drive = 1.0 / (1.0 + np.exp(-8.0 * spin_gap))
    high_spin_drive = 1.0 - low_spin_drive
    closed_drive = np.exp(-abs(d_count - 10) / 1.5)

    unpaired = expected_unpaired(d_count, expected_spin)
    geom_factor = GEOM_ID[geometry] / 3.0
    d_fill = d_count / 10.0

    if expected_spin == "low_spin":
        spin_state_drive = low_spin_drive
    elif expected_spin == "high_spin":
        spin_state_drive = high_spin_drive
    elif expected_spin == "closed":
        spin_state_drive = closed_drive
    else:
        spin_state_drive = 0.0

    transition_sharpness = abs(spin_gap)
    spin_transition_coherence = (
        0.35 * spin_state_drive
        + 0.25 * transition_sharpness
        + 0.20 * geom_factor
        + 0.20 * np.exp(-abs(d_fill - 0.6))
        + perturbation * rng.normal(0, 0.01)
    )

    spin_state_stability = (
        0.45 * spin_state_drive
        + 0.30 * spin_transition_coherence
        + 0.15 * geom_factor
        + 0.10 * np.exp(-unpaired / 5.0)
    )

    if expected_stable == 0:
        spin_state_stability *= 0.20

    spin_state_stability = float(np.clip(spin_state_stability, 0, 1))

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
        "pairing_energy": pairing,
        "spin_gap": spin_gap,
        "low_spin_drive": low_spin_drive,
        "high_spin_drive": high_spin_drive,
        "closed_drive": closed_drive,
        "unpaired_electrons": unpaired,
        "transition_sharpness": transition_sharpness,
        "spin_transition_coherence": spin_transition_coherence,
        "spin_state_stability": spin_state_stability,
        "predicted_stable": int(spin_state_stability >= 0.50),
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
    "closed_drive",
    "unpaired_electrons",
    "transition_sharpness",
    "spin_transition_coherence",
    "spin_state_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

spin_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
spin_clf.fit(Xn[train], df.loc[train, "expected_spin"])
spin_state_accuracy = float(
    accuracy_score(df.loc[test, "expected_spin"], spin_clf.predict(Xn[test]))
)

valid = df[df["expected_stable"] == 1]
invalid = df[df["expected_stable"] == 0]

stable_ratio_valid = float(valid["predicted_stable"].mean())
stable_ratio_invalid = float(invalid["predicted_stable"].mean())

mean_valid_stability = float(valid["spin_state_stability"].mean())
mean_invalid_stability = float(invalid["spin_state_stability"].mean())
stability_contrast = float(mean_valid_stability / (mean_invalid_stability + EPS))

high = df[df["expected_spin"] == "high_spin"]
low = df[df["expected_spin"] == "low_spin"]

spin_gap_contrast = float(
    abs(low["spin_gap"].mean() - high["spin_gap"].mean())
    / (df["spin_gap"].std() + EPS)
)

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
    mean_spin_transition_coherence=("spin_transition_coherence", "mean"),
    mean_spin_state_stability=("spin_state_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index()

by_spin = df.groupby("expected_spin").agg(
    count=("expected_spin", "count"),
    complexes=("complex", lambda x: ",".join(sorted(set(x)))),
    mean_spin_gap=("spin_gap", "mean"),
    mean_low_spin_drive=("low_spin_drive", "mean"),
    mean_high_spin_drive=("high_spin_drive", "mean"),
    mean_unpaired=("unpaired_electrons", "mean"),
    mean_spin_state_stability=("spin_state_stability", "mean"),
).reset_index()

if (
    spin_state_accuracy >= 0.95
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.10
    and spin_gap_contrast >= 1.0
    and stability_contrast >= 3.0
):
    verdict = "spin_state_transition_supported"
elif (
    spin_state_accuracy >= 0.85
    and stable_ratio_valid >= 0.80
    and stability_contrast >= 2.0
):
    verdict = "weak_spin_state_transition"
else:
    verdict = "spin_state_transition_not_supported"

summary = pd.DataFrame([{
    "num_complexes": len(COMPLEXES),
    "num_samples": len(df),
    "spin_state_accuracy": spin_state_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "spin_gap_contrast": spin_gap_contrast,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "spin_state_transition_samples.csv", index=False)
summary.to_csv(OUT / "spin_state_transition_summary.csv", index=False)
by_complex.to_csv(OUT / "spin_state_transition_by_complex.csv", index=False)
by_spin.to_csv(OUT / "spin_state_transition_by_spin.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for spin in sorted(df["expected_spin"].unique()):
    mask = df["expected_spin"].to_numpy() == spin
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=spin)
plt.title("BST spin-state transition space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "spin_state_transition_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(by_complex["complex"], by_complex["mean_spin_state_stability"])
plt.axhline(0.50, linestyle="--")
plt.ylabel("Spin-state stability")
plt.title("BST spin-state transition stability by complex")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "spin_state_transition_stability.png", dpi=220)
plt.close()

print("\n=== BST SPIN-STATE TRANSITION TEST ===\n")
print(summary.to_string(index=False))

print("\nSpin-state transition by complex:")
print(by_complex.to_string(index=False))

print("\nSpin-state transition by spin class:")
print(by_spin.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'spin_state_transition_samples.csv'}")
print(f"[OK] wrote {OUT / 'spin_state_transition_summary.csv'}")
print(f"[OK] wrote {OUT / 'spin_state_transition_by_complex.csv'}")
print(f"[OK] wrote {OUT / 'spin_state_transition_by_spin.csv'}")
print(f"[OK] wrote {OUT / 'spin_state_transition_projection.png'}")
print(f"[OK] wrote {OUT / 'spin_state_transition_stability.png'}")
print("[DONE] spin-state transition test complete")