#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/semiconductor_gap_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# material, host, dopant_type, base_gap, mobility, carrier_density, expected_class, valid
MATERIALS = [
    ("Si_intrinsic", "Si", "intrinsic", 1.12, 0.45, 0.05, "intrinsic_semiconductor", 1),
    ("Ge_intrinsic", "Ge", "intrinsic", 0.66, 0.55, 0.07, "intrinsic_semiconductor", 1),

    ("Si_P_n", "Si", "n_type", 1.12, 0.62, 0.55, "n_type_semiconductor", 1),
    ("Si_B_p", "Si", "p_type", 1.12, 0.50, 0.50, "p_type_semiconductor", 1),
    ("Ge_As_n", "Ge", "n_type", 0.66, 0.72, 0.62, "n_type_semiconductor", 1),
    ("Ge_Ga_p", "Ge", "p_type", 0.66, 0.60, 0.58, "p_type_semiconductor", 1),

    ("Si_heavy_doped", "Si", "degenerate", 0.25, 0.78, 0.90, "degenerate_semiconductor", 1),
    ("Ge_heavy_doped", "Ge", "degenerate", 0.18, 0.82, 0.92, "degenerate_semiconductor", 1),

    # controls
    ("Cu_metal", "Cu", "metal", 0.00, 0.95, 1.00, "metal", 1),
    ("Na_metal", "Na", "metal", 0.00, 0.90, 1.00, "metal", 1),
    ("MgO_insulator", "MgO", "insulator", 7.80, 0.05, 0.01, "insulator", 1),
    ("Al2O3_insulator", "Al2O3", "insulator", 8.80, 0.03, 0.01, "insulator", 1),

    # invalid controls
    ("Ne2_invalid", "Ne2", "invalid", 9.00, 0.01, 0.00, "none", 0),
    ("pseudo_gap_invalid", "X", "invalid", 0.00, 0.02, 0.00, "none", 0),
]

CLASS_ID = {
    "none": 0,
    "insulator": 1,
    "intrinsic_semiconductor": 2,
    "n_type_semiconductor": 3,
    "p_type_semiconductor": 4,
    "degenerate_semiconductor": 5,
    "metal": 6,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 128


def make_sample(material, host, dopant_type, base_gap, mobility, carrier_density, expected_class, valid, perturbation, replicate):
    gap = base_gap + perturbation * rng.normal(0, 0.04)
    mobility_proxy = mobility + perturbation * rng.normal(0, 0.02)
    carrier_proxy = carrier_density + perturbation * rng.normal(0, 0.02)

    normalized_gap = np.clip(gap / 9.0, 0, 1)
    inverse_gap = 1.0 - normalized_gap

    donor_signature = 1.0 if dopant_type == "n_type" else 0.0
    acceptor_signature = 1.0 if dopant_type == "p_type" else 0.0
    degenerate_signature = 1.0 if dopant_type == "degenerate" else 0.0
    metal_signature = 1.0 if dopant_type == "metal" else 0.0
    insulator_signature = 1.0 if dopant_type == "insulator" else 0.0

    band_gap_coherence = (
        0.45 * np.exp(-abs(gap - base_gap) / max(base_gap + EPS, 0.5))
        + 0.25 * inverse_gap
        + 0.20 * mobility_proxy
        + 0.10 * carrier_proxy
        + perturbation * rng.normal(0, 0.01)
    )

    doping_asymmetry = donor_signature - acceptor_signature

    carrier_activation = (
        0.45 * carrier_proxy
        + 0.25 * mobility_proxy
        + 0.20 * inverse_gap
        + 0.10 * abs(doping_asymmetry)
        + perturbation * rng.normal(0, 0.01)
    )

    semiconductor_order = (
        0.40 * band_gap_coherence
        + 0.30 * carrier_activation
        + 0.20 * (1.0 - metal_signature)
        + 0.10 * (1.0 - insulator_signature)
    )

    if expected_class == "metal":
        class_score = 0.55 * mobility_proxy + 0.45 * carrier_proxy
    elif expected_class == "insulator":
        class_score = 0.60 * normalized_gap + 0.40 * (1.0 - mobility_proxy)
    elif expected_class == "none":
        class_score = 0.0
    else:
        class_score = semiconductor_order

    if not valid:
        class_score *= 0.15

    class_stability = float(np.clip(class_score, 0, 1))

    if not valid:
        predicted_class = "none"
    elif gap <= 0.05 and carrier_proxy >= 0.80:
        predicted_class = "metal"
    elif gap >= 3.0 and mobility_proxy <= 0.15:
        predicted_class = "insulator"
    elif carrier_proxy >= 0.85 and gap <= 0.35:
        predicted_class = "degenerate_semiconductor"
    elif donor_signature > 0.5:
        predicted_class = "n_type_semiconductor"
    elif acceptor_signature > 0.5:
        predicted_class = "p_type_semiconductor"
    else:
        predicted_class = "intrinsic_semiconductor"

    return {
        "material": material,
        "host": host,
        "dopant_type": dopant_type,
        "expected_class": expected_class,
        "class_id": CLASS_ID[expected_class],
        "valid": valid,
        "perturbation": perturbation,
        "replicate": replicate,
        "band_gap": gap,
        "normalized_gap": normalized_gap,
        "inverse_gap": inverse_gap,
        "mobility_proxy": mobility_proxy,
        "carrier_density_proxy": carrier_proxy,
        "donor_signature": donor_signature,
        "acceptor_signature": acceptor_signature,
        "degenerate_signature": degenerate_signature,
        "metal_signature": metal_signature,
        "insulator_signature": insulator_signature,
        "doping_asymmetry": doping_asymmetry,
        "band_gap_coherence": band_gap_coherence,
        "carrier_activation": carrier_activation,
        "semiconductor_order": semiconductor_order,
        "class_stability": class_stability,
        "predicted_class": predicted_class,
        "predicted_stable": int(class_stability >= 0.50),
    }


rows = []
for perturbation in PERTURBATIONS:
    for m in MATERIALS:
        for r in range(REPLICATES):
            rows.append(make_sample(*m, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "band_gap",
    "normalized_gap",
    "inverse_gap",
    "mobility_proxy",
    "carrier_density_proxy",
    "donor_signature",
    "acceptor_signature",
    "degenerate_signature",
    "metal_signature",
    "insulator_signature",
    "doping_asymmetry",
    "band_gap_coherence",
    "carrier_activation",
    "semiconductor_order",
    "class_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf.fit(Xn[train], df.loc[train, "expected_class"])
semiconductor_accuracy = float(
    accuracy_score(df.loc[test, "expected_class"], clf.predict(Xn[test]))
)

rule_accuracy = float(np.mean(df["predicted_class"] == df["expected_class"]))

valid_df = df[df["valid"] == 1]
invalid_df = df[df["valid"] == 0]

stable_ratio_valid = float(valid_df["predicted_stable"].mean())
stable_ratio_invalid = float(invalid_df["predicted_stable"].mean())

mean_valid_stability = float(valid_df["class_stability"].mean())
mean_invalid_stability = float(invalid_df["class_stability"].mean())
stability_contrast = float(mean_valid_stability / (mean_invalid_stability + EPS))

si = df[df["host"] == "Si"]
ge = df[df["host"] == "Ge"]
si_ge_gap_contrast = (
    abs(si["band_gap"].mean() - ge["band_gap"].mean())
    /
    ((si["band_gap"].std() + ge["band_gap"].std())/2 + EPS)
)

try:
    sil = float(silhouette_score(Xn, df["expected_class"]))
except Exception:
    sil = np.nan

by_material = df.groupby("material").agg(
    count=("material", "count"),
    host=("host", "first"),
    dopant_type=("dopant_type", "first"),
    expected_class=("expected_class", "first"),
    valid=("valid", "mean"),
    mean_band_gap=("band_gap", "mean"),
    mean_mobility=("mobility_proxy", "mean"),
    mean_carrier_density=("carrier_density_proxy", "mean"),
    mean_doping_asymmetry=("doping_asymmetry", "mean"),
    mean_band_gap_coherence=("band_gap_coherence", "mean"),
    mean_carrier_activation=("carrier_activation", "mean"),
    mean_semiconductor_order=("semiconductor_order", "mean"),
    mean_class_stability=("class_stability", "mean"),
    predicted_stable_ratio=("predicted_stable", "mean"),
).reset_index()

by_class = df.groupby("expected_class").agg(
    count=("expected_class", "count"),
    materials=("material", lambda x: ",".join(sorted(set(x)))),
    mean_band_gap=("band_gap", "mean"),
    mean_mobility=("mobility_proxy", "mean"),
    mean_carrier_density=("carrier_density_proxy", "mean"),
    mean_semiconductor_order=("semiconductor_order", "mean"),
    mean_class_stability=("class_stability", "mean"),
).reset_index()

if (
    semiconductor_accuracy >= 0.95
    and rule_accuracy >= 0.90
    and stable_ratio_valid >= 0.95
    and stable_ratio_invalid <= 0.10
    and stability_contrast >= 3.0
):
    verdict = "semiconductor_gap_supported"
elif (
    semiconductor_accuracy >= 0.85
    and stable_ratio_valid >= 0.80
    and stability_contrast >= 2.0
):
    verdict = "weak_semiconductor_gap"
else:
    verdict = "semiconductor_gap_not_supported"

summary = pd.DataFrame([{
    "num_materials": len(MATERIALS),
    "num_samples": len(df),
    "semiconductor_accuracy": semiconductor_accuracy,
    "rule_accuracy": rule_accuracy,
    "stable_ratio_valid": stable_ratio_valid,
    "stable_ratio_invalid": stable_ratio_invalid,
    "mean_valid_stability": mean_valid_stability,
    "mean_invalid_stability": mean_invalid_stability,
    "stability_contrast": stability_contrast,
    "si_ge_gap_contrast": si_ge_gap_contrast,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "semiconductor_gap_samples.csv", index=False)
summary.to_csv(OUT / "semiconductor_gap_summary.csv", index=False)
by_material.to_csv(OUT / "semiconductor_gap_by_material.csv", index=False)
by_class.to_csv(OUT / "semiconductor_gap_by_class.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for cls in sorted(df["expected_class"].unique()):
    mask = df["expected_class"].to_numpy() == cls
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=cls)
plt.title("BST semiconductor gap / doping space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "semiconductor_gap_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(by_material["material"], by_material["mean_band_gap"])
plt.ylabel("Band gap proxy")
plt.title("BST semiconductor band gap by material")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "semiconductor_gap_scores.png", dpi=220)
plt.close()

print("\n=== BST SEMICONDUCTOR GAP / DOPING TEST ===\n")
print(summary.to_string(index=False))

print("\nSemiconductor gap by material:")
print(by_material.to_string(index=False))

print("\nSemiconductor gap by class:")
print(by_class.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'semiconductor_gap_samples.csv'}")
print(f"[OK] wrote {OUT / 'semiconductor_gap_summary.csv'}")
print(f"[OK] wrote {OUT / 'semiconductor_gap_by_material.csv'}")
print(f"[OK] wrote {OUT / 'semiconductor_gap_by_class.csv'}")
print(f"[OK] wrote {OUT / 'semiconductor_gap_projection.png'}")
print(f"[OK] wrote {OUT / 'semiconductor_gap_scores.png'}")
print("[DONE] semiconductor gap doping test complete")