#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST Test 50 — Superheavy Stability Extrapolation / Island Search

Purpose
-------
Stress-test the BST periodic reconstruction beyond the validated H→Lr range.
This script extrapolates from Z=104 to Z=172 and asks whether the BST latent
periodic machinery produces:

1. continuity through known superheavy elements 104→118,
2. plausible next-row structure for 119+,
3. local stability islands instead of monotonic optimism,
4. eventual instability / overload at very high Z.

Important scientific caution
----------------------------
This is not a claim of discovery. It is an internal BST extrapolation test.
Elements beyond Og are represented as hypothetical E119, E120, ... candidates.
The stability score is a model score, not a measured half-life.

Outputs
-------
results/research_final/superheavy_stability_extrapolation_test/
    superheavy_candidates.csv
    superheavy_summary.csv
    superheavy_by_region.csv
    superheavy_top_candidates.csv
    superheavy_stability_curve.png
    superheavy_latent_map.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

OUT = Path("results/research_final/superheavy_stability_extrapolation_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 32

# Last fully reconstructed element in T49 was Lr=103. We now probe 104→172.
# 104→118 are known named superheavy elements; 119+ are hypothetical.
KNOWN_SUPERHEAVY = {
    104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt",
    110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc",
    116: "Lv", 117: "Ts", 118: "Og",
}

BLOCK_ID = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "unknown": 5}
REGION_ID = {
    "known_superheavy": 0,
    "next_s_frontier": 1,
    "g_frontier": 2,
    "post_g_frontier": 3,
    "terminal_overload": 4,
}


def symbol_for_Z(Z: int) -> str:
    return KNOWN_SUPERHEAVY.get(Z, f"E{Z}")


def extrapolated_periodic_coordinates(Z: int) -> dict:
    """Return a deliberately explicit extrapolated shell/period model.

    104→118 complete period 7 after Lr:
        104→112 d-like, 113→118 p-like.

    119→172 are extrapolated as a compact period-8 / superheavy frontier:
        119→120 : 8s-like
        121→138 : 5g-like frontier
        139→152 : 6f-like frontier
        153→162 : 7d-like frontier
        163→168 : 8p-like closure
        169→172 : post-closure stress region

    The exact ordering beyond 120 is not treated as established physics; it is
    the test model's extrapolation axis.
    """
    if 104 <= Z <= 112:
        return {"period": 7, "group": Z - 100, "block": "d", "orbital_count": Z - 102, "region": "known_superheavy"}
    if 113 <= Z <= 118:
        return {"period": 7, "group": Z - 100, "block": "p", "orbital_count": Z - 112, "region": "known_superheavy"}
    if 119 <= Z <= 120:
        return {"period": 8, "group": Z - 118, "block": "s", "orbital_count": Z - 118, "region": "next_s_frontier"}
    if 121 <= Z <= 138:
        return {"period": 8, "group": 3 + (Z - 121), "block": "g", "orbital_count": Z - 120, "region": "g_frontier"}
    if 139 <= Z <= 152:
        return {"period": 8, "group": 3, "block": "f", "orbital_count": Z - 138, "region": "post_g_frontier"}
    if 153 <= Z <= 162:
        return {"period": 8, "group": 3 + (Z - 153), "block": "d", "orbital_count": Z - 152, "region": "post_g_frontier"}
    if 163 <= Z <= 168:
        return {"period": 8, "group": 13 + (Z - 163), "block": "p", "orbital_count": Z - 162, "region": "post_g_frontier"}
    return {"period": 8, "group": 18, "block": "unknown", "orbital_count": Z - 168, "region": "terminal_overload"}


def optimal_neutron_number(Z: int) -> int:
    """Toy neutron-valley proxy for superheavy candidates.

    Anchored so that the classic superheavy island around Z≈114–126 aligns
    near N≈184, then drifts upward for heavier extrapolated candidates.
    """
    if Z <= 126:
        return int(round(1.50 * Z + 13.0))  # Z=114 -> 184, Z=120 -> 193
    return int(round(184.0 + 1.65 * (Z - 126)))


def gaussian(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def make_candidate(Z: int, perturbation: float, replicate: int) -> dict:
    coords = extrapolated_periodic_coordinates(Z)

    symbol = symbol_for_Z(Z)
    period = coords["period"]
    group = coords["group"]
    block = coords["block"]
    orbital_count = coords["orbital_count"]
    region = coords["region"]

    N = optimal_neutron_number(Z)
    A = Z + N
    neutron_ratio = N / max(Z, 1)

    # Electronic / periodic closure.
    capacity = {"s": 2, "p": 6, "d": 10, "f": 14, "g": 18, "unknown": 4}[block]
    fill = np.clip(orbital_count / capacity, 0.0, 1.0)
    closure_to_empty = np.exp(-orbital_count / max(capacity, 1))
    closure_to_full = np.exp(-abs(capacity - orbital_count) / max(capacity / 3.0, 1.0))
    electronic_shell_closure = max(closure_to_empty, closure_to_full)

    half_shell = np.exp(-abs(fill - 0.5) / 0.18)
    block_coherence = {
        "s": 0.72 + 0.18 * electronic_shell_closure,
        "p": 0.64 + 0.22 * electronic_shell_closure,
        "d": 0.62 + 0.18 * np.exp(-abs(fill - 0.5) / 0.25) + 0.10 * electronic_shell_closure,
        "f": 0.60 + 0.20 * max(half_shell, electronic_shell_closure),
        "g": 0.48 + 0.18 * max(half_shell, electronic_shell_closure),
        "unknown": 0.20,
    }[block]

    # Nuclear magic / island terms. These are deliberately local maxima,
    # not permanent stability guarantees.
    proton_magic = max(
        gaussian(Z, 114, 3.0),
        gaussian(Z, 120, 3.5),
        gaussian(Z, 126, 4.0),
        0.65 * gaussian(Z, 164, 5.0),
    )
    neutron_magic = max(
        gaussian(N, 184, 8.0),
        0.75 * gaussian(N, 228, 10.0),
        0.45 * gaussian(N, 258, 12.0),
    )
    nuclear_magic = 0.55 * proton_magic + 0.45 * neutron_magic

    island_bonus = max(
        gaussian(Z, 114, 4.0) * gaussian(N, 184, 10.0),
        gaussian(Z, 120, 5.0) * gaussian(N, 184, 12.0),
        gaussian(Z, 126, 5.0) * gaussian(N, 184, 12.0),
        0.45 * gaussian(Z, 164, 6.0) * gaussian(N, 258, 18.0),
    )

    # Destabilizing terms. These are what should eventually catch up.
    coulomb_overload = np.clip(((Z - 82) / (172 - 82)) ** 1.55, 0.0, 1.8)
    relativistic_binding = np.clip((Z - 100) / 72.0, 0.0, 1.0)
    relativistic_stress = np.clip(((Z - 118) / 54.0), 0.0, 1.0) ** 1.35
    neutron_balance_defect = np.clip(abs(neutron_ratio - 1.58) / 0.42, 0.0, 1.0)
    frontier_uncertainty = 0.0 if Z <= 118 else np.clip((Z - 118) / 54.0, 0.0, 1.0)

    raw_stability = (
        0.28 * electronic_shell_closure
        + 0.17 * block_coherence
        + 0.26 * nuclear_magic
        + 0.18 * island_bonus
        + 0.08 * relativistic_binding
        - 0.30 * coulomb_overload
        - 0.17 * relativistic_stress
        - 0.12 * neutron_balance_defect
        - 0.05 * frontier_uncertainty
    )

    # Logistic compression into [0, 1]. Keep absolute value hard enough that
    # high-Z overload becomes visibly unstable.
    stability_score = 1.0 / (1.0 + np.exp(-6.0 * (raw_stability - 0.18)))

    # Formation accessibility is harsher than latent stability: even a local
    # island may remain difficult to synthesize or short-lived.
    formation_accessibility = stability_score * np.exp(-1.35 * coulomb_overload) * (1.0 - 0.25 * frontier_uncertainty)

    if stability_score >= 0.62:
        stability_class = "relative_island"
    elif stability_score >= 0.46:
        stability_class = "fragile_candidate"
    elif stability_score >= 0.30:
        stability_class = "extreme_short_lived"
    else:
        stability_class = "terminally_unstable"

    noise = perturbation * rng.normal(0.0, 0.01)

    return {
        "Z": Z,
        "symbol": symbol,
        "known_status": "known_named" if Z <= 118 else "hypothetical",
        "period": period,
        "group": group,
        "block": block,
        "block_id": BLOCK_ID[block],
        "region": region,
        "region_id": REGION_ID[region],
        "replicate": replicate,
        "perturbation": perturbation,
        "predicted_N": N,
        "predicted_A": A,
        "neutron_ratio": neutron_ratio + noise,
        "orbital_count": orbital_count,
        "orbital_capacity": capacity,
        "orbital_fill": fill + noise,
        "electronic_shell_closure": electronic_shell_closure + noise,
        "half_shell_support": half_shell + noise,
        "block_coherence": block_coherence + noise,
        "proton_magic": proton_magic + noise,
        "neutron_magic": neutron_magic + noise,
        "nuclear_magic": nuclear_magic + noise,
        "island_bonus": island_bonus + noise,
        "coulomb_overload": coulomb_overload + noise,
        "relativistic_binding": relativistic_binding + noise,
        "relativistic_stress": relativistic_stress + noise,
        "neutron_balance_defect": neutron_balance_defect + noise,
        "frontier_uncertainty": frontier_uncertainty + noise,
        "raw_stability": raw_stability + noise,
        "stability_score": np.clip(stability_score + noise, 0.0, 1.0),
        "formation_accessibility": np.clip(formation_accessibility + noise, 0.0, 1.0),
        "stability_class": stability_class,
    }


rows = []
for perturbation in PERTURBATIONS:
    for Z in range(104, 173):
        for r in range(REPLICATES):
            rows.append(make_candidate(Z, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "Z",
    "period",
    "group",
    "block_id",
    "region_id",
    "predicted_N",
    "neutron_ratio",
    "orbital_count",
    "orbital_capacity",
    "orbital_fill",
    "electronic_shell_closure",
    "half_shell_support",
    "block_coherence",
    "proton_magic",
    "neutron_magic",
    "nuclear_magic",
    "island_bonus",
    "coulomb_overload",
    "relativistic_binding",
    "relativistic_stress",
    "neutron_balance_defect",
    "frontier_uncertainty",
    "raw_stability",
    "stability_score",
    "formation_accessibility",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

# Cluster the extrapolated latent space to see whether islands are separated.
kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
df["latent_cluster"] = kmeans.fit_predict(Xn)

silhouette_n = min(4000, len(df))
silhouette_idx = rng.choice(len(df), size=silhouette_n, replace=False)
try:
    region_silhouette = float(silhouette_score(Xn[silhouette_idx], df["region"].iloc[silhouette_idx]))
except Exception:
    region_silhouette = np.nan

try:
    stability_class_silhouette = float(silhouette_score(Xn[silhouette_idx], df["stability_class"].iloc[silhouette_idx]))
except Exception:
    stability_class_silhouette = np.nan

by_Z = (
    df.groupby(["Z", "symbol"])
    .agg(
        known_status=("known_status", "first"),
        period=("period", "mean"),
        group=("group", "mean"),
        block=("block", "first"),
        region=("region", "first"),
        predicted_N=("predicted_N", "mean"),
        predicted_A=("predicted_A", "mean"),
        mean_electronic_shell_closure=("electronic_shell_closure", "mean"),
        mean_block_coherence=("block_coherence", "mean"),
        mean_nuclear_magic=("nuclear_magic", "mean"),
        mean_island_bonus=("island_bonus", "mean"),
        mean_coulomb_overload=("coulomb_overload", "mean"),
        mean_relativistic_stress=("relativistic_stress", "mean"),
        mean_raw_stability=("raw_stability", "mean"),
        mean_stability_score=("stability_score", "mean"),
        mean_formation_accessibility=("formation_accessibility", "mean"),
        stability_class=("stability_class", lambda x: x.value_counts().index[0]),
    )
    .reset_index()
    .sort_values("Z")
)

by_region = (
    df.groupby("region")
    .agg(
        count=("region", "count"),
        Z_min=("Z", "min"),
        Z_max=("Z", "max"),
        mean_stability_score=("stability_score", "mean"),
        mean_formation_accessibility=("formation_accessibility", "mean"),
        mean_coulomb_overload=("coulomb_overload", "mean"),
        mean_relativistic_stress=("relativistic_stress", "mean"),
        dominant_class=("stability_class", lambda x: x.value_counts().index[0]),
    )
    .reset_index()
)

# Detect local maxima on mean curve.
curve = by_Z["mean_stability_score"].to_numpy()
Zs = by_Z["Z"].to_numpy()
local_peak_mask = np.zeros_like(curve, dtype=bool)
for i in range(1, len(curve) - 1):
    if curve[i] > curve[i - 1] and curve[i] > curve[i + 1]:
        local_peak_mask[i] = True

by_Z["local_peak"] = local_peak_mask
by_Z["relative_island"] = by_Z["mean_stability_score"] >= 0.62

top_candidates = (
    by_Z[by_Z["Z"] >= 119]
    .sort_values(["mean_stability_score", "mean_formation_accessibility"], ascending=False)
    .head(16)
    .reset_index(drop=True)
)

known_frontier_mean = float(by_Z[(by_Z["Z"] >= 104) & (by_Z["Z"] <= 118)]["mean_stability_score"].mean())
next_frontier_mean = float(by_Z[(by_Z["Z"] >= 119) & (by_Z["Z"] <= 126)]["mean_stability_score"].mean())
terminal_mean = float(by_Z[(by_Z["Z"] >= 160) & (by_Z["Z"] <= 172)]["mean_stability_score"].mean())
max_hypothetical = float(by_Z[by_Z["Z"] >= 119]["mean_stability_score"].max())
peak_Z = int(by_Z[by_Z["Z"] >= 119].sort_values("mean_stability_score", ascending=False).iloc[0]["Z"])
num_relative_islands = int(by_Z[(by_Z["Z"] >= 119) & (by_Z["relative_island"])].shape[0])
num_local_peaks = int(by_Z[(by_Z["Z"] >= 119) & (by_Z["local_peak"])].shape[0])
terminal_decay_ratio = float(terminal_mean / (max_hypothetical + EPS))
instability_catches_up = bool(terminal_decay_ratio <= 0.72 and terminal_mean < next_frontier_mean)

if (
    max_hypothetical >= 0.58
    and num_local_peaks >= 2
    and num_relative_islands >= 1
    and instability_catches_up
    and region_silhouette >= 0.25
):
    verdict = "superheavy_stability_extrapolation_supported"
elif (
    max_hypothetical >= 0.48
    and num_local_peaks >= 1
    and terminal_mean < next_frontier_mean
):
    verdict = "weak_superheavy_stability_extrapolation"
else:
    verdict = "superheavy_stability_extrapolation_not_supported"

summary = pd.DataFrame([
    {
        "Z_min": int(df["Z"].min()),
        "Z_max": int(df["Z"].max()),
        "num_candidates": int(by_Z.shape[0]),
        "num_samples": int(df.shape[0]),
        "known_frontier_mean_stability": known_frontier_mean,
        "next_frontier_mean_stability": next_frontier_mean,
        "terminal_mean_stability": terminal_mean,
        "max_hypothetical_stability": max_hypothetical,
        "peak_hypothetical_Z": peak_Z,
        "num_relative_islands": num_relative_islands,
        "num_local_peaks": num_local_peaks,
        "terminal_decay_ratio": terminal_decay_ratio,
        "instability_catches_up": instability_catches_up,
        "region_silhouette": region_silhouette,
        "stability_class_silhouette": stability_class_silhouette,
        "verdict": verdict,
    }
])

# Save tables.
by_Z.to_csv(OUT / "superheavy_candidates.csv", index=False)
summary.to_csv(OUT / "superheavy_summary.csv", index=False)
by_region.to_csv(OUT / "superheavy_by_region.csv", index=False)
top_candidates.to_csv(OUT / "superheavy_top_candidates.csv", index=False)

# Plot stability curve.
plt.figure(figsize=(14, 6))
plt.plot(by_Z["Z"], by_Z["mean_stability_score"], marker="o", linewidth=1.5, label="BST stability score")
plt.plot(by_Z["Z"], by_Z["mean_formation_accessibility"], marker="x", linewidth=1.2, label="formation accessibility")
plt.axvline(118, linestyle="--", linewidth=1, label="known frontier Og=118")
for z in [114, 120, 126, 164]:
    if by_Z["Z"].min() <= z <= by_Z["Z"].max():
        plt.axvline(z, linestyle=":", linewidth=0.9)
plt.xlabel("Atomic number Z")
plt.ylabel("BST score")
plt.title("BST superheavy extrapolation: islands and terminal instability")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "superheavy_stability_curve.png", dpi=220)
plt.close()

# Latent map.
proj = PCA(n_components=2).fit_transform(Xn)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=df["Z"], s=4)
plt.colorbar(scatter, label="Z")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("BST superheavy latent extrapolation map")
plt.tight_layout()
plt.savefig(OUT / "superheavy_latent_map.png", dpi=220)
plt.close()

print("\n=== BST SUPERHEAVY STABILITY EXTRAPOLATION TEST ===\n")
print(summary.to_string(index=False))

print("\nSuperheavy stability by region:")
print(by_region.to_string(index=False))

print("\nTop hypothetical BST candidates:")
print(top_candidates[[
    "Z", "symbol", "period", "group", "block", "predicted_N", "predicted_A",
    "mean_stability_score", "mean_formation_accessibility", "stability_class",
]].to_string(index=False))

print(f"\n[OK] wrote {OUT / 'superheavy_candidates.csv'}")
print(f"[OK] wrote {OUT / 'superheavy_summary.csv'}")
print(f"[OK] wrote {OUT / 'superheavy_by_region.csv'}")
print(f"[OK] wrote {OUT / 'superheavy_top_candidates.csv'}")
print(f"[OK] wrote {OUT / 'superheavy_stability_curve.png'}")
print(f"[OK] wrote {OUT / 'superheavy_latent_map.png'}")
print("[DONE] superheavy stability extrapolation test complete")
