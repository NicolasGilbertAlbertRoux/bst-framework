#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/transition_metal_recurrence_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# Z, symbol, d_count, s_count, group
TRANSITION_ELEMENTS = [
    (21, "Sc", 1, 2, 3),
    (22, "Ti", 2, 2, 4),
    (23, "V",  3, 2, 5),
    (24, "Cr", 5, 1, 6),   # anomaly
    (25, "Mn", 5, 2, 7),
    (26, "Fe", 6, 2, 8),
    (27, "Co", 7, 2, 9),
    (28, "Ni", 8, 2, 10),
    (29, "Cu", 10, 1, 11), # anomaly
    (30, "Zn", 10, 2, 12),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 96


def make_sample(Z, symbol, d_count, s_count, group, perturbation, replicate):
    d_fill = d_count / 10.0
    s_fill = s_count / 2.0

    half_filled = np.exp(-abs(d_count - 5) / 2.0)
    filled = np.exp(-abs(d_count - 10) / 2.0)
    d_symmetry = max(half_filled, filled)

    anomaly_flag = int(symbol in ["Cr", "Cu"])

    exchange_stabilization = (
        0.45 * half_filled
        + 0.35 * filled
        + 0.20 * anomaly_flag
        + perturbation * rng.normal(0, 0.01)
    )

    coordination_drive = (
        0.35 * (1.0 - abs(d_fill - 0.5))
        + 0.25 * (1.0 - filled)
        + 0.20 * s_fill
        + 0.20 * np.exp(-abs(group - 8) / 5.0)
        + perturbation * rng.normal(0, 0.01)
    )

    magnetic_moment_proxy = (
        min(d_count, 10 - d_count) / 5.0
        + perturbation * rng.normal(0, 0.01)
    )

    transition_recurrence = (
        0.30 * d_fill
        + 0.20 * s_fill
        + 0.20 * d_symmetry
        + 0.15 * coordination_drive
        + 0.15 * exchange_stabilization
        + perturbation * rng.normal(0, 0.01)
    )

    d_block_stability = (
        0.30 * transition_recurrence
        + 0.25 * coordination_drive
        + 0.20 * d_symmetry
        + 0.15 * exchange_stabilization
        + 0.10 * (1.0 - abs(d_fill - 0.5))
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "group": group,
        "replicate": replicate,
        "perturbation": perturbation,
        "d_count": d_count,
        "s_count": s_count,
        "d_fill": d_fill,
        "s_fill": s_fill,
        "half_filled": half_filled,
        "filled": filled,
        "d_symmetry": d_symmetry,
        "anomaly_flag": anomaly_flag,
        "exchange_stabilization": exchange_stabilization,
        "coordination_drive": coordination_drive,
        "magnetic_moment_proxy": magnetic_moment_proxy,
        "transition_recurrence": transition_recurrence,
        "d_block_stability": d_block_stability,
    }


rows = []
for perturbation in PERTURBATIONS:
    for e in TRANSITION_ELEMENTS:
        for r in range(REPLICATES):
            rows.append(make_sample(*e, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "group",
    "d_count",
    "s_count",
    "d_fill",
    "s_fill",
    "half_filled",
    "filled",
    "d_symmetry",
    "exchange_stabilization",
    "coordination_drive",
    "magnetic_moment_proxy",
    "transition_recurrence",
    "d_block_stability",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

def knn_acc(label):
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf.fit(Xn[train], df.loc[train, label])
    return float(accuracy_score(df.loc[test, label], clf.predict(Xn[test])))

element_accuracy = knn_acc("symbol")
group_accuracy = knn_acc("group")
anomaly_accuracy = knn_acc("anomaly_flag")

try:
    element_silhouette = float(silhouette_score(Xn, df["symbol"]))
except Exception:
    element_silhouette = np.nan

try:
    anomaly_silhouette = float(silhouette_score(Xn, df["anomaly_flag"]))
except Exception:
    anomaly_silhouette = np.nan

anomaly = df[df["anomaly_flag"] == 1]
normal = df[df["anomaly_flag"] == 0]

anomaly_exchange_contrast = float(
    anomaly["exchange_stabilization"].mean()
    / (normal["exchange_stabilization"].mean() + EPS)
)

half_filled_strength = float(df[df["d_count"] == 5]["d_symmetry"].mean())
filled_strength = float(df[df["d_count"] == 10]["d_symmetry"].mean())
transition_stability = float(df["d_block_stability"].mean())

by_element = df.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    group=("group", "mean"),
    mean_d_count=("d_count", "mean"),
    mean_s_count=("s_count", "mean"),
    mean_d_fill=("d_fill", "mean"),
    mean_half_filled=("half_filled", "mean"),
    mean_filled=("filled", "mean"),
    mean_d_symmetry=("d_symmetry", "mean"),
    mean_exchange_stabilization=("exchange_stabilization", "mean"),
    mean_coordination_drive=("coordination_drive", "mean"),
    mean_magnetic_moment_proxy=("magnetic_moment_proxy", "mean"),
    mean_transition_recurrence=("transition_recurrence", "mean"),
    mean_d_block_stability=("d_block_stability", "mean"),
    anomaly_flag=("anomaly_flag", "mean"),
).reset_index().sort_values("Z")

if (
    element_accuracy >= 0.95
    and group_accuracy >= 0.95
    and anomaly_accuracy >= 0.95
    and half_filled_strength >= 0.95
    and filled_strength >= 0.95
    and transition_stability >= 0.45
):
    verdict = "transition_metal_recurrence_supported"
elif (
    group_accuracy >= 0.85
    and transition_stability >= 0.35
):
    verdict = "weak_transition_metal_recurrence"
else:
    verdict = "transition_metal_recurrence_not_supported"

summary = pd.DataFrame([{
    "num_transition_elements": len(TRANSITION_ELEMENTS),
    "num_samples": len(df),
    "element_accuracy": element_accuracy,
    "group_accuracy": group_accuracy,
    "anomaly_accuracy": anomaly_accuracy,
    "element_silhouette": element_silhouette,
    "anomaly_silhouette": anomaly_silhouette,
    "anomaly_exchange_contrast": anomaly_exchange_contrast,
    "half_filled_strength": half_filled_strength,
    "filled_strength": filled_strength,
    "mean_transition_stability": transition_stability,
    "verdict": verdict,
}])

df.to_csv(OUT / "transition_metal_samples.csv", index=False)
summary.to_csv(OUT / "transition_metal_summary.csv", index=False)
by_element.to_csv(OUT / "transition_metal_by_element.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(9, 6))
for symbol in [e[1] for e in TRANSITION_ELEMENTS]:
    mask = df["symbol"].to_numpy() == symbol
    plt.scatter(proj[mask, 0], proj[mask, 1], s=12, label=symbol)
plt.title("BST transition metal recurrence space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "transition_metal_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(by_element["symbol"], by_element["mean_d_block_stability"], marker="o", label="d-block stability")
plt.plot(by_element["symbol"], by_element["mean_exchange_stabilization"], marker="x", label="exchange stabilization")
plt.ylabel("BST score")
plt.title("BST transition metal recurrence")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "transition_metal_recurrence.png", dpi=220)
plt.close()

print("\n=== BST TRANSITION METAL RECURRENCE TEST ===\n")
print(summary.to_string(index=False))

print("\nTransition metal recurrence by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'transition_metal_samples.csv'}")
print(f"[OK] wrote {OUT / 'transition_metal_summary.csv'}")
print(f"[OK] wrote {OUT / 'transition_metal_by_element.csv'}")
print(f"[OK] wrote {OUT / 'transition_metal_projection.png'}")
print(f"[OK] wrote {OUT / 'transition_metal_recurrence.png'}")
print("[DONE] transition metal recurrence test complete")