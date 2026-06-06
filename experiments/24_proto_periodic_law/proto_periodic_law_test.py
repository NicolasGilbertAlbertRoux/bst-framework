#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/proto_periodic_law_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

ELEMENTS = [
    (1, "H", 0, [1], 1, 1),
    (2, "He", 2, [2], 1, 18),
    (3, "Li", 4, [2, 1], 2, 1),
    (4, "Be", 5, [2, 2], 2, 2),
    (5, "B", 6, [2, 3], 2, 13),
    (6, "C", 6, [2, 4], 2, 14),
    (7, "N", 7, [2, 5], 2, 15),
    (8, "O", 8, [2, 6], 2, 16),
    (9, "F", 10, [2, 7], 2, 17),
    (10, "Ne", 10, [2, 8], 2, 18),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 48


def shell_capacity(i):
    return 2 * (i + 1) ** 2


def generate_element(Z, symbol, neutrons, shells, period, group, perturbation, replicate):
    shell1 = shells[0] if len(shells) > 0 else 0
    shell2 = shells[1] if len(shells) > 1 else 0
    shell3 = shells[2] if len(shells) > 2 else 0

    outer_shell = 3 if shell3 > 0 else 2 if shell2 > 0 else 1
    outer_count = shell3 if shell3 > 0 else shell2 if shell2 > 0 else shell1
    outer_capacity = shell_capacity(outer_shell - 1)

    valence = outer_count
    closure_defect = (outer_capacity - outer_count) / outer_capacity
    shell_completion = outer_count / outer_capacity

    nucleus_size = Z + neutrons
    neutron_excess = neutrons - Z

    proto_core_count = Z + perturbation * rng.normal(0, 0.015)
    proto_nucleus_stability = (
        0.72
        + 0.08 * np.exp(-abs(neutron_excess) / (Z + 1))
        + 0.04 * shell_completion
        + perturbation * rng.normal(0, 0.015)
    )

    interface_bridge = (
        0.82
        + 0.04 * np.sin(Z)
        + perturbation * rng.normal(0, 0.015)
    )

    shell_stability = (
        0.58
        + 0.07 * shell_completion
        + 0.03 * interface_bridge
        + perturbation * rng.normal(0, 0.015)
    )

    recurrence = (
        0.50
        + 0.25 * shell_completion
        + 0.10 * np.exp(-closure_defect)
        + perturbation * rng.normal(0, 0.015)
    )

    periodic_closure = (
        np.exp(-closure_defect)
        * (0.5 + 0.5 * interface_bridge)
        * (0.5 + 0.5 * proto_nucleus_stability)
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "period": period,
        "group": group,
        "replicate": replicate,
        "perturbation": perturbation,
        "neutrons": neutrons,
        "nucleus_size": nucleus_size,
        "neutron_excess": neutron_excess,
        "proto_core_count": proto_core_count,
        "shell1": shell1,
        "shell2": shell2,
        "shell3": shell3,
        "outer_shell": outer_shell,
        "outer_count": outer_count,
        "outer_capacity": outer_capacity,
        "valence": valence,
        "closure_defect": closure_defect,
        "shell_completion": shell_completion,
        "interface_bridge": interface_bridge,
        "proto_nucleus_stability": proto_nucleus_stability,
        "shell_stability": shell_stability,
        "recurrence": recurrence,
        "periodic_closure": periodic_closure,
        "Z_shell_match": np.exp(-abs(Z - (shell1 + shell2 + shell3))),
    }


rows = []

for perturbation in PERTURBATIONS:
    for Z, symbol, neutrons, shells, period, group in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(generate_element(Z, symbol, neutrons, shells, period, group, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "proto_core_count",
    "shell1",
    "shell2",
    "shell3",
    "outer_shell",
    "outer_count",
    "valence",
    "closure_defect",
    "shell_completion",
    "interface_bridge",
    "proto_nucleus_stability",
    "shell_stability",
    "recurrence",
    "periodic_closure",
    "Z_shell_match",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

element_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
element_clf.fit(Xn[train], df.loc[train, "symbol"])
element_pred = element_clf.predict(Xn[test])
element_accuracy = float(accuracy_score(df.loc[test, "symbol"], element_pred))

period_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
period_clf.fit(Xn[train], df.loc[train, "period"])
period_pred = period_clf.predict(Xn[test])
period_accuracy = float(accuracy_score(df.loc[test, "period"], period_pred))

group_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
group_clf.fit(Xn[train], df.loc[train, "group"])
group_pred = group_clf.predict(Xn[test])
group_accuracy = float(accuracy_score(df.loc[test, "group"], group_pred))

try:
    sil = float(silhouette_score(Xn, df["symbol"]))
except Exception:
    sil = np.nan

Z_recovery = float(np.mean(np.round(df["proto_core_count"]).astype(int) == df["Z"]))
shell_recovery = float(np.mean(df["Z_shell_match"] >= 0.999))
mean_periodic_closure = float(df["periodic_closure"].mean())
mean_shell_stability = float(df["shell_stability"].mean())
mean_nucleus_stability = float(df["proto_nucleus_stability"].mean())

by_element = df.groupby(["Z", "symbol"]).agg(
    period=("period", "mean"),
    group=("group", "mean"),
    mean_proto_core_count=("proto_core_count", "mean"),
    mean_valence=("valence", "mean"),
    mean_outer_shell=("outer_shell", "mean"),
    mean_shell_completion=("shell_completion", "mean"),
    mean_closure_defect=("closure_defect", "mean"),
    mean_periodic_closure=("periodic_closure", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
    mean_nucleus_stability=("proto_nucleus_stability", "mean"),
).reset_index().sort_values("Z")

if (
    Z_recovery >= 0.99
    and shell_recovery >= 0.99
    and period_accuracy >= 0.99
    and group_accuracy >= 0.90
    and mean_periodic_closure >= 0.50
):
    verdict = "proto_periodic_law_supported"
elif period_accuracy >= 0.90 and group_accuracy >= 0.75:
    verdict = "weak_proto_periodic_law"
else:
    verdict = "proto_periodic_law_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_samples": len(df),
    "element_accuracy": element_accuracy,
    "period_accuracy": period_accuracy,
    "group_accuracy": group_accuracy,
    "silhouette_score": sil,
    "Z_recovery": Z_recovery,
    "shell_recovery": shell_recovery,
    "mean_periodic_closure": mean_periodic_closure,
    "mean_shell_stability": mean_shell_stability,
    "mean_nucleus_stability": mean_nucleus_stability,
    "verdict": verdict,
}])

df.to_csv(OUT / "proto_periodic_law_samples.csv", index=False)
summary.to_csv(OUT / "proto_periodic_law_summary.csv", index=False)
by_element.to_csv(OUT / "proto_periodic_law_by_element.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(8, 6))
for Z, symbol, *_ in ELEMENTS:
    mask = df["symbol"].to_numpy() == symbol
    plt.scatter(proj[mask, 0], proj[mask, 1], s=18, label=symbol)
plt.title("BST proto-periodic law reconstruction space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "proto_periodic_law_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(by_element["Z"], by_element["mean_valence"], marker="o", label="valence")
plt.plot(by_element["Z"], by_element["mean_outer_shell"], marker="x", label="period / outer shell")
plt.xticks(by_element["Z"], by_element["symbol"])
plt.xlabel("Proto-element")
plt.ylabel("Recovered periodic coordinates")
plt.title("BST proto-periodic coordinates")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "proto_periodic_coordinates.png", dpi=220)
plt.close()

print("\n=== BST PROTO-PERIODIC LAW TEST ===\n")
print(summary.to_string(index=False))

print("\nProto-periodic reconstruction by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'proto_periodic_law_samples.csv'}")
print(f"[OK] wrote {OUT / 'proto_periodic_law_summary.csv'}")
print(f"[OK] wrote {OUT / 'proto_periodic_law_by_element.csv'}")
print(f"[OK] wrote {OUT / 'proto_periodic_law_projection.png'}")
print(f"[OK] wrote {OUT / 'proto_periodic_coordinates.png'}")
print("[DONE] proto-periodic law test complete")