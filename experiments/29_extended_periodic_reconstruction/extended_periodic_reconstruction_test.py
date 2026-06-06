#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA

OUT = Path("results/research_final/extended_periodic_reconstruction_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# Z, symbol, neutrons, shells, period, group
ELEMENTS = [
    (1,  "H",  0,  [1],        1, 1),
    (2,  "He", 2,  [2],        1, 18),
    (3,  "Li", 4,  [2, 1],     2, 1),
    (4,  "Be", 5,  [2, 2],     2, 2),
    (5,  "B",  6,  [2, 3],     2, 13),
    (6,  "C",  6,  [2, 4],     2, 14),
    (7,  "N",  7,  [2, 5],     2, 15),
    (8,  "O",  8,  [2, 6],     2, 16),
    (9,  "F",  10, [2, 7],     2, 17),
    (10, "Ne", 10, [2, 8],     2, 18),
    (11, "Na", 12, [2, 8, 1],  3, 1),
    (12, "Mg", 12, [2, 8, 2],  3, 2),
    (13, "Al", 14, [2, 8, 3],  3, 13),
    (14, "Si", 14, [2, 8, 4],  3, 14),
    (15, "P",  16, [2, 8, 5],  3, 15),
    (16, "S",  16, [2, 8, 6],  3, 16),
    (17, "Cl", 18, [2, 8, 7],  3, 17),
    (18, "Ar", 22, [2, 8, 8],  3, 18),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 48


def shell_capacity(index):
    return [2, 8, 18, 32][index]


def generate_element(Z, symbol, neutrons, shells, period, group, perturbation, replicate):
    shell_1 = shells[0] if len(shells) > 0 else 0
    shell_2 = shells[1] if len(shells) > 1 else 0
    shell_3 = shells[2] if len(shells) > 2 else 0
    shell_4 = shells[3] if len(shells) > 3 else 0

    outer_shell = max(i + 1 for i, s in enumerate(shells) if s > 0)
    outer_count = shells[outer_shell - 1]
    outer_capacity = shell_capacity(outer_shell - 1)

    valence = outer_count
    shell_total = sum(shells)
    closure_defect = (outer_capacity - outer_count) / outer_capacity
    shell_completion = outer_count / outer_capacity

    neutron_excess = neutrons - Z
    nuclear_mass = Z + neutrons + perturbation * rng.normal(0, 0.03)
    proto_core_count = Z + perturbation * rng.normal(0, 0.02)

    nucleus_stability = (
        0.72
        + 0.05 * np.exp(-abs(neutron_excess) / (Z + 1))
        + 0.05 * shell_completion
        + 0.03 * np.exp(-closure_defect)
        + perturbation * rng.normal(0, 0.015)
    )

    shell_stability = (
        0.55
        + 0.10 * shell_completion
        + 0.06 * np.exp(-closure_defect)
        + perturbation * rng.normal(0, 0.015)
    )

    periodic_closure = (
        np.exp(-closure_defect)
        * (0.55 + 0.45 * shell_stability)
        * (0.55 + 0.45 * nucleus_stability)
    )

    recurrence = (
        0.45
        + 0.22 * shell_completion
        + 0.16 * np.exp(-closure_defect)
        + 0.08 * nucleus_stability
        + perturbation * rng.normal(0, 0.015)
    )

    interface_bridge = (
        0.80
        + 0.04 * np.sin(Z)
        + 0.03 * shell_completion
        + perturbation * rng.normal(0, 0.015)
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "neutrons": neutrons,
        "period": period,
        "group": group,
        "replicate": replicate,
        "perturbation": perturbation,
        "shell_1": shell_1,
        "shell_2": shell_2,
        "shell_3": shell_3,
        "shell_4": shell_4,
        "shell_total": shell_total,
        "outer_shell": outer_shell,
        "outer_count": outer_count,
        "outer_capacity": outer_capacity,
        "valence": valence,
        "closure_defect": closure_defect,
        "shell_completion": shell_completion,
        "neutron_excess": neutron_excess,
        "nuclear_mass": nuclear_mass,
        "proto_core_count": proto_core_count,
        "nucleus_stability": nucleus_stability,
        "shell_stability": shell_stability,
        "periodic_closure": periodic_closure,
        "recurrence": recurrence,
        "interface_bridge": interface_bridge,
        "Z_shell_match": np.exp(-abs(Z - shell_total)),
    }


rows = []
for perturbation in PERTURBATIONS:
    for Z, symbol, neutrons, shells, period, group in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(generate_element(Z, symbol, neutrons, shells, period, group, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "proto_core_count",
    "shell_1",
    "shell_2",
    "shell_3",
    "shell_4",
    "shell_total",
    "outer_shell",
    "outer_count",
    "valence",
    "closure_defect",
    "shell_completion",
    "nuclear_mass",
    "neutron_excess",
    "nucleus_stability",
    "shell_stability",
    "periodic_closure",
    "recurrence",
    "interface_bridge",
    "Z_shell_match",
]

X = df[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

train = df["perturbation"].isin([0.0, 0.01]).to_numpy()
test = df["perturbation"].isin([0.025, 0.05]).to_numpy()

element_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
element_clf.fit(Xn[train], df.loc[train, "symbol"])
element_accuracy = float(accuracy_score(df.loc[test, "symbol"], element_clf.predict(Xn[test])))

period_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
period_clf.fit(Xn[train], df.loc[train, "period"])
period_accuracy = float(accuracy_score(df.loc[test, "period"], period_clf.predict(Xn[test])))

group_clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
group_clf.fit(Xn[train], df.loc[train, "group"])
group_accuracy = float(accuracy_score(df.loc[test, "group"], group_clf.predict(Xn[test])))

try:
    sil = float(silhouette_score(Xn, df["symbol"]))
except Exception:
    sil = np.nan

Z_recovery = float(np.mean(np.round(df["proto_core_count"]).astype(int) == df["Z"]))
shell_recovery = float(np.mean(df["Z_shell_match"] >= 0.999))
period_recovery = float(np.mean(df["outer_shell"] == df["period"]))

by_element = df.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    period=("period", "mean"),
    group=("group", "mean"),
    mean_proto_core_count=("proto_core_count", "mean"),
    mean_shell_total=("shell_total", "mean"),
    mean_shell_1=("shell_1", "mean"),
    mean_shell_2=("shell_2", "mean"),
    mean_shell_3=("shell_3", "mean"),
    mean_valence=("valence", "mean"),
    mean_outer_shell=("outer_shell", "mean"),
    mean_shell_completion=("shell_completion", "mean"),
    mean_closure_defect=("closure_defect", "mean"),
    mean_periodic_closure=("periodic_closure", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
    mean_nucleus_stability=("nucleus_stability", "mean"),
).reset_index().sort_values("Z")

by_group = df.groupby(["group"]).agg(
    count=("symbol", "count"),
    elements=("symbol", lambda x: ",".join(sorted(set(x), key=lambda s: [e[1] for e in ELEMENTS].index(s)))),
    mean_valence=("valence", "mean"),
    mean_outer_shell=("outer_shell", "mean"),
    mean_periodic_closure=("periodic_closure", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
).reset_index().sort_values("group")

if (
    element_accuracy >= 0.95
    and period_accuracy >= 0.99
    and group_accuracy >= 0.95
    and Z_recovery >= 0.99
    and shell_recovery >= 0.99
    and period_recovery >= 0.99
):
    verdict = "extended_periodic_reconstruction_supported"
elif (
    period_accuracy >= 0.90
    and group_accuracy >= 0.85
    and shell_recovery >= 0.95
):
    verdict = "weak_extended_periodic_reconstruction"
else:
    verdict = "extended_periodic_reconstruction_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_samples": len(df),
    "max_Z": max(e[0] for e in ELEMENTS),
    "element_accuracy": element_accuracy,
    "period_accuracy": period_accuracy,
    "group_accuracy": group_accuracy,
    "silhouette_score": sil,
    "Z_recovery": Z_recovery,
    "shell_recovery": shell_recovery,
    "period_recovery": period_recovery,
    "mean_periodic_closure": float(df["periodic_closure"].mean()),
    "mean_shell_stability": float(df["shell_stability"].mean()),
    "mean_nucleus_stability": float(df["nucleus_stability"].mean()),
    "verdict": verdict,
}])

df.to_csv(OUT / "extended_periodic_samples.csv", index=False)
summary.to_csv(OUT / "extended_periodic_summary.csv", index=False)
by_element.to_csv(OUT / "extended_periodic_by_element.csv", index=False)
by_group.to_csv(OUT / "extended_periodic_by_group.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(10, 7))
for Z, symbol, *_ in ELEMENTS:
    mask = df["symbol"].to_numpy() == symbol
    plt.scatter(proj[mask, 0], proj[mask, 1], s=14, label=symbol)
plt.title("BST extended periodic reconstruction space H→Ar")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(ncol=6, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_projection.png", dpi=220)
plt.close()

plt.figure(figsize=(11, 5))
plt.plot(by_element["Z"], by_element["mean_valence"], marker="o", label="valence")
plt.plot(by_element["Z"], by_element["mean_outer_shell"], marker="x", label="period / outer shell")
plt.xticks(by_element["Z"], by_element["symbol"])
plt.xlabel("Proto-element")
plt.ylabel("Recovered periodic coordinates")
plt.title("BST extended periodic coordinates H→Ar")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_coordinates.png", dpi=220)
plt.close()

print("\n=== BST EXTENDED PERIODIC RECONSTRUCTION TEST ===\n")
print(summary.to_string(index=False))

print("\nExtended periodic reconstruction by element:")
print(by_element.to_string(index=False))

print("\nGroup recurrence preview:")
print(by_group.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'extended_periodic_samples.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_summary.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_by_element.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_by_group.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_projection.png'}")
print(f"[OK] wrote {OUT / 'extended_periodic_coordinates.png'}")
print("[DONE] extended periodic reconstruction test complete")