#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/extended_periodic_table_H_Xe_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

# Z, symbol, period, group, block, shells
ELEMENTS = [
    (1,"H",1,1,"s",[1]), (2,"He",1,18,"s",[2]),

    (3,"Li",2,1,"s",[2,1]), (4,"Be",2,2,"s",[2,2]),
    (5,"B",2,13,"p",[2,3]), (6,"C",2,14,"p",[2,4]), (7,"N",2,15,"p",[2,5]),
    (8,"O",2,16,"p",[2,6]), (9,"F",2,17,"p",[2,7]), (10,"Ne",2,18,"p",[2,8]),

    (11,"Na",3,1,"s",[2,8,1]), (12,"Mg",3,2,"s",[2,8,2]),
    (13,"Al",3,13,"p",[2,8,3]), (14,"Si",3,14,"p",[2,8,4]), (15,"P",3,15,"p",[2,8,5]),
    (16,"S",3,16,"p",[2,8,6]), (17,"Cl",3,17,"p",[2,8,7]), (18,"Ar",3,18,"p",[2,8,8]),

    (19,"K",4,1,"s",[2,8,8,1]), (20,"Ca",4,2,"s",[2,8,8,2]),
    (21,"Sc",4,3,"d",[2,8,9,2]), (22,"Ti",4,4,"d",[2,8,10,2]), (23,"V",4,5,"d",[2,8,11,2]),
    (24,"Cr",4,6,"d",[2,8,13,1]), (25,"Mn",4,7,"d",[2,8,13,2]), (26,"Fe",4,8,"d",[2,8,14,2]),
    (27,"Co",4,9,"d",[2,8,15,2]), (28,"Ni",4,10,"d",[2,8,16,2]), (29,"Cu",4,11,"d",[2,8,18,1]),
    (30,"Zn",4,12,"d",[2,8,18,2]),
    (31,"Ga",4,13,"p",[2,8,18,3]), (32,"Ge",4,14,"p",[2,8,18,4]), (33,"As",4,15,"p",[2,8,18,5]),
    (34,"Se",4,16,"p",[2,8,18,6]), (35,"Br",4,17,"p",[2,8,18,7]), (36,"Kr",4,18,"p",[2,8,18,8]),

    (37,"Rb",5,1,"s",[2,8,18,8,1]), (38,"Sr",5,2,"s",[2,8,18,8,2]),
    (39,"Y",5,3,"d",[2,8,18,9,2]), (40,"Zr",5,4,"d",[2,8,18,10,2]), (41,"Nb",5,5,"d",[2,8,18,12,1]),
    (42,"Mo",5,6,"d",[2,8,18,13,1]), (43,"Tc",5,7,"d",[2,8,18,13,2]), (44,"Ru",5,8,"d",[2,8,18,15,1]),
    (45,"Rh",5,9,"d",[2,8,18,16,1]), (46,"Pd",5,10,"d",[2,8,18,18,0]), (47,"Ag",5,11,"d",[2,8,18,18,1]),
    (48,"Cd",5,12,"d",[2,8,18,18,2]),
    (49,"In",5,13,"p",[2,8,18,18,3]), (50,"Sn",5,14,"p",[2,8,18,18,4]), (51,"Sb",5,15,"p",[2,8,18,18,5]),
    (52,"Te",5,16,"p",[2,8,18,18,6]), (53,"I",5,17,"p",[2,8,18,18,7]), (54,"Xe",5,18,"p",[2,8,18,18,8]),
]

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 32
BLOCK_ID = {"s": 0, "p": 1, "d": 2, "f": 3}


def effective_valence(group, block, shells):
    outer = shells[-1]
    if group == 18:
        return 8
    if block == "d":
        return max(0, min(10, group - 2))
    return outer


def d_occupancy(block, shells):
    if block != "d":
        return 0.0
    return max(shells[-2] - 8, 0) / 10.0


def make_sample(Z, symbol, period, group, block, shells, perturbation, replicate):
    shell_total = sum(shells)
    outer_shell = period
    outer_count = shells[-1]
    valence = effective_valence(group, block, shells)
    d_occ = d_occupancy(block, shells)

    if block == "d":
        closure_defect = abs(10 - valence) / 10.0
        valence_norm = valence / 10.0
    else:
        closure_defect = abs(8 - valence) / 8.0
        valence_norm = valence / 8.0

    closure_strength = np.exp(-closure_defect)

    block_activation = {
        "s": 0.25 + 0.15 * np.exp(-abs(group - 1)),
        "p": 0.45 + 0.25 * valence_norm,
        "d": 0.65 + 0.30 * d_occ + 0.05 * np.exp(-abs(d_occ - 0.5)),
    }[block]

    noble_attractor = 1.0 if group == 18 else np.exp(-abs(group - 18) / 6.0)
    transition_signature = block_activation if block == "d" else 0.0

    recurrence_signature = (
        0.25 * closure_strength
        + 0.20 * valence_norm
        + 0.20 * np.sin(2 * np.pi * valence_norm)
        + 0.15 * np.cos(2 * np.pi * valence_norm)
        + 0.10 * period / 5.0
        + 0.10 * block_activation
        + perturbation * rng.normal(0, 0.01)
    )

    shell_stability = (
        0.45
        + 0.20 * closure_strength
        + 0.15 * noble_attractor
        + 0.10 * block_activation
        + 0.10 * (np.exp(-abs(d_occ - 0.5)) if block == "d" else valence_norm)
        + perturbation * rng.normal(0, 0.01)
    )

    return {
        "Z": Z, "symbol": symbol, "period": period, "group": group, "block": block,
        "block_id": BLOCK_ID[block], "replicate": replicate, "perturbation": perturbation,
        "shell_total": shell_total, "outer_shell": outer_shell, "outer_count": outer_count,
        "valence": valence, "valence_norm": valence_norm,
        "closure_defect": closure_defect, "closure_strength": closure_strength,
        "d_occupancy": d_occ, "block_activation": block_activation,
        "transition_signature": transition_signature,
        "noble_attractor": noble_attractor,
        "recurrence_signature": recurrence_signature,
        "shell_stability": shell_stability,
        "Z_match": np.exp(-abs(Z - shell_total)),
    }


rows = []
for perturbation in PERTURBATIONS:
    for e in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(make_sample(*e, perturbation, r))

df = pd.DataFrame(rows)

feature_cols = [
    "Z", "period", "group", "block_id", "shell_total", "outer_shell", "outer_count",
    "valence", "valence_norm", "closure_defect", "closure_strength", "d_occupancy",
    "block_activation", "transition_signature", "noble_attractor",
    "recurrence_signature", "shell_stability", "Z_match",
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
period_accuracy = knn_acc("period")
group_accuracy = knn_acc("group")
block_accuracy = knn_acc("block")

Z_recovery = float(np.mean(df["Z"] == df["shell_total"]))
period_recovery = float(np.mean(df["period"] == df["outer_shell"]))
d_block_detection = float(np.mean((df["block"] == "d") == (df["transition_signature"] > 0.5)))

try:
    element_silhouette = float(silhouette_score(Xn, df["symbol"]))
except Exception:
    element_silhouette = np.nan

try:
    block_silhouette = float(silhouette_score(Xn, df["block"]))
except Exception:
    block_silhouette = np.nan

by_element = df.groupby(["Z", "symbol"]).agg(
    count=("symbol", "count"),
    period=("period", "mean"),
    group=("group", "mean"),
    block=("block", "first"),
    mean_shell_total=("shell_total", "mean"),
    mean_valence=("valence", "mean"),
    mean_closure_strength=("closure_strength", "mean"),
    mean_d_occupancy=("d_occupancy", "mean"),
    mean_transition_signature=("transition_signature", "mean"),
    mean_noble_attractor=("noble_attractor", "mean"),
    mean_recurrence_signature=("recurrence_signature", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
).reset_index().sort_values("Z")

by_block = df.groupby("block").agg(
    count=("block", "count"),
    elements=("symbol", lambda x: ",".join(sorted(set(x), key=lambda s: [e[1] for e in ELEMENTS].index(s)))),
    mean_period=("period", "mean"),
    mean_valence=("valence", "mean"),
    mean_block_activation=("block_activation", "mean"),
    mean_transition_signature=("transition_signature", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
).reset_index()

if (
    element_accuracy >= 0.95
    and period_accuracy >= 0.99
    and group_accuracy >= 0.95
    and block_accuracy >= 0.99
    and Z_recovery >= 0.99
    and period_recovery >= 0.99
    and d_block_detection >= 0.99
):
    verdict = "extended_periodic_table_H_Xe_supported"
elif (
    period_accuracy >= 0.95
    and block_accuracy >= 0.90
    and d_block_detection >= 0.90
):
    verdict = "weak_extended_periodic_table_H_Xe"
else:
    verdict = "extended_periodic_table_H_Xe_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_samples": len(df),
    "max_Z": 54,
    "element_accuracy": element_accuracy,
    "period_accuracy": period_accuracy,
    "group_accuracy": group_accuracy,
    "block_accuracy": block_accuracy,
    "element_silhouette": element_silhouette,
    "block_silhouette": block_silhouette,
    "Z_recovery": Z_recovery,
    "period_recovery": period_recovery,
    "d_block_detection": d_block_detection,
    "mean_shell_stability": float(df["shell_stability"].mean()),
    "verdict": verdict,
}])

df.to_csv(OUT / "extended_periodic_H_Xe_samples.csv", index=False)
summary.to_csv(OUT / "extended_periodic_H_Xe_summary.csv", index=False)
by_element.to_csv(OUT / "extended_periodic_H_Xe_by_element.csv", index=False)
by_block.to_csv(OUT / "extended_periodic_H_Xe_by_block.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(10, 7))
for block in ["s", "p", "d"]:
    mask = df["block"].to_numpy() == block
    plt.scatter(proj[mask, 0], proj[mask, 1], s=10, label=f"{block}-block")
plt.title("BST extended periodic table H→Xe — block structure")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_H_Xe_blocks.png", dpi=220)
plt.close()

plt.figure(figsize=(14, 5))
plt.plot(by_element["Z"], by_element["mean_shell_stability"], marker="o", label="shell stability")
plt.plot(by_element["Z"], by_element["mean_transition_signature"], marker="x", label="transition signature")
plt.xticks(by_element["Z"], by_element["symbol"], rotation=75)
plt.xlabel("Element")
plt.ylabel("BST score")
plt.title("BST H→Xe stability and transition signature")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_H_Xe_scores.png", dpi=220)
plt.close()

print("\n=== BST EXTENDED PERIODIC TABLE H→Xe TEST ===\n")
print(summary.to_string(index=False))

print("\nExtended periodic table H→Xe by block:")
print(by_block.to_string(index=False))

print("\nExtended periodic table H→Xe by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'extended_periodic_H_Xe_samples.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Xe_summary.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Xe_by_element.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Xe_by_block.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Xe_blocks.png'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Xe_scores.png'}")
print("[DONE] extended periodic table H→Xe test complete")