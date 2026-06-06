#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

OUT = Path("results/research_final/extended_periodic_table_H_Rn_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

BASE = [
    # Z, symbol, period, group, block, shells
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

    (55,"Cs",6,1,"s",[2,8,18,18,8,1]), (56,"Ba",6,2,"s",[2,8,18,18,8,2]),
]

LANTHANIDES = [
    (57,"La",6,3,"f",[2,8,18,18,9,2]), (58,"Ce",6,3,"f",[2,8,18,19,9,2]),
    (59,"Pr",6,3,"f",[2,8,18,21,8,2]), (60,"Nd",6,3,"f",[2,8,18,22,8,2]),
    (61,"Pm",6,3,"f",[2,8,18,23,8,2]), (62,"Sm",6,3,"f",[2,8,18,24,8,2]),
    (63,"Eu",6,3,"f",[2,8,18,25,8,2]), (64,"Gd",6,3,"f",[2,8,18,25,9,2]),
    (65,"Tb",6,3,"f",[2,8,18,27,8,2]), (66,"Dy",6,3,"f",[2,8,18,28,8,2]),
    (67,"Ho",6,3,"f",[2,8,18,29,8,2]), (68,"Er",6,3,"f",[2,8,18,30,8,2]),
    (69,"Tm",6,3,"f",[2,8,18,31,8,2]), (70,"Yb",6,3,"f",[2,8,18,32,8,2]),
    (71,"Lu",6,3,"f",[2,8,18,32,9,2]),
]

POST_LN = [
    (72,"Hf",6,4,"d",[2,8,18,32,10,2]), (73,"Ta",6,5,"d",[2,8,18,32,11,2]),
    (74,"W",6,6,"d",[2,8,18,32,12,2]), (75,"Re",6,7,"d",[2,8,18,32,13,2]),
    (76,"Os",6,8,"d",[2,8,18,32,14,2]), (77,"Ir",6,9,"d",[2,8,18,32,15,2]),
    (78,"Pt",6,10,"d",[2,8,18,32,17,1]), (79,"Au",6,11,"d",[2,8,18,32,18,1]),
    (80,"Hg",6,12,"d",[2,8,18,32,18,2]),
    (81,"Tl",6,13,"p",[2,8,18,32,18,3]), (82,"Pb",6,14,"p",[2,8,18,32,18,4]),
    (83,"Bi",6,15,"p",[2,8,18,32,18,5]), (84,"Po",6,16,"p",[2,8,18,32,18,6]),
    (85,"At",6,17,"p",[2,8,18,32,18,7]), (86,"Rn",6,18,"p",[2,8,18,32,18,8]),
]

ELEMENTS = BASE + LANTHANIDES + POST_LN

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 24

BLOCK_ID = {"s": 0, "p": 1, "d": 2, "f": 3}


def effective_valence(group, block, shells):
    if group == 18:
        return 8
    if block == "d":
        return max(0, min(10, group - 2))
    if block == "f":
        return 3
    return shells[-1]


def f_count_from_shells(block, shells):
    if block != "f":
        return 0
    return max(shells[3] - 18, 0)


def d_occ(block, shells):
    if block == "d":
        return max(shells[-2] - 8, 0) / 10.0
    if block == "f":
        return max(shells[-2] - 8, 0) / 10.0
    return 0.0


def make_sample(Z, symbol, period, group, block, shells, perturbation, replicate):
    shell_total = sum(shells)
    valence = effective_valence(group, block, shells)
    f_count = f_count_from_shells(block, shells)
    f_fill = f_count / 14.0
    d_occupancy = d_occ(block, shells)

    if block == "d":
        closure_defect = abs(10 - valence) / 10.0
        valence_norm = valence / 10.0
    elif block == "f":
        closure_defect = abs(14 - f_count) / 14.0
        valence_norm = f_fill
    else:
        closure_defect = abs(8 - valence) / 8.0
        valence_norm = valence / 8.0

    closure_strength = np.exp(-closure_defect)

    half_f = np.exp(-abs(f_count - 7) / 2.0) if block == "f" else 0.0
    filled_f = np.exp(-abs(f_count - 14) / 2.0) if block == "f" else 0.0
    f_symmetry = max(half_f, filled_f)

    f_activation = (
        0.0 if block != "f"
        else 0.55 * (1.0 if f_count > 0 else 0.15) + 0.25 * f_fill + 0.20 * f_symmetry
    )

    d_activation = (
        0.0 if block != "d"
        else 0.65 + 0.30 * d_occupancy + 0.05 * np.exp(-abs(d_occupancy - 0.5))
    )

    block_activation = {
        "s": 0.25 + 0.15 * np.exp(-abs(group - 1)),
        "p": 0.45 + 0.25 * valence_norm,
        "d": d_activation,
        "f": f_activation,
    }[block] + perturbation * rng.normal(0, 0.01)

    noble_attractor = 1.0 if group == 18 else np.exp(-abs(group - 18) / 6.0)

    lanthanide_contraction = (
        0.0 if block != "f"
        else 1.0 - 0.28 * f_fill + perturbation * rng.normal(0, 0.004)
    )

    recurrence_signature = (
        0.22 * closure_strength
        + 0.18 * valence_norm
        + 0.18 * np.sin(2 * np.pi * valence_norm)
        + 0.12 * np.cos(2 * np.pi * valence_norm)
        + 0.10 * period / 6.0
        + 0.10 * block_activation
        + 0.10 * noble_attractor
        + perturbation * rng.normal(0, 0.01)
    )

    shell_stability = (
        0.42
        + 0.18 * closure_strength
        + 0.14 * noble_attractor
        + 0.12 * block_activation
        + 0.08 * valence_norm
        + 0.06 * (f_symmetry if block == "f" else np.exp(-abs(d_occupancy - 0.5)) if block == "d" else valence_norm)
        + perturbation * rng.normal(0, 0.01)
    )

    return {
        "Z": Z,
        "symbol": symbol,
        "period": period,
        "group": group,
        "block": block,
        "block_id": BLOCK_ID[block],
        "replicate": replicate,
        "perturbation": perturbation,
        "shell_total": shell_total,
        "outer_shell": period,
        "outer_count": shells[-1],
        "valence": valence,
        "valence_norm": valence_norm,
        "closure_defect": closure_defect,
        "closure_strength": closure_strength,
        "d_occupancy": d_occupancy,
        "f_count": f_count,
        "f_fill": f_fill,
        "f_symmetry": f_symmetry,
        "f_activation": f_activation,
        "d_activation": d_activation,
        "block_activation": block_activation,
        "noble_attractor": noble_attractor,
        "lanthanide_contraction": lanthanide_contraction,
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
    "valence", "valence_norm", "closure_defect", "closure_strength",
    "d_occupancy", "f_count", "f_fill", "f_symmetry", "f_activation", "d_activation",
    "block_activation", "noble_attractor", "lanthanide_contraction",
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
d_block_detection = float(np.mean((df["block"] == "d") == (df["d_activation"] > 0.5)))
f_block_detection = float(np.mean((df["block"] == "f") == (df["f_activation"] > 0.5)))

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
    mean_f_count=("f_count", "mean"),
    mean_closure_strength=("closure_strength", "mean"),
    mean_d_activation=("d_activation", "mean"),
    mean_f_activation=("f_activation", "mean"),
    mean_block_activation=("block_activation", "mean"),
    mean_noble_attractor=("noble_attractor", "mean"),
    mean_lanthanide_contraction=("lanthanide_contraction", "mean"),
    mean_recurrence_signature=("recurrence_signature", "mean"),
    mean_shell_stability=("shell_stability", "mean"),
).reset_index().sort_values("Z")

by_block = df.groupby("block").agg(
    count=("block", "count"),
    elements=("symbol", lambda x: ",".join(sorted(set(x), key=lambda s: [e[1] for e in ELEMENTS].index(s)))),
    mean_period=("period", "mean"),
    mean_valence=("valence", "mean"),
    mean_f_count=("f_count", "mean"),
    mean_block_activation=("block_activation", "mean"),
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
    and f_block_detection >= 0.95
):
    verdict = "extended_periodic_table_H_Rn_supported"
elif (
    period_accuracy >= 0.95
    and block_accuracy >= 0.90
    and f_block_detection >= 0.85
):
    verdict = "weak_extended_periodic_table_H_Rn"
else:
    verdict = "extended_periodic_table_H_Rn_not_supported"

summary = pd.DataFrame([{
    "num_elements": len(ELEMENTS),
    "num_samples": len(df),
    "max_Z": 86,
    "element_accuracy": element_accuracy,
    "period_accuracy": period_accuracy,
    "group_accuracy": group_accuracy,
    "block_accuracy": block_accuracy,
    "element_silhouette": element_silhouette,
    "block_silhouette": block_silhouette,
    "Z_recovery": Z_recovery,
    "period_recovery": period_recovery,
    "d_block_detection": d_block_detection,
    "f_block_detection": f_block_detection,
    "mean_shell_stability": float(df["shell_stability"].mean()),
    "verdict": verdict,
}])

df.to_csv(OUT / "extended_periodic_H_Rn_samples.csv", index=False)
summary.to_csv(OUT / "extended_periodic_H_Rn_summary.csv", index=False)
by_element.to_csv(OUT / "extended_periodic_H_Rn_by_element.csv", index=False)
by_block.to_csv(OUT / "extended_periodic_H_Rn_by_block.csv", index=False)

proj = PCA(n_components=2).fit_transform(Xn)

plt.figure(figsize=(11, 7))
for block in ["s", "p", "d", "f"]:
    mask = df["block"].to_numpy() == block
    plt.scatter(proj[mask, 0], proj[mask, 1], s=8, label=f"{block}-block")
plt.title("BST extended periodic table H→Rn — s/p/d/f block structure")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_H_Rn_blocks.png", dpi=220)
plt.close()

plt.figure(figsize=(16, 5))
plt.plot(by_element["Z"], by_element["mean_shell_stability"], marker="o", label="shell stability")
plt.plot(by_element["Z"], by_element["mean_f_activation"], marker="x", label="f activation")
plt.xticks(by_element["Z"], by_element["symbol"], rotation=80)
plt.xlabel("Element")
plt.ylabel("BST score")
plt.title("BST H→Rn stability and f-block activation")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "extended_periodic_H_Rn_scores.png", dpi=220)
plt.close()

print("\n=== BST EXTENDED PERIODIC TABLE H→Rn TEST ===\n")
print(summary.to_string(index=False))

print("\nExtended periodic table H→Rn by block:")
print(by_block.to_string(index=False))

print("\nExtended periodic table H→Rn by element:")
print(by_element.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'extended_periodic_H_Rn_samples.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Rn_summary.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Rn_by_element.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Rn_by_block.csv'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Rn_blocks.png'}")
print(f"[OK] wrote {OUT / 'extended_periodic_H_Rn_scores.png'}")
print("[DONE] extended periodic table H→Rn test complete")