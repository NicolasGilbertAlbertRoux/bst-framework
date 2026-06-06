#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 64 — Electromagnetic-Like Interaction

Purpose:
    Test whether BST stress-energy / field / metric proxies generate an
    electromagnetic-like interaction layer.

Important:
    This is NOT a claim of Standard Model electromagnetism.
    It tests a BST effective interaction proxy:
        polarity + field gradient + gauge-like invariant + stress-energy flow
        -> attraction / repulsion / neutral screening.

Canonical hierarchy:
    W -> C -> field -> gauge-like invariants -> metric proxy
      -> stress-energy proxy -> electromagnetic-like interaction -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no particle identity;
    - no electron identity;
    - no photon identity;
    - no Standard Model EM claim.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


SEED = 1729
rng = np.random.default_rng(SEED)

INP = Path("results/research_final/wave_stress_energy_tensor_proxy_test/wave_stress_energy_samples.csv")
OUT = Path("results/research_final/electromagnetic_like_interaction_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "stress_regime",
    "metric_regime",
    "field_regime",
    "cluster_class",
    "is_valid_field",
    "field_density",
    "field_coherence",
    "field_range",
    "field_gradient",
    "field_flux",
    "field_order",
    "field_presence_score",
    "gauge_amplitude",
    "gauge_charge_proxy",
    "gauge_invariant_norm",
    "metric_potential",
    "metric_anisotropy",
    "curvature_density",
    "curvature_coherence",
    "effective_curvature_score",
    "energy_density_proxy",
    "pressure_proxy",
    "flux_proxy",
    "anisotropy_proxy",
    "tensor_norm_proxy",
    "stress_energy_coupling",
    "conservation_proxy",
]


def load_input():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing required input: {INP}\n"
            "Run experiments/63_wave_stress_energy_tensor_proxy/wave_stress_energy_tensor_proxy_test.py first."
        )

    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")

    return df


def classify_interaction_regime(row):
    if row["stress_regime"] == "vacuum_like":
        return "screened_neutral"

    if row["stress_regime"] == "localized_stress_energy":
        return "localized_polar_interaction"

    if row["stress_regime"] == "curvature_stress_energy_basin":
        return "extended_polar_field"

    return "unclassified_em_like"


def compute_em_like_features(df):
    df = df.copy()

    fd = df["field_density"].astype(float)
    fc = df["field_coherence"].astype(float)
    fr = df["field_range"].astype(float)
    fg = df["field_gradient"].astype(float)
    ff = df["field_flux"].astype(float)
    fo = df["field_order"].astype(float)
    fp = df["field_presence_score"].astype(float)

    ga = df["gauge_amplitude"].astype(float)
    gc = df["gauge_charge_proxy"].astype(float)
    gn = df["gauge_invariant_norm"].astype(float)

    mp = df["metric_potential"].astype(float)
    ma = df["metric_anisotropy"].astype(float)
    cd = df["curvature_density"].astype(float)
    cc = df["curvature_coherence"].astype(float)
    ecs = df["effective_curvature_score"].astype(float)

    ed = df["energy_density_proxy"].astype(float)
    pr = df["pressure_proxy"].astype(float)
    fl = df["flux_proxy"].astype(float)
    an = df["anisotropy_proxy"].astype(float)
    tn = df["tensor_norm_proxy"].astype(float)
    se = df["stress_energy_coupling"].astype(float)
    cons = df["conservation_proxy"].astype(float)

    df["em_like_regime"] = df.apply(classify_interaction_regime, axis=1)

    # Signed polarity proxy:
    # positive = outward/repulsive tendency
    # negative = inward/attractive tendency
    # neutral controls should remain near zero.
    df["polarity_proxy"] = np.clip(
        np.tanh(
            1.8 * (fl - pr)
            + 0.9 * (ff - fg)
            + 0.5 * (fr - fd)
        ),
        -1,
        1,
    )

    df["charge_separation_proxy"] = np.clip(
        np.abs(df["polarity_proxy"])
        * (0.35 + 0.30 * gn + 0.20 * fp + 0.15 * fc),
        0,
        3,
    )

    df["em_field_strength_proxy"] = np.clip(
        0.24 * df["charge_separation_proxy"]
        + 0.22 * ga
        + 0.18 * gn
        + 0.14 * fd
        + 0.12 * fg
        + 0.10 * se,
        0,
        5,
    )

    df["interaction_range_proxy"] = np.clip(
        fr * (0.45 + fc + 0.35 * np.tanh(se)),
        0,
        5,
    )

    df["screening_proxy"] = np.clip(
        1.0
        - (
            0.28 * df["em_field_strength_proxy"]
            + 0.24 * fp
            + 0.20 * gn
            + 0.16 * se
            + 0.12 * ed
        ),
        0,
        1,
    )

    df["attraction_proxy"] = np.clip(
        np.maximum(-df["polarity_proxy"], 0)
        * df["em_field_strength_proxy"]
        * (0.5 + cons),
        0,
        5,
    )

    df["repulsion_proxy"] = np.clip(
        np.maximum(df["polarity_proxy"], 0)
        * df["em_field_strength_proxy"]
        * (0.5 + cons),
        0,
        5,
    )

    df["neutrality_proxy"] = np.clip(
        1.0
        - df["charge_separation_proxy"]
        - 0.5 * df["em_field_strength_proxy"],
        0,
        1,
    )

    df["em_interaction_score"] = np.clip(
        0.30 * df["em_field_strength_proxy"]
        + 0.22 * df["charge_separation_proxy"]
        + 0.18 * df["interaction_range_proxy"]
        + 0.14 * np.maximum(df["attraction_proxy"], df["repulsion_proxy"])
        + 0.10 * (1.0 - df["screening_proxy"])
        + 0.06 * cons,
        0,
        5,
    )

    df["em_coupling_to_metric"] = np.clip(
        0.35 * df["em_interaction_score"]
        + 0.25 * mp
        + 0.20 * ecs
        + 0.10 * cc
        + 0.10 * ma,
        0,
        5,
    )

    df["em_like_supported"] = (
        (df["is_valid_field"] == 1)
        & (
            ((df["em_like_regime"] == "localized_polar_interaction") & (df["em_interaction_score"] >= 0.35))
            | ((df["em_like_regime"] == "extended_polar_field") & (df["em_interaction_score"] >= 0.80))
        )
    ).astype(int)

    df["localized_em_supported"] = (
        (df["em_like_regime"] == "localized_polar_interaction")
        & (df["em_interaction_score"] >= 0.35)
    ).astype(int)

    df["extended_em_supported"] = (
        (df["em_like_regime"] == "extended_polar_field")
        & (df["em_interaction_score"] >= 0.80)
    ).astype(int)

    return df


def supervised_accuracy(df, target):
    features = [
        "field_density",
        "field_coherence",
        "field_range",
        "field_gradient",
        "field_flux",
        "field_order",
        "field_presence_score",
        "gauge_amplitude",
        "gauge_charge_proxy",
        "gauge_invariant_norm",
        "metric_potential",
        "metric_anisotropy",
        "effective_curvature_score",
        "energy_density_proxy",
        "pressure_proxy",
        "flux_proxy",
        "anisotropy_proxy",
        "stress_energy_coupling",
        "conservation_proxy",
        "polarity_proxy",
        "charge_separation_proxy",
        "em_field_strength_proxy",
        "interaction_range_proxy",
        "screening_proxy",
        "attraction_proxy",
        "repulsion_proxy",
        "neutrality_proxy",
        "em_interaction_score",
        "em_coupling_to_metric",
    ]

    if len(df[target].unique()) < 2:
        return np.nan

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=280,
        max_depth=8,
        random_state=SEED,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    return float(accuracy_score(y_test, pred))


def safe_corr(a, b):
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return np.nan


def safe_silhouette(df, label_col, cols):
    try:
        if len(df[label_col].unique()) < 2:
            return np.nan
        return float(silhouette_score(df[cols], df[label_col]))
    except Exception:
        return np.nan


def summarize(df):
    invalid = df[df["em_like_regime"] == "screened_neutral"]
    local = df[df["em_like_regime"] == "localized_polar_interaction"]
    extended = df[df["em_like_regime"] == "extended_polar_field"]

    em_accuracy = supervised_accuracy(df, "em_like_supported")

    local_subset = df[df["em_like_regime"].isin(["localized_polar_interaction", "screened_neutral"])]
    extended_subset = df[df["em_like_regime"].isin(["extended_polar_field", "screened_neutral"])]

    local_em_accuracy = supervised_accuracy(local_subset, "localized_em_supported")
    extended_em_accuracy = supervised_accuracy(extended_subset, "extended_em_supported")

    local_support = float((local["em_interaction_score"] >= 0.35).mean()) if len(local) else 0.0
    extended_support = float((extended["em_interaction_score"] >= 0.80).mean()) if len(extended) else 0.0
    invalid_suppression = float((invalid["em_interaction_score"] < 0.35).mean()) if len(invalid) else 0.0

    mean_local = float(local["em_interaction_score"].mean()) if len(local) else 0.0
    mean_extended = float(extended["em_interaction_score"].mean()) if len(extended) else 0.0
    mean_invalid = float(invalid["em_interaction_score"].mean()) if len(invalid) else 0.0

    local_contrast = mean_local / max(mean_invalid, 1e-9)
    extended_contrast = mean_extended / max(mean_invalid, 1e-9)

    em_silhouette = safe_silhouette(
        df,
        "em_like_regime",
        [
            "polarity_proxy",
            "charge_separation_proxy",
            "em_field_strength_proxy",
            "interaction_range_proxy",
            "screening_proxy",
            "em_interaction_score",
            "em_coupling_to_metric",
        ],
    )

    charge_field_corr = safe_corr(df["charge_separation_proxy"], df["em_field_strength_proxy"])
    gauge_em_corr = safe_corr(df["gauge_invariant_norm"], df["em_interaction_score"])
    stress_em_corr = safe_corr(df["stress_energy_coupling"], df["em_interaction_score"])
    metric_em_corr = safe_corr(df["metric_potential"], df["em_coupling_to_metric"])
    polarity_flux_corr = safe_corr(df["polarity_proxy"], df["flux_proxy"] - df["pressure_proxy"])

    verdict = (
        "electromagnetic_like_interaction_supported"
        if (
            em_accuracy >= 0.98
            and local_support >= 0.95
            and extended_support >= 0.90
            and invalid_suppression >= 0.95
            and local_contrast > 1.5
            and extended_contrast > 3.0
            and charge_field_corr > 0.80
            and gauge_em_corr > 0.70
            and stress_em_corr > 0.70
            and metric_em_corr > 0.80
        )
        else "electromagnetic_like_interaction_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_screened_neutral": len(invalid),
        "num_localized_polar_interaction": len(local),
        "num_extended_polar_field": len(extended),
        "em_accuracy": em_accuracy,
        "local_em_accuracy": local_em_accuracy,
        "extended_em_accuracy": extended_em_accuracy,
        "em_silhouette": em_silhouette,
        "local_em_support": local_support,
        "extended_em_support": extended_support,
        "invalid_suppression": invalid_suppression,
        "mean_local_em_score": mean_local,
        "mean_extended_em_score": mean_extended,
        "mean_invalid_em_score": mean_invalid,
        "local_em_contrast": local_contrast,
        "extended_em_contrast": extended_contrast,
        "charge_field_corr": charge_field_corr,
        "gauge_em_corr": gauge_em_corr,
        "stress_em_corr": stress_em_corr,
        "metric_em_corr": metric_em_corr,
        "polarity_flux_corr": polarity_flux_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "electron_identity_used": False,
        "photon_identity_used": False,
        "standard_model_em_claim": False,
        "safe_hierarchy": "W -> C -> field -> gauge-like invariants -> metric proxy -> stress-energy proxy -> electromagnetic-like interaction -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(df):
    by_regime = df.groupby("em_like_regime").agg(
        count=("em_like_regime", "count"),
        mean_polarity=("polarity_proxy", "mean"),
        mean_abs_polarity=("polarity_proxy", lambda x: float(np.mean(np.abs(x)))),
        mean_charge_separation=("charge_separation_proxy", "mean"),
        mean_em_field_strength=("em_field_strength_proxy", "mean"),
        mean_interaction_range=("interaction_range_proxy", "mean"),
        mean_screening=("screening_proxy", "mean"),
        mean_attraction=("attraction_proxy", "mean"),
        mean_repulsion=("repulsion_proxy", "mean"),
        mean_neutrality=("neutrality_proxy", "mean"),
        mean_em_interaction=("em_interaction_score", "mean"),
        mean_em_coupling_to_metric=("em_coupling_to_metric", "mean"),
        support_ratio=("em_like_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        em_like_regime=("em_like_regime", lambda x: x.mode().iloc[0]),
        mean_charge_separation=("charge_separation_proxy", "mean"),
        mean_em_field_strength=("em_field_strength_proxy", "mean"),
        mean_interaction_range=("interaction_range_proxy", "mean"),
        mean_screening=("screening_proxy", "mean"),
        mean_em_interaction=("em_interaction_score", "mean"),
        mean_em_coupling_to_metric=("em_coupling_to_metric", "mean"),
    ).reset_index()

    by_stress = df.groupby("stress_regime").agg(
        count=("stress_regime", "count"),
        mean_em_field_strength=("em_field_strength_proxy", "mean"),
        mean_em_interaction=("em_interaction_score", "mean"),
        mean_em_coupling_to_metric=("em_coupling_to_metric", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_stress


def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("em_like_regime"):
        plt.scatter(
            sub["charge_separation_proxy"],
            sub["em_field_strength_proxy"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("charge_separation_proxy")
    plt.ylabel("em_field_strength_proxy")
    plt.title("BST electromagnetic-like field emergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "em_like_field_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("em_like_regime"):
        plt.scatter(
            sub["polarity_proxy"],
            sub["em_interaction_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("polarity_proxy")
    plt.ylabel("em_interaction_score")
    plt.title("BST polarity to EM-like interaction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "em_like_polarity_interaction.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(
        valid["cluster_class"].astype(str),
        valid["mean_em_interaction"],
    )
    plt.xlabel("cluster_class")
    plt.ylabel("mean_em_interaction")
    plt.title("BST EM-like interaction by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "em_like_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="em_like_regime",
        values="em_interaction_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="em_interaction_score")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("em_like_regime")
    plt.ylabel("cluster_class")
    plt.title("BST EM-like interaction heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "em_like_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 64 ELECTROMAGNETIC-LIKE INTERACTION TEST ===\n")

    base = load_input()
    samples = compute_em_like_features(base)

    summary = summarize(samples)
    by_regime, by_cluster, by_stress = aggregate(samples)

    samples.to_csv(OUT / "electromagnetic_like_samples.csv", index=False)
    summary.to_csv(OUT / "electromagnetic_like_summary.csv", index=False)
    by_regime.to_csv(OUT / "electromagnetic_like_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "electromagnetic_like_by_cluster.csv", index=False)
    by_stress.to_csv(OUT / "electromagnetic_like_by_stress_regime.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nElectromagnetic-like interaction by regime:")
    print(by_regime.to_string(index=False))

    print("\nElectromagnetic-like interaction by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nElectromagnetic-like interaction by stress regime:")
    print(by_stress.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'electromagnetic_like_samples.csv'}")
    print(f"[OK] wrote {OUT / 'electromagnetic_like_summary.csv'}")
    print(f"[OK] wrote {OUT / 'electromagnetic_like_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'electromagnetic_like_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'electromagnetic_like_by_stress_regime.csv'}")
    print(f"[OK] wrote {OUT / 'em_like_field_projection.png'}")
    print(f"[OK] wrote {OUT / 'em_like_polarity_interaction.png'}")
    print(f"[OK] wrote {OUT / 'em_like_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'em_like_heatmap.png'}")
    print("[DONE] electromagnetic-like interaction test complete")


if __name__ == "__main__":
    main()