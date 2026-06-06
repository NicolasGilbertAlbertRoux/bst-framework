#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 65 — Nuclear-Like Binding

Purpose:
    Test whether BST electromagnetic-like / stress-energy / metric proxies
    generate nuclear-like binding behavior.

Important:
    This is NOT a claim of Standard Model nuclear physics.
    It tests an effective BST binding proxy:
        cohesion + stress-energy confinement + curvature basin
        - EM-like polar overload - fragmentation
        -> bound / fragile / unbound regimes.

Canonical hierarchy:
    W -> C -> field -> gauge-like invariants -> metric proxy
      -> stress-energy proxy -> electromagnetic-like interaction
      -> nuclear-like binding -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no proton identity;
    - no neutron identity;
    - no quark identity;
    - no Standard Model nuclear claim.
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

INP = Path("results/research_final/electromagnetic_like_interaction_test/electromagnetic_like_samples.csv")
OUT = Path("results/research_final/nuclear_like_binding_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "em_like_regime",
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


def load_input():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing required input: {INP}\n"
            "Run experiments/64_electromagnetic_like_interaction/electromagnetic_like_interaction_test.py first."
        )

    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")

    return df


def classify_binding_regime(row):
    if row["em_like_regime"] == "screened_neutral":
        return "unbound_screened"

    if row["em_like_regime"] == "localized_polar_interaction":
        return "localized_bound_state"

    if row["em_like_regime"] == "extended_polar_field":
        return "extended_binding_basin"

    return "unclassified_binding"


def compute_binding_features(df):
    df = df.copy()

    fd = df["field_density"].astype(float)
    fc = df["field_coherence"].astype(float)
    fr = df["field_range"].astype(float)
    fg = df["field_gradient"].astype(float)
    ff = df["field_flux"].astype(float)
    fp = df["field_presence_score"].astype(float)

    gn = df["gauge_invariant_norm"].astype(float)
    gc = df["gauge_charge_proxy"].astype(float)

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

    polarity = df["polarity_proxy"].astype(float)
    charge_sep = df["charge_separation_proxy"].astype(float)
    em_strength = df["em_field_strength_proxy"].astype(float)
    em_range = df["interaction_range_proxy"].astype(float)
    screening = df["screening_proxy"].astype(float)
    attraction = df["attraction_proxy"].astype(float)
    repulsion = df["repulsion_proxy"].astype(float)
    em_score = df["em_interaction_score"].astype(float)

    df["binding_regime"] = df.apply(classify_binding_regime, axis=1)

    df["cohesion_proxy"] = np.clip(
        0.24 * fc
        + 0.20 * fp
        + 0.18 * cons
        + 0.16 * gn
        + 0.12 * cc
        + 0.10 * np.tanh(se),
        0,
        2,
    )

    df["confinement_proxy"] = np.clip(
        0.26 * mp
        + 0.24 * ecs
        + 0.18 * cd
        + 0.14 * tn
        + 0.10 * fr
        + 0.08 * (1.0 - screening),
        0,
        4,
    )

    df["binding_glue_proxy"] = np.clip(
        0.30 * attraction
        + 0.24 * df["cohesion_proxy"]
        + 0.20 * df["confinement_proxy"]
        + 0.16 * se
        + 0.10 * ed,
        0,
        5,
    )

    df["overload_proxy"] = np.clip(
        0.34 * repulsion
        + 0.26 * charge_sep
        + 0.18 * em_strength
        + 0.12 * np.abs(polarity)
        + 0.10 * an,
        0,
        5,
    )

    df["fragmentation_proxy"] = np.clip(
        0.30 * np.maximum(df["overload_proxy"] - df["binding_glue_proxy"], 0)
        + 0.22 * screening
        + 0.18 * np.abs(fl - pr)
        + 0.16 * ma
        + 0.14 * (1.0 - cons),
        0,
        5,
    )

    df["binding_energy_proxy"] = np.clip(
        df["binding_glue_proxy"]
        - 0.45 * df["overload_proxy"]
        - 0.35 * df["fragmentation_proxy"]
        + 0.20 * df["cohesion_proxy"],
        0,
        5,
    )

    df["binding_stability_proxy"] = np.clip(
        0.34 * df["binding_energy_proxy"]
        + 0.24 * df["cohesion_proxy"]
        + 0.18 * df["confinement_proxy"]
        + 0.14 * cons
        + 0.10 * (1.0 - df["fragmentation_proxy"] / 5.0),
        0,
        5,
    )

    df["island_stability_proxy"] = np.clip(
        df["binding_stability_proxy"]
        * (0.5 + 0.25 * np.tanh(fr) + 0.25 * np.tanh(se))
        * (1.0 - 0.25 * np.tanh(df["overload_proxy"])),
        0,
        5,
    )

    df["binding_supported"] = (
        (df["is_valid_field"] == 1)
        & (
            ((df["binding_regime"] == "localized_bound_state") & (df["binding_stability_proxy"] >= 0.45))
            | ((df["binding_regime"] == "extended_binding_basin") & (df["binding_stability_proxy"] >= 0.85))
        )
    ).astype(int)

    df["localized_binding_supported"] = (
        (df["binding_regime"] == "localized_bound_state")
        & (df["binding_stability_proxy"] >= 0.45)
    ).astype(int)

    df["extended_binding_supported"] = (
        (df["binding_regime"] == "extended_binding_basin")
        & (df["binding_stability_proxy"] >= 0.85)
    ).astype(int)

    df["fragmented_control"] = (
        (df["binding_regime"] == "unbound_screened")
        | (df["fragmentation_proxy"] > df["binding_glue_proxy"])
    ).astype(int)

    return df


def supervised_accuracy(df, target):
    features = [
        "field_density",
        "field_coherence",
        "field_range",
        "field_gradient",
        "field_flux",
        "field_presence_score",
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
        "polarity_proxy",
        "charge_separation_proxy",
        "em_field_strength_proxy",
        "interaction_range_proxy",
        "screening_proxy",
        "attraction_proxy",
        "repulsion_proxy",
        "em_interaction_score",
        "cohesion_proxy",
        "confinement_proxy",
        "binding_glue_proxy",
        "overload_proxy",
        "fragmentation_proxy",
        "binding_energy_proxy",
        "binding_stability_proxy",
        "island_stability_proxy",
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
        n_estimators=300,
        max_depth=9,
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
    invalid = df[df["binding_regime"] == "unbound_screened"]
    local = df[df["binding_regime"] == "localized_bound_state"]
    extended = df[df["binding_regime"] == "extended_binding_basin"]

    binding_accuracy = supervised_accuracy(df, "binding_supported")

    local_subset = df[df["binding_regime"].isin(["localized_bound_state", "unbound_screened"])]
    extended_subset = df[df["binding_regime"].isin(["extended_binding_basin", "unbound_screened"])]

    local_binding_accuracy = supervised_accuracy(local_subset, "localized_binding_supported")
    extended_binding_accuracy = supervised_accuracy(extended_subset, "extended_binding_supported")

    local_support = float((local["binding_stability_proxy"] >= 0.45).mean()) if len(local) else 0.0
    extended_support = float((extended["binding_stability_proxy"] >= 0.85).mean()) if len(extended) else 0.0
    invalid_suppression = float((invalid["binding_stability_proxy"] < 0.45).mean()) if len(invalid) else 0.0
    fragmentation_rejection = float((invalid["fragmented_control"] == 1).mean()) if len(invalid) else 0.0

    mean_local = float(local["binding_stability_proxy"].mean()) if len(local) else 0.0
    mean_extended = float(extended["binding_stability_proxy"].mean()) if len(extended) else 0.0
    mean_invalid = float(invalid["binding_stability_proxy"].mean()) if len(invalid) else 0.0

    local_contrast = mean_local / max(mean_invalid, 1e-9)
    extended_contrast = mean_extended / max(mean_invalid, 1e-9)

    binding_silhouette = safe_silhouette(
        df,
        "binding_regime",
        [
            "cohesion_proxy",
            "confinement_proxy",
            "binding_glue_proxy",
            "overload_proxy",
            "fragmentation_proxy",
            "binding_energy_proxy",
            "binding_stability_proxy",
            "island_stability_proxy",
        ],
    )

    cohesion_binding_corr = safe_corr(df["cohesion_proxy"], df["binding_stability_proxy"])
    confinement_binding_corr = safe_corr(df["confinement_proxy"], df["binding_stability_proxy"])
    em_overload_corr = safe_corr(df["em_interaction_score"], df["overload_proxy"])
    stress_binding_corr = safe_corr(df["stress_energy_coupling"], df["binding_stability_proxy"])
    curvature_binding_corr = safe_corr(df["effective_curvature_score"], df["binding_stability_proxy"])
    fragmentation_binding_corr = safe_corr(df["fragmentation_proxy"], df["binding_stability_proxy"])

    verdict = (
        "nuclear_like_binding_supported"
        if (
            binding_accuracy >= 0.98
            and local_support >= 0.95
            and extended_support >= 0.90
            and invalid_suppression >= 0.95
            and fragmentation_rejection >= 0.95
            and local_contrast > 1.5
            and extended_contrast > 2.5
            and cohesion_binding_corr > 0.75
            and confinement_binding_corr > 0.75
            and stress_binding_corr > 0.70
            and curvature_binding_corr > 0.70
        )
        else "nuclear_like_binding_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_unbound_screened": len(invalid),
        "num_localized_bound_state": len(local),
        "num_extended_binding_basin": len(extended),
        "binding_accuracy": binding_accuracy,
        "local_binding_accuracy": local_binding_accuracy,
        "extended_binding_accuracy": extended_binding_accuracy,
        "binding_silhouette": binding_silhouette,
        "local_binding_support": local_support,
        "extended_binding_support": extended_support,
        "invalid_suppression": invalid_suppression,
        "fragmentation_rejection": fragmentation_rejection,
        "mean_local_binding": mean_local,
        "mean_extended_binding": mean_extended,
        "mean_invalid_binding": mean_invalid,
        "local_binding_contrast": local_contrast,
        "extended_binding_contrast": extended_contrast,
        "cohesion_binding_corr": cohesion_binding_corr,
        "confinement_binding_corr": confinement_binding_corr,
        "em_overload_corr": em_overload_corr,
        "stress_binding_corr": stress_binding_corr,
        "curvature_binding_corr": curvature_binding_corr,
        "fragmentation_binding_corr": fragmentation_binding_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "proton_identity_used": False,
        "neutron_identity_used": False,
        "quark_identity_used": False,
        "standard_model_nuclear_claim": False,
        "safe_hierarchy": "W -> C -> field -> gauge-like invariants -> metric proxy -> stress-energy proxy -> electromagnetic-like interaction -> nuclear-like binding -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(df):
    by_regime = df.groupby("binding_regime").agg(
        count=("binding_regime", "count"),
        mean_cohesion=("cohesion_proxy", "mean"),
        mean_confinement=("confinement_proxy", "mean"),
        mean_binding_glue=("binding_glue_proxy", "mean"),
        mean_overload=("overload_proxy", "mean"),
        mean_fragmentation=("fragmentation_proxy", "mean"),
        mean_binding_energy=("binding_energy_proxy", "mean"),
        mean_binding_stability=("binding_stability_proxy", "mean"),
        mean_island_stability=("island_stability_proxy", "mean"),
        support_ratio=("binding_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        binding_regime=("binding_regime", lambda x: x.mode().iloc[0]),
        mean_cohesion=("cohesion_proxy", "mean"),
        mean_confinement=("confinement_proxy", "mean"),
        mean_binding_glue=("binding_glue_proxy", "mean"),
        mean_overload=("overload_proxy", "mean"),
        mean_fragmentation=("fragmentation_proxy", "mean"),
        mean_binding_energy=("binding_energy_proxy", "mean"),
        mean_binding_stability=("binding_stability_proxy", "mean"),
        mean_island_stability=("island_stability_proxy", "mean"),
    ).reset_index()

    by_em = df.groupby("em_like_regime").agg(
        count=("em_like_regime", "count"),
        mean_binding_stability=("binding_stability_proxy", "mean"),
        mean_binding_energy=("binding_energy_proxy", "mean"),
        mean_fragmentation=("fragmentation_proxy", "mean"),
        mean_island_stability=("island_stability_proxy", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_em


def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("binding_regime"):
        plt.scatter(
            sub["binding_glue_proxy"],
            sub["binding_stability_proxy"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("binding_glue_proxy")
    plt.ylabel("binding_stability_proxy")
    plt.title("BST nuclear-like binding stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "binding_stability_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("binding_regime"):
        plt.scatter(
            sub["overload_proxy"],
            sub["fragmentation_proxy"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("overload_proxy")
    plt.ylabel("fragmentation_proxy")
    plt.title("BST overload / fragmentation proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "binding_fragmentation_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(
        valid["cluster_class"].astype(str),
        valid["mean_binding_stability"],
    )
    plt.xlabel("cluster_class")
    plt.ylabel("mean_binding_stability")
    plt.title("BST binding stability by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "binding_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="binding_regime",
        values="binding_stability_proxy",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="binding_stability_proxy")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("binding_regime")
    plt.ylabel("cluster_class")
    plt.title("BST nuclear-like binding heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "binding_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 65 NUCLEAR-LIKE BINDING TEST ===\n")

    base = load_input()
    samples = compute_binding_features(base)

    summary = summarize(samples)
    by_regime, by_cluster, by_em = aggregate(samples)

    samples.to_csv(OUT / "nuclear_like_binding_samples.csv", index=False)
    summary.to_csv(OUT / "nuclear_like_binding_summary.csv", index=False)
    by_regime.to_csv(OUT / "nuclear_like_binding_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "nuclear_like_binding_by_cluster.csv", index=False)
    by_em.to_csv(OUT / "nuclear_like_binding_by_em_regime.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nNuclear-like binding by regime:")
    print(by_regime.to_string(index=False))

    print("\nNuclear-like binding by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nNuclear-like binding by EM-like regime:")
    print(by_em.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'nuclear_like_binding_samples.csv'}")
    print(f"[OK] wrote {OUT / 'nuclear_like_binding_summary.csv'}")
    print(f"[OK] wrote {OUT / 'nuclear_like_binding_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'nuclear_like_binding_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'nuclear_like_binding_by_em_regime.csv'}")
    print(f"[OK] wrote {OUT / 'binding_stability_projection.png'}")
    print(f"[OK] wrote {OUT / 'binding_fragmentation_projection.png'}")
    print(f"[OK] wrote {OUT / 'binding_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'binding_heatmap.png'}")
    print("[DONE] nuclear-like binding test complete")


if __name__ == "__main__":
    main()