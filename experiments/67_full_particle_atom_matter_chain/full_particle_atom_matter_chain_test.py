#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 67 — Full Particle–Atom–Matter Chain

Purpose:
    Test the full BST chain from wave-derived regimes to matter architectures.

Important:
    This is NOT a Standard Model particle identity test.
    It does not claim electron/proton/neutron/quark identities.
    It tests whether the already validated BST layers form a coherent
    end-to-end matter-architecture pipeline.

Canonical hierarchy:
    W -> C -> field -> gauge-like invariants -> metric proxy
      -> stress-energy proxy -> electromagnetic-like interaction
      -> nuclear-like binding -> weak-like transition channels
      -> matter architecture -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no electron identity;
    - no proton identity;
    - no neutron identity;
    - no quark identity;
    - no atom identity;
    - no Standard Model claim.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


SEED = 1729

INP = Path("results/research_final/weak_like_transition_channels_test/weak_like_transition_samples.csv")
OUT = Path("results/research_final/full_particle_atom_matter_chain_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "transition_regime",
    "binding_regime",
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
    "field_presence_score",
    "gauge_invariant_norm",
    "metric_potential",
    "curvature_density",
    "curvature_coherence",
    "effective_curvature_score",
    "energy_density_proxy",
    "pressure_proxy",
    "flux_proxy",
    "stress_energy_coupling",
    "conservation_proxy",
    "polarity_proxy",
    "charge_separation_proxy",
    "em_field_strength_proxy",
    "em_interaction_score",
    "cohesion_proxy",
    "confinement_proxy",
    "binding_glue_proxy",
    "binding_energy_proxy",
    "binding_stability_proxy",
    "island_stability_proxy",
    "instability_drive",
    "channel_opening_proxy",
    "family_transition_proxy",
    "decay_proxy",
    "restabilization_proxy",
    "weak_like_transition_score",
    "transition_balance_proxy",
]


def load_input():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing required input: {INP}\n"
            "Run experiments/66_weak_like_transition_channels/weak_like_transition_channels_test.py first."
        )

    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")

    return df


def classify_architecture(row):
    if row["transition_regime"] == "suppressed_transition":
        return "vacuum_architecture"

    if row["transition_regime"] == "local_reconfiguration_channel":
        return "localized_matter_architecture"

    if row["transition_regime"] == "basin_decay_channel":
        return "extended_matter_architecture"

    return "unclassified_architecture"


def architecture_depth(label):
    return {
        "vacuum_architecture": 0,
        "localized_matter_architecture": 1,
        "extended_matter_architecture": 2,
    }.get(label, -1)


def compute_chain_features(df):
    df = df.copy()

    fp = df["field_presence_score"].astype(float)
    gn = df["gauge_invariant_norm"].astype(float)
    mp = df["metric_potential"].astype(float)
    ecs = df["effective_curvature_score"].astype(float)
    se = df["stress_energy_coupling"].astype(float)
    em = df["em_interaction_score"].astype(float)
    bs = df["binding_stability_proxy"].astype(float)
    wt = df["weak_like_transition_score"].astype(float)

    fc = df["field_coherence"].astype(float)
    cd = df["curvature_density"].astype(float)
    ed = df["energy_density_proxy"].astype(float)
    cons = df["conservation_proxy"].astype(float)
    island = df["island_stability_proxy"].astype(float)
    decay = df["decay_proxy"].astype(float)
    restab = df["restabilization_proxy"].astype(float)
    balance = df["transition_balance_proxy"].astype(float)

    df["matter_architecture"] = df.apply(classify_architecture, axis=1)
    df["architecture_depth"] = df["matter_architecture"].apply(architecture_depth)

    df["wave_to_field_score"] = np.clip(
        0.45 * fp
        + 0.25 * fc
        + 0.20 * df["field_density"].astype(float)
        + 0.10 * df["field_range"].astype(float),
        0,
        3,
    )

    df["field_to_gauge_score"] = np.clip(
        0.42 * gn
        + 0.24 * fp
        + 0.18 * fc
        + 0.16 * df["field_flux"].astype(float),
        0,
        3,
    )

    df["gauge_to_metric_score"] = np.clip(
        0.42 * mp
        + 0.28 * gn
        + 0.18 * ecs
        + 0.12 * df["curvature_coherence"].astype(float),
        0,
        3,
    )

    df["metric_to_stress_score"] = np.clip(
        0.36 * se
        + 0.26 * ecs
        + 0.20 * ed
        + 0.18 * df["pressure_proxy"].astype(float),
        0,
        4,
    )

    df["stress_to_em_score"] = np.clip(
        0.40 * em
        + 0.24 * se
        + 0.16 * df["charge_separation_proxy"].astype(float)
        + 0.12 * df["em_field_strength_proxy"].astype(float)
        + 0.08 * cons,
        0,
        4,
    )

    df["em_to_binding_score"] = np.clip(
        0.38 * bs
        + 0.24 * df["binding_energy_proxy"].astype(float)
        + 0.18 * df["cohesion_proxy"].astype(float)
        + 0.12 * df["confinement_proxy"].astype(float)
        + 0.08 * np.maximum(df["attraction_proxy"].astype(float), 0),
        0,
        4,
    ) if "attraction_proxy" in df.columns else np.clip(
        0.44 * bs
        + 0.24 * df["binding_energy_proxy"].astype(float)
        + 0.18 * df["cohesion_proxy"].astype(float)
        + 0.14 * df["confinement_proxy"].astype(float),
        0,
        4,
    )

    df["binding_to_transition_score"] = np.clip(
        0.34 * wt
        + 0.22 * df["channel_opening_proxy"].astype(float)
        + 0.18 * restab
        + 0.14 * balance
        + 0.12 * island,
        0,
        4,
    )

    df["matter_architecture_score"] = np.clip(
        0.16 * df["wave_to_field_score"]
        + 0.14 * df["field_to_gauge_score"]
        + 0.14 * df["gauge_to_metric_score"]
        + 0.14 * df["metric_to_stress_score"]
        + 0.14 * df["stress_to_em_score"]
        + 0.15 * df["em_to_binding_score"]
        + 0.13 * df["binding_to_transition_score"],
        0,
        4,
    )

    df["hierarchy_consistency_score"] = np.clip(
        1.0
        - (
            0.15 * np.abs(df["wave_to_field_score"] - fp)
            + 0.14 * np.abs(df["field_to_gauge_score"] - gn)
            + 0.14 * np.abs(df["gauge_to_metric_score"] - mp)
            + 0.14 * np.abs(df["metric_to_stress_score"] - se)
            + 0.14 * np.abs(df["stress_to_em_score"] - em)
            + 0.15 * np.abs(df["em_to_binding_score"] - bs)
            + 0.14 * np.abs(df["binding_to_transition_score"] - wt)
        ),
        0,
        1,
    )

    df["architecture_stability_score"] = np.clip(
        0.26 * df["matter_architecture_score"]
        + 0.20 * bs
        + 0.16 * island
        + 0.14 * restab
        + 0.12 * cons
        + 0.12 * df["hierarchy_consistency_score"],
        0,
        4,
    )

    df["architecture_complexity_score"] = np.clip(
        0.24 * fp
        + 0.18 * gn
        + 0.16 * ecs
        + 0.14 * se
        + 0.12 * em
        + 0.10 * bs
        + 0.06 * wt,
        0,
        4,
    )

    df["architecture_supported"] = (
        (df["is_valid_field"] == 1)
        & (
            ((df["matter_architecture"] == "localized_matter_architecture") & (df["matter_architecture_score"] >= 0.35))
            | ((df["matter_architecture"] == "extended_matter_architecture") & (df["matter_architecture_score"] >= 0.70))
        )
    ).astype(int)

    df["localized_architecture_supported"] = (
        (df["matter_architecture"] == "localized_matter_architecture")
        & (df["matter_architecture_score"] >= 0.35)
    ).astype(int)

    df["extended_architecture_supported"] = (
        (df["matter_architecture"] == "extended_matter_architecture")
        & (df["matter_architecture_score"] >= 0.70)
    ).astype(int)

    return df


def supervised_accuracy(df, target):
    features = [
        "field_presence_score",
        "gauge_invariant_norm",
        "metric_potential",
        "effective_curvature_score",
        "energy_density_proxy",
        "stress_energy_coupling",
        "em_interaction_score",
        "binding_stability_proxy",
        "weak_like_transition_score",
        "wave_to_field_score",
        "field_to_gauge_score",
        "gauge_to_metric_score",
        "metric_to_stress_score",
        "stress_to_em_score",
        "em_to_binding_score",
        "binding_to_transition_score",
        "matter_architecture_score",
        "hierarchy_consistency_score",
        "architecture_stability_score",
        "architecture_complexity_score",
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
        n_estimators=320,
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
    vacuum = df[df["matter_architecture"] == "vacuum_architecture"]
    local = df[df["matter_architecture"] == "localized_matter_architecture"]
    extended = df[df["matter_architecture"] == "extended_matter_architecture"]

    chain_accuracy = supervised_accuracy(df, "architecture_supported")

    local_subset = df[df["matter_architecture"].isin(["localized_matter_architecture", "vacuum_architecture"])]
    extended_subset = df[df["matter_architecture"].isin(["extended_matter_architecture", "vacuum_architecture"])]

    local_architecture_accuracy = supervised_accuracy(local_subset, "localized_architecture_supported")
    extended_architecture_accuracy = supervised_accuracy(extended_subset, "extended_architecture_supported")

    local_support = float((local["matter_architecture_score"] >= 0.35).mean()) if len(local) else 0.0
    extended_support = float((extended["matter_architecture_score"] >= 0.70).mean()) if len(extended) else 0.0
    vacuum_suppression = float((vacuum["matter_architecture_score"] < 0.35).mean()) if len(vacuum) else 0.0

    mean_local = float(local["matter_architecture_score"].mean()) if len(local) else 0.0
    mean_extended = float(extended["matter_architecture_score"].mean()) if len(extended) else 0.0
    mean_vacuum = float(vacuum["matter_architecture_score"].mean()) if len(vacuum) else 0.0

    local_contrast = mean_local / max(mean_vacuum, 1e-9)
    extended_contrast = mean_extended / max(mean_vacuum, 1e-9)

    chain_silhouette = safe_silhouette(
        df,
        "matter_architecture",
        [
            "wave_to_field_score",
            "field_to_gauge_score",
            "gauge_to_metric_score",
            "metric_to_stress_score",
            "stress_to_em_score",
            "em_to_binding_score",
            "binding_to_transition_score",
            "matter_architecture_score",
            "hierarchy_consistency_score",
        ],
    )

    field_to_matter_corr = safe_corr(df["field_presence_score"], df["matter_architecture_score"])
    gauge_to_matter_corr = safe_corr(df["gauge_invariant_norm"], df["matter_architecture_score"])
    metric_to_matter_corr = safe_corr(df["metric_potential"], df["matter_architecture_score"])
    stress_to_matter_corr = safe_corr(df["stress_energy_coupling"], df["matter_architecture_score"])
    em_to_matter_corr = safe_corr(df["em_interaction_score"], df["matter_architecture_score"])
    binding_to_matter_corr = safe_corr(df["binding_stability_proxy"], df["matter_architecture_score"])
    transition_to_matter_corr = safe_corr(df["weak_like_transition_score"], df["matter_architecture_score"])
    consistency_to_matter_corr = safe_corr(df["hierarchy_consistency_score"], df["matter_architecture_score"])

    mean_hierarchy_consistency = float(df["hierarchy_consistency_score"].mean())
    local_hierarchy_consistency = float(local["hierarchy_consistency_score"].mean()) if len(local) else 0.0
    extended_hierarchy_consistency = float(extended["hierarchy_consistency_score"].mean()) if len(extended) else 0.0

    verdict = (
        "full_particle_atom_matter_chain_supported"
        if (
            chain_accuracy >= 0.98
            and local_support >= 0.95
            and extended_support >= 0.90
            and vacuum_suppression >= 0.95
            and local_contrast > 1.5
            and extended_contrast > 2.5
            and field_to_matter_corr > 0.80
            and gauge_to_matter_corr > 0.80
            and metric_to_matter_corr > 0.80
            and stress_to_matter_corr > 0.80
            and binding_to_matter_corr > 0.80
            and transition_to_matter_corr > 0.70
            and mean_hierarchy_consistency > 0.65
        )
        else "full_particle_atom_matter_chain_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_vacuum_architecture": len(vacuum),
        "num_localized_matter_architecture": len(local),
        "num_extended_matter_architecture": len(extended),
        "chain_accuracy": chain_accuracy,
        "local_architecture_accuracy": local_architecture_accuracy,
        "extended_architecture_accuracy": extended_architecture_accuracy,
        "chain_silhouette": chain_silhouette,
        "local_architecture_support": local_support,
        "extended_architecture_support": extended_support,
        "vacuum_suppression": vacuum_suppression,
        "mean_local_architecture_score": mean_local,
        "mean_extended_architecture_score": mean_extended,
        "mean_vacuum_architecture_score": mean_vacuum,
        "local_architecture_contrast": local_contrast,
        "extended_architecture_contrast": extended_contrast,
        "field_to_matter_corr": field_to_matter_corr,
        "gauge_to_matter_corr": gauge_to_matter_corr,
        "metric_to_matter_corr": metric_to_matter_corr,
        "stress_to_matter_corr": stress_to_matter_corr,
        "em_to_matter_corr": em_to_matter_corr,
        "binding_to_matter_corr": binding_to_matter_corr,
        "transition_to_matter_corr": transition_to_matter_corr,
        "consistency_to_matter_corr": consistency_to_matter_corr,
        "mean_hierarchy_consistency": mean_hierarchy_consistency,
        "local_hierarchy_consistency": local_hierarchy_consistency,
        "extended_hierarchy_consistency": extended_hierarchy_consistency,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "electron_identity_used": False,
        "proton_identity_used": False,
        "neutron_identity_used": False,
        "quark_identity_used": False,
        "atom_identity_used": False,
        "standard_model_claim": False,
        "safe_hierarchy": "W -> C -> field -> gauge-like invariants -> metric proxy -> stress-energy proxy -> electromagnetic-like interaction -> nuclear-like binding -> weak-like transition channels -> matter architecture -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(df):
    by_architecture = df.groupby("matter_architecture").agg(
        count=("matter_architecture", "count"),
        mean_wave_to_field=("wave_to_field_score", "mean"),
        mean_field_to_gauge=("field_to_gauge_score", "mean"),
        mean_gauge_to_metric=("gauge_to_metric_score", "mean"),
        mean_metric_to_stress=("metric_to_stress_score", "mean"),
        mean_stress_to_em=("stress_to_em_score", "mean"),
        mean_em_to_binding=("em_to_binding_score", "mean"),
        mean_binding_to_transition=("binding_to_transition_score", "mean"),
        mean_matter_architecture=("matter_architecture_score", "mean"),
        mean_hierarchy_consistency=("hierarchy_consistency_score", "mean"),
        mean_architecture_stability=("architecture_stability_score", "mean"),
        mean_architecture_complexity=("architecture_complexity_score", "mean"),
        support_ratio=("architecture_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        matter_architecture=("matter_architecture", lambda x: x.mode().iloc[0]),
        mean_matter_architecture=("matter_architecture_score", "mean"),
        mean_hierarchy_consistency=("hierarchy_consistency_score", "mean"),
        mean_architecture_stability=("architecture_stability_score", "mean"),
        mean_architecture_complexity=("architecture_complexity_score", "mean"),
    ).reset_index()

    by_transition = df.groupby("transition_regime").agg(
        count=("transition_regime", "count"),
        matter_architecture=("matter_architecture", lambda x: x.mode().iloc[0]),
        mean_matter_architecture=("matter_architecture_score", "mean"),
        mean_hierarchy_consistency=("hierarchy_consistency_score", "mean"),
        mean_architecture_stability=("architecture_stability_score", "mean"),
    ).reset_index()

    layer_scores = pd.DataFrame([
        {"layer": "wave_to_field", "mean_score": float(df["wave_to_field_score"].mean())},
        {"layer": "field_to_gauge", "mean_score": float(df["field_to_gauge_score"].mean())},
        {"layer": "gauge_to_metric", "mean_score": float(df["gauge_to_metric_score"].mean())},
        {"layer": "metric_to_stress", "mean_score": float(df["metric_to_stress_score"].mean())},
        {"layer": "stress_to_em", "mean_score": float(df["stress_to_em_score"].mean())},
        {"layer": "em_to_binding", "mean_score": float(df["em_to_binding_score"].mean())},
        {"layer": "binding_to_transition", "mean_score": float(df["binding_to_transition_score"].mean())},
        {"layer": "matter_architecture", "mean_score": float(df["matter_architecture_score"].mean())},
    ])

    return by_architecture, by_cluster, by_transition, layer_scores


def make_plots(df, by_cluster, layer_scores):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("matter_architecture"):
        plt.scatter(
            sub["hierarchy_consistency_score"],
            sub["matter_architecture_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("hierarchy_consistency_score")
    plt.ylabel("matter_architecture_score")
    plt.title("BST full chain: hierarchy to matter architecture")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "full_chain_architecture_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(layer_scores["layer"], layer_scores["mean_score"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean layer score")
    plt.title("BST full chain layer scores")
    plt.tight_layout()
    plt.savefig(OUT / "full_chain_layer_scores.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(valid["cluster_class"].astype(str), valid["mean_matter_architecture"])
    plt.xlabel("cluster_class")
    plt.ylabel("mean_matter_architecture")
    plt.title("BST matter architecture by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "full_chain_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="matter_architecture",
        values="matter_architecture_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="matter_architecture_score")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("matter_architecture")
    plt.ylabel("cluster_class")
    plt.title("BST full particle-atom-matter chain heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "full_chain_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 67 FULL PARTICLE–ATOM–MATTER CHAIN TEST ===\n")

    base = load_input()
    samples = compute_chain_features(base)

    summary = summarize(samples)
    by_architecture, by_cluster, by_transition, layer_scores = aggregate(samples)

    samples.to_csv(OUT / "full_chain_samples.csv", index=False)
    summary.to_csv(OUT / "full_chain_summary.csv", index=False)
    by_architecture.to_csv(OUT / "full_chain_by_architecture.csv", index=False)
    by_cluster.to_csv(OUT / "full_chain_by_cluster.csv", index=False)
    by_transition.to_csv(OUT / "full_chain_by_transition_regime.csv", index=False)
    layer_scores.to_csv(OUT / "full_chain_layer_scores.csv", index=False)

    make_plots(samples, by_cluster, layer_scores)

    print(summary.to_string(index=False))

    print("\nFull chain by matter architecture:")
    print(by_architecture.to_string(index=False))

    print("\nFull chain by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nFull chain by transition regime:")
    print(by_transition.to_string(index=False))

    print("\nFull chain layer scores:")
    print(layer_scores.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'full_chain_samples.csv'}")
    print(f"[OK] wrote {OUT / 'full_chain_summary.csv'}")
    print(f"[OK] wrote {OUT / 'full_chain_by_architecture.csv'}")
    print(f"[OK] wrote {OUT / 'full_chain_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'full_chain_by_transition_regime.csv'}")
    print(f"[OK] wrote {OUT / 'full_chain_layer_scores.csv'}")
    print(f"[OK] wrote {OUT / 'full_chain_architecture_projection.png'}")
    print(f"[OK] wrote {OUT / 'full_chain_layer_scores.png'}")
    print(f"[OK] wrote {OUT / 'full_chain_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'full_chain_heatmap.png'}")
    print("[DONE] full particle-atom-matter chain test complete")


if __name__ == "__main__":
    main()