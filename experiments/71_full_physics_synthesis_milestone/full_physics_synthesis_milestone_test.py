#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 71 — Full Physics Synthesis Milestone

Synthesizes validated BST chain:
51-56 condensed matter
57-59 wave/contact safety
60-67 fields/interactions/matter
68-70 cosmology/structure/gravity

No SM identity claim. No GR equivalence claim. No standard cosmology claim.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/research_final/full_physics_synthesis_milestone_test")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "56_condensed_matter": Path("results/research_final/full_condensed_matter_milestone_test/condensed_matter_summary.csv"),
    "60_field": Path("results/research_final/field_emergence_test/field_emergence_summary.csv"),
    "61_gauge": Path("results/research_final/gauge_like_symmetry_emergence_test/gauge_like_symmetry_summary.csv"),
    "62_metric": Path("results/research_final/curvature_metric_coupling_proxy_test/curvature_metric_summary.csv"),
    "63_stress_energy": Path("results/research_final/wave_stress_energy_tensor_proxy_test/wave_stress_energy_summary.csv"),
    "64_em_like": Path("results/research_final/electromagnetic_like_interaction_test/electromagnetic_like_summary.csv"),
    "65_nuclear_like": Path("results/research_final/nuclear_like_binding_test/nuclear_like_binding_summary.csv"),
    "66_weak_like": Path("results/research_final/weak_like_transition_channels_test/weak_like_transition_summary.csv"),
    "67_full_matter_chain": Path("results/research_final/full_particle_atom_matter_chain_test/full_chain_summary.csv"),
    "68_vacuum_background": Path("results/research_final/vacuum_wave_background_test/vacuum_wave_background_summary.csv"),
    "69_structure": Path("results/research_final/structure_formation_from_wave_clusters_test/structure_formation_summary.csv"),
    "70_gravity": Path("results/research_final/gravity_curvature_emergence_milestone_test/gravity_curvature_summary.csv"),
}

EXPECTED_VERDICTS = {
    "56_condensed_matter": "full_condensed_matter_milestone_supported",
    "60_field": "field_emergence_supported",
    "61_gauge": "gauge_like_symmetry_emergence_supported",
    "62_metric": "curvature_metric_coupling_supported",
    "63_stress_energy": "wave_stress_energy_tensor_supported",
    "64_em_like": "electromagnetic_like_interaction_supported",
    "65_nuclear_like": "nuclear_like_binding_supported",
    "66_weak_like": "weak_like_transition_channels_supported",
    "67_full_matter_chain": "full_particle_atom_matter_chain_supported",
    "68_vacuum_background": "vacuum_wave_background_supported",
    "69_structure": "structure_formation_from_wave_clusters_supported",
    "70_gravity": "gravity_curvature_emergence_supported",
}

SCORE_COLUMNS = {
    "56_condensed_matter": ["condensed_matter_chain_score", "condensed_matter_harmonic_score", "verdict_integrity", "cross_stage_consistency"],
    "60_field": ["field_presence_accuracy", "regime_silhouette", "local_field_support", "extended_field_support", "invalid_suppression"],
    "61_gauge": ["invariance_accuracy", "gauge_invariance_support", "valid_gauge_support", "invalid_suppression", "amplitude_conservation"],
    "62_metric": ["metric_accuracy", "metric_silhouette", "local_metric_support", "extended_metric_support", "invalid_suppression"],
    "63_stress_energy": ["tensor_accuracy", "tensor_silhouette", "local_tensor_support", "extended_tensor_support", "invalid_suppression"],
    "64_em_like": ["em_accuracy", "em_silhouette", "local_em_support", "extended_em_support", "invalid_suppression"],
    "65_nuclear_like": ["binding_accuracy", "binding_silhouette", "local_binding_support", "extended_binding_support", "fragmentation_rejection"],
    "66_weak_like": ["transition_accuracy", "transition_silhouette", "suppressed_rejection", "instability_decay_corr", "channel_transition_corr"],
    "67_full_matter_chain": ["chain_accuracy", "chain_silhouette", "local_architecture_support", "extended_architecture_support", "vacuum_suppression"],
    "68_vacuum_background": ["background_accuracy", "background_silhouette", "vacuum_support", "structured_support", "matter_seed_support"],
    "69_structure": ["structure_accuracy", "structure_silhouette", "diffuse_support", "clustered_support", "large_scale_support"],
    "70_gravity": ["gravity_accuracy", "gravity_silhouette", "flat_background_support", "localized_gravity_support", "extended_curvature_support"],
}

def load_stage(name, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary for {name}: {path}")
    df = pd.read_csv(path)
    if "verdict" not in df.columns:
        raise ValueError(f"Missing verdict column in {path}")
    return df.iloc[0].to_dict()

def clamp01(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return np.nan

def stage_score(name, row):
    vals = []
    for c in SCORE_COLUMNS[name]:
        if c in row:
            vals.append(clamp01(row[c]))
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0

def main():
    print("\n=== BST 71 FULL PHYSICS SYNTHESIS MILESTONE TEST ===\n")

    rows = []
    for name, path in FILES.items():
        row = load_stage(name, path)
        verdict = str(row["verdict"])
        expected = EXPECTED_VERDICTS[name]
        supported = verdict == expected
        score = stage_score(name, row)

        rows.append({
            "stage": name,
            "expected_verdict": expected,
            "actual_verdict": verdict,
            "supported": float(supported),
            "stage_score": score,
            "source_file": str(path),
        })

    stages = pd.DataFrame(rows)

    chain_integrity = float(stages["supported"].mean())
    mean_stage_score = float(stages["stage_score"].mean())
    weakest_idx = int(stages["stage_score"].idxmin())
    weakest_stage = str(stages.loc[weakest_idx, "stage"])
    weakest_stage_score = float(stages.loc[weakest_idx, "stage_score"])

    phase_rows = [
        {
            "phase": "condensed_matter",
            "stages": "56",
            "phase_score": float(stages[stages["stage"].isin(["56_condensed_matter"])]["stage_score"].mean()),
        },
        {
            "phase": "fields_metric_interactions",
            "stages": "60-66",
            "phase_score": float(stages[stages["stage"].isin([
                "60_field", "61_gauge", "62_metric", "63_stress_energy",
                "64_em_like", "65_nuclear_like", "66_weak_like"
            ])]["stage_score"].mean()),
        },
        {
            "phase": "matter_chain",
            "stages": "67",
            "phase_score": float(stages[stages["stage"].isin(["67_full_matter_chain"])]["stage_score"].mean()),
        },
        {
            "phase": "cosmology_gravity",
            "stages": "68-70",
            "phase_score": float(stages[stages["stage"].isin([
                "68_vacuum_background", "69_structure", "70_gravity"
            ])]["stage_score"].mean()),
        },
    ]
    phases = pd.DataFrame(phase_rows)

    phase_harmonic_score = float(len(phases) / np.sum(1.0 / np.maximum(phases["phase_score"].values, 1e-9)))
    stage_harmonic_score = float(len(stages) / np.sum(1.0 / np.maximum(stages["stage_score"].values, 1e-9)))

    guardrails = {
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "electron_identity_used": False,
        "proton_identity_used": False,
        "neutron_identity_used": False,
        "quark_identity_used": False,
        "atom_identity_used": False,
        "standard_model_claim": False,
        "general_relativity_claim": False,
        "standard_cosmology_claim": False,
    }

    # Safety audit: if any stage has these columns and any is True, fail guardrail.
    guardrail_violations = []
    for name, path in FILES.items():
        row = load_stage(name, path)
        for g in guardrails:
            if g in row and bool(row[g]) is True:
                guardrail_violations.append(f"{name}:{g}")

    guardrail_integrity = 1.0 if not guardrail_violations else 0.0

    synthesis_score = float(
        0.30 * chain_integrity
        + 0.25 * mean_stage_score
        + 0.20 * stage_harmonic_score
        + 0.15 * phase_harmonic_score
        + 0.10 * guardrail_integrity
    )

    verdict = (
        "full_physics_synthesis_milestone_supported"
        if (
            chain_integrity == 1.0
            and guardrail_integrity == 1.0
            and mean_stage_score >= 0.80
            and stage_harmonic_score >= 0.75
            and phase_harmonic_score >= 0.75
            and weakest_stage_score >= 0.60
        )
        else "full_physics_synthesis_milestone_not_supported"
    )

    summary = pd.DataFrame([{
        "num_stages": len(stages),
        "chain_integrity": chain_integrity,
        "mean_stage_score": mean_stage_score,
        "stage_harmonic_score": stage_harmonic_score,
        "phase_harmonic_score": phase_harmonic_score,
        "weakest_stage": weakest_stage,
        "weakest_stage_score": weakest_stage_score,
        "guardrail_integrity": guardrail_integrity,
        "guardrail_violations": ",".join(guardrail_violations) if guardrail_violations else "none",
        "synthesis_score": synthesis_score,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "standard_model_claim": False,
        "general_relativity_claim": False,
        "standard_cosmology_claim": False,
        "safe_hierarchy": "W/C -> field -> gauge-like invariants -> metric -> stress-energy -> interactions -> matter -> vacuum background -> structures -> gravity-like curvature",
        "verdict": verdict,
    }])

    stages.to_csv(OUT / "full_physics_synthesis_stage_audit.csv", index=False)
    phases.to_csv(OUT / "full_physics_synthesis_phase_scores.csv", index=False)
    summary.to_csv(OUT / "full_physics_synthesis_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(stages["stage"], stages["stage_score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("stage_score")
    plt.title("BST full physics synthesis — stage scores")
    plt.tight_layout()
    plt.savefig(OUT / "full_physics_stage_scores.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(phases["phase"], phases["phase_score"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("phase_score")
    plt.title("BST full physics synthesis — phase scores")
    plt.tight_layout()
    plt.savefig(OUT / "full_physics_phase_scores.png", dpi=220)
    plt.close()

    print(summary.to_string(index=False))

    print("\nStage audit:")
    print(stages.to_string(index=False))

    print("\nPhase scores:")
    print(phases.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'full_physics_synthesis_summary.csv'}")
    print(f"[OK] wrote {OUT / 'full_physics_synthesis_stage_audit.csv'}")
    print(f"[OK] wrote {OUT / 'full_physics_synthesis_phase_scores.csv'}")
    print(f"[OK] wrote {OUT / 'full_physics_stage_scores.png'}")
    print(f"[OK] wrote {OUT / 'full_physics_phase_scores.png'}")
    print("[DONE] full physics synthesis milestone complete")

if __name__ == "__main__":
    main()