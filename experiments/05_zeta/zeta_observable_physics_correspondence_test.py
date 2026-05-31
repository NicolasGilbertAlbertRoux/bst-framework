from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("results/research_final/zeta_physical_law_recovery_test")
OUT = Path("results/research_final/zeta_observable_physics_correspondence_test")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "zeta_physical_law_recovery_events.csv")

# Correspondances candidates avec signatures physiques minimales

df["relativistic_correspondence"] = (
    df["relativity_candidate"]
    / (1.0 + np.abs(df["pseudo_interval"]))
)

df["quantum_correspondence"] = (
    1.0
    / (1.0 + df["quantization_distance"])
)

df["conservation_correspondence"] = (
    1.0
    / (1.0 + np.abs(df["symmetry_invariant_2"] - 1.0))
)

df["observable_physics_score"] = (
    df["relativistic_correspondence"]
    * df["quantum_correspondence"]
    * df["conservation_correspondence"]
)

summary = (
    df.groupby("quantum_state")
    .agg({
        "relativistic_correspondence": ["mean", "std"],
        "quantum_correspondence": ["mean", "std"],
        "conservation_correspondence": ["mean", "std"],
        "observable_physics_score": ["mean", "std", "max"],
        "mode_transition": lambda x: x.value_counts().to_dict(),
    })
)

summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()

print("\n=== ZETA VI — OBSERVABLE PHYSICS CORRESPONDENCE TEST ===")

print("\nCorrespondence summary:")
print(summary.to_string(index=False))

print("\nBest observable-physics correspondences:")
print(
    df.sort_values("observable_physics_score", ascending=False)
    .head(30)[[
        "index_a",
        "index_b",
        "mode_transition",
        "quantum_state",
        "relativistic_correspondence",
        "quantum_correspondence",
        "conservation_correspondence",
        "observable_physics_score",
        "relativity_candidate",
        "proto_constant",
    ]]
    .to_string(index=False)
)

summary.to_csv(
    OUT / "zeta_observable_physics_correspondence_summary.csv",
    index=False
)

df.to_csv(
    OUT / "zeta_observable_physics_correspondence_events.csv",
    index=False
)

print(f"\n[OK] wrote {OUT}")
print("[DONE] zeta observable physics correspondence test complete")