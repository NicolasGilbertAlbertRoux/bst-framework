from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(
    "results/research_final/zeta_final_falsifiability_test"
)

OUT = Path(
    "results/research_final/zeta_minimal_final_formulation"
)

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    BASE / "zeta_final_falsifiability_sectors.csv"
)

# =========================================================
# MINIMAL FINAL INVARIANTS
# =========================================================

# Noyau minimal :
#
# 1. fermeture transverse
# 2. respiration cosmologique
# 3. horizon causal
# 4. expansion locale
#
# Toute la structure Zeta est supposée émerger
# de leur interaction.

df["minimal_generator"] = (
    (
        df["sector_transverse_closure"]
        *
        df["cosmic_breath"]
    )
    /
    (
        df["sector_horizon"]
        +
        1e-12
    )
)

df["minimal_stability"] = (
    np.exp(
        -df["expansion_factor"] * 1e5
    )
)

df["minimal_action"] = (
    df["minimal_generator"]
    *
    df["minimal_stability"]
)

# =========================================================
# GLOBAL SYNTHESIS
# =========================================================

summary = pd.DataFrame([{
    "minimal_generator_mean":
        df["minimal_generator"].mean(),

    "minimal_generator_std":
        df["minimal_generator"].std(),

    "minimal_stability_mean":
        df["minimal_stability"].mean(),

    "minimal_action_mean":
        df["minimal_action"].mean(),

    "generator_cv":
        (
            df["minimal_generator"].std()
            /
            (
                df["minimal_generator"].mean()
                + 1e-12
            )
        ),
}])

# =========================================================
# INTERPRETATION
# =========================================================

laws = pd.DataFrame([
    {
        "minimal_law":
        "causal structure emerges from transverse closure"
    },
    {
        "minimal_law":
        "metric emerges from horizon-constrained breathing"
    },
    {
        "minimal_law":
        "stable observables require bounded expansion"
    },
    {
        "minimal_law":
        "quantization emerges from admissible causal clustering"
    },
    {
        "minimal_law":
        "cosmology emerges from sector breathing synchronization"
    },
])

print("\n=== ZETA VI — MINIMAL FINAL FORMULATION ===")

print("\nMinimal formulation summary:")
print(summary.to_string(index=False))

print("\nMinimal sector generators:")
print(
    df[[
        "multisector_cluster",
        "sector_transverse_closure",
        "sector_horizon",
        "cosmic_breath",
        "expansion_factor",
        "minimal_generator",
        "minimal_stability",
        "minimal_action",
    ]]
    .sort_values(
        "minimal_action",
        ascending=False
    )
    .to_string(index=False)
)

print("\nMinimal reconstructed laws:")
print(laws.to_string(index=False))

summary.to_csv(
    OUT / "zeta_minimal_final_summary.csv",
    index=False
)

df.to_csv(
    OUT / "zeta_minimal_final_sectors.csv",
    index=False
)

laws.to_csv(
    OUT / "zeta_minimal_final_laws.csv",
    index=False
)

print(f"\n[OK] wrote {OUT}")
print("[DONE] zeta minimal final formulation complete")