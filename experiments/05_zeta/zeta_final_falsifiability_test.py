from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(
    "results/research_final/zeta_global_transverse_closure_test"
)

OUT = Path(
    "results/research_final/zeta_final_falsifiability_test"
)

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    BASE / "zeta_global_transverse_closure_sectors.csv"
)

# =========================================================
# FALSIFIABILITY CONDITIONS
# =========================================================

# critères interdits :
#
# 1. fermeture transverse trop faible
# 2. horizon négatif/impossible
# 3. expansion instable
# 4. respiration cosmologique incohérente

df["forbidden_closure"] = (
    df["sector_transverse_closure"] < 0.25
)

df["forbidden_horizon"] = (
    df["sector_horizon"] <= 0
)

df["forbidden_expansion"] = (
    df["expansion_factor"] <= 0
)

df["forbidden_breath"] = (
    df["cosmic_breath"] <= 0
)

# score global de cohérence

df["falsifiability_score"] = (
    (
        ~df["forbidden_closure"]
    ).astype(int)
    +
    (
        ~df["forbidden_horizon"]
    ).astype(int)
    +
    (
        ~df["forbidden_expansion"]
    ).astype(int)
    +
    (
        ~df["forbidden_breath"]
    ).astype(int)
)

summary = pd.DataFrame([{
    "total_sectors": len(df),
    "valid_sectors": int(
        (df["falsifiability_score"] == 4).sum()
    ),
    "invalid_sectors": int(
        (df["falsifiability_score"] < 4).sum()
    ),
    "mean_transverse_closure":
        df["sector_transverse_closure"].mean(),
    "min_transverse_closure":
        df["sector_transverse_closure"].min(),
}])

print("\n=== ZETA VI — FINAL FALSIFIABILITY TEST ===")

print("\nFalsifiability summary:")
print(summary.to_string(index=False))

print("\nSector validity:")
print(
    df[[
        "multisector_cluster",
        "sector_transverse_closure",
        "sector_horizon",
        "expansion_factor",
        "cosmic_breath",
        "forbidden_closure",
        "forbidden_horizon",
        "forbidden_expansion",
        "forbidden_breath",
        "falsifiability_score",
    ]]
    .sort_values(
        "falsifiability_score",
        ascending=False
    )
    .to_string(index=False)
)

summary.to_csv(
    OUT / "zeta_final_falsifiability_summary.csv",
    index=False
)

df.to_csv(
    OUT / "zeta_final_falsifiability_sectors.csv",
    index=False
)

print(f"\n[OK] wrote {OUT}")
print("[DONE] zeta final falsifiability test complete")