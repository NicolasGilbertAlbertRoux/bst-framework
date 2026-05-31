from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("results/research_final/zeta_emergent_causal_cone_test")
OUT = Path("results/research_final/zeta_relativistic_limit_test")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "zeta_emergent_causal_cone_events.csv")

df["relativistic_stability"] = (
    1.0 / (1.0 + np.abs(df["pseudo_interval"]))
)

df["lightcone_like_score"] = (
    df["relativistic_stability"]
    * (1.0 + df["cone_opening"])
)

summary = (
    df.groupby(["cluster_transition", "causal_region"])
    .agg({
        "pseudo_interval": ["mean", "std"],
        "cone_opening": ["mean", "std"],
        "lightcone_like_score": ["mean", "max"],
        "local_causal_speed": ["mean", "std"],
    })
)

summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()

print("\n=== ZETA VI — RELATIVISTIC LIMIT TEST ===")
print("\nRelativistic-limit summary:")
print(summary.to_string(index=False))

print("\nMost lightcone-like transitions:")
print(
    df.sort_values("lightcone_like_score", ascending=False)
    .head(20)[[
        "index_a",
        "index_b",
        "mode_transition",
        "cluster_transition",
        "causal_region",
        "pseudo_interval",
        "cone_opening",
        "local_causal_speed",
        "lightcone_like_score",
    ]]
    .to_string(index=False)
)

summary.to_csv(OUT / "zeta_relativistic_limit_summary.csv", index=False)
df.to_csv(OUT / "zeta_relativistic_limit_events.csv", index=False)

print(f"\n[OK] wrote {OUT}")
print("[DONE] zeta relativistic limit test complete")