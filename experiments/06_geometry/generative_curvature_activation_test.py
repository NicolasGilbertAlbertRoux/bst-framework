from pathlib import Path
import numpy as np
import pandas as pd

BASE_FUNC = Path("results/research_final/Rlambda_functional_role_test")
BASE_MODE = Path("results/research_final/mode_hierarchy_correspondence_test")

OUT = Path("results/research_final/generative_curvature_activation_test")
OUT.mkdir(parents=True, exist_ok=True)

func = pd.read_csv(BASE_FUNC / "Rlambda_functional_role_timeseries.csv")
modes = pd.read_csv(BASE_MODE / "mode_hierarchy_events.csv")

S = np.column_stack([
    func["R_lambda"].values,
    func["phase_error"].values,
    func["breath_error"].values,
])

def curvature(path):
    v = np.diff(path, axis=0)
    a = np.diff(v, axis=0)
    speed = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(a, axis=1)
    return np.mean(acc / (speed[:-1]**2 + 1e-9))

window = 16
rows = []

for _, row in modes.iterrows():
    i = int(row["index"])

    if i - window < 0 or i + window >= len(S):
        continue

    pre = S[i-window:i]
    post = S[i:i+window]

    pre_curv = curvature(pre)
    post_curv = curvature(post)

    rows.append({
        "index": i,
        "mode": row["predicted_mode"],
        "taxon": row["branch_taxon"],
        "pre_curvature": pre_curv,
        "post_curvature": post_curv,
        "curvature_gain": post_curv - pre_curv,
    })

df = pd.DataFrame(rows)

summary = (
    df.groupby("mode")
    .agg({
        "curvature_gain": ["mean", "std", "count"],
        "pre_curvature": "mean",
        "post_curvature": "mean",
    })
)

summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()

print("\n=== GENERATIVE CURVATURE ACTIVATION TEST ===")
print(summary.to_string(index=False))

print("\nTop curvature gains:")
print(
    df.sort_values("curvature_gain", ascending=False)
    .head(20)
    .to_string(index=False)
)

summary.to_csv(OUT / "generative_curvature_activation_summary.csv", index=False)
df.to_csv(OUT / "generative_curvature_activation_events.csv", index=False)

print(f"\n[OK] wrote {OUT}")
print("[DONE] generative curvature activation test complete")