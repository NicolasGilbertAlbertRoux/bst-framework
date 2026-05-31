from pathlib import Path
import numpy as np
import pandas as pd

BASE_FUNC = Path(
    "results/research_final/Rlambda_functional_role_test"
)

BASE_MODE = Path(
    "results/research_final/mode_hierarchy_correspondence_test"
)

OUT = Path(
    "results/research_final/limited_closure_invariants_test"
)

OUT.mkdir(parents=True, exist_ok=True)

func = pd.read_csv(
    BASE_FUNC / "Rlambda_functional_role_timeseries.csv"
)

modes = pd.read_csv(
    BASE_MODE / "mode_hierarchy_events.csv"
)

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

    return np.mean(
        acc / (speed[:-1]**2 + 1e-9)
    )

rows = []

window = 24

for _, row in modes.iterrows():

    i = int(row["index"])

    if i - window < 0 or i + window >= len(S):
        continue

    pre = S[i-window:i]
    post = S[i:i+window]

    pre_energy = np.linalg.norm(
        pre.mean(axis=0)
    )

    post_energy = np.linalg.norm(
        post.mean(axis=0)
    )

    pre_curv = curvature(pre)
    post_curv = curvature(post)

    closure_fraction = (
        np.linalg.norm(
            post.mean(axis=0)
        )
        /
        (
            np.linalg.norm(
                pre.mean(axis=0)
            ) + 1e-9
        )
    )

    curvature_ratio = (
        post_curv /
        (pre_curv + 1e-9)
    )

    invariant_score = (
        abs(closure_fraction - 1.0)
        +
        abs(curvature_ratio - 1.0)
    )

    rows.append({
        "index": i,
        "mode": row["predicted_mode"],
        "taxon": row["branch_taxon"],

        "closure_fraction":
            closure_fraction,

        "curvature_ratio":
            curvature_ratio,

        "invariant_score":
            invariant_score,

        "pre_energy":
            pre_energy,

        "post_energy":
            post_energy,

        "pre_curvature":
            pre_curv,

        "post_curvature":
            post_curv,
    })

df = pd.DataFrame(rows)

summary = (
    df.groupby("mode")
    .agg({
        "closure_fraction": [
            "mean",
            "std"
        ],

        "curvature_ratio": [
            "mean",
            "std"
        ],

        "invariant_score": [
            "mean",
            "std"
        ],
    })
)

summary.columns = [
    "_".join(c)
    for c in summary.columns
]

summary = summary.reset_index()

print("\n=== LIMITED CLOSURE + INVARIANTS TEST ===")
print(summary.to_string(index=False))

print("\nTop invariant events:")

print(
    df.sort_values(
        "invariant_score"
    )
    .head(20)
    .to_string(index=False)
)

summary.to_csv(
    OUT / "limited_closure_invariants_summary.csv",
    index=False
)

df.to_csv(
    OUT / "limited_closure_invariants_events.csv",
    index=False
)

print(f"\n[OK] wrote {OUT}")
print("[DONE] limited closure invariants test complete")