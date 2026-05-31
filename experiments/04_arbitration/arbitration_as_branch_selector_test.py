from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE_FUNC = Path("results/research_final/Rlambda_functional_role_test")
BASE_BRANCH = Path("results/research_final/branch_resolution_vs_coexistence_test")
OUT = Path("results/research_final/arbitration_as_branch_selector_test")
OUT.mkdir(parents=True, exist_ok=True)

func = pd.read_csv(BASE_FUNC / "Rlambda_functional_role_timeseries.csv")
events = pd.read_csv(BASE_BRANCH / "branch_resolution_events.csv")

S = np.column_stack([
    func["R_lambda"].values,
    func["phase_error"].values,
    func["breath_error"].values,
])

layers = [22, 36]
future_window = 32

def feats(i):
    chunks = []
    for w in layers:
        p = S[i-w:i]
        chunks += [p.mean(axis=0), p.std(axis=0)]
    chunks.append(S[i])
    return np.concatenate(chunks)

rows = []

for _, row in events.iterrows():
    i = int(row["index"])
    if i - max(layers) < 0 or i + future_window >= len(S):
        continue

    future = S[i:i+future_window]
    km = KMeans(n_clusters=2, random_state=42, n_init=20).fit(future)

    centers = km.cluster_centers_
    true_future = S[i+future_window]

    # sélection “arbitre idéal” : choisit la branche admissible la plus proche
    d0 = np.linalg.norm(true_future - centers[0])
    d1 = np.linalg.norm(true_future - centers[1])
    chosen = 0 if d0 < d1 else 1
    selected = centers[chosen]

    # prédiction substrat directe
    rows.append({
        "index": i,
        "branching_ratio": row["branching_ratio"],
        "silhouette": row["silhouette"],
        "cluster_distance": row["cluster_distance"],
        "true_R": true_future[0],
        "true_phase": true_future[1],
        "true_breath": true_future[2],
        "selected_R": selected[0],
        "selected_phase": selected[1],
        "selected_breath": selected[2],
    })

df = pd.DataFrame(rows)

truth = df[["true_R","true_phase","true_breath"]].values
sel = df[["selected_R","selected_phase","selected_breath"]].values

summary = pd.DataFrame([{
    "num_events": len(df),
    "selector_R2_Rlambda": r2_score(truth[:,0], sel[:,0]),
    "selector_R2_phase": r2_score(truth[:,1], sel[:,1]),
    "selector_R2_breath": r2_score(truth[:,2], sel[:,2]),
    "selector_R2_mean": np.mean([r2_score(truth[:,k], sel[:,k]) for k in range(3)]),
    "selector_MAE_mean": np.mean(np.abs(truth - sel)),
}])

print("\n=== ARBITRATION AS BRANCH SELECTOR TEST ===")
print(summary.to_string(index=False))

summary.to_csv(OUT / "arbitration_as_branch_selector_summary.csv", index=False)
df.to_csv(OUT / "arbitration_as_branch_selector_events.csv", index=False)

print(f"\n[OK] wrote {OUT}")
print("[DONE] arbitration as branch selector test complete")