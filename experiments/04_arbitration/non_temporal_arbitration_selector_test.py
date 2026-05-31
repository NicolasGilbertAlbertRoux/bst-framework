from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE_FUNC = Path("results/research_final/Rlambda_functional_role_test")
BASE_BRANCH = Path("results/research_final/branch_resolution_vs_coexistence_test")

OUT = Path("results/research_final/non_temporal_arbitration_selector_test")
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

def substrate_features(i):
    chunks = []
    for w in layers:
        p = S[i-w:i]
        chunks.extend([p.mean(axis=0), p.std(axis=0)])
    chunks.append(S[i])
    return np.concatenate(chunks)

rows = []

for _, row in events.iterrows():
    i = int(row["index"])

    if i - max(layers) < 0 or i + future_window >= len(S):
        continue

    # Futurs admissibles locaux : deux branches extraites du futur proche
    future_segment = S[i:i+future_window]

    km = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = km.fit_predict(future_segment)
    centers = km.cluster_centers_

    true_future = S[i+future_window]

    # Choix non-temporel : on ne prédit pas la trajectoire pas-à-pas,
    # on sélectionne la branche admissible minimisant une énergie de cohérence.
    # Énergie = distance à R_lambda futur + pénalité respiratoire.
    energies = []

    for c in centers:
        E_identity = (true_future[0] - c[0])**2
        E_phase = 0.5 * (true_future[1] - c[1])**2
        E_breath = 0.25 * (true_future[2] - c[2])**2

        energies.append(E_identity + E_phase + E_breath)

    chosen = int(np.argmin(energies))
    selected = centers[chosen]

    rows.append({
        "index": i,
        "branching_ratio": row["branching_ratio"],
        "silhouette": row["silhouette"],
        "cluster_distance": row["cluster_distance"],
        "selected_R": selected[0],
        "selected_phase": selected[1],
        "selected_breath": selected[2],
        "true_R": true_future[0],
        "true_phase": true_future[1],
        "true_breath": true_future[2],
        "chosen_branch": chosen,
        "energy_gap": abs(energies[0] - energies[1]),
    })

df = pd.DataFrame(rows)

truth = df[["true_R", "true_phase", "true_breath"]].values
sel = df[["selected_R", "selected_phase", "selected_breath"]].values

summary = pd.DataFrame([{
    "num_events": len(df),
    "R2_Rlambda": r2_score(truth[:,0], sel[:,0]),
    "R2_phase": r2_score(truth[:,1], sel[:,1]),
    "R2_breath": r2_score(truth[:,2], sel[:,2]),
    "R2_mean": np.mean([r2_score(truth[:,k], sel[:,k]) for k in range(3)]),
    "MAE_mean": np.mean(np.abs(truth - sel)),
    "mean_energy_gap": df["energy_gap"].mean(),
}])

print("\n=== NON-TEMPORAL ARBITRATION SELECTOR TEST ===")
print(summary.to_string(index=False))

summary.to_csv(OUT / "non_temporal_arbitration_selector_summary.csv", index=False)
df.to_csv(OUT / "non_temporal_arbitration_selector_events.csv", index=False)

print(f"\n[OK] wrote {OUT}")
print("[DONE] non-temporal arbitration selector test complete")