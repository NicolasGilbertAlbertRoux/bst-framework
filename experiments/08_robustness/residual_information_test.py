from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

BASE_MODES = Path("results/research_final/hidden_breathing_modes_test")
BASE_BRANCH = Path("results/research_final/branch_resolution_vs_coexistence_test")

OUT = Path("results/research_final/residual_information_test")
OUT.mkdir(parents=True, exist_ok=True)

modes = pd.read_csv(BASE_MODES / "hidden_modes_timeseries.csv")
events = pd.read_csv(BASE_BRANCH / "branch_resolution_events.csv")

mode_cols = [c for c in modes.columns if c.startswith("mode")]
M = modes[mode_cols].values

future_window = 32
memory_window = 21

# =========================
# modèle organisationnel
# =========================

X = []
y = []

for _, row in events.iterrows():

    i = int(row["index"])

    if i-memory_window < 0 or i+future_window >= len(M):
        continue

    past = M[i-memory_window:i]

    features = np.concatenate([
        past.mean(axis=0),
        past.std(axis=0),
        M[i],
    ])

    future = M[i+future_window]

    X.append(features)
    y.append(future)

X = np.array(X)
y = np.array(y)

reg = Ridge(alpha=1e-6)
pred = reg.fit(X, y).predict(X)

# =========================
# résidus
# =========================

residuals = y - pred

# variance résiduelle
residual_var = np.var(residuals)

# =========================
# PCA sur les résidus
# =========================

pca = PCA()
Z = pca.fit_transform(residuals)

explained = pca.explained_variance_ratio_

# =========================
# clustering des résidus
# =========================

km = KMeans(n_clusters=2, random_state=42, n_init=20)
labels = km.fit_predict(residuals)

cluster_centers = km.cluster_centers_

intra = []
for k in range(2):
    pts = residuals[labels == k]
    intra.append(np.mean(np.linalg.norm(
        pts - cluster_centers[k],
        axis=1
    )))

inter = np.linalg.norm(
    cluster_centers[0] - cluster_centers[1]
)

structure_ratio = inter / (np.mean(intra) + 1e-9)

summary = pd.DataFrame([{
    "mean_residual_variance": residual_var,
    "pca_mode1_variance": explained[0],
    "pca_mode2_variance": explained[1],
    "cumulative_2modes": explained[:2].sum(),
    "residual_structure_ratio": structure_ratio,
    "is_structured_residual":
        structure_ratio > 1.5
}])

pca_report = pd.DataFrame({
    "mode": np.arange(1, len(explained)+1),
    "explained_variance": explained,
    "cumulative": np.cumsum(explained),
})

print("\n=== RESIDUAL INFORMATION TEST ===")
print(summary.to_string(index=False))

print("\nResidual PCA:")
print(pca_report.head(10).to_string(index=False))

summary.to_csv(
    OUT / "residual_information_summary.csv",
    index=False
)

pca_report.to_csv(
    OUT / "residual_information_pca.csv",
    index=False
)

print(f"\n[OK] wrote {OUT}")
print("[DONE] residual information test complete")