# ============================================================
# canonical_closure_law_test.py
#
# Teste si le système possède une fermeture canonique :
# les observables émergent-elles d’un petit manifold latent ?
#
# Idée :
# - générer beaucoup de systèmes équivalents ;
# - réduire par PCA ;
# - reconstruire les observables ;
# - mesurer si la dynamique se ferme sur peu de dimensions.
#
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ============================================================
# CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)

N = 4000

OUTPUT_DIR = "results/research_final/canonical_closure_law_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GENERATE SYNTHETIC SYSTEMS
# ============================================================

memory_gain = np.random.uniform(0.5, 3.0, N)
sigma = np.random.uniform(0.2, 2.5, N)

# gauge-like nuisance parameters
noise = np.random.uniform(0.0, 1.0, N)
grad_coeff = np.random.uniform(0.0, 1.0, N)
phase_shift = np.random.uniform(-np.pi, np.pi, N)

# latent canonical coordinate
latent = (
    1.7 * memory_gain
    + 0.9 * sigma
)

# observables generated from latent manifold
obs_a = (
    np.sin(latent)
    + 0.03 * noise
)

obs_b = (
    np.cos(0.5 * latent)
    + 0.04 * grad_coeff
)

obs_c = (
    latent
    + 0.02 * phase_shift
)

obs_d = (
    latent**2
    + 0.05 * noise
)

obs_e = (
    np.exp(-0.15 * latent)
    + 0.03 * grad_coeff
)

X = np.column_stack([
    obs_a,
    obs_b,
    obs_c,
    obs_d,
    obs_e
])

# ============================================================
# PCA REDUCTION
# ============================================================

pca = PCA(n_components=5)
Z = pca.fit_transform(X)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

df_pca = pd.DataFrame({
    "component": np.arange(1, 6),
    "explained_variance_ratio": explained,
    "cumulative": cumulative
})

print("\n=== CANONICAL CLOSURE PCA ===")
print(df_pca)

# ============================================================
# FIND MINIMAL CLOSURE DIMENSION
# ============================================================

thresholds = [0.90, 0.95, 0.99]

rows = []

for t in thresholds:
    dim = np.argmax(cumulative >= t) + 1

    rows.append({
        "variance_threshold": t,
        "minimal_dimension": dim
    })

df_dim = pd.DataFrame(rows)

print("\n=== MINIMAL CLOSURE DIMENSION ===")
print(df_dim)

# ============================================================
# RECOVER OBSERVABLES FROM LOW DIMENSION
# ============================================================

recovery_rows = []

for dim in [1, 2, 3]:

    Z_low = Z[:, :dim]

    for i, name in enumerate([
        "obs_a",
        "obs_b",
        "obs_c",
        "obs_d",
        "obs_e"
    ]):

        y = X[:, i]

        model = LinearRegression()
        model.fit(Z_low, y)

        y_pred = model.predict(Z_low)

        r2 = r2_score(y, y_pred)

        recovery_rows.append({
            "dimension": dim,
            "observable": name,
            "R2": r2
        })

df_recovery = pd.DataFrame(recovery_rows)

print("\n=== OBSERVABLE RECOVERY ===")
print(df_recovery)

# ============================================================
# METRIC TENSOR
# ============================================================

metric = np.cov(Z.T)

metric_df = pd.DataFrame(
    metric,
    columns=[f"z{i+1}" for i in range(metric.shape[0])],
    index=[f"z{i+1}" for i in range(metric.shape[0])]
)

print("\n=== EFFECTIVE METRIC TENSOR ===")
print(metric_df.round(4))

# ============================================================
# PLOTS
# ============================================================

# cumulative variance
plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(1, 6),
    cumulative,
    marker="o"
)

plt.axhline(0.95, linestyle="--")

plt.xlabel("latent dimension")
plt.ylabel("cumulative explained variance")
plt.title("Canonical closure collapse")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "canonical_closure_collapse.png"
    )
)

# latent manifold
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    Z[:, 0],
    Z[:, 1],
    c=latent,
    alpha=0.5
)

plt.colorbar(scatter)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Canonical latent manifold")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "canonical_latent_manifold.png"
    )
)

# recovery
pivot = df_recovery.pivot(
    index="observable",
    columns="dimension",
    values="R2"
)

pivot.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.ylabel("R² recovery")
plt.title("Observable reconstruction from low-dimensional closure")

plt.ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "closure_recovery_scores.png"
    )
)

# ============================================================
# SAVE
# ============================================================

df_pca.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "canonical_closure_pca.csv"
    ),
    index=False
)

df_dim.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "minimal_closure_dimension.csv"
    ),
    index=False
)

df_recovery.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "observable_recovery.csv"
    ),
    index=False
)

metric_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "effective_metric_tensor.csv"
    )
)

print(f"\n[OK] wrote {OUTPUT_DIR}")
print("[DONE] canonical closure law test complete")