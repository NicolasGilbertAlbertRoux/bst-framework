from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

BASE_FUNC = Path("results/research_final/Rlambda_functional_role_test")
BASE_BRANCH = Path("results/research_final/branch_resolution_vs_coexistence_test")

OUT = Path("results/research_final/minimal_substrate_closure_test")
OUT.mkdir(parents=True, exist_ok=True)

func = pd.read_csv(BASE_FUNC / "Rlambda_functional_role_timeseries.csv")
events = pd.read_csv(BASE_BRANCH / "branch_resolution_events.csv")

signals = np.column_stack([
    func["R_lambda"].values,
    func["phase_error"].values,
    func["breath_error"].values,
])

future_horizon = 32

# =========================
# paramètres du noyau minimal
# =========================

LAYER_ORG = 22
LAYER_PHASE = 36
FORGET_RATIO = 0.66

retain22 = int(round(LAYER_ORG * FORGET_RATIO))
retain36 = int(round(LAYER_PHASE * FORGET_RATIO))

# =========================
# construction features
# =========================

X = []
y = []

for _, row in events.iterrows():

    i = int(row["index"])

    if i-max(LAYER_ORG, LAYER_PHASE) < 0:
        continue

    if i+future_horizon >= len(signals):
        continue

    past22 = signals[i-retain22:i]
    past36 = signals[i-retain36:i]

    feat22 = np.concatenate([
        past22.mean(axis=0),
        past22.std(axis=0),
    ])

    feat36 = np.concatenate([
        past36.mean(axis=0),
        past36.std(axis=0),
    ])

    features = np.concatenate([
        feat22,
        feat36,
        signals[i],
    ])

    future = signals[i+future_horizon]

    X.append(features)
    y.append(future)

X = np.array(X)
y = np.array(y)

# =========================
# split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.35,
    random_state=42
)

# =========================
# modèle principal
# =========================

reg = Ridge(alpha=1e-4)

reg.fit(X_train, y_train)

pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)

# =========================
# résidu latent
# =========================

residual_train = y_train - pred_train
residual_test = y_test - pred_test

pca = PCA(n_components=1)

latent_train = pca.fit_transform(residual_train)
latent_test = pca.transform(residual_test)

latent_var = pca.explained_variance_ratio_[0]

# =========================
# fermeture avec résidu latent
# =========================

X_train_latent = np.column_stack([
    X_train,
    latent_train
])

X_test_latent = np.column_stack([
    X_test,
    latent_test
])

reg2 = Ridge(alpha=1e-4)

reg2.fit(X_train_latent, y_train)

pred_test_latent = reg2.predict(X_test_latent)

# =========================
# scores
# =========================

summary = pd.DataFrame([{

    "train_R2_mean":
        np.mean([
            r2_score(y_train[:,k], pred_train[:,k])
            for k in range(y.shape[1])
        ]),

    "test_R2_mean":
        np.mean([
            r2_score(y_test[:,k], pred_test[:,k])
            for k in range(y.shape[1])
        ]),

    "test_R2_mean_with_latent":
        np.mean([
            r2_score(y_test[:,k], pred_test_latent[:,k])
            for k in range(y.shape[1])
        ]),

    "latent_residual_variance":
        latent_var,

    "test_MAE_mean":
        np.mean([
            mean_absolute_error(y_test[:,k], pred_test[:,k])
            for k in range(y.shape[1])
        ]),

    "test_MAE_mean_with_latent":
        np.mean([
            mean_absolute_error(y_test[:,k], pred_test_latent[:,k])
            for k in range(y.shape[1])
        ]),

}])

print("\n=== MINIMAL SUBSTRATE CLOSURE TEST ===")
print(summary.to_string(index=False))

summary.to_csv(
    OUT / "minimal_substrate_closure_summary.csv",
    index=False
)

print(f"\n[OK] wrote {OUT}")
print("[DONE] minimal substrate closure test complete")