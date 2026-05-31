from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE = Path("results/research_final/Rlambda_functional_role_test")
OUT = Path("results/research_final/Rlambda_generative_memory_test")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "Rlambda_functional_role_timeseries.csv")

t = df["t"].values
R = df["R_lambda"].values
phase = df["phase_error"].values
breath = df["breath_error"].values
dt = np.median(np.diff(t))

lag = 1
horizons = [1,2,4,8,12,16,24,32,48]

rows = []

for h in horizons:
    start = lag
    end = len(R) - h

    Rnow = R[start:end]
    Rlag = R[start-lag:end-lag]
    P = phase[start-lag:end-lag]
    B = breath[start-lag:end-lag]
    phi = np.arctan2(B, P)

    X_memory_only = np.column_stack([
        np.ones_like(Rnow),
        Rnow, Rlag,
        Rnow*Rlag,
        Rlag**2,
    ])

    X_full = np.column_stack([
        X_memory_only,
        P, B, P*B,
        np.sin(phi), np.cos(phi),
        np.sin(2*phi), np.cos(2*phi),
        Rlag*P, Rlag*B,
    ])

    y = R[start+h:end+h]

    pred_mem = Ridge(alpha=1e-6, fit_intercept=False).fit(X_memory_only, y).predict(X_memory_only)
    pred_full = Ridge(alpha=1e-6, fit_intercept=False).fit(X_full, y).predict(X_full)

    rows.append({
        "horizon_steps": h,
        "horizon_time": h*dt,
        "R2_memory_only": r2_score(y, pred_mem),
        "R2_full_generative": r2_score(y, pred_full),
        "generative_gain": r2_score(y, pred_full) - r2_score(y, pred_mem),
        "MAE_memory_only": np.mean(np.abs(y - pred_mem)),
        "MAE_full": np.mean(np.abs(y - pred_full)),
    })

report = pd.DataFrame(rows)

print("\n=== R_LAMBDA GENERATIVE MEMORY TEST ===")
print(report.to_string(index=False))

report.to_csv(OUT / "Rlambda_generative_memory_report.csv", index=False)

print(f"\n[OK] wrote {OUT}")
print("[DONE] R_lambda generative memory test complete")