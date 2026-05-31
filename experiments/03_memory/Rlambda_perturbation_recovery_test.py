from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

BASE = Path("results/research_final/delayed_synchronization_attractor_test")
OUT = Path("results/research_final/Rlambda_perturbation_recovery_test")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "delayed_synchronization_timeseries.csv")

t = df["t"].values
R = df["R_lambda"].values
Rdot = df["Rdot"].values
P = df["phase_lag"].values
B = df["breath_lag"].values
dt = np.median(np.diff(t))

phi = np.arctan2(B, P)
sync_sin = np.sin(phi)
sync_cos = np.cos(phi)

X = np.column_stack([
    np.ones_like(R),
    R, R**2, R**3,
    P, B, P*B,
    R*P, R*B,
    sync_sin, sync_cos,
    R*sync_sin, R*sync_cos,
])

reg = Ridge(alpha=1e-6, fit_intercept=False).fit(X, Rdot)

def features(Rv, Pv, Bv):
    ph = np.arctan2(Bv, Pv)
    ss = np.sin(ph)
    cc = np.cos(ph)
    return np.array([
        1.0,
        Rv, Rv**2, Rv**3,
        Pv, Bv, Pv*Bv,
        Rv*Pv, Rv*Bv,
        ss, cc,
        Rv*ss, Rv*cc,
    ])

perturb_index = len(R)//3
horizon = min(300, len(R) - perturb_index - 1)
perturbations = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

rows = []
trajectories = {"t": t[perturb_index:perturb_index+horizon]}

for amp in perturbations:
    R_roll = np.zeros(horizon)
    R_roll[0] = R[perturb_index] + amp

    for i in range(horizon-1):
        j = perturb_index + i
        x = features(R_roll[i], P[j], B[j])
        dR = float(reg.predict(x.reshape(1,-1))[0])
        dR = np.clip(dR, -5, 5)
        R_roll[i+1] = R_roll[i] + dt*dR

    R_ref = R[perturb_index:perturb_index+horizon]
    err = np.abs(R_roll - R_ref)

    initial_err = err[0]
    final_err = err[-1]
    recovery_ratio = final_err / (initial_err + 1e-9)

    below_half = np.where(err < 0.5 * initial_err)[0]
    half_recovery_time = np.nan if len(below_half) == 0 else below_half[0] * dt

    rows.append({
        "perturbation": amp,
        "initial_error": initial_err,
        "final_error": final_err,
        "recovery_ratio": recovery_ratio,
        "half_recovery_time": half_recovery_time,
        "recovered": recovery_ratio < 0.5,
    })

    trajectories[f"R_perturb_{amp}"] = R_roll

trajectories["R_reference"] = R[perturb_index:perturb_index+horizon]

report = pd.DataFrame(rows)

print("\n=== R_LAMBDA PERTURBATION RECOVERY TEST ===")
print(report.to_string(index=False))
print("\nMean recovery ratio:", report["recovery_ratio"].mean())
print("Recovered count:", int(report["recovered"].sum()), "/", len(report))

report.to_csv(OUT / "Rlambda_perturbation_recovery_report.csv", index=False)
pd.DataFrame(trajectories).to_csv(OUT / "Rlambda_perturbation_trajectories.csv", index=False)

print(f"\n[OK] wrote {OUT}")
print("[DONE] R_lambda perturbation recovery test complete")