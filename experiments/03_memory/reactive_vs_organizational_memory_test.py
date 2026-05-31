from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

BASE_FUNC = Path("results/research_final/Rlambda_functional_role_test")
BASE_BRANCH = Path("results/research_final/branch_resolution_vs_coexistence_test")

OUT = Path("results/research_final/reactive_vs_organizational_memory_test")
OUT.mkdir(parents=True, exist_ok=True)

func = pd.read_csv(BASE_FUNC / "Rlambda_functional_role_timeseries.csv")
events = pd.read_csv(BASE_BRANCH / "branch_resolution_events.csv")

R = func["R_lambda"].values
phase = func["phase_error"].values
breath = func["breath_error"].values

signals = np.column_stack([R, phase, breath])

future_horizon = 32
windows = [1,2,4,8,12,16,20,24,28,32,40]

rows = []

for w in windows:

    # =========================
    # mémoire réactive locale
    # =========================

    X_local = []
    y_local = []

    for i in range(w, len(signals)-1):

        seg = signals[i-w:i]

        X_local.append(seg.flatten())
        y_local.append(signals[i+1])

    X_local = np.array(X_local)
    y_local = np.array(y_local)

    reg_local = Ridge(alpha=1e-6)
    reg_local.fit(X_local, y_local)

    pred_local = reg_local.predict(X_local)

    reactive_r2 = np.mean([
        r2_score(y_local[:,k], pred_local[:,k])
        for k in range(y_local.shape[1])
    ])

    # =========================
    # mémoire organisationnelle
    # =========================

    X_org = []
    y_org = []

    for _, row in events.iterrows():

        i = int(row["index"])

        if i-w < 0 or i+future_horizon >= len(signals):
            continue

        past = signals[i-w:i]

        features = np.concatenate([
            past.mean(axis=0),
            past.std(axis=0),
            signals[i],
        ])

        future = signals[i+future_horizon]

        X_org.append(features)
        y_org.append(future)

    X_org = np.array(X_org)
    y_org = np.array(y_org)

    reg_org = Ridge(alpha=1e-6)
    reg_org.fit(X_org, y_org)

    pred_org = reg_org.predict(X_org)

    organizational_r2 = np.mean([
        r2_score(y_org[:,k], pred_org[:,k])
        for k in range(y_org.shape[1])
    ])

    rows.append({
        "window": w,
        "reactive_memory_R2": reactive_r2,
        "organizational_memory_R2": organizational_r2,
        "difference_org_minus_reactive":
            organizational_r2 - reactive_r2,
    })

report = pd.DataFrame(rows)

best_reactive = report.iloc[
    np.argmax(report["reactive_memory_R2"])
]

best_org = report.iloc[
    np.argmax(report["organizational_memory_R2"])
]

print("\n=== REACTIVE VS ORGANIZATIONAL MEMORY TEST ===")
print(report.to_string(index=False))

print("\nBEST REACTIVE:")
print(best_reactive.to_string())

print("\nBEST ORGANIZATIONAL:")
print(best_org.to_string())

report.to_csv(
    OUT / "reactive_vs_organizational_memory_report.csv",
    index=False
)

print(f"\n[OK] wrote {OUT}")
print("[DONE] reactive vs organizational memory test complete")