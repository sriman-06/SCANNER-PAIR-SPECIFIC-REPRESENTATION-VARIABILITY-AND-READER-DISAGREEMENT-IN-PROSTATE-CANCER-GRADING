
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(r"D:\ISBI_2027")
INPUT = (
    BASE / "outputs" / "patient_level_aggregation_audit_30"
    / "slide_level_table_with_share.csv"
)
OUT = (
    BASE / "outputs" / "patient_level_aggregation_audit_30"
)
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = [
    "case_id",
    "patient_id_inferred",
    "reader_ambiguity",
    "mean_directional_HL",
    "HL_directional_share",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()

def get_rho(data, metric):
    if (
        len(data) < 3
        or data["reader_ambiguity"].nunique() < 2
        or data[metric].nunique() < 2
    ):
        return np.nan

    return float(
        spearmanr(data["reader_ambiguity"], data[metric]).statistic
    )

metrics = ["mean_directional_HL", "HL_directional_share"]
patients = sorted(df["patient_id_inferred"].unique())

rows = []

for patient in patients:
    subset = df[df["patient_id_inferred"] != patient]

    for metric in metrics:
        rows.append({
            "excluded_patient": patient,
            "metric": metric,
            "remaining_slides": len(subset),
            "remaining_patients": subset["patient_id_inferred"].nunique(),
            "spearman_rho": get_rho(subset, metric),
        })

results = pd.DataFrame(rows)
results.to_csv(OUT / "leave_one_patient_out_correlations.csv", index=False)

print("Patients:", len(patients))
print("Saved:", OUT / "leave_one_patient_out_correlations.csv")

for metric in metrics:
    values = results.loc[
        results["metric"] == metric, "spearman_rho"
    ].dropna()

    print(f"\n{metric}")
    print(f"Valid leave-one-patient-out runs: {len(values)}")
    print(f"Minimum rho: {values.min():.6f}")
    print(f"Maximum rho: {values.max():.6f}")
    print(f"All correlations positive: {bool((values > 0).all())}")
