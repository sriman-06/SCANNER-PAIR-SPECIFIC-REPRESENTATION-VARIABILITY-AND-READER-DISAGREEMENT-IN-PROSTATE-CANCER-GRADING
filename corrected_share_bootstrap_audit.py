
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(r"D:\ISBI_2027")
INPUT = (
    BASE / "outputs" / "patient_pair_reproducibility_audit_27"
    / "corrected_patient_hl_share.csv"
)
OUT_DIR = BASE / "outputs" / "isbi_frozen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 10000
SEED = 20261016

PRIMARY = "HL_directional_share_corrected"
METRICS = [
    PRIMARY,
    "mean_directional_HL_stored",
    "overall_directional_mean",
    "mean_cosine_disagreement_HL",
    "mean_squared_distance_HL",
]


def safe_rho(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    return float(result.statistic)


def main():
    df = pd.read_csv(INPUT)

    # Build the overall directional metric from the patient-level values.
    gh = "mean_directional_GH_stored"
    gl = "mean_directional_GL_stored"
    hl = "mean_directional_HL_stored"

    required = [
        "patient_id",
        "reader_ambiguity",
        PRIMARY,
        gh,
        gl,
        hl,
        "mean_cosine_disagreement_HL",
        "mean_squared_distance_HL",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["overall_directional_mean"] = df[[gh, gl, hl]].mean(axis=1)

    df = df[required[:3] + [
        gh, gl, hl,
        "overall_directional_mean",
        "mean_cosine_disagreement_HL",
        "mean_squared_distance_HL",
    ]].copy()

    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient rows detected.")
    if df.isna().any().any():
        raise ValueError("Missing values detected.")

    n = len(df)
    if n < 5:
        raise ValueError(f"Too few patients for this audit: n={n}")

    x = df["reader_ambiguity"].to_numpy(dtype=float)
    observed_rows = []

    for metric in METRICS:
        y = df[metric].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = spearmanr(x, y)

        observed_rows.append({
            "metric": metric,
            "n_patients": n,
            "observed_rho": float(result.statistic),
            "observed_p": float(result.pvalue),
        })

    observed = pd.DataFrame(observed_rows)

    rng = np.random.default_rng(SEED)
    boot_values = {metric: [] for metric in METRICS}

    # Paired patient bootstrap: the same sampled patients are used
    # for every metric within each replicate.
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]

        for metric in METRICS:
            yb = df[metric].to_numpy(dtype=float)[idx]
            r = safe_rho(xb, yb)
            if np.isfinite(r):
                boot_values[metric].append(r)

    bootstrap_rows = []
    for metric in METRICS:
        vals = np.asarray(boot_values[metric], dtype=float)
        bootstrap_rows.append({
            "metric": metric,
            "valid_replicates": len(vals),
            "bootstrap_median_rho": float(np.median(vals)),
            "ci_95_low": float(np.quantile(vals, 0.025)),
            "ci_95_high": float(np.quantile(vals, 0.975)),
            "fraction_rho_positive": float(np.mean(vals > 0)),
        })

    bootstrap = pd.DataFrame(bootstrap_rows)

    # Paired bootstrap differences: corrected share minus each comparator.
    comparisons = [
        "mean_directional_HL_stored",
        "overall_directional_mean",
        "mean_cosine_disagreement_HL",
        "mean_squared_distance_HL",
    ]
    difference_rows = []

    for comparator in comparisons:
        primary_vals = df[PRIMARY].to_numpy(dtype=float)
        comparator_vals = df[comparator].to_numpy(dtype=float)

        observed_delta = (
            safe_rho(x, primary_vals)
            - safe_rho(x, comparator_vals)
        )

        deltas = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            xb = x[idx]
            rp = safe_rho(xb, primary_vals[idx])
            rc = safe_rho(xb, comparator_vals[idx])

            if np.isfinite(rp) and np.isfinite(rc):
                deltas.append(rp - rc)

        deltas = np.asarray(deltas, dtype=float)
        difference_rows.append({
            "comparison": f"{PRIMARY} minus {comparator}",
            "observed_delta_rho": observed_delta,
            "valid_replicates": len(deltas),
            "bootstrap_median_delta": float(np.median(deltas)),
            "delta_ci_95_low": float(np.quantile(deltas, 0.025)),
            "delta_ci_95_high": float(np.quantile(deltas, 0.975)),
            "fraction_delta_positive": float(np.mean(deltas > 0)),
        })

    differences = pd.DataFrame(difference_rows)

    # Save results.
    observed.to_csv(
        OUT_DIR / "corrected_share_observed_associations.csv", index=False
    )
    bootstrap.to_csv(
        OUT_DIR / "corrected_share_bootstrap_intervals.csv", index=False
    )
    differences.to_csv(
        OUT_DIR / "corrected_share_paired_differences.csv", index=False
    )

    print("Patients:", n)
    print("Bootstrap iterations requested:", N_BOOT)

    print("\n=== OBSERVED ASSOCIATIONS ===")
    print(observed.to_string(index=False))

    print("\n=== BOOTSTRAP CORRELATION INTERVALS ===")
    print(bootstrap.to_string(index=False))

    print("\n=== CORRECTED SHARE PAIRED COMPARISONS ===")
    print(differences.to_string(index=False))

    print("\nSaved files:")
    print(OUT_DIR / "corrected_share_observed_associations.csv")
    print(OUT_DIR / "corrected_share_bootstrap_intervals.csv")
    print(OUT_DIR / "corrected_share_paired_differences.csv")

    print(
        "\nCaution: these bootstrap intervals reflect resampling stability "
        "within this 27-patient ambiguity-stratified cohort. They do not "
        "establish population-level generalization."
    )


if __name__ == "__main__":
    main()
