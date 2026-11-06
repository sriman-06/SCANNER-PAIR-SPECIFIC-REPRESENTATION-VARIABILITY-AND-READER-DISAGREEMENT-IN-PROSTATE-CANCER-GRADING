
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent

PAIR_PATH = Path(
    os.environ.get(
        "ISBI_PAIR_ROBUSTNESS_OUTPUT_ROOT",
        PROJECT_ROOT / "outputs" / "scanner_pair_directional_robustness_30"
    )
).expanduser().resolve() / "scanner_pair_patch_level_900.csv"

MAP_PATH = Path(
    os.environ.get(
        "ISBI_PATIENT_MAPPING_PATH",
        PROJECT_ROOT / "outputs" / "patient_mapping_validation"
        / "documented_patient_mapping.csv"
    )
).expanduser().resolve()

PATIENT_PATH = Path(
    os.environ.get(
        "ISBI_PATIENT_TABLE_PATH",
        PROJECT_ROOT / "outputs" / "patient_level_aggregation_audit_30"
        / "patient_level_aggregated_table.csv"
    )
).expanduser().resolve()

OUT = Path(
    os.environ.get(
        "ISBI_PATIENT_INCREMENTAL_OUTPUT_ROOT",
        PROJECT_ROOT / "outputs" / "patient_incremental_value_audit_27"
    )
).expanduser().resolve()

N_BOOT = 10000
SEED = 20261016


def find_col(df, candidates):
    lookup = {
        str(c).strip().lower(): c for c in df.columns
    }
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"Could not find {candidates}; columns={df.columns.tolist()}"
    )


def rho(x, y):
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def bootstrap_rhos_and_differences(
    data, metrics, outcome="reader_ambiguity"
):
    """
    Resample patients as paired rows.
    Correlation differences are calculated on identical bootstrap samples.
    """
    rng = np.random.default_rng(SEED)
    n = len(data)
    x = data[outcome].to_numpy(dtype=float)

    boot = {m: [] for m in metrics}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]

        for metric in metrics:
            yb = data[metric].to_numpy(dtype=float)[idx]
            r, _ = rho(xb, yb)
            if np.isfinite(r):
                boot[metric].append(r)

    rows = []
    for metric in metrics:
        values = np.asarray(boot[metric], dtype=float)
        rows.append({
            "metric": metric,
            "bootstrap_valid": len(values),
            "bootstrap_median_rho": np.median(values),
            "bootstrap_ci_low": np.quantile(values, 0.025),
            "bootstrap_ci_high": np.quantile(values, 0.975),
            "fraction_rho_positive": np.mean(values > 0),
        })

    summary = pd.DataFrame(rows)

    # Paired bootstrap differences in Spearman correlations.
    # Compare every baseline against the H-L directional metric.
    primary = "mean_directional_HL"
    diff_rows = []

    for baseline in metrics:
        if baseline == primary:
            continue

        differences = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            xb = x[idx]

            yp = data[primary].to_numpy(dtype=float)[idx]
            yb = data[baseline].to_numpy(dtype=float)[idx]

            rp, _ = rho(xb, yp)
            rb, _ = rho(xb, yb)

            if np.isfinite(rp) and np.isfinite(rb):
                differences.append(rp - rb)

        differences = np.asarray(differences, dtype=float)
        diff_rows.append({
            "comparison": f"{primary} minus {baseline}",
            "bootstrap_valid": len(differences),
            "observed_delta_rho": (
                rho(x, data[primary])[0]
                - rho(x, data[baseline])[0]
            ),
            "bootstrap_median_delta": np.median(differences),
            "delta_ci_low": np.quantile(differences, 0.025),
            "delta_ci_high": np.quantile(differences, 0.975),
            "fraction_delta_positive": np.mean(differences > 0),
        })

    return summary, pd.DataFrame(diff_rows)


def main():
    for input_file in [PAIR_PATH, MAP_PATH, PATIENT_PATH]:
        if not input_file.is_file():
            raise FileNotFoundError(f"Required input file not found: {input_file}")

    OUT.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIR_PATH)
    mapping = pd.read_csv(MAP_PATH)
    patients = pd.read_csv(PATIENT_PATH)

    print("Patch-level columns:", pairs.columns.tolist())
    print("Mapping columns:", mapping.columns.tolist())
    print("Patient columns:", patients.columns.tolist())

    # Resolve input columns.
    case_col = find_col(pairs, ["case_id", "slide_id"])
    pair_col = find_col(pairs, ["pair", "scanner_pair"])
    directional_col = find_col(
        pairs, ["directional_component", "directional"]
    )
    cosine_col = find_col(
        pairs, ["cosine_disagreement", "cosine_distance"]
    )
    distance_col = find_col(
        pairs, ["squared_distance", "distance_squared"]
    )

    map_slide_col = find_col(mapping, ["slide_id", "case_id"])
    map_patient_col = find_col(
        mapping, ["patient_id_inferred", "patient_id"]
    )

    patient_id_col = find_col(patients, ["patient_id"])
    ambiguity_col = find_col(
        patients, ["reader_ambiguity", "ambiguity"]
    )
    grade_col = find_col(patients, ["mean_isup", "mean_grade"])

    pairs[case_col] = pairs[case_col].astype(str).str.strip()
    mapping[map_slide_col] = (
        mapping[map_slide_col].astype(str).str.strip()
    )
    mapping[map_patient_col] = (
        mapping[map_patient_col].astype(str).str.strip()
    )
    patients[patient_id_col] = (
        patients[patient_id_col].astype(str).str.strip()
    )

    pairs[pair_col] = pairs[pair_col].astype(str).str.upper().str.strip()

    # Ensure only the expected scanner pairs are included.
    expected_pairs = {"GH", "GL", "HL"}
    actual_pairs = set(pairs[pair_col].unique())
    if not expected_pairs.issubset(actual_pairs):
        raise ValueError(
            f"Expected pairs {expected_pairs}; found {actual_pairs}"
        )

    # Aggregate patch/location measurements to a slide-level profile.
    slide_profile = (
        pairs.groupby([case_col, pair_col], as_index=False)
        .agg(
            directional=(directional_col, "mean"),
            cosine_disagreement=(cosine_col, "mean"),
            squared_distance=(distance_col, "mean"),
            n_locations=(directional_col, "count"),
        )
    )

    # Each scanner pair becomes a separate set of columns.
    directional = slide_profile.pivot(
        index=case_col, columns=pair_col, values="directional"
    )
    directional.columns = [
        f"mean_directional_{c}" for c in directional.columns
    ]

    cosine = slide_profile.pivot(
        index=case_col, columns=pair_col, values="cosine_disagreement"
    )
    cosine.columns = [
        f"mean_cosine_disagreement_{c}" for c in cosine.columns
    ]

    distance = slide_profile.pivot(
        index=case_col, columns=pair_col, values="squared_distance"
    )
    distance.columns = [
        f"mean_squared_distance_{c}" for c in distance.columns
    ]

    slide_metrics = pd.concat(
        [directional, cosine, distance], axis=1
    ).reset_index().rename(columns={case_col: "slide_id"})

    # Preserve only slides that belong to the existing 27-patient cohort.
    selected_patient_ids = set(patients[patient_id_col])
    mapping_selected = mapping[
        mapping[map_patient_col].isin(selected_patient_ids)
    ].copy()

    slide_metrics = slide_metrics.merge(
        mapping_selected[[map_slide_col, map_patient_col]].drop_duplicates(),
        left_on="slide_id",
        right_on=map_slide_col,
        how="inner",
        validate="many_to_one",
    )

    # Aggregate slides equally within each patient.
    metric_cols = [
        c for c in slide_metrics.columns
        if c.startswith("mean_directional_")
        or c.startswith("mean_cosine_disagreement_")
        or c.startswith("mean_squared_distance_")
    ]

    patient_metrics = (
        slide_metrics.groupby(map_patient_col, as_index=False)[metric_cols]
        .mean()
        .rename(columns={map_patient_col: "patient_id"})
    )

# Keep the already validated patient-level directional metrics.
# Merge only the additional baseline metrics to avoid _x/_y suffixes.
    extra_metrics = [
        "mean_cosine_disagreement_GH",
        "mean_cosine_disagreement_GL",
        "mean_cosine_disagreement_HL",
        "mean_squared_distance_GH",
        "mean_squared_distance_GL",
        "mean_squared_distance_HL",
    ]

    data = patients.rename(
        columns={
            patient_id_col: "patient_id",
            ambiguity_col: "reader_ambiguity",
            grade_col: "mean_isup",
        }
    ).merge(
        patient_metrics[["patient_id"] + extra_metrics],
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    # Main metric and simple baselines.
    directional_cols = [
        "mean_directional_GH",
        "mean_directional_GL",
        "mean_directional_HL",
    ]
    required = directional_cols + [
        "mean_cosine_disagreement_HL",
        "mean_squared_distance_HL",
        "reader_ambiguity",
        "mean_isup",
    ]

    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required metrics {missing_cols}. "
            f"Available: {data.columns.tolist()}"
        )

    data["overall_directional_mean"] = data[
        directional_cols
    ].mean(axis=1)

    # Include the primary metric, overall metric, cosine baseline,
    # and squared-distance baseline.
    metrics = [
        "mean_directional_HL",
        "overall_directional_mean",
        "mean_cosine_disagreement_HL",
        "mean_squared_distance_HL",
    ]

    if data[required].isna().any().any():
        print("\nMissing values by column:")
        print(data[required].isna().sum())
        raise ValueError("Missing data detected; audit stopped.")

    data.to_csv(OUT / "patient_metric_comparison_table.csv", index=False)

    print("\nPatients:", len(data))
    print("\n=== OBSERVED SPEARMAN ASSOCIATIONS ===")
    observed = []
    for metric in metrics:
        r, p = rho(data["reader_ambiguity"], data[metric])
        observed.append({
            "metric": metric,
            "n_patients": len(data),
            "spearman_rho": r,
            "p_value": p,
        })
    observed_df = pd.DataFrame(observed)
    print(observed_df.to_string(index=False))
    observed_df.to_csv(
        OUT / "observed_metric_associations.csv", index=False
    )

    print("\nRunning paired patient bootstrap...")
    boot_summary, differences = bootstrap_rhos_and_differences(
        data, metrics
    )

    print("\n=== PATIENT BOOTSTRAP CORRELATIONS ===")
    print(boot_summary.to_string(index=False))
    boot_summary.to_csv(
        OUT / "patient_bootstrap_correlations.csv", index=False
    )

    print("\n=== PAIRED CORRELATION DIFFERENCES ===")
    print(differences.to_string(index=False))
    differences.to_csv(
        OUT / "paired_correlation_differences.csv", index=False
    )

    print("\nSaved outputs to:", OUT)
    print(
        "\nInterpretation reminder: this is a small, ambiguity-stratified "
        "cohort. Bootstrap intervals describe resampling stability within "
        "this cohort, not population-level generalization."
    )


if __name__ == "__main__":
    main()
