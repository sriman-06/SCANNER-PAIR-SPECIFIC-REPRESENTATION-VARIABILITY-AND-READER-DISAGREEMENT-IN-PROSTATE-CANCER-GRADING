
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# 1. PATHS
# ============================================================

BASE = Path(r"D:\ISBI_2027")

PROFILE_FILE = (
    BASE / "outputs" / "label_blind_scanner_pair_audit_30"
    / "label_blind_case_profile_30.csv"
)

LABEL_FILE = BASE / "annotations" / "slide_labels.tsv"

MAPPING_FILE = (
    BASE / "outputs" / "patient_mapping_validation"
    / "documented_patient_mapping.csv"
)

OUT = BASE / "outputs" / "patient_level_aggregation_audit_30"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE SCANNER PROFILE
# ============================================================

if not PROFILE_FILE.exists():
    raise FileNotFoundError(
        f"Scanner profile not found:\n{PROFILE_FILE}"
    )

profile = pd.read_csv(PROFILE_FILE)

required_profile_cols = [
    "case_id",
    "mean_directional_GH",
    "mean_directional_GL",
    "mean_directional_HL",
]

missing = [
    col for col in required_profile_cols
    if col not in profile.columns
]

if missing:
    raise ValueError(
        f"Missing scanner-profile columns: {missing}\n"
        f"Available columns: {profile.columns.tolist()}"
    )

profile["case_id"] = (
    profile["case_id"].astype(str).str.strip().str.upper()
)

if profile["case_id"].duplicated().any():
    duplicates = profile.loc[
        profile["case_id"].duplicated(keep=False), "case_id"
    ].tolist()
    raise ValueError(f"Duplicate scanner-profile slide IDs: {duplicates}")


# ============================================================
# 3. LOAD READER LABELS
# ============================================================

if not LABEL_FILE.exists():
    raise FileNotFoundError(
        f"Reader-label file not found:\n{LABEL_FILE}"
    )

labels = pd.read_csv(LABEL_FILE, sep="\t", dtype=str)

labels = labels.apply(
    lambda col: col.str.strip() if col.dtype == "object" else col
)

if "slide_id" not in labels.columns:
    raise ValueError("The reader-label file has no slide_id column.")

labels["case_id"] = (
    labels["slide_id"].astype(str).str.strip().str.upper()
)

if labels["case_id"].duplicated().any():
    duplicates = labels.loc[
        labels["case_id"].duplicated(keep=False), "case_id"
    ].tolist()
    raise ValueError(f"Duplicate slide IDs in labels: {duplicates}")

isup_cols = ["isup_1", "isup_r2", "isup_r3"]

missing = [col for col in isup_cols if col not in labels.columns]
if missing:
    raise ValueError(f"Missing reader ISUP columns: {missing}")

for col in isup_cols:
    labels[col] = pd.to_numeric(labels[col], errors="coerce")

# Reader ambiguity is the range across available reader grades.
# No consensus label is constructed.
labels["reader_ambiguity"] = (
    labels[isup_cols].max(axis=1, skipna=True)
    - labels[isup_cols].min(axis=1, skipna=True)
)

labels["mean_isup"] = labels[isup_cols].mean(axis=1, skipna=True)


# ============================================================
# 4. LOAD VALIDATED PATIENT MAPPING
# ============================================================

if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"Validated mapping not found:\n{MAPPING_FILE}"
    )

mapping = pd.read_csv(MAPPING_FILE, dtype=str)

mapping = mapping.apply(
    lambda col: col.str.strip() if col.dtype == "object" else col
)

required_mapping_cols = ["slide_id", "patient_id_inferred"]

missing = [
    col for col in required_mapping_cols
    if col not in mapping.columns
]

if missing:
    raise ValueError(
        f"Missing mapping columns: {missing}\n"
        f"Available columns: {mapping.columns.tolist()}"
    )

mapping["slide_id"] = (
    mapping["slide_id"].astype(str).str.strip().str.upper()
)

mapping["patient_id_inferred"] = (
    mapping["patient_id_inferred"].astype(str).str.strip().str.upper()
)

if mapping["slide_id"].duplicated().any():
    raise ValueError("Duplicate slide IDs in mapping CSV.")

if mapping["patient_id_inferred"].isna().any():
    raise ValueError("Missing patient IDs in mapping CSV.")

if (mapping["patient_id_inferred"] == "").any():
    raise ValueError("Empty patient IDs in mapping CSV.")

print("=" * 65)
print("PATIENT MAPPING VALIDATION")
print("=" * 65)
print("Mapping rows:", len(mapping))
print("Unique mapped slides:", mapping["slide_id"].nunique())
print("Unique mapped patients:", mapping["patient_id_inferred"].nunique())


# ============================================================
# 5. ATTACH MAPPING TO LABELS
# ============================================================

labels = labels.merge(
    mapping[["slide_id", "patient_id_inferred"]],
    on="slide_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

unmapped = labels.loc[
    labels["_merge"] != "both", "slide_id"
].tolist()

if unmapped:
    raise ValueError(
        f"Slides missing from patient mapping: {unmapped}"
    )

labels = labels.drop(columns="_merge")

# Keep one row per slide.
label_cols = [
    "case_id",
    "patient_id_inferred",
    "reader_ambiguity",
    "mean_isup",
]

labels = labels[label_cols].copy()


# ============================================================
# 6. MERGE SCANNER PROFILE WITH LABELS
# ============================================================

df = profile.merge(
    labels,
    on="case_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

unmatched = df.loc[df["_merge"] != "both", "case_id"].tolist()

if unmatched:
    raise ValueError(
        f"Scanner-profile slides missing labels/mapping: {unmatched}"
    )

df = df.drop(columns="_merge")

if len(df) != 30:
    raise ValueError(
        f"Expected 30 selected slides, found {len(df)}."
    )

if df["patient_id_inferred"].isna().any():
    raise ValueError("Some selected slides have no patient mapping.")

if df["reader_ambiguity"].isna().any():
    missing_ids = df.loc[
        df["reader_ambiguity"].isna(), "case_id"
    ].tolist()
    raise ValueError(
        f"Reader ambiguity unavailable for selected slides: {missing_ids}"
    )

n_slides = len(df)
n_patients = df["patient_id_inferred"].nunique()

print("\n" + "=" * 65)
print("SELECTED COHORT")
print("=" * 65)
print(f"Matched slides: {n_slides}")
print(f"Mapped patients: {n_patients}")
print(f"Patients with multiple selected slides: "
      f"{(df.groupby('patient_id_inferred').size() > 1).sum()}")


# ============================================================
# 7. CALCULATE H-L DIRECTIONAL SHARE
# ============================================================

directional_cols = [
    "mean_directional_GH",
    "mean_directional_GL",
    "mean_directional_HL",
]

if not all(col in df.columns for col in directional_cols):
    raise ValueError("One or more directional columns are missing.")

if df[directional_cols].isna().any().any():
    raise ValueError("Missing scanner directional values.")

denominator = df[directional_cols].sum(axis=1)

if (denominator <= 0).any():
    raise ValueError(
        "Non-positive directional sum prevents calculating H-L share."
    )

df["HL_directional_share"] = (
    df["mean_directional_HL"] / denominator
)


# ============================================================
# 8. SAVE SLIDE-LEVEL TABLE
# ============================================================

slide_table = df.copy()

slide_table.to_csv(
    OUT / "slide_level_table_with_share.csv",
    index=False,
)


# ============================================================
# 9. AGGREGATE SELECTED SLIDES BY PATIENT
# ============================================================

# Average slide-level measurements within each inferred patient.
# Average reader ambiguity across available slide-level labels.
# This is not a consensus grade and does not imply patient-level
# pathologist annotation.

aggregation = {
    "reader_ambiguity": "mean",
    "mean_isup": "mean",
    "mean_directional_GH": "mean",
    "mean_directional_GL": "mean",
    "mean_directional_HL": "mean",
    "HL_directional_share": "mean",
    "case_id": ["count", lambda x: ";".join(sorted(x.astype(str)))],
}

patient_table = (
    df.groupby("patient_id_inferred", as_index=False)
    .agg(aggregation)
)

# Flatten the multi-level column names generated by aggregation.
flattened = []
for col in patient_table.columns:
    if isinstance(col, tuple):
        parts = [str(x) for x in col if x]
        flattened.append("_".join(parts))
    else:
        flattened.append(str(col))

patient_table.columns = flattened

patient_table = patient_table.rename(
    columns={
        "patient_id_inferred": "patient_id",
        "reader_ambiguity_mean": "reader_ambiguity",
        "mean_isup_mean": "mean_isup",
        "mean_directional_GH_mean": "mean_directional_GH",
        "mean_directional_GL_mean": "mean_directional_GL",
        "mean_directional_HL_mean": "mean_directional_HL",
        "HL_directional_share_mean": "HL_directional_share",
        "case_id_count": "number_of_selected_slides",
        "case_id_<lambda>": "selected_slide_ids",
    }
)

patient_table.to_csv(
    OUT / "patient_level_aggregated_table.csv",
    index=False,
)


# ============================================================
# 10. SPEARMAN ASSOCIATION FUNCTION
# ============================================================

def association_table(data, unit_name):
    metrics = [
        "mean_directional_GH",
        "mean_directional_GL",
        "mean_directional_HL",
        "HL_directional_share",
    ]

    results = []

    for metric in metrics:
        valid = data[["reader_ambiguity", metric]].dropna()

        if len(valid) < 3:
            rho, p_value = np.nan, np.nan
        elif (
            valid["reader_ambiguity"].nunique() < 2
            or valid[metric].nunique() < 2
        ):
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(
                valid["reader_ambiguity"],
                valid[metric],
            )

        results.append({
            "analysis_unit": unit_name,
            "metric": metric,
            "n_observations": len(valid),
            "spearman_rho": rho,
            "p_value": p_value,
        })

    return results


# ============================================================
# 11. SLIDE-LEVEL AND PATIENT-LEVEL ASSOCIATIONS
# ============================================================

results = []

results.extend(association_table(df, "slide"))

results.extend(association_table(patient_table, "patient"))

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "slide_vs_patient_associations.csv",
    index=False,
)


# ============================================================
# 12. REPORT
# ============================================================

print("\n" + "=" * 65)
print("SLIDE-LEVEL VS PATIENT-LEVEL ASSOCIATIONS")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nSaved outputs:")
print(OUT / "slide_vs_patient_associations.csv")
print(OUT / "patient_level_aggregated_table.csv")
print(OUT / "slide_level_table_with_share.csv")

print("\nInterpretation cautions:")
print("- The cohort contains only 30 selected slides.")
print("- Patient groups are reconstructed from the documented naming rule.")
print("- The cohort was selected with reader-disagreement stratification.")
print("- Patient-level ambiguity is averaged across available slide labels.")
print("- These associations are exploratory, not population-level validation.")
print("- No consensus pathology label is created.")
