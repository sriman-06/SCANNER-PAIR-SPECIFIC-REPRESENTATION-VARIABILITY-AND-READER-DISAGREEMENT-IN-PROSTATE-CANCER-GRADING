
from pathlib import Path
import ast
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata
import statsmodels.api as sm

BASE = Path(r"D:\ISBI_2027")

MANIFEST_PATH = (
    BASE / "outputs" / "uni2h_30" / "embedding_manifest_30.csv"
)
PATIENT_PATH = (
    BASE / "outputs" / "patient_level_aggregation_audit_30"
    / "patient_level_aggregated_table.csv"
)
MAPPING_PATH = (
    BASE / "outputs" / "patient_mapping_validation"
    / "documented_patient_mapping.csv"
)
OUT = BASE / "outputs" / "patient_adjusted_audit_27"
OUT.mkdir(parents=True, exist_ok=True)


def norm_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def find_col(df, candidates, required=True):
    lookup = {norm_name(c): c for c in df.columns}
    for candidate in candidates:
        if norm_name(candidate) in lookup:
            return lookup[norm_name(candidate)]
    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return None


def load_embedding(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE / path

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        if len(loaded.files) != 1:
            raise ValueError(
                f"NPZ has multiple arrays; inspect manually: {path}"
            )
        arr = loaded[loaded.files[0]]
    elif suffix in {".pt", ".pth"}:
        try:
            obj = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except TypeError:
            obj = torch.load(path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            tensors = [
                v for v in obj.values()
                if isinstance(v, torch.Tensor)
            ]
            arrays = [
                v for v in obj.values()
                if isinstance(v, np.ndarray)
            ]
            if len(tensors) == 1:
                arr = tensors[0].detach().cpu().numpy()
            elif len(arrays) == 1:
                arr = arrays[0]
            else:
                raise ValueError(
                    f"Cannot uniquely identify embedding array in {path}"
                )
        else:
            arr = np.asarray(obj)
    else:
        raise ValueError(
            f"Unsupported embedding format {suffix}: {path}"
        )

    arr = np.asarray(arr, dtype=np.float64).squeeze()
    if arr.ndim != 1:
        raise ValueError(
            f"Expected one embedding vector, got {arr.shape}: {path}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"NaN/Inf in embedding: {path}")
    return arr


def spearman(x, y):
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def partial_spearman(x, y, controls):
    """Rank-transform variables, then correlate residuals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = np.asarray(controls, dtype=float)

    rx = rankdata(x)
    ry = rankdata(y)
    rc = np.column_stack(
        [rankdata(controls[:, j]) for j in range(controls.shape[1])]
    )
    design = sm.add_constant(rc)

    rx_resid = rx - design @ np.linalg.lstsq(
        design, rx, rcond=None
    )[0]
    ry_resid = ry - design @ np.linalg.lstsq(
        design, ry, rcond=None
    )[0]

    rho, p = spearman(rx_resid, ry_resid)
    return rho, p


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    patient_table = pd.read_csv(PATIENT_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    print("Manifest columns:", manifest.columns.tolist())
    print("Mapping columns:", mapping.columns.tolist())

    path_col = find_col(
        manifest, ["embedding_path", "path", "file_path", "embedding_file"]
    )
    slide_col = find_col(
        manifest, ["case_id", "slide_id", "slide", "case"]
    )
    scanner_col = find_col(
        manifest, ["scanner", "scanner_id", "scanner_name", "source"]
    )

    map_slide_col = find_col(
        mapping, ["slide_id", "case_id", "slide"]
    )
    map_patient_col = find_col(
        mapping,
        ["patient_id", "patient_id_inferred", "patient", "patient_key"]
    )

    # The existing patient table is the source of the selected cohort.
    pt_id_col = find_col(patient_table, ["patient_id"])
    ambiguity_col = find_col(
        patient_table, ["reader_ambiguity", "ambiguity"]
    )
    grade_col = find_col(
        patient_table, ["mean_isup", "mean_grade", "mean_isup_grade"]
    )

    metric_col = find_col(
        patient_table, ["mean_directional_HL"]
    )
    share_col = find_col(
        patient_table, ["HL_directional_share"]
    )

    patient_table[pt_id_col] = patient_table[pt_id_col].astype(str).str.strip()
    mapping[map_slide_col] = mapping[map_slide_col].astype(str).str.strip()
    mapping[map_patient_col] = mapping[map_patient_col].astype(str).str.strip()
    manifest[slide_col] = manifest[slide_col].astype(str).str.strip()

    # Keep the same selected 27-patient cohort.
    selected_patients = set(patient_table[pt_id_col])
    mapping = mapping[
        mapping[map_patient_col].isin(selected_patients)
    ].copy()

    # Calculate the L2 norm from each saved embedding vector.
    norm_rows = []
    cache = {}
    for _, row in manifest.iterrows():
        path_value = str(row[path_col])
        if path_value not in cache:
            cache[path_value] = load_embedding(path_value)
        vec = cache[path_value]

        norm_rows.append({
            "slide_id": str(row[slide_col]).strip(),
            "scanner": str(row[scanner_col]).strip(),
            "embedding_norm": float(np.linalg.norm(vec)),
        })

    norms = pd.DataFrame(norm_rows)

    # Aggregate patch norms within each slide and scanner.
    slide_scanner = (
        norms.groupby(["slide_id", "scanner"], as_index=False)
        ["embedding_norm"].mean()
    )

    # One row per slide: average across scanners equally.
    slide_norm = (
        slide_scanner.groupby("slide_id", as_index=False)
        ["embedding_norm"].mean()
        .rename(columns={"embedding_norm": "mean_embedding_norm"})
    )

    # Map selected slides to documented patient IDs.
    slide_norm = slide_norm.merge(
        mapping[[map_slide_col, map_patient_col]].drop_duplicates(),
        left_on="slide_id",
        right_on=map_slide_col,
        how="inner",
        validate="many_to_one",
    )

    # Aggregate selected slides to patient level.
    patient_norm = (
        slide_norm.groupby(map_patient_col, as_index=False)
        ["mean_embedding_norm"].mean()
        .rename(columns={map_patient_col: "patient_id"})
    )

    patient_norm["patient_id"] = patient_norm["patient_id"].astype(str)
    patient_table["patient_id"] = patient_table[pt_id_col].astype(str)

    data = patient_table.merge(
        patient_norm, on="patient_id", how="left", validate="one_to_one"
    )

    if data["mean_embedding_norm"].isna().any():
        missing = data.loc[
            data["mean_embedding_norm"].isna(), "patient_id"
        ].tolist()
        raise ValueError(
            f"Missing patient embedding norms for: {missing}"
        )

    # Standardize for interpretable OLS coefficients.
    variables = {
        "ambiguity": ambiguity_col,
        "mean_isup": grade_col,
        "mean_embedding_norm": "mean_embedding_norm",
        "mean_directional_HL": metric_col,
        "HL_directional_share": share_col,
    }
    for new_name, source_col in variables.items():
        data[new_name] = pd.to_numeric(data[source_col], errors="raise")

    data.to_csv(OUT / "patient_level_norm_adjusted_table.csv", index=False)

    print("\nPatients analyzed:", len(data))
    print("Missing values in analysis columns:")
    print(data[list(variables)].isna().sum())

    results = []
    for outcome in ["mean_directional_HL", "HL_directional_share"]:
        x = data["ambiguity"].to_numpy()
        y = data[outcome].to_numpy()
        controls = data[
            ["mean_isup", "mean_embedding_norm"]
        ].to_numpy()

        raw_rho, raw_p = spearman(x, y)
        partial_rho, partial_p = partial_spearman(x, y, controls)

        # OLS outcome ~ ambiguity + mean ISUP + embedding norm.
        model_data = data[
            [outcome, "ambiguity", "mean_isup", "mean_embedding_norm"]
        ].astype(float)

        z = (model_data - model_data.mean()) / model_data.std(ddof=1)
        X = sm.add_constant(
            z[["ambiguity", "mean_isup", "mean_embedding_norm"]]
        )
        model = sm.OLS(z[outcome], X).fit(cov_type="HC3")

        ci = model.conf_int().loc["ambiguity"]
        results.append({
            "outcome": outcome,
            "n_patients": len(data),
            "raw_spearman_rho": raw_rho,
            "raw_spearman_p": raw_p,
            "partial_spearman_rho": partial_rho,
            "partial_spearman_p_approx": partial_p,
            "standardized_ols_beta_ambiguity": model.params["ambiguity"],
            "hc3_p_ambiguity": model.pvalues["ambiguity"],
            "hc3_ci_low": ci.iloc[0],
            "hc3_ci_high": ci.iloc[1],
            "r_squared": model.rsquared,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "patient_adjusted_association_results.csv", index=False)

    print("\n=== PATIENT-LEVEL ADJUSTED RESULTS ===")
    print(result_df.to_string(index=False))
    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()
