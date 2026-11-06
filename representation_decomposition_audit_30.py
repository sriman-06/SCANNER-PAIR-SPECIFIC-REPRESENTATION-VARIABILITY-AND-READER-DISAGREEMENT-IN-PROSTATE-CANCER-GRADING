# ============================================================
# ISBI 2027
# Representation Decomposition Audit — 30 Cases
#
# Existing data only:
#   900 UNI2-h embeddings
#   300 matched spatial locations
#   30 cases
#
# No model training.
# No embedding extraction.
# No patch modification.
#
# Main question:
#   Can scanner-induced representation variability be
#   decomposed into magnitude-related and directional
#   components, and how do these relate to grade and
#   reader-level diagnostic ambiguity?
# ============================================================

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, kruskal
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------

# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

EMBED_ROOT = Path(
    os.environ.get(
        "ISBI_EMBED_OUTPUT_ROOT",
        PROJECT_ROOT / "outputs" / "uni2h_30"
    )
).expanduser().resolve()

EMBED_MANIFEST = EMBED_ROOT / "embedding_manifest_30.csv"

LABEL_FILE = Path(
    os.environ.get(
        "ISBI_LABEL_PATH",
        PROJECT_ROOT / "annotations" / "slide_labels.tsv"
    )
).expanduser().resolve()

OUT = Path(
    os.environ.get(
        "ISBI_DECOMPOSITION_OUTPUT_ROOT",
        PROJECT_ROOT / "outputs" / "representation_decomposition_30"
    )
).expanduser().resolve()

OUT.mkdir(parents=True, exist_ok=True)

# PARAMETERS
# ------------------------------------------------------------

SCANNERS = [
    "grundium",
    "hamamatsu",
    "leica",
]

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def clean_id(x):
    return str(x).strip().upper()


def corr(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan

    sr, sp = spearmanr(x[mask], y[mask])
    pr, pp = pearsonr(x[mask], y[mask])

    return (
        float(sr),
        float(sp),
        float(pr),
        float(pp),
    )


def standardized_ols(df, outcome, predictors):

    cols = [outcome] + predictors

    tmp = df[cols].copy()

    for c in cols:
        tmp[c] = pd.to_numeric(
            tmp[c],
            errors="coerce"
        )

    tmp = tmp.dropna()

    if len(tmp) < len(predictors) + 4:
        return None

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_x.fit_transform(
        tmp[predictors]
    )

    y = scaler_y.fit_transform(
        tmp[[outcome]]
    ).ravel()

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    result = {
        "outcome": outcome,
        "n": len(tmp),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
    }

    for i, p in enumerate(
        predictors,
        start=1
    ):
        result[p + "_beta"] = float(
            model.params[i]
        )
        result[p + "_p"] = float(
            model.pvalues[i]
        )

    return result


# ------------------------------------------------------------
# LOAD MANIFEST
# ------------------------------------------------------------

print("=" * 70)
print("ISBI 2027 — REPRESENTATION DECOMPOSITION AUDIT")
print("=" * 70)

print("\nLoading manifest:")
print(EMBED_MANIFEST)

manifest = pd.read_csv(
    EMBED_MANIFEST,
    dtype=str
)

manifest = manifest.apply(
    lambda c: c.str.strip()
    if c.dtype == "object"
    else c
)

manifest["case_id"] = manifest[
    "case_id"
].map(clean_id)

manifest["patch_id"] = pd.to_numeric(
    manifest["patch_id"],
    errors="coerce"
)

manifest["scanner"] = (
    manifest["scanner"]
    .str.lower()
)

print(
    f"\nManifest rows: {len(manifest)}"
)

print(
    f"Cases: {manifest['case_id'].nunique()}"
)

print(
    f"Scanners: {manifest['scanner'].nunique()}"
)

# ------------------------------------------------------------
# EMBEDDING LOADING
# ------------------------------------------------------------

print("\nLoading embeddings...")

records = []

for idx, row in manifest.iterrows():

    raw_path = Path(str(row["embedding_path"]).strip())

    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            EMBED_ROOT / raw_path,
            PROJECT_ROOT / raw_path
        ])

    # Recover paths recorded under an older project root.
    parts = list(raw_path.parts)
    lower_parts = [part.lower() for part in parts]

    if "embeddings" in lower_parts:
        index = lower_parts.index("embeddings")
        relative_parts = parts[index + 1:]

        if relative_parts:
            candidates.append(
                EMBED_ROOT / Path(*relative_parts)
            )

    path = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None
    )

    # Last fallback: accept only a unique filename match.
    if path is None:
        matches = list(EMBED_ROOT.rglob(raw_path.name))

        if len(matches) == 1:
            path = matches[0]

    if path is None:
        raise FileNotFoundError(
            f"Could not resolve embedding: {raw_path}"
        )

    obj = torch_load = None

    # --------------------------------------------------------
    # torch import here to keep startup clean
    # --------------------------------------------------------

    import torch

    z = torch.load(
        path,
        map_location="cpu"
    )

    if isinstance(z, dict):

        # Support common checkpoint structures
        if "embedding" in z:
            z = z["embedding"]

        elif "features" in z:
            z = z["features"]

        elif "z" in z:
            z = z["z"]

        else:
            raise RuntimeError(
                f"Unknown dictionary format:\n{path}"
            )

    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()

    z = np.asarray(z)

    z = np.squeeze(z)

    if z.ndim != 1:
        raise RuntimeError(
            f"Unexpected embedding shape "
            f"{z.shape}: {path}"
        )

    if z.shape[0] != 1536:
        raise RuntimeError(
            f"Expected 1536-D embedding, "
            f"got {z.shape[0]}: {path}"
        )

    if not np.all(
        np.isfinite(z)
    ):
        raise RuntimeError(
            f"NaN/Inf embedding: {path}"
        )

    records.append({
        "case_id": row["case_id"],
        "patch_id": int(
            row["patch_id"]
        ),
        "scanner": row["scanner"],
        "embedding": z.astype(
            np.float32
        ),
    })

print(
    f"Loaded embeddings: {len(records)}"
)

# ------------------------------------------------------------
# BUILD TRIPLETS
# ------------------------------------------------------------

lookup = {}

for r in records:

    key = (
        r["case_id"],
        r["patch_id"],
        r["scanner"],
    )

    lookup[key] = r["embedding"]

cases = sorted(
    manifest["case_id"].unique()
)

triplets = []

for case in cases:

    patches = sorted(
        manifest.loc[
            manifest["case_id"] == case,
            "patch_id"
        ].dropna().unique()
    )

    for patch in patches:

        vectors = {}

        for scanner in SCANNERS:

            key = (
                case,
                int(patch),
                scanner,
            )

            if key not in lookup:
                raise RuntimeError(
                    f"Missing triplet: "
                    f"{key}"
                )

            vectors[scanner] = lookup[key]

        triplets.append({
            "case_id": case,
            "patch_id": int(patch),
            **{
                f"z_{s}": vectors[s]
                for s in SCANNERS
            }
        })

print(
    f"Matched triplets: {len(triplets)}"
)

if len(triplets) != 300:
    raise RuntimeError(
        "Expected exactly 300 matched locations."
    )

# ------------------------------------------------------------
# COMPUTE DECOMPOSED METRICS
# ------------------------------------------------------------

rows = []

for t in triplets:

    Z = np.stack(
        [
            t["z_grundium"],
            t["z_hamamatsu"],
            t["z_leica"],
        ],
        axis=0
    )

    # --------------------------------------------------------
    # Individual scanner magnitudes
    # --------------------------------------------------------

    norms = np.linalg.norm(
        Z,
        axis=1
    )

    mean_norm = float(
        np.mean(norms)
    )

    std_norm = float(
        np.std(norms)
    )

    cv_norm = float(
        std_norm / mean_norm
    ) if mean_norm > 0 else np.nan

    norm_range = float(
        np.max(norms)
        - np.min(norms)
    )

    # --------------------------------------------------------
    # Mean representation
    # --------------------------------------------------------

    z_bar = np.mean(
        Z,
        axis=0
    )

    mean_vector_norm = float(
        np.linalg.norm(z_bar)
    )

    # --------------------------------------------------------
    # Raw scanner variability
    # --------------------------------------------------------

    deviations = (
        Z - z_bar
    )

    raw_variance = float(
        np.mean(
            np.sum(
                deviations ** 2,
                axis=1
            )
        )
    )

    # --------------------------------------------------------
    # Normalize each scanner representation
    # --------------------------------------------------------

    Z_unit = (
        Z
        /
        np.maximum(
            norms[:, None],
            1e-12
        )
    )

    unit_bar = np.mean(
        Z_unit,
        axis=0
    )

    normalized_deviations = (
        Z_unit - unit_bar
    )

    normalized_variance = float(
        np.mean(
            np.sum(
                normalized_deviations ** 2,
                axis=1
            )
        )
    )

    # --------------------------------------------------------
    # Pairwise cosine disagreement
    # --------------------------------------------------------

    cosines = []

    for i in range(3):

        for j in range(i + 1, 3):

            cos = float(
                np.dot(
                    Z_unit[i],
                    Z_unit[j]
                )
            )

            cosines.append(cos)

    mean_cosine = float(
        np.mean(cosines)
    )

    cosine_disagreement = (
        1.0 - mean_cosine
    )

    # --------------------------------------------------------
    # Pairwise Euclidean distances
    # --------------------------------------------------------

    pairwise_distances = []

    for i in range(3):

        for j in range(i + 1, 3):

            pairwise_distances.append(
                float(
                    np.linalg.norm(
                        Z[i] - Z[j]
                    )
                )
            )

    mean_pairwise_distance = float(
        np.mean(
            pairwise_distances
        )
    )

    # --------------------------------------------------------
    # Magnitude vs directional decomposition
    #
    # Total pairwise squared distance:
    #
    # ||z_i-z_j||²
    #
    # contains both norm and angular terms.
    #
    # We explicitly record both components.
    # --------------------------------------------------------

    magnitude_component = float(
        np.mean(
            [
                (
                    norms[i]
                    - norms[j]
                ) ** 2
                for i in range(3)
                for j in range(i + 1, 3)
            ]
        )
    )

    directional_component = float(
        np.mean(
            [
                2.0
                * norms[i]
                * norms[j]
                * (
                    1.0
                    - cosines[
                        0
                    ]
                )
                for i, j in []
            ]
        )
    )

    # More explicit pairwise angular component
    angular_components = []

    pair_index = 0

    for i in range(3):

        for j in range(i + 1, 3):

            angular_components.append(
                2.0
                * norms[i]
                * norms[j]
                * (
                    1.0
                    - float(
                        np.dot(
                            Z_unit[i],
                            Z_unit[j]
                        )
                    )
                )
            )

            pair_index += 1

    directional_component = float(
        np.mean(
            angular_components
        )
    )

    rows.append({
        "case_id": t["case_id"],
        "patch_id": t["patch_id"],

        "mean_embedding_norm":
            mean_norm,

        "embedding_norm_std":
            std_norm,

        "embedding_norm_cv":
            cv_norm,

        "embedding_norm_range":
            norm_range,

        "mean_vector_norm":
            mean_vector_norm,

        "raw_scanner_variance":
            raw_variance,

        "normalized_scanner_variance":
            normalized_variance,

        "mean_cosine_similarity":
            mean_cosine,

        "cosine_disagreement":
            cosine_disagreement,

        "mean_pairwise_distance":
            mean_pairwise_distance,

        "magnitude_component":
            magnitude_component,

        "directional_component":
            directional_component,
    })

patch_df = pd.DataFrame(rows)

# ------------------------------------------------------------
# CHECK DECOMPOSITION
# ------------------------------------------------------------

print("\nChecking decomposition...")

# For each pair:
#
# ||a-b||² =
# (||a||-||b||)²
# + 2||a||||b||(1-cos(theta))
#
# Therefore the average pairwise squared distance
# should equal the sum of the two components.

pairwise_squared = []

for _, r in patch_df.iterrows():

    # Reconstruct from component relationship.
    pairwise_squared.append(
        r["magnitude_component"]
        + r["directional_component"]
    )

patch_df[
    "decomposed_pairwise_squared"
] = pairwise_squared

# Mean pairwise squared distance directly
direct_pairwise_squared = []

for t in triplets:

    Z = np.stack(
        [
            t["z_grundium"],
            t["z_hamamatsu"],
            t["z_leica"],
        ]
    )

    vals = []

    for i in range(3):

        for j in range(i + 1, 3):

            vals.append(
                np.sum(
                    (
                        Z[i]
                        - Z[j]
                    ) ** 2
                )
            )

    direct_pairwise_squared.append(
        np.mean(vals)
    )

patch_df[
    "direct_pairwise_squared"
] = direct_pairwise_squared

decomp_error = np.abs(
    patch_df[
        "direct_pairwise_squared"
    ]
    -
    patch_df[
        "decomposed_pairwise_squared"
    ]
)

print(
    "Maximum decomposition error:",
    float(decomp_error.max())
)

if decomp_error.max() > 1e-3:
    print(
        "WARNING: decomposition "
        "numerical discrepancy detected."
    )
else:
    print(
        "Decomposition identity: PASS"
    )

# ------------------------------------------------------------
# LOAD LABELS
# ------------------------------------------------------------

labels = pd.read_csv(
    LABEL_FILE,
    sep="\t",
    dtype=str
)

labels = labels.apply(
    lambda c:
    c.str.strip()
    if c.dtype == "object"
    else c
)

labels["case_id"] = labels[
    "slide_id"
].map(clean_id)

isup_cols = [
    "isup_1",
    "isup_r2",
    "isup_r3",
]

for c in isup_cols:

    labels[c] = pd.to_numeric(
        labels[c],
        errors="coerce"
    )

labels["mean_isup"] = labels[
    isup_cols
].mean(
    axis=1,
    skipna=True
)

labels["reader_ambiguity"] = (
    labels[isup_cols].max(
        axis=1
    )
    -
    labels[isup_cols].min(
        axis=1
    )
)

label_df = labels[
    [
        "case_id",
        "mean_isup",
        "reader_ambiguity",
    ]
].copy()

# ------------------------------------------------------------
# CASE-LEVEL AGGREGATION
# ------------------------------------------------------------

patch_case = (
    patch_df
    .groupby("case_id")
    .agg(
        patch_count=(
            "patch_id",
            "count"
        ),

        mean_embedding_norm=(
            "mean_embedding_norm",
            "mean"
        ),

        embedding_norm_std=(
            "embedding_norm_std",
            "mean"
        ),

        embedding_norm_cv=(
            "embedding_norm_cv",
            "mean"
        ),

        embedding_norm_range=(
            "embedding_norm_range",
            "mean"
        ),

        mean_vector_norm=(
            "mean_vector_norm",
            "mean"
        ),

        raw_scanner_variance=(
            "raw_scanner_variance",
            "mean"
        ),

        normalized_scanner_variance=(
            "normalized_scanner_variance",
            "mean"
        ),

        cosine_disagreement=(
            "cosine_disagreement",
            "mean"
        ),

        mean_pairwise_distance=(
            "mean_pairwise_distance",
            "mean"
        ),

        magnitude_component=(
            "magnitude_component",
            "mean"
        ),

        directional_component=(
            "directional_component",
            "mean"
        ),
    )
    .reset_index()
)

case_df = patch_case.merge(
    label_df,
    on="case_id",
    how="inner"
)

print(
    "\nCase-level dataset:"
)

print(
    f"Cases: {len(case_df)}"
)

if len(case_df) != 30:
    raise RuntimeError(
        "Expected 30 cases."
    )

# ------------------------------------------------------------
# SAVE PATCH-LEVEL + CASE-LEVEL
# ------------------------------------------------------------

patch_file = (
    OUT
    / "decomposition_patch_level_300.csv"
)

case_file = (
    OUT
    / "decomposition_case_level_30.csv"
)

patch_df.to_csv(
    patch_file,
    index=False
)

case_df.to_csv(
    case_file,
    index=False
)

# ------------------------------------------------------------
# CORRELATION MATRIX
# ------------------------------------------------------------

analysis_vars = [
    "mean_isup",
    "reader_ambiguity",
    "mean_embedding_norm",
    "embedding_norm_range",
    "raw_scanner_variance",
    "normalized_scanner_variance",
    "cosine_disagreement",
    "magnitude_component",
    "directional_component",
]

correlation_rows = []

for i in range(len(analysis_vars)):

    for j in range(
        i + 1,
        len(analysis_vars)
    ):

        x = analysis_vars[i]
        y = analysis_vars[j]

        sr, sp, pr, pp = corr(
            case_df[x],
            case_df[y]
        )

        correlation_rows.append({
            "x": x,
            "y": y,
            "spearman_rho": sr,
            "spearman_p": sp,
            "pearson_r": pr,
            "pearson_p": pp,
        })

corr_df = pd.DataFrame(
    correlation_rows
)

corr_file = (
    OUT
    / "decomposition_correlations.csv"
)

corr_df.to_csv(
    corr_file,
    index=False
)

# ------------------------------------------------------------
# PRINT KEY CORRELATIONS
# ------------------------------------------------------------

key_pairs = [

    (
        "reader_ambiguity",
        "mean_embedding_norm"
    ),

    (
        "reader_ambiguity",
        "magnitude_component"
    ),

    (
        "reader_ambiguity",
        "directional_component"
    ),

    (
        "reader_ambiguity",
        "normalized_scanner_variance"
    ),

    (
        "mean_isup",
        "mean_embedding_norm"
    ),

    (
        "mean_isup",
        "magnitude_component"
    ),

    (
        "mean_isup",
        "directional_component"
    ),

    (
        "mean_isup",
        "normalized_scanner_variance"
    ),

    (
        "reader_ambiguity",
        "raw_scanner_variance"
    ),
]

print("\n" + "=" * 70)
print("KEY CORRELATIONS")
print("=" * 70)

for x, y in key_pairs:

    sr, sp, pr, pp = corr(
        case_df[x],
        case_df[y]
    )

    print(
        f"\n{x}  <->  {y}"
    )

    print(
        f"  Spearman rho = {sr:.6f}, "
        f"p = {sp:.6f}"
    )

    print(
        f"  Pearson r    = {pr:.6f}, "
        f"p = {pp:.6f}"
    )

# ------------------------------------------------------------
# COMPONENT RELATIVE CONTRIBUTION
# ------------------------------------------------------------

case_df[
    "directional_fraction"
] = (
    case_df["directional_component"]
    /
    (
        case_df["magnitude_component"]
        +
        case_df["directional_component"]
    )
)

case_df[
    "magnitude_fraction"
] = (
    case_df["magnitude_component"]
    /
    (
        case_df["magnitude_component"]
        +
        case_df["directional_component"]
    )
)

print("\n" + "=" * 70)
print("DECOMPOSITION FRACTIONS")
print("=" * 70)

print(
    "\nMean magnitude fraction:",
    case_df[
        "magnitude_fraction"
    ].mean()
)

print(
    "Mean directional fraction:",
    case_df[
        "directional_fraction"
    ].mean()
)

# ------------------------------------------------------------
# REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REGRESSION AUDIT")
print("=" * 70)

regression_results = []

# Directional component explained by grade + ambiguity
r1 = standardized_ols(
    case_df,
    "directional_component",
    [
        "mean_isup",
        "reader_ambiguity",
    ]
)

if r1:

    r1["model"] = (
        "directional_component ~ "
        "mean_isup + reader_ambiguity"
    )

    regression_results.append(r1)

    print(
        "\nDirectional component:"
    )

    print(
        f"  R² = {r1['r_squared']:.6f}"
    )

    print(
        f"  Mean ISUP beta = "
        f"{r1['mean_isup_beta']:.6f}, "
        f"p = "
        f"{r1['mean_isup_p']:.6f}"
    )

    print(
        f"  Reader ambiguity beta = "
        f"{r1['reader_ambiguity_beta']:.6f}, "
        f"p = "
        f"{r1['reader_ambiguity_p']:.6f}"
    )

# Magnitude component explained by grade + ambiguity
r2 = standardized_ols(
    case_df,
    "magnitude_component",
    [
        "mean_isup",
        "reader_ambiguity",
    ]
)

if r2:

    r2["model"] = (
        "magnitude_component ~ "
        "mean_isup + reader_ambiguity"
    )

    regression_results.append(r2)

    print(
        "\nMagnitude component:"
    )

    print(
        f"  R² = {r2['r_squared']:.6f}"
    )

    print(
        f"  Mean ISUP beta = "
        f"{r2['mean_isup_beta']:.6f}, "
        f"p = "
        f"{r2['mean_isup_p']:.6f}"
    )

    print(
        f"  Reader ambiguity beta = "
        f"{r2['reader_ambiguity_beta']:.6f}, "
        f"p = "
        f"{r2['reader_ambiguity_p']:.6f}"
    )

# Directional component vs grade + ambiguity + magnitude
r3 = standardized_ols(
    case_df,
    "directional_component",
    [
        "mean_isup",
        "reader_ambiguity",
        "mean_embedding_norm",
    ]
)

if r3:

    r3["model"] = (
        "directional_component ~ "
        "mean_isup + reader_ambiguity "
        "+ mean_embedding_norm"
    )

    regression_results.append(r3)

    print(
        "\nDirectional component with "
        "embedding magnitude:"
    )

    print(
        f"  R² = {r3['r_squared']:.6f}"
    )

    print(
        f"  Mean ISUP beta = "
        f"{r3['mean_isup_beta']:.6f}, "
        f"p = "
        f"{r3['mean_isup_p']:.6f}"
    )

    print(
        f"  Reader ambiguity beta = "
        f"{r3['reader_ambiguity_beta']:.6f}, "
        f"p = "
        f"{r3['reader_ambiguity_p']:.6f}"
    )

    print(
        f"  Embedding norm beta = "
        f"{r3['mean_embedding_norm_beta']:.6f}, "
        f"p = "
        f"{r3['mean_embedding_norm_p']:.6f}"
    )

reg_df = pd.DataFrame(
    regression_results
)

reg_file = (
    OUT
    / "decomposition_regressions.csv"
)

reg_df.to_csv(
    reg_file,
    index=False
)

# ------------------------------------------------------------
# GROUP SUMMARY
# ------------------------------------------------------------

case_df["ambiguity_group"] = pd.cut(
    case_df["reader_ambiguity"],
    bins=[
        -np.inf,
        0,
        1,
        np.inf
    ],
    labels=[
        "LOW",
        "MODERATE",
        "HIGH"
    ]
)

group_summary = (
    case_df
    .groupby(
        "ambiguity_group",
        observed=True
    )
    [
        [
            "mean_embedding_norm",
            "magnitude_component",
            "directional_component",
            "normalized_scanner_variance",
            "cosine_disagreement",
        ]
    ]
    .agg(
        [
            "mean",
            "median",
            "std",
            "count",
        ]
    )
)

group_file = (
    OUT
    / "decomposition_group_summary.csv"
)

group_summary.to_csv(
    group_file
)

print("\n" + "=" * 70)
print("AMBIGUITY GROUP SUMMARY")
print("=" * 70)

print(
    group_summary.to_string()
)

# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

def scatter_plot(
    x,
    y,
    xlabel,
    ylabel,
    title,
    filename
):

    xval = pd.to_numeric(
        case_df[x],
        errors="coerce"
    )

    yval = pd.to_numeric(
        case_df[y],
        errors="coerce"
    )

    mask = (
        np.isfinite(xval)
        &
        np.isfinite(yval)
    )

    sr, sp, _, _ = corr(
        xval[mask],
        yval[mask]
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        xval[mask],
        yval[mask],
        s=55
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        f"{title}\n"
        f"Spearman ρ={sr:.3f}, "
        f"p={sp:.4f}"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=300
    )

    plt.close()


scatter_plot(
    "reader_ambiguity",
    "magnitude_component",
    "Reader-level diagnostic ambiguity",
    "Magnitude component",
    "Reader ambiguity vs magnitude component",
    "ambiguity_vs_magnitude_component.png"
)

scatter_plot(
    "reader_ambiguity",
    "directional_component",
    "Reader-level diagnostic ambiguity",
    "Directional scanner component",
    "Reader ambiguity vs directional component",
    "ambiguity_vs_directional_component.png"
)

scatter_plot(
    "mean_isup",
    "magnitude_component",
    "Mean reader ISUP",
    "Magnitude component",
    "Grade vs magnitude component",
    "grade_vs_magnitude_component.png"
)

scatter_plot(
    "mean_isup",
    "directional_component",
    "Mean reader ISUP",
    "Directional scanner component",
    "Grade vs directional component",
    "grade_vs_directional_component.png"
)

scatter_plot(
    "magnitude_component",
    "directional_component",
    "Magnitude component",
    "Directional scanner component",
    "Magnitude vs directional scanner components",
    "magnitude_vs_directional_component.png"
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary_file = (
    OUT
    / "representation_decomposition_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ISBI 2027 — Representation Decomposition Audit\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Cohort: 30 cases\n"
    )

    f.write(
        "Paired locations: 300\n"
    )

    f.write(
        "Embeddings: 900\n"
    )

    f.write(
        "Embedding dimension: 1536\n\n"
    )

    f.write(
        "The scanner-induced representation difference was "
        "decomposed into:\n"
    )

    f.write(
        "1. Magnitude component\n"
    )

    f.write(
        "2. Directional component\n\n"
    )

    f.write(
        "Reader ambiguity = max(ISUP)-min(ISUP)\n"
    )

    f.write(
        "Mean ISUP = mean available reader ISUP\n\n"
    )

    f.write(
        "KEY CORRELATIONS\n"
        "----------------\n"
    )

    for x, y in key_pairs:

        sr, sp, pr, pp = corr(
            case_df[x],
            case_df[y]
        )

        f.write(
            f"{x} vs {y}: "
            f"Spearman rho={sr:.6f}, "
            f"p={sp:.6f}; "
            f"Pearson r={pr:.6f}, "
            f"p={pp:.6f}\n"
        )

    f.write(
        "\nDECOMPOSITION\n"
        "-------------\n"
    )

    f.write(
        f"Mean magnitude fraction = "
        f"{case_df['magnitude_fraction'].mean():.6f}\n"
    )

    f.write(
        f"Mean directional fraction = "
        f"{case_df['directional_fraction'].mean():.6f}\n"
    )

    f.write(
        f"Maximum decomposition error = "
        f"{float(decomp_error.max()):.10f}\n"
    )

    f.write(
        "\nIMPORTANT:\n"
        "This analysis is exploratory and does not establish "
        "causality or biological mechanisms.\n"
    )

print("\n" + "=" * 70)
print("REPRESENTATION DECOMPOSITION AUDIT COMPLETE")
print("=" * 70)

print("\nOutputs:")

for p in sorted(OUT.iterdir()):
    print(" ", p)

print(
    "\nSend the complete terminal output."
)

print(
    "Do not train a model or modify the dataset yet."
)