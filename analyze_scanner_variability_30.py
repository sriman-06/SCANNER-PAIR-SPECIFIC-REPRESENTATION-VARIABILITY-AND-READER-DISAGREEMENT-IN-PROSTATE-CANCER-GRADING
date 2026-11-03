# ================================================================
# ISBI 2027
# Scanner-Induced Representation Variability Analysis
#
# Cohort:
#   30 cases
#   10 matched locations / case
#   3 scanners / location
#   900 UNI2-h embeddings
#
# Primary scientific question:
#   How does scanner-induced representation variability relate
#   to reader-level diagnostic ambiguity?
#
# IMPORTANT:
#   - No Domain Adaptation
#   - No model training
#   - No patch resampling
#   - No cohort modification
#   - Case-level inference is primary
# ================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal


# ================================================================
# 1. PATHS
# ================================================================

ROOT = Path(r"D:\ISBI_2027")

EMB_DIR = ROOT / "outputs" / "uni2h_30"
MANIFEST_PATH = EMB_DIR / "embedding_manifest_30.csv"

LABEL_PATH = ROOT / "annotations" / "slide_labels.tsv"

OUT_DIR = ROOT / "outputs" / "scanner_variability_30"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_OUT = OUT_DIR / "scanner_variability_300.csv"
CASE_OUT = OUT_DIR / "case_level_scanner_variability_30.csv"
PAIR_OUT = OUT_DIR / "scanner_pair_statistics.csv"
SUMMARY_OUT = OUT_DIR / "scanner_variability_summary.txt"

SCATTER_OUT = OUT_DIR / "case_scatter_reader_ambiguity.png"
GROUP_OUT = OUT_DIR / "ambiguity_group_comparison.png"
PAIR_FIG_OUT = OUT_DIR / "scanner_pair_comparison.png"


# ================================================================
# 2. SETTINGS
# ================================================================

EXPECTED_CASES = 30
EXPECTED_LOCATIONS = 300
EXPECTED_IMAGES = 900
EXPECTED_DIM = 1536

SCANNERS = ["grundium", "hamamatsu", "leica"]


# ================================================================
# 3. HELPERS
# ================================================================

def clean_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_embedding(path):
    """
    Load UNI2-h embedding robustly.

    Expected:
        torch.Tensor of shape [1536]
    """

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.Tensor):
        z = obj
    elif isinstance(obj, dict):
        # Try common possibilities
        possible_keys = [
            "embedding",
            "features",
            "feature",
            "z",
            "emb",
        ]

        z = None

        for key in possible_keys:
            if key in obj:
                z = obj[key]
                break

        if z is None:
            raise ValueError(
                f"Could not find embedding tensor in dictionary: {path}"
            )
    else:
        raise TypeError(
            f"Unsupported embedding object type {type(obj)}: {path}"
        )

    z = torch.as_tensor(z).detach().cpu().float().numpy()
    z = np.asarray(z).reshape(-1)

    return z


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    return float(np.dot(a, b) / (na * nb))


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def interpret_p(p):
    if not np.isfinite(p):
        return "not available"
    if p < 0.001:
        return "highly significant"
    if p < 0.01:
        return "very significant"
    if p < 0.05:
        return "significant"
    return "not statistically significant"


# ================================================================
# 4. LOAD EMBEDDING MANIFEST
# ================================================================

print("=" * 80)
print("ISBI 2027 — FULL SCANNER VARIABILITY ANALYSIS")
print("=" * 80)

print("\nLoading embedding manifest:")
print(MANIFEST_PATH)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Embedding manifest not found:\n{MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

print(f"Manifest rows: {len(manifest)}")
print("Columns:")
print(list(manifest.columns))


# ================================================================
# 5. NORMALIZE MANIFEST COLUMN NAMES
# ================================================================

manifest.columns = [
    str(c).strip().lower()
    for c in manifest.columns
]

print("\nNormalized columns:")
print(list(manifest.columns))


# ================================================================
# 6. FIND REQUIRED COLUMNS
# ================================================================

def find_column(df, candidates, required=True):

    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Tried: {candidates}\n"
            f"Available: {list(df.columns)}"
        )

    return None


case_col = find_column(
    manifest,
    ["case", "case_id", "slide_id", "id"]
)

scanner_col = find_column(
    manifest,
    ["scanner", "scanner_name"]
)

patch_col = find_column(
    manifest,
    ["patch_id", "patch", "location_id"]
)

embedding_col = find_column(
    manifest,
    [
        "embedding_path",
        "embedding_file",
        "embedding",
        "output_path",
        "path",
    ]
)


manifest["case"] = manifest[case_col].astype(str).str.strip()
manifest["scanner"] = (
    manifest[scanner_col]
    .astype(str)
    .str.strip()
    .str.lower()
)
manifest["patch_id"] = manifest[patch_col].astype(str).str.strip()
manifest["embedding_path"] = (
    manifest[embedding_col].astype(str).str.strip()
)


# ================================================================
# 7. BASIC MANIFEST QC
# ================================================================

print("\n" + "=" * 80)
print("MANIFEST QC")
print("=" * 80)

print("Rows:", len(manifest))
print("Cases:", manifest["case"].nunique())
print("Scanners:", sorted(manifest["scanner"].unique()))
print("Matched locations:",
      manifest.groupby(["case", "patch_id"]).ngroups)

if len(manifest) != EXPECTED_IMAGES:
    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} manifest rows, "
        f"found {len(manifest)}."
    )

if manifest["case"].nunique() != EXPECTED_CASES:
    raise RuntimeError(
        f"Expected {EXPECTED_CASES} cases, "
        f"found {manifest['case'].nunique()}."
    )

if manifest.groupby(["case", "patch_id"]).ngroups != EXPECTED_LOCATIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_LOCATIONS} matched locations."
    )


# ================================================================
# 8. RESOLVE EMBEDDING PATHS
# ================================================================

def resolve_embedding_path(p):

    p = Path(p)

    if p.is_absolute() and p.exists():
        return p

    # Relative to embedding directory
    candidate = EMB_DIR / p

    if candidate.exists():
        return candidate

    # Relative to project root
    candidate = ROOT / p

    if candidate.exists():
        return candidate

    # Filename search as final fallback
    matches = list(EMB_DIR.rglob(p.name))

    if len(matches) == 1:
        return matches[0]

    return p


manifest["resolved_embedding_path"] = (
    manifest["embedding_path"]
    .apply(resolve_embedding_path)
)


missing = []

for p in manifest["resolved_embedding_path"]:
    if not Path(p).exists():
        missing.append(str(p))

if missing:
    print("\nMissing embedding files:")
    for p in missing[:20]:
        print(p)

    raise FileNotFoundError(
        f"{len(missing)} embedding files are missing."
    )

print("Embedding files: 900/900 found")


# ================================================================
# 9. LOAD READER LABELS
# ================================================================

print("\n" + "=" * 80)
print("LOADING READER LABELS")
print("=" * 80)

if not LABEL_PATH.exists():
    raise FileNotFoundError(
        f"Reader label file not found:\n{LABEL_PATH}"
    )

labels = pd.read_csv(
    LABEL_PATH,
    sep="\t",
    dtype=str
)

labels = labels.apply(
    lambda col: (
        col.str.strip()
        if col.dtype == "object"
        else col
    )
)

print("Label rows:", len(labels))
print("Label columns:")
print(list(labels.columns))


# ================================================================
# 10. NORMALIZE LABEL ID
# ================================================================

label_id_col = "slide_id"

if label_id_col not in labels.columns:

    candidates = [
        "case",
        "case_id",
        "id",
    ]

    found = None

    for c in candidates:
        if c in labels.columns:
            found = c
            break

    if found is None:
        raise ValueError(
            "Could not identify slide ID column."
        )

    label_id_col = found

labels["case"] = (
    labels[label_id_col]
    .astype(str)
    .str.strip()
)


# ================================================================
# 11. FIND ISUP COLUMNS
# ================================================================

isup_candidates = [
    "isup_1",
    "isup_r1",
    "isup_reader1",
    "isup_reader_1",
]

isup_cols = [
    c for c in isup_candidates
    if c in labels.columns
]

# Explicitly support the dataset's known naming pattern.
known_candidates = [
    "isup_1",
    "isup_r2",
    "isup_r3",
]

isup_cols = [
    c for c in known_candidates
    if c in labels.columns
]

if not isup_cols:
    raise ValueError(
        "No ISUP reader columns found."
    )

print("Reader ISUP columns:", isup_cols)


# ================================================================
# 12. COMPUTE READER-LEVEL DIAGNOSTIC AMBIGUITY
# ================================================================

def compute_reader_metrics(row):

    vals = []

    for c in isup_cols:

        value = row[c]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value == "":
            continue

        try:
            vals.append(float(value))
        except ValueError:
            pass

    if len(vals) == 0:
        return pd.Series({
            "reader_n": 0,
            "reader_isup_min": np.nan,
            "reader_isup_max": np.nan,
            "reader_isup_mean": np.nan,
            "reader_ambiguity": np.nan,
        })

    return pd.Series({
        "reader_n": len(vals),
        "reader_isup_min": min(vals),
        "reader_isup_max": max(vals),
        "reader_isup_mean": np.mean(vals),
        "reader_ambiguity": max(vals) - min(vals),
    })


reader_metrics = labels.apply(
    compute_reader_metrics,
    axis=1
)

labels = pd.concat(
    [labels, reader_metrics],
    axis=1
)


# ================================================================
# 13. AMBIGUITY GROUPS
# ================================================================

def ambiguity_group(x):

    if not np.isfinite(x):
        return "UNKNOWN"

    if x == 0:
        return "LOW"

    if x == 1:
        return "MODERATE"

    return "HIGH"


labels["ambiguity_group"] = (
    labels["reader_ambiguity"]
    .apply(ambiguity_group)
)


print("\nReader ambiguity distribution:")

print(
    labels[
        [
            "case",
            "reader_n",
            "reader_isup_min",
            "reader_isup_max",
            "reader_ambiguity",
            "ambiguity_group",
        ]
    ]
    .to_string(index=False)
)


# ================================================================
# 14. CREATE LABEL LOOKUP
# ================================================================

label_lookup = labels.set_index("case")


# ================================================================
# 15. LOAD ALL EMBEDDINGS
# ================================================================

print("\n" + "=" * 80)
print("LOADING 900 UNI2-h EMBEDDINGS")
print("=" * 80)

embedding_cache = {}

for idx, row in manifest.iterrows():

    path = Path(row["resolved_embedding_path"])

    key = str(path)

    if key not in embedding_cache:

        z = load_embedding(path)

        if z.shape[0] != EXPECTED_DIM:
            raise RuntimeError(
                f"Unexpected embedding dimension:\n"
                f"{path}\n"
                f"Expected: {EXPECTED_DIM}\n"
                f"Found: {z.shape[0]}"
            )

        if not np.all(np.isfinite(z)):
            raise RuntimeError(
                f"NaN/Inf detected in embedding:\n{path}"
            )

        embedding_cache[key] = z

    if (idx + 1) % 100 == 0:
        print(
            f"Loaded {idx + 1}/{len(manifest)}"
        )

print("All embeddings loaded successfully.")


# ================================================================
# 16. BUILD MATCHED TRIPLETS
# ================================================================

print("\n" + "=" * 80)
print("MATCHED-TRIPLET ANALYSIS")
print("=" * 80)

triplet_records = []

grouped = manifest.groupby(
    ["case", "patch_id"],
    sort=True
)

for (case, patch_id), group in grouped:

    scanner_map = {}

    for _, row in group.iterrows():

        scanner = row["scanner"]

        if scanner in scanner_map:
            raise RuntimeError(
                f"Duplicate scanner for {case}, {patch_id}: "
                f"{scanner}"
            )

        path = str(row["resolved_embedding_path"])

        scanner_map[scanner] = embedding_cache[path]

    missing_scanners = [
        s for s in SCANNERS
        if s not in scanner_map
    ]

    if missing_scanners:

        raise RuntimeError(
            f"Missing scanner(s) for "
            f"{case}, patch {patch_id}: "
            f"{missing_scanners}"
        )

    z_g = scanner_map["grundium"]
    z_h = scanner_map["hamamatsu"]
    z_l = scanner_map["leica"]

    # ------------------------------------------------------------
    # Mean representation
    # ------------------------------------------------------------

    z_bar = (
        z_g +
        z_h +
        z_l
    ) / 3.0

    # ------------------------------------------------------------
    # Raw scanner-induced representation variance
    # ------------------------------------------------------------

    d_g = np.linalg.norm(z_g - z_bar) ** 2
    d_h = np.linalg.norm(z_h - z_bar) ** 2
    d_l = np.linalg.norm(z_l - z_bar) ** 2

    v_scanner = (
        d_g +
        d_h +
        d_l
    ) / 3.0

    # ------------------------------------------------------------
    # Mean norm normalization
    #
    # This is a sensitivity metric, not the primary metric.
    # ------------------------------------------------------------

    mean_norm_sq = np.linalg.norm(z_bar) ** 2

    if mean_norm_sq > 0:

        normalized_variance = (
            v_scanner /
            mean_norm_sq
        )

    else:
        normalized_variance = np.nan

    # ------------------------------------------------------------
    # Pairwise Euclidean distances
    # ------------------------------------------------------------

    d_gh = np.linalg.norm(z_g - z_h)
    d_gl = np.linalg.norm(z_g - z_l)
    d_hl = np.linalg.norm(z_h - z_l)

    mean_pairwise_distance = np.mean(
        [d_gh, d_gl, d_hl]
    )

    # ------------------------------------------------------------
    # Pairwise cosine similarities
    # ------------------------------------------------------------

    c_gh = cosine_similarity(z_g, z_h)
    c_gl = cosine_similarity(z_g, z_l)
    c_hl = cosine_similarity(z_h, z_l)

    mean_cosine = np.nanmean(
        [c_gh, c_gl, c_hl]
    )

    cosine_disagreement = (
        1.0 - mean_cosine
    )

    # ------------------------------------------------------------
    # Unit-vector Euclidean distances
    # ------------------------------------------------------------

    def normalize(z):

        norm = np.linalg.norm(z)

        if norm == 0:
            return np.full_like(z, np.nan)

        return z / norm

    u_g = normalize(z_g)
    u_h = normalize(z_h)
    u_l = normalize(z_l)

    unit_d_gh = np.linalg.norm(u_g - u_h)
    unit_d_gl = np.linalg.norm(u_g - u_l)
    unit_d_hl = np.linalg.norm(u_h - u_l)

    mean_unit_distance = np.mean(
        [
            unit_d_gh,
            unit_d_gl,
            unit_d_hl,
        ]
    )

    # ------------------------------------------------------------
    # Save patch-level record
    # ------------------------------------------------------------

    triplet_records.append({

        "case": case,
        "patch_id": patch_id,

        "v_scanner": v_scanner,
        "normalized_variance": normalized_variance,

        "d_gh": d_gh,
        "d_gl": d_gl,
        "d_hl": d_hl,

        "mean_pairwise_distance":
            mean_pairwise_distance,

        "cosine_gh": c_gh,
        "cosine_gl": c_gl,
        "cosine_hl": c_hl,

        "mean_cosine":
            mean_cosine,

        "cosine_disagreement":
            cosine_disagreement,

        "unit_d_gh": unit_d_gh,
        "unit_d_gl": unit_d_gl,
        "unit_d_hl": unit_d_hl,

        "mean_unit_distance":
            mean_unit_distance,
    })


patch_df = pd.DataFrame(
    triplet_records
)

print(
    f"\nMatched triplets generated: "
    f"{len(patch_df)}"
)

if len(patch_df) != EXPECTED_LOCATIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_LOCATIONS} "
        f"matched locations."
    )


# ================================================================
# 17. MERGE READER AMBIGUITY
# ================================================================

merge_cols = [
    "reader_n",
    "reader_isup_min",
    "reader_isup_max",
    "reader_isup_mean",
    "reader_ambiguity",
    "ambiguity_group",
]

patch_df = patch_df.merge(
    labels[
        ["case"] + merge_cols
    ],
    on="case",
    how="left",
    validate="many_to_one",
)


# ================================================================
# 18. PATCH-LEVEL OUTPUT
# ================================================================

patch_df = patch_df.sort_values(
    ["case", "patch_id"]
).reset_index(drop=True)

patch_df.to_csv(
    PATCH_OUT,
    index=False
)

print("\nSaved:")
print(PATCH_OUT)


# ================================================================
# 19. CASE-LEVEL AGGREGATION
# ================================================================

print("\n" + "=" * 80)
print("CASE-LEVEL AGGREGATION")
print("=" * 80)

metric_cols = [
    "v_scanner",
    "normalized_variance",
    "mean_pairwise_distance",
    "cosine_disagreement",
    "mean_unit_distance",
    "d_gh",
    "d_gl",
    "d_hl",
    "cosine_gh",
    "cosine_gl",
    "cosine_hl",
]

case_df = (
    patch_df
    .groupby("case")
    .agg(
        patch_count=("patch_id", "count"),

        v_scanner_mean=(
            "v_scanner",
            "mean"
        ),

        v_scanner_median=(
            "v_scanner",
            "median"
        ),

        v_scanner_std=(
            "v_scanner",
            "std"
        ),

        normalized_variance_mean=(
            "normalized_variance",
            "mean"
        ),

        normalized_variance_median=(
            "normalized_variance",
            "median"
        ),

        mean_pairwise_distance_mean=(
            "mean_pairwise_distance",
            "mean"
        ),

        cosine_disagreement_mean=(
            "cosine_disagreement",
            "mean"
        ),

        mean_unit_distance_mean=(
            "mean_unit_distance",
            "mean"
        ),

        d_gh_mean=("d_gh", "mean"),
        d_gl_mean=("d_gl", "mean"),
        d_hl_mean=("d_hl", "mean"),

        cosine_gh_mean=("cosine_gh", "mean"),
        cosine_gl_mean=("cosine_gl", "mean"),
        cosine_hl_mean=("cosine_hl", "mean"),
    )
    .reset_index()
)


# Merge reader metrics at case level
case_df = case_df.merge(
    labels[
        [
            "case",
            "reader_n",
            "reader_isup_min",
            "reader_isup_max",
            "reader_isup_mean",
            "reader_ambiguity",
            "ambiguity_group",
        ]
    ],
    on="case",
    how="left",
    validate="one_to_one",
)


# ================================================================
# 20. CASE QC
# ================================================================

if len(case_df) != EXPECTED_CASES:
    raise RuntimeError(
        f"Expected {EXPECTED_CASES} cases "
        f"after aggregation, found {len(case_df)}."
    )

if not np.all(
    case_df["patch_count"].values == 10
):
    raise RuntimeError(
        "Not every case contains exactly "
        "10 matched locations."
    )

print(
    f"Cases with exactly 10 locations: "
    f"{len(case_df)}/{EXPECTED_CASES}"
)


# ================================================================
# 21. SAVE CASE-LEVEL DATA
# ================================================================

case_df.to_csv(
    CASE_OUT,
    index=False
)

print("\nSaved:")
print(CASE_OUT)


# ================================================================
# 22. PRIMARY STATISTICAL ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("PRIMARY CASE-LEVEL ANALYSIS")
print("=" * 80)

x = case_df["reader_ambiguity"].to_numpy(
    dtype=float
)

metrics_for_test = {
    "v_scanner_mean":
        case_df["v_scanner_mean"].to_numpy(
            dtype=float
        ),

    "normalized_variance_mean":
        case_df["normalized_variance_mean"].to_numpy(
            dtype=float
        ),

    "mean_pairwise_distance_mean":
        case_df["mean_pairwise_distance_mean"].to_numpy(
            dtype=float
        ),

    "cosine_disagreement_mean":
        case_df["cosine_disagreement_mean"].to_numpy(
            dtype=float
        ),

    "mean_unit_distance_mean":
        case_df["mean_unit_distance_mean"].to_numpy(
            dtype=float
        ),
}


correlation_results = []

for metric_name, y in metrics_for_test.items():

    valid = (
        np.isfinite(x) &
        np.isfinite(y)
    )

    if valid.sum() < 4:
        rho = np.nan
        p = np.nan
        direction = "not available"

    else:

        rho, p = spearmanr(
            x[valid],
            y[valid]
        )

        if rho > 0:
            direction = "positive"
        elif rho < 0:
            direction = "negative"
        else:
            direction = "zero"

    correlation_results.append({
        "metric": metric_name,
        "n_cases": int(valid.sum()),
        "spearman_rho": rho,
        "spearman_p": p,
        "direction": direction,
    })


correlation_df = pd.DataFrame(
    correlation_results
)

print(
    correlation_df.to_string(
        index=False
    )
)


# ================================================================
# 23. PEARSON SENSITIVITY ANALYSIS
# ================================================================

pearson_results = []

for metric_name, y in metrics_for_test.items():

    valid = (
        np.isfinite(x) &
        np.isfinite(y)
    )

    if valid.sum() >= 4:

        r, p = pearsonr(
            x[valid],
            y[valid]
        )

    else:

        r = np.nan
        p = np.nan

    pearson_results.append({
        "metric": metric_name,
        "n_cases": int(valid.sum()),
        "pearson_r": r,
        "pearson_p": p,
    })


pearson_df = pd.DataFrame(
    pearson_results
)


# ================================================================
# 24. GROUPED DESCRIPTIVE ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("AMBIGUITY GROUP ANALYSIS")
print("=" * 80)

group_order = [
    "LOW",
    "MODERATE",
    "HIGH",
]

group_summary = (
    case_df
    .groupby("ambiguity_group", sort=False)
    .agg(
        cases=("case", "count"),

        scanner_variability_mean=(
            "v_scanner_mean",
            "mean"
        ),

        scanner_variability_median=(
            "v_scanner_mean",
            "median"
        ),

        scanner_variability_std=(
            "v_scanner_mean",
            "std"
        ),

        normalized_variance_mean=(
            "normalized_variance_mean",
            "mean"
        ),

        cosine_disagreement_mean=(
            "cosine_disagreement_mean",
            "mean"
        ),
    )
    .reindex(group_order)
    .reset_index()
)

print(
    group_summary.to_string(
        index=False
    )
)


# ================================================================
# 25. NONPARAMETRIC GROUP TEST
# ================================================================

group_arrays = []

for group in group_order:

    vals = case_df.loc[
        case_df["ambiguity_group"] == group,
        "v_scanner_mean"
    ].dropna().to_numpy()

    if len(vals) > 0:
        group_arrays.append(vals)


kruskal_h = np.nan
kruskal_p = np.nan

if len(group_arrays) >= 2:

    try:
        kruskal_h, kruskal_p = kruskal(
            *group_arrays
        )
    except Exception as e:
        warnings.warn(
            f"Kruskal-Wallis failed: {e}"
        )


# ================================================================
# 26. PAIRWISE SCANNER STATISTICS
# ================================================================

print("\n" + "=" * 80)
print("SCANNER-PAIR ANALYSIS")
print("=" * 80)

pair_names = [
    ("grundium", "hamamatsu", "GH"),
    ("grundium", "leica", "GL"),
    ("hamamatsu", "leica", "HL"),
]

pair_records = []

for s1, s2, pair_name in pair_names:

    d_col = f"d_{pair_name.lower()}"
    c_col = f"cosine_{pair_name.lower()}"

    # Case-level values
    d = case_df[
        f"{d_col}_mean"
    ].to_numpy(dtype=float)

    c = case_df[
        f"{c_col}_mean"
    ].to_numpy(dtype=float)

    pair_records.append({
        "pair": pair_name,
        "scanner_1": s1,
        "scanner_2": s2,
        "mean_euclidean_distance":
            np.nanmean(d),
        "median_euclidean_distance":
            np.nanmedian(d),
        "std_euclidean_distance":
            np.nanstd(d, ddof=1),
        "mean_cosine_similarity":
            np.nanmean(c),
        "median_cosine_similarity":
            np.nanmedian(c),
    })


pair_df = pd.DataFrame(
    pair_records
)

pair_df.to_csv(
    PAIR_OUT,
    index=False
)

print(
    pair_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(PAIR_OUT)


# ================================================================
# 27. CASE-LEVEL SCATTER PLOT
# ================================================================

plt.figure(figsize=(8, 6))

for group in group_order:

    sub = case_df[
        case_df["ambiguity_group"] == group
    ]

    if len(sub) == 0:
        continue

    plt.scatter(
        sub["reader_ambiguity"],
        sub["v_scanner_mean"],
        label=group,
        s=70,
        alpha=0.85,
    )

# Primary correlation
valid = (
    np.isfinite(
        case_df["reader_ambiguity"]
    )
    &
    np.isfinite(
        case_df["v_scanner_mean"]
    )
)

if valid.sum() >= 4:

    rho, p = spearmanr(
        case_df.loc[
            valid,
            "reader_ambiguity"
        ],
        case_df.loc[
            valid,
            "v_scanner_mean"
        ],
    )

    text = (
        f"Spearman ρ = {rho:.3f}\n"
        f"p = {p:.4g}\n"
        f"n = {valid.sum()}"
    )

    plt.text(
        0.97,
        0.97,
        text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

plt.xlabel(
    "Reader-level diagnostic ambiguity "
    "(ISUP range)"
)

plt.ylabel(
    "Mean scanner-induced representation variability"
)

plt.title(
    "Reader-level diagnostic ambiguity vs.\n"
    "scanner-induced representation variability"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    SCATTER_OUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(SCATTER_OUT)


# ================================================================
# 28. GROUP COMPARISON PLOT
# ================================================================

plt.figure(figsize=(8, 6))

data = []

labels_plot = []

for group in group_order:

    vals = case_df.loc[
        case_df["ambiguity_group"] == group,
        "v_scanner_mean"
    ].dropna().to_numpy()

    if len(vals) > 0:

        data.append(vals)
        labels_plot.append(group)

if data:

    plt.boxplot(
        data,
        labels=labels_plot,
        showmeans=True,
    )

    # Overlay individual case points
    for i, vals in enumerate(data, start=1):

        jitter = (
            np.random.default_rng(42)
            .normal(
                i,
                0.04,
                size=len(vals)
            )
        )

        plt.scatter(
            jitter,
            vals,
            alpha=0.7,
            s=35,
        )

plt.xlabel(
    "Reader-level diagnostic ambiguity group"
)

plt.ylabel(
    "Mean scanner-induced representation variability"
)

plt.title(
    "Scanner-induced representation variability\n"
    "across reader-ambiguity groups"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    GROUP_OUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(GROUP_OUT)


# ================================================================
# 29. SCANNER-PAIR COMPARISON
# ================================================================

plt.figure(figsize=(8, 6))

pair_labels = pair_df["pair"].tolist()
pair_values = (
    pair_df[
        "mean_euclidean_distance"
    ].to_numpy()
)

xpos = np.arange(
    len(pair_labels)
)

plt.bar(
    xpos,
    pair_values
)

plt.xticks(
    xpos,
    pair_labels
)

plt.ylabel(
    "Mean pairwise embedding distance"
)

plt.xlabel(
    "Scanner pair"
)

plt.title(
    "Scanner-pair representation differences"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PAIR_FIG_OUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(PAIR_FIG_OUT)


# ================================================================
# 30. SCIENTIFIC INTERPRETATION
# ================================================================

primary_row = correlation_df[
    correlation_df["metric"] == "v_scanner_mean"
].iloc[0]

rho = primary_row["spearman_rho"]
p = primary_row["spearman_p"]

if np.isfinite(rho):

    if rho >= 0.30 and p < 0.05:
        interpretation = (
            "The 30-case analysis shows a positive and "
            "statistically significant association between "
            "reader-level diagnostic ambiguity and "
            "scanner-induced representation variability."
        )

    elif rho <= -0.30 and p < 0.05:
        interpretation = (
            "The 30-case analysis shows a negative and "
            "statistically significant association between "
            "reader-level diagnostic ambiguity and "
            "scanner-induced representation variability."
        )

    elif abs(rho) < 0.30:
        interpretation = (
            "The 30-case analysis does not show a strong "
            "monotonic association between reader-level "
            "diagnostic ambiguity and scanner-induced "
            "representation variability."
        )

    else:
        interpretation = (
            "The 30-case analysis shows an association whose "
            "strength does not provide strong evidence of a "
            "statistically significant monotonic relationship."
        )

else:

    interpretation = (
        "The primary association could not be estimated."
    )


# ================================================================
# 31. WRITE SUMMARY
# ================================================================

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ISBI 2027 — Scanner-Induced Representation "
        "Variability Analysis\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        "COHORT\n"
    )

    f.write(
        f"Cases: {len(case_df)}\n"
    )

    f.write(
        f"Matched locations: {len(patch_df)}\n"
    )

    f.write(
        "Scanners: Grundium, Hamamatsu, Leica\n"
    )

    f.write(
        "UNI2-h embedding dimension: 1536\n\n"
    )

    f.write(
        "PRIMARY DEFINITION\n"
    )

    f.write(
        "V_scanner = (1/3) * "
        "sum_s ||z_s - z_bar||^2\n\n"
    )

    f.write(
        "Reader-level diagnostic ambiguity:\n"
    )

    f.write(
        "A_reader = max(reader ISUP) - "
        "min(reader ISUP)\n\n"
    )

    f.write(
        "PRIMARY CASE-LEVEL RESULT\n"
    )

    f.write(
        f"Spearman rho: {rho:.6f}\n"
    )

    f.write(
        f"Spearman p-value: {p:.6g}\n"
    )

    f.write(
        f"Interpretation: {interpret_p(p)}\n\n"
    )

    f.write(
        "SCIENTIFIC INTERPRETATION\n"
    )

    f.write(
        interpretation + "\n\n"
    )

    f.write(
        "AMBIGUITY GROUP SUMMARY\n"
    )

    f.write(
        group_summary.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "KRUSKAL-WALLIS GROUP TEST\n"
    )

    f.write(
        f"H = {kruskal_h:.6f}\n"
    )

    f.write(
        f"p = {kruskal_p:.6g}\n\n"
    )

    f.write(
        "SPEARMAN SENSITIVITY RESULTS\n"
    )

    f.write(
        correlation_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "PEARSON SENSITIVITY RESULTS\n"
    )

    f.write(
        pearson_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "SCANNER PAIRS\n"
    )

    f.write(
        pair_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "IMPORTANT INTERPRETATION NOTE\n"
    )

    f.write(
        "Patch-level measurements are nested within slides. "
        "Therefore the primary association is evaluated "
        "at the case/slide level using the mean of the 10 "
        "matched locations per case. The 300 patch observations "
        "are not treated as 300 independent cases.\n"
    )

    f.write(
        "This analysis does not establish causal independence "
        "between scanner-induced representation variability "
        "and reader-level diagnostic ambiguity.\n"
    )

print("\nSaved:")
print(SUMMARY_OUT)


# ================================================================
# 32. FINAL CONSOLE REPORT
# ================================================================

print("\n")
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Cases:                 {len(case_df)}"
)

print(
    f"Matched locations:     {len(patch_df)}"
)

print(
    f"Images represented:    {EXPECTED_IMAGES}"
)

print(
    f"Embedding dimension:   {EXPECTED_DIM}"
)

print(
    f"Primary Spearman rho:  {rho:.6f}"
)

print(
    f"Primary Spearman p:    {p:.6g}"
)

print(
    f"Primary interpretation:"
)

print(
    interpretation
)

print("\nOutputs:")
print(PATCH_OUT)
print(CASE_OUT)
print(PAIR_OUT)
print(SUMMARY_OUT)
print(SCATTER_OUT)
print(GROUP_OUT)
print(PAIR_FIG_OUT)

print("\n" + "=" * 80)
print("DO NOT TRAIN A DECOMPOSITION MODEL YET.")
print("First inspect these full-cohort results.")
print("=" * 80)