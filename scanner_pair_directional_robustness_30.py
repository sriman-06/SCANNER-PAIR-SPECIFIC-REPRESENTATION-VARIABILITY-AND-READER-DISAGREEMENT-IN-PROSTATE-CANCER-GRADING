# ============================================================
# ISBI 2027
# Scanner-Pair + Within-Case Directional Robustness Audit
#
# Existing data ONLY:
#   900 UNI2-h embeddings
#   300 matched spatial locations
#   30 cases
#
# NO new extraction
# NO new patches
# NO model training
#
# Questions:
#   1. Is directional scanner disagreement consistent across
#      G-H, G-L and H-L?
#   2. Does each scanner pair relate to reader ambiguity?
#   3. Is the association driven by a few cases?
#   4. Is the association stable across the 10 locations/case?
#   5. Does leave-one-patch-out aggregation change conclusions?
# ============================================================

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr

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

MANIFEST = EMBED_ROOT / "embedding_manifest_30.csv"

LABEL_FILE = Path(
    os.environ.get(
        "ISBI_LABEL_PATH",
        PROJECT_ROOT / "annotations" / "slide_labels.tsv"
    )
).expanduser().resolve()

OUT = Path(
    os.environ.get(
        "ISBI_PAIR_ROBUSTNESS_OUTPUT_ROOT",
        PROJECT_ROOT / "outputs" / "scanner_pair_directional_robustness_30"
    )
).expanduser().resolve()

SCANNERS = [
    "grundium",
    "hamamatsu",
    "leica",
]

PAIRS = [
    ("grundium", "hamamatsu", "GH"),
    ("grundium", "leica", "GL"),
    ("hamamatsu", "leica", "HL"),
]

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def clean_id(x):
    return str(x).strip().upper()


def correlation(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if mask.sum() < 4:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            int(mask.sum()),
        )

    sr, sp = spearmanr(
        x[mask],
        y[mask]
    )

    pr, pp = pearsonr(
        x[mask],
        y[mask]
    )

    return (
        float(sr),
        float(sp),
        float(pr),
        float(pp),
        int(mask.sum()),
    )


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.std(x, ddof=1))


# ------------------------------------------------------------
# LOAD MANIFEST
# ------------------------------------------------------------

print("=" * 70)
print("ISBI 2027 — SCANNER-PAIR + WITHIN-CASE ROBUSTNESS AUDIT")
print("=" * 70)

print("\nLoading manifest:")
print(MANIFEST)

manifest = pd.read_csv(
    MANIFEST,
    dtype=str
)

manifest = manifest.apply(
    lambda c:
    c.str.strip()
    if c.dtype == "object"
    else c
)

manifest["case_id"] = (
    manifest["case_id"]
    .map(clean_id)
)

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

if len(manifest) != 900:
    raise RuntimeError(
        "Expected exactly 900 embedding records."
    )

# ------------------------------------------------------------
# LOAD EMBEDDINGS
# ------------------------------------------------------------

print("\nLoading embeddings...")

import torch

lookup = {}

for _, row in manifest.iterrows():

    raw_path = Path(
        str(row["embedding_path"]).strip().replace("\\\\", "/")
    )

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([EMBED_ROOT / raw_path, PROJECT_ROOT / raw_path])

    parts = list(raw_path.parts)
    lower_parts = [part.lower() for part in parts]
    if "embeddings" in lower_parts:
        index = lower_parts.index("embeddings")
        relative_parts = parts[index + 1:]
        if relative_parts:
            candidates.append(EMBED_ROOT / Path(*relative_parts))

    path = next((candidate for candidate in candidates if candidate.is_file()), None)

    if path is None:
        matches = list(EMBED_ROOT.rglob(raw_path.name))
        if len(matches) == 1:
            path = matches[0]

    if path is None:
        path = EMBED_ROOT / raw_path

    if not path.exists():
        raise FileNotFoundError(
            f"Missing embedding:\n{path}"
        )

    z = torch.load(
        path,
        map_location="cpu"
    )

    if isinstance(z, dict):

        if "embedding" in z:
            z = z["embedding"]

        elif "features" in z:
            z = z["features"]

        elif "z" in z:
            z = z["z"]

        else:
            raise RuntimeError(
                f"Unknown embedding dictionary:\n{path}"
            )

    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()

    z = np.asarray(z).squeeze()

    if z.ndim != 1 or z.shape[0] != 1536:
        raise RuntimeError(
            f"Bad embedding shape {z.shape}:\n{path}"
        )

    if not np.all(np.isfinite(z)):
        raise RuntimeError(
            f"NaN/Inf embedding:\n{path}"
        )

    key = (
        row["case_id"],
        int(row["patch_id"]),
        row["scanner"],
    )

    lookup[key] = z.astype(
        np.float64
    )

print(
    f"Loaded embeddings: {len(lookup)}"
)

# ------------------------------------------------------------
# LOAD READER LABELS
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
    labels[isup_cols].max(axis=1)
    -
    labels[isup_cols].min(axis=1)
)

label_df = labels[
    [
        "case_id",
        "mean_isup",
        "reader_ambiguity",
    ]
].copy()

# ------------------------------------------------------------
# BUILD PAIRWISE PATCH DATA
# ------------------------------------------------------------

print("\nComputing scanner-pair metrics...")

pair_rows = []

for case in sorted(
    manifest["case_id"].unique()
):

    patches = sorted(
        manifest.loc[
            manifest["case_id"] == case,
            "patch_id"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    for patch in patches:

        for s1, s2, pair_name in PAIRS:

            z1 = lookup[
                (case, patch, s1)
            ]

            z2 = lookup[
                (case, patch, s2)
            ]

            n1 = np.linalg.norm(z1)
            n2 = np.linalg.norm(z2)

            if n1 <= 0 or n2 <= 0:
                raise RuntimeError(
                    f"Invalid norm: {case}, "
                    f"{patch}, {pair_name}"
                )

            cosine = float(
                np.dot(z1, z2)
                /
                (n1 * n2)
            )

            cosine = np.clip(
                cosine,
                -1.0,
                1.0
            )

            directional = float(
                2.0
                * n1
                * n2
                * (1.0 - cosine)
            )

            magnitude = float(
                (n1 - n2) ** 2
            )

            squared_distance = float(
                np.sum(
                    (z1 - z2) ** 2
                )
            )

            pair_rows.append({
                "case_id": case,
                "patch_id": patch,
                "pair": pair_name,

                "scanner_1": s1,
                "scanner_2": s2,

                "norm_1": n1,
                "norm_2": n2,

                "norm_difference": abs(
                    n1 - n2
                ),

                "magnitude_component":
                    magnitude,

                "directional_component":
                    directional,

                "cosine_similarity":
                    cosine,

                "cosine_disagreement":
                    1.0 - cosine,

                "squared_distance":
                    squared_distance,
            })

pair_df = pd.DataFrame(
    pair_rows
)

print(
    f"Pairwise rows: {len(pair_df)}"
)

if len(pair_df) != 900:
    raise RuntimeError(
        "Expected 900 pairwise rows "
        "(300 locations × 3 scanner pairs)."
    )

# ------------------------------------------------------------
# VERIFY DECOMPOSITION
# ------------------------------------------------------------

pair_df["decomposition_error"] = np.abs(
    pair_df["squared_distance"]
    -
    (
        pair_df["magnitude_component"]
        +
        pair_df["directional_component"]
    )
)

print(
    "\nMaximum pair decomposition error:",
    pair_df[
        "decomposition_error"
    ].max()
)

if (
    pair_df[
        "decomposition_error"
    ].max()
    > 1e-3
):
    raise RuntimeError(
        "Pairwise decomposition failed."
    )

# ------------------------------------------------------------
# CASE-LEVEL PAIR AGGREGATION
# ------------------------------------------------------------

case_pair = (
    pair_df
    .groupby(
        [
            "case_id",
            "pair",
        ]
    )
    .agg(
        patch_count=(
            "patch_id",
            "count"
        ),

        mean_directional_component=(
            "directional_component",
            "mean"
        ),

        median_directional_component=(
            "directional_component",
            "median"
        ),

        std_directional_component=(
            "directional_component",
            "std"
        ),

        mean_magnitude_component=(
            "magnitude_component",
            "mean"
        ),

        mean_cosine_disagreement=(
            "cosine_disagreement",
            "mean"
        ),

        mean_squared_distance=(
            "squared_distance",
            "mean"
        ),

        mean_norm_difference=(
            "norm_difference",
            "mean"
        ),
    )
    .reset_index()
)

case_pair = case_pair.merge(
    label_df,
    on="case_id",
    how="inner"
)

if len(case_pair) != 90:
    raise RuntimeError(
        "Expected 90 case-pair rows."
    )

# ------------------------------------------------------------
# PRINT SCANNER-PAIR SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SCANNER-PAIR SUMMARY")
print("=" * 70)

for pair_name in [
    "GH",
    "GL",
    "HL",
]:

    sub = case_pair[
        case_pair["pair"] == pair_name
    ]

    print(
        f"\n{pair_name}"
    )

    print(
        "  Directional component mean:",
        f"{sub['mean_directional_component'].mean():.6f}"
    )

    print(
        "  Magnitude component mean:",
        f"{sub['mean_magnitude_component'].mean():.6f}"
    )

    print(
        "  Cosine disagreement mean:",
        f"{sub['mean_cosine_disagreement'].mean():.6f}"
    )

    sr, sp, pr, pp, n = correlation(
        sub["reader_ambiguity"],
        sub[
            "mean_directional_component"
        ]
    )

    print(
        "  Ambiguity ↔ directional:"
    )

    print(
        f"    Spearman rho = {sr:.6f}, "
        f"p = {sp:.6f}"
    )

    print(
        f"    Pearson r    = {pr:.6f}, "
        f"p = {pp:.6f}"
    )

# ------------------------------------------------------------
# CORRELATION TABLE
# ------------------------------------------------------------

corr_rows = []

for pair_name in [
    "GH",
    "GL",
    "HL",
]:

    sub = case_pair[
        case_pair["pair"] == pair_name
    ]

    metrics = [
        "mean_directional_component",
        "mean_magnitude_component",
        "mean_cosine_disagreement",
        "mean_squared_distance",
        "mean_norm_difference",
    ]

    for metric in metrics:

        sr, sp, pr, pp, n = correlation(
            sub["reader_ambiguity"],
            sub[metric]
        )

        corr_rows.append({
            "pair": pair_name,
            "metric": metric,
            "spearman_rho": sr,
            "spearman_p": sp,
            "pearson_r": pr,
            "pearson_p": pp,
            "n": n,
        })

pair_corr_df = pd.DataFrame(
    corr_rows
)

# ------------------------------------------------------------
# WITHIN-CASE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WITHIN-CASE ROBUSTNESS")
print("=" * 70)

within_rows = []

for pair_name in [
    "GH",
    "GL",
    "HL",
]:

    sub = pair_df[
        pair_df["pair"] == pair_name
    ]

    case_stats = (
        sub
        .groupby("case_id")
        .agg(
            patch_mean=(
                "directional_component",
                "mean"
            ),

            patch_median=(
                "directional_component",
                "median"
            ),

            patch_std=(
                "directional_component",
                "std"
            ),

            patch_min=(
                "directional_component",
                "min"
            ),

            patch_max=(
                "directional_component",
                "max"
            ),

            patch_count=(
                "patch_id",
                "count"
            ),
        )
        .reset_index()
    )

    case_stats = case_stats.merge(
        label_df,
        on="case_id",
        how="inner"
    )

    # Coefficient of variation
    case_stats[
        "within_case_cv"
    ] = (
        case_stats["patch_std"]
        /
        case_stats["patch_mean"]
    )

    sr, sp, pr, pp, n = correlation(
        case_stats["reader_ambiguity"],
        case_stats["patch_std"]
    )

    print(
        f"\n{pair_name}:"
    )

    print(
        "  Mean within-case SD =",
        f"{case_stats['patch_std'].mean():.6f}"
    )

    print(
        "  Mean within-case CV =",
        f"{case_stats['within_case_cv'].mean():.6f}"
    )

    print(
        "  Ambiguity ↔ within-case SD:"
    )

    print(
        f"    Spearman rho = {sr:.6f}, "
        f"p = {sp:.6f}"
    )

    within_rows.append({
        "pair": pair_name,
        "mean_within_case_sd":
            case_stats["patch_std"].mean(),
        "mean_within_case_cv":
            case_stats["within_case_cv"].mean(),
        "ambiguity_vs_within_sd_spearman":
            sr,
        "ambiguity_vs_within_sd_p":
            sp,
    })

within_df = pd.DataFrame(
    within_rows
)

# ------------------------------------------------------------
# LEAVE-ONE-PATCH-OUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LEAVE-ONE-PATCH-OUT SENSITIVITY")
print("=" * 70)

loo_rows = []

for pair_name in [
    "GH",
    "GL",
    "HL",
]:

    sub = pair_df[
        pair_df["pair"] == pair_name
    ].copy()

    # Baseline case means
    baseline = (
        sub
        .groupby("case_id")
        [
            "directional_component"
        ]
        .mean()
        .rename("baseline")
        .reset_index()
    )

    baseline = baseline.merge(
        label_df,
        on="case_id",
        how="inner"
    )

    base_rho, base_p, _, _, _ = correlation(
        baseline["reader_ambiguity"],
        baseline["baseline"]
    )

    print(
        f"\n{pair_name} baseline:"
    )

    print(
        f"  rho = {base_rho:.6f}, "
        f"p = {base_p:.6f}"
    )

    # Recompute correlation after removing
    # each patch index from every case.
    #
    # This tests sensitivity to spatial locations
    # without treating patches as independent cases.

    patch_ids = sorted(
        sub["patch_id"].unique()
    )

    loo_values = []

    for removed_patch in patch_ids:

        reduced = sub[
            sub["patch_id"]
            != removed_patch
        ]

        case_means = (
            reduced
            .groupby("case_id")
            [
                "directional_component"
            ]
            .mean()
            .rename("directional_mean")
            .reset_index()
        )

        case_means = case_means.merge(
            label_df,
            on="case_id",
            how="inner"
        )

        rho, p, _, _, _ = correlation(
            case_means[
                "reader_ambiguity"
            ],
            case_means[
                "directional_mean"
            ]
        )

        loo_values.append({
            "pair": pair_name,
            "removed_patch": removed_patch,
            "rho": rho,
            "p": p,
        })

    loo_values_df = pd.DataFrame(
        loo_values
    )

    loo_rows.append(
        loo_values_df
    )

    print(
        "  LOO rho range:",
        f"{loo_values_df['rho'].min():.6f}",
        "to",
        f"{loo_values_df['rho'].max():.6f}"
    )

    print(
        "  LOO p range:",
        f"{loo_values_df['p'].min():.6f}",
        "to",
        f"{loo_values_df['p'].max():.6f}"
    )

loo_df = pd.concat(
    loo_rows,
    ignore_index=True
)

# ------------------------------------------------------------
# LEAVE-ONE-CASE-OUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LEAVE-ONE-CASE-OUT SENSITIVITY")
print("=" * 70)

case_loo_rows = []

for pair_name in [
    "GH",
    "GL",
    "HL",
]:

    sub = case_pair[
        case_pair["pair"] == pair_name
    ].copy()

    metric = (
        "mean_directional_component"
    )

    full_rho, full_p, _, _, _ = correlation(
        sub["reader_ambiguity"],
        sub[metric]
    )

    loo_rhos = []
    loo_ps = []

    for case in sub["case_id"]:

        tmp = sub[
            sub["case_id"] != case
        ]

        rho, p, _, _, _ = correlation(
            tmp["reader_ambiguity"],
            tmp[metric]
        )

        loo_rhos.append(rho)
        loo_ps.append(p)

        case_loo_rows.append({
            "pair": pair_name,
            "removed_case": case,
            "rho": rho,
            "p": p,
        })

    print(
        f"\n{pair_name}:"
    )

    print(
        f"  Full rho = "
        f"{full_rho:.6f}"
    )

    print(
        f"  LOO rho range = "
        f"{min(loo_rhos):.6f} "
        f"to "
        f"{max(loo_rhos):.6f}"
    )

    print(
        f"  LOO p range = "
        f"{min(loo_ps):.6f} "
        f"to "
        f"{max(loo_ps):.6f}"
    )

case_loo_df = pd.DataFrame(
    case_loo_rows
)

# ------------------------------------------------------------
# CONSISTENCY ACROSS SCANNER PAIRS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PAIR CONSISTENCY")
print("=" * 70)

pair_case_pivot = (
    case_pair
    .pivot(
        index="case_id",
        columns="pair",
        values="mean_directional_component"
    )
)

print(
    "\nDirectional component pairwise correlations:"
)

for p1, p2 in [
    ("GH", "GL"),
    ("GH", "HL"),
    ("GL", "HL"),
]:

    sr, sp, pr, pp, n = correlation(
        pair_case_pivot[p1],
        pair_case_pivot[p2]
    )

    print(
        f"\n{p1} vs {p2}:"
    )

    print(
        f"  Spearman rho = {sr:.6f}, "
        f"p = {sp:.6f}"
    )

    print(
        f"  Pearson r = {pr:.6f}, "
        f"p = {pp:.6f}"
    )

# ------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------

pair_file = (
    OUT
    / "scanner_pair_patch_level_900.csv"
)

case_pair_file = (
    OUT
    / "scanner_pair_case_level_90.csv"
)

corr_file = (
    OUT
    / "scanner_pair_correlations.csv"
)

within_file = (
    OUT
    / "within_case_robustness.csv"
)

loo_patch_file = (
    OUT
    / "leave_one_patch_out.csv"
)

loo_case_file = (
    OUT
    / "leave_one_case_out.csv"
)

pair_df.to_csv(
    pair_file,
    index=False
)

case_pair.to_csv(
    case_pair_file,
    index=False
)

pair_corr_df.to_csv(
    corr_file,
    index=False
)

within_df.to_csv(
    within_file,
    index=False
)

loo_df.to_csv(
    loo_patch_file,
    index=False
)

case_loo_df.to_csv(
    loo_case_file,
    index=False
)

# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------


# Pair-specific ambiguity plots — publication-readable formatting
for pair_name in ["GH", "GL", "HL"]:

    sub = case_pair[case_pair["pair"] == pair_name]

    x = sub["reader_ambiguity"]
    y = sub["mean_directional_component"]

    rho, p, _, _, _ = correlation(x, y)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.scatter(
        x,
        y,
        s=70,
        alpha=0.85
    )

    ax.set_xlabel(
        "Reader-level diagnostic ambiguity",
        fontsize=15,
        labelpad=10
    )

    ax.set_ylabel(
        "Directional scanner component",
        fontsize=15,
        labelpad=10
    )

    ax.set_title(
        f"{pair_name}: Ambiguity vs directional component\n"
        f"Spearman ρ = {rho:.3f}, p = {p:.4f}",
        fontsize=16,
        pad=14
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=13,
        length=5,
        width=1.2
    )

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.8
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()

    fig.savefig(
        OUT / f"{pair_name}_ambiguity_directional.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


# LOO patch robustness plot
plt.figure(
    figsize=(8, 5)
)

for pair_name in [
    "GH",
    "GL",
    "HL",
]:

    sub = loo_df[
        loo_df["pair"] == pair_name
    ]

    plt.plot(
        sub["removed_patch"],
        sub["rho"],
        marker="o",
        label=pair_name
    )

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Patch index removed"
)

plt.ylabel(
    "Spearman correlation"
)

plt.title(
    "Leave-one-patch-out robustness"
)

plt.legend()

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUT
    /
    "leave_one_patch_out_rho.png",
    dpi=300
)

plt.close()

# LOO case robustness
plt.figure(
    figsize=(8, 5)
)

for pair_name in [
    "GH",
    "GL",
    "HL",
]:

    sub = case_loo_df[
        case_loo_df["pair"] == pair_name
    ]

    plt.plot(
        range(len(sub)),
        sub["rho"],
        marker="o",
        label=pair_name
    )

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Removed-case iteration"
)

plt.ylabel(
    "Spearman correlation"
)

plt.title(
    "Leave-one-case-out robustness"
)

plt.legend()

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUT
    /
    "leave_one_case_out_rho.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary_file = (
    OUT
    / "scanner_pair_directional_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ISBI 2027 — Scanner-Pair + "
        "Within-Case Directional Robustness Audit\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Cohort: 30 cases\n"
    )

    f.write(
        "Spatial locations: 300\n"
    )

    f.write(
        "Scanner pairs: GH, GL, HL\n"
    )

    f.write(
        "Analysis unit for inference: case\n\n"
    )

    f.write(
        "SCANNER-PAIR ASSOCIATIONS\n"
    )

    f.write(
        "--------------------------\n"
    )

    for pair_name in [
        "GH",
        "GL",
        "HL",
    ]:

        sub = case_pair[
            case_pair["pair"]
            == pair_name
        ]

        for metric in [
            "mean_directional_component",
            "mean_magnitude_component",
            "mean_cosine_disagreement",
        ]:

            sr, sp, pr, pp, n = correlation(
                sub["reader_ambiguity"],
                sub[metric]
            )

            f.write(
                f"{pair_name} | "
                f"{metric}: "
                f"Spearman rho={sr:.6f}, "
                f"p={sp:.6f}; "
                f"Pearson r={pr:.6f}, "
                f"p={pp:.6f}\n"
            )

    f.write(
        "\nWITHIN-CASE ROBUSTNESS\n"
    )

    f.write(
        "----------------------\n"
    )

    for _, row in within_df.iterrows():

        f.write(
            f"{row['pair']}: "
            f"mean within-case SD="
            f"{row['mean_within_case_sd']:.6f}; "
            f"mean within-case CV="
            f"{row['mean_within_case_cv']:.6f}; "
            f"ambiguity-vs-SD rho="
            f"{row['ambiguity_vs_within_sd_spearman']:.6f}; "
            f"p="
            f"{row['ambiguity_vs_within_sd_p']:.6f}\n"
        )

    f.write(
        "\nIMPORTANT\n"
    )

    f.write(
        "All inferential correlations are case-level.\n"
        "Patch locations are not treated as independent cases.\n"
        "Leave-one-patch-out and leave-one-case-out analyses "
        "are sensitivity analyses.\n"
    )

    f.write(
        "These analyses are exploratory and do not establish "
        "causality.\n"
    )

# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SCANNER-PAIR ROBUSTNESS AUDIT COMPLETE")
print("=" * 70)

print("\nOutputs:")

for p in sorted(
    OUT.iterdir()
):

    print(
        " ",
        p
    )

print(
    "\nSend me the COMPLETE terminal output."
)

print(
    "Do not train a model or modify the dataset yet."
)