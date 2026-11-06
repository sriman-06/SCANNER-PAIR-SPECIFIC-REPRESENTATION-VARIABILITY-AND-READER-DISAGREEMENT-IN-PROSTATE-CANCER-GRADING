from pathlib import Path
import os
import re
import time

import pandas as pd
import torch
from PIL import Image
import timm

from timm.data import create_transform, resolve_data_config


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PATCH_ROOT = Path(
    os.environ.get(
        "ISBI_PATCH_ROOT",
        PROJECT_ROOT / "outputs" / "paired_patches_30"
    )
).expanduser().resolve()

MANIFEST_PATH = PATCH_ROOT / "paired_patch_manifest_30.csv"

OUTPUT_ROOT = Path(
    os.environ.get(
        "ISBI_EMBED_OUTPUT_ROOT",
        PROJECT_ROOT / "outputs" / "uni2h_30"
    )
).expanduser().resolve()

EMBED_ROOT = OUTPUT_ROOT / "embeddings"
EMBED_MANIFEST = OUTPUT_ROOT / "embedding_manifest_30.csv"
EMBED_SUMMARY = OUTPUT_ROOT / "embedding_summary_30.csv"
EMBED_QC = OUTPUT_ROOT / "embedding_qc_30.csv"
MODEL_NAME = "hf-hub:MahmoodLab/UNI2-h"

EXPECTED_DIM = 1536

EXPECTED_CASES = 30

EXPECTED_LOCATIONS = 300

EXPECTED_LOCATIONS_PER_CASE = 10

EXPECTED_SCANNERS = 3

EXPECTED_IMAGES = 900

SCANNERS = [
    "grundium",
    "hamamatsu",
    "leica",
]

SCANNER_FILE_COLUMNS = {
    "grundium": "grundium_file",
    "hamamatsu": "hamamatsu_file",
    "leica": "leica_file",
}

CASE_COLUMN = "case"

PATCH_COLUMN = "patch_id"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("UNI2-h FULL 900-IMAGE EMBEDDING EXTRACTION")
print("=" * 80)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Model: {MODEL_NAME}")
print(f"Expected paired locations: {EXPECTED_LOCATIONS}")
print(f"Expected images: {EXPECTED_IMAGES}")
print(f"Expected embedding dimension: {EXPECTED_DIM}")

print("=" * 80)


# ============================================================
# LOAD PATCH MANIFEST
# ============================================================

if not MANIFEST_PATH.exists():

    raise FileNotFoundError(
        f"\nPatch manifest not found:\n"
        f"{MANIFEST_PATH}"
    )

manifest = pd.read_csv(
    MANIFEST_PATH
)

print("\nPatch manifest loaded.")

print(f"Manifest rows: {len(manifest)}")

print("\nColumns:")

for column in manifest.columns:

    print(f"  {column}")


# ============================================================
# MANIFEST SCHEMA VALIDATION
# ============================================================

required_columns = [

    CASE_COLUMN,

    PATCH_COLUMN,

    "candidate_id",

    "grundium_file",

    "hamamatsu_file",

    "leica_file",

    "valid_triplet",

]

missing_columns = [

    c
    for c in required_columns
    if c not in manifest.columns

]

if missing_columns:

    raise RuntimeError(
        "\nManifest is missing required columns:\n"
        + "\n".join(
            f"  {c}"
            for c in missing_columns
        )
    )


# ============================================================
# MANIFEST QC
# ============================================================

print("\n" + "=" * 80)
print("PATCH MANIFEST QC")
print("=" * 80)

if len(manifest) != EXPECTED_LOCATIONS:

    raise RuntimeError(
        f"Expected {EXPECTED_LOCATIONS} paired locations, "
        f"but found {len(manifest)}."
    )

print(
    f"Paired locations: {len(manifest)} / "
    f"{EXPECTED_LOCATIONS} PASS"
)


# ------------------------------------------------------------
# ------------------------------------------------------------
# CASE AND PATCH IDENTIFIER INTEGRITY
# ------------------------------------------------------------

identifier_columns = [
    CASE_COLUMN,
    PATCH_COLUMN
]

for column in identifier_columns:
    invalid_ids = (
        manifest[column].isna()
        | manifest[column].astype(str).str.strip().eq("")
    )

    if invalid_ids.any():
        raise RuntimeError(
            f"Found {int(invalid_ids.sum())} missing or empty "
            f"identifiers in column: {column}"
        )

duplicate_locations = manifest[
    manifest.duplicated(
        subset=[CASE_COLUMN, PATCH_COLUMN],
        keep=False
    )
].sort_values(
    [CASE_COLUMN, PATCH_COLUMN]
)

if not duplicate_locations.empty:
    print("\nDuplicate case/patch identifiers:")
    print(
        duplicate_locations[
            [CASE_COLUMN, PATCH_COLUMN]
        ].to_string(index=False)
    )

    raise RuntimeError(
        "Duplicate case/patch identifiers detected."
    )

print("Case/patch identifier integrity: PASS")

# Cases
# ------------------------------------------------------------

cases = sorted(
    manifest[CASE_COLUMN]
    .astype(str)
    .unique()
)

print(
    f"Cases: {len(cases)} / "
    f"{EXPECTED_CASES}"
)

if len(cases) != EXPECTED_CASES:

    raise RuntimeError(
        f"Expected {EXPECTED_CASES} cases, "
        f"found {len(cases)}."
    )


# ------------------------------------------------------------
# Locations per case
# ------------------------------------------------------------

case_counts = (
    manifest
    .groupby(CASE_COLUMN)
    .size()
)

bad_cases = case_counts[
    case_counts != EXPECTED_LOCATIONS_PER_CASE
]

if len(bad_cases) > 0:

    print("\nBad case counts:")

    print(
        bad_cases.to_string()
    )

    raise RuntimeError(
        "Each case must contain exactly "
        "10 paired locations."
    )

print(
    "Locations/case: 10 / 10 PASS"
)


# ------------------------------------------------------------
# valid_triplet
# ------------------------------------------------------------

valid_values = (
    manifest["valid_triplet"]
    .astype(str)
    .str.lower()
)

invalid_triplets = manifest[
    ~valid_values.isin(
        ["true", "1", "yes"]
    )
]

if len(invalid_triplets) > 0:

    raise RuntimeError(
        f"Found {len(invalid_triplets)} "
        "manifest rows with invalid paired-triplet status."
    )

print(
    "All paired locations marked valid_triplet: PASS"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

EMBED_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(
    value
):

    raw = str(value).strip()

    path = Path(raw)

    # --------------------------------------------------------
    # Absolute path
    # --------------------------------------------------------

    if path.is_absolute():

        if path.exists():

            return path.resolve()

    # --------------------------------------------------------
    # Relative path candidates
    # --------------------------------------------------------

    candidates = [

        PATCH_ROOT / path,

        ROOT / path,

        path,

    ]

    for candidate in candidates:

        if candidate.exists():

            return candidate.resolve()

    # --------------------------------------------------------
    # Return unresolved path
    # --------------------------------------------------------

    return path


# ============================================================
# UNI2-h SINGLE IMAGE INFERENCE
# ============================================================

@torch.inference_mode()
def extract_embedding(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        non_blocking=True
    )

    if DEVICE.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            embedding = model(
                tensor
            )

    else:

        embedding = model(
            tensor
        )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if embedding.ndim != 2:

        raise RuntimeError(
            "Unexpected embedding tensor shape: "
            f"{tuple(embedding.shape)}"
        )

    if embedding.shape[0] != 1:

        raise RuntimeError(
            "Expected batch size 1, "
            f"got {embedding.shape[0]}"
        )

    if embedding.shape[1] != EXPECTED_DIM:

        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{embedding.shape[1]} "
            f"(expected {EXPECTED_DIM})"
        )

    # --------------------------------------------------------
    # CPU float32
    # --------------------------------------------------------

    embedding = (
        embedding
        .squeeze(0)
        .float()
        .cpu()
    )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not torch.isfinite(
        embedding
    ).all():

        raise RuntimeError(
            "Embedding contains NaN or Inf."
        )

    return embedding


# ============================================================
# BUILD 900 IMAGE JOBS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING 900 EMBEDDING JOBS")
print("=" * 80)

jobs = []

for _, row in manifest.iterrows():

    case_id = str(
        row[CASE_COLUMN]
    )

    patch_id = str(
        row[PATCH_COLUMN]
    )

    for scanner in SCANNERS:

        file_column = (
            SCANNER_FILE_COLUMNS[
                scanner
            ]
        )

        image_path = resolve_image_path(
            row[file_column]
        )

        jobs.append({

            "case_id": case_id,

            "patch_id": patch_id,

            "scanner": scanner,

            "image_path": image_path,

        })


print(
    f"Embedding jobs created: "
    f"{len(jobs)}"
)

if len(jobs) != EXPECTED_IMAGES:

    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} "
        f"embedding jobs, got {len(jobs)}."
    )

print(
    "900-job construction: PASS"
)


# ============================================================
# INPUT IMAGE PREFLIGHT

missing_images = [
    str(job["image_path"])
    for job in jobs
    if not job["image_path"].is_file()
]

if missing_images:
    print(f"\nMissing input images: {len(missing_images)}")
    for image_path in missing_images[:20]:
        print(f"  {image_path}")
    raise FileNotFoundError(
        "Input image preflight failed. UNI2-h was not loaded."
    )

print("All 900 input image paths exist: PASS")

# LOAD UNI2-h
# ============================================================

print("\nLoading UNI2-h...")

timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    **timm_kwargs,
)

model = model.to(DEVICE)

model.eval()

transform = create_transform(
    **resolve_data_config(
        model.pretrained_cfg,
        model=model,
    )
)

print("UNI2-h loaded successfully.")
print(f"Embedding dimension: {EXPECTED_DIM}")

print("=" * 80)


# ============================================================
# EXTRACTION
# ============================================================

print("\n" + "=" * 80)
print("STARTING UNI2-h EXTRACTION")
print("=" * 80)

records = []

start_time = time.time()

successful = 0

failed = 0

for job_index, job in enumerate(
    jobs,
    start=1
):

    case_id = job["case_id"]

    patch_id = job["patch_id"]

    scanner = job["scanner"]

    image_path = job["image_path"]

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    case_dir = (
        EMBED_ROOT /
        case_id
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Safe names
    # --------------------------------------------------------

    safe_patch = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        patch_id
    )

    safe_scanner = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        scanner
    )

    output_path = (
        case_dir /
        f"patch_{safe_patch}_{safe_scanner}_uni2h.pt"
    )

    # --------------------------------------------------------
    # Existing valid embedding
    # --------------------------------------------------------

    if output_path.exists():

        try:

            existing = torch.load(
                output_path,
                map_location="cpu"
            )

            if (
                isinstance(
                    existing,
                    torch.Tensor
                )
                and
                tuple(
                    existing.shape
                ) == (EXPECTED_DIM,)
                and
                bool(
                    torch.isfinite(
                        existing
                    ).all()
                )
            ):

                successful += 1

                records.append({

                    "case_id": case_id,

                    "patch_id": patch_id,

                    "scanner": scanner,

                    "image_path": str(
                        image_path
                    ),

                    "embedding_path": str(
                        output_path
                    ),

                    "embedding_dim":
                        EXPECTED_DIM,

                    "status":
                        "EXISTING_VALID",

                })

                print(
                    f"[{job_index:03d}/900] "
                    f"{case_id} | "
                    f"{patch_id} | "
                    f"{scanner:<10} | "
                    f"EXISTING VALID"
                )

                continue

        except Exception:

            pass

    # --------------------------------------------------------
    # Image existence
    # --------------------------------------------------------

    if not image_path.exists():

        failed += 1

        records.append({

            "case_id": case_id,

            "patch_id": patch_id,

            "scanner": scanner,

            "image_path": str(
                image_path
            ),

            "embedding_path": str(
                output_path
            ),

            "embedding_dim": "",

            "status":
                "FAILED_IMAGE_MISSING",

        })

        print(
            f"[{job_index:03d}/900] "
            f"{case_id} | "
            f"{patch_id} | "
            f"{scanner:<10} | "
            f"IMAGE MISSING"
        )

        continue

    # --------------------------------------------------------
    # Extract embedding
    # --------------------------------------------------------

    try:

        embedding = extract_embedding(
            image_path
        )

        # ----------------------------------------------------
        # Temporary file
        # ----------------------------------------------------

        temp_path = (
            output_path
            .with_suffix(
                ".tmp.pt"
            )
        )

        torch.save(
            embedding,
            temp_path
        )

        temp_path.replace(
            output_path
        )

        successful += 1

        records.append({

            "case_id": case_id,

            "patch_id": patch_id,

            "scanner": scanner,

            "image_path": str(
                image_path
            ),

            "embedding_path": str(
                output_path
            ),

            "embedding_dim":
                int(
                    embedding.shape[0]
                ),

            "status":
                "PASS",

        })

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            successful / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            EXPECTED_IMAGES
            - job_index
        )

        eta_seconds = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"[{job_index:03d}/900] "
            f"{case_id} | "
            f"{patch_id} | "
            f"{scanner:<10} | "
            f"PASS | "
            f"1536-D | "
            f"ETA {eta_seconds/60:.1f} min"
        )

    except Exception as e:

        failed += 1

        records.append({

            "case_id": case_id,

            "patch_id": patch_id,

            "scanner": scanner,

            "image_path": str(
                image_path
            ),

            "embedding_path": str(
                output_path
            ),

            "embedding_dim": "",

            "status":
                f"FAILED: {repr(e)}",

        })

        print(
            f"[{job_index:03d}/900] "
            f"{case_id} | "
            f"{patch_id} | "
            f"{scanner:<10} | "
            f"FAILED"
        )

        print(
            f"    Error: {repr(e)}"
        )

    # --------------------------------------------------------
    # Periodic CUDA cleanup
    # --------------------------------------------------------

    if (
        DEVICE.type == "cuda"
        and
        job_index % 50 == 0
    ):

        torch.cuda.empty_cache()


# ============================================================
# SAVE MANIFEST
# ============================================================

embedding_df = pd.DataFrame(
    records
)

embedding_df.to_csv(
    EMBED_MANIFEST,
    index=False
)

print("\n" + "=" * 80)
print("EXTRACTION FINISHED")
print("=" * 80)

print(
    f"Expected:  {EXPECTED_IMAGES}"
)

print(
    f"Successful: {successful}"
)

print(
    f"Failed:     {failed}"
)

print(
    f"\nManifest:\n{EMBED_MANIFEST}"
)


# ============================================================
# FINAL COUNT CHECK
# ============================================================

if len(embedding_df) != EXPECTED_IMAGES:
    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} manifest records, "
        f"found {len(embedding_df)}."
    )

job_duplicates = embedding_df[
    embedding_df.duplicated(
        subset=["case_id", "patch_id", "scanner"],
        keep=False
    )
]

if not job_duplicates.empty:
    raise RuntimeError(
        "Duplicate embedding jobs found in embedding manifest."
    )

if embedding_df["embedding_path"].duplicated().any():
    raise RuntimeError(
        "Duplicate embedding output paths found."
    )

print("900 manifest records and unique jobs: PASS")

valid_statuses = [
    "PASS",
    "EXISTING_VALID",
]

valid_df = embedding_df[
    embedding_df["status"].isin(valid_statuses)
].copy()

if len(valid_df) != EXPECTED_IMAGES:
    print("\nINVALID EMBEDDING COUNT.")
    print(
        embedding_df[
            ["case_id", "patch_id", "scanner", "status"]
        ].to_string(index=False)
    )
    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} valid embeddings, "
        f"found {len(valid_df)}."
    )

# PHYSICAL EMBEDDING QC
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING ALL 900 EMBEDDING FILES")
print("=" * 80)

qc_records = []

for _, row in valid_df.iterrows():

    embedding_path = Path(
        row["embedding_path"]
    )

    status = "PASS"

    shape = ""

    finite = False

    try:

        if not embedding_path.exists():

            status = "MISSING"

        else:

            embedding = torch.load(
                embedding_path,
                map_location="cpu"
            )

            shape = str(
                tuple(
                    embedding.shape
                )
            )

            finite = bool(
                torch.isfinite(
                    embedding
                ).all()
            )

            if (
                tuple(
                    embedding.shape
                )
                !=
                (EXPECTED_DIM,)
            ):

                status = "BAD_SHAPE"

            elif not finite:

                status = "NONFINITE"

    except Exception as e:

        status = (
            "LOAD_ERROR: "
            + repr(e)
        )

    qc_records.append({

        "case_id":
            row["case_id"],

        "patch_id":
            row["patch_id"],

        "scanner":
            row["scanner"],

        "embedding_path":
            str(embedding_path),

        "shape":
            shape,

        "finite":
            finite,

        "status":
            status,

    })


qc_df = pd.DataFrame(
    qc_records
)

qc_df.to_csv(
    EMBED_QC,
    index=False
)


bad_qc = qc_df[
    qc_df["status"] != "PASS"
]

print(
    f"Files checked: {len(qc_df)}"
)

print(
    f"QC failures:  {len(bad_qc)}"
)

if len(bad_qc) > 0:

    print("\nQC FAILURES:")

    print(
        bad_qc.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "UNI2-h embedding QC FAILED."
    )

print(
    "Embedding file QC: PASS"
)


# ============================================================
# CASE / SCANNER STRUCTURE QC
# ============================================================

print("\n" + "=" * 80)
print("CASE / SCANNER STRUCTURE QC")
print("=" * 80)

structure = (
    qc_df
    .groupby(["case_id", "scanner"])
    .size()
    .reindex(
        pd.MultiIndex.from_product(
            [cases, SCANNERS],
            names=["case_id", "scanner"]
        ),
        fill_value=0
    )
    .reset_index(name="count")
)

bad_structure = structure[
    structure["count"] != EXPECTED_LOCATIONS_PER_CASE
]

if not bad_structure.empty:
    print(bad_structure.to_string(index=False))
    raise RuntimeError(
        "Case/scanner structure QC FAILED."
    )

if len(structure) != EXPECTED_CASES * EXPECTED_SCANNERS:
    raise RuntimeError(
        "Expected exactly 90 case/scanner combinations."
    )

print("30 cases x 3 scanners x 10 patches: PASS")

# PATCH MATCHING QC
# ============================================================

print("\n" + "=" * 80)
print("MATCHED PATCH QC")
print("=" * 80)

unique_triplets = (
    qc_df
    .groupby(
        [
            "case_id",
            "patch_id"
        ]
    )["scanner"]
    .nunique()
)

bad_triplets = unique_triplets[
    unique_triplets != EXPECTED_SCANNERS
]

print(
    f"Matched spatial locations: "
    f"{len(unique_triplets)}"
)

if len(unique_triplets) != EXPECTED_LOCATIONS:

    raise RuntimeError(
        "Expected exactly 300 "
        "matched spatial locations."
    )

if len(bad_triplets) > 0:

    print(
        bad_triplets.to_string()
    )

    raise RuntimeError(
        "Some spatial locations "
        "do not contain all 3 scanners."
    )

print(
    "300/300 locations have all 3 scanners: PASS"
)


# ============================================================
# SUMMARY
# ============================================================

summary_records = []

for case_id in cases:

    case_qc = qc_df[
        qc_df["case_id"] ==
        case_id
    ]

    for scanner in SCANNERS:

        scanner_count = len(
            case_qc[
                case_qc["scanner"] ==
                scanner
            ]
        )

        summary_records.append({

            "case_id":
                case_id,

            "scanner":
                scanner,

            "expected":
                EXPECTED_LOCATIONS_PER_CASE,

            "actual":
                scanner_count,

            "status":
                (
                    "PASS"
                    if
                    scanner_count ==
                    EXPECTED_LOCATIONS_PER_CASE
                    else
                    "FAIL"
                ),

        })


summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    EMBED_SUMMARY,
    index=False
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("ALL UNI2-h EMBEDDING QC PASSED")
print("=" * 80)

print(
    f"Cases:                 {EXPECTED_CASES}"
)

print(
    f"Paired locations:      {EXPECTED_LOCATIONS}"
)

print(
    f"Images:                {EXPECTED_IMAGES}"
)

print(
    f"Embeddings:            {len(qc_df)}"
)

print(
    f"Embedding dimension:   {EXPECTED_DIM}"
)

print(
    "NaN/Inf:               0"
)

print(
    "Case/scanner structure: PASS"
)

print(
    "Matched triplets:       300/300"
)

print(
    "\nEmbedding manifest:"
)

print(
    EMBED_MANIFEST
)

print(
    "\nEmbedding QC:"
)

print(
    EMBED_QC
)

print(
    "\nEmbedding summary:"
)

print(
    EMBED_SUMMARY
)

print("\nNEXT STEP:")
print(
    "Proceed to matched-triplet "
    "scanner-induced representation "
    "variability analysis."
)

print("=" * 80)