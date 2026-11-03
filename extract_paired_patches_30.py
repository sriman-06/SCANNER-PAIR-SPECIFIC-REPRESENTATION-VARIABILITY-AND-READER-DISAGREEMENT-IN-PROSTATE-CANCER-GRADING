from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import openslide
from PIL import Image
import json
import math
import random


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"D:\ISBI_2027\data")

REG_ROOT = Path(
    r"D:\ISBI_2027\outputs\registration_all"
)

OUT_ROOT = Path(
    r"D:\ISBI_2027\outputs\paired_patches_30"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_OUT = (
    OUT_ROOT /
    "paired_patch_manifest_30.csv"
)

SUMMARY_OUT = (
    OUT_ROOT /
    "paired_patch_extraction_summary.csv"
)


# ------------------------------------------------------------
# Determinism
# ------------------------------------------------------------

# Not used for random sampling at the moment, but kept
# available so any future randomized tie-breaking is
# reproducible.
RANDOM_SEED = 20261007


# ------------------------------------------------------------
# Cohort
# ------------------------------------------------------------

COHORT = [
    "C001A",
    "C002A",
    "C002B",
    "C008A",
    "C011A",
    "C025B",
    "C027",
    "C032B",
    "C034B",
    "C036B",
    "C038A",
    "C038B",
    "C040A",
    "C040B",
    "C045B",
    "C049B",
    "C058",
    "C059A",
    "C063B",
    "C072A",
    "C077A",
    "C081B",
    "C082",
    "C092B",
    "C098A",
    "C113A",
    "C120A",
    "C134A",
    "C151A",
    "C159B",
]


SCANNERS = [
    "grundium",
    "hamamatsu",
    "leica",
]

REFERENCE = "grundium"

MOVING = [
    "hamamatsu",
    "leica",
]


# ============================================================
# PATCH PARAMETERS
# ============================================================

# Target physical patch:
# 256 um x 256 um
PATCH_UM = 256.0

# All final images are resampled to:
# 0.5 um/pixel -> 512 x 512
TARGET_MPP = 0.5

OUTPUT_SIZE = 512

# Number of locations per slide that we actually SAVE
PATCHES_PER_CASE = 10

# Number of spatially distributed candidates to SELECT
# before extraction.  We extract the first PATCHES_PER_CASE
# candidates that yield complete triplets; the rest of the
# pool is available as replacements.
CANDIDATE_POOL_SIZE = 30

# Reference thumbnail used for sampling
THUMBNAIL_MAX_DIM = 1400

# Tissue thresholds
REFERENCE_MIN_TISSUE = 0.50
ALL_SCANNERS_MIN_TISSUE = 0.30

# Candidate spacing in reference thumbnail
CANDIDATE_STEP = 8

# Minimum distance between selected points
# expressed relative to thumbnail diagonal
MIN_SPACING_FRACTION = 0.08


# ============================================================
# EXPECTED MPP FALLBACKS
# ============================================================

# These are only fallbacks if OpenSlide metadata does not
# contain MPP values.

FALLBACK_MPP = {
    "grundium": 0.250552,
    "hamamatsu": 0.22658781410735732,
    "leica": 0.263516,
}


# ============================================================
# FILE NAMES
# ============================================================

def filename_for(case, scanner):

    c = case.lower()

    if scanner == "grundium":
        return f"{c}_grundium.svs"

    if scanner == "hamamatsu":
        return f"{c}_hamamatsu.ndpi"

    if scanner == "leica":
        return f"{c}_leica.svs"

    raise ValueError(scanner)


# ============================================================
# MPP
# ============================================================

def get_mpp(slide, scanner):

    keys_x = [
        openslide.PROPERTY_NAME_MPP_X,
        "openslide.mpp-x",
        "aperio.MPP",
    ]

    for key in keys_x:

        value = slide.properties.get(key)

        if value is not None:

            try:

                value = float(value)

                if value > 0:
                    return value

            except Exception:
                pass

    return FALLBACK_MPP[scanner]


# ============================================================
# THUMBNAIL
# ============================================================

def load_thumbnail(path):

    slide = openslide.OpenSlide(
        str(path)
    )

    width, height = slide.dimensions

    scale = min(
        THUMBNAIL_MAX_DIM / width,
        THUMBNAIL_MAX_DIM / height,
        1.0,
    )

    tw = max(
        1,
        int(round(width * scale))
    )

    th = max(
        1,
        int(round(height * scale))
    )

    thumb = slide.get_thumbnail(
        (tw, th)
    )

    mpp = None

    try:
        mpp = get_mpp(
            slide,
            ""
        )
    except Exception:
        pass

    slide.close()

    rgb = np.asarray(
        thumb.convert("RGB")
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    return rgb, gray


# ============================================================
# TISSUE MASK
# ============================================================

def tissue_mask(rgb):

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    mask = (
        (
            (saturation > 20) &
            (value < 250)
        )
        |
        (gray < 220)
    )

    mask = mask.astype(
        np.uint8
    )

    # Remove tiny isolated noise
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# ============================================================
# TISSUE FRACTION
# ============================================================

def patch_tissue_fraction(
    rgb,
    x,
    y,
    width,
    height
):

    h, w = rgb.shape[:2]

    x0 = int(
        round(x - width / 2)
    )

    y0 = int(
        round(y - height / 2)
    )

    x1 = x0 + int(width)
    y1 = y0 + int(height)

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):
        return 0.0

    patch = rgb[
        y0:y1,
        x0:x1
    ]

    if patch.size == 0:
        return 0.0

    mask = tissue_mask(
        patch
    )

    return float(
        mask.mean()
    )


# ============================================================
# LOAD WSI
# ============================================================

def open_case_slides(case):

    slides = {}

    for scanner in SCANNERS:

        path = (
            DATA_ROOT /
            case /
            filename_for(
                case,
                scanner
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        slide = openslide.OpenSlide(
            str(path)
        )

        slides[scanner] = {
            "slide": slide,
            "path": path,
            "dimensions": slide.dimensions,
            "mpp": get_mpp(
                slide,
                scanner
            ),
        }

    return slides


# ============================================================
# ROTATION HELPERS
# ============================================================

def inverse_rotation(
    x,
    y,
    original_width,
    original_height,
    rotation
):

    # Coordinates are in the rotated thumbnail.
    # Return coordinates in original moving thumbnail.

    if rotation == 0:

        return x, y

    if rotation == 90:

        return (
            y,
            original_height - 1 - x
        )

    if rotation == 180:

        return (
            original_width - 1 - x,
            original_height - 1 - y
        )

    if rotation == 270:

        return (
            original_width - 1 - y,
            x
        )

    raise ValueError(rotation)


# ============================================================
# HOMOGRAPHY MAPPING
# ============================================================

def apply_homography(
    H,
    x,
    y
):

    point = np.array(
        [
            [x, y]
        ],
        dtype=np.float64
    ).reshape(
        1,
        1,
        2
    )

    mapped = cv2.perspectiveTransform(
        point,
        H
    )

    return (
        float(mapped[0, 0, 0]),
        float(mapped[0, 0, 1])
    )


# ============================================================
# THUMBNAIL → LEVEL 0
# ============================================================

def thumb_to_level0(
    x,
    y,
    thumb_width,
    thumb_height,
    level0_width,
    level0_height
):

    return (
        x * level0_width / thumb_width,
        y * level0_height / thumb_height,
    )


# ============================================================
# LEVEL 0 PATCH SIZE
# ============================================================

def native_patch_size(mpp):

    size = int(
        round(
            PATCH_UM / mpp
        )
    )

    return size


# ============================================================
# EXTRACT NATIVE PATCH
# ============================================================

def read_native_patch(
    slide,
    center_x,
    center_y,
    size
):

    x = int(
        round(
            center_x - size / 2
        )
    )

    y = int(
        round(
            center_y - size / 2
        )
    )

    width, height = slide.dimensions

    if (
        x < 0
        or y < 0
        or x + size > width
        or y + size > height
    ):
        return None

    rgba = slide.read_region(
        (x, y),
        0,
        (size, size)
    )

    rgb = np.asarray(
        rgba.convert("RGB")
    )

    return rgb


# ============================================================
# RESAMPLE TO 512×512
# ============================================================

def resample_patch(
    patch
):

    if (
        patch.shape[0] == OUTPUT_SIZE
        and patch.shape[1] == OUTPUT_SIZE
    ):
        return patch

    return cv2.resize(
        patch,
        (
            OUTPUT_SIZE,
            OUTPUT_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# VALIDATE MAPPED CENTER
# ============================================================

def mapped_center_for_scanner(
    ref_x,
    ref_y,
    scanner,
    registration
):

    H = registration["H"]
    rotation = registration["rotation"]

    # H maps rotated moving thumbnail → reference thumbnail.
    # Therefore invert H to map reference → rotated moving.

    H_inv = np.linalg.inv(H)

    rx, ry = apply_homography(
        H_inv,
        ref_x,
        ref_y
    )

    moving_thumb_w = registration[
        "moving_thumb_width"
    ]

    moving_thumb_h = registration[
        "moving_thumb_height"
    ]

    ox, oy = inverse_rotation(
        rx,
        ry,
        moving_thumb_w,
        moving_thumb_h,
        rotation
    )

    return ox, oy


# ============================================================
# LOAD REGISTRATIONS
# ============================================================

def load_registration(
    case,
    scanner
):

    H_path = (
        REG_ROOT /
        case /
        f"{scanner}_to_grundium_homography.npy"
    )

    if not H_path.exists():
        raise FileNotFoundError(
            H_path
        )

    H = np.load(
        H_path
    )

    summary_path = (
        REG_ROOT /
        "registration_summary_all_30.csv"
    )

    summary = pd.read_csv(
        summary_path
    )

    row = summary[
        (summary["case"] == case)
        &
        (
            summary[
                "moving_scanner"
            ] == scanner
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Registration record not found: "
            f"{case} {scanner}"
        )

    row = row.iloc[0]

    # Need the actual thumbnail dimensions used
    # during registration.
    moving_path = (
        DATA_ROOT /
        case /
        filename_for(
            case,
            scanner
        )
    )

    moving_rgb, _ = load_thumbnail(
        moving_path
    )

    return {
        "H": H,
        "rotation": int(
            row["rotation"]
        ),
        "moving_thumb_width":
            moving_rgb.shape[1],
        "moving_thumb_height":
            moving_rgb.shape[0],
    }


# ============================================================
# GENERATE REFERENCE CANDIDATES
# ============================================================

def generate_candidates(
    ref_rgb,
    ref_mpp
):

    h, w = ref_rgb.shape[:2]

    tissue = tissue_mask(
        ref_rgb
    )

    # Physical patch size in reference thumbnail pixels.
    patch_level0 = (
        PATCH_UM / ref_mpp
    )

    # We need level-0 dimensions separately,
    # so approximate thumbnail patch size here
    # from the reference physical scale.
    #
    # This is replaced below in process_case()
    # with the exact thumbnail scaling.

    return tissue


# ============================================================
# SELECT SPATIALLY DISTRIBUTED LOCATIONS
# ============================================================

def select_candidates(
    ref_rgb,
    ref_slide,
    ref_mpp,
    registrations,
    slides
):

    ref_h, ref_w = ref_rgb.shape[:2]

    level0_w, level0_h = (
        ref_slide.dimensions
    )

    # Exact physical patch size in thumbnail pixels.
    scale_x = (
        ref_w / level0_w
    )

    scale_y = (
        ref_h / level0_h
    )

    patch_native = (
        PATCH_UM / ref_mpp
    )

    patch_thumb_w = (
        patch_native * scale_x
    )

    patch_thumb_h = (
        patch_native * scale_y
    )

    # --------------------------------------------------------
    # Reference tissue mask
    # --------------------------------------------------------

    mask = tissue_mask(
        ref_rgb
    )

    integral = cv2.integral(
        mask.astype(
            np.float32
        )
    )

    half_w = patch_thumb_w / 2
    half_h = patch_thumb_h / 2

    # --------------------------------------------------------
    # Precompute the physical patch half-size expressed in
    # EACH moving scanner's thumbnail pixel space.
    # --------------------------------------------------------

    moving_half = {}

    for scanner in MOVING:

        reg = registrations[scanner]

        mpp = slides[scanner]["mpp"]

        lvl0_w, lvl0_h = slides[
            scanner
        ]["dimensions"]

        thumb_w = reg["moving_thumb_width"]
        thumb_h = reg["moving_thumb_height"]

        native = PATCH_UM / mpp

        s_x = thumb_w / lvl0_w
        s_y = thumb_h / lvl0_h

        half_moving_w = (native * s_x) / 2.0
        half_moving_h = (native * s_y) / 2.0

        moving_half[scanner] = (
            half_moving_w,
            half_moving_h,
            thumb_w,
            thumb_h,
        )

    candidates = []

    # --------------------------------------------------------
    # Candidate grid
    # --------------------------------------------------------

    for y in range(
        int(math.ceil(half_h)),
        int(
            ref_h -
            math.ceil(half_h)
        ),
        CANDIDATE_STEP
    ):

        for x in range(
            int(math.ceil(half_w)),
            int(
                ref_w -
                math.ceil(half_w)
            ),
            CANDIDATE_STEP
        ):

            x0 = int(
                round(x - half_w)
            )
            y0 = int(
                round(y - half_h)
            )

            x1 = int(
                round(x + half_w)
            )
            y1 = int(
                round(y + half_h)
            )

            x0 = max(
                0,
                x0
            )

            y0 = max(
                0,
                y0
            )

            x1 = min(
                ref_w,
                x1
            )

            y1 = min(
                ref_h,
                y1
            )

            area = (
                (x1 - x0)
                *
                (y1 - y0)
            )

            if area <= 0:
                continue

            tissue_sum = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )

            fraction = (
                tissue_sum / area
            )

            if fraction < REFERENCE_MIN_TISSUE:
                continue

            # ------------------------------------------------
            # Check that the COMPLETE physical patch fits
            # inside EVERY moving scanner.
            # ------------------------------------------------

            valid = True

            for scanner in MOVING:

                ox, oy = (
                    mapped_center_for_scanner(
                        x,
                        y,
                        scanner,
                        registrations[
                            scanner
                        ]
                    )
                )

                (
                    half_moving_w,
                    half_moving_h,
                    thumb_w,
                    thumb_h,
                ) = moving_half[scanner]

                if (
                    ox - half_moving_w < 0
                    or oy - half_moving_h < 0
                    or ox + half_moving_w > thumb_w
                    or oy + half_moving_h > thumb_h
                ):
                    valid = False
                    break

            if not valid:
                continue

            candidates.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "reference_tissue":
                        float(fraction),
                }
            )

    if len(candidates) < CANDIDATE_POOL_SIZE:
        raise RuntimeError(
            f"Only {len(candidates)} valid "
            f"reference candidates found "
            f"(need {CANDIDATE_POOL_SIZE})."
        )

    # --------------------------------------------------------
    # Spatially distributed selection
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda z:
            z["reference_tissue"],
        reverse=True
    )

    selected = []

    diagonal = math.sqrt(
        ref_w ** 2 +
        ref_h ** 2
    )

    min_distance = (
        MIN_SPACING_FRACTION *
        diagonal
    )

    # First point = highest tissue fraction
    selected.append(
        candidates[0]
    )

    remaining = candidates[1:]

    # Build a POOL larger than PATCHES_PER_CASE so that
    # candidates which fail extraction can be replaced.
    while (
        len(selected)
        < CANDIDATE_POOL_SIZE
        and remaining
    ):

        best = None
        best_score = -1

        for candidate in remaining:

            x = candidate["x"]
            y = candidate["y"]

            min_dist = min(
                math.sqrt(
                    (
                        x - s["x"]
                    ) ** 2
                    +
                    (
                        y - s["y"]
                    ) ** 2
                )
                for s in selected
            )

            score = (
                min_dist
                *
                (
                    0.5
                    +
                    0.5 *
                    candidate[
                        "reference_tissue"
                    ]
                )
            )

            if (
                min_dist >= min_distance
                and score > best_score
            ):
                best_score = score
                best = candidate

        if best is None:
            # Relax spacing only if necessary.
            for candidate in remaining:

                x = candidate["x"]
                y = candidate["y"]

                min_dist = min(
                    math.sqrt(
                        (
                            x - s["x"]
                        ) ** 2
                        +
                        (
                            y - s["y"]
                        ) ** 2
                    )
                    for s in selected
                )

                score = (
                    min_dist
                    *
                    candidate[
                        "reference_tissue"
                    ]
                )

                if score > best_score:
                    best_score = score
                    best = candidate

        selected.append(
            best
        )

        remaining.remove(
            best
        )

    # Keep a larger spatially distributed candidate pool.
    # Actual extraction will select the first 10 candidates
    # that pass complete-triplet QC.
    return selected + remaining


# ============================================================
# PROCESS CASE
# ============================================================

def process_case(case):

    print("\n")
    print("=" * 80)
    print(f"CASE: {case}")
    print("=" * 80)

    slides = open_case_slides(
        case
    )

    ref_slide = slides[
        REFERENCE
    ]["slide"]

    ref_path = slides[
        REFERENCE
    ]["path"]

    ref_mpp = slides[
        REFERENCE
    ]["mpp"]

    # --------------------------------------------------------
    # Reference thumbnail
    # --------------------------------------------------------

    ref_rgb, _ = load_thumbnail(
        ref_path
    )

    print(
        f"Reference dimensions: "
        f"{ref_slide.dimensions}"
    )

    print(
        f"Reference MPP: "
        f"{ref_mpp:.6f}"
    )

    print(
        f"Reference thumbnail: "
        f"{ref_rgb.shape[1]} × "
        f"{ref_rgb.shape[0]}"
    )

    # --------------------------------------------------------
    # Registrations
    # --------------------------------------------------------

    registrations = {}

    for scanner in MOVING:

        registrations[
            scanner
        ] = load_registration(
            case,
            scanner
        )

        print(
            f"{scanner}: "
            f"rotation="
            f"{registrations[scanner]['rotation']}°"
        )

    # --------------------------------------------------------
    # Select candidate pool
    # --------------------------------------------------------

    candidates = select_candidates(
        ref_rgb,
        ref_slide,
        ref_mpp,
        registrations,
        slides
    )

    print(
        f"Candidate pool size: "
        f"{len(candidates)}"
    )

    case_out = (
        OUT_ROOT /
        case
    )

    case_out.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    accepted_idx = 0

    # --------------------------------------------------------
    # Extract each candidate until we have accepted enough
    # complete triplets.
    # --------------------------------------------------------

    for candidate_idx, candidate in enumerate(
        candidates,
        start=1
    ):

        if accepted_idx >= PATCHES_PER_CASE:
            break

        ref_x_thumb = candidate[
            "x"
        ]

        ref_y_thumb = candidate[
            "y"
        ]

        # Reference center → level 0
        ref_x0, ref_y0 = (
            thumb_to_level0(
                ref_x_thumb,
                ref_y_thumb,
                ref_rgb.shape[1],
                ref_rgb.shape[0],
                ref_slide.dimensions[0],
                ref_slide.dimensions[1]
            )
        )

        patch_data = {}

        valid_triplet = True

        scanner_info = {}

        # ----------------------------------------------------
        # Extract all three scanners
        # ----------------------------------------------------

        for scanner in SCANNERS:

            slide = slides[
                scanner
            ]["slide"]

            mpp = slides[
                scanner
            ]["mpp"]

            native_size = native_patch_size(
                mpp
            )

            if scanner == REFERENCE:

                center_x0 = ref_x0
                center_y0 = ref_y0

            else:

                mx_thumb, my_thumb = (
                    mapped_center_for_scanner(
                        ref_x_thumb,
                        ref_y_thumb,
                        scanner,
                        registrations[
                            scanner
                        ]
                    )
                )

                moving_rgb, _ = (
                    load_thumbnail(
                        slides[
                            scanner
                        ]["path"]
                    )
                )

                center_x0, center_y0 = (
                    thumb_to_level0(
                        mx_thumb,
                        my_thumb,
                        moving_rgb.shape[1],
                        moving_rgb.shape[0],
                        slide.dimensions[0],
                        slide.dimensions[1]
                    )
                )

            patch = read_native_patch(
                slide,
                center_x0,
                center_y0,
                native_size
            )

            if patch is None:

                valid_triplet = False

                scanner_info[
                    scanner
                ] = {
                    "valid": False
                }

                continue

            patch = resample_patch(
                patch
            )

            tissue_fraction = (
                float(
                    tissue_mask(
                        patch
                    ).mean()
                )
            )

            scanner_info[
                scanner
            ] = {
                "valid": True,
                "center_x_level0":
                    center_x0,
                "center_y_level0":
                    center_y0,
                "native_patch_size":
                    native_size,
                "mpp": mpp,
                "tissue_fraction":
                    tissue_fraction,
            }

            if (
                tissue_fraction
                <
                ALL_SCANNERS_MIN_TISSUE
            ):
                valid_triplet = False

            patch_data[
                scanner
            ] = patch

        # ----------------------------------------------------
        # Reject incomplete triplet
        # ----------------------------------------------------

        if not valid_triplet:

            failed_scanners = [
                scanner
                for scanner in SCANNERS
                if scanner not in patch_data
            ]

            print(
                f"  Candidate {candidate_idx:02d}: "
                f"REJECTED | "
                f"missing={failed_scanners}"
            )

            continue

        # ----------------------------------------------------
        # Accept and renumber
        # ----------------------------------------------------

        accepted_idx += 1
        idx = accepted_idx

        filenames = {}

        for scanner in SCANNERS:

            filename = (
                f"patch_{idx:02d}_"
                f"{scanner}.png"
            )

            path = (
                case_out /
                filename
            )

            Image.fromarray(
                patch_data[
                    scanner
                ]
            ).save(
                path,
                format="PNG"
            )

            filenames[
                scanner
            ] = str(path)

        # ----------------------------------------------------
        # Manifest record
        # ----------------------------------------------------

        record = {
            "case": case,
            "patch_id": idx,
            "candidate_id": candidate_idx,

            "reference_x_thumbnail":
                ref_x_thumb,

            "reference_y_thumbnail":
                ref_y_thumb,

            "reference_x_level0":
                ref_x0,

            "reference_y_level0":
                ref_y0,

            "physical_patch_um":
                PATCH_UM,

            "output_size_px":
                OUTPUT_SIZE,

            "reference_mpp":
                ref_mpp,

            "reference_tissue_fraction":
                scanner_info[
                    "grundium"
                ][
                    "tissue_fraction"
                ],

            "grundium_x_level0":
                scanner_info[
                    "grundium"
                ][
                    "center_x_level0"
                ],

            "grundium_y_level0":
                scanner_info[
                    "grundium"
                ][
                    "center_y_level0"
                ],

            "hamamatsu_x_level0":
                scanner_info[
                    "hamamatsu"
                ][
                    "center_x_level0"
                ],

            "hamamatsu_y_level0":
                scanner_info[
                    "hamamatsu"
                ][
                    "center_y_level0"
                ],

            "leica_x_level0":
                scanner_info[
                    "leica"
                ][
                    "center_x_level0"
                ],

            "leica_y_level0":
                scanner_info[
                    "leica"
                ][
                    "center_y_level0"
                ],

            "grundium_mpp":
                scanner_info[
                    "grundium"
                ][
                    "mpp"
                ],

            "hamamatsu_mpp":
                scanner_info[
                    "hamamatsu"
                ][
                    "mpp"
                ],

            "leica_mpp":
                scanner_info[
                    "leica"
                ][
                    "mpp"
                ],

            "grundium_tissue_fraction":
                scanner_info[
                    "grundium"
                ][
                    "tissue_fraction"
                ],

            "hamamatsu_tissue_fraction":
                scanner_info[
                    "hamamatsu"
                ][
                    "tissue_fraction"
                ],

            "leica_tissue_fraction":
                scanner_info[
                    "leica"
                ][
                    "tissue_fraction"
                ],

            "grundium_file":
                filenames[
                    "grundium"
                ],

            "hamamatsu_file":
                filenames[
                    "hamamatsu"
                ],

            "leica_file":
                filenames[
                    "leica"
                ],

            "valid_triplet":
                True,
        }

        records.append(
            record
        )

        print(
            f"  Candidate {candidate_idx:02d}: "
            f"PASS (patch_{idx:02d}) | "
            f"G={scanner_info['grundium']['tissue_fraction']:.3f} "
            f"H={scanner_info['hamamatsu']['tissue_fraction']:.3f} "
            f"L={scanner_info['leica']['tissue_fraction']:.3f}"
        )

    # --------------------------------------------------------
    # Close slides
    # --------------------------------------------------------

    for scanner in SCANNERS:

        slides[
            scanner
        ]["slide"].close()

    print(
        f"\nCASE RESULT: "
        f"{len(records)}/{PATCHES_PER_CASE} "
        f"complete triplets"
    )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 80)
    print("30-SLIDE PAIRED PHYSICAL-PATCH EXTRACTION")
    print("=" * 80)

    print(
        f"Cases: {len(COHORT)}"
    )

    print(
        f"Target: "
        f"{PATCHES_PER_CASE} locations/case"
    )

    print(
        f"Candidate pool: "
        f"{CANDIDATE_POOL_SIZE}/case"
    )

    print(
        f"Physical patch: "
        f"{PATCH_UM} µm × {PATCH_UM} µm"
    )

    print(
        f"Output resolution: "
        f"{TARGET_MPP} µm/pixel"
    )

    print(
        f"Output image: "
        f"{OUTPUT_SIZE} × {OUTPUT_SIZE}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    print("=" * 80)

    all_records = []
    case_summary = []

    for i, case in enumerate(
        COHORT,
        start=1
    ):

        print(
            f"\n[{i}/{len(COHORT)}]"
        )

        records = process_case(
            case
        )

        all_records.extend(
            records
        )

        case_summary.append(
            {
                "case": case,
                "expected_triplets":
                    PATCHES_PER_CASE,
                "successful_triplets":
                    len(records),
                "status":
                    (
                        "PASS"
                        if len(records)
                        ==
                        PATCHES_PER_CASE
                        else
                        "FAIL"
                    ),
            }
        )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        all_records
    )

    summary = pd.DataFrame(
        case_summary
    )

    manifest.to_csv(
        MANIFEST_OUT,
        index=False
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False
    )

    # --------------------------------------------------------
    # Final QC
    # --------------------------------------------------------

    expected = (
        len(COHORT)
        *
        PATCHES_PER_CASE
    )

    actual = len(
        manifest
    )

    print("\n")
    print("=" * 80)
    print("PATCH EXTRACTION SUMMARY")
    print("=" * 80)

    print(
        f"Expected paired locations: "
        f"{expected}"
    )

    print(
        f"Successful paired locations: "
        f"{actual}"
    )

    print(
        f"Expected images: "
        f"{expected * 3}"
    )

    print(
        f"Actual images: "
        f"{actual * 3}"
    )

    complete_cases = int(
        (
            summary["successful_triplets"]
            == PATCHES_PER_CASE
        ).sum()
    )

    print(
        f"Complete cases: "
        f"{complete_cases}/{len(COHORT)}"
    )

    print("\nCase summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nManifest:")
    print(
        MANIFEST_OUT
    )

    print("\nSummary:")
    print(
        SUMMARY_OUT
    )

    print("=" * 80)

    if (
        actual != expected
        or
        (
            summary[
                "successful_triplets"
            ]
            != PATCHES_PER_CASE
        ).any()
    ):

        raise RuntimeError(
            "PATCH EXTRACTION QC FAILED. "
            "Do not proceed to UNI2-h."
        )

    print(
        "\nALL 900 PAIRED PATCHES PASSED QC."
    )


if __name__ == "__main__":
    main()