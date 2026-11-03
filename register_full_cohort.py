from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import openslide
from PIL import Image
import json
import traceback


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"D:\ISBI_2027\data")

OUTPUT_ROOT = Path(
    r"D:\ISBI_2027\outputs\registration_all"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_ROOT / "registration_summary_all_30.csv"


# Reference scanner
REFERENCE_SCANNER = "grundium"

# Moving scanners
MOVING_SCANNERS = [
    "hamamatsu",
    "leica",
]

# Thumbnail configuration
THUMBNAIL_MAX_DIM = 1400

# SIFT configuration
SIFT_NFEATURES = 8000
SIFT_CONTRAST_THRESHOLD = 0.02

# Matching
LOWE_RATIO = 0.75

# RANSAC
RANSAC_THRESHOLD = 5.0

# Rotations to test
ROTATIONS = [0, 90, 180, 270]


# ============================================================
# EXPECTED COHORT
# ============================================================

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


# ============================================================
# FILE NAMES
# ============================================================

def get_filename(case, scanner):

    case = case.lower()

    if scanner == "grundium":
        return f"{case}_grundium.svs"

    if scanner == "hamamatsu":
        return f"{case}_hamamatsu.ndpi"

    if scanner == "leica":
        return f"{case}_leica.svs"

    raise ValueError(scanner)


# ============================================================
# LOAD THUMBNAIL
# ============================================================

def load_thumbnail(path):

    slide = openslide.OpenSlide(str(path))

    width, height = slide.dimensions

    scale = min(
        THUMBNAIL_MAX_DIM / width,
        THUMBNAIL_MAX_DIM / height,
        1.0
    )

    thumb_width = max(1, int(round(width * scale)))
    thumb_height = max(1, int(round(height * scale)))

    thumbnail = slide.get_thumbnail(
        (thumb_width, thumb_height)
    )

    slide.close()

    thumbnail = np.array(thumbnail)

    if thumbnail.ndim == 2:
        gray = thumbnail

    else:
        if thumbnail.shape[2] == 4:
            thumbnail = thumbnail[:, :, :3]

        gray = cv2.cvtColor(
            thumbnail,
            cv2.COLOR_RGB2GRAY
        )

    return thumbnail, gray


# ============================================================
# ROTATION
# ============================================================

def rotate_image(image, rotation):

    if rotation == 0:
        return image

    if rotation == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if rotation == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    if rotation == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    raise ValueError(rotation)


# ============================================================
# ROTATE POINTS
# ============================================================

def rotate_points(points, width, height, rotation):

    pts = np.asarray(points).reshape(-1, 2)

    if rotation == 0:

        return pts

    if rotation == 90:

        x = pts[:, 0]
        y = pts[:, 1]

        new_x = height - 1 - y
        new_y = x

        return np.column_stack([
            new_x,
            new_y
        ])

    if rotation == 180:

        x = pts[:, 0]
        y = pts[:, 1]

        new_x = width - 1 - x
        new_y = height - 1 - y

        return np.column_stack([
            new_x,
            new_y
        ])

    if rotation == 270:

        x = pts[:, 0]
        y = pts[:, 1]

        new_x = y
        new_y = width - 1 - x

        return np.column_stack([
            new_x,
            new_y
        ])

    raise ValueError(rotation)


# ============================================================
# SIFT REGISTRATION
# ============================================================

def register_pair(
    reference_rgb,
    reference_gray,
    moving_rgb,
    moving_gray
):

    sift = cv2.SIFT_create(
        nfeatures=SIFT_NFEATURES,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD
    )

    kp_ref, des_ref = sift.detectAndCompute(
        reference_gray,
        None
    )

    kp_mov_original, des_mov_original = (
        sift.detectAndCompute(
            moving_gray,
            None
        )
    )

    if des_ref is None or des_mov_original is None:
        raise RuntimeError(
            "SIFT descriptor extraction failed."
        )

    best = None

    # --------------------------------------------------------
    # Test all rotations
    # --------------------------------------------------------

    for rotation in ROTATIONS:

        rotated_gray = rotate_image(
            moving_gray,
            rotation
        )

        # Recompute descriptors after rotation.
        kp_mov, des_mov = sift.detectAndCompute(
            rotated_gray,
            None
        )

        if des_mov is None:
            continue

        matcher = cv2.BFMatcher(
            cv2.NORM_L2
        )

        knn_matches = matcher.knnMatch(
            des_mov,
            des_ref,
            k=2
        )

        good_matches = []

        for pair in knn_matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            continue

        src_pts = np.float32([
            kp_mov[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp_ref[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            RANSAC_THRESHOLD
        )

        if H is None or mask is None:
            continue

        inlier_mask = mask.ravel().astype(bool)

        inliers = int(
            np.sum(inlier_mask)
        )

        total_good = len(good_matches)

        inlier_ratio = (
            inliers / total_good
            if total_good > 0
            else 0.0
        )

        candidate = {
            "rotation": rotation,
            "H": H,
            "good_matches": total_good,
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "kp_ref": len(kp_ref),
            "kp_mov": len(kp_mov),
            "kp_mov_original": len(kp_mov_original),
            "src_pts": src_pts,
            "dst_pts": dst_pts,
            "mask": inlier_mask,
        }

        # ----------------------------------------------------
        # Select using inlier count first, then ratio
        # ----------------------------------------------------

        if best is None:

            best = candidate

        else:

            if (
                candidate["inliers"],
                candidate["inlier_ratio"]
            ) > (
                best["inliers"],
                best["inlier_ratio"]
            ):
                best = candidate

    if best is None:
        raise RuntimeError(
            "No valid homography found."
        )

    return best


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    reference_rgb,
    moving_rgb,
    H
):

    h_ref, w_ref = reference_rgb.shape[:2]

    warped = cv2.warpPerspective(
        moving_rgb,
        H,
        (w_ref, h_ref)
    )

    # Convert RGB → BGR for OpenCV
    ref_bgr = cv2.cvtColor(
        reference_rgb,
        cv2.COLOR_RGB2BGR
    )

    warped_bgr = cv2.cvtColor(
        warped,
        cv2.COLOR_RGB2BGR
    )

    # Checker-style blend
    overlay = cv2.addWeighted(
        ref_bgr,
        0.5,
        warped_bgr,
        0.5,
        0
    )

    return overlay


# ============================================================
# SAVE MATCH VISUALIZATION
# ============================================================

def create_match_visualization(
    reference_rgb,
    moving_rgb,
    result
):

    src_pts = result["src_pts"]
    dst_pts = result["dst_pts"]
    mask = result["mask"]

    # Draw only inliers
    draw_mask = [
        1 if x else 0
        for x in mask
    ]

    vis = cv2.drawMatches(
        moving_rgb,
        [],
        reference_rgb,
        [],
        [],
        None
    )

    # We deliberately save the overlay separately.
    # Match drawing is not reliable without original keypoints,
    # so this function is retained only as a placeholder.

    return vis


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(case):

    print("\n")
    print("=" * 80)
    print(f"CASE: {case}")
    print("=" * 80)

    case_dir = DATA_ROOT / case

    output_case = OUTPUT_ROOT / case
    output_case.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    reference_path = (
        case_dir /
        get_filename(case, REFERENCE_SCANNER)
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            reference_path
        )

    # --------------------------------------------------------
    # Load reference
    # --------------------------------------------------------

    print(
        f"Loading reference: "
        f"{reference_path.name}"
    )

    ref_rgb, ref_gray = load_thumbnail(
        reference_path
    )

    print(
        f"Reference thumbnail: "
        f"{ref_rgb.shape[1]} × {ref_rgb.shape[0]}"
    )

    case_results = []

    # --------------------------------------------------------
    # Moving scanners
    # --------------------------------------------------------

    for scanner in MOVING_SCANNERS:

        moving_path = (
            case_dir /
            get_filename(case, scanner)
        )

        print("\n" + "-" * 70)

        print(
            f"{scanner.upper()} → "
            f"{REFERENCE_SCANNER.upper()}"
        )

        result_record = {
            "case": case,
            "reference_scanner": REFERENCE_SCANNER,
            "moving_scanner": scanner,
            "reference_file": str(
                reference_path
            ),
            "moving_file": str(
                moving_path
            ),
            "rotation": None,
            "keypoints_reference": None,
            "keypoints_moving": None,
            "good_matches": None,
            "ransac_inliers": None,
            "inlier_ratio": None,
            "status": "FAIL",
            "homography": None,
            "error": None,
        }

        try:

            if not moving_path.exists():
                raise FileNotFoundError(
                    moving_path
                )

            moving_rgb, moving_gray = (
                load_thumbnail(
                    moving_path
                )
            )

            print(
                f"Moving thumbnail: "
                f"{moving_rgb.shape[1]} × "
                f"{moving_rgb.shape[0]}"
            )

            # ------------------------------------------------
            # Registration
            # ------------------------------------------------

            result = register_pair(
                ref_rgb,
                ref_gray,
                moving_rgb,
                moving_gray
            )

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            result_record[
                "rotation"
            ] = result["rotation"]

            result_record[
                "keypoints_reference"
            ] = result["kp_ref"]

            result_record[
                "keypoints_moving"
            ] = result["kp_mov"]

            result_record[
                "good_matches"
            ] = result["good_matches"]

            result_record[
                "ransac_inliers"
            ] = result["inliers"]

            result_record[
                "inlier_ratio"
            ] = result["inlier_ratio"]

            result_record[
                "homography"
            ] = json.dumps(
                result["H"].tolist()
            )

            result_record[
                "status"
            ] = "PASS"

            # ------------------------------------------------
            # Save homography
            # ------------------------------------------------

            H_path = (
                output_case /
                f"{scanner}_to_grundium_homography.npy"
            )

            np.save(
                H_path,
                result["H"]
            )

            # ------------------------------------------------
            # Rotate moving image for visualization
            # ------------------------------------------------

            moving_rotated = rotate_image(
                moving_rgb,
                result["rotation"]
            )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            overlay = create_overlay(
                ref_rgb,
                moving_rotated,
                result["H"]
            )

            overlay_path = (
                output_case /
                f"{scanner}_to_grundium_overlay.png"
            )

            cv2.imwrite(
                str(overlay_path),
                overlay
            )

            # ------------------------------------------------
            # Save rotated thumbnail
            # ------------------------------------------------

            rotated_path = (
                output_case /
                f"{scanner}_rotated_thumbnail.png"
            )

            cv2.imwrite(
                str(rotated_path),
                cv2.cvtColor(
                    moving_rotated,
                    cv2.COLOR_RGB2BGR
                )
            )

            print(
                f"Rotation: "
                f"{result['rotation']}°"
            )

            print(
                f"Good matches: "
                f"{result['good_matches']}"
            )

            print(
                f"RANSAC inliers: "
                f"{result['inliers']}"
            )

            print(
                f"Inlier ratio: "
                f"{result['inlier_ratio']:.6f}"
            )

            print("STATUS: PASS")

        except Exception as e:

            result_record[
                "error"
            ] = str(e)

            print(
                f"STATUS: FAIL"
            )

            print(
                f"ERROR: {e}"
            )

            traceback.print_exc()

        case_results.append(
            result_record
        )

    return case_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FULL 30-SLIDE / 3-SCANNER REGISTRATION")
    print("=" * 80)

    print(
        f"Reference scanner: "
        f"{REFERENCE_SCANNER}"
    )

    print(
        f"Moving scanners: "
        f"{MOVING_SCANNERS}"
    )

    print(
        f"Cases: "
        f"{len(COHORT)}"
    )

    print(
        f"Expected registrations: "
        f"{len(COHORT) * len(MOVING_SCANNERS)}"
    )

    print("=" * 80)

    all_results = []

    for i, case in enumerate(COHORT, start=1):

        print(
            f"\n[{i}/{len(COHORT)}]"
        )

        results = process_case(case)

        all_results.extend(
            results
        )

    # --------------------------------------------------------
    # Save master CSV
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_results
    )

    df.to_csv(
        SUMMARY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("REGISTRATION SUMMARY")
    print("=" * 80)

    print(
        f"Expected registrations: "
        f"{len(COHORT) * 2}"
    )

    print(
        f"Completed rows: "
        f"{len(df)}"
    )

    print(
        f"PASS: "
        f"{(df['status'] == 'PASS').sum()}"
    )

    print(
        f"FAIL: "
        f"{(df['status'] == 'FAIL').sum()}"
    )

    if len(df):

        passed = df[
            df["status"] == "PASS"
        ]

        if len(passed):

            print("\n")
            print("Inlier ratio statistics:")

            print(
                passed[
                    "inlier_ratio"
                ].describe()
            )

            print("\n")
            print("Inlier statistics:")

            print(
                passed[
                    "ransac_inliers"
                ].describe()
            )

            print("\n")
            print("Good-match statistics:")

            print(
                passed[
                    "good_matches"
                ].describe()
            )

            print("\n")
            print("Scanner-pair summary:")

            print(
                passed.groupby(
                    "moving_scanner"
                )[
                    [
                        "good_matches",
                        "ransac_inliers",
                        "inlier_ratio"
                    ]
                ].agg(
                    [
                        "count",
                        "mean",
                        "median",
                        "min",
                        "max"
                    ]
                )
            )

            print("\n")
            print("Rotation distribution:")

            print(
                passed[
                    "rotation"
                ].value_counts().sort_index()
            )

    print("\n")
    print("Master report:")
    print(SUMMARY_CSV)

    print("=" * 80)


if __name__ == "__main__":
    main()