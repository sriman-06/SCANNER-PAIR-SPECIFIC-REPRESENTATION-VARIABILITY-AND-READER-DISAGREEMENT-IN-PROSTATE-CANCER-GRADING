# Scanner-Pair-Specific Representation Variability and Reader Disagreement in Prostate Cancer Grading

This repository contains the experimental code for the ISBI 2027 study investigating the association between scanner-specific variability in UNI2-h representations and reader-level grading ambiguity in prostate biopsy whole-slide images (WSIs).

## Study Overview

The study analyzes 30 selected prostate biopsy slides acquired using three whole-slide scanners:

- Grundium
- Hamamatsu
- Leica

Three independent readers assigned ISUP grade groups. The study evaluates scanner-pair-specific embedding variability and its association with reader disagreement.

## Experimental Pipeline

1. **WSI registration:** Global SIFT feature matching and RANSAC homography estimation, using Grundium as the reference scanner.
2. **Matched patch extraction:** Extraction of 10 matched tissue locations per slide across the three scanners.
3. **Feature extraction:** Generation of 1536-dimensional UNI2-h embeddings.
4. **Representation analysis:** Decomposition of squared embedding distances into magnitude-related and magnitude-weighted directional components.
5. **Statistical analysis:** Evaluation of GH, GL, and HL scanner-pair associations with reader-level grading ambiguity.
6. **Patient-level analysis:** Analysis across 27 dataset-derived patient groups, including adjusted associations, bootstrap comparisons, and sensitivity assessments.

## Main Scripts

| Script | Description |
|---|---|
| `register_full_cohort.py` | Whole-slide image registration |
| `extract_paired_patches_30.py` | Matched patch extraction |
| `extract_uni2h_900.py` | UNI2-h embedding extraction |
| `analyze_scanner_variability_30.py` | Scanner variability analysis |
| `representation_decomposition_audit_30.py` | Embedding distance decomposition |
| `scanner_pair_directional_robustness_30.py` | Scanner-pair robustness analysis |
| `scanner_pair_directional_robustness_30_publication_figure.py` | Publication figure generation |
| `patient_level_aggregation_audit_30.py` | Patient-level aggregation |
| `patient_adjusted_audit_27.py` | Adjusted patient-level analysis |
| `patient_incremental_value_audit_27.py` | Incremental-value analysis |
| `patient_leave_one_out_audit_30.py` | Leave-one-patient-out sensitivity |
| `corrected_share_bootstrap_audit.py` | Corrected directional-share bootstrap |
| `corrected_incremental_evaluation_30.py` | Corrected incremental evaluation |

## Dataset

The study uses the publicly available PAR prostate biopsy dataset.

- Accession: S-BIAD2323
- Dataset DOI: `10.6019/S-BIAD2323`
- Publication DOI: `10.1038/s41597-026-07798-9`

The dataset is not included in this repository. Whole-slide images, annotations, extracted patches, embeddings, and intermediate outputs are excluded.

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and torchvision installations may need to be adapted to the available CUDA version and hardware.

## Reproducibility

The scripts require access to the corresponding dataset, UNI2-h model weights, and the appropriate Python environment.

File paths and environment-specific settings may need to be configured before execution. The scripts are provided as experimental research code; complete reproduction requires the relevant input data and dependencies.

## Citation

If you use this code, please cite the associated ISBI 2027 paper:

**Scanner-Pair-Specific Representation Variability and Reader Disagreement in Prostate Cancer Grading.**

## License

A license has not yet been assigned to this repository.