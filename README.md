# Scanner-Pair-Specific Representation Variability and Reader Disagreement in Prostate Cancer Grading

This repository contains the experimental code for the ISBI 2027 study investigating the association between scanner-specific variability in UNI2-h representations and reader-level grading ambiguity in prostate biopsy whole-slide images (WSIs).

## Study Overview

The study analyzes 30 selected prostate biopsy slides acquired using three whole-slide scanners:

- Grundium
- Hamamatsu
- Leica

Three independent readers assigned ISUP grade groups. The study evaluates scanner-pair-specific embedding variability and its association with reader disagreement.

The cohort contains 30 slides from 27 dataset-derived patient groups. Patient groups were inferred from the PAR filename convention and were not independently verified against clinical identities.

## Experimental Pipeline

1. **WSI registration:** Global SIFT feature matching and RANSAC homography estimation, using Grundium as the reference scanner.
2. **Matched patch extraction:** Extraction of 10 matched tissue locations per slide across the three scanners.
3. **Feature extraction:** Generation of 1536-dimensional UNI2-h embeddings.
4. **Representation analysis:** Decomposition of squared embedding distances into magnitude-related and magnitude-weighted directional components.
5. **Scanner-pair analysis:** Evaluation of GH, GL, and HL associations with reader-level grading ambiguity.
6. **Patient-level analysis:** Analysis across 27 dataset-derived patient groups, including adjusted associations, bootstrap comparisons, and sensitivity assessments.

## Repository Structure

The scripts expect the following directory structure:

```text
ISBI_2027/
├── annotations/
│   └── slide_labels.tsv
├── data/
│   └── [PAR whole-slide images]
├── outputs/
│   ├── registration_all/
│   ├── paired_patches_30/
│   ├── uni2h_30/
│   └── [analysis output directories]
├── register_full_cohort.py
├── extract_paired_patches_30.py
├── extract_uni2h_900.py
├── analyze_scanner_variability_30.py
├── representation_decomposition_audit_30.py
├── scanner_pair_directional_robustness_30.py
├── scanner_pair_directional_robustness_30_publication_figure.py
├── patient_level_aggregation_audit_30.py
├── patient_adjusted_audit_27.py
├── patient_incremental_value_audit_27.py
├── patient_leave_one_out_audit_30.py
├── corrected_share_bootstrap_audit.py
├── corrected_incremental_evaluation_30.py
├── requirements.txt
└── README.md
```

The actual dataset and annotation filenames must match those expected by the scripts.

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

The dataset is not included in this repository. Whole-slide images, annotations, extracted patches, embeddings, model weights, and intermediate outputs are excluded.

Users must obtain the dataset separately and organize the files according to the expected scanner-specific naming convention.

## Installation

Use Python 3.10 and create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the repository dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch and torchvision must also be installed with compatible versions for the available operating system and CUDA configuration. Select the appropriate installation command from the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Verify the installation:

```python
import torch
import timm
import openslide

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("timm:", timm.__version__)
print("OpenSlide:", openslide.__version__)
```

## Path Configuration

The scripts support environment variables so that users can configure dataset and output locations without modifying the source code.

The default paths are relative to the repository directory.

| Environment variable | Purpose |
|---|---|
| `ISBI_DATA_ROOT` | WSI dataset directory |
| `ISBI_OUTPUT_ROOT` | Registration output directory |
| `ISBI_REG_ROOT` | Registration results used for patch extraction |
| `ISBI_PATCH_OUTPUT_ROOT` | Matched patch output directory |
| `ISBI_PATCH_ROOT` | Matched patch input directory for embedding extraction |
| `ISBI_EMBED_OUTPUT_ROOT` | UNI2-h embedding directory |
| `ISBI_EMBED_OUTPUT_ROOT` | Embedding input directory for downstream analyses |
| `ISBI_LABEL_PATH` | Reader annotation file |
| `ISBI_VARIABILITY_OUTPUT_ROOT` | Scanner variability analysis output |
| `ISBI_DECOMPOSITION_OUTPUT_ROOT` | Representation decomposition output |
| `ISBI_PAIR_ROBUSTNESS_OUTPUT_ROOT` | Scanner-pair robustness output |
| `ISBI_PROFILE_PATH` | Label-blind scanner profile input |
| `ISBI_PATIENT_MAPPING_PATH` | Dataset-derived patient mapping file |
| `ISBI_PATIENT_TABLE_PATH` | Patient-level aggregated table |
| `ISBI_PATIENT_SLIDE_TABLE_PATH` | Patient-level slide table |
| `ISBI_PATIENT_AGGREGATION_OUTPUT_ROOT` | Patient aggregation output |
| `ISBI_PATIENT_ADJUSTED_OUTPUT_ROOT` | Adjusted patient-level analysis output |
| `ISBI_PATIENT_INCREMENTAL_OUTPUT_ROOT` | Patient-level incremental analysis output |
| `ISBI_CORRECTED_SHARE_INPUT` | Corrected directional-share input table |
| `ISBI_FROZEN_RESULTS_ROOT` | Frozen results output directory |
| `ISBI_CORRECTED_EVALUATION_OUTPUT_ROOT` | Corrected incremental evaluation output |

For example, on Windows PowerShell:

```powershell
$env:ISBI_DATA_ROOT = "D:\ISBI_2027\data"
$env:ISBI_LABEL_PATH = "D:\ISBI_2027\annotations\slide_labels.tsv"
$env:ISBI_EMBED_OUTPUT_ROOT = "D:\ISBI_2027\outputs\uni2h_30"
```

Set the variables before running the relevant scripts. They only need to be specified when the default directory structure is not used.

## Execution Order

The main image-processing pipeline is:

```powershell
python register_full_cohort.py
python extract_paired_patches_30.py
python extract_uni2h_900.py
```

The analysis scripts can then be executed using their required input files:

```powershell
python analyze_scanner_variability_30.py
python representation_decomposition_audit_30.py
python scanner_pair_directional_robustness_30.py
python scanner_pair_directional_robustness_30_publication_figure.py
```

The patient-level analysis scripts require additional supporting tables, including the label-blind scanner profile, dataset-derived patient mapping, and patient-level aggregation table. These inputs must be generated or supplied before running the corresponding scripts.

```powershell
python patient_level_aggregation_audit_30.py
python patient_adjusted_audit_27.py
python patient_incremental_value_audit_27.py
python patient_leave_one_out_audit_30.py
python corrected_share_bootstrap_audit.py
python corrected_incremental_evaluation_30.py
```

The execution order of the patient-level scripts depends on the availability of their required input tables. Check the configured input paths before execution.

## Reproducibility

The scripts require access to the PAR dataset, the UNI2-h model weights, reader annotations, and the appropriate Python environment.

The patch extraction and embedding stages use fixed study configurations. Statistical scripts retain their specified random seeds and analysis settings.

The repository contains experimental research code. Reproduction of the reported results requires the corresponding input data, supporting patient-level tables, dependencies, and model weights.

The scripts include input-path checks, but successful syntax compilation alone does not establish that the complete pipeline has been executed or reproduced.

## Citation

If you use this code, please cite the associated ISBI 2027 paper:

**Scanner-Pair-Specific Representation Variability and Reader Disagreement in Prostate Cancer Grading.**

Dataset reference:

Muhammad Ali et al. PAR prostate biopsy dataset. Scientific Data (2026). DOI: `10.1038/s41597-026-07798-9`.

## License

A license has not yet been assigned to this repository.