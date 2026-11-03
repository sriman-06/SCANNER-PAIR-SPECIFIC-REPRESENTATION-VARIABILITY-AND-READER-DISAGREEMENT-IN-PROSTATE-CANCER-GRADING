\# Scanner-Pair-Specific Representation Variability and Reader Disagreement in Prostate Cancer Grading



This repository contains the experimental code for the ISBI 2027 study investigating the association between scanner-specific variability in UNI2-h representations and reader-level grading ambiguity in prostate biopsy whole-slide images.



\## Study Overview



The study analyzes 30 selected prostate biopsy slides acquired using three whole-slide scanners:



\- Grundium

\- Hamamatsu

\- Leica



Three independent readers assigned ISUP grade groups. The study evaluates scanner-pair-specific embedding variability and its association with reader disagreement.



\## Experimental Pipeline



1\. \*\*WSI registration:\*\* Global SIFT feature matching and RANSAC homography estimation.

2\. \*\*Matched patch extraction:\*\* Extraction of 10 matched tissue locations per slide across the three scanners.

3\. \*\*Feature extraction:\*\* Generation of 1536-dimensional UNI2-h embeddings.

4\. \*\*Representation analysis:\*\* Decomposition of squared embedding distances into magnitude-related and magnitude-weighted directional components.

5\. \*\*Statistical analysis:\*\* Scanner-pair-specific associations with reader-level grading ambiguity.

6\. \*\*Patient-level analysis:\*\* Aggregation across 27 dataset-derived patient groups, adjusted analyses, and sensitivity assessments.



\## Main Scripts



| Script | Description |

|---|---|

| `register\_full\_cohort.py` | Whole-slide image registration |

| `extract\_paired\_patches\_30.py` | Matched patch extraction |

| `extract\_uni2h\_900.py` | UNI2-h embedding extraction |

| `analyze\_scanner\_variability\_30.py` | Scanner variability analysis |

| `representation\_decomposition\_audit\_30.py` | Embedding distance decomposition |

| `scanner\_pair\_directional\_robustness\_30.py` | Scanner-pair robustness analysis |

| `scanner\_pair\_directional\_robustness\_30\_publication\_figure.py` | Publication figure generation |

| `patient\_level\_aggregation\_audit\_30.py` | Patient-level aggregation |

| `patient\_adjusted\_audit\_27.py` | Adjusted patient-level analysis |

| `patient\_incremental\_value\_audit\_27.py` | Incremental-value analysis |

| `patient\_leave\_one\_out\_audit\_30.py` | Leave-one-patient-out sensitivity |

| `corrected\_share\_bootstrap\_audit.py` | Corrected directional-share bootstrap |

| `corrected\_incremental\_evaluation\_30.py` | Corrected incremental evaluation |



\## Dataset



The study uses the publicly available PAR prostate biopsy dataset.



\- Accession: S-BIAD2323

\- DOI: 10.6019/S-BIAD2323

\- Dataset publication DOI: 10.1038/s41597-026-07798-9



Whole-slide images, annotations, extracted patches, embeddings, and intermediate outputs are not included in this repository.



\## Reproducibility



The scripts require access to the corresponding dataset, model weights, and the appropriate Python environment. File paths and environment-specific settings may need to be configured before execution.



The repository contains experimental scripts; complete reproduction requires the relevant input data and dependencies.



\## Citation



If you use this code, please cite the associated ISBI 2027 paper.



\*\*Paper:\*\* Scanner-Pair-Specific Representation Variability and Reader Disagreement in Prostate Cancer Grading.