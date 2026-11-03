
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(r"D:\ISBI_2027")

PROFILE_FILE = (
    BASE / "outputs" / "label_blind_scanner_pair_audit_30" /
    "label_blind_case_profile_30.csv"
)
LABEL_FILE = BASE / "annotations" / "slide_labels.tsv"

OUT = BASE / "outputs" / "corrected_incremental_evaluation_30"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20271016
N_BOOT = 20000
ALPHA = 10.0

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------
# Load case-level profile and reader labels
# ------------------------------------------------------------
profile = pd.read_csv(PROFILE_FILE)
profile["case_id"] = (
    profile["case_id"].astype(str).str.strip().str.upper()
)

labels = pd.read_csv(LABEL_FILE, sep="\t", dtype=str)
labels = labels.apply(
    lambda col: col.str.strip() if col.dtype == "object" else col
)
labels["case_id"] = (
    labels["slide_id"].astype(str).str.strip().str.upper()
)

isup_cols = ["isup_1", "isup_r2", "isup_r3"]
for col in isup_cols:
    labels[col] = pd.to_numeric(labels[col], errors="coerce")

labels["reader_ambiguity"] = (
    labels[isup_cols].max(axis=1) -
    labels[isup_cols].min(axis=1)
)
labels["mean_isup"] = labels[isup_cols].mean(axis=1)

labels = labels[
    ["case_id", "reader_ambiguity", "mean_isup"]
].drop_duplicates("case_id")

df = profile.merge(
    labels,
    on="case_id",
    how="inner",
    validate="one_to_one",
).sort_values("case_id").reset_index(drop=True)

if len(df) != 30:
    raise ValueError(f"Expected 30 cases; found {len(df)}")

# Confirm required measurements exist.
required = [
    "mean_distance_HL",
    "mean_directional_GH",
    "mean_directional_GL",
    "mean_directional_HL",
    "reader_ambiguity",
    "mean_isup",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\nAvailable: {list(df.columns)}"
    )

# Genuine one-feature overall directional baseline.
df["overall_directional_mean"] = df[
    [
        "mean_directional_GH",
        "mean_directional_GL",
        "mean_directional_HL",
    ]
].mean(axis=1)

# ------------------------------------------------------------
# Four feature sets
# ------------------------------------------------------------
feature_sets = {
    "A_HL_distance": [
        "mean_distance_HL",
    ],
    "B_overall_directional_mean": [
        "overall_directional_mean",
    ],
    "C_three_pair_profile": [
        "mean_directional_GH",
        "mean_directional_GL",
        "mean_directional_HL",
    ],
    "D_profile_plus_mean_grade": [
        "mean_directional_GH",
        "mean_directional_GL",
        "mean_directional_HL",
        "mean_isup",
    ],
}

y = df["reader_ambiguity"].to_numpy(dtype=float)
n = len(df)

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])

# ------------------------------------------------------------
# Manual leave-one-case-out evaluation
# All learned preprocessing stays inside the training fold.
# The constant predictor also uses training cases only.
# ------------------------------------------------------------
prediction_rows = []

for model_name, features in feature_sets.items():
    X = df[features].to_numpy(dtype=float)

    for test_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False

        model = make_model()
        model.fit(X[train_mask], y[train_mask])

        pred = float(model.predict(X[[test_idx]])[0])
        train_mean_pred = float(np.mean(y[train_mask]))

        prediction_rows.append({
            "model": model_name,
            "case_id": df.loc[test_idx, "case_id"],
            "observed_ambiguity": y[test_idx],
            "prediction": pred,
            "absolute_error": abs(pred - y[test_idx]),
            "training_mean_prediction": train_mean_pred,
            "constant_absolute_error": abs(
                train_mean_pred - y[test_idx]
            ),
        })

preds = pd.DataFrame(prediction_rows)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
metric_rows = []

for model_name, group in preds.groupby("model", sort=False):
    observed = group["observed_ambiguity"].to_numpy(float)
    predicted = group["prediction"].to_numpy(float)

    # Every model has the same cases and targets.
    const_pred = group["training_mean_prediction"].to_numpy(float)

    mae = mean_absolute_error(observed, predicted)
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    r2 = r2_score(observed, predicted)
    const_mae = mean_absolute_error(observed, const_pred)

    metric_rows.append({
        "model": model_name,
        "features": ";".join(feature_sets[model_name]),
        "n_cases": len(group),
        "LOOCV_MAE": mae,
        "LOOCV_RMSE": rmse,
        "LOOCV_R2": r2,
        "fold_specific_constant_MAE": const_mae,
        "MAE_improvement_vs_constant": const_mae - mae,
    })

metrics = pd.DataFrame(metric_rows)

# ------------------------------------------------------------
# Paired case bootstrap for MAE differences
# Positive delta = model A has higher MAE (worse) than model B.
# This quantifies uncertainty in this fixed set of OOF errors.
# It does not fully account for model refitting uncertainty.
# ------------------------------------------------------------
error_wide = preds.pivot(
    index="case_id",
    columns="model",
    values="absolute_error",
).reindex(df["case_id"])

comparisons = [
    ("A_HL_distance", "B_overall_directional_mean"),
    ("A_HL_distance", "C_three_pair_profile"),
    ("B_overall_directional_mean", "C_three_pair_profile"),
    ("A_HL_distance", "D_profile_plus_mean_grade"),
]

difference_rows = []

for model_a, model_b in comparisons:
    ea = error_wide[model_a].to_numpy(float)
    eb = error_wide[model_b].to_numpy(float)
    d = ea - eb

    indices = rng.integers(0, n, size=(N_BOOT, n))
    boot_d = d[indices].mean(axis=1)

    difference_rows.append({
        "comparison": f"{model_a} minus {model_b}",
        "mean_MAE_difference": float(d.mean()),
        "bootstrap_ci_low": float(np.quantile(boot_d, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot_d, 0.975)),
        "fraction_bootstrap_difference_positive": float(
            np.mean(boot_d > 0)
        ),
        "n_cases": n,
    })

differences = pd.DataFrame(difference_rows)

# ------------------------------------------------------------
# Save and print
# ------------------------------------------------------------
metrics.to_csv(OUT / "corrected_loocv_metrics.csv", index=False)
preds.to_csv(OUT / "corrected_loocv_predictions.csv", index=False)
differences.to_csv(OUT / "paired_mae_bootstrap_comparisons.csv", index=False)

print("=" * 76)
print("CORRECTED INCREMENTAL EVALUATION")
print("=" * 76)
print(f"Cases: {n}")
print(f"Ridge alpha fixed before evaluation: {ALPHA}")
print("Validation: leave-one-case-out")
print("Constant predictor: training-fold mean only")

print("\nMODEL METRICS")
print(metrics.to_string(index=False))

print("\nPAIRED MAE DIFFERENCES")
print("Positive difference means the first model has higher/worse MAE.")
print(differences.to_string(index=False))

print("\nSaved outputs:")
for path in sorted(OUT.glob("*.csv")):
    print(path)

print(
    "\nCAUTION: the 30-case cohort was stratified by reader ambiguity. "
    "These out-of-fold estimates remain exploratory and are not "
    "independent population validation."
)
