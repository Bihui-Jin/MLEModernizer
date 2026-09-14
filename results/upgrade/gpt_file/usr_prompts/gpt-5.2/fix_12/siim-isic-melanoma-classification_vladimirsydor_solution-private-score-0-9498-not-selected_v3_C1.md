# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.9498

# 6. Current score

0.78013

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66782) has done: 'The current notebook fails because it tries to read a submission file from a non-existent `../input/best-private-sub/` dataset. To make it run end-to-end and produce a valid `submission.csv`, I replaced that external dependency with a small, stable baseline that trains a model using only the provided `train.csv` metadata and predicts on `test.csv`. This keeps the pipeline simple and within the available packages (pandas + scikit-learn), and guarantees correct submission columns/order by starting from `sample_submission.csv`. The resulting score won’t match a top deep-learning image model, but it generate a valid submission reliably.'
- What this solution (achieved 0.66923) has done: 'Your current score (0.66782) is far below the target (0.9498), so we should improve AUC with minimal, metric-aligned changes while keeping the same “metadata-only + LogisticRegression + GroupKFold” core logic. The biggest easy win here is removing `class_weight="balanced"`: for AUC (a ranking metric), reweighting can distort the learned ranking and often hurts compared to plain logistic regression on this dataset. I also add a tiny amount of regularization tuning (a small grid of `C` values) inside the same CV loop using AUC on the validation fold to pick `C`, which keeps the same model family/training approach but usually improves ranking noticeably. Finally, I ensure categorical missing values treat blank strings as missing (common in this dataset), which prevents “empty string” becoming its own category and can improve generalization slightly.'
- What this solution (achieved 0.78023) has done: 'We’re far below the target AUC, so the smallest legitimate way to move toward it (without changing the overall “metadata-only + LogisticRegression + GroupKFold” approach) is to add a few high-signal, still-metadata-derived features. I keep the same preprocessing pattern but extend it to include `patient_id` (as a categorical feature) and add one leakage-free aggregated feature computed from train only: per-patient lesion count (then mapped onto test), which helps ranking because patients can have multiple images. I also expand the regularization grid slightly (still the same model family and CV selection logic) to better match the optimal regularization for the richer feature set. Submission creation stays identical and still merges onto `sample_submission.csv` to guarantee correct order/format.'
- What this solution (achieved 0.77778) has done: 'We’re still far below the target AUC, so the smallest likely improvement while preserving your exact “metadata-only + LogisticRegression + GroupKFold + CV-selected C” core is to (1) add a couple of leakage-safe, metadata-derived numeric features that often improve ranking: a missingness flag for age and a simple patient-level mean target prior computed out-of-fold (so it can’t leak within each validation fold). I also switch `OneHotEncoder` to `min_frequency` grouping to reduce very-rare category noise (especially for `patient_id`) without changing the model family, and keep the same CV loop and test-time fold ensembling. These changes are directly aimed at better AUC via better ranking/calibration, and still produce the same submission format and file.'
- What this solution (achieved 0.78196) has done: 'You’re still far below the target AUC, so we should make the smallest metadata-only changes that usually improve ranking without changing the core “LogisticRegression + GroupKFold + CV-selected C + fold ensembling” approach. The biggest low-risk gain here is to avoid letting `patient_id` one-hot dominate (it can overfit fold-specific patient patterns) by moving it out of the one-hot set and instead using leakage-safe patient aggregate features. I add two out-of-fold, fold-computed aggregates derived from train only: (1) smoothed per-patient target mean and (2) logit of that mean (both filled with the fold global mean), and keep your existing patient lesion count. Finally, I slightly widen the `C` grid to include a bit stronger regularization which often helps when adding aggregated features, while leaving the overall loop and semantics unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved 0.7633) has done: 'Your current approach is already stable and fairly strong for metadata-only; to move closer to the 0.9498 target with minimal core-logic change, the biggest remaining low-risk gain is improving the *patient aggregation* signal without leakage by (1) using multiple smoothing strengths (alpha grid) selected on each fold’s validation AUC, and (2) adding a second leakage-safe aggregate: per-patient lesion count computed on the training part of each fold (not globally), plus its log transform. This keeps the same LogisticRegression + GroupKFold + per-fold hyperparameter selection + fold-ensemble structure, but makes the patient-derived features more robust and less biased. I’m also keeping the submission creation unchanged (merge onto sample_submission) to guarantee correct ordering/format. These changes should increase AUC somewhat (toward the target) without changing the model family or training semantics.'
- What this solution (achieved 0.76001) has done: 'I keep your exact metadata-only + GroupKFold + LogisticRegression + per-fold hyperparameter selection/ensembling core, but make two minimal, AUC-relevant improvements. First, I add one high-signal leakage-safe numeric feature: a smoothed per–anatom_site target prior computed from the training split of each fold (and applied to val/test for that fold), mirroring your patient prior logic. Second, I add a tiny “missingness” flag for anatom_site (blank/NaN), since missingness itself is predictive and helps ranking without changing the model family. Everything else (data loading, CV, selection over C and alpha grids, submission creation and ordering) stays the same.'
- What this solution (achieved 0.76818) has done: 'To move your AUC upward toward the target while keeping the same core “metadata-only + GroupKFold + LogisticRegression + per-fold hyperparameter selection + fold ensembling” logic, I make two minimal, metric-aligned changes. First, I standardize all numeric features inside the pipeline (including the fold-created aggregate/logit features) because LogisticRegression is sensitive to feature scaling and this often improves ranking/AUC without changing the model family. Second, I expand the `C` grid slightly to include stronger regularization values that tend to work better once scaling is applied (still the same per-fold model selection mechanism). Everything else (leakage-safe fold aggregates, CV structure, submission creation/ordering) remains unchanged.'
- What this solution (achieved 0.76817) has done: 'Your current score (0.76818) is far below the target (0.9498), so we should make a small, safe change that tends to improve AUC without changing the core “metadata + GroupKFold + LogisticRegression + per-fold hyperparameter selection + fold ensembling” logic. The biggest low-risk gain here is fixing a subtle evaluation mismatch: you’re selecting `alpha`/`site_alpha` using the *logit of the prior alone*, but the prior is ultimately used as an input feature to a multivariate logistic model, so that selection criterion can pick suboptimal smoothing. I instead select `alpha` and `site_alpha` by training the same LogisticRegression pipeline on the fold’s training split and scoring AUC on the fold’s validation split, keeping everything else (feature set, model family, C-grid selection, no leakage) the same. This preserves your approach but aligns hyperparameter choice with the actual model/metric, which should move AUC upward toward the target while remaining stable and within time.'
- What this solution (achieved 0.76946) has done: 'We’re far below the target AUC, so the smallest likely boost while preserving your exact “metadata + GroupKFold + LogisticRegression + per-fold hyperparameter selection + ensembling” core is to add one more leakage-safe aggregate that captures strong signal in this competition: patient-level sex and age context. Concretely, inside each fold we compute (from that fold’s training split only) smoothed per-patient mean age and majority sex, then map them to val/test for that fold; these features improve ranking without using images or changing the model family. I keep your existing alpha/site_alpha selection by fold AUC and your C-grid selection unchanged, only expanding the numeric feature list to include these new fold-derived aggregates (with scaling still applied). Submission writing and ordering remain identical.'
- What this solution (achieved 0.78013) has done: 'I keep your exact metadata-only + GroupKFold + LogisticRegression + per-fold hyperparameter selection + fold-ensembling structure, but fix one major generalization issue: you currently tune `alpha/site_alpha` and `C` on the same validation fold, which overfits the fold and tends to depress leaderboard AUC. I add a tiny *inner* GroupKFold (on the training part of each outer fold only) to select `(alpha, site_alpha, C)` by mean inner AUC, then fit once on the full outer-train split and predict the outer-val/test—same model family and features, but properly nested selection. This is a minimal change to the training semantics (no new model, no new features) that should move AUC upward toward your 0.9498 target. Submission writing/order remains unchanged and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_DIR_CANDIDATES:
        nested = os.path.join(base, "siim-isic-melanoma-classification", filename)
        if os.path.exists(nested):
            return nested
    raise FileNotFoundError(
        f"Could not find {filename} in known Kaggle input locations: {DATA_DIR_CANDIDATES}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "target" in train.columns, "train.csv must contain 'target'"
assert (
    "image_name" in test.columns and "image_name" in sample_sub.columns
), "Missing 'image_name' column"
assert list(sample_sub.columns) == [
    "image_name",
    "target",
], "sample_submission.csv must have columns: image_name,target"



## === cell 1
feature_cols = ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]

X = train[feature_cols].copy()
y = train["target"].astype(int).values
groups = (
    train["patient_id"].values
    if "patient_id" in train.columns
    else np.arange(len(train))
)

X_test = test[feature_cols].copy()

for col in ["sex", "anatom_site_general_challenge", "patient_id"]:
    X[col] = X[col].replace("", np.nan)
    X_test[col] = X_test[col].replace("", np.nan)

X["anatom_missing"] = X["anatom_site_general_challenge"].isna().astype(float)
X_test["anatom_missing"] = X_test["anatom_site_general_challenge"].isna().astype(float)

X["age_missing"] = X["age_approx"].isna().astype(float)
X_test["age_missing"] = X_test["age_approx"].isna().astype(float)

patient_counts = train["patient_id"].value_counts(dropna=False)
X["patient_lesion_count"] = train["patient_id"].map(patient_counts).astype(float)
X_test["patient_lesion_count"] = test["patient_id"].map(patient_counts).astype(float)
X_test["patient_lesion_count"] = X_test["patient_lesion_count"].fillna(1.0)

categorical_features = ["sex", "anatom_site_general_challenge"]


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)




## === cell 2
gkf = GroupKFold(n_splits=5)

test_pred_accum = np.zeros(len(test), dtype=float)
oof_pred = np.zeros(len(train), dtype=float)

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
ALPHA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
SITE_ALPHA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]

fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_tr_base, X_va_base = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    tr_pids = train.iloc[tr_idx]["patient_id"]
    va_pids = train.iloc[va_idx]["patient_id"]

    tr_patient_counts = tr_pids.value_counts(dropna=False)
    X_tr_base["patient_lesion_count_fold"] = tr_pids.map(tr_patient_counts).astype(
        float
    )
    X_va_base["patient_lesion_count_fold"] = va_pids.map(tr_patient_counts).astype(
        float
    )

    X_test_fold = X_test.copy()
    X_test_fold["patient_lesion_count_fold"] = (
        test["patient_id"].map(tr_patient_counts).astype(float)
    )

    X_tr_base["patient_lesion_count_fold"] = X_tr_base[
        "patient_lesion_count_fold"
    ].fillna(1.0)
    X_va_base["patient_lesion_count_fold"] = X_va_base[
        "patient_lesion_count_fold"
    ].fillna(1.0)
    X_test_fold["patient_lesion_count_fold"] = X_test_fold[
        "patient_lesion_count_fold"
    ].fillna(1.0)

    X_tr_base["patient_lesion_count_log_fold"] = np.log1p(
        X_tr_base["patient_lesion_count_fold"].values
    )
    X_va_base["patient_lesion_count_log_fold"] = np.log1p(
        X_va_base["patient_lesion_count_fold"].values
    )
    X_test_fold["patient_lesion_count_log_fold"] = np.log1p(
        X_test_fold["patient_lesion_count_fold"].values
    )

    global_mean_outer = float(np.mean(y_tr))

    patient_stats_outer = (
        pd.DataFrame({"patient_id": tr_pids.values, "y": y_tr})
        .groupby("patient_id")["y"]
        .agg(["mean", "count"])
    )

    tr_sites_outer = train.iloc[tr_idx]["anatom_site_general_challenge"].replace(
        "", np.nan
    )
    va_sites_outer = train.iloc[va_idx]["anatom_site_general_challenge"].replace(
        "", np.nan
    )
    te_sites_outer = test["anatom_site_general_challenge"].replace("", np.nan)

    site_stats_outer = (
        pd.DataFrame({"site": tr_sites_outer.values, "y": y_tr})
        .groupby("site")["y"]
        .agg(["mean", "count"])
    )

    tr_age_outer = pd.to_numeric(train.iloc[tr_idx]["age_approx"], errors="coerce")
    tr_sex_outer = train.iloc[tr_idx]["sex"].replace("", np.nan)

    pid_age_mean_outer = (
        pd.DataFrame({"patient_id": tr_pids.values, "age": tr_age_outer.values})
        .groupby("patient_id")["age"]
        .mean()
    )
    global_age_median_outer = (
        float(np.nanmedian(tr_age_outer.values))
        if np.isfinite(np.nanmedian(tr_age_outer.values))
        else float(
            np.nanmedian(pd.to_numeric(train["age_approx"], errors="coerce").values)
        )
    )

    sex_bin_tr_outer = tr_sex_outer.map({"male": 1.0, "female": 0.0})
    pid_male_rate_outer = (
        pd.DataFrame({"patient_id": tr_pids.values, "male": sex_bin_tr_outer.values})
        .groupby("patient_id")["male"]
        .mean()
    )
    global_male_rate_outer = (
        float(np.nanmean(sex_bin_tr_outer.values))
        if np.isfinite(np.nanmean(sex_bin_tr_outer.values))
        else 0.5
    )

    X_tr_base["patient_age_mean_fold"] = (
        tr_pids.map(pid_age_mean_outer).astype(float).fillna(global_age_median_outer)
    )
    X_va_base["patient_age_mean_fold"] = (
        va_pids.map(pid_age_mean_outer).astype(float).fillna(global_age_median_outer)
    )
    X_test_fold["patient_age_mean_fold"] = (
        test["patient_id"]
        .map(pid_age_mean_outer)
        .astype(float)
        .fillna(global_age_median_outer)
    )

    X_tr_base["patient_male_rate_fold"] = (
        tr_pids.map(pid_male_rate_outer).astype(float).fillna(global_male_rate_outer)
    )
    X_va_base["patient_male_rate_fold"] = (
        va_pids.map(pid_male_rate_outer).astype(float).fillna(global_male_rate_outer)
    )
    X_test_fold["patient_male_rate_fold"] = (
        test["patient_id"]
        .map(pid_male_rate_outer)
        .astype(float)
        .fillna(global_male_rate_outer)
    )

    numeric_features_fold = [
        "age_approx",
        "patient_lesion_count",
        "patient_lesion_count_fold",
        "patient_lesion_count_log_fold",
        "age_missing",
        "anatom_missing",
        "patient_age_mean_fold",
        "patient_male_rate_fold",
        "patient_target_mean",
        "patient_target_logit",
        "site_target_mean",
        "site_target_logit",
    ]

    preprocess_fold = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features_fold,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    inner_splits = 3
    unique_groups = pd.Series(tr_pids.values).nunique(dropna=False)
    inner_n_splits = min(inner_splits, int(unique_groups)) if unique_groups >= 2 else 2
    inner_n_splits = max(2, inner_n_splits)
    inner_gkf = GroupKFold(n_splits=inner_n_splits)

    def build_features_with_alphas_for_split(
        X_base_local: pd.DataFrame,
        pids_local: pd.Series,
        sites_local: pd.Series,
        patient_stats_local: pd.DataFrame,
        site_stats_local: pd.DataFrame,
        global_mean_local: float,
        alpha_patient: float,
        alpha_site: float,
    ) -> pd.DataFrame:
        smoothed_patient = (
            patient_stats_local["mean"] * patient_stats_local["count"]
            + global_mean_local * alpha_patient
        ) / (patient_stats_local["count"] + alpha_patient)

        smoothed_site = (
            site_stats_local["mean"] * site_stats_local["count"]
            + global_mean_local * alpha_site
        ) / (site_stats_local["count"] + alpha_site)

        X_out = X_base_local.copy()

        X_out["patient_target_mean"] = (
            pids_local.map(smoothed_patient).astype(float).fillna(global_mean_local)
        )
        ptm = X_out["patient_target_mean"].values
        X_out["patient_target_logit"] = np.log(
            _clip_prob(ptm) / (1.0 - _clip_prob(ptm))
        )

        X_out["site_target_mean"] = (
            sites_local.map(smoothed_site).astype(float).fillna(global_mean_local)
        )
        stm = X_out["site_target_mean"].values
        X_out["site_target_logit"] = np.log(_clip_prob(stm) / (1.0 - _clip_prob(stm)))

        return X_out

    best_params = None
    best_inner_auc = -1.0

    for alpha in ALPHA_GRID:
        for site_alpha in SITE_ALPHA_GRID:
            for C in C_GRID:
                inner_aucs = []
                for inner_tr_rel, inner_va_rel in inner_gkf.split(
                    X_tr_base, y_tr, groups=tr_pids.values
                ):
                    X_inner_tr_base = X_tr_base.iloc[inner_tr_rel].copy()
                    X_inner_va_base = X_tr_base.iloc[inner_va_rel].copy()
                    y_inner_tr = y_tr[inner_tr_rel]
                    y_inner_va = y_tr[inner_va_rel]

                    pids_inner_tr = tr_pids.iloc[inner_tr_rel]
                    pids_inner_va = tr_pids.iloc[inner_va_rel]

                    sites_inner_tr = tr_sites_outer.iloc[inner_tr_rel]
                    sites_inner_va = tr_sites_outer.iloc[inner_va_rel]

                    global_mean_inner = float(np.mean(y_inner_tr))

                    patient_stats_inner = (
                        pd.DataFrame(
                            {"patient_id": pids_inner_tr.values, "y": y_inner_tr}
                        )
                        .groupby("patient_id")["y"]
                        .agg(["mean", "count"])
                    )
                    site_stats_inner = (
                        pd.DataFrame({"site": sites_inner_tr.values, "y": y_inner_tr})
                        .groupby("site")["y"]
                        .agg(["mean", "count"])
                    )

                    X_inner_tr = build_features_with_alphas_for_split(
                        X_inner_tr_base,
                        pids_inner_tr,
                        sites_inner_tr,
                        patient_stats_inner,
                        site_stats_inner,
                        global_mean_inner,
                        alpha,
                        site_alpha,
                    )
                    X_inner_va = build_features_with_alphas_for_split(
                        X_inner_va_base,
                        pids_inner_va,
                        sites_inner_va,
                        patient_stats_inner,
                        site_stats_inner,
                        global_mean_inner,
                        alpha,
                        site_alpha,
                    )

                    model = Pipeline(
                        steps=[
                            ("preprocess", preprocess_fold),
                            (
                                "clf",
                                LogisticRegression(
                                    solver="lbfgs",
                                    max_iter=2000,
                                    C=C,
                                    random_state=RANDOM_STATE,
                                ),
                            ),
                        ]
                    )
                    model.fit(X_inner_tr, y_inner_tr)
                    pred_inner = model.predict_proba(X_inner_va)[:, 1]
                    inner_aucs.append(roc_auc_score(y_inner_va, pred_inner))

                mean_inner_auc = float(np.mean(inner_aucs)) if len(inner_aucs) else -1.0
                if mean_inner_auc > best_inner_auc:
                    best_inner_auc = mean_inner_auc
                    best_params = (alpha, site_alpha, C)

    best_alpha, best_site_alpha, best_C = best_params

    X_tr_final = build_features_with_alphas_for_split(
        X_tr_base,
        tr_pids,
        tr_sites_outer,
        patient_stats_outer,
        site_stats_outer,
        global_mean_outer,
        best_alpha,
        best_site_alpha,
    )
    X_va_final = build_features_with_alphas_for_split(
        X_va_base,
        va_pids,
        va_sites_outer,
        patient_stats_outer,
        site_stats_outer,
        global_mean_outer,
        best_alpha,
        best_site_alpha,
    )
    X_te_final = build_features_with_alphas_for_split(
        X_test_fold,
        test["patient_id"],
        te_sites_outer,
        patient_stats_outer,
        site_stats_outer,
        global_mean_outer,
        best_alpha,
        best_site_alpha,
    )

    best_model = Pipeline(
        steps=[
            ("preprocess", preprocess_fold),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    C=best_C,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    best_model.fit(X_tr_final, y_tr)

    va_pred = best_model.predict_proba(X_va_final)[:, 1]
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(fold_auc)
    oof_pred[va_idx] = va_pred
    test_pred_accum += best_model.predict_proba(X_te_final)[:, 1]

    print(
        f"Fold {fold}: best_alpha={best_alpha} best_site_alpha={best_site_alpha} best_C={best_C} "
        f"(inner_auc={best_inner_auc:.6f}) val_auc={fold_auc:.6f}"
    )

test_pred = test_pred_accum / gkf.get_n_splits()
print(
    f"OOF AUC: {roc_auc_score(y, oof_pred):.6f} (mean fold AUC={np.mean(fold_aucs):.6f})"
)



## === cell 3
sub = sample_sub.copy()
pred_df = pd.DataFrame({"image_name": test["image_name"].values, "target": test_pred})
sub = sub.drop(columns=["target"]).merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(train["target"].mean())

sub["target"] = sub["target"].clip(0.0, 1.0).astype(float)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={sub.shape} and columns={list(sub.columns)}")
print(sub.head())
