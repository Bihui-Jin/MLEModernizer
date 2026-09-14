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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
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

0.9347764514003416

# 6. Current score

0.75928

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails immediately because it depends on external prediction CSVs (`../input/rcsiimpreds/...`) that do not exist in your provided Kaggle filesystem, so none of the downstream merges can run and no submission is produced. To keep the “ensemble merge + average/geometric-mean” core logic intact while making it runnable, I replace those missing inputs with a minimal, legitimate fallback that reads the competition’s `sample_submission.csv` and uses it as a baseline prediction vector (then applies the same gmean/mean aggregation pattern on duplicated columns). I also make the data-path resolution robust to either `/kaggle/input/...` or `/kaggle/data/...` layouts, and ensure the final `submission.csv` has exactly the required columns and row alignment with `test.csv`. This run end-to-end and yield a valid `.csv` submission (score won’t reach the original ensemble target without the missing model files, but it be valid and stable).'
- What this solution (achieved 0.66789) has done: 'Your current 0.5 AUC comes from submitting (effectively) constant predictions because the external model prediction CSVs are missing and you fall back to `sample_submission.csv`, which is all zeros. To move toward the target AUC with minimal change and without altering the ensemble “merge + gmean/mean” semantics, I replace that fallback vector with a legitimate, lightweight metadata-based probability model trained on `train.csv` (using the same available columns as `test.csv`). This keeps the rest of your pipeline intact (same merges, same gmean aggregation), but ensures the base predictions contain signal so the final submission can exceed random performance. I also keep paths robust and ensure the submission row alignment strictly follows `test.csv`.'
- What this solution (achieved 0.77211) has done: 'Your current pipeline builds many “ensemble” columns that are all identical copies of the same metadata logistic-regression prediction, so the final mean/gmean doesn’t add any new signal and AUC stalls around ~0.67. To move your score upward toward the 0.9348 target while preserving the same overall approach (metadata model → create multiple prediction columns → merge → mean/gmean), the smallest effective change is to (1) add a few strong, still-test-available metadata features (`patient_id` and missingness indicators) and (2) train the logistic regression with `class_weight='balanced'` to better handle the heavy class imbalance. These tweaks keep the same model family and pipeline semantics, but typically yield a sizable AUC jump for this competition’s metadata-only baseline. The submission writing and row alignment with `test.csv` are kept strict and unchanged.'
- What this solution (achieved 0.76392) has done: 'Your current AUC (0.77211) is far below the target (0.93478), so we should improve the underlying signal while keeping the same “metadata model → duplicate into many columns → mean/gmean ensemble → submission.csv” core logic unchanged. The smallest, high-impact fix is to prevent patient-level leakage by using a `GroupShuffleSplit` on `patient_id` and then train the same logistic-regression pipeline with `C` selected from a tiny grid using that grouped validation AUC; this typically improves generalization and Kaggle AUC for this competition’s metadata baseline. I also add `StandardScaler` for numeric metadata features (still the same model family/architecture) to stabilize LR optimization and calibration. The rest of your merging/aggregation code remains intact, and the script still writes a valid `submission.csv` with the required columns and correct row alignment.'
- What this solution (achieved 0.67119) has done: 'Your current score (0.76392) is far below the target (0.93478), so we should improve the underlying signal while keeping the same metadata-logistic-regression → duplicated “ensemble columns” → mean/gmean aggregation semantics intact. The biggest win that doesn’t change the modeling family is to reduce overfitting from `patient_id` one-hot by hashing it into numeric features (still using the same LogisticRegression pipeline), while preserving grouped validation and the rest of the merge/gmean logic. I also slightly expand the tiny `C` grid (same selection method) so regularization can land in a better range for the hashed features. All file paths and the submission writing remain unchanged, and the script still produces a valid `submission.csv`.'
- What this solution (achieved 0.75737) has done: 'Your current score (0.67119) is far below the target (0.93478), and the main issue is that every “ensemble” column is still the exact same metadata prediction, so the mean/gmean adds no new signal. To move upward while preserving the same core approach (metadata logistic regression → create multiple prediction columns → merge → mean/gmean), I add patient-level aggregated metadata features computed on train and merged into both train/test (using only columns available at test time and grouped splitting to avoid leakage across patients). I also slightly widen the existing tiny `C` grid so the same LogisticRegression can regularize better with the added numeric features, without changing the model family or training loop structure. All merges and the submission-writing logic remain intact, and the script still produces `submission.csv` with the required columns and correct alignment.'
- What this solution (achieved 0.75841) has done: 'Your current “ensemble” columns are all identical copies of the same meta-model prediction, so the mean/gmean aggregation cannot add signal and AUC stalls. To move the score upward toward the 0.9348 target while preserving the same pipeline shape (metadata logistic regression → multiple prediction columns → merge → mean/gmean → submission), I keep your existing model and features, but train a few *slightly different* logistic-regression variants (different `C` and `class_weight`) and feed those distinct probabilities into the existing ensemble columns. This is a minimal change that adds real diversity without changing the core approach or introducing new data sources. I also keep the row alignment strictly on `test.csv` and ensure `submission.csv` is always produced with the required columns.'
- What this solution (achieved 0.75697) has done: 'Your current score (0.75841) is well below the target (0.93478), so we should improve generalization with minimal disruption to your existing “metadata LR → create multiple prediction columns → merge → mean/gmean ensemble → submission.csv” pipeline. The biggest low-risk gain here is to stop training/predicting on raw `age_approx` (which is noisy and inconsistently distributed) and instead add a simple, train-derived risk encoding for `age_approx` and `anatom_site_general_challenge` (out-of-fold on train, then fitted on full train for test) while keeping the same LogisticRegression family and the same ensemble merge/gmean logic. This adds strong signal without using images and without changing the downstream aggregation semantics. I also keep the group split by `patient_id` intact and keep all paths/output format unchanged.'
- What this solution (achieved 0.76348) has done: 'Your current score (0.75697) is far below the target (0.93478), so we should add a bit more real signal while keeping your core pipeline intact (metadata LR → multiple variants → mean/gmean ensemble → submission.csv). The smallest high-impact change is to make the out-of-fold target encodings *patient-group aware* by using `GroupKFold` (your current `GroupShuffleSplit` can repeat the same patient in validation across splits and is noisier), and to add one more simple, test-available encoding: a smoothed target encoding for `sex`. I keep the same LogisticRegression model family, the same grid-search-on-one-split, the same ensemble column construction and gmean aggregation, and the same output format/path. These changes typically improve ranking quality (AUC) without changing the overall approach.'
- What this solution (achieved 0.75927) has done: 'Your current score (0.76348) is far below the target (0.93478), so we should improve the underlying *ranking signal* while keeping the same metadata→LogisticRegression→multi-variant→gmean/mean ensemble pipeline intact. The biggest low-risk gain available within your constraints is to make the categorical target encodings *strictly out-of-fold* and *group-aware* (by patient) and to avoid using the full train to compute encodings that the model then trains on (subtle overfit/miscalibration). Concretely, I replace the single-pass OOF encoding with a “fit OOF encodings on each fold → train LR on that fold’s training split → predict that fold’s validation” scheme for model selection, then refit on full train with full-data encodings for test predictions (same model family, same feature set, same ensemble/gmean). This typically improves generalization AUC without changing architecture or adding data, and it stays fast enough for the 600s limit.'
- What this solution (achieved 0.75856) has done: 'Your current score (0.75927) is far below the target (0.93478), so we should increase AUC with the smallest changes that preserve your existing “metadata LogisticRegression variants → merge → gmean ensemble → submission.csv” core logic. The biggest issue is that your current “OOF encoding” is not actually OOF (you assign training rows using encodings fit on the same rows), which can select a bad `C` and hurt generalization; I replace it with true group-aware OOF encodings built via `GroupKFold` while keeping the same encodings/features. Next, I make the `C` selection more stable by evaluating on true OOF predictions (still same LR family/training semantics), then refit on full train and keep your same multi-variant prediction + gmean aggregation. All paths remain unchanged and the script still write a valid `submission.csv` with correct alignment to `test.csv`.'
- What this solution (achieved 0.75928) has done: 'Your current score (0.75856) is far below the target (0.93478), so we should increase AUC with minimal, low-risk changes while preserving your core “metadata LogisticRegression variants → merge → gmean ensemble → submission.csv” pipeline. The biggest fix is to align training with the competition’s patient-level structure by using a group-aware cross-validation scheme not only for TE creation (already group-aware) but also to generate diverse, better-calibrated ensemble members via out-of-fold training for each variant, then averaging those fold models at test time (same model family and prediction semantics, just less overfit and more robust). I also add `solver`-compatible parameters (`n_jobs` is not used by lbfgs) and increase `max_iter` slightly to avoid rare convergence issues that can hurt ranking. Finally, I keep your existing merge/gmean logic and ensure the submission stays strictly aligned to `test.csv` with the required columns.'

# 9. Code solution

## === cell 0
"""
submit of only B3 B4 & B5 models
"""

import os
import numpy as np
import pandas as pd




## === cell 1
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = _first_existing(
    [
        "/kaggle/input/siim-isic-melanoma-classification",
        "/kaggle/data/siim-isic-melanoma-classification",
        "/kaggle/input",
        "/kaggle/data",
    ]
)

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input/data directory in expected paths."
    )

TEST_CSV = _first_existing(
    [
        os.path.join(BASE_DIR, "test.csv"),
        os.path.join(BASE_DIR, "siim-isic-melanoma-classification", "test.csv"),
    ]
)
TRAIN_CSV = _first_existing(
    [
        os.path.join(BASE_DIR, "train.csv"),
        os.path.join(BASE_DIR, "siim-isic-melanoma-classification", "train.csv"),
    ]
)

SAMPLE_SUB = _first_existing(
    [
        os.path.join(BASE_DIR, "sample_submission.csv"),
        os.path.join(
            BASE_DIR, "siim-isic-melanoma-classification", "sample_submission.csv"
        ),
    ]
)

if TEST_CSV is None or TRAIN_CSV is None or SAMPLE_SUB is None:
    raise FileNotFoundError(
        f"Missing required files. TRAIN_CSV={TRAIN_CSV}, TEST_CSV={TEST_CSV}, SAMPLE_SUB={SAMPLE_SUB}"
    )



## === cell 2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _stable_hash_to_unit_interval(arr, salt="pid_v1"):
    arr = pd.Series(arr).astype(str).fillna("NA")
    h = (
        pd.util.hash_pandas_object(arr + "_" + salt, index=False)
        .astype(np.uint64)
        .values
    )
    return (h % np.uint64(1_000_000)).astype(np.float64) / 1_000_000.0


def _normalize_str(s):
    s = s.astype(str)
    s = s.where(~s.isna(), "NA")
    s = s.str.strip()
    s = s.replace({"": "NA", "nan": "NA", "None": "NA"})
    return s


def _fit_enc_map(train_df, col, smoothing, global_mean):
    stats = train_df.groupby(col)["target"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
        stats["count"] + smoothing
    )
    return enc


def _apply_enc_map(series, enc_map, global_mean):
    return series.map(enc_map).fillna(global_mean).astype(float).values


def _add_group_oof_te(train_df, test_df, group_arr, col, smoothing=50.0, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    global_mean = float(train_df["target"].mean())

    oof = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in gkf.split(
        train_df, train_df["target"].values, groups=group_arr
    ):
        enc_map = _fit_enc_map(
            train_df.iloc[tr_idx], col=col, smoothing=smoothing, global_mean=global_mean
        )
        oof[va_idx] = _apply_enc_map(train_df.iloc[va_idx][col], enc_map, global_mean)

    enc_full = _fit_enc_map(
        train_df, col=col, smoothing=smoothing, global_mean=global_mean
    )
    test_enc = _apply_enc_map(test_df[col], enc_full, global_mean)
    return oof, test_enc


train_cols = ["patient_id", "sex", "age_approx", "anatom_site_general_challenge"]
train_df_full = pd.read_csv(TRAIN_CSV, usecols=["target"] + train_cols)
test_df_full = pd.read_csv(TEST_CSV, usecols=["image_name"] + train_cols)

for df in (train_df_full, test_df_full):
    df["age_missing"] = df["age_approx"].isna().astype(int)
    df["sex_missing"] = (
        df["sex"].isna() | (df["sex"].astype(str).str.strip() == "")
    ).astype(int)
    df["site_missing"] = (
        df["anatom_site_general_challenge"].isna()
        | (df["anatom_site_general_challenge"].astype(str).str.strip() == "")
    ).astype(int)

train_df_full["patient_hash"] = _stable_hash_to_unit_interval(
    train_df_full["patient_id"], salt="pid"
)
train_df_full["patient_hash2"] = _stable_hash_to_unit_interval(
    train_df_full["patient_id"], salt="pid2"
)
test_df_full["patient_hash"] = _stable_hash_to_unit_interval(
    test_df_full["patient_id"], salt="pid"
)
test_df_full["patient_hash2"] = _stable_hash_to_unit_interval(
    test_df_full["patient_id"], salt="pid2"
)

train_meta = train_df_full.copy()
test_meta = test_df_full.copy()
train_meta["sex_norm"] = _normalize_str(train_meta["sex"])
test_meta["sex_norm"] = _normalize_str(test_meta["sex"])
train_meta["site_norm"] = _normalize_str(train_meta["anatom_site_general_challenge"])
test_meta["site_norm"] = _normalize_str(test_meta["anatom_site_general_challenge"])

pid_counts = (
    train_meta.groupby("patient_id").size().rename("pid_image_count").reset_index()
)

pid_age = (
    train_meta.groupby("patient_id")["age_approx"]
    .agg(["mean", "median", "min", "max", "count"])
    .rename(
        columns={
            "mean": "pid_age_mean",
            "median": "pid_age_median",
            "min": "pid_age_min",
            "max": "pid_age_max",
            "count": "pid_age_nonnull_count",
        }
    )
    .reset_index()
)

pid_site_nuniq = (
    train_meta.groupby("patient_id")["site_norm"]
    .nunique(dropna=False)
    .rename("pid_site_nunique")
    .reset_index()
)

pid_agg = pid_counts.merge(pid_age, on="patient_id", how="left").merge(
    pid_site_nuniq, on="patient_id", how="left"
)

train_df_full = train_df_full.merge(pid_agg, on="patient_id", how="left")
test_df_full = test_df_full.merge(pid_agg, on="patient_id", how="left")

agg_cols = [
    "pid_image_count",
    "pid_age_mean",
    "pid_age_median",
    "pid_age_min",
    "pid_age_max",
    "pid_age_nonnull_count",
    "pid_site_nunique",
]
for c in agg_cols:
    med = train_df_full[c].median()
    train_df_full[c] = train_df_full[c].fillna(med)
    test_df_full[c] = test_df_full[c].fillna(med)

groups = train_df_full["patient_id"].astype(str).fillna("NA").values

train_df_full["age_bin"] = pd.cut(
    train_df_full["age_approx"],
    bins=[0, 30, 45, 55, 65, 75, 90, 120],
    include_lowest=True,
)
test_df_full["age_bin"] = pd.cut(
    test_df_full["age_approx"],
    bins=[0, 30, 45, 55, 65, 75, 90, 120],
    include_lowest=True,
)

train_df_full["site_norm2"] = _normalize_str(
    train_df_full["anatom_site_general_challenge"]
)
test_df_full["site_norm2"] = _normalize_str(
    test_df_full["anatom_site_general_challenge"]
)

train_df_full["sex_norm2"] = _normalize_str(train_df_full["sex"])
test_df_full["sex_norm2"] = _normalize_str(test_df_full["sex"])

train_df_full["te_age_bin"], test_df_full["te_age_bin"] = _add_group_oof_te(
    train_df_full.assign(age_bin=train_df_full["age_bin"].astype(str)),
    test_df_full.assign(age_bin=test_df_full["age_bin"].astype(str)),
    group_arr=groups,
    col="age_bin",
    smoothing=50.0,
    n_splits=5,
)
train_df_full["te_site"], test_df_full["te_site"] = _add_group_oof_te(
    train_df_full.assign(site_norm2=train_df_full["site_norm2"].astype(str)),
    test_df_full.assign(site_norm2=test_df_full["site_norm2"].astype(str)),
    group_arr=groups,
    col="site_norm2",
    smoothing=50.0,
    n_splits=5,
)
train_df_full["te_sex"], test_df_full["te_sex"] = _add_group_oof_te(
    train_df_full.assign(sex_norm2=train_df_full["sex_norm2"].astype(str)),
    test_df_full.assign(sex_norm2=test_df_full["sex_norm2"].astype(str)),
    group_arr=groups,
    col="sex_norm2",
    smoothing=50.0,
    n_splits=5,
)

model_cols = [
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "age_missing",
    "sex_missing",
    "site_missing",
    "patient_hash",
    "patient_hash2",
    "te_age_bin",
    "te_site",
    "te_sex",
] + agg_cols

X_train = train_df_full[model_cols].copy()
y_train = train_df_full["target"].astype(int).values
X_test = test_df_full[model_cols].copy()

numeric_features = [
    "age_approx",
    "age_missing",
    "sex_missing",
    "site_missing",
    "patient_hash",
    "patient_hash2",
    "te_age_bin",
    "te_site",
    "te_sex",
] + agg_cols
categorical_features = ["sex", "anatom_site_general_challenge"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)


def _oof_auc_for_C(C, class_weight="balanced", n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(X_train), dtype=float)

    for tr_idx, va_idx in gkf.split(X_train, y_train, groups=groups):
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1200,
            C=C,
            class_weight=class_weight,
        )
        model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_pred[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]

    return roc_auc_score(y_train, oof_pred)


best_auc = -np.inf
best_C = 1.0
for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
    auc = _oof_auc_for_C(C, class_weight="balanced", n_splits=5)
    if auc > best_auc:
        best_auc = auc
        best_C = C


def _fit_predict_variant_cv_ensemble(C, class_weight, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    p_test = np.zeros(len(X_test), dtype=float)

    for tr_idx, _ in gkf.split(X_train, y_train, groups=groups):
        clf_v = LogisticRegression(
            solver="lbfgs",
            max_iter=1200,
            C=C,
            class_weight=class_weight,
        )
        model_v = Pipeline(steps=[("preprocess", preprocess), ("clf", clf_v)])
        model_v.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        p_test += model_v.predict_proba(X_test)[:, 1].astype(float) / n_splits

    return np.clip(p_test, 1e-7, 1 - 1e-7)


proba_best = _fit_predict_variant_cv_ensemble(best_C, "balanced", n_splits=5)
proba_hiC = _fit_predict_variant_cv_ensemble(
    min(30.0, best_C * 3.0), "balanced", n_splits=5
)
proba_loC = _fit_predict_variant_cv_ensemble(
    max(0.01, best_C / 3.0), "balanced", n_splits=5
)
proba_nobal = _fit_predict_variant_cv_ensemble(best_C, None, n_splits=5)

test_df = test_df_full[["image_name"]].copy()

base_best = test_df.copy()
base_best["target"] = proba_best

base_hiC = test_df.copy()
base_hiC["target"] = proba_hiC

base_loC = test_df.copy()
base_loC["target"] = proba_loC

base_nobal = test_df.copy()
base_nobal["target"] = proba_nobal

sample_sub = pd.read_csv(SAMPLE_SUB)



## === cell 3
pred_b3 = base_loC.rename(columns={"target": "target"}).copy()
pred_b4 = base_best.rename(columns={"target": "target"}).copy()
pred_b5 = base_hiC.rename(columns={"target": "target"}).copy()
pred_b6 = base_nobal.rename(columns={"target": "target"}).copy()

pred_cw_b4 = base_best.rename(columns={"target": "target_cw_b4"}).copy()



## === cell 4
pred_512_B6 = base_hiC.rename(columns={"target": "target_B6_512"}).copy()



## === cell 5
pred_tta_b3 = base_loC.rename(columns={"target": "target_tta_b3"}).copy()
pred_tta_b4 = base_best.rename(columns={"target": "target_tta_b4"}).copy()

result_tta = pd.merge(
    pred_tta_b3, pred_tta_b4, on="image_name", suffixes=("_tta_b3", "_tta_b4")
)
result_tta.head()



## === cell 6
pass



## === cell 7
pass



## === cell 8
result1 = pd.merge(pred_b3, pred_b4, on="image_name", suffixes=("_b3", "_b4"))



## === cell 9
result1.head()



## === cell 10
result2 = pd.merge(pred_b5, pred_b6, on="image_name", suffixes=("_b5", "_b6"))



## === cell 11
result2.head()



## === cell 12
semi_final = pd.merge(result1, result2, on="image_name")



## === cell 13
semi_final.head()



## === cell 14
result3 = pd.merge(
    pred_cw_b4, pred_512_B6, on="image_name", suffixes=("_cw_b4", "_B6_512")
)
result3.head()



## === cell 15
final = pd.merge(semi_final, result3, on="image_name")
final.head()



## === cell 16
final = pd.merge(final, result_tta, on="image_name")
final.head()



## === cell 17
pred_kr_b3 = base_loC.rename(columns={"target": "target_kr_b3"}).copy()
pred_kr_b4 = base_best.rename(columns={"target": "target_kr_b4"}).copy()
pred_kr_eb3 = base_hiC.rename(columns={"target": "target_kr_eb3"}).copy()



## === cell 18
kr_result = pd.merge(pred_kr_b3, pred_kr_b4, on="image_name", suffixes=("_b3", "_b4"))
kr_result = pd.merge(kr_result, pred_kr_eb3, on="image_name", suffixes=("_b3", "_b4"))
kr_result.head()



## === cell 19
final = pd.merge(final, kr_result, on="image_name")
final.head()



## === cell 20
pred_256_b4 = base_nobal.rename(columns={"target": "target_256_b4"}).copy()
pred_256_b4.head()



## === cell 21
final = pd.merge(final, pred_256_b4, on="image_name")
final.head()



## === cell 22
final["target"] = (
    (final["target_b4"])
    + (final["target_b6"])
    + final["target_tta_b4"]
    + final["target_kr_b3"]
    + final["target_kr_b4"]
    + final["target_kr_eb3"]
    + final["target_256_b4"]
) / 7.0
final.head()



## === cell 23
from scipy.stats import gmean



## === cell 24
cols_for_gmean = [
    "target_b4",
    "target_b6",
    "target_tta_b4",
    "target_B6_512",
    "target_kr_b3",
    "target_kr_b4",
    "target_kr_eb3",
    "target_256_b4",
]
for c in cols_for_gmean:
    final[c] = final[c].astype(float).clip(1e-7, 1 - 1e-7)

target_array = np.vstack([final[c].values for c in cols_for_gmean])
final["target"] = gmean(target_array, axis=0)
final.head()



## === cell 25
final.head()



## === cell 26
submit_file = test_df.merge(
    final[["image_name", "target"]], on="image_name", how="left"
)
if submit_file["target"].isna().any():
    submit_file["target"] = submit_file["target"].fillna(submit_file["target"].median())

submit_file = submit_file[["image_name", "target"]]



## === cell 27
submit_file.head()



## === cell 28
submit_file.to_csv("submission.csv", index=False)



## === cell 29
try:
    ax = submit_file["target"].hist(bins=50)
except Exception:
    pass
