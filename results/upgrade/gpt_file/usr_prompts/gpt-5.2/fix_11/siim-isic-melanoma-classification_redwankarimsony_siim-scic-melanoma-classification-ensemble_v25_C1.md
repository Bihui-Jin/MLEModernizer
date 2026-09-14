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

0.9294022564192178

# 6. Current score

0.74466

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66776) has done: 'I remove the hard dependency on missing “public submission” CSV files (those paths don’t exist in your environment), and instead generate predictions from the provided metadata in `train.csv`/`test.csv` so the notebook runs end-to-end. To keep changes minimal and stable, I implement a lightweight preprocessing + logistic regression pipeline using only `numpy/pandas/scikit-learn`, which fits the competition metric (ROC-AUC) by producing probabilities. I also ensure the submission rows align exactly to `sample_submission.csv` by merging on `image_name`, and I always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.74421) has done: 'Your current score (0.66776) is far below the target AUC (0.9294), so we need a meaningful but still “same-core-logic” upgrade to move toward the target. We keep the exact same sklearn Pipeline + LogisticRegression approach, but add a few metadata-derived features (still from the same CSV columns) that are known to be strongly predictive in this competition: one-hot encoding of `diagnosis` and `benign_malignant` (train-only) plus simple group prevalence features by `patient_id`. To avoid leakage-like train/test inconsistency, we compute these group statistics using out-of-fold (OOF) encoding on the train set and apply full-train statistics to the test set. This preserves evaluation semantics (probability output for ROC-AUC) while giving the linear model substantially more signal, typically lifting AUC toward the desired band.'
- What this solution (achieved 0.63394) has done: 'Your current AUC (0.74421) is far below the target (0.9294), so we need a modest but meaningful upgrade without changing the core “metadata-only + sklearn logistic regression” approach. The biggest missing signal in your current setup is that you compute patient-level target stats, but you do not compute analogous high-signal encodings for other important categorical groups (notably `anatom_site_general_challenge`, and optionally `sex`) in an out-of-fold way. I add OOF mean/count target encodings for `anatom_site_general_challenge` (and `sex`), computed safely with StratifiedKFold to avoid leakage, and apply full-train stats to test—this typically provides a noticeable lift while keeping the same model, loss, and training procedure. I keep everything else (pipeline, LogisticRegression, one-hot usage, submission alignment) unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.72723) has done: 'We keep your exact “metadata-only + sklearn LogisticRegression pipeline” core intact, but fix the main reason your AUC is stuck low: you’re leaking train-only columns (`diagnosis`, `benign_malignant`) into the model even though they are missing in test, so the model can overfit to those signals yet can’t use them at inference (they become all-missing → constant), hurting ranking on the test set. The minimal, safe change is to drop any feature that is not present (non-all-missing) in both train and test, while keeping all your existing target-encoding features (patient/site/sex OOF means/counts) and the same preprocessing/model. This typically increases leaderboard AUC substantially (toward your 0.929 target) without changing the training approach, loss, or model family. We also keep the submission alignment/merge logic unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.72727) has done: 'Your current AUC (0.72723) is far below the target (0.9294), so we should improve signal while keeping the same metadata-only LogisticRegression pipeline. The smallest high-impact fix is to add a few standard, test-available metadata feature interactions and missingness indicators (e.g., `age_approx` missing flag, age-binned categories, and combined `sex x site`), which often improve ranking without changing the model family or training loop. We also add light smoothing to the existing target-encoding features (patient/site/sex mean encodings) to reduce noise for rare groups, while preserving the same OOF encoding approach and semantics. All I/O paths and the submission alignment logic remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.72618) has done: 'We fix the runtime error that stops training by ensuring `age_bin` is treated as a plain string (not a pandas Categorical) before doing target-encoding arithmetic inside the OOF loop. This allow `test_pred` to be created, unblocking the downstream submission cells. We keep the same metadata-only feature set, OOF target-encoding approach, and LogisticRegression pipeline; the changes are strictly type-handling and stability fixes. We also add a small safeguard to keep `age_approx` numeric consistently (without changing semantics) so the preprocess step is robust.'
- What this solution (achieved 0.72611) has done: 'We keep your exact metadata-only + OOF target-encoding + LogisticRegression pipeline, but make two minimal changes that typically lift ROC-AUC materially for this competition: (1) calibrate the target-encoding smoothing strength based on the dataset size (your current K=20 is often too weak and can add noise), and (2) tune LogisticRegression regularization (C) slightly to better match the expanded feature space while keeping the same solver and probability output. These changes do not alter the modeling approach, loss, training loop, or feature sources; they only adjust regularization/smoothing to improve ranking toward your target AUC. All paths and submission alignment remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.74395) has done: 'Your current AUC (0.72611) is far below the target (0.9294), so we need a real lift while keeping the same core “metadata + OOF target encoding + LogisticRegression” approach. The biggest minimal-change gain is to train the LogisticRegression on out-of-fold (OOF) predicted probabilities from the same model (trained per fold) and then fit a final LogisticRegression “blender” on those OOF scores (and optionally a couple stable numeric encodings) to improve ranking calibration without changing the model family or loss. This preserves identical semantics (still LogisticRegression probabilities), avoids leakage (OOF for train, full-train for test), and typically boosts ROC-AUC materially versus a single fit. All paths and submission-writing logic stay the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.74466) has done: 'We keep your exact metadata-only + OOF target-encoding + LogisticRegression + second-stage LogisticRegression blender, but make two small changes that typically improve ROC-AUC ranking without changing the core approach. First, we add one additional stable numeric signal (`age_approx`) into the blender inputs so the second stage can correct residual ranking errors tied to age without relying on sparse one-hot space. Second, we slightly increase the base model’s regularization strength (lower `C`) to reduce overfitting/noise from the expanded one-hot + TE features, which should move your score upward toward the 0.9294 target. All file paths, feature generation, OOF semantics, and submission alignment/writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE1 = "/kaggle/input/siim-isic-melanoma-classification"
BASE2 = "/kaggle/data/siim-isic-melanoma-classification"
BASE3 = "/kaggle/input"  # contains train.csv/test.csv in this environment description
BASE4 = "/kaggle/data"


def pick_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


train_csv = pick_path(
    os.path.join(BASE1, "train.csv"),
    os.path.join(BASE2, "train.csv"),
    os.path.join(BASE3, "train.csv"),
    os.path.join(BASE4, "train.csv"),
)
test_csv = pick_path(
    os.path.join(BASE1, "test.csv"),
    os.path.join(BASE2, "test.csv"),
    os.path.join(BASE3, "test.csv"),
    os.path.join(BASE4, "test.csv"),
)
sample_csv = pick_path(
    os.path.join(BASE1, "sample_submission.csv"),
    os.path.join(BASE2, "sample_submission.csv"),
    os.path.join(BASE3, "sample_submission.csv"),
    os.path.join(BASE4, "sample_submission.csv"),
)

if train_csv is None or test_csv is None or sample_csv is None:
    raise FileNotFoundError(
        f"Could not find required CSVs. Found train={train_csv}, test={test_csv}, sample={sample_csv}"
    )

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
sub = pd.read_csv(sample_csv)

assert "target" in train.columns, "train.csv must contain 'target'"
assert (
    "image_name" in test.columns and "image_name" in sub.columns
), "image_name must exist in test and sample_submission"



## === cell 2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

train_feat = train.copy()
test_feat = test.copy()

for col in ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]:
    if col not in train_feat.columns:
        train_feat[col] = np.nan
    if col not in test_feat.columns:
        test_feat[col] = np.nan

for col in ["diagnosis", "benign_malignant"]:
    if col in train_feat.columns and col not in test_feat.columns:
        test_feat[col] = np.nan

y_train = train_feat["target"].astype(int).values
global_mean = float(train_feat["target"].mean())


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["age_approx"] = pd.to_numeric(out["age_approx"], errors="coerce")
    out["age_missing"] = out["age_approx"].isna().astype(np.int8)

    bins = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
    labels = [
        "<=10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "80+",
    ]
    out["age_bin"] = pd.cut(out["age_approx"], bins=bins, labels=labels)
    out["age_bin"] = out["age_bin"].astype(str).replace("nan", np.nan)

    s = out["sex"].fillna("unknown").astype(str)
    site = out["anatom_site_general_challenge"].fillna("unknown").astype(str)
    out["sex_x_site"] = (s + "__" + site).astype(str)

    return out


train_feat = add_derived_features(train_feat)
test_feat = add_derived_features(test_feat)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_patient_mean = np.zeros(len(train_feat), dtype=np.float32)
oof_patient_count = np.zeros(len(train_feat), dtype=np.float32)

oof_site_mean = np.zeros(len(train_feat), dtype=np.float32)
oof_site_count = np.zeros(len(train_feat), dtype=np.float32)
oof_sex_mean = np.zeros(len(train_feat), dtype=np.float32)
oof_sex_count = np.zeros(len(train_feat), dtype=np.float32)

oof_sexsite_mean = np.zeros(len(train_feat), dtype=np.float32)
oof_sexsite_count = np.zeros(len(train_feat), dtype=np.float32)
oof_agebin_mean = np.zeros(len(train_feat), dtype=np.float32)
oof_agebin_count = np.zeros(len(train_feat), dtype=np.float32)

SMOOTH_K = 100.0

for tr_idx, va_idx in skf.split(train_feat, y_train):
    tr_fold = train_feat.iloc[tr_idx]

    pat_stats = tr_fold.groupby("patient_id")["target"].agg(["mean", "count"])
    va_pat = train_feat.iloc[va_idx]["patient_id"]
    va_pat_mean = va_pat.map(pat_stats["mean"])
    va_pat_cnt = va_pat.map(pat_stats["count"])
    va_pat_smooth = (va_pat_mean * va_pat_cnt + global_mean * SMOOTH_K) / (
        va_pat_cnt + SMOOTH_K
    )
    oof_patient_mean[va_idx] = (
        va_pat_smooth.fillna(global_mean).astype(np.float32).values
    )
    oof_patient_count[va_idx] = va_pat_cnt.fillna(0).astype(np.float32).values

    site_stats = tr_fold.groupby("anatom_site_general_challenge")["target"].agg(
        ["mean", "count"]
    )
    va_site = train_feat.iloc[va_idx]["anatom_site_general_challenge"]
    va_site_mean = va_site.map(site_stats["mean"])
    va_site_cnt = va_site.map(site_stats["count"])
    va_site_smooth = (va_site_mean * va_site_cnt + global_mean * SMOOTH_K) / (
        va_site_cnt + SMOOTH_K
    )
    oof_site_mean[va_idx] = va_site_smooth.fillna(global_mean).astype(np.float32).values
    oof_site_count[va_idx] = va_site_cnt.fillna(0).astype(np.float32).values

    sex_stats = tr_fold.groupby("sex")["target"].agg(["mean", "count"])
    va_sex = train_feat.iloc[va_idx]["sex"]
    va_sex_mean = va_sex.map(sex_stats["mean"])
    va_sex_cnt = va_sex.map(sex_stats["count"])
    va_sex_smooth = (va_sex_mean * va_sex_cnt + global_mean * SMOOTH_K) / (
        va_sex_cnt + SMOOTH_K
    )
    oof_sex_mean[va_idx] = va_sex_smooth.fillna(global_mean).astype(np.float32).values
    oof_sex_count[va_idx] = va_sex_cnt.fillna(0).astype(np.float32).values

    sexsite_stats = tr_fold.groupby("sex_x_site")["target"].agg(["mean", "count"])
    va_sexsite = train_feat.iloc[va_idx]["sex_x_site"]
    va_sexsite_mean = va_sexsite.map(sexsite_stats["mean"])
    va_sexsite_cnt = va_sexsite.map(sexsite_stats["count"])
    va_sexsite_smooth = (va_sexsite_mean * va_sexsite_cnt + global_mean * SMOOTH_K) / (
        va_sexsite_cnt + SMOOTH_K
    )
    oof_sexsite_mean[va_idx] = (
        va_sexsite_smooth.fillna(global_mean).astype(np.float32).values
    )
    oof_sexsite_count[va_idx] = va_sexsite_cnt.fillna(0).astype(np.float32).values

    agebin_stats = tr_fold.groupby("age_bin")["target"].agg(["mean", "count"])
    va_agebin = train_feat.iloc[va_idx]["age_bin"]
    va_agebin_mean = va_agebin.map(agebin_stats["mean"])
    va_agebin_cnt = va_agebin.map(agebin_stats["count"])
    va_agebin_smooth = (va_agebin_mean * va_agebin_cnt + global_mean * SMOOTH_K) / (
        va_agebin_cnt + SMOOTH_K
    )
    oof_agebin_mean[va_idx] = (
        va_agebin_smooth.fillna(global_mean).astype(np.float32).values
    )
    oof_agebin_count[va_idx] = va_agebin_cnt.fillna(0).astype(np.float32).values

full_pat = train_feat.groupby("patient_id")["target"].agg(["mean", "count"])
test_pat_mean = test_feat["patient_id"].map(full_pat["mean"])
test_pat_cnt = test_feat["patient_id"].map(full_pat["count"])
test_feat["patient_target_mean"] = (
    (
        (test_pat_mean * test_pat_cnt + global_mean * SMOOTH_K)
        / (test_pat_cnt + SMOOTH_K)
    )
    .fillna(global_mean)
    .astype(np.float32)
)
test_feat["patient_target_count"] = test_pat_cnt.fillna(0).astype(np.float32)

full_site = train_feat.groupby("anatom_site_general_challenge")["target"].agg(
    ["mean", "count"]
)
test_site_mean = test_feat["anatom_site_general_challenge"].map(full_site["mean"])
test_site_cnt = test_feat["anatom_site_general_challenge"].map(full_site["count"])
test_feat["site_target_mean"] = (
    (
        (test_site_mean * test_site_cnt + global_mean * SMOOTH_K)
        / (test_site_cnt + SMOOTH_K)
    )
    .fillna(global_mean)
    .astype(np.float32)
)
test_feat["site_target_count"] = test_site_cnt.fillna(0).astype(np.float32)

full_sex = train_feat.groupby("sex")["target"].agg(["mean", "count"])
test_sex_mean = test_feat["sex"].map(full_sex["mean"])
test_sex_cnt = test_feat["sex"].map(full_sex["count"])
test_feat["sex_target_mean"] = (
    (
        (test_sex_mean * test_sex_cnt + global_mean * SMOOTH_K)
        / (test_sex_cnt + SMOOTH_K)
    )
    .fillna(global_mean)
    .astype(np.float32)
)
test_feat["sex_target_count"] = test_sex_cnt.fillna(0).astype(np.float32)

full_sexsite = train_feat.groupby("sex_x_site")["target"].agg(["mean", "count"])
test_sexsite_mean = test_feat["sex_x_site"].map(full_sexsite["mean"])
test_sexsite_cnt = test_feat["sex_x_site"].map(full_sexsite["count"])
test_feat["sexsite_target_mean"] = (
    (
        (test_sexsite_mean * test_sexsite_cnt + global_mean * SMOOTH_K)
        / (test_sexsite_cnt + SMOOTH_K)
    )
    .fillna(global_mean)
    .astype(np.float32)
)
test_feat["sexsite_target_count"] = test_sexsite_cnt.fillna(0).astype(np.float32)

full_agebin = train_feat.groupby("age_bin")["target"].agg(["mean", "count"])
test_agebin_mean = test_feat["age_bin"].map(full_agebin["mean"])
test_agebin_cnt = test_feat["age_bin"].map(full_agebin["count"])
test_feat["agebin_target_mean"] = (
    (
        (test_agebin_mean * test_agebin_cnt + global_mean * SMOOTH_K)
        / (test_agebin_cnt + SMOOTH_K)
    )
    .fillna(global_mean)
    .astype(np.float32)
)
test_feat["agebin_target_count"] = test_agebin_cnt.fillna(0).astype(np.float32)

train_feat["patient_target_mean"] = oof_patient_mean
train_feat["patient_target_count"] = oof_patient_count
train_feat["site_target_mean"] = oof_site_mean
train_feat["site_target_count"] = oof_site_count
train_feat["sex_target_mean"] = oof_sex_mean
train_feat["sex_target_count"] = oof_sex_count
train_feat["sexsite_target_mean"] = oof_sexsite_mean
train_feat["sexsite_target_count"] = oof_sexsite_count
train_feat["agebin_target_mean"] = oof_agebin_mean
train_feat["agebin_target_count"] = oof_agebin_count

feature_cols = [
    "sex",
    "age_approx",
    "age_missing",
    "age_bin",
    "anatom_site_general_challenge",
    "sex_x_site",
    "patient_target_mean",
    "patient_target_count",
    "site_target_mean",
    "site_target_count",
    "sex_target_mean",
    "sex_target_count",
    "sexsite_target_mean",
    "sexsite_target_count",
    "agebin_target_mean",
    "agebin_target_count",
]

for col in ["diagnosis", "benign_malignant"]:
    if col in train_feat.columns and col in test_feat.columns:
        if (train_feat[col].notna().any()) and (test_feat[col].notna().any()):
            feature_cols.append(col)

X_train = train_feat[feature_cols].copy()
X_test = test_feat[feature_cols].copy()

numeric_features = [
    c
    for c in feature_cols
    if c
    in [
        "age_approx",
        "age_missing",
        "patient_target_mean",
        "patient_target_count",
        "site_target_mean",
        "site_target_count",
        "sex_target_mean",
        "sex_target_count",
        "sexsite_target_mean",
        "sexsite_target_count",
        "agebin_target_mean",
        "agebin_target_count",
    ]
]
categorical_features = [c for c in feature_cols if c not in numeric_features]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    C=0.25,
    n_jobs=None,
)

base_model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

oof_pred = np.zeros(len(X_train), dtype=np.float32)
test_pred_folds = np.zeros((len(X_test), skf.n_splits), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va = X_train.iloc[va_idx]

    m = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
    m.fit(X_tr, y_tr)

    oof_pred[va_idx] = m.predict_proba(X_va)[:, 1].astype(np.float32)
    test_pred_folds[:, fold] = m.predict_proba(X_test)[:, 1].astype(np.float32)

test_pred_base = test_pred_folds.mean(axis=1).astype(np.float32)

blend_train = pd.DataFrame(
    {
        "base_pred": oof_pred,
        "patient_te": X_train["patient_target_mean"].astype(np.float32).values,
        "site_te": X_train["site_target_mean"].astype(np.float32).values,
        "age_approx": pd.to_numeric(X_train["age_approx"], errors="coerce")
        .astype(np.float32)
        .fillna(np.float32(np.nan)),
    }
)
blend_test = pd.DataFrame(
    {
        "base_pred": test_pred_base,
        "patient_te": X_test["patient_target_mean"].astype(np.float32).values,
        "site_te": X_test["site_target_mean"].astype(np.float32).values,
        "age_approx": pd.to_numeric(X_test["age_approx"], errors="coerce")
        .astype(np.float32)
        .fillna(np.float32(np.nan)),
    }
)

blend_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
            ),
        ),
    ]
)

blend_model.fit(blend_train, y_train)
test_pred = blend_model.predict_proba(blend_test)[:, 1].astype(np.float32)



## === cell 3
pred_df = pd.DataFrame({"image_name": test["image_name"].values, "target": test_pred})

sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(float(train["target"].mean()))

sub_out["target"] = sub_out["target"].clip(0.0, 1.0)



## === cell 4
sub_out.to_csv("submission.csv", index=False)
print(sub_out.head())
print(
    f"Wrote submission.csv with shape={sub_out.shape} and columns={list(sub_out.columns)}"
)



## === cell 5
assert list(sub_out.columns) == ["image_name", "target"]
assert len(sub_out) == len(sub)
assert sub_out["target"].between(0, 1).all()
