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

0.9188839498622048

# 6. Current score

0.66768

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the dependency on missing external “public submission” CSVs (those files aren’t present in your provided `/kaggle/input` tree), which is what currently prevents any submission from being created. To keep the core behavior minimal and stable, I fall back to using the competition’s provided `sample_submission.csv` and populate it with a constant probability equal to the training-set prevalence of `target` (a simple, legitimate baseline that runs end-to-end). I also make the path handling robust to either `/kaggle/input/...` or `../input/...` layouts and add basic checks to guarantee the output has the required `image_name,target` columns and correct row count. This reliably produce a valid `submission.csv` without changing any modeling code (since none exists in the original script).'
- What this solution (achieved 0.66789) has done: 'To move your AUC up from ~0.5 toward the 0.9189 target without changing any “model architecture/training” (none exists here), the smallest legitimate step is to replace the constant prior with a real, fast tabular baseline using only the provided metadata (sex, age, site) and a leakage-safe train/validation split by `patient_id`. This keeps the overall approach simple (CSV-in/CSV-out) while producing non-constant predictions, which is necessary to exceed random AUC. I also align the encoding between train/test, fill missing values deterministically, and use out-of-fold validation AUC to sanity-check that the model is learning signal before writing `submission.csv`. All paths and the submission schema (`image_name,target`) remain unchanged and a valid `submission.csv` is always produced.'
- What this solution (achieved 0.66733) has done: 'I keep your exact metadata-only pipeline and LogisticRegression core, but make two small changes that typically improve AUC materially for this competition: (1) set `class_weight="balanced"` to counter the heavy class imbalance, and (2) slightly strengthen regularization (`C=0.3`) to reduce overfitting noise from one-hot site/sex. I also make the GroupKFold split robust by ensuring each missing `patient_id` becomes a unique group (instead of all missing collapsing into one group), which avoids leakage-like artifacts and usually yields cleaner generalization. The rest (features, preprocessing, CV loop, submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.6672) has done: 'Your current score (0.66733) is far below the target (0.91888), so we should improve AUC with minimal, safe changes while keeping the same metadata-only LogisticRegression pipeline. The biggest likely issue is underfitting: `C=0.3` is quite strong regularization for sparse one-hot features, and `lbfgs` is not the best choice for high-dimensional sparse-ish problems. I switch the solver to `liblinear` (stable for smaller tabular problems) and relax regularization back toward a reasonable default (`C=1.0`) while keeping `class_weight="balanced"`, GroupKFold by `patient_id`, and the exact same preprocessing/features. This should increase separability (and thus AUC) without changing the overall approach or submission semantics.'
- What this solution (achieved 0.66274) has done: 'We keep the same metadata-only LogisticRegression + GroupKFold-by-patient pipeline, but make two small adjustments aimed at improving separability (and thus AUC) without changing the overall approach. First, we switch the one-hot encoder to output a sparse matrix, which is a more natural fit for this kind of high-dimensional categorical expansion and works better with a linear model. Second, we move from `liblinear` to `saga` (still LogisticRegression) with a slightly higher `max_iter`, which typically optimizes sparse logistic models more effectively and should raise your score from ~0.667 toward the 0.9189 target while preserving semantics. Submission writing, columns, paths, and leakage-safe grouping remain unchanged.'
- What this solution (achieved 0.66687) has done: 'Your current AUC (0.66274) is far below the target (0.91888), so we should improve separability with the smallest changes while keeping the same metadata-only LogisticRegression + GroupKFold-by-patient pipeline. The main low-risk gain here is better handling of the extreme class imbalance by explicitly calibrating the intercept via `fit_intercept=False` and adding the log-odds prior as a fixed feature, which often improves ranking when only weak metadata is available. We also switch to `solver="lbfgs"` with a slightly higher `C` while keeping the same one-hot + imputation preprocessing; this typically converges more stably for this small feature space than `saga` and can improve AUC without changing the overall approach. Submission generation and schema remain identical and robust.'
- What this solution (achieved 0.66776) has done: 'We keep your exact metadata-only LogisticRegression + GroupKFold-by-patient pipeline, but fix two small choices that are likely suppressing AUC. First, `fit_intercept=False` combined with a constant `prior_logit` feature is redundant and can hurt ranking; we restore a normal intercept (`fit_intercept=True`) and drop the constant prior feature while keeping the same inputs (sex/age/site) and preprocessing. Second, we slightly ease regularization back toward the default (`C=1.0`) to reduce underfitting on sparse one-hot features, without changing the modeling family or evaluation semantics. Submission writing remains identical and always produces a valid `submission.csv`.'
- What this solution (achieved 0.66956) has done: 'We keep your exact metadata-only LogisticRegression + GroupKFold-by-patient pipeline, but make two minimal, score-relevant adjustments that typically improve AUC ranking without changing the overall approach. First, we add `diagnosis` as an additional categorical feature (train-only column; in test it be missing/NaN and safely handled by the existing imputer/one-hot), which gives the model extra supervised signal at train time without any leakage. Second, we add a small amount of label smoothing at prediction time via blending the model probabilities with the global prior (a calibration/ranking stabilizer); this is a tiny post-processing change that can improve generalization on this competition while preserving semantics and producing a valid submission. Everything else (paths, splitting by `patient_id`, preprocessing style, LogisticRegression, and submission writing) stays the same.'
- What this solution (achieved 0.66776) has done: 'Your current AUC (0.66956) is far below the target (0.91888), so we should improve ranking with the smallest safe changes while keeping the same metadata-only LogisticRegression + GroupKFold-by-patient pipeline. The biggest likely issue is that `diagnosis` is train-only and becomes all-missing in test, so its learned weights can act like noise at inference; we remove `diagnosis` from the feature set to reduce that mismatch while preserving the same model family and preprocessing approach. We also remove the probability blending with the global prior (post-processing that can compress score range and harm ROC ordering) and keep only clipping for numerical safety. Everything else (paths, split strategy, preprocessing blocks, LogisticRegression, and submission writing) remains the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.67153) has done: 'Your current AUC is far below the target, so we should cautiously improve ranking while keeping the same metadata-only LogisticRegression + GroupKFold-by-patient core. The smallest high-impact change here is to add the train-provided `diagnosis` column back in, but in a leakage-safe way: we compute an out-of-fold (GroupKFold) target-mean encoding per diagnosis for train, and a global-mean fallback for unseen diagnoses in test. This preserves the same model family/training loop and just adds one numeric feature with deterministic CV encoding, which typically adds meaningful signal on this competition. Everything else (paths, preprocessing, submission schema, and end-to-end execution) remains unchanged.'
- What this solution (achieved 0.67139) has done: 'I keep your metadata-only LogisticRegression + GroupKFold-by-patient pipeline intact, but fix the main weak spot: the `diagnosis` target-encoding is currently computed but not included in the model’s categorical/numeric feature definitions (so it can be underused/misaligned). I also add a tiny amount of smoothing to the diagnosis mean-encoding (within each fold and for the full-train map) to reduce overfitting/noise from rare diagnoses while preserving the same semantics and CV leakage-safety. Finally, I ensure the exact same preprocessing is applied consistently by explicitly appending the engineered `diag_te` feature to `feature_cols` and using it only as numeric input, which should improve ranking signal and move AUC upward toward your target.'
- What this solution (achieved 0.66717) has done: 'I keep your exact metadata + diagnosis target-encoding + LogisticRegression pipeline, but fix two issues that likely cap performance: (1) `diagnosis` is missing in test.csv, so your test-time `diag_te` collapses to the global mean and contributes no ranking signal; we should instead build `diag_te` from `benign_malignant` (available only in train, but it defines a stable mapping from diagnosis→malignancy label that can be applied to test diagnosis inferred from metadata is not possible, so we must drop the test-side `diag_te` to avoid noise). (2) To legitimately add more signal without changing the modeling family, we add simple frequency/count encodings for `anatom_site_general_challenge` and `sex` (computed on train and applied to test) as extra numeric features; this is minimal, fast, and often improves AUC when metadata is weak. Everything else (GroupKFold by patient, preprocessing, LogisticRegression settings, submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.66724) has done: 'Your current AUC (0.66717) is far below the target (0.91888), so we should improve ranking with minimal, legitimate changes while keeping the same metadata-only LogisticRegression + GroupKFold-by-patient pipeline. The biggest low-risk gain is to make the linear model better suited for sparse one-hot features by switching the solver to `saga` (same LogisticRegression family) and increasing `max_iter` to ensure convergence, which often improves AUC for this exact setup. I also add simple, leakage-safe interaction features between `sex` and `anatom_site_general_challenge` (one extra categorical column), which typically adds signal without changing the overall approach. Everything else (paths, CV grouping, preprocessing, submission format, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.66768) has done: 'Your score is far below the 0.9189 target, so we should improve AUC with the smallest changes that keep your exact metadata-only LogisticRegression + GroupKFold-by-patient pipeline intact. The most likely reason the model is stuck around ~0.67 is that sex/age/site alone is weak; a minimal, legitimate boost is to add a leakage-safe, out-of-fold target-mean encoding for `anatom_site_general_challenge` and `sex` (computed within each fold, then applied to validation and test). This adds just two numeric features while preserving the same model family, training loop, and submission semantics. Everything else (paths, one-hot+imputation preprocessing, solver, CV, and CSV writing) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_sub_path = _first_existing_path(
    [
        "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv",
        "/kaggle/data/siim-isic-melanoma-classification/sample_submission.csv",
        "../input/siim-isic-melanoma-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "../input/sample_submission.csv",
    ]
)
train_csv_path = _first_existing_path(
    [
        "/kaggle/input/siim-isic-melanoma-classification/train.csv",
        "/kaggle/data/siim-isic-melanoma-classification/train.csv",
        "../input/siim-isic-melanoma-classification/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
        "../input/train.csv",
    ]
)
test_csv_path = _first_existing_path(
    [
        "/kaggle/input/siim-isic-melanoma-classification/test.csv",
        "/kaggle/data/siim-isic-melanoma-classification/test.csv",
        "../input/siim-isic-melanoma-classification/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
        "../input/test.csv",
    ]
)

if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input/data paths."
    )
if train_csv_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv in expected Kaggle input/data paths."
    )
if test_csv_path is None:
    raise FileNotFoundError(
        "Could not locate test.csv in expected Kaggle input/data paths."
    )

sub = pd.read_csv(sample_sub_path)
train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)

if "image_name" not in sub.columns:
    raise ValueError(
        f"sample_submission is missing required column 'image_name'. Columns={list(sub.columns)}"
    )
if "image_name" not in test.columns:
    raise ValueError(
        f"test.csv is missing required column 'image_name'. Columns={list(test.columns)}"
    )
if "target" not in train.columns:
    raise ValueError(
        f"train.csv is missing required column 'target'. Columns={list(train.columns)}"
    )

prior = float(np.clip(train["target"].mean(), 1e-6, 1 - 1e-6))



## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

feature_cols = ["sex", "age_approx", "anatom_site_general_challenge"]
group_col = "patient_id"
target_col = "target"
diag_col = "diagnosis"

for c in feature_cols + [group_col, diag_col]:
    if c not in train.columns:
        train[c] = np.nan
    if c not in test.columns:
        test[c] = np.nan

train[diag_col] = train[diag_col].astype("object")
test[diag_col] = test[diag_col].astype("object")

X = train[feature_cols].copy()
y = train[target_col].astype(int).values
X_test = test[feature_cols].copy()


def _add_interaction_cat(X_df, X_test_df, a, b, out_col):
    X_df = X_df.copy()
    X_test_df = X_test_df.copy()
    X_df[out_col] = (
        X_df[a].astype("object").fillna("__nan__").astype(str)
        + "__"
        + X_df[b].astype("object").fillna("__nan__").astype(str)
    ).astype("object")
    X_test_df[out_col] = (
        X_test_df[a].astype("object").fillna("__nan__").astype(str)
        + "__"
        + X_test_df[b].astype("object").fillna("__nan__").astype(str)
    ).astype("object")
    return X_df, X_test_df


X, X_test = _add_interaction_cat(
    X, X_test, "sex", "anatom_site_general_challenge", "sex_x_site"
)

categorical_features = ["sex", "anatom_site_general_challenge", "sex_x_site"]

grp = train[group_col].astype("object")
missing = grp.isna()
if missing.any():
    grp.loc[missing] = "__missing__" + train.loc[missing].index.astype(str)
groups = grp.astype(str).values

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)


def _add_freq_features(X_df, X_test_df, cols):
    X_df = X_df.copy()
    X_test_df = X_test_df.copy()
    n_train = float(len(X_df))
    for c in cols:
        vc = X_df[c].astype("object").value_counts(dropna=False)
        X_df[f"{c}__freq"] = (
            X_df[c].astype("object").map(vc).fillna(0).astype(float) / n_train
        )
        X_test_df[f"{c}__freq"] = (
            X_test_df[c].astype("object").map(vc).fillna(0).astype(float) / n_train
        )
    return X_df, X_test_df


X, X_test = _add_freq_features(
    X, X_test, cols=["sex", "anatom_site_general_challenge", "sex_x_site"]
)


def _oof_target_mean_encode(
    X_df, X_test_df, y_arr, groups_arr, cols, n_splits=5, smoothing=20.0
):
    X_df = X_df.copy()
    X_test_df = X_test_df.copy()

    global_mean = float(np.mean(y_arr))
    gkf_local = GroupKFold(n_splits=n_splits)
    oof_encoded = {c: np.zeros(len(X_df), dtype=np.float64) for c in cols}

    for tr_idx, va_idx in gkf_local.split(X_df, y_arr, groups=groups_arr):
        tr = X_df.iloc[tr_idx]
        y_tr = y_arr[tr_idx]
        for c in cols:
            key_tr = tr[c].astype("object").fillna("__nan__")
            stats = pd.DataFrame({"k": key_tr.values, "y": y_tr})
            agg = stats.groupby("k")["y"].agg(["mean", "count"])
            enc_map = (
                (agg["mean"] * agg["count"] + global_mean * smoothing)
                / (agg["count"] + smoothing)
            ).to_dict()

            key_va = X_df.iloc[va_idx][c].astype("object").fillna("__nan__").values
            oof_encoded[c][va_idx] = (
                pd.Series(key_va).map(enc_map).fillna(global_mean).values
            )

    for c in cols:
        key_all = X_df[c].astype("object").fillna("__nan__")
        stats_all = pd.DataFrame({"k": key_all.values, "y": y_arr})
        agg_all = stats_all.groupby("k")["y"].agg(["mean", "count"])
        enc_map_all = (
            (agg_all["mean"] * agg_all["count"] + global_mean * smoothing)
            / (agg_all["count"] + smoothing)
        ).to_dict()

        X_df[f"{c}__te"] = oof_encoded[c].astype(np.float64)
        key_test = X_test_df[c].astype("object").fillna("__nan__").values
        X_test_df[f"{c}__te"] = (
            pd.Series(key_test)
            .map(enc_map_all)
            .fillna(global_mean)
            .astype(np.float64)
            .values
        )

    return X_df, X_test_df


X, X_test = _oof_target_mean_encode(
    X,
    X_test,
    y_arr=y,
    groups_arr=groups,
    cols=["sex", "anatom_site_general_challenge"],
    n_splits=n_splits,
    smoothing=20.0,
)

numeric_features = [
    "age_approx",
    "sex__freq",
    "anatom_site_general_challenge__freq",
    "sex_x_site__freq",
    "sex__te",
    "anatom_site_general_challenge__te",
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

clf = LogisticRegression(
    solver="saga",
    max_iter=6000,
    C=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
    fit_intercept=True,
)

model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

oof = np.zeros(len(train), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    model.fit(X.iloc[tr_idx], y[tr_idx])
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

oof_std = float(np.std(oof))
try:
    auc = roc_auc_score(y, oof)
except Exception:
    auc = float("nan")

print(
    f"OOF AUC (GroupKFold={n_splits}): {auc:.6f} | oof_std={oof_std:.6e} | prior={prior:.6f}"
)

use_prior_fallback = (not np.isfinite(auc)) or (oof_std < 1e-12)

if use_prior_fallback:
    print(
        "Model predictions appear degenerate; falling back to constant prior for submission."
    )
    test_pred = np.full(len(test), prior, dtype=np.float64)
else:
    model.fit(X, y)
    test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)
    test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)



## === cell 2
sub = sub.copy()

pred_df = pd.DataFrame({"image_name": test["image_name"].values, "target": test_pred})
pred_map = dict(zip(pred_df["image_name"].values, pred_df["target"].values))

sub["target"] = sub["image_name"].map(pred_map).fillna(prior).astype(float)

sub = sub[["image_name", "target"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert sub.shape[1] == 2
assert list(sub.columns) == ["image_name", "target"]
assert sub["target"].between(0, 1).all()
assert sub["image_name"].notna().all()
"""submission.csv is now ready."""
