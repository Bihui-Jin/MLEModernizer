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

0.8856343494136878

# 6. Current score

0.77491

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66776) has done: 'I fix the runtime errors by removing the dependency on a missing `../input/efficientnets/` directory and instead generate a valid prediction file directly from the provided competition data. Since your current run produces no submission, the minimal stable approach is to create a metadata-only baseline model (logistic regression with proper preprocessing) trained on `train.csv` and used to predict probabilities for `test.csv`, which yields a valid AUC-oriented probabilistic submission. I also ensure the submission strictly matches `sample_submission.csv` ordering and column names (`image_name,target`) and is saved with a `.csv` suffix. This keeps the spirit of the original “blend submissions” approach (no image model) while making it executable end-to-end in your environment.'
- What this solution (achieved 0.66724) has done: 'Your current score (0.66776) is far below the target (0.8856), so we should make a small, low-risk improvement that keeps the same “metadata-only logistic regression” core while better matching the competition’s patient-level structure and reducing leakage. The biggest gain with minimal logic change is to use a group-aware split by `patient_id` to tune only the regularization strength `C` for AUC (no architecture change), then refit on all data with the best `C`. I also add a standardization step for the numeric age feature (commonly helps LR calibration/ranking) while keeping the same preprocessing approach. Finally, the submission alignment remain exactly matched to `sample_submission.csv` order and columns.'
- What this solution (achieved 0.66725) has done: 'Your current pipeline is a metadata-only logistic regression, so the smallest legitimate way to move AUC upward (toward 0.8856) without changing the modeling approach is to (1) reduce avoidable noise from extreme `C` selection by using a more targeted, denser `C` grid and (2) increase CV stability by using a fixed shuffle-free GroupKFold but selecting `C` by mean fold AUC (not a single pooled OOF AUC), which is less sensitive to fold prevalence differences in this dataset. I also add `missing_values=np.nan` explicitly and cast `age_approx` to numeric to avoid silent object coercion issues that can hurt the ranking quality. Everything else (features, LR model, preprocessing, group-aware CV, probabilistic submission format/alignment) is preserved.'
- What this solution (achieved 0.66942) has done: 'Your current AUC (0.66725) is far below the target (0.8856), so we should make the smallest change that can legitimately improve ranking while keeping the same metadata-only logistic regression core. The biggest low-risk gain here is to stop forcing `class_weight="balanced"` (which often hurts ROC-AUC ranking for this competition’s severe imbalance by overcompensating the minority class) and instead tune this choice via the same GroupKFold CV you already have. We keep the exact same features, preprocessing, solver, and CV procedure, and only extend the hyperparameter search to include `class_weight ∈ {None, "balanced"}` while still selecting by mean fold AUC. This stays within your current approach and should move the score upward toward the target without changing submission semantics.'
- What this solution (achieved 0.68031) has done: 'Your current AUC (0.669) is far below the target (0.886), so we need a legitimate lift while keeping the same “metadata-only logistic regression” core. The smallest high-impact change for this competition is to add a few well-known strong metadata signals already present in `train.csv/test.csv` (notably `patient_id` and `diagnosis`) as additional categorical features via the same OneHot+LogReg pipeline; this preserves the exact model family and training loop. To avoid leakage/overfitting risk as much as possible, we keep the GroupKFold by `patient_id` and only extend the feature set (no change to loss/solver/approach). Submission writing/order stays identical and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.73353) has done: 'Your current gap to the target AUC is large, so the smallest legitimate lift without changing the “metadata-only logistic regression” core is to remove a high-leakage/overfit categorical (`diagnosis`) and replace it with a safer, competition-standard metadata signal: `anatom_site_general_challenge` frequency encoding (computed on train+test) plus one extra numeric (`log1p(patient_lesion_count)` derived from `patient_id` counts). This keeps the same training loop (GroupKFold CV to pick `C`/`class_weight`, then refit once) and the same model family, but typically improves ranking on this dataset versus high-cardinality one-hot IDs/diagnosis. I also reduce the damage from extremely high-cardinality `patient_id` one-hot by dropping it from one-hot entirely (still using it only for grouping and derived counts), which usually generalizes better to the test set. Submission writing/order is unchanged and still aligned exactly to `sample_submission.csv`.'
- What this solution (achieved 0.73343) has done: 'Your current score (0.73353) is far below the target (0.88563), so we should make a small, low-risk improvement that keeps the same metadata-only LogisticRegression + GroupKFold selection core. The biggest gain we can get without changing the model family is to add a couple of proven metadata-derived numeric signals (still fed into the same preprocessing + LR): (1) per-patient mean target (OOF target encoding) to capture patient-level propensity while avoiding leakage via GroupKFold, and (2) a simple interaction between age and site frequency. I implement the patient target encoding in an out-of-fold way for training and a smoothed mapping for test (global mean fallback for unseen patients), which tends to lift AUC on this competition while preserving evaluation semantics. Everything else (CV to pick C/class_weight, preprocessing, probability submission alignment) remains the same.'
- What this solution (achieved 0.39676) has done: 'Your current score (0.733) is far below the target (0.886), so we need a legitimate lift while keeping your exact metadata-only LogisticRegression + GroupKFold selection core. The smallest high-impact fix is to remove train/test distribution shift created by using test data inside your `anatom_site_freq` and `patient_lesion_count` features; we compute those encodings from train only (then apply to test with safe defaults), which typically improves generalization and AUC on this competition. Separately, your patient target encoding smoothing currently uses the global mean from *all* data (including the validation fold); we compute the global mean per fold from the fold’s training split only to avoid subtle leakage and improve CV selection stability. Everything else (features, model family, CV procedure, submission writing/alignment) remains the same.'
- What this solution (achieved 0.39676) has done: 'Your current score (0.3968) is far below the target (0.8856), and the largest likely cause is that the feature engineering for `pid_target_enc` is broken due to inconsistent `patient_id` typing: you build the per-fold mapping with raw `patient_id` values but look up using strings in `groups`, which makes most lookups miss and collapses the encoding toward the global mean (hurting AUC). I make the smallest fix by creating a single canonical `patient_id_str` used consistently for grouping, counting, and target encoding (both OOF and full-fit), without changing the model family, CV procedure, or submission semantics. I also ensure the encodings are computed from the intended columns and that mapping lookups match the mapping key dtype, which should materially improve ranking toward your target while keeping everything else intact. The rest of your pipeline (LogReg + GroupKFold selection, preprocessing, and submission alignment) remains unchanged.'
- What this solution (achieved 0.39676) has done: 'Your current score (0.3968) is far below the target (0.8856), so we should make a small, legitimate improvement without changing the model family or training loop. The biggest likely issue is that `pid_target_enc` is computed OOF on the full dataset, but then you do GroupKFold CV using those same precomputed features—so each validation fold gets a feature that was trained using its own labels (leakage), causing CV to select hyperparameters that do not generalize to the real test set. I fix this by recomputing `pid_target_enc` *inside each CV fold* (fit encoding on fold-train only, apply to fold-valid), and then train the final model using a proper OOF encoding for train and full-train mapping for test (same feature, same LR pipeline). This keeps the exact core logic (LogReg + GroupKFold selection + same feature set) but removes the mismatch between CV and final training, which should move AUC upward toward your target.'
- What this solution (achieved 0.39676) has done: 'Your current score is far below the target, so we should only make changes that plausibly improve AUC without changing the core “metadata + LogisticRegression + GroupKFold selection” approach. The biggest bug-like issue is that you create a global OOF `pid_target_enc` and then overwrite it inside CV folds, but the final model is trained on a different `pid_target_enc` definition than what CV optimizes against; we align CV and final training by generating the *same* encoding recipe for both (OOF for training, full-fit mapping for test) and use that consistently. Next, we remove the train/test feature inconsistency caused by fitting numeric imputers/scalers on all training rows including those whose `pid_target_enc` came from OOF; instead, we keep OOF (still no leakage) but ensure the final model’s `pid_target_enc` is computed in a full-fit way consistent with the feature definition used at inference. Finally, we add a very small, safe improvement for LR ranking stability: increase `max_iter` to ensure convergence and set `random_state` for determinism (does not change model family or training loop).'
- What this solution (achieved 0.77491) has done: 'Your current score is far below the target, so we should only make small, legitimate changes that can materially improve AUC without changing the core “metadata features + LogisticRegression + GroupKFold selection” approach. The biggest issue is that `pid_target_enc` is computed once globally (OOF) and then reused inside CV, which leaks validation labels into fold features and leads to hyperparameters that don’t generalize to the test set; we fix this by recomputing `pid_target_enc` inside each CV fold and using that same recipe when training the final model. We also ensure `patient_id_str` treats missing IDs consistently (true missing stays missing rather than the literal string `"nan"`), which improves grouping/encoding quality. Everything else (features, LR, CV, submission alignment/format) stays the same.'
- What this solution (achieved 0.77491) has done: 'Your current gap to the target AUC is still large, so the smallest legitimate lift while preserving the same metadata+LogisticRegression core is to fix two feature-quality issues that commonly hurt ranking: (1) `patient_id_str` currently turns missing IDs into the literal string `"<NA>"`, which then becomes a “real” patient group and pollutes counts/target-encoding; we keep true missing as missing and only fill a sentinel for `GroupKFold`. (2) `anatom_site_general_challenge` is frequency-encoded using `astype(str)`, which similarly turns missing into `"nan"`; we compute site frequency on a clean, missing-aware key so train/test missingness is handled consistently. These are minimal, “bug-fix” style changes that keep the exact same model family, CV loop, and feature set, but should improve generalization and move AUC upward toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]

BASE_DIR = None
for d in BASE_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        BASE_DIR = d
        break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv under expected Kaggle directories. "
        f"Tried: {BASE_DIR_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

print("Using BASE_DIR =", BASE_DIR)
print("TRAIN_CSV =", TRAIN_CSV)
print("TEST_CSV  =", TEST_CSV)
print("SAMPLE_SUB_CSV =", SAMPLE_SUB_CSV)



## === cell 2
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert "target" in train.columns, "train.csv must contain 'target'"
assert set(sample_sub.columns) == {
    "image_name",
    "target",
}, "sample_submission must have columns: image_name,target"
assert "image_name" in test.columns, "test.csv must contain 'image_name'"

print(train.shape, test.shape, sample_sub.shape)
train.head()



## === cell 3
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

train_local = train.copy()
test_local = test.copy()

for df in (train_local, test_local):
    if "sex" not in df.columns:
        df["sex"] = np.nan
    if "age_approx" not in df.columns:
        df["age_approx"] = np.nan
    if "anatom_site_general_challenge" not in df.columns:
        df["anatom_site_general_challenge"] = np.nan
    if "patient_id" not in df.columns:
        df["patient_id"] = np.nan

for df in (train_local, test_local):
    df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")

train_local["patient_id_str"] = train_local["patient_id"].astype("string")
test_local["patient_id_str"] = test_local["patient_id"].astype("string")

_pid_key_tr = train_local["patient_id_str"].fillna("__MISSING__")
pid_counts_tr = _pid_key_tr.value_counts(dropna=False)
train_local["patient_lesion_count"] = _pid_key_tr.map(pid_counts_tr).astype(float)
test_local["patient_lesion_count"] = (
    test_local["patient_id_str"]
    .fillna("__MISSING__")
    .map(pid_counts_tr)
    .astype(float)
    .fillna(1.0)
)

site_key_tr = (
    train_local["anatom_site_general_challenge"].astype("string").fillna("__MISSING__")
)
site_counts_tr = site_key_tr.value_counts(dropna=False)
total_n_tr = float(len(train_local))

train_local["anatom_site_freq"] = (
    train_local["anatom_site_general_challenge"]
    .astype("string")
    .fillna("__MISSING__")
    .map(site_counts_tr)
    .astype(float)
    / total_n_tr
)
test_local["anatom_site_freq"] = (
    test_local["anatom_site_general_challenge"]
    .astype("string")
    .fillna("__MISSING__")
    .map(site_counts_tr)
    .astype(float)
    / total_n_tr
).fillna(0.0)

train_local["log_patient_lesion_count"] = np.log1p(
    train_local["patient_lesion_count"].astype(float)
)
test_local["log_patient_lesion_count"] = np.log1p(
    test_local["patient_lesion_count"].astype(float)
)

train_local["age_x_sitefreq"] = train_local["age_approx"].astype(float) * train_local[
    "anatom_site_freq"
].astype(float)
test_local["age_x_sitefreq"] = test_local["age_approx"].astype(float) * test_local[
    "anatom_site_freq"
].astype(float)

groups = train_local["patient_id_str"].fillna("__MISSING__").astype("string").values
y_train = train_local["target"].astype(int).values

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

alpha = 20.0  # smoothing strength


def fit_pid_target_encoder(tr_df: pd.DataFrame, alpha: float):
    """Fit smoothed target mean per patient on tr_df only."""
    fold_global_mean = float(tr_df["target"].mean())
    stats = tr_df.groupby("patient_id_str")["target"].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + fold_global_mean * alpha) / (
        stats["count"] + alpha
    )
    return smooth.to_dict(), fold_global_mean


def apply_pid_target_encoder(df: pd.DataFrame, mapping: dict, default: float):
    return df["patient_id_str"].map(mapping).astype(float).fillna(float(default)).values


feature_cols = [
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "anatom_site_freq",
    "log_patient_lesion_count",
    "age_x_sitefreq",
    "pid_target_enc",
]

numeric_features = [
    "age_approx",
    "anatom_site_freq",
    "log_patient_lesion_count",
    "age_x_sitefreq",
    "pid_target_enc",
]
categorical_features = ["sex", "anatom_site_general_challenge"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)


def make_model(C: float, class_weight):
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        n_jobs=None,
        class_weight=class_weight,
        C=C,
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", clf)])


C_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
class_weight_grid = [None, "balanced"]

cv_results = []
for cw in class_weight_grid:
    for C in C_grid:
        fold_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(train_local, y_train, groups=groups), start=1
        ):
            tr_df = train_local.iloc[tr_idx].copy()
            va_df = train_local.iloc[va_idx].copy()

            mapping, fold_mean = fit_pid_target_encoder(tr_df, alpha=alpha)
            tr_df["pid_target_enc"] = apply_pid_target_encoder(
                tr_df, mapping, fold_mean
            )
            va_df["pid_target_enc"] = apply_pid_target_encoder(
                va_df, mapping, fold_mean
            )

            X_tr = tr_df[feature_cols]
            X_va = va_df[feature_cols]

            m = make_model(C, cw)
            m.fit(X_tr, y_train[tr_idx])
            va_pred = m.predict_proba(X_va)[:, 1]
            fold_auc = roc_auc_score(y_train[va_idx], va_pred)
            fold_aucs.append(fold_auc)

        mean_auc = float(np.mean(fold_aucs))
        cv_results.append((cw, C, mean_auc))
        print(
            f"class_weight={str(cw):>8} C={C:<4} GroupKFold mean AUC={mean_auc:.6f} "
            f"(folds: {[round(a,6) for a in fold_aucs]})"
        )

best_cw, best_C, best_auc = max(cv_results, key=lambda x: x[2])
print(
    "Selected best class_weight =",
    best_cw,
    "best_C =",
    best_C,
    "with mean CV AUC =",
    best_auc,
)

global_mean_full = float(np.mean(y_train))
full_stats = train_local.groupby("patient_id_str")["target"].agg(["mean", "count"])
full_smooth = (full_stats["mean"] * full_stats["count"] + global_mean_full * alpha) / (
    full_stats["count"] + alpha
)
full_mapping = full_smooth.to_dict()

train_local["pid_target_enc"] = apply_pid_target_encoder(
    train_local, full_mapping, global_mean_full
)
test_local["pid_target_enc"] = apply_pid_target_encoder(
    test_local, full_mapping, global_mean_full
)

X_train = train_local[feature_cols].copy()
X_test = test_local[feature_cols].copy()

model = make_model(best_C, best_cw)
model.fit(X_train, y_train)



## === cell 4
test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

test_pred = np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0)
test_pred = np.clip(test_pred, 0.0, 1.0)

print(
    "Pred summary:",
    float(test_pred.min()),
    float(test_pred.mean()),
    float(test_pred.max()),
)



## === cell 5
sub = sample_sub[["image_name"]].copy()

pred_map = pd.Series(test_pred, index=test_local["image_name"]).to_dict()
sub["target"] = sub["image_name"].map(pred_map)
sub["target"] = sub["target"].astype(float).fillna(0.5)

assert (
    sub.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample_submission"
assert sub["target"].between(0, 1).all(), "All predictions must be within [0,1]"

sub.head()



## === cell 6
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.describe(include="all"))
