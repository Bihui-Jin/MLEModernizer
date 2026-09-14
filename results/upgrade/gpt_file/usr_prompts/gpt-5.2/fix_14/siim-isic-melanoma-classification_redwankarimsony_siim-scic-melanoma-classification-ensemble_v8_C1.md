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

0.914893167760655

# 6. Current score

0.74016

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66789) has done: 'I remove the dependency on missing `../input/public-submission-melanoma-95/*.csv` files (which is causing the FileNotFoundError) and replace it with a self-contained baseline that uses only the provided `train.csv` and `test.csv`. To keep core logic minimal and stable while achieving a reasonable AUC, I train a straightforward tabular model (logistic regression with preprocessing for categorical + numeric metadata) and generate probabilistic predictions for the test set. I also ensure the submission uses the exact required columns (`image_name,target`), correct row alignment, and writes to `submission.csv` with a `.csv` suffix. This run end-to-end in the given environment without extra packages.'
- What this solution (achieved 0.66725) has done: 'I fix the training crash caused by pandas `pd.NA` values leaking into the sklearn pipeline, which triggers “boolean value of NA is ambiguous” inside `SimpleImputer`. The minimal robust fix is to convert the feature matrices to plain numpy missing values (`np.nan`) by replacing `pd.NA` and also forcing any remaining pandas nullable dtypes to regular numpy-friendly types before fitting. Once the pipeline fits successfully, `test_pred` be defined and the submission-writing cell run and create a valid `submission.csv` with the required columns. These changes are correctness/stability fixes and should also restore the intended baseline score by allowing the model to train properly.'
- What this solution (achieved 0.76804) has done: 'Your current score (0.66725) is far below the target (0.91489), so we should improve performance while keeping your tabular-logistic core intact. The largest safe gain here is to prevent patient-level leakage in training by doing out-of-fold (OOF) training with `GroupKFold` on `patient_id`, then using the fold-trained models to predict test and average probabilities (same model/pipeline, just repeated fits). This typically yields a materially better AUC than a single fit on all rows because it reduces overfitting and stabilizes categorical handling, while keeping the exact same preprocessing + LogisticRegression logic. I also keep the submission alignment strict by merging predictions onto `sample_submission` by `image_name` so ordering cannot silently mismatch.'
- What this solution (achieved 0.74983) has done: 'Your current score (0.76804) is well below the target (0.91489), so we should improve AUC while keeping the same tabular + LogisticRegression core. The smallest reliable gain is to add a few well-known, leakage-safe metadata features derived from existing columns (age missingness indicator, age binning, and site×sex interaction) while keeping the exact same preprocessing (impute + one-hot) and the same GroupKFold training/averaging approach. These features often help linear models capture non-linear risk patterns and interactions that matter for melanoma, improving ranking (AUC) without changing the overall method. I also set `random_state` where applicable for stability, but won’t change the training loop, model family, or evaluation semantics.'
- What this solution (achieved 0.74657) has done: 'Your current AUC (0.74983) is far below the target (0.91489), so we should carefully improve ranking while keeping your same metadata + preprocessing + LogisticRegression + GroupKFold core intact. The most impactful minimal change is to tune LogisticRegression’s regularization strength (C) and solver choice within the same model family, because your current C=0.5 may be underfitting and hurting AUC. To avoid overfitting/leakage and keep semantics identical, we select C (and solver) using GroupKFold cross-validated AUC on the training set, then refit per-fold and average test probabilities exactly as you already do. This is a small, legitimate change that usually yields a noticeable AUC gain for linear tabular baselines without changing the approach.'
- What this solution (achieved 0.73441) has done: 'Your timeout is driven by repeatedly refitting an expensive `ColumnTransformer+OneHotEncoder` inside 5-fold CV for each of 13 hyperparameter settings (65 full preprocess+fit passes), and then again for the final 5 folds. The safe speed fix is to fit the preprocessing **once**, transform train/test **once**, and then run the exact same `GroupKFold` loops and `LogisticRegression` fits on the resulting numeric matrices (identical features/semantics). Additionally, we enable OHE sparsity and pass sparse matrices through to LogisticRegression (much faster and less memory than dense), and we avoid repeated pandas `.iloc` slicing by using numpy/sparse indexing. These changes preserve the model, loss, CV logic, and predictions up to negligible floating-point differences while cutting the dominant runtime drastically.'
- What this solution (achieved 0.735) has done: 'We keep your exact metadata→OHE→LogisticRegression + GroupKFold averaging core, but make two minimal changes aimed at lifting AUC toward the target: (1) expand the regularization search a bit around the current region (including higher C) using the same CV protocol, and (2) add a single additional interaction feature (`age_bin x site`) you already partially use but can strengthen in a linear model. These are small, leakage-safe changes that often improve ranking without changing evaluation semantics. The rest (single preprocess fit/transform, fold training, submission merge/alignment) stays the same to preserve stability and runtime.'
- What this solution (achieved 0.733) has done: 'We keep your exact metadata→OHE→LogisticRegression + GroupKFold averaging core, but make two small, score-relevant changes that typically improve AUC ranking without changing evaluation semantics. First, we expand the derived feature set with a couple of lightweight, leakage-safe interactions (site×sex×agebin and patient-level lesion count) that linear models benefit from. Second, we slightly widen the regularization search around higher-capacity settings (still LogisticRegression) while keeping the same CV selection and test-fold averaging. These are minimal additions aimed at lifting AUC from ~0.735 toward your 0.915 target while staying within runtime and package constraints.'
- What this solution (achieved 0.74016) has done: 'Your current AUC (0.733) is far below the target (0.9149), so we need a legitimate boost while keeping the same metadata→OHE→LogisticRegression + GroupKFold core. The biggest issue is that `patient_lesion_count` is currently computed separately for train and test, which makes it inconsistent across splits and hurts generalization; we compute it once from the concatenated train+test patient pool and then split back (same feature, just corrected). Next, we remove the extra regularization bias introduced by `class_weight="balanced"` (AUC is ranking-based, and the class weighting often distorts probabilities/ranking when you’re already doing proper CV), keeping LogisticRegression otherwise identical. Finally, we make the `GroupKFold` parameter selection more robust by choosing the best config using mean fold AUC (instead of a single global OOF AUC that can be noisier), without changing the CV scheme or model family.'
- What this solution (achieved 0.74016) has done: 'We keep your exact metadata→OHE→LogisticRegression + GroupKFold core, but make two minimal, score-relevant fixes that usually improve AUC without changing the modeling approach. First, we switch the CV scoring used for selecting `C/solver` to be **patient-weighted AUC** (weight each lesion by `1 / lesions_per_patient`), which better matches the competition’s patient-grouped generalization and tends to reduce domination by patients with many images. Second, we use `class_weight="balanced"` only during the hyperparameter search (to avoid under-selecting recall-heavy settings) and then refit the final fold models **without** class weighting (often yields better probability ranking for AUC). The rest of the pipeline (features, preprocessing, GroupKFold splits, averaging fold predictions, and submission formatting) is unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "/kaggle/data" if os.path.exists("/kaggle/data") else "/kaggle/input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)

y = train["target"].astype(int)

drop_cols = ["image_name", "diagnosis", "benign_malignant", "target"]
feature_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]

X_train_full = train[feature_cols].copy()
X_test = test[feature_cols].copy()

X_train_full = X_train_full.replace({pd.NA: np.nan})
X_test = X_test.replace({pd.NA: np.nan})


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "age_approx" in df.columns:
        age = pd.to_numeric(df["age_approx"], errors="coerce")
        df["age_missing"] = age.isna().astype(np.int8)
        df["age_bin"] = pd.cut(
            age,
            bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
            labels=["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"],
        ).astype("object")

    if "anatom_site_general_challenge" in df.columns:
        df["site_missing"] = df["anatom_site_general_challenge"].isna().astype(np.int8)

    if "sex" in df.columns:
        df["sex_missing"] = df["sex"].isna().astype(np.int8)

    if "anatom_site_general_challenge" in df.columns and "sex" in df.columns:
        site = df["anatom_site_general_challenge"].astype("object")
        sex = df["sex"].astype("object")
        df["site_x_sex"] = (site.fillna("NA") + "__" + sex.fillna("NA")).astype(
            "object"
        )

    if "age_bin" in df.columns and "sex" in df.columns:
        df["sex_x_agebin"] = (
            df["sex"].astype("object").fillna("NA")
            + "__"
            + df["age_bin"].astype("object").fillna("NA")
        ).astype("object")

    if "age_bin" in df.columns and "anatom_site_general_challenge" in df.columns:
        df["site_x_agebin"] = (
            df["anatom_site_general_challenge"].astype("object").fillna("NA")
            + "__"
            + df["age_bin"].astype("object").fillna("NA")
        ).astype("object")
        df["agebin_x_site"] = (
            df["age_bin"].astype("object").fillna("NA")
            + "__"
            + df["anatom_site_general_challenge"].astype("object").fillna("NA")
        ).astype("object")

    if (
        "age_bin" in df.columns
        and "sex" in df.columns
        and "anatom_site_general_challenge" in df.columns
    ):
        df["site_x_sex_x_agebin"] = (
            df["anatom_site_general_challenge"].astype("object").fillna("NA")
            + "__"
            + df["sex"].astype("object").fillna("NA")
            + "__"
            + df["age_bin"].astype("object").fillna("NA")
        ).astype("object")

    return df


X_train_full = add_derived_features(X_train_full)
X_test = add_derived_features(X_test)

if "patient_id" in X_train_full.columns:
    tr_n = len(X_train_full)
    all_pid = pd.concat([X_train_full["patient_id"], X_test["patient_id"]], axis=0)
    all_pid = all_pid.fillna("NA").astype(str)
    all_counts = all_pid.value_counts()

    all_patient_lesion_count = all_pid.map(all_counts).astype(np.float64).to_numpy()
    X_train_full["patient_lesion_count"] = all_patient_lesion_count[:tr_n]
    X_test["patient_lesion_count"] = all_patient_lesion_count[tr_n:]

feature_cols = list(X_train_full.columns)

cat_cols = []
num_cols = []
for c in feature_cols:
    if X_train_full[c].dtype == "object" or str(X_train_full[c].dtype).startswith(
        "string"
    ):
        cat_cols.append(c)
    else:
        num_cols.append(c)

for c in num_cols:
    X_train_full[c] = pd.to_numeric(X_train_full[c], errors="coerce").astype(np.float64)
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(np.float64)

for c in cat_cols:
    X_train_full[c] = X_train_full[c].astype("object")
    X_test[c] = X_test[c].astype("object")

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0,
)

groups = train["patient_id"].fillna("NA").astype(str).values
gkf = GroupKFold(n_splits=5)

param_grid = [
    {"solver": "lbfgs", "C": 0.05},
    {"solver": "lbfgs", "C": 0.1},
    {"solver": "lbfgs", "C": 0.2},
    {"solver": "lbfgs", "C": 0.5},
    {"solver": "lbfgs", "C": 1.0},
    {"solver": "lbfgs", "C": 2.0},
    {"solver": "lbfgs", "C": 5.0},
    {"solver": "lbfgs", "C": 10.0},
    {"solver": "lbfgs", "C": 20.0},
    {"solver": "lbfgs", "C": 50.0},
    {"solver": "liblinear", "C": 0.1},
    {"solver": "liblinear", "C": 0.2},
    {"solver": "liblinear", "C": 0.5},
    {"solver": "liblinear", "C": 1.0},
    {"solver": "liblinear", "C": 2.0},
    {"solver": "liblinear", "C": 5.0},
    {"solver": "liblinear", "C": 10.0},
    {"solver": "liblinear", "C": 20.0},
]

X_train_all = preprocess.fit_transform(X_train_full)
X_test_all = preprocess.transform(X_test)

splits = list(gkf.split(X_train_full, y, groups=groups))

best_cfg = None
best_auc = -np.inf


def _take_rows(X, idx):
    return X[idx]


pid_train = train["patient_id"].fillna("NA").astype(str)
pid_counts_train = pid_train.value_counts()
sample_weight_all = (
    (1.0 / pid_train.map(pid_counts_train)).astype(np.float64).to_numpy()
)

for cfg in param_grid:
    fold_aucs = []
    for tr_idx, va_idx in splits:
        X_tr = _take_rows(X_train_all, tr_idx)
        y_tr = y.iloc[tr_idx].to_numpy()
        X_va = _take_rows(X_train_all, va_idx)
        y_va = y.iloc[va_idx].to_numpy()

        clf = LogisticRegression(
            max_iter=800,
            solver=cfg["solver"],
            C=cfg["C"],
            n_jobs=None,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_tr, y_tr)
        va_pred = clf.predict_proba(X_va)[:, 1]

        w_va = sample_weight_all[va_idx]
        fold_aucs.append(roc_auc_score(y_va, va_pred, sample_weight=w_va))

    auc = float(np.mean(fold_aucs))
    if auc > best_auc:
        best_auc = auc
        best_cfg = cfg

test_pred_folds = []
for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
    X_tr = _take_rows(X_train_all, tr_idx)
    y_tr = y.iloc[tr_idx].to_numpy()

    clf = LogisticRegression(
        max_iter=800,
        solver=best_cfg["solver"],
        C=best_cfg["C"],
        n_jobs=None,
        random_state=42,
        class_weight=None,
    )
    clf.fit(X_tr, y_tr)

    fold_test_pred = clf.predict_proba(X_test_all)[:, 1]
    test_pred_folds.append(np.asarray(fold_test_pred, dtype=np.float64))

test_pred = np.mean(np.vstack(test_pred_folds), axis=0)
test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 1
pred_df = pd.DataFrame({"image_name": test["image_name"].values, "target": test_pred})
sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

assert (
    sub_out.shape[0] == sub.shape[0]
), "Submission row count must match sample submission row count."
assert sub_out["target"].notna().all(), "All test image_name must receive a prediction."
assert list(sub_out.columns) == [
    "image_name",
    "target",
], "Submission columns must be ['image_name', 'target']."

sub_out.to_csv("submission.csv", index=False)
sub_out.head()
