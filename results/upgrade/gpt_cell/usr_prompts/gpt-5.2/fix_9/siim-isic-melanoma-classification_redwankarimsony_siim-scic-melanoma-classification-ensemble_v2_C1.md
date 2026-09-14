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

0.9191

# 6. Current score

0.79538

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes because it tries to read three external submission CSVs from `../input/public-submission-melanoma-95/`, a directory that does not exist in this environment. This causes a `FileNotFoundError` before `sub` and the three `public_sub_*` DataFrames can be created, which also blocks cell 2 that depends on them.

Patch summary: Modify only cell 1 to (a) load the provided `sample_submission.csv` from the available dataset path, and (b) gracefully fall back to using that same sample submission as a stand-in for the missing external submissions (with deterministic `target` defaults) so that downstream blending in cell 2 can run without changing its logic.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: The patch preserves the variables `public_sub_mean_9533`, `public_sub_median_9533`, `public_sub_meta_ens_9577`, and `sub` as pandas DataFrames with a `target` column, so cell 2’s weighted assignment remains valid and executes without interface changes.

Assumptions: The competition-provided `sample_submission.csv` exists at one of the listed paths, and it contains `image_name` and `target` columns; when external submissions are unavailable, using zeros for their `target` values is an acceptable deterministic fallback to avoid crashing.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score happens because all three “public submission” inputs are missing, so the fallback makes every prediction exactly 0.0; this yields AUC=0.5 (random/constant). To move toward the 0.9191 target with minimal change and without changing the core “blend three submissions” logic, I keep the same variables/DataFrame interfaces but replace the fallback targets with a deterministic, reasonable prior derived from `train.csv`’s malignancy rate (overall mean of `target`). This produces non-constant probabilities (still simple and stable) and should improve AUC above 0.5 while remaining minimal and metric-consistent. I also ensure alignment by `image_name` when constructing the fallback so the blended result matches the sample submission order exactly.'
- What this solution (achieved 0.66725) has done: 'Your current 0.5 AUC comes from constant predictions because the three external “public submissions” aren’t available and your fallback uses a single prior for every test row. To move toward the 0.9191 target with minimal changes while preserving the exact “blend three submissions” core logic, I make the fallback produce *non-constant* probabilities using only the provided metadata (`sex`, `age_approx`, `anatom_site_general_challenge`) by fitting a simple, deterministic logistic regression on `train.csv` and predicting on `test.csv`. I keep the same DataFrame interfaces (`public_sub_*` with `image_name,target`) so cell 2 stays unchanged, and I align by `image_name` to avoid any row-order mismatches that would hurt AUC. This should substantially improve over 0.5 while remaining lightweight and within the constraints.'
- What this solution (achieved 0.65058) has done: 'Your current 0.66725 AUC is limited because all three blended inputs are identical fallbacks, so the ensemble provides no diversity and the final prediction is effectively just one metadata model. To move your score upward toward 0.9191 with minimal changes and the same core “blend three submissions” logic, I keep the exact blending in cell 2 but make each fallback slightly different by varying only LogisticRegression hyperparameters/feature variants in a deterministic way (still metadata-only, same training approach). This preserves the same DataFrame interfaces and evaluation semantics while creating a real ensemble from three related but non-identical metadata models, which should improve AUC without large code changes. I also keep strict `image_name` alignment and deterministic settings to avoid accidental score drops from misalignment.'
- What this solution (achieved 0.77387) has done: 'Your current gap to the target is large (0.65058 vs 0.9191), and the main limitation is that your “three-public-submissions blend” is effectively just three similar metadata-only models. To move the AUC upward without changing the blending core logic, I keep the same interfaces and LogisticRegression approach but add a few strong, competition-relevant tabular features that are already in `train.csv`/`test.csv` (notably `patient_id`, plus simple engineered age terms) to increase signal. I also make the three fallback variants genuinely different via feature subsets/hyperparameters while staying within the same training approach and ensuring strict `image_name` alignment. This should improve AUC materially while remaining lightweight and deterministic.'
- What this solution (achieved 0.73591) has done: 'Your current score (0.77387) is still well below the target (0.9191), so we should improve AUC with the smallest, safe changes while preserving the same “blend three submissions” core logic. The simplest high-impact upgrade is to make the metadata fallback models stronger without changing the modeling family: keep LogisticRegression, but use out-of-fold target encoding for `patient_id` (a very strong signal here) plus a couple of lightweight frequency/interaction features, all computed strictly within train data (no leakage). To preserve the ensemble diversity, each of the three fallbacks use the same pipeline style but slightly different feature sets/hyperparameters as before. We also keep strict `image_name` alignment and still write `submission.csv` with the required columns.'
- What this solution (achieved 0.79338) has done: 'Your current gap to the target is still large (0.73591 vs 0.9191), so we should cautiously increase AUC with minimal, low-risk changes while preserving the same overall “blend three submissions” logic and LogisticRegression metadata fallback. The biggest safe win is to fix a likely bug in `patient_count` for the test set (it’s currently computed from test-only counts, which is inconsistent with train) and to strengthen the very strong `patient_id` signal using leakage-safe out-of-fold *logit* target encoding with smoothing (still computed strictly within train folds). I keep the same three fallback variants and the same weighted blend in cell 2, but each variant benefits from the improved encoding and consistent count feature. These are small, deterministic changes that should move your score upward toward the target without altering the core architecture or submission semantics.'
- What this solution (achieved 0.79538) has done: 'Your current AUC (0.79338) is still well below the 0.9191 target, so we should increase score with the smallest, low-risk change while keeping the same “blend three submissions” core logic and LogisticRegression metadata fallback. The strongest remaining signal we can safely add without changing the model family is leakage-safe target encoding for `anatom_site_general_challenge` (computed out-of-fold on train, then applied to test with smoothing) and a consistent train-based frequency feature for that site; these typically help a lot in this competition’s tabular baseline. I add these features to all three fallback variants while leaving the rest of the pipeline, blending weights, and submission writing unchanged. This should move AUC upward toward the target with minimal additional complexity and within the runtime budget.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
import os

base_candidates = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "../input/siim-isic-melanoma-classification",
    "../data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
    "../data",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_paths = [os.path.join(b, "sample_submission.csv") for b in base_candidates]
sample_path = _first_existing(sample_paths)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {sample_paths}"
    )

sub = pd.read_csv(sample_path)

if "target" not in sub.columns:
    sub["target"] = 0.0
else:
    sub["target"] = (
        pd.to_numeric(sub["target"], errors="coerce").fillna(0.0).astype(float)
    )

train_paths = [os.path.join(b, "train.csv") for b in base_candidates]
train_path = _first_existing(train_paths)
if train_path is None:
    raise FileNotFoundError(f"Could not find train.csv in any of: {train_paths}")

test_paths = [os.path.join(b, "test.csv") for b in base_candidates]
test_path = _first_existing(test_paths)
if test_path is None:
    raise FileNotFoundError(f"Could not find test.csv in any of: {test_paths}")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _safe_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _safe_sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _add_oof_patient_target_encoding(
    train_df, test_df, y_col="target", n_splits=5, seed=0, smoothing=20.0
):
    tr = train_df.copy()
    te = test_df.copy()

    global_mean = float(tr[y_col].mean())
    global_mean = float(np.clip(global_mean, 1e-6, 1 - 1e-6))
    global_logit = float(_safe_logit(global_mean))

    tr["patient_te"] = np.nan
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y = tr[y_col].astype(int).values
    for tr_idx, val_idx in skf.split(tr, y):
        fold_tr = tr.iloc[tr_idx]
        grp = fold_tr.groupby("patient_id")[y_col].agg(["mean", "count"])
        smoothed_p = (grp["mean"] * grp["count"] + global_mean * smoothing) / (
            grp["count"] + smoothing
        )
        stats_logit = smoothed_p.map(_safe_logit)

        tr.iloc[val_idx, tr.columns.get_loc("patient_te")] = tr.iloc[val_idx][
            "patient_id"
        ].map(stats_logit)

    tr["patient_te"] = tr["patient_te"].astype(float).fillna(global_logit)

    grp_full = tr.groupby("patient_id")[y_col].agg(["mean", "count"])
    smoothed_p_full = (
        grp_full["mean"] * grp_full["count"] + global_mean * smoothing
    ) / (grp_full["count"] + smoothing)
    stats_logit_full = smoothed_p_full.map(_safe_logit)
    te["patient_te"] = (
        te["patient_id"].map(stats_logit_full).astype(float).fillna(global_logit)
    )

    tr["patient_count"] = (
        tr.groupby("patient_id")["patient_id"].transform("size").astype(np.int16)
    )
    train_counts = tr["patient_id"].value_counts()
    te["patient_count"] = (
        te["patient_id"].map(train_counts).fillna(1.0).astype(np.int16)
    )

    return tr, te


def _add_oof_site_target_encoding(
    train_df,
    test_df,
    site_col="anatom_site_general_challenge",
    y_col="target",
    n_splits=5,
    seed=0,
    smoothing=50.0,
):
    tr = train_df.copy()
    te = test_df.copy()

    tr_site = tr[site_col].astype("object").fillna("__MISSING__")
    te_site = te[site_col].astype("object").fillna("__MISSING__")

    global_mean = float(tr[y_col].mean())
    global_mean = float(np.clip(global_mean, 1e-6, 1 - 1e-6))
    global_logit = float(_safe_logit(global_mean))

    tr["site_te"] = np.nan
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = tr[y_col].astype(int).values

    for tr_idx, val_idx in skf.split(tr, y):
        fold_tr = tr.iloc[tr_idx]
        fold_site = fold_tr[site_col].astype("object").fillna("__MISSING__")
        grp = (
            fold_tr.assign(_site=fold_site)
            .groupby("_site")[y_col]
            .agg(["mean", "count"])
        )
        smoothed_p = (grp["mean"] * grp["count"] + global_mean * smoothing) / (
            grp["count"] + smoothing
        )
        stats_logit = smoothed_p.map(_safe_logit)

        val_site = tr_site.iloc[val_idx]
        tr.iloc[val_idx, tr.columns.get_loc("site_te")] = val_site.map(
            stats_logit
        ).values

    tr["site_te"] = tr["site_te"].astype(float).fillna(global_logit)

    grp_full = tr.assign(_site=tr_site).groupby("_site")[y_col].agg(["mean", "count"])
    smoothed_p_full = (
        grp_full["mean"] * grp_full["count"] + global_mean * smoothing
    ) / (grp_full["count"] + smoothing)
    stats_logit_full = smoothed_p_full.map(_safe_logit)
    te["site_te"] = te_site.map(stats_logit_full).astype(float).fillna(global_logit)

    site_counts = tr_site.value_counts()
    tr["site_count"] = tr_site.map(site_counts).astype(np.int32)
    te["site_count"] = te_site.map(site_counts).fillna(1.0).astype(np.int32)

    return tr, te


def _fit_predict_metadata(train_csv, test_csv, sub_df, variant="base"):
    use_cols = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "target",
    ]
    train_df = pd.read_csv(train_csv, usecols=use_cols)
    test_df = pd.read_csv(
        test_csv,
        usecols=[
            "image_name",
            "patient_id",
            "sex",
            "age_approx",
            "anatom_site_general_challenge",
        ],
    )

    train_df["age_approx"] = pd.to_numeric(train_df["age_approx"], errors="coerce")
    test_df["age_approx"] = pd.to_numeric(test_df["age_approx"], errors="coerce")

    for df in (train_df, test_df):
        df["age_missing"] = df["age_approx"].isna().astype(np.int8)
        df["age2"] = (df["age_approx"].fillna(df["age_approx"].median()) ** 2).astype(
            float
        )

    train_df, test_df = _add_oof_patient_target_encoding(
        train_df, test_df, y_col="target", n_splits=5, seed=0, smoothing=20.0
    )

    train_df, test_df = _add_oof_site_target_encoding(
        train_df,
        test_df,
        site_col="anatom_site_general_challenge",
        y_col="target",
        n_splits=5,
        seed=0,
        smoothing=50.0,
    )

    for df in (train_df, test_df):
        df["age_x_patient_te"] = df["age_approx"].fillna(
            train_df["age_approx"].median()
        ).astype(float) * df["patient_te"].astype(float)

    if variant == "base":
        feat_cols = [
            "patient_te",
            "patient_count",
            "site_te",
            "site_count",
            "sex",
            "age_approx",
            "age2",
            "age_missing",
            "age_x_patient_te",
            "anatom_site_general_challenge",
        ]
        C = 0.7
        class_weight = "balanced"
    elif variant == "strong_reg":
        feat_cols = [
            "patient_te",
            "patient_count",
            "site_te",
            "site_count",
            "sex",
            "age_approx",
            "age_missing",
            "anatom_site_general_challenge",
        ]
        C = 0.25
        class_weight = "balanced"
    elif variant == "no_age":
        feat_cols = [
            "patient_te",
            "patient_count",
            "site_te",
            "site_count",
            "sex",
            "anatom_site_general_challenge",
        ]
        C = 0.9
        class_weight = "balanced"
    else:
        feat_cols = [
            "patient_te",
            "patient_count",
            "site_te",
            "site_count",
            "sex",
            "age_approx",
            "age2",
            "age_missing",
            "age_x_patient_te",
            "anatom_site_general_challenge",
        ]
        C = 0.7
        class_weight = "balanced"

    X_train = train_df[feat_cols]
    y_train = train_df["target"].astype(int)
    X_test = test_df[feat_cols]

    numeric_features = [
        c
        for c in feat_cols
        if c
        in (
            "age_approx",
            "age2",
            "age_missing",
            "patient_te",
            "patient_count",
            "age_x_patient_te",
            "site_te",
            "site_count",
        )
    ]
    categorical_features = [c for c in feat_cols if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        C=C,
        class_weight=class_weight,
        random_state=0,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict_proba(X_test)[:, 1].astype(float)
    pred = np.clip(pred, 1e-6, 1 - 1e-6)

    pred_df = pd.DataFrame({"image_name": test_df["image_name"].values, "target": pred})

    pred_map = pred_df.set_index("image_name")["target"]
    out = sub_df[["image_name"]].copy()
    out["target"] = out["image_name"].map(pred_map).astype(float)

    prior = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    out["target"] = out["target"].fillna(prior).clip(0.0, 1.0)
    return out


ext_base = "../input/public-submission-melanoma-95"
mean_path = os.path.join(ext_base, "submission_mean.csv")
median_path = os.path.join(ext_base, "submission_median.csv")
meta_path = os.path.join(ext_base, "external_meta_ensembled.csv")


def _load_or_fallback(path, fallback_df, train_csv, test_csv, variant):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            return _fit_predict_metadata(
                train_csv, test_csv, fallback_df, variant=variant
            )
        df["target"] = (
            pd.to_numeric(df["target"], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .clip(0.0, 1.0)
        )
        df = fallback_df[["image_name"]].merge(
            df[["image_name", "target"]], on="image_name", how="left"
        )
        if df["target"].isna().any():
            meta_pred = _fit_predict_metadata(
                train_csv, test_csv, fallback_df, variant=variant
            ).set_index("image_name")["target"]
            df["target"] = df["target"].fillna(df["image_name"].map(meta_pred))
        df["target"] = df["target"].astype(float).clip(0.0, 1.0)
        return df

    return _fit_predict_metadata(train_csv, test_csv, fallback_df, variant=variant)


public_sub_mean_9533 = _load_or_fallback(
    mean_path, sub, train_path, test_path, variant="strong_reg"
)
public_sub_median_9533 = _load_or_fallback(
    median_path, sub, train_path, test_path, variant="base"
)
public_sub_meta_ens_9577 = _load_or_fallback(
    meta_path, sub, train_path, test_path, variant="no_age"
)



## === cell 2
sub.target = (
    public_sub_meta_ens_9577.target * 0.40
    + public_sub_median_9533.target * 0.30
    + public_sub_mean_9533.target * 0.30
)



## === cell 3
sub.head()
sub.to_csv("submission.csv", index=False)
