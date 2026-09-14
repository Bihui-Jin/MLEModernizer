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

0.9331869828129152

# 6. Current score

0.31222

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66776) has done: 'I remove the dependency on missing `../input/public-submission-melanoma-95/*` files (the root cause of the FileNotFoundError) and instead build a valid submission from the provided competition files. To keep the core “blend submissions” intent with minimal changes, the script fall back to a simple, stable tabular baseline using `train.csv`/`test.csv` with scikit-learn `LogisticRegression` and standard preprocessing when the external submissions aren’t available. This produces a properly formatted `submission.csv` with `image_name,target` and probabilities in `[0,1]`. The changes are focused on unblocking runtime and yielding a reasonable AUC-oriented probability model without changing paths beyond using the provided `/kaggle/input/...` data.'
- What this solution (achieved 0.76834) has done: 'Your current score (0.66776) is far below the target (0.93319), so we should improve the model while keeping the same “tabular LogisticRegression baseline when external blend isn’t available” core logic. The biggest score gain available with minimal semantic change is to prevent patient-level leakage by splitting/grouping by `patient_id` and to add a few strong, still-tabular features already present in `train.csv`/`test.csv` (e.g., `patient_id` and simple age missingness). We also switch the logistic solver to one that is robust with high-cardinality one-hot features (`saga`) and slightly increase `max_iter` to ensure convergence; this keeps the model family and training approach unchanged. Finally, we keep the external-blend path intact and still write a valid `submission.csv`.'
- What this solution (achieved 0.75423) has done: 'Your gap to target is large (0.76834 vs 0.93319, higher-is-better), so we should improve AUC while keeping your core “tabular LogisticRegression fallback when external blend isn’t available” logic unchanged. The smallest, most relevant boost is to add a couple of strong, leakage-safe tabular signals already in `train.csv`/`test.csv`: a binned age feature (nonlinear age effect) and a couple of simple interaction/missingness indicators, while keeping the same model family, preprocessing style, and training loop. Additionally, calibrating regularization strength (`C`) slightly upward can help this linear model fit these sparse one-hot features better without changing the approach. The external-blend path and submission writing remain intact.'
- What this solution (achieved 0.66713) has done: 'We keep your exact fallback modeling approach (single scikit-learn `LogisticRegression` with the same preprocessing style) but make two minimal, score-relevant adjustments that typically lift AUC for this competition’s tabular baseline: (1) stop one-hot encoding the high-cardinality `patient_id` (it tends to overfit and harms generalization on the public LB), and (2) add a couple of simple leakage-safe tabular features that are strongly predictive here (`age*sex` interaction and a missingness flag for `anatom_site_general_challenge`). Everything else—including the external-blend path, logistic model family, and submission writing—stays the same, and the script still runs end-to-end to produce a valid `submission.csv`.'
- What this solution (achieved 0.61738) has done: 'Your current AUC (0.66713) is far below the target (0.93319), so we should make a small, legitimate improvement without changing the core “tabular LogisticRegression fallback” approach. The biggest low-risk gain here is to stop using `class_weight="balanced"` (it optimizes a different objective and often hurts ROC-AUC ranking) and to add `diagnosis` as a categorical feature (it exists in `train.csv` and is a strong tabular predictor) while safely handling its absence in `test.csv`. We also add a simple `patient_image_count` feature derived from the available metadata only (no leakage from labels), which helps calibration/ranking while preserving the same single-model pipeline. All paths, the external-blend logic, and submission writing remain unchanged.'
- What this solution (achieved 0.62347) has done: 'Your current score (0.617) is far below the target (0.933), so we should improve AUC while keeping your same core “single LogisticRegression tabular fallback (or external blend if available)” approach. The biggest low-risk gain for AUC here is to prevent overfitting to rare `diagnosis` categories by collapsing infrequent diagnoses into an “other” bucket using only training frequencies, while keeping the same model and preprocessing. In addition, scaling/normalizing the numeric features (especially `patient_image_count`) typically improves LogisticRegression stability and ranking with minimal semantic change. These are small, legitimate adjustments that preserve your pipeline structure and still produce a valid `submission.csv`.'
- What this solution (achieved 0.72364) has done: 'We keep your single `LogisticRegression` tabular fallback unchanged in spirit, but make two minimal, score-relevant fixes to improve ROC-AUC ranking. First, we stop using the leakage-prone `diagnosis` column (it’s train-only and its “unknown in test” handling tends to hurt generalization), while retaining the rest of your engineered metadata features. Second, we add a very small, legitimate unsupervised signal from the provided metadata only: a target-agnostic `patient_test_count` feature derived from `test.csv` patient frequency (this often helps rank-order risk without changing the model family). The external-blend path and submission writing remain intact.'
- What this solution (achieved 0.70714) has done: 'Your current score (0.72364) is well below the target (0.93319), so we should make a small, legitimate improvement while keeping the same core “single LogisticRegression tabular fallback (or external blend if available)” approach. The most impactful minimal change here is to explicitly encode information in `patient_id` without one-hot overfitting by adding two simple, leakage-safe aggregates computed from metadata only: each patient’s mean age and most common anatomical site in train, then mapping those to both train/test. In addition, because AUC is rank-based, we can safely use cross-validated out-of-fold prediction to choose a light monotonic calibration (logit scaling) and apply it to test probabilities; this does not change the model family or training loop, but often improves ranking slightly. All paths remain unchanged, the external blend logic is preserved, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.70715) has done: 'We keep your exact “external blend if available, otherwise single LogisticRegression tabular model + monotonic logit calibration” core logic, but make two minimal, score-relevant fixes that typically lift ROC-AUC here. First, we add `patient_id` frequency features computed separately for train and test (you already do this for test-only; adding the symmetric train-only frequency is leakage-safe and often improves ranking). Second, we make the one-hot encoding slightly more robust by grouping rare `anatom_site_general_challenge` categories into an “Other” bucket using train frequencies only, reducing noisy sparsity without changing the model family or training loop. Everything else (paths, model, CV grouping, calibration, and submission writing) stays the same and still outputs a valid `submission.csv`.'
- What this solution (achieved 0.70615) has done: 'We keep your exact fallback approach (single LogisticRegression on engineered tabular metadata + GroupKFold OOF logit calibration) but fix two score-relevant issues that can suppress AUC: (1) your `StandardScaler(with_mean=True)` can break/behave poorly with sparse one-hot outputs from `ColumnTransformer`, so we switch to `with_mean=False` (safe for sparse) to stabilize training; and (2) `patient_mode_site` currently uses already-collapsed `site_col`, which can wash out signal—compute the patient mode from the original raw site before rare-category collapsing, then apply the same rare collapsing only to the main `site_col`. These are minimal changes that preserve model family/training semantics, but typically improve ranking reliability and thus ROC-AUC. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.70556) has done: 'We’re far below the target AUC, so the smallest legitimate way to move upward while preserving your exact “single tabular LogisticRegression + GroupKFold OOF logit calibration” core logic is to reduce overfitting from high-cardinality categorical one-hot features. I keep your feature set and training loop intact, but make OneHotEncoder less brittle by (1) grouping rare categories in the derived categorical interaction columns (which can explode cardinality) using train-frequency thresholds, and (2) using `min_frequency` in `OneHotEncoder` to automatically bucket infrequent categories during encoding. These changes preserve evaluation semantics and model family, but usually improve public-LB AUC by stabilizing ranks. The external-blend path and submission writing remain unchanged.'
- What this solution (achieved 0.70332) has done: 'Your current AUC (0.70556) is far below the target (0.93319), so we should make small, legitimate improvements without changing the overall “single tabular LogisticRegression fallback + GroupKFold OOF logit calibration” approach. The biggest issue suppressing AUC here is that the model never sees `patient_id` at all (it’s in `base_features` but then omitted from both numeric and categorical transformers), so we include `patient_id` as a categorical feature while controlling overfitting by collapsing rare patients into an “Other” bucket using only train frequencies. Additionally, we simplify duplicated count features (`patient_test_count` vs `patient_test_count_self`) to reduce noise/collinearity while keeping the same model family and training semantics. These changes keep the architecture/training loop intact and should move AUC upward toward the target with minimal risk.'
- What this solution (achieved 0.31222) has done: 'Your current AUC (0.70332) is far below the target (0.93319), so we should make a small, safe improvement without changing the overall “single tabular LogisticRegression fallback + GroupKFold OOF logit calibration” approach. The biggest likely issue is that `patient_id` and patient-derived aggregates are being computed from the original IDs, but then `patient_id` is collapsed to `"Other"` **afterwards**, making these aggregates inconsistent with the model’s effective grouping signal. I collapse `patient_id` first into a stable `patient_id_collapsed` used consistently for grouping, CV groups, counts, and per-patient aggregates, while keeping the same model, preprocessing, and calibration steps. This should improve generalization and ranking (AUC) with minimal semantic change and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = _find_first_existing(DATA_ROOT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate Kaggle input data directory.")

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

assert {"image_name", "target"}.issubset(
    sub.columns
), "sample_submission.csv must have image_name,target"
assert "target" in train_df.columns, "train.csv must have target column"

train_df.shape, test_df.shape, sub.shape



## === cell 2
EXT_ROOT = "../input/public-submission-melanoma-95"

ext_files = {
    "public_sub_mean_9533": "submission_mean.csv",
    "public_sub_median_9533": "submission_median.csv",
    "public_sub_meta_ens_9577": "external_meta_ensembled.csv",
    "public_sub_9581": "submission_9581.csv",
    "public_sub_tabular": "submission_tabular_only.csv",
    "public_sub_9619": "submission_9619.csv",
    "public_sub_9606": "submission_9606.csv",
    "public_sub_9603": "submission_9603.csv",
}

loaded_ext = {}
if os.path.isdir(EXT_ROOT):
    for k, fn in ext_files.items():
        p = os.path.join(EXT_ROOT, fn)
        if os.path.exists(p):
            df = pd.read_csv(p)
            if {"image_name", "target"}.issubset(df.columns):
                loaded_ext[k] = df[["image_name", "target"]].copy()

loaded_ext.keys()



## === cell 3
use_external_blend = all(
    k in loaded_ext
    for k in [
        "public_sub_meta_ens_9577",
        "public_sub_9619",
        "public_sub_9606",
        "public_sub_tabular",
    ]
)

if use_external_blend:
    base = sub[["image_name"]].copy()
    for k in [
        "public_sub_meta_ens_9577",
        "public_sub_9619",
        "public_sub_9606",
        "public_sub_tabular",
    ]:
        base = base.merge(
            loaded_ext[k], on="image_name", how="left", suffixes=("", f"_{k}")
        )

    for k in [
        "public_sub_meta_ens_9577",
        "public_sub_9619",
        "public_sub_9606",
        "public_sub_tabular",
    ]:
        col = "target" if k == "public_sub_meta_ens_9577" else f"target_{k}"
        if col not in base.columns:
            base[col] = np.nan
        base[col] = base[col].astype(float).fillna(0.5)

    sub["target"] = (
        base["target"] * 0.40
        + base["target_public_sub_9619"] * 0.25
        + base["target_public_sub_9606"] * 0.15
        + base["target_public_sub_tabular"] * 0.20
    ).clip(0.0, 1.0)
else:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    tr = train_df.copy()
    te = test_df.copy()

    site_col = "anatom_site_general_challenge"
    raw_site_tr = tr[site_col].copy()

    def _mode_or_nan(s):
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return s.value_counts().index[0]

    def _collapse_rare_train_categories(
        train_series, test_series, min_count=200, other="Other"
    ):
        vc = train_series.value_counts(dropna=False)
        rare = vc[vc < min_count].index
        train_series = train_series.where(~train_series.isin(rare), other=other)
        test_series = test_series.where(~test_series.isin(rare), other=other)
        return train_series, test_series

    tr["patient_id"] = tr["patient_id"].astype(str).fillna("NA")
    te["patient_id"] = te["patient_id"].astype(str).fillna("NA")
    tr["patient_id_collapsed"], te["patient_id_collapsed"] = (
        _collapse_rare_train_categories(
            tr["patient_id"], te["patient_id"], min_count=5, other="Other"
        )
    )

    patient_mode_site = (
        tr.assign(_raw_site=raw_site_tr)
        .groupby("patient_id_collapsed")["_raw_site"]
        .apply(_mode_or_nan)
    )

    tr_age = tr[["patient_id_collapsed", "age_approx"]].copy()
    patient_mean_age = tr_age.groupby("patient_id_collapsed")["age_approx"].mean()

    site_counts = tr[site_col].value_counts(dropna=True)
    rare_sites = site_counts[site_counts < 50].index
    tr[site_col] = tr[site_col].where(~tr[site_col].isin(rare_sites), other="Other")
    te[site_col] = te[site_col].where(~te[site_col].isin(rare_sites), other="Other")

    global_age_median = tr["age_approx"].median()

    base_features = [
        "patient_id_collapsed",
        "sex",
        "age_approx",
        site_col,
    ]

    X = tr[base_features].copy()
    y = tr["target"].astype(int).copy()
    X_test = te.reindex(columns=base_features).copy()

    X["age_missing"] = X["age_approx"].isna().astype(np.int8)
    X_test["age_missing"] = X_test["age_approx"].isna().astype(np.int8)

    X["site_missing"] = X[site_col].isna().astype(np.int8)
    X_test["site_missing"] = X_test[site_col].isna().astype(np.int8)

    age_bins = [-1, 20, 30, 40, 50, 60, 70, 80, 120]
    X["age_bin"] = pd.cut(X["age_approx"], bins=age_bins)
    X_test["age_bin"] = pd.cut(X_test["age_approx"], bins=age_bins)

    X["sex_x_site"] = (
        X["sex"].astype("object").fillna("NA").astype(str)
        + "__"
        + X[site_col].astype("object").fillna("NA").astype(str)
    )
    X_test["sex_x_site"] = (
        X_test["sex"].astype("object").fillna("NA").astype(str)
        + "__"
        + X_test[site_col].astype("object").fillna("NA").astype(str)
    )

    X["sex_x_agebin"] = (
        X["sex"].astype("object").fillna("NA").astype(str)
        + "__"
        + X["age_bin"].astype("object").fillna("NA").astype(str)
    )
    X_test["sex_x_agebin"] = (
        X_test["sex"].astype("object").fillna("NA").astype(str)
        + "__"
        + X_test["age_bin"].astype("object").fillna("NA").astype(str)
    )

    train_counts = tr["patient_id_collapsed"].value_counts()
    X["patient_image_count"] = (
        X["patient_id_collapsed"].map(train_counts).fillna(1).astype(np.float32)
    )
    X_test["patient_image_count"] = (
        X_test["patient_id_collapsed"].map(train_counts).fillna(1).astype(np.float32)
    )

    test_counts = te["patient_id_collapsed"].value_counts()
    X["patient_test_count"] = (
        X["patient_id_collapsed"].map(test_counts).fillna(0).astype(np.float32)
    )
    X_test["patient_test_count"] = (
        X_test["patient_id_collapsed"].map(test_counts).fillna(0).astype(np.float32)
    )

    X["patient_train_count"] = (
        X["patient_id_collapsed"].map(train_counts).fillna(1).astype(np.float32)
    )
    X_test["patient_train_count"] = (
        X_test["patient_id_collapsed"].map(train_counts).fillna(1).astype(np.float32)
    )

    X["patient_mean_age"] = (
        X["patient_id_collapsed"]
        .map(patient_mean_age)
        .fillna(global_age_median)
        .astype(np.float32)
    )
    X_test["patient_mean_age"] = (
        X_test["patient_id_collapsed"]
        .map(patient_mean_age)
        .fillna(global_age_median)
        .astype(np.float32)
    )

    X["patient_mode_site"] = (
        X["patient_id_collapsed"].map(patient_mode_site).fillna("NA").astype(str)
    )
    X_test["patient_mode_site"] = (
        X_test["patient_id_collapsed"].map(patient_mode_site).fillna("NA").astype(str)
    )

    X["sex_x_site"], X_test["sex_x_site"] = _collapse_rare_train_categories(
        X["sex_x_site"], X_test["sex_x_site"], min_count=200, other="Other"
    )
    X["sex_x_agebin"], X_test["sex_x_agebin"] = _collapse_rare_train_categories(
        X["sex_x_agebin"], X_test["sex_x_agebin"], min_count=200, other="Other"
    )
    X["patient_mode_site"], X_test["patient_mode_site"] = (
        _collapse_rare_train_categories(
            X["patient_mode_site"],
            X_test["patient_mode_site"],
            min_count=200,
            other="Other",
        )
    )

    numeric_features = [
        "age_approx",
        "age_missing",
        "site_missing",
        "patient_image_count",
        "patient_test_count",
        "patient_train_count",
        "patient_mean_age",
    ]
    categorical_features = [
        "patient_id_collapsed",
        "sex",
        site_col,
        "age_bin",
        "sex_x_site",
        "sex_x_agebin",
        "patient_mode_site",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False, with_std=True)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        C=2.0,
        class_weight=None,
        n_jobs=None,
        random_state=42,
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    gkf = GroupKFold(n_splits=5)
    groups = tr["patient_id_collapsed"].astype(str).values
    oof = np.zeros(len(X), dtype=np.float64)

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        model_cv = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        model_cv.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model_cv.predict_proba(X.iloc[va_idx])[:, 1]

    eps = 1e-6
    oof_clip = np.clip(oof, eps, 1 - eps)
    logit_oof = np.log(oof_clip / (1 - oof_clip)).reshape(-1, 1)

    cal = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        random_state=42,
    )
    cal.fit(logit_oof, y.values)

    model.fit(X, y)
    proba_test = model.predict_proba(X_test)[:, 1]
    proba_test_clip = np.clip(proba_test, eps, 1 - eps)
    logit_test = np.log(proba_test_clip / (1 - proba_test_clip)).reshape(-1, 1)
    proba_cal = cal.predict_proba(logit_test)[:, 1]

    try:
        auc_oof = roc_auc_score(y, oof)
        auc_oof_cal = roc_auc_score(y, cal.predict_proba(logit_oof)[:, 1])
        print(f"OOF AUC raw: {auc_oof:.5f} | OOF AUC calibrated: {auc_oof_cal:.5f}")
    except Exception as e:
        print("AUC check skipped:", repr(e))

    sub["target"] = np.clip(proba_cal, 0.0, 1.0)



## === cell 4
sub = sub[["image_name", "target"]].copy()
sub["target"] = sub["target"].astype(float).clip(0.0, 1.0)

assert len(sub) == len(test_df), "Submission must have one row per test image."
assert sub["image_name"].isna().sum() == 0, "image_name cannot be null."
assert sub["target"].isna().sum() == 0, "target cannot be null."

sub.head()



## === cell 5
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {sub.shape} and columns {list(sub.columns)}")
print(sub["target"].describe())
