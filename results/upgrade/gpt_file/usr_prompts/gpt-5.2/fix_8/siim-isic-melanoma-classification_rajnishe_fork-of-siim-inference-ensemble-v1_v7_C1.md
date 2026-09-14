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

0.9359800206055708

# 6. Current score

0.74393

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66776) has done: 'Your notebook fails because it expects external prediction CSVs from `../input/rcsiimpreds/`, which are not present in this Kaggle environment, causing `FileNotFoundError` and cascading `NameError`s. To make it run end-to-end and still follow the same “blend multiple model predictions” core idea, I replace those missing inputs with a lightweight, deterministic out-of-fold style baseline using only the provided metadata (sex/age/anatom_site) and a Logistic Regression model. This produces valid probability predictions for the required `image_name,target` submission format and writes `submission.csv`. I also make path handling robust by reading `sample_submission.csv` from the available `/kaggle/data/` tree.'
- What this solution (achieved 0.65669) has done: 'Your current 0.66776 score is far below the 0.93598 target, so we need a real lift while keeping the same “metadata-only LogisticRegression fallback” core. The biggest win with minimal semantic change is to make the train/validation split patient-wise and use out-of-fold (OOF) predictions to learn a simple probability calibration mapping (isotonic), which usually improves AUC for this competition’s metadata baseline without changing the underlying model. I also add a tiny, safe feature: a missing-age indicator (still metadata-only) which often helps without altering the training approach. Finally, I keep the same submission alignment with `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.65654) has done: 'To move your AUC closer to the 0.93598 target without changing the core “metadata-only LogisticRegression + patient-wise CV + isotonic calibration” approach, I make two minimal, high-impact fixes: (1) prevent isotonic overfitting by fitting it on out-of-fold predictions and applying it only to the model’s *raw logit scores* (monotonic mapping preserves ranking, typically improving AUC stability), and (2) strengthen the logistic regression slightly (same model family) via a small `C` tweak while keeping everything else intact. I also add a deterministic, patient-wise fold assignment to remove any split nondeterminism. The pipeline still run end-to-end and write a valid `submission.csv` with the required columns and alignment to `sample_submission.csv`.'
- What this solution (achieved 0.69248) has done: 'Your current score (0.65654) is far below the target (0.93598), so we should increase AUC with minimal, low-risk changes while keeping the same metadata-only LogisticRegression + GroupKFold + isotonic calibration core. The largest safe gain here is to add a small set of well-known high-signal metadata features for this specific competition (per-patient lesion counts and basic interactions like age×sex), without changing the model family or training loop. I also make the fold splitting deterministic and stratified by target at the patient level (still group-wise, still 5 folds) to improve the quality of the OOF scores used by isotonic calibration. Submission writing/alignment stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.74419) has done: 'Your current AUC (0.69248) is far below the target (0.93598), so we should improve it with the smallest safe changes while keeping the same metadata-only LogisticRegression + patient-wise folds + isotonic calibration core. The main issue is that your engineered “patient_lesion_count” features for the test set are computed using only test rows, which makes them not comparable to train and can hurt generalization; we recompute patient-level aggregates on the concatenated train+test patient table and then map them back. Second, we calibrate the *same* score type the calibrator was trained on by fitting isotonic on base-model `predict_proba` (probabilities) rather than mixing `decision_function` at train-time with probabilities at test-time. These two fixes preserve the model family, training loop, and overall semantics but should move AUC upward toward the target while remaining deterministic and within Kaggle constraints.'
- What this solution (achieved 0.74393) has done: 'We’re far below the target AUC (0.744 vs 0.936), so we should improve ranking quality with minimal, low-risk changes while keeping your metadata-only LogisticRegression + patient-wise CV + isotonic calibration core intact. The biggest issue is that isotonic calibration is being trained on out-of-fold predictions from fold-specific models, but then applied to predictions from a different (full-data) model—this distribution shift can hurt AUC; we instead generate calibrated test predictions by averaging fold models’ calibrated outputs (same model family, same training loop, just consistent application). We also switch the fold base predictions from `predict_proba` to `decision_function` for isotonic fitting/transforming (still LogisticRegression), which often yields a cleaner monotonic mapping and slightly better AUC stability. Finally, we keep the same features and submission alignment, and still write `submission.csv`.'
- What this solution (achieved 0.74393) has done: 'We’re far below the target AUC, so we should improve ranking quality while keeping your exact metadata-only LogisticRegression + patient-wise CV + isotonic calibration core. The smallest high-impact fix is to correct the isotonic calibration direction: isotonic expects a score where larger means “more likely malignant”, but `LogisticRegression.decision_function` is oriented to whatever class is `classes_[1]`, and with `y` in {0,1} that’s usually fine but can silently flip if anything changes—so we explicitly align the decision scores to the positive class by using `predict_proba[:, 1]` consistently for both OOF and test before isotonic fitting/transforming (still monotonic mapping, same calibrator). Additionally, we reduce fold-to-fold variance by using the same pre-fitted `preprocessor` object via cloning inside each fold (no logic change, just avoids any accidental shared-state issues) and we keep everything deterministic. This should move the score upward toward the target without changing the overall approach or submission semantics.'

# 9. Code solution

## === cell 0
"""
submit of only B3 B4 & B5 models

NOTE (bugfix for this environment):
The original notebook blended multiple external prediction CSVs from ../input/rcsiimpreds/,
but that dataset is not available here, causing FileNotFoundError and no submission output.
To keep the core semantic intent (produce probabilistic predictions and output submission.csv),
we fall back to a minimal, deterministic metadata-only model when those files are missing.
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = _first_existing_path(
    [
        "/kaggle/data/siim-isic-melanoma-classification",
        "/kaggle/input/siim-isic-melanoma-classification",
        "/kaggle/data",
        "/kaggle/input",
    ]
)

if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate Kaggle data root in expected locations.")

train_csv = _first_existing_path(
    [
        os.path.join(DATA_ROOT, "train.csv"),
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
    ]
)
test_csv = _first_existing_path(
    [
        os.path.join(DATA_ROOT, "test.csv"),
        "/kaggle/data/test.csv",
        "/kaggle/input/test.csv",
    ]
)
sample_sub_csv = _first_existing_path(
    [
        os.path.join(DATA_ROOT, "sample_submission.csv"),
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_sub = pd.read_csv(sample_sub_csv)

for c in ["image_name"]:
    if c not in test_df.columns or c not in sample_sub.columns:
        raise ValueError(f"Missing required column '{c}' in test/sample_submission.")
if "target" not in train_df.columns:
    raise ValueError("Missing 'target' in train.csv; cannot train fallback model.")

print("DATA_ROOT:", DATA_ROOT)
print(
    "train_df:",
    train_df.shape,
    "test_df:",
    test_df.shape,
    "sample_sub:",
    sample_sub.shape,
)



## === cell 1
RC_PREDS_DIR = _first_existing_path(
    [
        "../input/rcsiimpreds",  # original path from the notebook
        "/kaggle/input/rcsiimpreds",
    ]
)


def _safe_read_pred(path, rename_to=None):
    if path is None or (not os.path.exists(path)):
        return None
    df = pd.read_csv(path)
    if rename_to is not None and "target" in df.columns:
        df = df.rename(columns={"target": rename_to})
    return df


use_external_blend = False
if RC_PREDS_DIR is not None and os.path.isdir(RC_PREDS_DIR):
    rep = os.path.join(RC_PREDS_DIR, "sub_EfficientNetB3_384_9460.csv")
    if os.path.exists(rep):
        use_external_blend = True

print("RC_PREDS_DIR:", RC_PREDS_DIR, "use_external_blend:", use_external_blend)



## === cell 2
if use_external_blend:
    pred_b3 = pd.read_csv(os.path.join(RC_PREDS_DIR, "sub_EfficientNetB3_384_9460.csv"))
    pred_b4 = pd.read_csv(os.path.join(RC_PREDS_DIR, "sub_EfficientNetB4_384_9498.csv"))
    pred_b5 = pd.read_csv(os.path.join(RC_PREDS_DIR, "sub_EfficientNetB5_384_9454.csv"))
    pred_b6 = pd.read_csv(os.path.join(RC_PREDS_DIR, "sub_EfficientNetB6_384_9481.csv"))
    pred_cw_b4 = pd.read_csv(
        os.path.join(RC_PREDS_DIR, "sub_EfficientNetB4_CW_384_9457.csv")
    )
    pred_cw_b4.rename(columns={"target": "target_cw_b4"}, inplace=True)

    pred_512_B6 = pd.read_csv(os.path.join(RC_PREDS_DIR, "sub_B5_512_3fold_9466.csv"))
    pred_512_B6.rename(columns={"target": "target_B6_512"}, inplace=True)

    pred_tta_b3 = pd.read_csv(os.path.join(RC_PREDS_DIR, "siim_tta_b3_9458.csv"))
    pred_tta_b3.rename(columns={"target": "target_tta_b3"}, inplace=True)

    pred_tta_b4 = pd.read_csv(os.path.join(RC_PREDS_DIR, "siim_tta_b4_9473.csv"))
    pred_tta_b4.rename(columns={"target": "target_tta_b4"}, inplace=True)

    result_tta = pd.merge(
        pred_b3, pred_b4, on="image_name", suffixes=("_tta_b3", "_tta_b4")
    )

    result1 = pd.merge(pred_b3, pred_b4, on="image_name", suffixes=("_b3", "_b4"))
    result2 = pd.merge(pred_b5, pred_b6, on="image_name", suffixes=("_b5", "_b6"))
    semi_final = pd.merge(result1, result2, on="image_name")

    result3 = pd.merge(
        pred_cw_b4, pred_512_B6, on="image_name", suffixes=("_cw_b4", "_B6_512")
    )
    final = pd.merge(semi_final, result3, on="image_name")

    final = pd.merge(final, result_tta, on="image_name")

    pred_kr_b3 = pd.read_csv(
        os.path.join(RC_PREDS_DIR, "KR_sub_EfficientNetB3_512_9520.csv")
    )
    pred_kr_b3.rename(columns={"target": "target_kr_b3"}, inplace=True)

    pred_kr_b4 = pd.read_csv(
        os.path.join(RC_PREDS_DIR, "KR_sub_EfficientNetB4_512_9499.csv")
    )
    pred_kr_b4.rename(columns={"target": "target_kr_b4"}, inplace=True)

    pred_kr_eb3 = pd.read_csv(os.path.join(RC_PREDS_DIR, "KR_sub_eb3_512_9554.csv"))
    pred_kr_eb3.rename(columns={"target": "target_kr_eb3"}, inplace=True)

    kr_result = pd.merge(
        pred_kr_b3, pred_kr_b4, on="image_name", suffixes=("_b3", "_b4")
    )
    kr_result = pd.merge(
        kr_result, pred_kr_eb3, on="image_name", suffixes=("_b3", "_b4")
    )

    final = pd.merge(final, kr_result, on="image_name")

    pred_256_b4 = pd.read_csv(
        os.path.join(RC_PREDS_DIR, "sub_EfficientNetB4_256_9496.csv")
    )
    pred_256_b4.rename(columns={"target": "target_256_b4"}, inplace=True)

    final = pd.merge(final, pred_256_b4, on="image_name")

    final["target"] = (
        (final.target_b4)
        + (final.target_b6)
        + final.target_tta_b4
        + final.target_kr_b3
        + final.target_kr_b4
        + final.target_kr_eb3
        + final.target_256_b4
    ) / 7.0

    submit_file = final[["image_name", "target"]].copy()
else:
    submit_file = None



## === cell 3
if not use_external_blend:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.isotonic import IsotonicRegression
    from sklearn.base import clone

    train_df = train_df.copy()
    test_df = test_df.copy()

    all_patients = pd.concat(
        [
            train_df[["patient_id", "image_name"]].copy(),
            test_df[["patient_id", "image_name"]].copy(),
        ],
        axis=0,
        ignore_index=True,
    )
    all_patients["patient_id"] = all_patients["patient_id"].astype(str)
    pid_counts = (
        all_patients.groupby("patient_id")["image_name"]
        .count()
        .astype(np.int16)
        .rename("patient_lesion_count_all")
    )

    for df in (train_df, test_df):
        df["patient_id"] = df["patient_id"].astype(str)
        df["sex"] = df["sex"].astype("object")
        df["anatom_site_general_challenge"] = df[
            "anatom_site_general_challenge"
        ].astype("object")

        df["age_missing"] = df["age_approx"].isna().astype(np.int8)
        age_filled = df["age_approx"].fillna(df["age_approx"].median())
        df["age_x10"] = (age_filled / 10.0).astype(np.float32)
        df["age_sq"] = (age_filled**2).astype(np.float32)

        df["patient_lesion_count"] = (
            df["patient_id"].map(pid_counts).fillna(1).astype(np.int16)
        )
        df["patient_lesion_count_log1p"] = np.log1p(df["patient_lesion_count"]).astype(
            np.float32
        )

        df["sex_is_male"] = (df["sex"].fillna("").str.lower() == "male").astype(np.int8)
        df["age_x_sex_male"] = (df["age_x10"] * df["sex_is_male"]).astype(np.float32)

    train_df = train_df.sort_values(["patient_id", "image_name"]).reset_index(drop=True)

    features = [
        "sex",
        "age_approx",
        "age_missing",
        "age_x10",
        "age_sq",
        "anatom_site_general_challenge",
        "patient_lesion_count",
        "patient_lesion_count_log1p",
        "sex_is_male",
        "age_x_sex_male",
    ]

    X_train = train_df[features].copy()
    y_train = train_df["target"].astype(int).values
    X_test = test_df[features].copy()

    numeric_features = [
        "age_approx",
        "age_missing",
        "age_x10",
        "age_sq",
        "patient_lesion_count",
        "patient_lesion_count_log1p",
        "sex_is_male",
        "age_x_sex_male",
    ]
    categorical_features = ["sex", "anatom_site_general_challenge"]

    preprocessor = ColumnTransformer(
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
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        n_jobs=None,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        C=2.0,
    )

    base_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    patient_tbl = (
        train_df.groupby("patient_id")["target"]
        .max()
        .reset_index()
        .rename(columns={"target": "patient_target"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    patient_tbl["fold"] = -1
    for f, (_, va_pat_idx) in enumerate(
        skf.split(
            patient_tbl["patient_id"].values, patient_tbl["patient_target"].values
        )
    ):
        patient_tbl.loc[va_pat_idx, "fold"] = f

    pid_to_fold = dict(
        zip(patient_tbl["patient_id"].values, patient_tbl["fold"].values)
    )
    folds = train_df["patient_id"].map(pid_to_fold).values.astype(int)

    oof_score = np.zeros(len(train_df), dtype=np.float64)
    test_pred_accum = np.zeros(len(test_df), dtype=np.float64)

    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        m = clone(base_model)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])

        oof_score[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1].astype(
            np.float64
        )

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_score, y_train)

    for f in range(5):
        tr_idx = np.where(folds != f)[0]

        m = clone(base_model)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])

        test_score = m.predict_proba(X_test)[:, 1].astype(np.float64)
        test_pred_accum += calibrator.transform(test_score).astype(np.float64) / 5.0

    submit_file = pd.DataFrame(
        {
            "image_name": test_df["image_name"].values,
            "target": test_pred_accum,
        }
    )

    submit_file = sample_sub[["image_name"]].merge(
        submit_file, on="image_name", how="left"
    )
    if submit_file["target"].isna().any():
        submit_file["target"] = submit_file["target"].fillna(
            float(np.nanmean(test_pred_accum))
        )

submit_file.head()



## === cell 4
submit_file = submit_file[["image_name", "target"]].copy()
submit_file["target"] = submit_file["target"].clip(0.0, 1.0)

out_path = "submission.csv"
submit_file.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape", submit_file.shape)
print(submit_file.head())
