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

0.9331527615469468

# 6. Current score

0.74128

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.77397) has done: 'The main issue is that this notebook depends on external prediction CSVs in `../input/rcsiimpreds/`, which are not available in your environment, causing `FileNotFoundError` and cascading `NameError`s. To keep the “ensemble submission” core logic intact while making it runnable, I add a small loader that uses those files if they exist, but otherwise falls back to a simple metadata-only model trained from the provided `train.csv` and applied to `test.csv`. This guarantees an end-to-end run and produces a valid `submission.csv` with the required columns and ordering. The fallback uses a lightweight sklearn pipeline (logistic regression with one-hot encoding + imputation) so it stays within the installed packages and the 600s budget.'
- What this solution (achieved 0.78005) has done: 'Your current fallback is a metadata-only logistic regression, which is inherently capped and explains the 0.77397 AUC gap to the 0.933 target. To move toward the target without changing the overall “fallback model” approach, I keep the same sklearn Pipeline + LogisticRegression core logic but (1) add a small set of proven, competition-relevant engineered metadata features (age missing flag + one-hot for anatom/sex + patient_id frequency) and (2) switch to a slightly stronger regularized solver setup (same model family) that typically improves ranking/AUC while staying fast. These are minimal, legitimate changes that should increase AUC materially but won’t introduce any new data sources or change submission semantics. The ensemble branch is left untouched and still be used automatically if the external prediction files exist.'
- What this solution (achieved 0.75419) has done: 'Your current score (0.78005) is well below the target (0.93315), so we should legitimately increase AUC while keeping your fallback “metadata-only LogisticRegression pipeline” intact. The smallest high-impact change for this specific competition is to prevent patient leakage by doing out-of-fold (OOF) target encoding for high-cardinality `patient_id` (and optionally `anatom_site_general_challenge`), which improves ranking without changing the model family or training loop semantics. I keep the same Pipeline + LogisticRegression, but replace the leaky `patient_count` with OOF mean target encodings (plus a test-time mapping), which is a minimal feature-engineering adjustment directly aimed at AUC. The external ensemble branch remains untouched and still run if those files exist, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.77048) has done: 'We need to improve AUC from 0.754 toward 0.933 (higher is better), so we should strengthen the existing metadata-only LogisticRegression fallback without changing its core model family. The biggest minimal win here is to add a few well-known, leakage-safe engineered metadata features that improve ranking: log-transformed patient frequency, age bins, and interaction between sex and anatom site, while keeping the same pipeline/training approach. I also make the target encoding slightly more robust by computing the global mean inside each fold (instead of using the full-data mean) to avoid subtle leakage and improve generalization. The external ensemble branch stays untouched and still be used automatically if those files exist, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.68645) has done: 'We’re still far below the target AUC (0.77048 vs 0.93315, higher is better), so we should make a small, legitimate improvement to the existing metadata-only LogisticRegression fallback without changing the model family or training semantics. The biggest low-risk gain here is to add a second leakage-safe OOF target-encoded feature for `patient_id`: a smoothed OOF encoding of the within-patient lesion index (capturing “first vs later lesion” signal) plus its test-time counterpart; this keeps the same pipeline and just adds one numeric feature. I also fix a small bug in `available_preds` where `pred_kr_eb3` incorrectly points to `pred_kr_b4` (harmless now but correctness-critical if files appear). Everything else (reading paths, ensemble branch, logistic regression, and submission writing) stays the same.'
- What this solution (achieved 0.72484) has done: 'Your current AUC (0.68645) is far below the target (0.93315), so we should make a small, safe improvement in the fallback metadata LogisticRegression without changing the model family or training semantics. The biggest minimal win here is to avoid patient-level leakage/shift by generating OOF target encodings with **GroupKFold by patient_id** (instead of StratifiedKFold), which typically improves public LB AUC for this competition while keeping the same “OOF target encoding + LogisticRegression pipeline” core logic. I also add an OOF target encoding for `sex_x_site` (you already build this interaction; encoding it numerically is a low-risk ranking boost). Finally, I fix the `pid_rank_te` encoding helper to pass the correct train/test frames (previously it relied on a rename that did nothing), improving correctness and stability.'
- What this solution (achieved 0.7254) has done: 'We need to increase AUC (0.72484 → 0.93315), but keep your fallback LogisticRegression + OOF target encoding core intact. The smallest high-impact fix is to make the target encodings and leakage-sensitive engineered features consistent with the **same GroupKFold (by patient_id)** used for OOF, so the model doesn’t learn patient-specific prevalence in a way that fails on LB. Concretely, we (1) compute `patient_count_log1p` and `lesion_rank_log1p` in a leakage-safe OOF way (train-side computed per fold; test computed from full train mapping) and (2) slightly increase smoothing for the extremely high-cardinality `pid_rank_key` encoding to reduce noise/overfit. Everything else (paths, ensemble branch, model family, pipeline, submission writing) stays unchanged.'
- What this solution (achieved 0.74128) has done: 'I keep your existing “external-preds ensemble else metadata LogisticRegression” structure intact and only adjust the fallback metadata model to better match what works for this competition. The main minimal gain is to switch the final fit/predict step to a leakage-safe **GroupKFold (by patient_id) bagging**: train 5 LogisticRegression models on group-split folds and average test probabilities, which usually improves AUC noticeably without changing the model family or features. I also make the target-encoding strength slightly more stable by increasing smoothing for the highest-cardinality encoding (`patient_id`) to reduce fold noise/overfit while preserving the same OOF target-encoding approach. The script still run end-to-end and write a valid `submission.csv` with the required columns and ordering.'

# 9. Code solution

## === cell 0
"""
submit of only B3 B4 & B5 models
"""


## === cell 1
import os
import numpy as np
import pandas as pd



## === cell 2
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
]


def _find_existing_file(rel_paths):
    for base in DATA_ROOT_CANDIDATES:
        for rp in rel_paths:
            p = os.path.join(base, rp)
            if os.path.exists(p):
                return p
    return None


def safe_read_pred_csv(path, rename_to=None):
    """Read prediction CSV if present; else return None."""
    if path is None:
        return None
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "image_name" not in df.columns:
            for c in df.columns:
                if c.lower() in ("image", "imagename", "id"):
                    df = df.rename(columns={c: "image_name"})
                    break
        if "target" not in df.columns:
            for c in df.columns:
                if c.lower() in ("pred", "prediction", "prob", "probability"):
                    df = df.rename(columns={c: "target"})
                    break
        if rename_to is not None and "target" in df.columns:
            df = df.rename(columns={"target": rename_to})
        return df[["image_name"] + ([rename_to] if rename_to else ["target"])]
    return None




## === cell 3
rc_preds_root = "../input/rcsiimpreds"
pred_b3 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "sub_EfficientNetB3_384_9460.csv"),
    rename_to="target_b3",
)
pred_b4 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "sub_EfficientNetB4_384_9498.csv"),
    rename_to="target_b4",
)
pred_b5 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "sub_EfficientNetB5_384_9454.csv"),
    rename_to="target_b5",
)
pred_b6 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "sub_EfficientNetB6_384_9481.csv"),
    rename_to="target_b6",
)
pred_cw_b4 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "sub_EfficientNetB4_CW_384_9457.csv"),
    rename_to="target_cw_b4",
)

pred_512_B6 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "sub_B5_512_3fold_9466.csv"), rename_to="target_B6_512"
)

pred_tta_b3 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "siim_tta_b3_9458.csv"), rename_to="target_tta_b3"
)
pred_tta_b4 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "siim_tta_b4_9473.csv"), rename_to="target_tta_b4"
)

pred_kr_b3 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "KR_sub_EfficientNetB3_512_9520.csv"),
    rename_to="target_kr_b3",
)
pred_kr_b4 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "KR_sub_EfficientNetB4_512_9499.csv"),
    rename_to="target_kr_b4",
)
pred_kr_eb3 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "KR_sub_eb3_512_9554.csv"), rename_to="target_kr_eb3"
)

pred_256_b4 = safe_read_pred_csv(
    os.path.join(rc_preds_root, "sub_EfficientNetB4_256_9496.csv"),
    rename_to="target_256_b4",
)

available_preds = {
    "pred_b3": pred_b3,
    "pred_b4": pred_b4,
    "pred_b5": pred_b5,
    "pred_b6": pred_b6,
    "pred_cw_b4": pred_cw_b4,
    "pred_512_B6": pred_512_B6,
    "pred_tta_b3": pred_tta_b3,
    "pred_tta_b4": pred_tta_b4,
    "pred_kr_b3": pred_kr_b3,
    "pred_kr_b4": pred_kr_b4,
    "pred_kr_eb3": pred_kr_eb3,
    "pred_256_b4": pred_256_b4,
}
missing = [k for k, v in available_preds.items() if v is None]
print(f"Missing external prediction files for: {missing}")



## === cell 4
sample_path = _find_existing_file(
    ["sample_submission.csv", "siim-isic-melanoma-classification/sample_submission.csv"]
)
test_path = _find_existing_file(
    ["test.csv", "siim-isic-melanoma-classification/test.csv"]
)
train_path = _find_existing_file(
    ["train.csv", "siim-isic-melanoma-classification/train.csv"]
)

if sample_path is None or test_path is None or train_path is None:
    raise FileNotFoundError(
        f"Could not locate required competition CSVs. "
        f"sample_path={sample_path}, test_path={test_path}, train_path={train_path}"
    )

sample_sub = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)

assert "image_name" in sample_sub.columns and "target" in sample_sub.columns
assert "image_name" in test_df.columns
assert "target" in train_df.columns



## === cell 5
required_for_original = [
    "pred_b4",
    "pred_b6",
    "pred_tta_b4",
    "pred_kr_b3",
    "pred_kr_b4",
    "pred_kr_eb3",
    "pred_256_b4",
]
have_all = all(available_preds[k] is not None for k in required_for_original)

if have_all:
    final = sample_sub[["image_name"]].copy()
    for k in required_for_original:
        final = final.merge(available_preds[k], on="image_name", how="left")

    pred_cols = [c for c in final.columns if c.startswith("target_") and c != "target"]
    for c in pred_cols:
        if final[c].isna().any():
            final[c] = final[c].fillna(final[c].mean())

    from scipy.stats import gmean

    target_array = np.vstack([final[c].to_numpy() for c in pred_cols])
    final["target"] = gmean(target_array)
    submit_file = final[["image_name", "target"]].copy()
else:
    submit_file = None

print("Using original ensemble preds pipeline?", have_all)



## === cell 6
if submit_file is None:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, GroupKFold

    def add_oof_target_encoding(
        train_df_in: pd.DataFrame,
        test_df_in: pd.DataFrame,
        col: str,
        target_col: str = "target",
        n_splits: int = 5,
        seed: int = 0,
        min_count_smooth: int = 20,
        groups=None,
    ):
        """
        OOF target encoding; if groups provided (patient_id), use GroupKFold to avoid leakage.
        """
        tr = train_df_in[[col, target_col]].copy()
        te = test_df_in[[col]].copy()

        tr[col] = tr[col].astype("object").fillna("__MISSING__")
        te[col] = te[col].astype("object").fillna("__MISSING__")

        oof = np.zeros(len(tr), dtype=np.float32)

        full_global_mean = tr[target_col].mean()
        full_stats = tr.groupby(col)[target_col].agg(["mean", "count"])
        smooth = (
            full_stats["mean"] * full_stats["count"]
            + full_global_mean * min_count_smooth
        ) / (full_stats["count"] + min_count_smooth)
        test_enc = (
            te[col].map(smooth).fillna(full_global_mean).astype(np.float32).to_numpy()
        )

        y = tr[target_col].astype(int).to_numpy()

        if groups is None:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            split_iter = splitter.split(tr, y)
        else:
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(tr, y, groups=groups)

        for tr_idx, val_idx in split_iter:
            fold_tr = tr.iloc[tr_idx]
            fold_val = tr.iloc[val_idx]

            fold_global_mean = fold_tr[target_col].mean()
            stats = fold_tr.groupby(col)[target_col].agg(["mean", "count"])
            fold_smooth = (
                stats["mean"] * stats["count"] + fold_global_mean * min_count_smooth
            ) / (stats["count"] + min_count_smooth)
            oof[val_idx] = (
                fold_val[col]
                .map(fold_smooth)
                .fillna(fold_global_mean)
                .astype(np.float32)
            )

        return oof, test_enc

    def add_oof_group_count_log1p(
        train_ids: pd.Series,
        test_ids: pd.Series,
        groups: np.ndarray,
        n_splits: int = 5,
    ):
        train_ids = train_ids.astype("object").fillna("__MISSING__")
        test_ids = test_ids.astype("object").fillna("__MISSING__")

        splitter = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(train_ids), dtype=np.float32)

        full_counts = train_ids.value_counts()
        test_feat = np.log1p(
            test_ids.map(full_counts).fillna(0).astype(np.float32)
        ).to_numpy()

        for tr_idx, val_idx in splitter.split(train_ids, None, groups=groups):
            fold_tr_ids = train_ids.iloc[tr_idx]
            fold_counts = fold_tr_ids.value_counts()
            oof[val_idx] = np.log1p(
                train_ids.iloc[val_idx].map(fold_counts).fillna(0).astype(np.float32)
            ).to_numpy()

        return oof, test_feat

    def add_oof_lesion_rank_log1p(
        train_image: pd.Series,
        train_pid: pd.Series,
        test_image: pd.Series,
        test_pid: pd.Series,
        groups: np.ndarray,
        n_splits: int = 5,
    ):
        train_pid = train_pid.astype("object").fillna("__MISSING__")
        test_pid = test_pid.astype("object").fillna("__MISSING__")
        train_image = train_image.astype(str)
        test_image = test_image.astype(str)

        splitter = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(train_pid), dtype=np.float32)

        test_rank = (
            pd.DataFrame({"pid": test_pid.to_numpy(), "img": test_image.to_numpy()})
            .sort_values(["pid", "img"])
            .groupby("pid")
            .cumcount()
            .astype(np.int32)
        )
        test_feat = np.log1p(test_rank.to_numpy(np.float32))

        for tr_idx, val_idx in splitter.split(train_pid, None, groups=groups):
            fold_tr = pd.DataFrame(
                {
                    "pid": train_pid.iloc[tr_idx].to_numpy(),
                    "img": train_image.iloc[tr_idx].to_numpy(),
                }
            ).sort_values(["pid", "img"])
            fold_counts_rank = fold_tr.groupby("pid").cumcount().astype(np.int32)

            fold_tr = fold_tr.assign(rank=fold_counts_rank.to_numpy())
            key_to_rank = pd.Series(
                fold_tr["rank"].to_numpy(),
                index=(
                    fold_tr["pid"].astype(str) + "__" + fold_tr["img"].astype(str)
                ).to_numpy(),
            )

            val_keys = (
                train_pid.iloc[val_idx].astype(str)
                + "__"
                + train_image.iloc[val_idx].astype(str)
            ).to_numpy()
            val_rank = (
                pd.Series(val_keys)
                .map(key_to_rank)
                .fillna(0)
                .astype(np.float32)
                .to_numpy()
            )
            oof[val_idx] = np.log1p(val_rank)

        return oof, test_feat

    base_feature_cols = [
        c
        for c in ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]
        if c in test_df.columns and c in train_df.columns
    ]

    X_train = train_df[base_feature_cols].copy()
    y_train = train_df["target"].astype(int).to_numpy()
    X_test = test_df[base_feature_cols].copy()

    if "age_approx" in base_feature_cols:
        X_train["age_missing"] = X_train["age_approx"].isna().astype(np.int8)
        X_test["age_missing"] = X_test["age_approx"].isna().astype(np.int8)

        age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 200]
        X_train["age_bin"] = pd.cut(
            X_train["age_approx"], bins=age_bins, right=False
        ).astype("object")
        X_test["age_bin"] = pd.cut(
            X_test["age_approx"], bins=age_bins, right=False
        ).astype("object")

    patient_groups = None
    if "patient_id" in train_df.columns:
        patient_groups = (
            train_df["patient_id"].astype("object").fillna("__MISSING__").to_numpy()
        )

    if "patient_id" in base_feature_cols:
        oof_pid, test_pid = add_oof_target_encoding(
            train_df_in=train_df,
            test_df_in=test_df,
            col="patient_id",
            target_col="target",
            n_splits=5,
            seed=0,
            min_count_smooth=80,
            groups=patient_groups,
        )
        X_train["patient_te"] = oof_pid
        X_test["patient_te"] = test_pid

        oof_cnt, test_cnt = add_oof_group_count_log1p(
            train_ids=train_df["patient_id"],
            test_ids=test_df["patient_id"],
            groups=patient_groups,
            n_splits=5,
        )
        X_train["patient_count_log1p"] = oof_cnt
        X_test["patient_count_log1p"] = test_cnt

        oof_rank, test_rank_feat = add_oof_lesion_rank_log1p(
            train_image=train_df["image_name"],
            train_pid=train_df["patient_id"],
            test_image=test_df["image_name"],
            test_pid=test_df["patient_id"],
            groups=patient_groups,
            n_splits=5,
        )
        X_train["lesion_rank_log1p"] = oof_rank
        X_test["lesion_rank_log1p"] = test_rank_feat

        train_pid = train_df["patient_id"].astype("object").fillna("__MISSING__")
        test_pid_col = test_df["patient_id"].astype("object").fillna("__MISSING__")

        train_rank_full = (
            train_df.assign(_pid=train_pid, _img=train_df["image_name"].astype(str))
            .sort_values(["_pid", "_img"])
            .groupby("_pid")
            .cumcount()
            .astype(np.int32)
            .reindex(train_df.index)
        )
        test_rank_full = (
            test_df.assign(_pid=test_pid_col, _img=test_df["image_name"].astype(str))
            .sort_values(["_pid", "_img"])
            .groupby("_pid")
            .cumcount()
            .astype(np.int32)
            .reindex(test_df.index)
        )

        tr_key_df = pd.DataFrame(
            {
                "pid_rank_key": (
                    train_pid.astype(str) + "__" + train_rank_full.astype(str)
                ).to_numpy(),
                "target": train_df["target"].to_numpy(),
            },
            index=train_df.index,
        )
        te_key_df = pd.DataFrame(
            {
                "pid_rank_key": (
                    test_pid_col.astype(str) + "__" + test_rank_full.astype(str)
                ).to_numpy()
            },
            index=test_df.index,
        )

        oof_pidrank, test_pidrank = add_oof_target_encoding(
            train_df_in=tr_key_df,
            test_df_in=te_key_df,
            col="pid_rank_key",
            target_col="target",
            n_splits=5,
            seed=0,
            min_count_smooth=50,
            groups=patient_groups,
        )
        X_train["pid_rank_te"] = oof_pidrank
        X_test["pid_rank_te"] = test_pidrank

    if "anatom_site_general_challenge" in base_feature_cols:
        oof_site, test_site = add_oof_target_encoding(
            train_df_in=train_df,
            test_df_in=test_df,
            col="anatom_site_general_challenge",
            target_col="target",
            n_splits=5,
            seed=0,
            min_count_smooth=20,
            groups=patient_groups,
        )
        X_train["site_te"] = oof_site
        X_test["site_te"] = test_site

    if ("sex" in base_feature_cols) and (
        "anatom_site_general_challenge" in base_feature_cols
    ):
        X_train["sex_x_site"] = (
            X_train["sex"].astype("object").fillna("__MISSING__")
            + "_"
            + X_train["anatom_site_general_challenge"]
            .astype("object")
            .fillna("__MISSING__")
        )
        X_test["sex_x_site"] = (
            X_test["sex"].astype("object").fillna("__MISSING__")
            + "_"
            + X_test["anatom_site_general_challenge"]
            .astype("object")
            .fillna("__MISSING__")
        )

        tmp_tr = pd.DataFrame(
            {"sex_x_site": X_train["sex_x_site"], "target": train_df["target"]},
            index=train_df.index,
        )
        tmp_te = pd.DataFrame({"sex_x_site": X_test["sex_x_site"]}, index=test_df.index)
        oof_sxs, test_sxs = add_oof_target_encoding(
            train_df_in=tmp_tr,
            test_df_in=tmp_te,
            col="sex_x_site",
            target_col="target",
            n_splits=5,
            seed=0,
            min_count_smooth=20,
            groups=patient_groups,
        )
        X_train["sex_x_site_te"] = oof_sxs
        X_test["sex_x_site_te"] = test_sxs

    feature_cols = list(X_train.columns)

    numeric_features = [
        c
        for c in feature_cols
        if c
        in [
            "age_approx",
            "age_missing",
            "patient_te",
            "site_te",
            "patient_count_log1p",
            "lesion_rank_log1p",
            "pid_rank_te",
            "sex_x_site_te",
        ]
    ]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=4000,
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        n_jobs=None,
        random_state=0,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    if patient_groups is not None:
        splitter = GroupKFold(n_splits=5)
        test_pred_accum = np.zeros(len(X_test), dtype=np.float64)
        for fold_i, (tr_idx, val_idx) in enumerate(
            splitter.split(X_train, y_train, groups=patient_groups)
        ):
            X_tr = X_train.iloc[tr_idx].copy()
            y_tr = y_train[tr_idx]
            fold_model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
            fold_model.fit(X_tr, y_tr)
            test_pred_accum += fold_model.predict_proba(X_test)[:, 1]
        test_pred = (test_pred_accum / 5.0).astype(np.float32)
    else:
        model.fit(X_train, y_train)
        test_pred = model.predict_proba(X_test)[:, 1]

    submit_file = test_df[["image_name"]].copy()
    submit_file["target"] = test_pred



## === cell 7
submit_file = sample_sub[["image_name"]].merge(submit_file, on="image_name", how="left")

if submit_file["target"].isna().any():
    submit_file["target"] = submit_file["target"].fillna(submit_file["target"].mean())

submit_file["target"] = submit_file["target"].clip(0.0, 1.0)

submit_file.head()



## === cell 8
submit_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submit_file.shape)
print(submit_file.describe(include="all"))
