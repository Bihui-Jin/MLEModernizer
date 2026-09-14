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

0.9312218084118784

# 6. Current score

0.75719

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to read out-of-environment prediction files from `../input/rcsiimpreds`, which don’t exist in your provided filesystem. To make the pipeline run end-to-end and still produce a valid submission, I keep the ensemble logic structure but add a robust fallback: if those external prediction CSVs aren’t available, we generate a safe baseline prediction from `sample_submission.csv` (constant probability). I also ensure merges don’t crash, columns are created consistently, and the final submission is aligned to `test.csv` image order with the required `image_name,target` columns. This yield a valid `submission.csv` file (score won’t reach the original ~0.94 ensemble without those missing prediction inputs, but it run reliably).'
- What this solution (achieved 0.66776) has done: 'Your current 0.5 AUC comes from producing constant 0.5 predictions because the referenced ensemble prediction files aren’t present. To move the score toward the 0.931 target without changing the overall “merge multiple model preds then average” core logic, I keep the same ensemble structure but add a lightweight fallback that generates non-constant, data-driven probabilities from `train.csv` metadata and applies them to `test.csv`. This uses only sklearn (already available), avoids any image modeling changes, and still writes a valid `submission.csv` with the required columns and row alignment. If the external prediction CSVs ever do exist, the code continue to use them as before.'
- What this solution (achieved 0.6565) has done: 'To move your AUC up toward the 0.931 target without changing the core “load multiple external preds → merge → average” logic, I strengthen only the fallback branch (which is what’s actually producing your current 0.66776). Specifically, I keep the same sklearn LogisticRegression meta-model but add a small set of safe, metadata-only engineered features (age missing indicator, sex/site missing indicators, and simple interactions) that typically improve separability for this competition while staying within the same training approach and loss. I also ensure the fallback triggers not only when predictions are constant 0.5, but also when most ensemble inputs are missing (to avoid averaging mostly-baseline 0.5s). This should raise the fallback AUC materially, moving closer to the 0.931 band, while still producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.64519) has done: 'Your current score (0.6565) is far below the target (0.9312), so we should improve the fallback branch that’s actually driving performance when external prediction files are missing. Keeping the same “(try to) load external preds → merge → average → if missing then metadata LR fallback” core logic, I (1) make the fallback stronger by training the same LogisticRegression on richer but still metadata-only features (add patient_id leakage-safe aggregation features computed from train only, plus a couple of simple interactions), and (2) make the fallback trigger slightly more reliably when the ensemble is effectively baseline. These are minimal, sklearn-only changes that tend to materially increase AUC on this competition without changing the overall approach or producing an invalid submission. The script still writes a valid `submission.csv` with `image_name,target` aligned to `test.csv`.'
- What this solution (achieved 0.75719) has done: 'Your current gap to the target AUC (0.645 → 0.931) is large, and the only place we can legitimately gain signal without changing your “load/merge external preds then average, else fallback” core logic is the metadata fallback. I keep the ensemble path exactly as-is, but strengthen the fallback by (1) switching the patient-level target encoding to a strict out-of-fold (OOF) computation to avoid overfitting artifacts that can hurt generalization AUC, and (2) using calibrated smoothing for categorical mean encodings (sex/site) so rare categories don’t get extreme means. These are minimal, sklearn-only changes that keep the same LogisticRegression approach/loss and should move AUC upward toward your target when external prediction files are missing. The script still run end-to-end and write a valid `submission.csv` with `image_name,target` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
"""
submit of only B3 B4 & B5 models
"""
import os
import numpy as np
import pandas as pd

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = _first_existing(DATA_ROOT_CANDIDATES) or "."
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")

if not os.path.exists(TEST_CSV):
    alt = os.path.join(DATA_ROOT, "siim-isic-melanoma-classification", "test.csv")
    if os.path.exists(alt):
        TEST_CSV = alt
if not os.path.exists(SAMPLE_SUB):
    alt = os.path.join(
        DATA_ROOT, "siim-isic-melanoma-classification", "sample_submission.csv"
    )
    if os.path.exists(alt):
        SAMPLE_SUB = alt
if not os.path.exists(TRAIN_CSV):
    alt = os.path.join(DATA_ROOT, "siim-isic-melanoma-classification", "train.csv")
    if os.path.exists(alt):
        TRAIN_CSV = alt

test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

assert "image_name" in test_df.columns
assert "image_name" in sample_sub.columns and "target" in sample_sub.columns


def load_pred_or_baseline(path, target_col_name=None, baseline_value=0.5):
    """
    Try to load pred CSVs from an external dataset path.
    If missing, return baseline predictions with correct schema for merging.
    """
    if target_col_name is None:
        target_col_name = "target"
    if path is not None and os.path.exists(path):
        df = pd.read_csv(path)
        if "image_name" not in df.columns:
            raise ValueError(f"Prediction file {path} missing 'image_name' column.")
        if "target" not in df.columns and target_col_name not in df.columns:
            raise ValueError(f"Prediction file {path} missing 'target' column.")
        if "target" in df.columns and target_col_name != "target":
            df = df.rename(columns={"target": target_col_name})
        return df[["image_name", target_col_name]].copy()

    base = test_df[["image_name"]].copy()
    base[target_col_name] = float(baseline_value)
    return base


RC_PRED_DIR_CANDIDATES = [
    "../input/rcsiimpreds",
    "/kaggle/input/rcsiimpreds",
    "/kaggle/data/rcsiimpreds",
]
RC_PRED_DIR = _first_existing(RC_PRED_DIR_CANDIDATES)


def rc_path(filename):
    if RC_PRED_DIR is None:
        return None
    return os.path.join(RC_PRED_DIR, filename)




## === cell 1
pred_b3 = load_pred_or_baseline(
    rc_path("sub_EfficientNetB3_384_9460.csv"),
    target_col_name="target_b3",
    baseline_value=0.5,
)
pred_b4 = load_pred_or_baseline(
    rc_path("sub_EfficientNetB4_384_9498.csv"),
    target_col_name="target_b4",
    baseline_value=0.5,
)
pred_b5 = load_pred_or_baseline(
    rc_path("sub_EfficientNetB5_384_9454.csv"),
    target_col_name="target_b5",
    baseline_value=0.5,
)
pred_b6 = load_pred_or_baseline(
    rc_path("sub_EfficientNetB6_384_9481.csv"),
    target_col_name="target_b6",
    baseline_value=0.5,
)

pred_cw_b4 = load_pred_or_baseline(
    rc_path("sub_EfficientNetB4_CW_384_9457.csv"),
    target_col_name="target_cw_b4",
    baseline_value=0.5,
)



## === cell 2
pred_512_B6 = load_pred_or_baseline(
    rc_path("sub_B5_512_3fold_9466.csv"),
    target_col_name="target_B6_512",
    baseline_value=0.5,
)



## === cell 3
pred_tta_b3 = load_pred_or_baseline(
    rc_path("siim_tta_b3_9458.csv"), target_col_name="target_tta_b3", baseline_value=0.5
)
pred_tta_b4 = load_pred_or_baseline(
    rc_path("siim_tta_b4_9473.csv"), target_col_name="target_tta_b4", baseline_value=0.5
)

result_tta = pd.merge(pred_tta_b3, pred_tta_b4, on="image_name", how="inner")



## === cell 4
result1 = pd.merge(pred_b3, pred_b4, on="image_name", how="inner")



## === cell 5
result2 = pd.merge(pred_b5, pred_b6, on="image_name", how="inner")



## === cell 6
semi_final = pd.merge(result1, result2, on="image_name", how="inner")



## === cell 7
result3 = pd.merge(pred_cw_b4, pred_512_B6, on="image_name", how="inner")



## === cell 8
final = pd.merge(semi_final, result3, on="image_name", how="inner")
final = pd.merge(final, result_tta, on="image_name", how="inner")



## === cell 9
final["target"] = (
    final["target_b3"]
    + final["target_b4"]
    + final["target_b5"]
    + final["target_b6"]
    + final["target_B6_512"]
    + final["target_cw_b4"]
    + final["target_tta_b3"]
    + final["target_tta_b4"]
) / 8.0

final["target"] = (
    pd.to_numeric(final["target"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
)




## === cell 10
def _all_constant_half(df, col="target"):
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.fillna(0.5)
    return float(s.nunique()) == 1.0 and float(s.iloc[0]) == 0.5


def _ensemble_effectively_missing(
    df, pred_cols, baseline_value=0.5, min_non_baseline_frac=0.05
):
    """
    Change rationale (score-toward-target): if most ensemble columns are just the baseline 0.5
    (because files are missing), averaging them yields weak predictions. Triggering the metadata
    fallback in this case improves AUC while keeping the ensemble core logic intact.
    """
    non_base_fracs = []
    for c in pred_cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(baseline_value)
        non_base_fracs.append(float((s != baseline_value).mean()))
    return float(np.mean(non_base_fracs)) < float(min_non_baseline_frac)


need_fallback = _all_constant_half(final, "target") or _ensemble_effectively_missing(
    final,
    pred_cols=[
        "target_b3",
        "target_b4",
        "target_b5",
        "target_b6",
        "target_B6_512",
        "target_cw_b4",
        "target_tta_b3",
        "target_tta_b4",
    ],
    baseline_value=0.5,
    min_non_baseline_frac=0.10,
)

if need_fallback:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    train_df = pd.read_csv(TRAIN_CSV)

    base_feature_cols = [
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
    ]

    def _prep_base_cols(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        for c in ["sex", "anatom_site_general_challenge"]:
            if c in df.columns:
                df[c] = df[c].replace("", np.nan)
        if "age_approx" in df.columns:
            df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
        return df

    train_base = _prep_base_cols(train_df[base_feature_cols + ["target"]].copy())
    test_base = _prep_base_cols(test_df[base_feature_cols].copy())

    def add_patient_oof_mean(
        train_local: pd.DataFrame, n_splits: int = 5, seed: int = 42
    ):
        train_local = train_local.copy()
        y = train_local["target"].astype(int).values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        global_mean = float(np.mean(y))
        oof_mean = np.full(len(train_local), np.nan, dtype=float)
        oof_cnt = np.full(len(train_local), np.nan, dtype=float)

        for tr_idx, val_idx in skf.split(train_local, y):
            tr_part = train_local.iloc[tr_idx]
            agg = (
                tr_part.groupby("patient_id", dropna=False)["target"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "m", "count": "c"})
            )
            val_part = train_local.iloc[val_idx][["patient_id"]].join(
                agg, on="patient_id"
            )
            oof_mean[val_idx] = val_part["m"].astype(float).values
            oof_cnt[val_idx] = val_part["c"].astype(float).values

        train_local["patient_target_mean"] = (
            pd.Series(oof_mean).fillna(global_mean).astype(float)
        )
        train_local["patient_image_count"] = (
            pd.Series(oof_cnt).fillna(0.0).astype(float)
        )
        return train_local

    def add_patient_full_agg(train_local: pd.DataFrame, test_local: pd.DataFrame):
        agg = (
            train_local.groupby("patient_id", dropna=False)["target"]
            .agg(["mean", "count"])
            .rename(
                columns={"mean": "patient_target_mean", "count": "patient_image_count"}
            )
            .reset_index()
        )
        test_local = test_local.merge(agg, on="patient_id", how="left")
        return test_local

    train_base = add_patient_oof_mean(train_base, n_splits=5, seed=42)
    test_base = add_patient_full_agg(
        train_base[base_feature_cols + ["target"]], test_base
    )

    def add_cat_mean_smooth(
        train_df_local, test_df_local, col, out_col, smoothing=20.0
    ):
        train_df_local = train_df_local.copy()
        test_df_local = test_df_local.copy()
        global_mean = float(train_df_local["target"].mean())

        stats = (
            train_df_local.groupby(col, dropna=False)["target"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stats[out_col] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        train_df_local = train_df_local.merge(stats[[col, out_col]], on=col, how="left")
        test_df_local = test_df_local.merge(stats[[col, out_col]], on=col, how="left")
        return train_df_local, test_df_local

    train_base, test_base = add_cat_mean_smooth(
        train_base, test_base, "sex", "sex_target_mean", smoothing=20.0
    )
    train_base, test_base = add_cat_mean_smooth(
        train_base,
        test_base,
        "anatom_site_general_challenge",
        "site_target_mean",
        smoothing=20.0,
    )

    def make_meta_features(df_in: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = df_in.copy()

        df["age_missing"] = df["age_approx"].isna().astype(int)
        df["sex_missing"] = df["sex"].isna().astype(int)
        df["site_missing"] = df["anatom_site_general_challenge"].isna().astype(int)

        age = df["age_approx"]
        df["age_bin"] = pd.cut(
            age,
            bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
            labels=[
                "<=20",
                "21-30",
                "31-40",
                "41-50",
                "51-60",
                "61-70",
                "71-80",
                "80+",
            ],
        ).astype(object)

        df["sex_site"] = (
            df["sex"].fillna("UNK").astype(str)
            + "__"
            + df["anatom_site_general_challenge"].fillna("UNK").astype(str)
        )

        df["age_x_sex_male"] = df["age_approx"].fillna(0.0) * (
            df["sex"].fillna("UNK").astype(str).str.lower() == "male"
        ).astype(float)
        df["age_x_site_known"] = df["age_approx"].fillna(0.0) * (
            (~df["anatom_site_general_challenge"].isna()).astype(float)
        )

        if "patient_target_mean" not in df.columns:
            df["patient_target_mean"] = np.nan
        if "patient_image_count" not in df.columns:
            df["patient_image_count"] = np.nan

        return df

    X_train = make_meta_features(train_base.drop(columns=["target"]), is_train=True)
    y_train = train_base["target"].astype(int).values
    X_test = make_meta_features(test_base, is_train=False)

    numeric_features = [
        "age_approx",
        "age_missing",
        "sex_missing",
        "site_missing",
        "patient_target_mean",
        "patient_image_count",
        "sex_target_mean",
        "site_target_mean",
        "age_x_sex_male",
        "age_x_site_known",
    ]
    categorical_features = [
        "sex",
        "anatom_site_general_challenge",
        "age_bin",
        "sex_site",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=700,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )

    meta_model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    meta_model.fit(X_train, y_train)
    meta_pred = meta_model.predict_proba(X_test)[:, 1].astype(float)
    meta_pred = np.clip(meta_pred, 0.0, 1.0)

    final = test_df[["image_name"]].copy()
    final["target"] = meta_pred

submit_file = test_df[["image_name"]].merge(
    final[["image_name", "target"]], on="image_name", how="left"
)
submit_file["target"] = submit_file["target"].fillna(0.5).astype(float).clip(0.0, 1.0)

submit_file = submit_file[["image_name", "target"]]
assert submit_file.shape[0] == test_df.shape[0]
assert list(submit_file.columns) == ["image_name", "target"]



## === cell 11
submit_file.to_csv("submission.csv", index=False)
print(submit_file.head())
print("Wrote submission.csv with", len(submit_file), "rows")
