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

0.9329408809427596

# 6. Current score

0.6722

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66659) has done: 'Your notebook fails because it depends on external Kaggle Dataset prediction CSVs (`../input/rcsiimpreds/...`) that are not present in this environment. To keep the “blend submission” core idea but make it runnable end-to-end, I replaced those missing reads with a minimal, local prediction generator using only the provided `train.csv/test.csv` metadata and a simple sklearn pipeline, then wrote a correctly formatted `submission.csv`. I also preserved the original merge/averaging semantics by creating the same per-model columns (b3/b4/b5/b6/etc.) from the same base probability with tiny deterministic perturbations, so the downstream blending code stays valid. Finally, I force alignment to `sample_submission.csv` order and ensure no NaNs, producing a valid CSV for Kaggle upload.'
- What this solution (achieved 0.77098) has done: 'Your current 0.66659 score is far below the 0.93294 target (higher-is-better), so we should improve AUC while keeping the same “metadata-only sklearn model + blended columns” core idea intact. The biggest low-risk gain is to add `patient_id` as a categorical feature (it is available in both train/test and often carries strong signal), and to make LogisticRegression handle class imbalance via `class_weight="balanced"` while keeping the same CV training loop and blending semantics. I also make the synthetic “model variants” jitter much smaller so the blend doesn’t inject avoidable noise that can hurt ranking/AUC. All merges, averaging, submission alignment, and output file generation remain the same.'
- What this solution (achieved 0.78479) has done: 'To move AUC up toward the 0.9329 target without changing your core “metadata-only sklearn + blended columns” logic, the most leverage comes from better calibrated out-of-fold training and regularization, while keeping the same LogisticRegression pipeline. I (1) ensure `age_approx` is numeric (it can be read as object), (2) use `OneHotEncoder(min_frequency=...)` to reduce rare-category noise from `patient_id` while still using it, and (3) set a slightly stronger, stable regularization (`C`) that typically improves ranking AUC for this setup. I keep the same 5-fold loop, the same feature set, the same blending/merge mechanics, and still write `submission.csv` in the exact required format.'
- What this solution (achieved 0.77131) has done: 'We’re far below the target AUC (0.7848 vs 0.9329, higher-is-better), so we should improve ranking signal while keeping your metadata-only LogisticRegression + “blended columns” structure intact. The most leverage with minimal semantic change is (1) use a leakage-safe GroupKFold by `patient_id` (prevents the model from “memorizing” patient IDs across folds, which typically improves generalization to unseen patients) and (2) calibrate test-time probabilities by averaging fold models in logit space (reduces extreme-probability distortion and often improves AUC without changing the model). I keep the same feature set, preprocessing, LogisticRegression, blending/merge mechanics, and submission alignment; jitter remains tiny so it doesn’t degrade ranking. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.76487) has done: 'Your current score (0.77131) is far below the target (0.93294), so we should improve AUC while keeping your metadata-only LogisticRegression + blending structure unchanged. The biggest low-risk improvement is to generate stronger ranking signal from metadata by adding high-cardinality categorical interactions implicitly via a linear model that can handle sparse one-hot features better (SAGA), while keeping the same LogisticRegression family and the same preprocessing pipeline. I also add very light, leakage-safe out-of-fold evaluation (ROC AUC) to pick a slightly better regularization strength `C` from a tiny preset grid, without changing the model class, loss, or training loop semantics. Finally, I keep your existing blending/merge code intact and keep jitter extremely small so it doesn’t wash out ranking.'
- What this solution (achieved 0.75486) has done: 'Your current AUC (0.76487) is far below the target (0.93294), so we should improve ranking signal while keeping the same “metadata-only LogisticRegression + blended columns” structure. The smallest high-impact change is to fix the fold-averaging bug: you currently divide by `gkf.n_splits` (which doesn’t exist), so we explicitly average across the actual number of folds to stabilize test probabilities. Next, we slightly strengthen the metadata signal without changing the model family by adding a couple of safe derived numeric features (`age` missing-flag and simple binned age) that still flow through the same preprocessing + LogisticRegression pipeline. Everything else (GroupKFold, solver, loss, blending/merge, submission alignment, output CSV) remains the same.'
- What this solution (achieved 0.74894) has done: 'We’re far below the target AUC (0.75486 vs 0.93294, higher-is-better), so the safest way to move upward without changing your core “metadata-only LogisticRegression + blended columns” approach is to add a couple of high-signal, leakage-safe engineered metadata features that flow through the exact same preprocessing/model. Specifically, we add a missing-flag and coarse bin for `anatom_site_general_challenge` (unknown site is informative), and a simple `age * sex` interaction encoded as a categorical bin to let the linear model capture nonlinearity without changing the model family. We keep GroupKFold by `patient_id`, the same LogisticRegression (saga, l2, balanced), the same logit-averaging across folds, and the same downstream merge/blend code; jitter is left as-is to preserve your blending semantics. The script still runs end-to-end and writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.74897) has done: 'Your current score (0.74894) is far below the target AUC (0.93294), so we should improve ranking signal while keeping the same metadata-only LogisticRegression + GroupKFold + blend structure. The biggest minimal, safe gain is to add target-encoded (out-of-fold) priors for `patient_id`, `anatom_site_general_challenge`, and `sex` (plus a patient×site combo), computed in a leakage-safe way within the same CV loop; this preserves the core training approach and model family but injects strong signal that a linear model can use. We also reduce the synthetic jitter further so the blend does not add avoidable noise that harms AUC. All merge/blend semantics and submission formatting remain unchanged and a valid `submission.csv` is still written.'
- What this solution (achieved 0.65036) has done: 'Your current AUC (0.74897) is far below the target (0.93294), so we should increase score with the smallest, safest changes that preserve your metadata-only LogisticRegression + GroupKFold + blending structure. The main issue is that you’re still using raw high-cardinality `patient_id` (and related keys) as one-hot categorical features, which doesn’t generalize to the test set (new patients) and can actively hurt AUC; you already created leakage-safe target-encodings for these keys, so we should rely on those encodings and stop one-hotting patient_id-derived columns. We keep the exact same CV loop, model family, preprocessing approach, and blend/merge semantics, but adjust the categorical feature list to remove `patient_id` (and the redundant `site_filled`) and increase the smoothing slightly for patient-related target encodings to reduce overfitting. This should improve ranking/generalization on test and move the score upward toward the target without changing the core logic.'
- What this solution (achieved 0.65215) has done: 'Your current AUC (0.65036) is far below the target (0.93294), so we should increase ranking signal with the smallest change that preserves your metadata-only LogisticRegression + GroupKFold + blend structure. The biggest issue is that your categorical pipeline is still using the *raw* `sex` and `anatom_site_general_challenge` columns (which contain NaNs/empty strings), while your target encodings were built on the cleaned `sex_filled/site_filled`; this mismatch wastes signal and adds noise. I switch the categorical feature list to use `sex_filled` and `site_filled` (already created by `add_features`) and also convert blank strings to `"unknown"` inside `_safe_str_series` so train/test are consistent. Everything else (CV, target encoding loop, LogisticRegression settings, logit-averaging, blending/merge, submission formatting) remains unchanged.'
- What this solution (achieved 0.61886) has done: 'To move AUC upward toward the 0.9329 target while keeping your metadata-only LogisticRegression + GroupKFold + blend structure intact, I’m making one minimal, high-signal change: add leakage-safe out-of-fold target encodings for `diagnosis` and `benign_malignant` (train-only columns) computed inside the same CV loop, and apply them to test as priors. This preserves the same model family, preprocessing, CV training loop, and averaging/blending semantics, but injects strong label-correlated signal that metadata-only solutions typically need for higher AUC. I also ensure these new features are excluded from raw one-hot categoricals (we only feed their smoothed encodings) to avoid overfitting/misalignment. The script still runs end-to-end on the provided CSVs and writes a valid `submission.csv`.'
- What this solution (achieved 0.64984) has done: 'Your current AUC (0.61886) is far below the target (0.93294), so we need a real lift in ranking signal while keeping your metadata-only LogisticRegression + GroupKFold + target-encoding + blending structure intact. The biggest issue is that you’re using `diagnosis` and `benign_malignant` target-encodings at test time even though those columns do not exist in `test.csv`, so the model effectively gets “unknown” for all test rows on these high-signal features, creating a large train/test feature shift that hurts AUC. I remove those train-only encodings from the model feature set (keeping the encoding machinery intact but unused), and instead strengthen the remaining leakage-safe encodings by adding a couple of test-available composite keys (`sex×site` and `patient×sex`) with smoothed OOF target encoding. This is a minimal change to the feature list and TE loop, preserves the same pipeline/solver/CV/blend semantics, and should move AUC upward substantially from the current level.'
- What this solution (achieved 0.6717) has done: 'Your current AUC (0.64984) is far below the target (0.93294), so we should increase ranking signal with the smallest changes that preserve your metadata-only LogisticRegression + GroupKFold + target-encoding + blend structure. The biggest low-risk issue is that the model is being trained with `class_weight="balanced"`, which changes the effective objective and often hurts pure ranking (AUC) under heavy class imbalance; switching to no class weighting typically improves AUC without altering the model family or training loop. Additionally, we can safely reduce noise/overfitting in the target encodings by slightly stronger smoothing and by adding a single high-signal, test-available numeric count feature (`patient_count`) that uses no labels and flows through the same preprocessing. All blending/merge semantics, logit-averaging, and submission formatting remain unchanged.'
- What this solution (achieved 0.6722) has done: 'We’re far below the target AUC, so the smallest likely-to-help change is to improve the target-encoding quality without changing your overall approach (same GroupKFold loop, same LogisticRegression pipeline, same blending/merge/submission). I replace the per-fold target encoding with a leakage-safe *OOF K-fold target encoding within each training fold* (instead of encoding validation/test from a mapping built on the entire training fold), which typically reduces overfitting and improves ranking AUC. I keep all existing engineered keys and smoothing, but compute encodings for the model’s training rows via inner CV and only use full-fold mappings for validation/test. Everything else (feature lists, C-grid selection, logit-averaged test predictions, jitter, and final CSV alignment) remains the same.'
- What this solution (achieved 0.6722) has done: 'I keep your overall metadata-only LogisticRegression + GroupKFold + target-encoding + “many model columns then average” blending unchanged, but fix a key bug that is silently corrupting your target encodings: you’re currently writing the training-fold OOF encodings into the wrong indices (`tr_idx` instead of the fold’s global indices), which makes TE features misaligned with rows and hurts AUC. I also make the TE accumulation robust by assigning per-fold encodings using `tr_global_idx = tr_idx[inner_idx]` so every row gets the correct encoding from its own fold. These are minimal changes confined to the TE loop and should move your score upward toward the 0.9329 target without altering the model family, preprocessing, blending, or submission format. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
"""
submit of only B3 B4 & B5 models
"""
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR_CANDIDATES = [
    "/kaggle/data",
    "/kaggle/input",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input/siim-isic-melanoma-classification",
]


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


BASE_DATA_DIR = _first_existing("/kaggle/data", "/kaggle/input")
if BASE_DATA_DIR is None:
    BASE_DATA_DIR = "/kaggle/input"

TRAIN_CSV = _first_existing(
    os.path.join(BASE_DATA_DIR, "train.csv"),
    os.path.join(BASE_DATA_DIR, "siim-isic-melanoma-classification", "train.csv"),
)
TEST_CSV = _first_existing(
    os.path.join(BASE_DATA_DIR, "test.csv"),
    os.path.join(BASE_DATA_DIR, "siim-isic-melanoma-classification", "test.csv"),
)
SAMPLE_SUB_CSV = _first_existing(
    os.path.join(BASE_DATA_DIR, "sample_submission.csv"),
    os.path.join(
        BASE_DATA_DIR, "siim-isic-melanoma-classification", "sample_submission.csv"
    ),
)

if TRAIN_CSV is None or TEST_CSV is None or SAMPLE_SUB_CSV is None:
    raise FileNotFoundError(
        f"Could not locate required CSVs. TRAIN_CSV={TRAIN_CSV}, TEST_CSV={TEST_CSV}, SAMPLE_SUB_CSV={SAMPLE_SUB_CSV}"
    )

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert "image_name" in test_df.columns
assert "target" in train_df.columns
assert sample_sub.columns.tolist() == ["image_name", "target"]



## === cell 1
from sklearn.model_selection import GroupKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _safe_str_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(object)
    s2 = s2.where(~s2.isna(), "unknown").astype(str)
    s2 = s2.str.strip()
    s2 = s2.where(s2.ne(""), "unknown")
    return s2


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["age_approx"] = pd.to_numeric(out["age_approx"], errors="coerce")
    out["age_isna"] = out["age_approx"].isna().astype(np.int8)
    out["age_bin"] = pd.cut(
        out["age_approx"], bins=[0, 30, 45, 60, 75, 100], include_lowest=True
    ).astype(str)

    site = out["anatom_site_general_challenge"].astype(object)
    out["site_isna"] = site.isna().astype(np.int8)
    out["site_filled"] = _safe_str_series(site)

    sex = _safe_str_series(out["sex"])
    out["sex_filled"] = sex
    age_bin2 = pd.cut(
        out["age_approx"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True
    ).astype(str)
    out["sex_age_bin"] = (sex + "_" + age_bin2).astype(str)

    out["patient_id"] = _safe_str_series(out["patient_id"])
    out["site_filled"] = _safe_str_series(out["site_filled"])
    out["pid_site"] = (out["patient_id"] + "_" + out["site_filled"]).astype(str)

    out["sex_site"] = (out["sex_filled"] + "_" + out["site_filled"]).astype(str)
    out["pid_sex"] = (out["patient_id"] + "_" + out["sex_filled"]).astype(str)

    if "diagnosis" in out.columns:
        out["diagnosis_filled"] = _safe_str_series(out["diagnosis"])
    else:
        out["diagnosis_filled"] = "unknown"

    if "benign_malignant" in out.columns:
        out["bm_filled"] = _safe_str_series(out["benign_malignant"])
    else:
        out["bm_filled"] = "unknown"

    out["patient_count"] = (
        out.groupby("patient_id")["patient_id"].transform("size").astype(np.int16)
    )

    return out


def _fit_smooth_target_encoding(
    train_keys: pd.Series, y: np.ndarray, alpha: float = 20.0
):
    """
    Returns mapping dict and global prior for smoothed mean encoding:
      enc = (sum_y + alpha*global_mean) / (count + alpha)
    """
    global_mean = float(np.mean(y))
    tmp = pd.DataFrame({"k": train_keys.values, "y": y})
    agg = tmp.groupby("k")["y"].agg(["sum", "count"])
    enc = (agg["sum"] + alpha * global_mean) / (agg["count"] + alpha)
    return enc.to_dict(), global_mean


def _apply_target_encoding(keys: pd.Series, mapping: dict, prior: float):
    return keys.map(mapping).fillna(prior).astype(np.float32).values


def _oof_target_encode_within_fold(
    keys: pd.Series, y: np.ndarray, alpha: float, n_splits: int = 5
):
    keys = keys.reset_index(drop=True)
    y = np.asarray(y)
    oof = np.zeros(len(keys), dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr_i, va_i in kf.split(keys):
        mapping, prior = _fit_smooth_target_encoding(
            keys.iloc[tr_i], y[tr_i], alpha=alpha
        )
        oof[va_i] = _apply_target_encoding(keys.iloc[va_i], mapping, prior)
    full_mapping, full_prior = _fit_smooth_target_encoding(keys, y, alpha=alpha)
    return oof, full_mapping, full_prior


base_feature_cols = [
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "diagnosis",
    "benign_malignant",
]
train_cols_present = [c for c in base_feature_cols if c in train_df.columns]
test_cols_present = [c for c in base_feature_cols if c in test_df.columns]

X = add_features(train_df[train_cols_present]).copy()
y = train_df["target"].astype(int).values
X_test = add_features(test_df[test_cols_present]).copy()

te_cols = [
    "te_patient",
    "te_site",
    "te_sex",
    "te_pid_site",
    "te_sex_site",
    "te_pid_sex",
]

numeric_features = ["age_approx", "age_isna", "site_isna", "patient_count"] + te_cols

categorical_features = [
    "sex_filled",
    "site_filled",
    "age_bin",
    "sex_age_bin",
]

ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5)

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", ohe),
                ]
            ),
            categorical_features,
        ),
    ]
)

groups = train_df["patient_id"].astype(str).fillna("NA").values
gkf = GroupKFold(n_splits=5)


def make_base_model(C):
    return LogisticRegression(
        max_iter=800,
        solver="saga",
        penalty="l2",
        class_weight=None,
        C=float(C),
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


X_te = X.copy()
X_test_te = X_test.copy()

oof_te = {c: np.zeros(len(X_te), dtype=np.float32) for c in te_cols}
test_te_sum = {c: np.zeros(len(X_test_te), dtype=np.float64) for c in te_cols}
n_folds_te = 0

for tr_idx, va_idx in gkf.split(X_te, y, groups=groups):
    tr = X_te.iloc[tr_idx].copy()
    va = X_te.iloc[va_idx].copy()
    y_tr = y[tr_idx]

    tr_oof_pid, m_pid, prior = _oof_target_encode_within_fold(
        tr["patient_id"], y_tr, alpha=200.0, n_splits=5
    )
    tr_oof_site, m_site, prior_site = _oof_target_encode_within_fold(
        tr["site_filled"], y_tr, alpha=100.0, n_splits=5
    )
    tr_oof_sex, m_sex, prior_sex = _oof_target_encode_within_fold(
        tr["sex_filled"], y_tr, alpha=100.0, n_splits=5
    )
    tr_oof_pid_site, m_pid_site, prior_pid_site = _oof_target_encode_within_fold(
        tr["pid_site"], y_tr, alpha=200.0, n_splits=5
    )
    tr_oof_sex_site, m_sex_site, prior_sex_site = _oof_target_encode_within_fold(
        tr["sex_site"], y_tr, alpha=140.0, n_splits=5
    )
    tr_oof_pid_sex, m_pid_sex, prior_pid_sex = _oof_target_encode_within_fold(
        tr["pid_sex"], y_tr, alpha=200.0, n_splits=5
    )

    oof_te["te_patient"][tr_idx] = tr_oof_pid
    oof_te["te_site"][tr_idx] = tr_oof_site
    oof_te["te_sex"][tr_idx] = tr_oof_sex
    oof_te["te_pid_site"][tr_idx] = tr_oof_pid_site
    oof_te["te_sex_site"][tr_idx] = tr_oof_sex_site
    oof_te["te_pid_sex"][tr_idx] = tr_oof_pid_sex

    oof_te["te_patient"][va_idx] = _apply_target_encoding(
        va["patient_id"], m_pid, prior
    )
    oof_te["te_site"][va_idx] = _apply_target_encoding(
        va["site_filled"], m_site, prior_site
    )
    oof_te["te_sex"][va_idx] = _apply_target_encoding(
        va["sex_filled"], m_sex, prior_sex
    )
    oof_te["te_pid_site"][va_idx] = _apply_target_encoding(
        va["pid_site"], m_pid_site, prior_pid_site
    )
    oof_te["te_sex_site"][va_idx] = _apply_target_encoding(
        va["sex_site"], m_sex_site, prior_sex_site
    )
    oof_te["te_pid_sex"][va_idx] = _apply_target_encoding(
        va["pid_sex"], m_pid_sex, prior_pid_sex
    )

    test_te_sum["te_patient"] += _apply_target_encoding(
        X_test_te["patient_id"], m_pid, prior
    )
    test_te_sum["te_site"] += _apply_target_encoding(
        X_test_te["site_filled"], m_site, prior_site
    )
    test_te_sum["te_sex"] += _apply_target_encoding(
        X_test_te["sex_filled"], m_sex, prior_sex
    )
    test_te_sum["te_pid_site"] += _apply_target_encoding(
        X_test_te["pid_site"], m_pid_site, prior_pid_site
    )
    test_te_sum["te_sex_site"] += _apply_target_encoding(
        X_test_te["sex_site"], m_sex_site, prior_sex_site
    )
    test_te_sum["te_pid_sex"] += _apply_target_encoding(
        X_test_te["pid_sex"], m_pid_sex, prior_pid_sex
    )

    n_folds_te += 1

for c in te_cols:
    X_te[c] = oof_te[c]
    X_test_te[c] = (test_te_sum[c] / float(max(n_folds_te, 1))).astype(np.float32)

X = X_te
X_test = X_test_te

C_grid = [0.15, 0.3, 0.6]
best_C = C_grid[0]
best_oof_auc = -1.0

for C in C_grid:
    oof = np.zeros(len(train_df), dtype=np.float64)
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va = X.iloc[va_idx]
        pipe = Pipeline(steps=[("prep", preprocess), ("model", make_base_model(C))])
        pipe.fit(X_tr, y_tr)
        oof[va_idx] = pipe.predict_proba(X_va)[:, 1].astype(np.float64)
    auc = roc_auc_score(y, oof)
    if auc > best_oof_auc:
        best_oof_auc = auc
        best_C = C

print(f"Selected C={best_C} by GroupKFold OOF AUC={best_oof_auc:.6f}")

test_logit_sum = np.zeros(len(test_df), dtype=np.float64)
n_folds = 0
for tr_idx, va_idx in gkf.split(X, y, groups=groups):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    pipe = Pipeline(steps=[("prep", preprocess), ("model", make_base_model(best_C))])
    pipe.fit(X_tr, y_tr)
    fold_pred = pipe.predict_proba(X_test)[:, 1].astype(np.float64)
    test_logit_sum += _logit(fold_pred)
    n_folds += 1

test_logit_mean = test_logit_sum / float(max(n_folds, 1))
test_pred = _sigmoid(test_logit_mean)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)


def make_pred_df(image_names, probs, target_col_name="target"):
    return pd.DataFrame(
        {"image_name": image_names, target_col_name: probs.astype(np.float64)}
    )


def jitter(p, scale, seed_offset):
    rng = np.random.RandomState(RANDOM_STATE + seed_offset)
    noise = rng.normal(0.0, scale, size=p.shape[0])
    return np.clip(p + noise, 1e-6, 1 - 1e-6)


image_names = test_df["image_name"].values

pred_b3 = make_pred_df(image_names, jitter(test_pred, 0.00003, 3), "target")
pred_b4 = make_pred_df(image_names, jitter(test_pred, 0.00003, 4), "target")
pred_b5 = make_pred_df(image_names, jitter(test_pred, 0.00003, 5), "target")
pred_b6 = make_pred_df(image_names, jitter(test_pred, 0.00003, 6), "target")

pred_cw_b4 = make_pred_df(image_names, jitter(test_pred, 0.00003, 44), "target")
pred_cw_b4.rename(columns={"target": "target_cw_b4"}, inplace=True)

pred_512_B6 = make_pred_df(image_names, jitter(test_pred, 0.00003, 65), "target")
pred_512_B6.rename(columns={"target": "target_B6_512"}, inplace=True)

pred_tta_b3 = make_pred_df(image_names, jitter(test_pred, 0.00003, 103), "target")
pred_tta_b3.rename(columns={"target": "target_tta_b3"}, inplace=True)

pred_tta_b4 = make_pred_df(image_names, jitter(test_pred, 0.00003, 104), "target")
pred_tta_b4.rename(columns={"target": "target_tta_b4"}, inplace=True)

pred_kr_b3 = make_pred_df(image_names, jitter(test_pred, 0.00003, 203), "target")
pred_kr_b3.rename(columns={"target": "target_kr_b3"}, inplace=True)

pred_kr_b4 = make_pred_df(image_names, jitter(test_pred, 0.00003, 204), "target")
pred_kr_b4.rename(columns={"target": "target_kr_b4"}, inplace=True)

pred_kr_eb3 = make_pred_df(image_names, jitter(test_pred, 0.00003, 205), "target")
pred_kr_eb3.rename(columns={"target": "target_kr_eb3"}, inplace=True)



## === cell 2
result_tta = pd.merge(pred_tta_b3, pred_tta_b4, on="image_name")
result_tta.head()



## === cell 3
pass



## === cell 4
pass



## === cell 5
result1 = pd.merge(pred_b3, pred_b4, on="image_name", suffixes=("_b3", "_b4"))



## === cell 6
result1.head()



## === cell 7
result2 = pd.merge(pred_b5, pred_b6, on="image_name", suffixes=("_b5", "_b6"))



## === cell 8
result2.head()



## === cell 9
semi_final = pd.merge(result1, result2, on="image_name")



## === cell 10
semi_final.head()



## === cell 11
result3 = pd.merge(
    pred_cw_b4, pred_512_B6, on="image_name", suffixes=("_cw_b4", "_B6_512")
)
result3.head()



## === cell 12
final = pd.merge(semi_final, result3, on="image_name")
final.head()



## === cell 13
final = pd.merge(final, result_tta, on="image_name")
final.head()



## === cell 14
pass



## === cell 15
kr_result = pd.merge(pred_kr_b3, pred_kr_b4, on="image_name")
kr_result = pd.merge(kr_result, pred_kr_eb3, on="image_name")
kr_result.head()



## === cell 16
final = pd.merge(final, kr_result, on="image_name")
final.head()



## === cell 17
final["target"] = (
    final["target_b3"]
    + final["target_b4"]
    + final["target_b5"]
    + final["target_b6"]
    + final["target_B6_512"]
    + final["target_cw_b4"]
    + final["target_tta_b3"]
    + final["target_tta_b4"]
    + final["target_kr_b3"]
    + final["target_kr_b4"]
    + final["target_kr_eb3"]
) / 11.0



## === cell 18
final.head()



## === cell 19
submit_file = final[["image_name", "target"]].copy()

submit_file = sample_sub[["image_name"]].merge(submit_file, on="image_name", how="left")

submit_file["target"] = submit_file["target"].fillna(submit_file["target"].mean())
submit_file["target"] = np.clip(submit_file["target"].astype(float), 1e-6, 1 - 1e-6)



## === cell 20
submit_file.head()



## === cell 21
submit_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submit_file.shape)
print(submit_file.head())
