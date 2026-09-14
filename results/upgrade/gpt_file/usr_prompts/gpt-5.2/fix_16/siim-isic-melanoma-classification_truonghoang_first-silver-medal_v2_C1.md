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

0.9411769203044964

# 6. Current score

0.74543

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.77414) has done: 'I remove the hard dependency on the missing `../input/ensemble-melanoma` dataset (which is causing the FileNotFoundError) and instead build a valid, score-safe submission directly from the provided competition CSV metadata. To keep core intent similar (producing probabilistic predictions without training images/models in this environment), I implement a minimal metadata-only baseline: impute missing values, one-hot encode categoricals, and train a regularized logistic regression on `train.csv`, then predict probabilities for `test.csv`. I also strictly enforce the required submission columns and alignment to `sample_submission.csv` so the output is always valid and ordered correctly. The script always write a `.csv` submission file to the working directory.'
- What this solution (achieved 0.73489) has done: 'Your current baseline is a metadata-only logistic regression; to move AUC upward toward the target with minimal disruption, I keep the same pipeline and model family but improve signal using one additional high-value metadata feature: per-patient lesion count (patients with multiple images tend to have different risk profiles). I also make `age_approx` robustly numeric (some versions contain non-numeric strings) so it’s reliably used as a numeric feature instead of being one-hot encoded as text. Finally, I slightly increase the logistic regression capacity (higher `C`) while keeping `class_weight="balanced"` and the same training approach, which typically improves AUC for this setup without changing evaluation semantics.'
- What this solution (achieved 0.72478) has done: 'Your current score (0.73489) is far below the target (0.94118), so we should improve AUC while keeping the same metadata-only LogisticRegression pipeline. The smallest high-impact change here is to add a few well-known strong metadata-derived risk features (age bins, per-patient target prior with proper out-of-fold encoding to avoid leakage, and sex/site target encodings computed on train only). These features preserve the same core model family and training approach (single LogisticRegression fit), but give it substantially more signal than raw one-hot alone. I also keep the existing submission alignment logic untouched so the output remains valid and ordered like `sample_submission.csv`.'
- What this solution (achieved 0.72365) has done: 'Your current AUC (0.72478) is far below the target (0.94118), so we should improve the model’s discriminative signal while keeping the same metadata-only LogisticRegression pipeline and single-fit training approach. The biggest leak-safe boost available in metadata is replacing the weak patient target mean with a proper **out-of-fold** patient posterior computed via a smoothed leave-one-fold-out estimate (no using a row’s own label), and using that both for train (OOF) and test (full-train mapping). I also add one more standard, low-risk engineered feature: a missingness indicator for `anatom_site_general_challenge`, and ensure categorical blanks are treated consistently. These are minimal changes that preserve the core logic (same preprocessing + LogisticRegression) but typically move AUC upward materially.'
- What this solution (achieved 0.70449) has done: 'Your current AUC is far below the target, so we should improve discriminative signal while keeping the same single-fit metadata LogisticRegression pipeline. The biggest low-risk gain here is to fix the patient out-of-fold computation so it is truly *leave-one-out per row* (your current version subtracts the whole fold’s patient contribution, which can overly wash out signal and behave inconsistently). I keep the same engineered feature name and smoothing, but compute OOF patient means efficiently via per-patient totals and per-row leave-one-out subtraction. Everything else (preprocess, model, submission alignment and writing) stays the same to minimize disruption while plausibly increasing AUC toward the target.'
- What this solution (achieved 0.73021) has done: 'Your current pipeline is already a metadata-only LogisticRegression, so the safest way to move AUC upward toward the target is to strengthen the same features without changing the model family or training loop. I make the `sex` and `anatom_site_general_challenge` target-encoding **leak-safe** by computing them out-of-fold (your current version uses full-train means on train rows, which inflates training signal and can hurt generalization). I keep your patient leave-one-out feature (already leak-safe per row) and add one more small, standard, leak-safe numeric feature: `patient_age_mean` (mean age per patient), which often helps and doesn’t change the core approach. Submission writing/alignment remains identical to ensure a valid `submission.csv`.'
- What this solution (achieved 0.73227) has done: 'We keep your exact metadata-only LogisticRegression pipeline, but make one targeted change to reduce overfitting and improve generalization AUC: compute the CV folds for out-of-fold target encoding using a **stratified split** (preserving the malignant rate per fold) instead of random fold IDs. This keeps the same engineered features and smoothing math, but makes the OOF encodings more stable for an imbalanced dataset, which typically nudges AUC upward without changing core logic. Additionally, we ensure the OOF fold assignment is deterministic and reproducible with the same seed. Submission creation/alignment and the model remain unchanged so the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.73255) has done: 'We fix the runtime failure by ensuring the feature columns used for train and test are aligned: `diagnosis` and `benign_malignant` exist only in train, so we must drop train-only columns (and any other non-common columns) when building `feature_cols`. This also resolve the cascading `NameError`s since `model`, `X_test`, and `sub` were never created after the exception. Changes are minimal and score-neutral: the same feature engineering and LogisticRegression pipeline are preserved, we just prevent leaking/using unavailable columns and guarantee consistent columns for both splits. Finally, we keep the same submission alignment to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.73484) has done: 'Your current AUC (0.73255) is far below the target (0.94118), so we should improve generalization while keeping the exact same metadata-only LogisticRegression pipeline and single `fit()` approach. The smallest, high-impact change that doesn’t alter core modeling is to add two well-known strong, leak-safe engineered features: a log transform of `patient_image_count` (to reduce skew) and an age–sex interaction (`age_x_male`) to capture risk differences by sex across ages. These are simple numeric columns that integrate into your existing preprocessing/model unchanged, and typically lift AUC for this competition’s metadata baseline. Submission alignment and writing remain identical to ensure a valid `submission.csv`.'
- What this solution (achieved 0.73778) has done: 'Your current score (0.73484) is far below the target (0.94118), so we should improve AUC while keeping the same metadata-only LogisticRegression pipeline and single `fit()` approach. The most “minimal but high-signal” fix is to add one more leak-safe engineered feature: a smoothed out-of-fold target encoding for `patient_id` computed with StratifiedKFold (instead of per-row leave-one-out), which better matches the competition’s strong patient-level signal while still avoiding leakage. This preserves the same model family, preprocessing style, and training semantics (still one LogisticRegression fit on engineered metadata), but should move the AUC upward. Submission alignment/ordering and CSV writing are kept identical to ensure a valid `submission.csv`.'
- What this solution (achieved 0.7437) has done: 'We keep your exact metadata-only LogisticRegression pipeline and feature set, but make the target encoding features more reliable by (1) using **patient-grouped** stratified folds (so the same patient never appears in both train/validation for an OOF encoding) and (2) computing the patient encoding as a true **leave-one-out within fold** (so a row never uses its own label even indirectly). This reduces leakage/overfitting in the engineered TE features and typically improves public/private AUC without changing the core modeling approach. We also deterministically set seeds for reproducibility and keep the submission alignment/writing logic unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.7437) has done: 'We keep your exact metadata-only LogisticRegression pipeline and engineered feature set, but make one minimal, score-relevant change: calibrate the output probabilities with a single monotonic “sharpening” power transform tuned by cross-validated AUC on the training set (this preserves ranking behavior in a controlled way and often lifts AUC slightly). To avoid leakage, the tuning uses out-of-fold predictions from the same pipeline (no target data from the validation fold is used to fit the model for that fold). We then refit once on all training data and apply the chosen transform to test probabilities before writing the submission. All paths and submission alignment remain unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.7437) has done: 'Your current gap to the target AUC is large, so we should improve discriminative signal while keeping your exact metadata-only LogisticRegression pipeline and single-fit training semantics. The most impactful minimal tweak here is to make the **OOF target encoding for `patient_id` truly leave-one-out within the training fold**, so each row’s encoding never uses its own label even indirectly (this tends to improve generalization AUC for this competition). I keep your grouped-stratified fold assignment, the same preprocessing, the same LogisticRegression, and the same submission alignment/writing. I also keep the gamma “sharpening” step unchanged, since it’s already CV-tuned without leakage.'
- What this solution (achieved 0.74543) has done: 'Your current score (0.7437) is far below the target (0.94118), so we should cautiously improve AUC while keeping the same metadata-only LogisticRegression pipeline, preprocessing, and single final fit. The smallest high-impact tweak that preserves core logic is to add one more leak-safe engineered feature: a smoothed out-of-fold target encoding for the combined `sex`+`anatom_site_general_challenge` interaction, which often captures extra signal beyond each field alone. This reuses your existing grouped-stratified fold scheme and the same smoothed OOF encoding function, so it’s consistent with your current semantics and should move the score upward without changing architecture/training approach. Submission creation and alignment stay identical to ensure a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE = "/kaggle/data" if os.path.exists("/kaggle/data") else "/kaggle/input"

train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

train_df.shape, test_df.shape, sample_sub.shape



## === cell 2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

target_col = "target"
id_col = "image_name"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

for _df in (train_df, test_df):
    if "age_approx" in _df.columns:
        _df["age_approx"] = pd.to_numeric(_df["age_approx"], errors="coerce")


def _add_engineered_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
):
    tr = train_df.copy()
    te = test_df.copy()

    for df in (tr, te):
        if "sex" in df.columns:
            df["sex"] = (
                df["sex"].astype("object").fillna("Unknown").replace("", "Unknown")
            )
        if "anatom_site_general_challenge" in df.columns:
            df["anatom_site_general_challenge"] = (
                df["anatom_site_general_challenge"]
                .astype("object")
                .fillna("Unknown")
                .replace("", "Unknown")
            )
            df["site_isna"] = (df["anatom_site_general_challenge"] == "Unknown").astype(
                np.int8
            )

    tr["patient_image_count"] = (
        tr.groupby("patient_id")[id_col].transform("count").astype(np.float32)
    )
    te["patient_image_count"] = (
        te.groupby("patient_id")[id_col].transform("count").astype(np.float32)
    )

    if "age_approx" in tr.columns and "patient_id" in tr.columns:
        tr["patient_age_mean"] = (
            tr.groupby("patient_id")["age_approx"].transform("mean").astype(np.float32)
        )
        te["patient_age_mean"] = (
            te.groupby("patient_id")["age_approx"].transform("mean").astype(np.float32)
        )

    for df in (tr, te):
        df["age_bin"] = pd.cut(
            df["age_approx"], bins=[0, 30, 45, 60, 75, 120], include_lowest=True
        ).astype(str)
        df["age_isna"] = df["age_approx"].isna().astype(np.int8)

    y = tr[target_col].astype(float)
    global_mean = float(y.mean())

    def _make_patient_group_stratified_folds(n_splits: int = 5, seed: int = 42):
        pid_series = tr["patient_id"].astype(str).fillna("NA_PATIENT")
        pid_mean = tr.groupby(pid_series)[target_col].mean()
        pid_label = (pid_mean >= 0.5).astype(int)

        unique_pids = pid_mean.index.to_numpy()
        unique_labels = pid_label.values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        pid_to_fold = {}
        for fold_id, (_, va_pid_idx) in enumerate(
            skf.split(np.zeros(len(unique_pids)), unique_labels)
        ):
            for pid in unique_pids[va_pid_idx]:
                pid_to_fold[pid] = fold_id

        fold_ids = pid_series.map(pid_to_fold).astype(int).values
        return fold_ids

    fold_ids = _make_patient_group_stratified_folds(n_splits=5, seed=RANDOM_SEED)

    def add_smoothed_oof_te(col: str, alpha: float):
        tr_te = np.empty(len(tr), dtype=np.float64)

        for fold in range(int(fold_ids.max()) + 1):
            in_tr_idx = np.where(fold_ids != fold)[0]
            in_va_idx = np.where(fold_ids == fold)[0]

            stats = tr.iloc[in_tr_idx].groupby(col)[target_col].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * alpha) / (
                stats["count"] + alpha
            )

            tr_te[in_va_idx] = (
                tr.iloc[in_va_idx][col]
                .map(smooth)
                .fillna(global_mean)
                .astype(np.float64)
                .values
            )

        te_name = f"{col}_te"
        tr[te_name] = tr_te.astype(np.float32)

        stats_full = tr.groupby(col)[target_col].agg(["mean", "count"])
        smooth_full = (
            stats_full["mean"] * stats_full["count"] + global_mean * alpha
        ) / (stats_full["count"] + alpha)
        te[te_name] = te[col].map(smooth_full).fillna(global_mean).astype(np.float32)

    if "sex" in tr.columns:
        add_smoothed_oof_te("sex", alpha=25.0)
    if "anatom_site_general_challenge" in tr.columns:
        add_smoothed_oof_te("anatom_site_general_challenge", alpha=50.0)

    if ("sex" in tr.columns) and ("anatom_site_general_challenge" in tr.columns):
        for df in (tr, te):
            df["sex_site"] = (
                df["sex"].astype(str)
                + "||"
                + df["anatom_site_general_challenge"].astype(str)
            ).astype("object")
        add_smoothed_oof_te("sex_site", alpha=100.0)

    if "patient_id" in tr.columns:
        tr["patient_id"] = tr["patient_id"].astype(str).fillna("NA_PATIENT")
        te["patient_id"] = te["patient_id"].astype(str).fillna("NA_PATIENT")

        alpha_patient = 50.0
        pid_oof = np.empty(len(tr), dtype=np.float64)

        for fold in range(int(fold_ids.max()) + 1):
            in_tr_idx = np.where(fold_ids != fold)[0]
            in_va_idx = np.where(fold_ids == fold)[0]

            df_in = tr.iloc[in_tr_idx][["patient_id", target_col]]
            pid_sum = df_in.groupby("patient_id")[target_col].sum()
            pid_cnt = df_in.groupby("patient_id")[target_col].count()

            va_pids = tr.iloc[in_va_idx]["patient_id"].values

            base_sum = pd.Series(va_pids).map(pid_sum).astype(np.float64).values
            base_cnt = pd.Series(va_pids).map(pid_cnt).astype(np.float64).values

            s = np.nan_to_num(base_sum, nan=np.nan)
            c = np.nan_to_num(base_cnt, nan=np.nan)

            unseen = np.isnan(base_sum) | np.isnan(base_cnt)
            loo_sum = s
            loo_cnt = c
            mean_smooth = (loo_sum + global_mean * alpha_patient) / (
                loo_cnt + alpha_patient
            )
            mean_smooth[unseen] = global_mean

            pid_oof[in_va_idx] = mean_smooth

        tr["patient_target_mean_oof"] = pid_oof.astype(np.float32)

        stats_pid_full = tr.groupby("patient_id")[target_col].agg(["sum", "count"])
        smooth_pid_full = (stats_pid_full["sum"] + global_mean * alpha_patient) / (
            stats_pid_full["count"] + alpha_patient
        )
        te["patient_target_mean_oof"] = (
            te["patient_id"].map(smooth_pid_full).fillna(global_mean).astype(np.float32)
        )

    for df in (tr, te):
        df["patient_image_count_log1p"] = np.log1p(
            df["patient_image_count"].astype(np.float32)
        ).astype(np.float32)

        sex_lower = df["sex"].astype(str).str.lower()
        df["is_male"] = (sex_lower == "male").astype(np.int8)

        age_num = pd.to_numeric(df["age_approx"], errors="coerce").astype(np.float32)
        df["age_x_male"] = (age_num * df["is_male"].astype(np.float32)).astype(
            np.float32
        )

    return tr, te


train_df_fe, test_df_fe = _add_engineered_features(
    train_df, test_df, target_col=target_col
)

common_cols = sorted(set(train_df_fe.columns).intersection(set(test_df_fe.columns)))
feature_cols = [c for c in common_cols if c not in (id_col, target_col)]

X_train = train_df_fe[feature_cols].copy()
y_train = train_df_fe[target_col].astype(int).values
X_test = test_df_fe[feature_cols].copy()

numeric_cols = []
categorical_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(X_train[c]):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

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

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=3000,
    class_weight="balanced",
    C=3.0,
    n_jobs=None,
)

model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
model




## === cell 3
def _sharpen_probs(p: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-12, 1.0 - 1e-12)
    a = np.power(p, gamma)
    b = np.power(1.0 - p, gamma)
    out = a / (a + b)
    return np.clip(out, 0.0, 1.0)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof_pred = np.zeros(len(X_train), dtype=np.float64)

for tr_idx, va_idx in skf.split(X_train, y_train):
    fold_model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
    fold_model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    oof_pred[va_idx] = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1].astype(
        np.float64
    )

oof_pred = np.clip(np.nan_to_num(oof_pred, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

gammas = [0.85, 1.0, 1.15, 1.3]
best_gamma = 1.0
best_auc = -np.inf
for g in gammas:
    auc = roc_auc_score(y_train, _sharpen_probs(oof_pred, g))
    if auc > best_auc:
        best_auc = auc
        best_gamma = g

print(f"OOF AUC (for gamma tuning): best_gamma={best_gamma}, best_auc={best_auc:.6f}")



## === cell 4
model.fit(X_train, y_train)
test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

test_pred = np.clip(np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

test_pred = _sharpen_probs(test_pred, best_gamma)

pred_df = pd.DataFrame({id_col: test_df_fe[id_col].values, "target": test_pred})

sub = sample_sub[[id_col]].merge(pred_df, on=id_col, how="left")
sub["target"] = sub["target"].astype(float).fillna(0.5)

assert list(sub.columns) == [id_col, "target"]
assert len(sub) == len(sample_sub)

sub.head(), sub["target"].describe()



## === cell 5
out_path = "submission.csv"
sub.to_csv(out_path, index=False, float_format="%.6f")
print(f"Wrote {out_path} with shape {sub.shape}")
print(sub.head())
