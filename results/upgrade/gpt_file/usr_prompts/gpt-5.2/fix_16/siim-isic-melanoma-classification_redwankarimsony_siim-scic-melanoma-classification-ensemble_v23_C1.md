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

0.9421

# 6. Current score

0.71529

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crashes come from trying to load several “public submission” CSVs that are not present in your environment; this prevents `sub` from ever being created and written. I keep the same high-level intent (produce predictions for the provided `sample_submission.csv`) but make it robust by conditionally loading any available ensemble files and otherwise falling back to a simple, score-neutral baseline (constant probability) so a valid `submission.csv` is always generated. I also ensure predictions are aligned by `image_name` and clipped to \[0,1\] to avoid format/validity issues. This run end-to-end in your Kaggle environment and create a proper `.csv` submission.'
- What this solution (achieved 0.66776) has done: 'Your current 0.5 AUC comes from emitting a constant 0.5 prediction when the external “public submission” files aren’t present; that yields near-random ranking. To move toward the 0.9421 target with minimal change, I keep the same “produce `sample_submission.csv`-aligned predictions and write `submission.csv`” core flow, but replace the fallback with a lightweight, fully local metadata-based model trained from `train.csv` and applied to `test.csv`. This uses only the provided CSV metadata (sex, age_approx, anatomic site) with simple preprocessing and logistic regression to create a non-constant probability ranking, which should substantially raise AUC versus 0.5 without changing evaluation semantics. The code still uses the ensemble files if they happen to exist, and otherwise produces a valid submission reliably.'
- What this solution (achieved 0.70903) has done: 'Your current AUC (0.66776) is far below the target (0.9421), so we should improve ranking quality with minimal changes while keeping the same “metadata-only logistic regression fallback” core flow. The smallest high-impact fix is to add more informative, still-local metadata features already present in `train.csv` (especially `patient_id` and `diagnosis`) and to use a slightly less-regularized logistic regression (`C` higher) so the model can fit these strong predictors better. To avoid leakage-like overfitting to rare categories while staying simple, we rare-group infrequent `patient_id`/`diagnosis` values into an `"__RARE__"` bucket based only on training frequency, then one-hot encode as before. Submission creation, alignment to `sample_submission.csv`, and clipping remain unchanged.'
- What this solution (achieved 0.74179) has done: 'Your current AUC (0.709) is still far below the 0.9421 target, so we should improve ranking signal with minimal, “same core” metadata-only modeling changes. The biggest low-risk gain is to add a leakage-safe aggregation feature: per-`patient_id` malignancy rate computed on train only and mapped to both train/test (with global-mean fallback), which typically improves AUC substantially without changing the overall pipeline (still a single LogisticRegression on tabular metadata). I also make the one-hot encoding a bit more expressive by increasing `min_frequency` while keeping rare-grouping logic (this reduces extreme sparsity while preserving signal). Submission alignment, clipping, and the “use ensemble if present, else fallback model” logic remain unchanged.'
- What this solution (achieved 0.71282) has done: 'Your current score (0.74179) is far below the target (0.9421), so we should improve ranking signal with minimal, “same pipeline” metadata-only changes (still one LogisticRegression with the same preprocessing style). The biggest safe gain here is to avoid target leakage in `patient_target_rate`: compute it out-of-fold on the training set (so each row’s feature doesn’t directly encode its own label), and use a smoothed mean to reduce noise for small patient groups. We then map a separate, train-fitted (smoothed) patient rate to the test set as before, keeping the rest of the model and submission logic unchanged. This typically improves generalization (and thus Kaggle test AUC) while preserving the core logic and producing the same submission format.'
- What this solution (achieved 0.71267) has done: 'Your current gap to the target is large (0.71282 vs 0.9421), so we should improve ranking signal without changing the overall “metadata-only logistic regression fallback” pipeline. The smallest high-impact fix is to add one more leakage-safe aggregation feature: a smoothed malignancy rate per `anatom_site_general_challenge` computed out-of-fold for train and train-fitted for test, analogous to your existing out-of-fold `patient_target_rate`. This keeps the same model class (single LogisticRegression), same preprocessing style (impute + one-hot), and same submission alignment, but typically boosts AUC because site is a strong melanoma prior. I’m also keeping determinism and runtime constraints by using pure pandas groupbys (no CV loops beyond the OOF math already present).'
- What this solution (achieved 0.71456) has done: 'Your current score (0.71267) is far below the target (0.9421), so we should improve rank quality with the smallest possible change while keeping the same “metadata-only logistic regression fallback” pipeline. The biggest missing signal in this competition is that `patient_id` appears in both train and test, so using a train-fitted, smoothed per-patient malignancy prior for test (without the out-of-fold exclusion needed only for train rows) usually boosts AUC a lot. I keep your existing OOF `patient_target_rate` for training (to avoid leakage) but add a separate `patient_target_rate_full` feature: for train it is also OOF (same leakage-safe style), and for test it uses the full per-patient smoothed rate fit on all training rows. Everything else (preprocessing, LogisticRegression, submission alignment) remains the same.'
- What this solution (achieved 0.70816) has done: 'We’re still far below the target AUC (0.71456 vs 0.9421), so the smallest meaningful change is to improve the fallback metadata model’s ranking signal without changing the overall approach (single LogisticRegression with the same preprocessing). The most direct low-risk gain is to add a leakage-safe, smoothed out-of-fold malignancy prior for `diagnosis` (analogous to what you already do for `patient_id` and `anatom_site`), plus a train-fitted full `diagnosis` prior for test-time inference. This preserves evaluation semantics, keeps the same model family/training loop, and only adds two numeric features derived strictly from `train.csv`. Submission creation/alignment and the “use ensemble if present else train local model” behavior stay unchanged.'
- What this solution (achieved 0.71078) has done: 'Your current score (0.70816) is far below the target (0.9421), so we should improve the ranking signal while keeping the same metadata-only LogisticRegression fallback pipeline. The most likely bug hurting AUC is that your “_full” target-rate features are accidentally identical to the out-of-fold ones for train and identical to the OOF-mapped ones for test, so they add no new information. I minimally fix this by computing true full-sample smoothed priors for `patient_id` and `diagnosis` (and also add a consistent full-sample prior for `anatom_site_general_challenge`) for test-time inference, while keeping the OOF versions for training to avoid leakage. Everything else (preprocessing, model, submission alignment, clipping, and optional ensemble loading) remains unchanged.'
- What this solution (achieved 0.71078) has done: 'The score is far below the target AUC, so we should increase ranking signal while keeping the same core “metadata-only LogisticRegression fallback” approach. The biggest issue in your current fallback is that the `_full` target-rate features for `patient_id` and `diagnosis` are not actually “full for test” (they’re just copies of the OOF-mapped versions), which removes a key generalization boost when train/test share IDs. I minimally fix this by (1) computing true train-fitted smoothed priors for test for `patient_id`, `diagnosis`, and `anatom_site_general_challenge`, and (2) also setting the test `diagnosis` to `"__MISSING__"` before mapping so it’s consistent. Everything else (preprocessing, one-hot, LogisticRegression hyperparams, submission alignment and clipping) stays the same.'
- What this solution (achieved 0.708) has done: 'To move your AUC upward toward 0.9421 without changing the core “metadata-only LogisticRegression fallback” approach, I make one high-signal but still minimal correction: compute **true out-of-fold (OOF) target-encoding features** for `patient_id`, `diagnosis`, and `anatom_site_general_challenge` using a deterministic KFold, instead of the current leave-one-out style that is noisier and can hurt generalization. I also keep your existing “full-sample smoothed prior” mappings for test-time (`*_full`) so shared IDs between train/test still help, but ensure the OOF features are genuinely fold-excluded for train rows. Finally, I keep the same preprocessing + LogisticRegression pipeline and the same submission alignment/clipping, so evaluation semantics and runtime remain stable.'
- What this solution (achieved 0.70976) has done: 'Your current fallback model is underperforming mainly because the “*_full” target-encoding features are accidentally computed as OOF again for train and then copied for test, so they add almost no new signal (especially important since `patient_id` overlaps heavily between train/test). I minimally fix this by computing true full-sample smoothed priors (fit on all train) for `patient_id`, `anatom_site_general_challenge`, and `diagnosis`, using them for test-time mapping, while keeping the OOF versions for training to avoid leakage. I also ensure `diagnosis` is handled consistently (missing in test) and align rare-grouping with the tokens used in encoding to avoid accidental mismatches. This keeps the same overall pipeline (metadata-only + target-encoding + single LogisticRegression) and should move AUC upward toward your 0.9421 target.'
- What this solution (achieved 0.70762) has done: 'Your current AUC (0.70976) is far below the target (0.9421), so we should add more real signal while keeping the same core “metadata-only LogisticRegression fallback” pipeline. The minimal high-impact fix here is to compute leakage-safe **out-of-fold target encodings** using `StratifiedKFold` (instead of plain `KFold`) and to make the encodings more stable by choosing `alpha` based on category type (patient stronger smoothing than site/diagnosis). I also correct an issue where the “*_full” features in train were computed but never used meaningfully (they were derived from the same mapping as OOF in earlier iterations); now train gets true OOF versions while test gets true full-train mappings, which should improve ranking on the public/private test. Submission alignment and output format remain unchanged.'
- What this solution (achieved 0.71596) has done: 'Your current score (0.70762) is far below the target (0.9421), so we should add a small amount of real signal without changing the overall “metadata-only logistic regression with target-encoding” pipeline. The main low-risk issue is that you’re using `diagnosis` as a categorical and for target-encoding, but `diagnosis` is not available in `test.csv`, so it mostly collapses to a constant `"__MISSING__"` at inference and can hurt calibration/weighting. I keep the same model class and preprocessing, but (1) drop `diagnosis` from categorical/OHE features while keeping the already-computed numeric diagnosis prior feature, and (2) add one more strong, still-leakage-safe OOF target-encoding numeric feature for `sex` (and full mapping for test) to improve ranking. Everything else (submission alignment, clipping, optional ensemble loading) stays the same.'
- What this solution (achieved 0.71529) has done: 'Your current AUC (0.71596) is far below the target (0.9421), so we should add a small amount of additional ranking signal while keeping the same “metadata + target-encoding + single LogisticRegression” pipeline. The most impactful minimal fix is to include `anatom_site_general_challenge` as a numeric target-encoded feature based on the **full train** mapping at test time (you already compute `site_target_rate_full` but currently set it equal to `site_target_rate`, losing the intended extra signal). I also add a leakage-safe out-of-fold target encoding for `age_approx` (binned) and include it as numeric features; this preserves the same model and preprocessing while typically improving AUC by refining risk ranking. The submission writing, alignment to `sample_submission.csv`, clipping, and optional ensemble-loading behavior remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if os.path.exists(p)), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find Kaggle data directory under expected paths."
    )

sample_path_candidates = [
    os.path.join(DATA_DIR, "sample_submission.csv"),
    "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv",
    "/kaggle/data/siim-isic-melanoma-classification/sample_submission.csv",
]
SAMPLE_PATH = next((p for p in sample_path_candidates if os.path.exists(p)), None)
if SAMPLE_PATH is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

train_path_candidates = [
    os.path.join(DATA_DIR, "train.csv"),
    "/kaggle/input/siim-isic-melanoma-classification/train.csv",
    "/kaggle/data/siim-isic-melanoma-classification/train.csv",
]
TRAIN_PATH = next((p for p in train_path_candidates if os.path.exists(p)), None)
if TRAIN_PATH is None:
    raise FileNotFoundError("train.csv not found in expected locations.")

test_path_candidates = [
    os.path.join(DATA_DIR, "test.csv"),
    "/kaggle/input/siim-isic-melanoma-classification/test.csv",
    "/kaggle/data/siim-isic-melanoma-classification/test.csv",
]
TEST_PATH = next((p for p in test_path_candidates if os.path.exists(p)), None)
if TEST_PATH is None:
    raise FileNotFoundError("test.csv not found in expected locations.")

sub = pd.read_csv(SAMPLE_PATH)

ensemble_candidates = [
    "/kaggle/input/public-submission-melanoma-95/submission_mean.csv",
    "/kaggle/input/public-submission-melanoma-95/submission_median.csv",
    "/kaggle/input/public-submission-melanoma-95/external_meta_ensembled.csv",
    "/kaggle/input/public-submission-melanoma-95/submission_9581.csv",
    "/kaggle/input/public-submission-melanoma-95/submission_tabular_only.csv",
    "/kaggle/input/public-submission-melanoma-95/submission_9619.csv",
    "/kaggle/input/public-submission-melanoma-95/submission_9606.csv",
    "/kaggle/input/public-submission-melanoma-95/submission_9603.csv",
]


def _load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_name" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"Submission at {path} must have columns ['image_name','target']"
        )
    return df[["image_name", "target"]].copy()


available_paths = [p for p in ensemble_candidates if os.path.exists(p)]

if len(available_paths) > 0:
    merged = sub[["image_name"]].copy()
    preds = []
    for p in available_paths:
        dfp = _load_submission(p).rename(columns={"target": os.path.basename(p)})
        merged = merged.merge(dfp, on="image_name", how="left")
        preds.append(os.path.basename(p))
    for c in preds:
        merged[c] = merged[c].astype(float)
        merged[c] = merged[c].fillna(merged[c].mean())
    sub["target"] = merged[preds].mean(axis=1).astype(float)
else:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    global_prior = float(train_df["target"].mean())
    train_df = train_df.copy()
    test_df = test_df.copy()

    alpha_patient = 50.0
    alpha_site = 20.0
    alpha_diag = 30.0
    alpha_sex = 10.0
    alpha_agebin = 20.0

    if "diagnosis" not in test_df.columns:
        test_df["diagnosis"] = np.nan

    def _oof_target_encode_stratified(
        df: pd.DataFrame,
        col: str,
        y_col: str,
        n_splits: int,
        seed: int,
        alpha: float,
        global_prior: float,
        missing_token: str = "__MISSING__",
    ) -> np.ndarray:
        x = df[col].astype("object").where(df[col].notna(), other=missing_token)
        yv = df[y_col].astype(float).values
        oof = np.empty(len(df), dtype=float)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(df, (yv > 0.5).astype(int)):
            x_tr = x.iloc[tr_idx]
            y_tr = yv[tr_idx]

            tr_tmp = pd.DataFrame({"x": x_tr.values, "y": y_tr})
            sums = tr_tmp.groupby("x")["y"].sum()
            cnts = tr_tmp.groupby("x")["y"].count()

            x_va = x.iloc[va_idx]
            sum_va = x_va.map(sums).astype(float).fillna(0.0)
            cnt_va = x_va.map(cnts).astype(float).fillna(0.0)
            oof[va_idx] = (sum_va + alpha * global_prior) / (cnt_va + alpha)

        return oof

    def _full_target_encode_map(
        train_df: pd.DataFrame,
        train_key_series: pd.Series,
        test_key_series: pd.Series,
        y_col: str,
        alpha: float,
        global_prior: float,
    ) -> pd.Series:
        tmp = train_df.assign(_k=train_key_series.values)
        sums = tmp.groupby("_k")[y_col].sum()
        cnts = tmp.groupby("_k")[y_col].count()

        test_sum = test_key_series.map(sums).astype(float).fillna(0.0)
        test_cnt = test_key_series.map(cnts).astype(float).fillna(0.0)
        return ((test_sum + alpha * global_prior) / (test_cnt + alpha)).astype(float)

    pid_col = "patient_id"
    train_pid = (
        train_df[pid_col]
        .astype("object")
        .where(train_df[pid_col].notna(), "__MISSING__")
    )
    test_pid = (
        test_df[pid_col].astype("object").where(test_df[pid_col].notna(), "__MISSING__")
    )

    train_df["patient_target_rate"] = _oof_target_encode_stratified(
        train_df,
        pid_col,
        "target",
        n_splits=5,
        seed=0,
        alpha=alpha_patient,
        global_prior=global_prior,
    )
    train_df["patient_target_rate_full"] = _full_target_encode_map(
        train_df,
        train_pid,
        train_pid,
        "target",
        alpha=alpha_patient,
        global_prior=global_prior,
    ).values
    test_df["patient_target_rate"] = _full_target_encode_map(
        train_df,
        train_pid,
        test_pid,
        "target",
        alpha=alpha_patient,
        global_prior=global_prior,
    )
    test_df["patient_target_rate_full"] = test_df["patient_target_rate"].astype(float)

    site_col = "anatom_site_general_challenge"
    train_site = (
        train_df[site_col]
        .astype("object")
        .where(train_df[site_col].notna(), "__MISSING__")
    )
    test_site = (
        test_df[site_col]
        .astype("object")
        .where(test_df[site_col].notna(), "__MISSING__")
    )

    train_df["site_target_rate"] = _oof_target_encode_stratified(
        train_df,
        site_col,
        "target",
        n_splits=5,
        seed=1,
        alpha=alpha_site,
        global_prior=global_prior,
    )
    train_df["site_target_rate_full"] = _full_target_encode_map(
        train_df,
        train_site,
        train_site,
        "target",
        alpha=alpha_site,
        global_prior=global_prior,
    ).values
    test_df["site_target_rate"] = _full_target_encode_map(
        train_df,
        train_site,
        test_site,
        "target",
        alpha=alpha_site,
        global_prior=global_prior,
    )
    test_df["site_target_rate_full"] = _full_target_encode_map(
        train_df,
        train_site,
        test_site,
        "target",
        alpha=alpha_site,
        global_prior=global_prior,
    ).astype(float)

    diag_col = "diagnosis"
    train_diag = (
        train_df[diag_col]
        .astype("object")
        .where(train_df[diag_col].notna(), "__MISSING__")
    )
    test_diag = (
        test_df[diag_col]
        .astype("object")
        .where(test_df[diag_col].notna(), "__MISSING__")
    )

    train_df["diagnosis_target_rate"] = _oof_target_encode_stratified(
        train_df,
        diag_col,
        "target",
        n_splits=5,
        seed=2,
        alpha=alpha_diag,
        global_prior=global_prior,
    )
    train_df["diagnosis_target_rate_full"] = _full_target_encode_map(
        train_df,
        train_diag,
        train_diag,
        "target",
        alpha=alpha_diag,
        global_prior=global_prior,
    ).values
    test_df["diagnosis_target_rate"] = _full_target_encode_map(
        train_df,
        train_diag,
        test_diag,
        "target",
        alpha=alpha_diag,
        global_prior=global_prior,
    )
    test_df["diagnosis_target_rate_full"] = test_df["diagnosis_target_rate"].astype(
        float
    )

    sex_col = "sex"
    train_sex = (
        train_df[sex_col]
        .astype("object")
        .where(train_df[sex_col].notna(), "__MISSING__")
    )
    test_sex = (
        test_df[sex_col].astype("object").where(test_df[sex_col].notna(), "__MISSING__")
    )

    train_df["sex_target_rate"] = _oof_target_encode_stratified(
        train_df,
        sex_col,
        "target",
        n_splits=5,
        seed=3,
        alpha=alpha_sex,
        global_prior=global_prior,
    )
    train_df["sex_target_rate_full"] = _full_target_encode_map(
        train_df,
        train_sex,
        train_sex,
        "target",
        alpha=alpha_sex,
        global_prior=global_prior,
    ).values
    test_df["sex_target_rate"] = _full_target_encode_map(
        train_df,
        train_sex,
        test_sex,
        "target",
        alpha=alpha_sex,
        global_prior=global_prior,
    )
    test_df["sex_target_rate_full"] = test_df["sex_target_rate"].astype(float)

    def _make_age_bin(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        b = (np.floor(x / 5.0) * 5.0).astype("Float64")
        return b.astype("object").where(b.notna(), "__MISSING__")

    train_age_bin = _make_age_bin(train_df["age_approx"])
    test_age_bin = _make_age_bin(test_df["age_approx"])
    train_df["age_bin_target_rate"] = _oof_target_encode_stratified(
        train_df.assign(_age_bin=train_age_bin),
        "_age_bin",
        "target",
        n_splits=5,
        seed=4,
        alpha=alpha_agebin,
        global_prior=global_prior,
    )
    train_df["age_bin_target_rate_full"] = _full_target_encode_map(
        train_df,
        train_age_bin,
        train_age_bin,
        "target",
        alpha=alpha_agebin,
        global_prior=global_prior,
    ).values
    test_df["age_bin_target_rate"] = _full_target_encode_map(
        train_df,
        train_age_bin,
        test_age_bin,
        "target",
        alpha=alpha_agebin,
        global_prior=global_prior,
    )
    test_df["age_bin_target_rate_full"] = test_df["age_bin_target_rate"].astype(float)

    feat_cols = [
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "patient_id",
        "patient_target_rate",
        "patient_target_rate_full",
        "site_target_rate",
        "site_target_rate_full",
        "diagnosis_target_rate",
        "diagnosis_target_rate_full",
        "sex_target_rate",
        "sex_target_rate_full",
        "age_bin_target_rate",
        "age_bin_target_rate_full",
    ]

    missing = [c for c in feat_cols + ["target"] if c not in train_df.columns]
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")
    missing_t = [c for c in feat_cols if c not in test_df.columns]
    if missing_t:
        raise ValueError(f"test.csv missing required columns: {missing_t}")

    X_train = train_df[feat_cols].copy()
    y_train = train_df["target"].astype(int).values
    X_test = test_df[feat_cols].copy()

    def _rare_group_train_test(
        train_s: pd.Series,
        test_s: pd.Series,
        min_count: int,
        missing_token: str = "__MISSING__",
        rare_token: str = "__RARE__",
    ):
        tr = train_s.astype("object").where(train_s.notna(), other=missing_token)
        te = test_s.astype("object").where(test_s.notna(), other=missing_token)
        vc = tr.value_counts(dropna=False)
        keep = set(vc[vc >= min_count].index.tolist())
        tr2 = tr.where(tr.isin(keep), other=rare_token)
        te2 = te.where(te.isin(keep), other=rare_token)
        return tr2, te2

    X_train["patient_id"], X_test["patient_id"] = _rare_group_train_test(
        X_train["patient_id"],
        X_test["patient_id"],
        min_count=5,
        missing_token="__MISSING__",
        rare_token="__RARE__",
    )

    numeric_features = [
        "age_approx",
        "patient_target_rate",
        "patient_target_rate_full",
        "site_target_rate",
        "site_target_rate_full",
        "diagnosis_target_rate",
        "diagnosis_target_rate_full",
        "sex_target_rate",
        "sex_target_rate_full",
        "age_bin_target_rate",
        "age_bin_target_rate_full",
    ]
    categorical_features = [
        "sex",
        "anatom_site_general_challenge",
        "patient_id",
    ]

    pre = ColumnTransformer(
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
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        C=3.0,
        random_state=0,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    test_pred = pipe.predict_proba(X_test)[:, 1].astype(float)

    pred_df = pd.DataFrame(
        {"image_name": test_df["image_name"].values, "target": test_pred}
    )
    sub = sub[["image_name"]].merge(pred_df, on="image_name", how="left")
    sub["target"] = sub["target"].fillna(global_prior)

sub["target"] = sub["target"].astype(float).clip(0.0, 1.0)



## === cell 2
assert list(sub.columns) == [
    "image_name",
    "target",
], f"Unexpected submission columns: {sub.columns.tolist()}"
sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 3
sub.head()
