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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.0059828002176315

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00032) has done: 'I fix the training failure by ensuring all remaining non-numeric features are encoded (the crash comes from a string column like `density` leaking into the feature matrix). I also make the resampling logic actually upsample the minority class to match the majority (current code accidentally keeps the same minority size), which should modestly improve pF1 while keeping the same overall approach. Finally, I guard inference/submission creation so `model`, `prediction_ids`, and `y_pred1` are always defined, and I write a valid `submission.csv` matching `sample_submission.csv`’s row set and order.'
- What this solution (achieved 0.00013) has done: 'Your current pipeline is likely under-scoring because the metric is pF1 (probabilistic F1), which is very sensitive to calibration and class imbalance: averaging raw probabilities across multiple images per `prediction_id` often yields overly small probabilities for positives, crushing recall. I keep the same model/feature logic, but (1) calibrate the final probabilities using the validation split (Platt scaling via `CalibratedClassifierCV`), and (2) change the per-`prediction_id` aggregation from mean to max to better preserve a “any-view positive” signal, which typically increases pF1 with minimal semantic change. I also add `class_weight="balanced"` to the logistic regression fallback to reduce the risk that LGBM is unavailable and the model collapses to near-zero probabilities. These are minimal, safe changes that should move your score upward toward the target without changing the overall approach.'
- What this solution (achieved 0.00013) has done: 'I make two minimal changes aimed at increasing pF1 toward your target: (1) fix the calibration step so it actually fits on `(x_val, y_val)` (your current code mistakenly calls `.fit(x_val, y_val)` where `x_val` is treated as labels), and (2) set `scale_pos_weight` for LightGBM using the post-upsampling class ratio to keep predicted probabilities from collapsing toward zero under imbalance. These preserve your exact feature engineering, model choices, and training flow, but correct a bug that likely hurts probabilities and improve probability scale in a metric (pF1) that is very sensitive to recall. Submission creation and ordering remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.00192) has done: 'Your score is far below the target (higher is better), so we should make small, low-risk fixes that specifically improve probability quality for pF1 without changing your overall feature pipeline or model family. The biggest issue is the calibration step: with `cv="prefit"` you must fit the base model on `x_train,y_train` and then calibrate on a *separate* set; right now you’re calibrating using `x_val,y_val` after the model already trained on `x_train`, which is fine, but your current `.fit(x_val, y_val)` call for the calibrator is correct only if `model` is already trained (it is) — the main improvement is to calibrate using **un-upsamped, original-distribution validation** so probabilities aren’t distorted by the artificial 50/50 upsampling. Second, for pF1 it usually helps to keep some probability mass (avoid near-zeros), so we apply a tiny, monotonic smoothing to predictions (a very small floor) which improves recall without changing ranking much. These are minimal changes and keep your training loop/models/features intact while nudging probabilities toward better pF1.'
- What this solution (achieved 0.00041) has done: 'Your current score (0.00192) is below the target (0.00598), so we should gently increase pF1 by nudging recall upward without changing the model family or feature pipeline. The smallest reliable lever for pF1 here is post-processing: instead of a fixed probability floor, we fit a single scalar “temperature” on the calibration split to slightly sharpen probabilities (monotonic transform) and then apply a tiny prior-based smoothing so outputs aren’t crushed to ~0. This keeps the same training, same calibration split, same aggregation (max over images per prediction_id), but typically yields better pF1 because positives retain more probability mass. I also make the scaler fit consistently on the original-distribution training fold used for calibration to reduce distribution shift between calibration and inference features.'
- What this solution (achieved 0.0) has done: 'Your current gap to the target is large (0.00041 vs 0.00598, higher-is-better), so we need a small but direct pF1 lift without changing the model family or feature pipeline. The most likely issue is that you train on an aggressively upsampled (≈50/50) distribution but then calibrate and smooth with an extremely tiny prior mix (alpha=0.01), which still leaves many predictions near zero and hurts pF1 recall. I keep the same training, encoding, calibration (sigmoid), temperature selection, and max-aggregation, but tune the *prior smoothing strength* `alpha` using the held-out calibration split by directly maximizing pF1—this is a minimal post-processing change aligned to the metric. I also reuse the already-scaled test predictions (no extra fitting) and keep submission ordering identical to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score is 0.0 (far below the target 0.00598, higher-is-better), which strongly suggests a submission validity issue rather than a modeling ceiling. I make two minimal, score-relevant fixes: (1) ensure every `prediction_id` that appears in `sample_submission.csv` receives a non-null prediction by aggregating test rows to the correct key before merging, and (2) make the `prediction_id` alignment deterministic by explicitly carrying `prediction_id` alongside predictions prior to grouping. These changes preserve your exact model/features/training/calibration/post-processing, but fix the most common reason pF1 becomes exactly zero: misaligned/mostly-missing predictions after the merge. The output still be `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a submission validity/alignment problem (most predictions missing or duplicated/misaligned by `prediction_id`), not with the model being too weak. I make the smallest changes to ensure we generate exactly one prediction per `prediction_id` with deterministic alignment: carry `prediction_id` alongside the row-level predictions, aggregate to `prediction_id` (max, as you already intended), then *reindex* to `sample_submission.csv` order and fill any missing IDs with the train prior. I also add a strict sanity check that the output row count and `prediction_id` set exactly match the sample submission so Kaggle doesn’t score it as effectively all-wrong/empty. These changes preserve your model, calibration, temperature scaling, smoothing, and aggregation semantics; they only harden the submission construction to avoid a zero score.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from a submission alignment/schema problem (missing/NaN cancers for many `prediction_id`s after grouping/mapping) rather than purely weak modeling, so I harden the prediction-to-`prediction_id` join to be exact and deterministic. I also ensure the features used for calibration (`x0_*_scaled`) apply the same numeric coercion + column order as `common_cols` before scaling, preventing silent column misalignment that can yield degenerate probabilities. Finally, I add strict sanity checks that the aggregated predictions cover the sample submission IDs and that the final file has exactly the right rows/columns, while keeping your model, calibration, temperature scaling, and max-aggregation logic unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score (vs target 0.00598, higher-is-better) strongly indicates Kaggle is effectively scoring you as invalid/mostly-misaligned, so the most direct improvement is to harden `prediction_id` alignment and ensure the row-level probabilities correspond 1:1 to `test.csv` rows before aggregation. I make minimal changes to (1) ensure `y_pred1` length exactly matches `test1` and is attached by position, (2) group/aggregate deterministically, and (3) guarantee the final submission has exactly the `sample_submission.csv` IDs in the same order (with a safe fallback prior if anything is missing). This preserves your exact model/features/training/calibration/temperature/smoothing logic; it only fixes the most likely source of the zero score. I also add a couple of strict assertions that fail early if any length/order mismatch would lead to a broken submission.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score (vs target 0.00598, higher-is-better) is most consistent with a *submission validity/ID alignment* issue rather than model quality, so I make the smallest changes that guarantee a correct 1:1 mapping from `test.csv` rows → `prediction_id` → `sample_submission.csv`. Specifically, I stop reading `test.csv` twice and instead use a single `test_df` everywhere so `prediction_ids` are guaranteed to align by row order with `y_pred1`. I also add strict checks that the aggregated `prediction_id` set exactly matches the sample submission’s IDs and fail early if not, instead of silently filling many missing IDs (which can yield an effectively-zero pF1). These changes preserve your model training, calibration, temperature scaling, smoothing, and max-aggregation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly caused by the submission being effectively “all wrong/invalid” due to the assertion that every `prediction_id` in `sample_submission.csv` must appear in `pred_agg`: in this competition, `test.csv` contains many more `prediction_id`s than the downloadable `sample_submission.csv`, so that assertion can break locally or (if you removed it) your merge can still silently misalign. I make the smallest changes to build the submission *exactly* on the `sample_submission.csv` ID set: filter predictions to those IDs, aggregate, then reindex to sample order and fill any missing with the train prior (so no NaNs). This preserves your model, calibration, temperature scaling, smoothing, and max-aggregation semantics; it only hardens ID alignment to avoid producing an “empty/mostly-missing” submission that scores as 0.0. I also keep strict shape/NaN checks, but remove the incorrect “must cover all sample IDs” assumption and replace it with deterministic reindexing.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score (vs target 0.00598, higher-is-better) is far enough that the most likely root cause is still submission misalignment/coverage rather than pure model quality. I make the smallest changes that guarantee row-level predictions are attached to the exact `test_df` row order, then aggregated to `prediction_id` and reindexed to `sample_submission.csv` deterministically (no silent drops). I also remove the filtering step that can discard most predictions when the downloadable `sample_submission.csv` is only a subset of the full hidden test IDs, and instead build the submission strictly by reindexing to the sample IDs (filling any missing with the train prior). These changes preserve your model training, calibration, temperature scaling, smoothing, and max-aggregation semantics while making a valid, fully-populated submission that should move the score up from 0 toward the target.'

# 9. Code solution

## === cell 0
import os
import glob
import random

import numpy as np
import pandas as pd



## === cell 1
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None



## === cell 2
try:
    import pydicom  # noqa: F401
except Exception:
    pydicom = None

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

try:
    import PIL
    from PIL import Image  # noqa: F401
except Exception:
    PIL = None



## === cell 3
SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 4
DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"



## === cell 5
train = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
img_data = DATA_DIR



## === cell 6
required_train_cols = {
    "cancer",
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "age",
    "implant",
    "machine_id",
    "site_id",
}
required_test_cols = {
    "patient_id",
    "image_id",
    "laterality",
    "view",
    "age",
    "implant",
    "machine_id",
    "site_id",
    "prediction_id",
}

missing_train = required_train_cols - set(train.columns)
missing_test = required_test_cols - set(test_df.columns)
assert not missing_train, f"Train missing columns: {missing_train}"
assert not missing_test, f"Test missing columns: {missing_test}"



## === cell 7
train_images = glob.glob(os.path.join(img_data, "train_images", "*", "*.dcm"))
test_images = glob.glob(os.path.join(img_data, "test_images", "*", "*.dcm"))



## === cell 8
test = test_df.copy()

num_cols_train = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols_test = test.select_dtypes(include=[np.number]).columns.tolist()

train[num_cols_train] = train[num_cols_train].fillna(
    train[num_cols_train].mean(numeric_only=True)
)
test[num_cols_test] = test[num_cols_test].fillna(
    test[num_cols_test].mean(numeric_only=True)
)

cat_cols = ["laterality", "view", "implant"]
for c in cat_cols:
    if c in train.columns:
        train[c] = train[c].fillna("missing")
    if c in test.columns:
        test[c] = test[c].fillna("missing")



## === cell 9
from sklearn.utils import resample

df_new_0 = train[train["cancer"] == 0]
df_new_1 = train[train["cancer"] == 1]

if len(df_new_1) == 0 or len(df_new_0) == 0:
    data_upsampled = train.copy()
else:
    maj = df_new_0 if len(df_new_0) >= len(df_new_1) else df_new_1
    mino = df_new_1 if len(df_new_0) >= len(df_new_1) else df_new_0
    mino_up = resample(mino, replace=True, n_samples=len(maj), random_state=20)

    data_upsampled = (
        pd.concat([maj, mino_up], axis=0)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )



## === cell 10
train_orig = train.copy()

y = data_upsampled["cancer"].astype(int)
x = data_upsampled.drop(columns=["cancer"])

test_features = test.drop(columns=["prediction_id"])

combined = pd.concat([x, test_features], axis=0, ignore_index=True)

cat_all = combined.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()
combined = pd.get_dummies(combined, columns=cat_all, dummy_na=True)

x_enc = combined.iloc[: len(x)].copy()
test_enc = combined.iloc[len(x) :].copy()



## === cell 11
from sklearn.preprocessing import StandardScaler

div_col_scale = ["age", "machine_id"]
for c in div_col_scale:
    if c not in x_enc.columns:
        x_enc[c] = 0.0
    if c not in test_enc.columns:
        test_enc[c] = 0.0

stand_data = StandardScaler()
x_enc[div_col_scale] = stand_data.fit_transform(x_enc[div_col_scale])
test_enc[div_col_scale] = stand_data.transform(test_enc[div_col_scale])



## === cell 12
x_enc = x_enc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
test_enc = test_enc.apply(pd.to_numeric, errors="coerce").fillna(0.0)

common_cols = sorted(list(set(x_enc.columns) & set(test_enc.columns)))
x_enc = x_enc[common_cols]
test_enc = test_enc[common_cols]



## === cell 13
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x_enc,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y if y.nunique() > 1 else None,
)



## === cell 14
from sklearn.linear_model import LogisticRegression

l_r = LogisticRegression(
    random_state=0, max_iter=500, solver="lbfgs", class_weight="balanced"
)
l_r.fit(x_train, y_train)
_ = l_r.predict(x_val)



## === cell 15
model = None
try:
    from lightgbm import LGBMClassifier  # type: ignore

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = (n_neg / max(1, n_pos)) if (n_pos > 0 and n_neg > 0) else 1.0

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.08,
        random_state=SEED,
        scale_pos_weight=spw,
    )
    model.fit(x_train, y_train)
except Exception:
    model = l_r



## === cell 16
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

y0 = train_orig["cancer"].astype(int)
x0 = train_orig.drop(columns=["cancer"])
combined0 = pd.concat(
    [x0, test.drop(columns=["prediction_id"])], axis=0, ignore_index=True
)
cat0 = combined0.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
combined0 = pd.get_dummies(combined0, columns=cat0, dummy_na=True)

x0_enc = combined0.iloc[: len(x0)].copy()

for c in common_cols:
    if c not in x0_enc.columns:
        x0_enc[c] = 0.0
extra_cols = [c for c in x0_enc.columns if c not in common_cols]
if extra_cols:
    x0_enc = x0_enc.drop(columns=extra_cols)

x0_enc = x0_enc[common_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

for c in div_col_scale:
    if c not in x0_enc.columns:
        x0_enc[c] = 0.0

x0_tr, x0_cal, y0_tr, y0_cal = train_test_split(
    x0_enc,
    y0,
    test_size=0.20,
    random_state=SEED,
    stratify=y0 if y0.nunique() > 1 else None,
)

stand_data0 = StandardScaler()
x0_tr_scaled = x0_tr.copy()
x0_cal_scaled = x0_cal.copy()
test_enc_scaled = test_enc.copy()

x0_tr_scaled[div_col_scale] = stand_data0.fit_transform(x0_tr_scaled[div_col_scale])
x0_cal_scaled[div_col_scale] = stand_data0.transform(x0_cal_scaled[div_col_scale])
test_enc_scaled[div_col_scale] = stand_data0.transform(test_enc_scaled[div_col_scale])

base_for_cal = model
try:
    if base_for_cal.__class__.__name__ == "LGBMClassifier":
        base_for_cal = base_for_cal.__class__(**base_for_cal.get_params())
        base_for_cal.fit(x0_tr_scaled, y0_tr)
    else:
        base_for_cal = LogisticRegression(
            random_state=0, max_iter=500, solver="lbfgs", class_weight="balanced"
        )
        base_for_cal.fit(x0_tr_scaled, y0_tr)
except Exception:
    base_for_cal = model

calibrated_model = base_for_cal
try:
    if hasattr(base_for_cal, "predict_proba"):
        calibrated_model = CalibratedClassifierCV(
            base_for_cal, method="sigmoid", cv="prefit"
        )
        calibrated_model.fit(x0_cal_scaled, y0_cal)
except Exception:
    calibrated_model = base_for_cal

if hasattr(calibrated_model, "predict_proba"):
    y_pred1 = calibrated_model.predict_proba(test_enc_scaled)[:, 1]
else:
    y_pred1 = calibrated_model.predict(test_enc_scaled).astype(float)


def _safe_logit(p, eps=1e-9):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _safe_sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def pf1_score(y_true, p):
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    pTP = np.sum(p * y_true)
    pFP = np.sum(p * (1.0 - y_true))
    pFN = np.sum((1.0 - p) * y_true)
    denom = 2.0 * pTP + pFP + pFN
    return (2.0 * pTP / denom) if denom > 0 else 0.0


try:
    if hasattr(calibrated_model, "predict_proba"):
        p_cal = calibrated_model.predict_proba(x0_cal_scaled)[:, 1]
    else:
        p_cal = calibrated_model.predict(x0_cal_scaled).astype(float)
    p_cal = np.clip(np.asarray(p_cal, dtype=float), 1e-9, 1 - 1e-9)
    z_cal = _safe_logit(p_cal)

    temps = np.array([0.7, 0.85, 1.0, 1.15, 1.3], dtype=float)
    best_t = 1.0
    best_pf1 = -1.0
    for t in temps:
        p_adj = _safe_sigmoid(z_cal / t)
        s = pf1_score(y0_cal.values, p_adj)
        if s > best_pf1:
            best_pf1 = s
            best_t = float(t)

    y_pred1 = _safe_sigmoid(_safe_logit(np.asarray(y_pred1, dtype=float)) / best_t)
except Exception:
    y_pred1 = np.asarray(y_pred1, dtype=float)

prediction_ids = test_df["prediction_id"].astype(str).to_numpy(copy=False)



## === cell 17
y_pred1 = np.asarray(y_pred1, dtype=float).reshape(-1)
y_pred1 = np.clip(y_pred1, 0.0, 1.0)

assert len(y_pred1) == len(
    test_df
), f"Pred length {len(y_pred1)} != test rows {len(test_df)}"
assert len(prediction_ids) == len(test_df), "prediction_ids length mismatch"

pos_rate = float(train_orig["cancer"].mean())

alpha = 0.01
try:
    if hasattr(calibrated_model, "predict_proba"):
        p_cal2 = calibrated_model.predict_proba(x0_cal_scaled)[:, 1]
    else:
        p_cal2 = calibrated_model.predict(x0_cal_scaled).astype(float)
    p_cal2 = np.clip(np.asarray(p_cal2, dtype=float), 0.0, 1.0)

    alpha_grid = np.array([0.00, 0.01, 0.03, 0.07, 0.12], dtype=float)
    best_a = float(alpha_grid[0])
    best_s = -1.0
    for a in alpha_grid:
        p_mix = (1.0 - a) * p_cal2 + a * pos_rate
        s = pf1_score(y0_cal.values, p_mix)
        if s > best_s:
            best_s = s
            best_a = float(a)
    alpha = best_a
except Exception:
    alpha = 0.01

y_pred1 = (1.0 - alpha) * y_pred1 + alpha * pos_rate
y_pred1 = np.clip(y_pred1, 0.0, 1.0)

test_pred_rows = pd.DataFrame({"prediction_id": prediction_ids, "cancer": y_pred1})
assert test_pred_rows.shape[0] == test_df.shape[0]
assert test_pred_rows["prediction_id"].isna().sum() == 0
assert test_pred_rows["cancer"].isna().sum() == 0

sample_sub = pd.read_csv(sample_sub_path)
sample_sub["prediction_id"] = sample_sub["prediction_id"].astype(str)

pred_agg = test_pred_rows.groupby("prediction_id", sort=False, as_index=True)[
    "cancer"
].max()

submission = sample_sub.set_index("prediction_id").copy()
submission["cancer"] = pred_agg.reindex(submission.index).astype(float)

submission["cancer"] = submission["cancer"].fillna(pos_rate).clip(1e-6, 1.0 - 1e-6)
submission = submission.reset_index()

assert list(submission.columns) == ["prediction_id", "cancer"]
assert submission.shape[0] == sample_sub.shape[0]
assert submission["prediction_id"].isna().sum() == 0
assert submission["cancer"].isna().sum() == 0
assert (submission["prediction_id"].values == sample_sub["prediction_id"].values).all()

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Chosen alpha:", alpha, "pos_rate:", pos_rate)
print("Wrote submission.csv with shape:", submission.shape)
print(
    "cancer min/max:",
    float(submission["cancer"].min()),
    float(submission["cancer"].max()),
)
print("Done")
