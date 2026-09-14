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

0.9150191311439004

# 6. Current score

0.63591

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66738) has done: 'Your notebook fails because it depends on an external Kaggle dataset (`public-submission-melanoma-95`) that is not present in this environment, so none of the loaded “public_sub_*” DataFrames exist and the ensemble step crashes. To make it run end-to-end and still be competitive, I replaced that unavailable-input ensemble with a minimal, self-contained metadata-only model trained from `train.csv` and applied to `test.csv` (same evaluation semantics: predict `target` probabilities). The fix also ensures robust preprocessing (missing values, unseen categories) and writes a valid `submission.csv` with the exact required columns and row alignment to `sample_submission.csv`. This should yield a non-trivial AUC (likely below top image models but far better than a constant baseline) and, crucially, produces a valid submission file.'
- What this solution (achieved 0.65555) has done: 'I keep your metadata-only LogisticRegression pipeline and 5-fold CV intact, but add two minimal improvements that typically boost AUC for this competition without changing the overall approach. First, I switch the CV split to be grouped by `patient_id` (still stratified) to reduce leakage and make the learned probabilities generalize better to unseen patients, which usually improves leaderboard AUC. Second, I add `patient_id`-level target encoding learned only on each fold (and applied to test using out-of-fold statistics), which is a small, legitimate feature addition that often gives a sizable lift over using only sex/age/site. The submission writing and schema stay the same and still produce `submission.csv` end-to-end.'
- What this solution (achieved 0.73847) has done: 'Your current score (0.65555) is far below the target (0.9150), so we should improve performance while keeping the same metadata-only LogisticRegression + 5-fold StratifiedGroupKFold core logic. The biggest low-risk gain here is to fix leakage in the “patient_target_mean” feature: right now it uses the full fold’s patient mean for the same rows (perfectly encoding y for each training patient), which hurts generalization; we replace it with out-of-fold (leave-one-out style) patient means computed within each training fold. We also add simple Bayesian smoothing of the patient mean toward the fold prior (very small change, same semantics), and keep everything else (features, model, CV, submission writing) intact. These changes typically increase AUC substantially for this competition without changing the modeling approach.'
- What this solution (achieved 0.73883) has done: 'Your current solution is a metadata-only LogisticRegression with a patient-level target-encoding feature; to move the AUC meaningfully toward the target without changing the core approach, the smallest reliable gain is to (1) make the fold-based test predictions consistent by using a single globally-smoothed patient prior for test-time encoding (instead of a different prior per fold), and (2) tune the smoothing strength `alpha` downward a bit so the patient signal isn’t over-shrunk. These changes keep the exact same model, preprocessing, CV strategy, and feature set, but reduce fold-to-fold calibration mismatch and typically improve ranking (AUC). I also add out-of-fold AUC reporting (doesn’t affect submission) to verify the change is moving in the right direction. The script still runs end-to-end and writes a valid `submission.csv` with correct columns and alignment.'
- What this solution (achieved 0.47373) has done: 'Your current AUC (0.73883) is far below the target (0.9150), so we should make a small, legitimate change that can improve ranking without changing the core “metadata + LogisticRegression + StratifiedGroupKFold + patient target mean encoding” approach. The biggest low-risk gain here is to add one more standard, competition-relevant metadata signal (`diagnosis`) into the same one-hot encoded categorical pipeline; this keeps the exact same model family, training loop, and evaluation semantics while often providing a sizable uplift. I also make sure `diagnosis` is safely available at inference by filling test-time with a constant “unknown” category (no leakage). Everything else (CV, smoothing/LOO encoding, submission writing) stays the same.'
- What this solution (achieved 0.73883) has done: 'Your current drop (0.73883 → 0.47373) is almost certainly because adding `diagnosis` as a categorical feature while filling test with `"unknown"` creates a severe train/test shift (the model learns strong signals from diagnosis categories that never appear in test), hurting ranking AUC. To move the score back up toward the 0.915 target with minimal changes and identical core logic (same LogisticRegression + StratifiedGroupKFold + patient target-mean encoding), I remove `diagnosis` from the feature set and keep everything else unchanged. I also keep the patient encoding exactly as-is (including smoothing and global prior for test) to preserve evaluation semantics and stability. This is the smallest change that should recover the previous ~0.74 AUC behavior and move you closer to the target.'
- What this solution (achieved 0.73744) has done: 'Your current AUC (0.73883) is still far below the target (0.9150), so we should make a small, low-risk improvement that keeps the same metadata-only LogisticRegression + StratifiedGroupKFold + patient target-mean encoding core logic. The most direct lift, without changing the model family or training loop, is to add a couple of well-known strong metadata interactions for this specific competition: (1) log-transform `age_approx` (helps linear models) and (2) add a smoothed target-encoding for `anatom_site_general_challenge` computed strictly fold-wise (OOF for train, fold stats for val, global prior for test). This preserves evaluation semantics (still producing probabilities) and avoids leakage by computing encodings only from the training fold. Submission writing, column names, and paths remain unchanged and a valid `submission.csv` is produced.'
- What this solution (achieved 0.7381) has done: 'To move your AUC upward toward 0.915 with minimal disruption, I keep the exact same metadata-only LogisticRegression + StratifiedGroupKFold + smoothed LOO target encodings, but fix two small issues that can materially hurt ranking. First, your patient/site encodings currently treat missing mappings as `0` before smoothing (because `.map(...).astype(float)` turns NaNs into NaNs but later arithmetic can propagate oddly); I make the mapping arithmetic explicitly NaN-safe and only compute means where counts>0, then fill with the appropriate prior—this prevents systematic bias for unseen patients/sites in a fold. Second, I add a very small, competition-standard numeric feature interaction (`age_approx` missing indicator) that often improves linear models without changing the approach. These are minimal feature/encoding corrections intended to increase AUC, while keeping the training loop, model family, CV, and submission format unchanged.'
- What this solution (achieved 0.7381) has done: 'Your current gap to the target AUC is large (0.7381 vs 0.9150), so the smallest likely win while preserving the same LogisticRegression + StratifiedGroupKFold + smoothed target-encoding core is to fix a subtle bug: you compute the site target-encoding “leave-one-out” using per-row labels (`y_tr`) instead of the per-row site sum/count, which injects wrong subtraction and damages the feature signal. I correct the LOO subtraction to use each row’s site aggregated target sum/count, keeping the exact same smoothing, priors, CV, model, and feature set. I also make the mapping NaN-safe for the LOO computation by filling missing sums/counts with 0 before arithmetic (doesn’t change semantics, just prevents NaN propagation). This should move AUC upward toward your target without changing the overall approach or producing any format changes in the submission.'
- What this solution (achieved 0.73709) has done: 'We’re still far below the target AUC (0.7381 vs 0.9150), so we should improve ranking with the smallest change that keeps your metadata-only LogisticRegression + StratifiedGroupKFold + smoothed LOO target-encoding core intact. The biggest remaining low-risk issue is that the fold-wise site encoding is computed on the raw site string, which has missing/blank values and inconsistent “unknown” handling; that weakens the signal and creates avoidable train/val/test mismatch. I standardize `anatom_site_general_challenge` (fill missing with a single token and strip whitespace) and also add a single additional smoothed target-encoding for `sex` computed fold-wise (same exact encoding pattern you already use for patient/site), which is a minimal feature addition that often gives a modest AUC lift without changing the model family or training loop. Submission writing/format stays identical and still produces `submission.csv`.'
- What this solution (achieved 0.73307) has done: 'We’re far below the target AUC, so the smallest likely improvement without changing the core “metadata + LogisticRegression + StratifiedGroupKFold + smoothed LOO target encodings” approach is to (1) add a missingness indicator for `anatom_site_general_challenge` (helps linear models capture systematic missingness) and (2) tune the smoothing strengths slightly so the patient encoding is trusted a bit more while site encoding is shrunk a bit more (reduces noise from many site categories). These are minimal feature/regularization adjustments that preserve the exact training loop, model family, and evaluation semantics, and should move the score upward without introducing leakage. I also keep all paths and the submission-writing logic identical to ensure a valid `submission.csv` is produced end-to-end.'
- What this solution (achieved 0.74943) has done: 'Your current AUC (0.733) is far below the target (0.915), so we should make a small, safe lift without changing the core “metadata + LogisticRegression + StratifiedGroupKFold + smoothed LOO target encodings” approach. The most likely easy win is improving the numeric scaling for LogisticRegression by adding a `StandardScaler` to the numeric pipeline (keeps the same model family and training loop, but typically improves ranking when you mix raw age/log/target-encodings). I also slightly relax regularization (increase `C`) to reduce underfitting, which often helps AUC for this setup while staying within identical semantics. Everything else (features, fold logic, encodings, paths, and submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.74538) has done: 'We keep your exact metadata-only LogisticRegression + StratifiedGroupKFold + smoothed LOO target-encoding approach, but make two small changes that typically lift AUC for this competition without altering the core training loop. First, we let the solver use a better optimization path for dense OHE features by switching from `liblinear` to `lbfgs` (still LogisticRegression, same probability semantics), which often improves ranking. Second, we slightly reduce underfitting by increasing `max_iter` to ensure convergence and nudging `C` up a bit; this is a minimal regularization adjustment intended to move your score upward toward the 0.915 target band. Everything else (features, encodings, folds, file paths, and submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.63489) has done: 'Your current AUC (0.745) is far below the 0.915 target, so we should improve ranking with the smallest changes that keep your exact “metadata + LogisticRegression + StratifiedGroupKFold + smoothed LOO target encodings” core intact. The biggest low-risk gain here is to fix the subtle mismatch where test-time target-encodings are computed per-fold but shrunk toward a global prior (creating fold-dependent calibration noise); instead, we compute test encodings once from full training data (still legitimate, no test leakage) and reuse them for every fold. Additionally, we slightly trust the strongest metadata signal (`patient_id`) more by reducing its smoothing `alpha` a bit, while keeping site/sex smoothing unchanged to avoid adding noise. Everything else (features, model family, CV, preprocessing, and submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.63591) has done: 'We’re far below the target AUC (0.6349 vs 0.9150), so we should make the smallest legitimate changes that improve ranking while keeping your exact metadata + LogisticRegression + StratifiedGroupKFold + smoothed LOO target-encoding core intact. The most likely issue suppressing generalization is that the target-encoding priors are computed per-fold (`prior_fold`), which adds fold-dependent noise; we switch the *smoothing prior* for patient/site/sex encodings to the single `global_prior` consistently for train/val/test (still no leakage, and still fold-wise stats). Separately, `age_approx` is currently both a raw numeric feature and also implicitly present in `base_features`; to reduce collinearity and stabilize the linear model, we explicitly use the numeric `age_approx` column (not the raw string) and ensure it is numeric in both train/test before preprocessing. These are minimal changes (no new model, no new features beyond proper numeric casting) and should move AUC back upward toward your target band while still producing the same valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sub = pd.read_csv(SAMPLE_SUB_CSV)

assert {"image_name", "target"}.issubset(train.columns)
assert {"image_name"}.issubset(test.columns)
assert list(sub.columns) == ["image_name", "target"]

train.shape, test.shape, sub.shape



## === cell 2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

base_features = ["sex", "age_approx", "anatom_site_general_challenge"]
extra_cat_features = []
group_col = "patient_id"
target_col = "target"

train_feat = train[base_features + extra_cat_features + [group_col]].copy()
test_feat = test[base_features + [group_col]].copy()


def _std_cat(s: pd.Series, missing_token: str) -> pd.Series:
    s = s.astype("object")
    s = s.where(~s.isna(), missing_token)
    s = s.astype(str).str.strip()
    s = s.replace({"": missing_token, "nan": missing_token, "None": missing_token})
    return s


train_feat["age_approx"] = pd.to_numeric(train_feat["age_approx"], errors="coerce")
test_feat["age_approx"] = pd.to_numeric(test_feat["age_approx"], errors="coerce")

train_feat["site_missing"] = (
    train_feat["anatom_site_general_challenge"].isna().astype(np.int8)
)
test_feat["site_missing"] = (
    test_feat["anatom_site_general_challenge"].isna().astype(np.int8)
)

train_feat["anatom_site_general_challenge"] = _std_cat(
    train_feat["anatom_site_general_challenge"], "site_unknown"
)
test_feat["anatom_site_general_challenge"] = _std_cat(
    test_feat["anatom_site_general_challenge"], "site_unknown"
)
train_feat["sex"] = _std_cat(train_feat["sex"], "sex_unknown")
test_feat["sex"] = _std_cat(test_feat["sex"], "sex_unknown")


def _add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_numeric(df["age_approx"], errors="coerce")
    df["age_log1p"] = np.log1p(s.clip(lower=0))
    df["age_missing"] = s.isna().astype(np.int8)
    return df


train_feat = _add_age_features(train_feat)
test_feat = _add_age_features(test_feat)

X = train_feat.copy()
y = train[target_col].astype(int).values
X_test = test_feat.copy()

numeric_features = [
    "age_approx",
    "age_log1p",
    "age_missing",
    "site_missing",
    "patient_target_mean",
    "site_target_mean",
    "sex_target_mean",
]
categorical_features = ["sex", "anatom_site_general_challenge"]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
    C=3.0,
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ]
)



## === cell 3
n_splits = 5
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_pred = np.zeros(len(test), dtype=np.float64)
oof_pred = np.zeros(len(train), dtype=np.float64)

global_prior = float(np.mean(y))

alpha_patient = 2.0
alpha_site = 30.0
alpha_sex = 40.0


def _safe_mean(sum_s: pd.Series, cnt_s: pd.Series) -> pd.Series:
    sum_s = sum_s.astype(float)
    cnt_s = cnt_s.astype(float)
    out = sum_s / cnt_s
    out = out.where(cnt_s > 0, np.nan)
    return out


full_df = pd.DataFrame(
    {
        group_col: X[group_col].values,
        "site": X["anatom_site_general_challenge"].astype("object").values,
        "sex": X["sex"].astype("object").values,
        "y": y,
    }
)

full_grp_sum = full_df.groupby(group_col)["y"].sum()
full_grp_cnt = full_df.groupby(group_col)["y"].count()
full_site_sum = full_df.groupby("site")["y"].sum()
full_site_cnt = full_df.groupby("site")["y"].count()
full_sex_sum = full_df.groupby("sex")["y"].sum()
full_sex_cnt = full_df.groupby("sex")["y"].count()

X_test_global = X_test.copy()

te_sum = X_test_global[group_col].map(full_grp_sum)
te_cnt = X_test_global[group_col].map(full_grp_cnt)
te_mean = _safe_mean(te_sum, te_cnt)
X_test_global["patient_target_mean"] = (
    te_mean * te_cnt.astype(float) + global_prior * alpha_patient
) / (te_cnt.astype(float) + alpha_patient)
X_test_global["patient_target_mean"] = X_test_global["patient_target_mean"].fillna(
    global_prior
)

te_site = X_test_global["anatom_site_general_challenge"].astype("object")
te_site_sum = te_site.map(full_site_sum)
te_site_cnt = te_site.map(full_site_cnt)
te_site_mean = _safe_mean(te_site_sum, te_site_cnt)
X_test_global["site_target_mean"] = (
    te_site_mean * te_site_cnt.astype(float) + global_prior * alpha_site
) / (te_site_cnt.astype(float) + alpha_site)
X_test_global["site_target_mean"] = X_test_global["site_target_mean"].fillna(
    global_prior
)

te_sex = X_test_global["sex"].astype("object")
te_sex_sum = te_sex.map(full_sex_sum)
te_sex_cnt = te_sex.map(full_sex_cnt)
te_sex_mean = _safe_mean(te_sex_sum, te_sex_cnt)
X_test_global["sex_target_mean"] = (
    te_sex_mean * te_sex_cnt.astype(float) + global_prior * alpha_sex
) / (te_sex_cnt.astype(float) + alpha_sex)
X_test_global["sex_target_mean"] = X_test_global["sex_target_mean"].fillna(global_prior)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=X[group_col].values), 1):
    X_tr = X.iloc[tr_idx].copy()
    y_tr = y[tr_idx]
    X_va = X.iloc[va_idx].copy()
    y_va = y[va_idx]

    tr_df = pd.DataFrame(
        {
            group_col: X_tr[group_col].values,
            "site": X_tr["anatom_site_general_challenge"].astype("object").values,
            "sex": X_tr["sex"].astype("object").values,
            "y": y_tr,
        }
    )

    prior_fold = global_prior

    grp_sum = tr_df.groupby(group_col)["y"].sum()
    grp_cnt = tr_df.groupby(group_col)["y"].count()

    sum_map_tr = X_tr[group_col].map(grp_sum)
    cnt_map_tr = X_tr[group_col].map(grp_cnt)

    loo_sum = sum_map_tr.astype(float) - y_tr
    loo_cnt = (cnt_map_tr.astype(float) - 1.0).clip(lower=0.0)
    loo_mean = (loo_sum / loo_cnt).where(loo_cnt > 0, np.nan)
    X_tr["patient_target_mean"] = (loo_mean * loo_cnt + prior_fold * alpha_patient) / (
        loo_cnt + alpha_patient
    )

    sum_map_va = X_va[group_col].map(grp_sum)
    cnt_map_va = X_va[group_col].map(grp_cnt)
    va_mean = _safe_mean(sum_map_va, cnt_map_va)
    X_va["patient_target_mean"] = (
        va_mean * cnt_map_va.astype(float) + prior_fold * alpha_patient
    ) / (cnt_map_va.astype(float) + alpha_patient)

    X_tr["patient_target_mean"] = X_tr["patient_target_mean"].fillna(prior_fold)
    X_va["patient_target_mean"] = X_va["patient_target_mean"].fillna(prior_fold)

    site_sum = tr_df.groupby("site")["y"].sum()
    site_cnt = tr_df.groupby("site")["y"].count()

    tr_site = X_tr["anatom_site_general_challenge"].astype("object")
    va_site = X_va["anatom_site_general_challenge"].astype("object")

    tr_site_sum = tr_site.map(site_sum).fillna(0.0).astype(float)
    tr_site_cnt = tr_site.map(site_cnt).fillna(0.0).astype(float)

    tr_site_loo_sum = tr_site_sum - y_tr
    tr_site_loo_cnt = (tr_site_cnt - 1.0).clip(lower=0.0)
    tr_site_loo_mean = (tr_site_loo_sum / tr_site_loo_cnt).where(
        tr_site_loo_cnt > 0, np.nan
    )

    X_tr["site_target_mean"] = (
        tr_site_loo_mean * tr_site_loo_cnt + prior_fold * alpha_site
    ) / (tr_site_loo_cnt + alpha_site)

    va_site_sum = va_site.map(site_sum)
    va_site_cnt = va_site.map(site_cnt)
    va_site_mean = _safe_mean(va_site_sum, va_site_cnt)
    X_va["site_target_mean"] = (
        va_site_mean * va_site_cnt.astype(float) + prior_fold * alpha_site
    ) / (va_site_cnt.astype(float) + alpha_site)

    X_tr["site_target_mean"] = X_tr["site_target_mean"].fillna(prior_fold)
    X_va["site_target_mean"] = X_va["site_target_mean"].fillna(prior_fold)

    sex_sum = tr_df.groupby("sex")["y"].sum()
    sex_cnt = tr_df.groupby("sex")["y"].count()

    tr_sex = X_tr["sex"].astype("object")
    va_sex = X_va["sex"].astype("object")

    tr_sex_sum = tr_sex.map(sex_sum).fillna(0.0).astype(float)
    tr_sex_cnt = tr_sex.map(sex_cnt).fillna(0.0).astype(float)

    tr_sex_loo_sum = tr_sex_sum - y_tr
    tr_sex_loo_cnt = (tr_sex_cnt - 1.0).clip(lower=0.0)
    tr_sex_loo_mean = (tr_sex_loo_sum / tr_sex_loo_cnt).where(
        tr_sex_loo_cnt > 0, np.nan
    )

    X_tr["sex_target_mean"] = (
        tr_sex_loo_mean * tr_sex_loo_cnt + prior_fold * alpha_sex
    ) / (tr_sex_loo_cnt + alpha_sex)

    va_sex_sum = va_sex.map(sex_sum)
    va_sex_cnt = va_sex.map(sex_cnt)
    va_sex_mean = _safe_mean(va_sex_sum, va_sex_cnt)
    X_va["sex_target_mean"] = (
        va_sex_mean * va_sex_cnt.astype(float) + prior_fold * alpha_sex
    ) / (va_sex_cnt.astype(float) + alpha_sex)

    X_tr["sex_target_mean"] = X_tr["sex_target_mean"].fillna(prior_fold)
    X_va["sex_target_mean"] = X_va["sex_target_mean"].fillna(prior_fold)

    feats = (
        base_features
        + extra_cat_features
        + [
            "age_log1p",
            "age_missing",
            "site_missing",
            "patient_target_mean",
            "site_target_mean",
            "sex_target_mean",
        ]
    )

    model.fit(X_tr[feats], y_tr)

    oof_pred[va_idx] = model.predict_proba(X_va[feats])[:, 1]
    test_pred += model.predict_proba(X_test_global[feats])[:, 1] / n_splits

print("OOF AUC:", roc_auc_score(y, oof_pred))
test_pred.min(), test_pred.max(), float(test_pred.mean())



## === cell 4
pred_by_image = pd.DataFrame(
    {"image_name": test["image_name"].values, "target": test_pred}
)
sub = sub[["image_name"]].merge(pred_by_image, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(float(np.nanmean(test_pred)))

sub["target"] = sub["target"].clip(0.0, 1.0)
sub.head()



## === cell 5
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={sub.shape} and columns={list(sub.columns)}")
print(sub["target"].describe())
