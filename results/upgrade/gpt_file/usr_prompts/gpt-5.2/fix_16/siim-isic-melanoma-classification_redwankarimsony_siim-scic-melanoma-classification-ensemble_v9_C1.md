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

0.913388160170087

# 6. Current score

0.75293

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.6672) has done: 'Your current notebook fails because it tries to read external “public submission” CSVs that don’t exist in this environment, so `sub` is never created and later cells crash. To keep the core idea (a simple ensemble-like prediction) but make it runnable end-to-end, I replace those missing external inputs with a lightweight, fully-local tabular baseline trained from `train.csv` metadata and used to score `test.csv`. This preserves the “metadata-based prediction” evaluation semantics, avoids any unavailable files, and guarantees a valid `submission.csv` with the required columns is written. I also make the file-path handling robust to both `/kaggle/input/...` and `/kaggle/data/...` layouts shown in your file tree.'
- What this solution (achieved 0.66728) has done: 'Your current score (0.6672) is far below the target (0.9134), so we should improve the metadata-only baseline while keeping the same core approach (a logistic regression over tabular metadata). The smallest high-impact fix is to prevent patient-level leakage/shift issues by using GroupKFold on `patient_id` to choose a better regularization strength (`C`) via out-of-fold ROC-AUC, then refit once on all training data. This keeps the exact same model family and preprocessing, but makes the hyperparameters better aligned to the AUC metric. I also add a lightweight interaction feature (`age_missing` and age*sex) without changing the overall pipeline structure, which typically gives a modest AUC lift for this dataset.'
- What this solution (achieved 0.70151) has done: 'Your current approach is a metadata-only logistic regression, so the most direct way to move AUC upward (without changing the model family or training loop structure) is to (1) add one or two high-signal metadata features that exist in `train.csv` and are derivable at test time, and (2) tune regularization a bit more finely using the same GroupKFold OOF-AUC selection you already do. Concretely, `patient_id` (as a categorical/group-level effect) and a simple nonlinearity for age (age-binning) often provide a noticeable lift for this competition while keeping the pipeline identical in spirit (same preprocessing + LogisticRegression). I keep GroupKFold, keep LogisticRegression, keep the submission writing logic unchanged, and only add these minimal features plus a slightly wider `C_grid` to better match the AUC objective.'
- What this solution (achieved 0.70325) has done: 'We keep your metadata-only LogisticRegression + GroupKFold selection intact, but make two minimal, high-signal tweaks that typically move AUC upward for this competition without changing the modeling approach. First, we add a couple of simple derived categorical features (`age_bin_x_sex` and `site_x_agebin`) that help a linear model capture mild nonlinear interactions already present in your existing features. Second, we slightly expand/refine the `C_grid` around the mid-range so the same OOF AUC selection has a better chance of landing on a stronger regularization setting. Everything else (paths, preprocessing structure, GroupKFold, LogisticRegression, submission writing) remains the same.'
- What this solution (achieved 0.70552) has done: 'To move your AUC upward toward the 0.913 target while keeping the exact same “metadata-only LogisticRegression + GroupKFold tuning” core, I make two minimal changes that usually help this specific competition: (1) add a couple of additional high-signal metadata features available in `train.csv` that are also present in `test.csv` (patient-level counts and sex-aware age normalization), and (2) slightly refine the regularization search grid around the best-performing region by making it denser (still the same tuning loop, same model). These tweaks preserve your training approach, model family, and evaluation semantics, but give the linear model a bit more predictive signal without touching images or changing the algorithm. The submission writing and paths remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.74855) has done: 'Your current metadata-only LogisticRegression is likely being held back by a train/test feature mismatch: `patient_img_count` is computed separately within train and test, which makes the same numeric feature mean different things across splits and can hurt AUC. I keep the exact same model, preprocessing, and GroupKFold tuning loop, but compute `patient_img_count` from the combined (train+test) patient frequency so the feature has consistent semantics at test time. Additionally, I compute `sex_age_median` using only non-missing ages (to avoid NaN medians for rare/unknown groups) and ensure `groups` uses the same filled `patient_id` representation as the features, improving stability without changing the approach. These minimal changes should move the score upward toward your target while preserving the core logic and producing the same submission format.'
- What this solution (achieved 0.74348) has done: 'Your current metadata-only LogisticRegression is already tuned with GroupKFold, so the smallest likely lift toward your target AUC is to reduce noisy/overpowering high-cardinality patient effects while keeping the same model family and training loop. I keep the exact pipeline and tuning logic, but (1) collapse `patient_id` into a frequency-binned categorical (instead of raw ID one-hot), and (2) use a log-scaled version of `patient_img_count` to stabilize its influence. This preserves evaluation semantics and uses only information available at test time (counts computed on combined train+test), and it still write a valid `submission.csv`.'
- What this solution (achieved 0.7483) has done: 'We keep your exact metadata-only LogisticRegression + GroupKFold tuning pipeline, but make two small changes that typically lift ROC-AUC for this competition without changing the modeling approach. First, we add one more lightweight interaction feature (`site_x_sex_x_agebin`) so the linear model can capture a common three-way dependency that’s hard to express otherwise. Second, we calibrate the raw probabilities using out-of-fold predictions (a monotonic Platt-style logistic calibrator), which often improves ranking quality for AUC when class_weight and regularization skew probability scales. Everything else (data paths, preprocessing style, tuning loop, final fit, submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.75068) has done: 'We keep your exact metadata-only LogisticRegression + GroupKFold tuning + Platt calibration structure, but make two minimal changes that usually improve ROC-AUC here. First, we use a group-aware out-of-fold calibrator (fit the calibrator on truly out-of-fold predictions rather than in-fold best_oof used for C selection), which reduces subtle overfitting in the calibration step without changing the modeling approach. Second, we add a very small, robust numeric feature (`age_squared`) to let the linear model capture mild nonlinearity in age while preserving the same pipeline and loss. These are low-risk changes intended to increase AUC from 0.7483 toward your 0.9134 target.'
- What this solution (achieved 0.76173) has done: 'The timeout is dominated by repeatedly fitting hundreds of `LogisticRegression(max_iter=2000)` models inside nested GroupKFold loops (outer 5 folds × inner 4 folds × 25 C values, plus another 5×25 grid), plus repeated creation of split indices and dense temporary arrays. I keep the exact same model, CV structure, C grid, and calibration approach, but cut overhead by (1) precomputing all fold indices once, (2) parallelizing the per‑C evaluation with joblib (process-based, deterministic), and (3) reusing constant arrays and avoiding repeated `np.zeros(...)` allocations in every split call. These changes are algorithmically identical (same fits, same data, same scoring) but remove a large amount of Python-loop overhead and enable multi-core CPU utilization.'
- What this solution (achieved 0.75286) has done: 'I keep your exact metadata-only LogisticRegression + GroupKFold tuning + OOF Platt calibration pipeline, but fix one high-impact source of overfitting that’s holding AUC down: the `patient_target_prior` feature currently uses the full training labels for every row (including each row’s own target and fold’s validation targets), which leaks labels into training/CV and harms generalization to the test leaderboard. I rebuild `patient_target_prior` as a group-aware out-of-fold (OOF) smoothed prior computed only from each fold’s training portion, and use a separately smoothed full-train mapping only for the final test-time feature generation. This is the smallest change that directly aligns the feature with the evaluation setting and typically produces a material AUC lift while preserving your model family, CV/training structure, and submission semantics. All paths remain unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.75286) has done: 'We’re still far below the target AUC, so the smallest likely improvement while keeping the same metadata-only LogisticRegression + GroupKFold + Platt-calibration core is to remove a remaining form of train/test mismatch in the `patient_target_prior` feature. Right now, test-time priors use only training labels, but the prior is meant to represent a “patient’s propensity” and your other patient features already use combined (train+test) patient frequency; making the prior consistent by using a combined-patient prior (train-label-informed, but computed for all patient_ids seen in train+test) typically improves generalization without changing the model family or training loop. Concretely, we compute a smoothed prior keyed by patient_id where the denominator uses combined counts (train+test) while the numerator uses train label sums, then use this mapping for test (and still keep the leakage-safe OOF prior for training). This is a minimal feature-engineering correction and should nudge leaderboard AUC upward toward the target while preserving everything else.'
- What this solution (achieved 0.75293) has done: 'Your current pipeline is a metadata-only LogisticRegression with GroupKFold tuning and a Platt-style calibrator; to improve AUC toward the target without changing the core approach, the highest-impact minimal change is to reduce instability from the very noisy/high-cardinality patient prior. I keep the exact same model family, CV structure, and calibration, but make the `patient_target_prior` more robust by (1) using stronger smoothing and (2) shrinking the prior toward the global mean (a monotonic linear transform that typically improves ranking generalization). This preserves evaluation semantics (still a probability model; still same features and LR) but usually yields a measurable lift when patient_id leakage-like signals don’t transfer to test. Everything else (paths, preprocessing, training loop, submission writing) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
np.random.seed(42)


def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the paths exist: {paths}")


DATA_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]

base_dir = _first_existing_path(DATA_CANDIDATES)

train_csv = _first_existing_path(
    [
        os.path.join(base_dir, "train.csv"),
        "/kaggle/input/siim-isic-melanoma-classification/train.csv",
        "/kaggle/data/siim-isic-melanoma-classification/train.csv",
    ]
)
test_csv = _first_existing_path(
    [
        os.path.join(base_dir, "test.csv"),
        "/kaggle/input/siim-isic-melanoma-classification/test.csv",
        "/kaggle/data/siim-isic-melanoma-classification/test.csv",
    ]
)
sample_sub_csv = _first_existing_path(
    [
        os.path.join(base_dir, "sample_submission.csv"),
        "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv",
        "/kaggle/data/siim-isic-melanoma-classification/sample_submission.csv",
    ]
)

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
sub = pd.read_csv(sample_sub_csv)

_all_pid = pd.concat(
    [
        train["patient_id"].fillna("unknown").astype(str),
        test["patient_id"].fillna("unknown").astype(str),
    ],
    axis=0,
    ignore_index=True,
)
all_pid_counts = _all_pid.value_counts(dropna=False)

_train_sex = train["sex"].fillna("unknown").astype(str)
_train_age = train["age_approx"].astype(float)
global_age_median = float(_train_age.median())

sex_age_median = (
    pd.DataFrame({"sex": _train_sex, "age": _train_age})
    .dropna(subset=["age"])
    .groupby("sex")["age"]
    .median()
    .to_dict()
)

train_pid = train["patient_id"].fillna("unknown").astype(str)
test_pid = test["patient_id"].fillna("unknown").astype(str)
global_target_mean = float(train["target"].mean())

alpha = 50.0  # was 10.0


def _smoothed_pid_prior(pid_series, y_series, global_mean, alpha_):
    pid_series = pid_series.astype(str)
    pid_mean = y_series.groupby(pid_series).mean()
    pid_cnt = pid_series.value_counts()
    smooth = (pid_mean * pid_cnt + global_mean * alpha_) / (pid_cnt + alpha_)
    return smooth.to_dict()


def _smoothed_pid_prior_combined_counts(
    train_pid_s, y_s, all_counts_s, global_mean, alpha_
):
    train_pid_s = train_pid_s.astype(str)
    y_s = pd.Series(y_s).astype(float).reset_index(drop=True)
    train_pid_s = train_pid_s.reset_index(drop=True)

    sum_y = y_s.groupby(train_pid_s).sum()
    cnt_train = train_pid_s.value_counts()

    cnt_all = all_counts_s.astype(float)

    idx = cnt_all.index
    sum_y = sum_y.reindex(idx).fillna(0.0)
    denom = cnt_all.reindex(idx).fillna(cnt_train.reindex(idx).fillna(0.0))
    numer = sum_y + global_mean * alpha_
    denom2 = denom + alpha_
    prior = (numer / denom2).astype(float)
    return prior.to_dict()


groups = train["patient_id"].fillna("unknown").astype(str).to_numpy()
gkf = GroupKFold(n_splits=5)

n_train = len(train)
dummy_X = np.empty(n_train, dtype=np.uint8)
outer_splits = [
    (tr_idx, va_idx)
    for tr_idx, va_idx in gkf.split(dummy_X, train["target"].values, groups=groups)
]

pid_prior_oof = np.full(n_train, global_target_mean, dtype=np.float32)
for tr_idx, va_idx in outer_splits:
    pid_map = _smoothed_pid_prior(
        train_pid.iloc[tr_idx],
        train.loc[tr_idx, "target"],
        global_target_mean,
        alpha,
    )
    pid_prior_oof[va_idx] = (
        train_pid.iloc[va_idx]
        .map(pid_map)
        .fillna(global_target_mean)
        .astype(np.float32)
        .to_numpy()
    )

pid_target_smooth_full = _smoothed_pid_prior_combined_counts(
    train_pid_s=train_pid,
    y_s=train["target"].values,
    all_counts_s=all_pid_counts,
    global_mean=global_target_mean,
    alpha_=alpha,
)

PRIOR_SHRINK = 0.35  # 0 -> use only global mean, 1 -> keep original prior

pid_prior_oof = (
    global_target_mean + PRIOR_SHRINK * (pid_prior_oof - global_target_mean)
).astype(np.float32)
pid_target_smooth_full = {
    k: float(global_target_mean + PRIOR_SHRINK * (v - global_target_mean))
    for k, v in pid_target_smooth_full.items()
}


def add_features(
    df: pd.DataFrame, patient_prior_map=None, patient_prior_oof=None
) -> pd.DataFrame:
    out = df.copy()

    out["patient_id"] = out["patient_id"].fillna("unknown").astype(str)

    out["age_missing"] = out["age_approx"].isna().astype(np.int8)
    sex = out["sex"].fillna("unknown").astype(str)
    site = out["anatom_site_general_challenge"].fillna("unknown").astype(str)
    out["sex_x_site"] = sex + "_" + site

    age = out["age_approx"].astype(float)
    age_filled = age.fillna(global_age_median)

    out["age_bin"] = pd.cut(
        age_filled,
        bins=[-np.inf, 20, 35, 45, 55, 65, 75, np.inf],
        labels=["a0_20", "a20_35", "a35_45", "a45_55", "a55_65", "a65_75", "a75_inf"],
    ).astype(str)

    out["age_bin_x_sex"] = out["age_bin"].astype(str) + "_" + sex
    out["site_x_agebin"] = site + "_" + out["age_bin"].astype(str)
    out["site_x_sex_x_agebin"] = site + "_" + sex + "_" + out["age_bin"].astype(str)

    pid_count = out["patient_id"].map(all_pid_counts).fillna(1).astype(np.float32)
    out["patient_img_count"] = pid_count
    out["patient_img_count_log1p"] = np.log1p(pid_count).astype(np.float32)

    out["patient_count_bin"] = pd.cut(
        pid_count,
        bins=[0, 1, 2, 3, 5, 10, 20, np.inf],
        labels=["c1", "c2", "c3", "c4_5", "c6_10", "c11_20", "c21p"],
        include_lowest=True,
        right=True,
    ).astype(str)

    sex_med = sex.map(sex_age_median).astype(float)
    sex_med = sex_med.fillna(global_age_median)
    out["age_centered_by_sex"] = (age_filled - sex_med).astype(np.float32)
    out["age_squared"] = (age_filled**2).astype(np.float32)

    out["patient_freq_bin"] = pd.cut(
        pid_count,
        bins=[0, 1, 2, 3, 5, 10, 20, np.inf],
        labels=["p1", "p2", "p3", "p4_5", "p6_10", "p11_20", "p21p"],
        include_lowest=True,
        right=True,
    ).astype(str)

    if patient_prior_oof is not None:
        out["patient_target_prior"] = patient_prior_oof.astype(np.float32)
    else:
        if patient_prior_map is None:
            patient_prior_map = {}
        out["patient_target_prior"] = (
            out["patient_id"]
            .map(patient_prior_map)
            .fillna(global_target_mean)
            .astype(np.float32)
        )

    return out


feature_cols = [
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "age_missing",
    "sex_x_site",
    "patient_count_bin",
    "age_bin",
    "age_bin_x_sex",
    "site_x_agebin",
    "site_x_sex_x_agebin",
    "patient_img_count",
    "patient_img_count_log1p",
    "age_centered_by_sex",
    "age_squared",
    "patient_freq_bin",
    "patient_target_prior",
]

X_train_df = add_features(train, patient_prior_oof=pid_prior_oof)[feature_cols].copy()
y_train = train["target"].astype(int).to_numpy()
X_test_df = add_features(test, patient_prior_map=pid_target_smooth_full)[
    feature_cols
].copy()

categorical_features = [
    "sex",
    "anatom_site_general_challenge",
    "age_missing",
    "sex_x_site",
    "patient_count_bin",
    "age_bin",
    "age_bin_x_sex",
    "site_x_agebin",
    "site_x_sex_x_agebin",
    "patient_freq_bin",
]
numeric_features = [
    "age_approx",
    "patient_img_count",
    "patient_img_count_log1p",
    "age_centered_by_sex",
    "age_squared",
    "patient_target_prior",
]

preprocess = ColumnTransformer(
    transformers=[
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
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            ),
            numeric_features,
        ),
    ],
    remainder="drop",
)

X_train = preprocess.fit_transform(X_train_df)
X_test = preprocess.transform(X_test_df)

C_grid = [
    0.003,
    0.005,
    0.007,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.06,
    0.08,
    0.1,
    0.12,
    0.15,
    0.18,
    0.2,
    0.25,
    0.3,
    0.4,
    0.6,
    0.8,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
]


def _cv_auc_for_C(C, X, y, splits):
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in splits:
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=1,  # single-thread per process; outer parallelism handled by joblib
            class_weight="balanced",
            random_state=42,
            C=C,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return float(roc_auc_score(y, oof)), C, oof


oof_for_cal = np.zeros(len(y_train), dtype=np.float64)
for outer_tr_idx, outer_va_idx in outer_splits:
    X_outer_tr = X_train[outer_tr_idx]
    y_outer_tr = y_train[outer_tr_idx]
    groups_outer_tr = groups[outer_tr_idx]

    inner_gkf = GroupKFold(n_splits=4)
    dummy_inner = np.empty(len(outer_tr_idx), dtype=np.uint8)
    inner_splits = [
        (itr, iva)
        for itr, iva in inner_gkf.split(dummy_inner, y_outer_tr, groups=groups_outer_tr)
    ]

    def _inner_auc_for_C(C):
        inner_oof = np.zeros(len(outer_tr_idx), dtype=np.float64)
        for in_tr_rel, in_va_rel in inner_splits:
            clf = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                n_jobs=1,
                class_weight="balanced",
                random_state=42,
                C=C,
            )
            clf.fit(X_outer_tr[in_tr_rel], y_outer_tr[in_tr_rel])
            inner_oof[in_va_rel] = clf.predict_proba(X_outer_tr[in_va_rel])[:, 1]
        return float(roc_auc_score(y_outer_tr, inner_oof)), C

    inner_results = Parallel(n_jobs=-1, prefer="processes", batch_size=1)(
        delayed(_inner_auc_for_C)(C) for C in C_grid
    )
    best_auc_inner, best_C_inner = max(inner_results, key=lambda t: t[0])

    final_inner_clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        n_jobs=1,
        class_weight="balanced",
        random_state=42,
        C=best_C_inner,
    )
    final_inner_clf.fit(X_outer_tr, y_outer_tr)
    oof_for_cal[outer_va_idx] = final_inner_clf.predict_proba(X_train[outer_va_idx])[
        :, 1
    ]

results = Parallel(n_jobs=-1, prefer="processes", batch_size=1)(
    delayed(_cv_auc_for_C)(C, X_train, y_train, outer_splits) for C in C_grid
)
best_auc, best_C, best_oof = max(results, key=lambda t: t[0])

final_clf = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    n_jobs=-1,  # final single fit can use all cores safely
    class_weight="balanced",
    random_state=42,
    C=best_C,
)
final_clf.fit(X_train, y_train)
test_pred = final_clf.predict_proba(X_test)[:, 1].astype(np.float64)

calibrator = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    n_jobs=-1,
    class_weight=None,
    random_state=42,
    C=1.0,
)
calibrator.fit(oof_for_cal.reshape(-1, 1), y_train)
test_pred = calibrator.predict_proba(test_pred.reshape(-1, 1))[:, 1]



## === cell 1
pred_df = pd.DataFrame(
    {
        "image_name": test["image_name"].values,
        "target": test_pred.astype(np.float32),
    }
)

sub = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(float(train["target"].mean()))

sub.to_csv("submission.csv", index=False)
sub.head()
