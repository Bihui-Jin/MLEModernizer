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

0.918821332226603

# 6. Current score

0.76345

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66776) has done: 'I remove the dependency on the missing external `public-submission-melanoma-95` files that cause the FileNotFoundError, and instead generate predictions from the provided competition metadata (train/test CSVs) so the notebook runs end-to-end in your environment. To keep changes minimal while still producing a meaningful AUC-oriented submission, I train a simple sklearn logistic regression on the same metadata columns and output predicted probabilities for the test set. I also make the data path robust to both `/kaggle/input/...` and the provided `/kaggle/data/...` mirrors, and ensure the written file is exactly `submission.csv` with `image_name,target`. This should yield a valid submission and typically a reasonable baseline score (though exact AUC depends on the split/feature signal).'
- What this solution (achieved 0.67512) has done: 'Your current 0.66776 score is far below the 0.9188 target (gap ≈ -0.251), so we should legitimately improve predictive signal while keeping the same “metadata-only sklearn model” core logic. The biggest low-risk gain is to add the already-available metadata columns `patient_id` and `diagnosis` (train-only) via a target-mean encoding learned on train and applied to test, while keeping the logistic regression pipeline. This preserves the training approach (single LogisticRegression on tabular features) but gives the model much stronger categorical signal than one-hot alone. I also ensure the encoded features are computed without leaking test targets and that submission alignment remains one-to-one with `test.csv`.'
- What this solution (achieved 0.75956) has done: 'Your current score (0.67512) is far below the target (0.91882), so we should improve predictive signal while keeping the same metadata-only LogisticRegression core. The biggest issue is that `diagnosis` is train-only but you currently set a constant for all test rows, which adds no signal and can even hurt; we remove that test-time constant feature. Then we add a minimal, leakage-safe “grouped patient” target-encoding variant using per-patient counts and smoothed mean (still the same single LogisticRegression pipeline) to better leverage `patient_id` without changing the training approach. Finally, we fix the submission alignment to use `test.csv` order directly (no merge), eliminating any risk of misalignment.'
- What this solution (achieved 0.7431) has done: 'We keep your metadata-only single LogisticRegression pipeline intact, but tighten the target-encoding for `patient_id` to avoid optimistic leakage from using each row’s own label in its encoding (this typically lifts AUC materially while preserving core logic). Concretely, we replace the in-sample patient target-mean encoding with an out-of-fold (OOF) smoothed encoding computed via stratified CV on the training set, and then apply the full-data encoding to the test set. We also add one additional patient-derived numeric feature (`patient_id_unique_images` is already your count; we add a stabilized log1p transform alongside it) without changing the model type or training loop. These are minimal, legitimate changes aimed at increasing your 0.75956 score toward the 0.91882 target.'
- What this solution (achieved 0.77024) has done: 'To move your AUC up toward the 0.9188 target while keeping the same “metadata-only + single LogisticRegression pipeline” core, I make the smallest changes that add legitimate signal from `patient_id` without changing the training approach. Specifically, I (1) fix the OOF target-encoding to compute the global mean *within each fold’s train split* (reduces fold-noise/shift), and (2) add a couple of additional leakage-safe patient aggregate features derived only from training labels: a smoothed per-patient mean target (fit on full train, applied to train/test) and a per-patient malignant/benign log-odds proxy with smoothing. These are just extra numeric columns (same model, same loss/solver), and typically give a material AUC lift for this competition. Submission writing stays identical (`submission.csv` with `image_name,target` in `test.csv` order).'
- What this solution (achieved 0.77827) has done: 'Your current score (0.77024) is well below the target (0.91882), so we should add a bit more legitimate signal while keeping the same metadata-only single `LogisticRegression` pipeline intact. The biggest low-risk gain here is to add one more leakage-safe patient-derived feature: the **OOF** (out-of-fold) smoothed target mean *and* the corresponding **OOF logit** (rather than only full-train logit), because patient-level label information is highly predictive in this competition but must be encoded without using each row’s own label. This preserves your exact training approach (single sklearn model; same preprocessing; no new loops beyond the existing CV encoding), but typically improves generalization and AUC. I also keep your existing full-train patient aggregates (applied to test) because they provide test-time signal, while ensuring the train-side versions are OOF to avoid in-sample inflation.'
- What this solution (achieved 0.68904) has done: 'We keep your exact “metadata + patient OOF target encoding + single LogisticRegression” approach, but fix two issues that likely cap AUC: (1) your StratifiedKFold OOF encoding currently splits by rows, so the same `patient_id` can appear in both train/valid within a fold (patient leakage); we switch to a patient-grouped fold split so each patient is confined to a single fold. (2) You currently duplicate `patient_id_te` and `patient_id_smoothed_mean` with identical values, which is redundant and can slightly hurt calibration; we keep just one copy (still the same features, just removing a duplicate column). These are minimal, metric-relevant changes and should move your score upward toward the 0.9188 target while preserving the core modeling logic and producing the same `submission.csv` format.'
- What this solution (achieved 0.68907) has done: 'Your current AUC (0.689) is far below the target (0.9188), so we need a legitimate signal lift while keeping the same “metadata + patient OOF target encoding + single LogisticRegression pipeline” core intact. The biggest low-risk issue is that patient-level encodings and counts are currently computed using only the **training** patient history, leaving **test patients unseen in train** with weak/zero patient features; we add a leakage-safe set of **test-only patient aggregate features** (count, count_log1p) derived from `test.csv` alone to better characterize repeated test patients without using labels. We also add a tiny amount of regularization tuning (C) consistent with the same solver/model (still LogisticRegression) to reduce underfitting from the new features. Submission writing and row alignment remain unchanged.'
- What this solution (achieved 0.70625) has done: 'Your current AUC (0.689) is far below the 0.9188 target, so we need a legitimate lift while keeping the exact same “metadata + patient encodings + single LogisticRegression” core. The biggest issue is that the current patient encoding uses only `patient_id`, but in this competition the *combination* of `patient_id` and `anatom_site_general_challenge` captures much more consistent lesion history; we add a leakage-safe grouped OOF target encoding for the pair `(patient_id, anatom_site_general_challenge)` and its OOF logit, computed with the same grouped-fold logic to avoid patient leakage. This is a minimal feature addition (no architecture/training loop change) and should materially improve ranking/AUC. We keep submission writing identical and in `test.csv` order.'
- What this solution (achieved 0.7085) has done: 'We’re far below the target AUC, so the smallest legitimate move is to add a bit more leakage-safe signal without changing the core approach (metadata features + OOF/group target encodings + single LogisticRegression). I add one additional OOF encoding for the pair `(patient_id, sex)` (same encoding machinery, still grouped folds by patient_id to prevent leakage) because sex is informative and interacts with patient history similarly to the existing `(patient_id, anatom_site)` feature. I also add a stable “age bin” categorical feature (derived from `age_approx`) to improve non-linear age effects while keeping the same preprocessing/model. Everything else (paths, model type, solver, training flow, submission format and order) remains unchanged.'
- What this solution (achieved 0.71198) has done: 'Your current AUC (0.7085) is far below the 0.9188 target, so we should add a small amount of legitimate signal while keeping the same metadata-only + OOF/group target-encodings + single LogisticRegression pipeline intact. The largest missing low-risk signal is **(patient_id, age_bin)** interactions: patients often have consistent lesion behavior and age is strongly correlated, so adding an OOF/group target encoding for `(patient_id, age_bin)` should improve ranking without changing the modeling approach. I compute this encoding with the exact same grouped-fold machinery (grouped by patient_id to prevent leakage), add the resulting mean/logit features to the numeric set, and keep everything else (paths, preprocessing, classifier, submission writing) unchanged. This is a minimal feature addition aimed specifically at increasing AUC toward your target.'
- What this solution (achieved 0.32912) has done: 'Your current AUC (0.71198) is far below the 0.91882 target, so we should add a small amount of legitimate signal without changing your core approach (metadata + leakage-safe grouped OOF target encodings + single LogisticRegression). The biggest low-risk missing metadata signal in this competition is the train-only `diagnosis` column, which we can use *without leaking labels* by creating an OOF smoothed target-encoding for `diagnosis` on train, and mapping a full-train smoothed encoding to test with an “unknown diagnosis” fallback to the global mean. This preserves the same model (LogisticRegression), same training flow (fit once), and just adds 2 numeric columns (`diagnosis_te`, `diagnosis_logit`) that typically improve ranking. I also add a single interaction encoding `(diagnosis, anatom_site_general_challenge)` with the same OOF machinery; it’s the same encoding logic you already use for other group keys, just another key that can capture clinically meaningful structure.'
- What this solution (achieved 0.76367) has done: 'Your current score (0.32912) is far below the target AUC (0.91882), so we should make a small, high-impact fix rather than adding more features. The big regression is that you’re using train-only `diagnosis` encodings but the test set has no `diagnosis`, so your `diagnosis_te/diag_site_te` collapse to (near) constants and can seriously hurt ranking; we remove those diagnosis-derived features entirely (while keeping the same LogisticRegression + preprocessing + OOF/group-TE core). We also undo the forced `class_weight="balanced"` (use default `None`) because Kaggle AUC is threshold-free and balanced weighting often worsens probability ranking calibration here; this is a minimal parameter change that frequently improves AUC without changing the model family or training flow. Everything else (paths, patient-grouped OOF encodings, pipeline, submission writing and order) stays the same and still produce `submission.csv`.'
- What this solution (achieved 0.76345) has done: 'Your current AUC (0.76367) is far below the target (0.91882), so we should make a small, legitimate improvement without changing the core “metadata + OOF/group target encodings + single LogisticRegression” approach. The biggest low-risk gain here is to stop underfitting by slightly increasing LogisticRegression capacity and letting the solver converge more reliably on the expanded feature set (this often improves AUC ranking without changing semantics). Concretely, I only tune `C` upward (less regularization) and increase `max_iter` to ensure convergence; everything else (features/encodings, grouped-fold logic, preprocessing, and submission format/order) remains the same. This is minimal, stable, and should move the score upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = _first_existing(
    [
        "/kaggle/input/siim-isic-melanoma-classification",
        "/kaggle/data/siim-isic-melanoma-classification",
        "../input/siim-isic-melanoma-classification",
        "../kaggle/input/siim-isic-melanoma-classification",
        "../kaggle/data/siim-isic-melanoma-classification",
    ]
)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate siim-isic-melanoma-classification dataset directory."
    )

train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_sub_path)

required_train_cols = {
    "image_name",
    "target",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "patient_id",
    "diagnosis",
}
required_test_cols = {
    "image_name",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "patient_id",
}
if not required_train_cols.issubset(train.columns):
    raise ValueError(
        f"train.csv missing columns: {sorted(required_train_cols - set(train.columns))}"
    )
if not required_test_cols.issubset(test.columns):
    raise ValueError(
        f"test.csv missing columns: {sorted(required_test_cols - set(test.columns))}"
    )
if not {"image_name", "target"}.issubset(sub.columns):
    raise ValueError("sample_submission.csv must have columns: image_name,target")

train.shape, test.shape, sub.shape



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def _iterative_group_stratified_folds(groups, y, n_splits=5, seed=42):
    """
    Patient-grouped folds (no patient appears in both train/valid in a fold),
    while approximately stratifying on patient-level target mean.
    """
    rng = np.random.RandomState(seed)

    groups = pd.Series(groups).astype(str)
    y = np.asarray(y).astype(int)

    df = pd.DataFrame({"pid": groups, "y": y})
    pid_stats = df.groupby("pid")["y"].agg(["mean", "count"])
    pid_list = pid_stats.index.to_numpy()
    pid_mean = pid_stats["mean"].to_numpy()

    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001])
    pid_bin = np.digitize(pid_mean, bins) - 1

    fold_pids = [[] for _ in range(n_splits)]
    fold_bin_counts = np.zeros((n_splits, len(bins) - 1), dtype=int)

    for b in range(len(bins) - 1):
        pids_in_bin = pid_list[pid_bin == b]
        rng.shuffle(pids_in_bin)
        for pid in pids_in_bin:
            sizes = np.array([len(f) for f in fold_pids])
            min_bin = fold_bin_counts[:, b].min()
            candidate_folds = np.where(fold_bin_counts[:, b] == min_bin)[0]
            j = candidate_folds[np.argmin(sizes[candidate_folds])]
            fold_pids[j].append(pid)
            fold_bin_counts[j, b] += 1

    pid_to_rows = df.groupby("pid").indices  # pid -> np.array(row_idxs)
    folds = []
    all_idx = np.arange(len(df))
    for k in range(n_splits):
        va_idx = (
            np.concatenate([pid_to_rows[pid] for pid in fold_pids[k]])
            if len(fold_pids[k])
            else np.array([], dtype=int)
        )
        va_mask = np.zeros(len(df), dtype=bool)
        va_mask[va_idx] = True
        tr_idx = all_idx[~va_mask]
        folds.append((tr_idx, va_idx))
    return folds


def _oof_patient_target_stats(
    train_df,
    test_df,
    patient_col="patient_id",
    target_col="target",
    m_mean=200.0,
    alpha_logit=1.0,
    n_splits=5,
    seed=42,
):
    """
    Grouped OOF encodings by patient_id (prevents patient leakage).
    Returns:
      oof_smoothed_mean, test_smoothed_mean,
      oof_logit,         test_logit
    """
    tr_pid = train_df[patient_col].fillna("__MISSING__").astype(str).to_numpy()
    te_pid = test_df[patient_col].fillna("__MISSING__").astype(str).to_numpy()
    y = train_df[target_col].astype(float).to_numpy()

    oof_mean = np.empty(train_df.shape[0], dtype=np.float64)
    oof_logit = np.empty(train_df.shape[0], dtype=np.float64)

    folds = _iterative_group_stratified_folds(
        tr_pid, y.astype(int), n_splits=n_splits, seed=seed
    )

    for tr_idx, va_idx in folds:
        fold_tr_pid = tr_pid[tr_idx]
        fold_tr_y = y[tr_idx]

        fold_global_mean = float(np.mean(fold_tr_y))
        fold_global_p = float(np.clip(fold_global_mean, 1e-6, 1.0 - 1e-6))
        fold_global_logit = float(np.log(fold_global_p / (1.0 - fold_global_p)))

        fold = pd.DataFrame({"pid": fold_tr_pid, "y": fold_tr_y})
        grp = fold.groupby("pid")["y"].agg(["sum", "count"])

        sm_mean = (grp["sum"] + fold_global_mean * m_mean) / (grp["count"] + m_mean)

        p = (grp["sum"] + alpha_logit) / (grp["count"] + 2.0 * alpha_logit)
        p = np.clip(p.to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
        sm_logit = pd.Series(np.log(p / (1.0 - p)), index=grp.index)

        va_pid = pd.Series(tr_pid[va_idx])
        oof_mean[va_idx] = (
            va_pid.map(sm_mean).astype(np.float64).fillna(fold_global_mean).to_numpy()
        )
        oof_logit[va_idx] = (
            va_pid.map(sm_logit).astype(np.float64).fillna(fold_global_logit).to_numpy()
        )

    full_global_mean = float(np.mean(y))
    full_global_p = float(np.clip(full_global_mean, 1e-6, 1.0 - 1e-6))
    full_global_logit = float(np.log(full_global_p / (1.0 - full_global_p)))

    full = pd.DataFrame({"pid": tr_pid, "y": y})
    grp_full = full.groupby("pid")["y"].agg(["sum", "count"])

    test_sm_mean_map = (grp_full["sum"] + full_global_mean * m_mean) / (
        grp_full["count"] + m_mean
    )
    p_full = (grp_full["sum"] + alpha_logit) / (grp_full["count"] + 2.0 * alpha_logit)
    p_full = np.clip(p_full.to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    test_sm_logit_map = pd.Series(np.log(p_full / (1.0 - p_full)), index=grp_full.index)

    te_pid_s = pd.Series(te_pid)
    test_mean = (
        te_pid_s.map(test_sm_mean_map)
        .astype(np.float64)
        .fillna(full_global_mean)
        .to_numpy()
    )
    test_logit = (
        te_pid_s.map(test_sm_logit_map)
        .astype(np.float64)
        .fillna(full_global_logit)
        .to_numpy()
    )

    return oof_mean, test_mean, oof_logit, test_logit


def _oof_groupkey_target_stats(
    train_df,
    test_df,
    group_key_tr,  # pd.Series aligned to train_df
    group_key_te,  # pd.Series aligned to test_df
    patient_groups,  # pd.Series aligned to train_df (used only for grouped folds)
    target_col="target",
    m_mean=200.0,
    alpha_logit=1.0,
    n_splits=5,
    seed=42,
):
    y = train_df[target_col].astype(float).to_numpy()

    g_tr = group_key_tr.fillna("__MISSING__").astype(str).to_numpy()
    g_te = group_key_te.fillna("__MISSING__").astype(str).to_numpy()
    pid = patient_groups.fillna("__MISSING__").astype(str).to_numpy()

    oof_mean = np.empty(train_df.shape[0], dtype=np.float64)
    oof_logit = np.empty(train_df.shape[0], dtype=np.float64)

    folds = _iterative_group_stratified_folds(
        pid, y.astype(int), n_splits=n_splits, seed=seed
    )

    for tr_idx, va_idx in folds:
        fold_tr_g = g_tr[tr_idx]
        fold_tr_y = y[tr_idx]

        fold_global_mean = float(np.mean(fold_tr_y))
        fold_global_p = float(np.clip(fold_global_mean, 1e-6, 1.0 - 1e-6))
        fold_global_logit = float(np.log(fold_global_p / (1.0 - fold_global_p)))

        fold = pd.DataFrame({"g": fold_tr_g, "y": fold_tr_y})
        grp = fold.groupby("g")["y"].agg(["sum", "count"])

        sm_mean = (grp["sum"] + fold_global_mean * m_mean) / (grp["count"] + m_mean)

        p = (grp["sum"] + alpha_logit) / (grp["count"] + 2.0 * alpha_logit)
        p = np.clip(p.to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
        sm_logit = pd.Series(np.log(p / (1.0 - p)), index=grp.index)

        va_g = pd.Series(g_tr[va_idx])
        oof_mean[va_idx] = (
            va_g.map(sm_mean).astype(np.float64).fillna(fold_global_mean).to_numpy()
        )
        oof_logit[va_idx] = (
            va_g.map(sm_logit).astype(np.float64).fillna(fold_global_logit).to_numpy()
        )

    full_global_mean = float(np.mean(y))
    full_global_p = float(np.clip(full_global_mean, 1e-6, 1.0 - 1e-6))
    full_global_logit = float(np.log(full_global_p / (1.0 - full_global_p)))

    full = pd.DataFrame({"g": g_tr, "y": y})
    grp_full = full.groupby("g")["y"].agg(["sum", "count"])
    test_sm_mean_map = (grp_full["sum"] + full_global_mean * m_mean) / (
        grp_full["count"] + m_mean
    )

    p_full = (grp_full["sum"] + alpha_logit) / (grp_full["count"] + 2.0 * alpha_logit)
    p_full = np.clip(p_full.to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    test_sm_logit_map = pd.Series(np.log(p_full / (1.0 - p_full)), index=grp_full.index)

    te_g = pd.Series(g_te)
    test_mean = (
        te_g.map(test_sm_mean_map)
        .astype(np.float64)
        .fillna(full_global_mean)
        .to_numpy()
    )
    test_logit = (
        te_g.map(test_sm_logit_map)
        .astype(np.float64)
        .fillna(full_global_logit)
        .to_numpy()
    )

    return oof_mean, test_mean, oof_logit, test_logit


def _make_age_bin(s):
    s_num = pd.to_numeric(s, errors="coerce")
    bins = [-np.inf, 30, 45, 55, 65, 75, np.inf]
    labels = ["<30", "30-45", "45-55", "55-65", "65-75", "75+"]
    return pd.cut(s_num, bins=bins, labels=labels).astype(object)


base_feature_cols = ["sex", "age_approx", "anatom_site_general_challenge"]

X_train_base = train[base_feature_cols].copy()
y_train = train["target"].astype(int).copy()
X_test_base = test[base_feature_cols].copy()

X_train_base["age_bin"] = _make_age_bin(train["age_approx"])
X_test_base["age_bin"] = _make_age_bin(test["age_approx"])

pid_oof_mean_tr, pid_mean_te, pid_oof_logit_tr, pid_logit_te = (
    _oof_patient_target_stats(
        train,
        test,
        patient_col="patient_id",
        target_col="target",
        m_mean=200.0,
        alpha_logit=1.0,
        n_splits=5,
        seed=RANDOM_STATE,
    )
)

train_pid = train["patient_id"].fillna("__MISSING__").astype(str)
test_pid = test["patient_id"].fillna("__MISSING__").astype(str)

pid_counts_train = train_pid.value_counts(dropna=False)
pid_count_tr = train_pid.map(pid_counts_train).astype(np.float64).to_numpy()
pid_count_te_from_train = (
    test_pid.map(pid_counts_train).fillna(0.0).astype(np.float64).to_numpy()
)

pid_counts_test = test_pid.value_counts(dropna=False)
pid_count_te_self = test_pid.map(pid_counts_test).astype(np.float64).to_numpy()

pid_count_log_tr = np.log1p(pid_count_tr)
pid_count_log_te_from_train = np.log1p(pid_count_te_from_train)
pid_count_log_te_self = np.log1p(pid_count_te_self)

train_site = train["anatom_site_general_challenge"].fillna("__MISSING__").astype(str)
test_site = test["anatom_site_general_challenge"].fillna("__MISSING__").astype(str)
group_key_tr = train_pid.astype(str) + "||" + train_site
group_key_te = test_pid.astype(str) + "||" + test_site

pid_site_oof_mean_tr, pid_site_mean_te, pid_site_oof_logit_tr, pid_site_logit_te = (
    _oof_groupkey_target_stats(
        train_df=train,
        test_df=test,
        group_key_tr=group_key_tr,
        group_key_te=group_key_te,
        patient_groups=train_pid,
        target_col="target",
        m_mean=200.0,
        alpha_logit=1.0,
        n_splits=5,
        seed=RANDOM_STATE,
    )
)

train_sex = train["sex"].fillna("__MISSING__").astype(str)
test_sex = test["sex"].fillna("__MISSING__").astype(str)
group_key_pid_sex_tr = train_pid.astype(str) + "||" + train_sex
group_key_pid_sex_te = test_pid.astype(str) + "||" + test_sex

pid_sex_oof_mean_tr, pid_sex_mean_te, pid_sex_oof_logit_tr, pid_sex_logit_te = (
    _oof_groupkey_target_stats(
        train_df=train,
        test_df=test,
        group_key_tr=group_key_pid_sex_tr,
        group_key_te=group_key_pid_sex_te,
        patient_groups=train_pid,
        target_col="target",
        m_mean=200.0,
        alpha_logit=1.0,
        n_splits=5,
        seed=RANDOM_STATE,
    )
)

group_key_pid_agebin_tr = (
    train_pid.astype(str)
    + "||"
    + X_train_base["age_bin"].fillna("__MISSING__").astype(str)
)
group_key_pid_agebin_te = (
    test_pid.astype(str)
    + "||"
    + X_test_base["age_bin"].fillna("__MISSING__").astype(str)
)

(
    pid_agebin_oof_mean_tr,
    pid_agebin_mean_te,
    pid_agebin_oof_logit_tr,
    pid_agebin_logit_te,
) = _oof_groupkey_target_stats(
    train_df=train,
    test_df=test,
    group_key_tr=group_key_pid_agebin_tr,
    group_key_te=group_key_pid_agebin_te,
    patient_groups=train_pid,
    target_col="target",
    m_mean=200.0,
    alpha_logit=1.0,
    n_splits=5,
    seed=RANDOM_STATE,
)

X_train = X_train_base.copy()
X_test = X_test_base.copy()

X_train["patient_id_te"] = pid_oof_mean_tr
X_test["patient_id_te"] = pid_mean_te

X_train["patient_id_count"] = pid_count_tr
X_test["patient_id_count"] = pid_count_te_from_train
X_train["patient_id_count_log1p"] = pid_count_log_tr
X_test["patient_id_count_log1p"] = pid_count_log_te_from_train

X_train["patient_id_logit"] = pid_oof_logit_tr
X_test["patient_id_logit"] = pid_logit_te

X_train["patient_id_test_count"] = 0.0
X_test["patient_id_test_count"] = pid_count_te_self
X_train["patient_id_test_count_log1p"] = 0.0
X_test["patient_id_test_count_log1p"] = pid_count_log_te_self

X_train["pid_site_te"] = pid_site_oof_mean_tr
X_test["pid_site_te"] = pid_site_mean_te
X_train["pid_site_logit"] = pid_site_oof_logit_tr
X_test["pid_site_logit"] = pid_site_logit_te

X_train["pid_sex_te"] = pid_sex_oof_mean_tr
X_test["pid_sex_te"] = pid_sex_mean_te
X_train["pid_sex_logit"] = pid_sex_oof_logit_tr
X_test["pid_sex_logit"] = pid_sex_logit_te

X_train["pid_agebin_te"] = pid_agebin_oof_mean_tr
X_test["pid_agebin_te"] = pid_agebin_mean_te
X_train["pid_agebin_logit"] = pid_agebin_oof_logit_tr
X_test["pid_agebin_logit"] = pid_agebin_logit_te

numeric_features = [
    "age_approx",
    "patient_id_te",
    "patient_id_count",
    "patient_id_count_log1p",
    "patient_id_logit",
    "patient_id_test_count",
    "patient_id_test_count_log1p",
    "pid_site_te",
    "pid_site_logit",
    "pid_sex_te",
    "pid_sex_logit",
    "pid_agebin_te",
    "pid_agebin_logit",
]
categorical_features = ["sex", "anatom_site_general_challenge", "age_bin"]

preprocess = ColumnTransformer(
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
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    class_weight=None,
    random_state=RANDOM_STATE,
    C=5.0,
)

model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
model.fit(X_train, y_train)

test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)
test_pred = np.nan_to_num(test_pred, nan=np.mean(y_train), posinf=1.0, neginf=0.0)
test_pred = np.clip(test_pred, 0.0, 1.0)

test_pred[:10], test_pred.min(), test_pred.max()



## === cell 2
out = test[["image_name"]].copy()
out["target"] = test_pred

if out.shape[0] != test.shape[0]:
    raise ValueError(
        f"Submission row count mismatch: out={out.shape[0]} vs test={test.shape[0]}"
    )
if out["image_name"].isna().any():
    raise ValueError("Found NaN image_name in submission.")
if out["target"].isna().any():
    raise ValueError("Found NaN target in submission.")

out.to_csv("submission.csv", index=False)
out.head()
