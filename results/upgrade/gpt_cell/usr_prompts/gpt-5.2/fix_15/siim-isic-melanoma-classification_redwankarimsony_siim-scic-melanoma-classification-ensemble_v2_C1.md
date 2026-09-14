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

0.9191

# 6. Current score

0.79046

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes because it tries to read three external submission CSVs from `../input/public-submission-melanoma-95/`, a directory that does not exist in this environment. This causes a `FileNotFoundError` before `sub` and the three `public_sub_*` DataFrames can be created, which also blocks cell 2 that depends on them.

Patch summary: Modify only cell 1 to (a) load the provided `sample_submission.csv` from the available dataset path, and (b) gracefully fall back to using that same sample submission as a stand-in for the missing external submissions (with deterministic `target` defaults) so that downstream blending in cell 2 can run without changing its logic.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: The patch preserves the variables `public_sub_mean_9533`, `public_sub_median_9533`, `public_sub_meta_ens_9577`, and `sub` as pandas DataFrames with a `target` column, so cell 2’s weighted assignment remains valid and executes without interface changes.

Assumptions: The competition-provided `sample_submission.csv` exists at one of the listed paths, and it contains `image_name` and `target` columns; when external submissions are unavailable, using zeros for their `target` values is an acceptable deterministic fallback to avoid crashing.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score happens because all three “public submission” inputs are missing, so the fallback makes every prediction exactly 0.0; this yields AUC=0.5 (random/constant). To move toward the 0.9191 target with minimal change and without changing the core “blend three submissions” logic, I keep the same variables/DataFrame interfaces but replace the fallback targets with a deterministic, reasonable prior derived from `train.csv`’s malignancy rate (overall mean of `target`). This produces non-constant probabilities (still simple and stable) and should improve AUC above 0.5 while remaining minimal and metric-consistent. I also ensure alignment by `image_name` when constructing the fallback so the blended result matches the sample submission order exactly.'
- What this solution (achieved 0.66725) has done: 'Your current 0.5 AUC comes from constant predictions because the three external “public submissions” aren’t available and your fallback uses a single prior for every test row. To move toward the 0.9191 target with minimal changes while preserving the exact “blend three submissions” core logic, I make the fallback produce *non-constant* probabilities using only the provided metadata (`sex`, `age_approx`, `anatom_site_general_challenge`) by fitting a simple, deterministic logistic regression on `train.csv` and predicting on `test.csv`. I keep the same DataFrame interfaces (`public_sub_*` with `image_name,target`) so cell 2 stays unchanged, and I align by `image_name` to avoid any row-order mismatches that would hurt AUC. This should substantially improve over 0.5 while remaining lightweight and within the constraints.'
- What this solution (achieved 0.65058) has done: 'Your current 0.66725 AUC is limited because all three blended inputs are identical fallbacks, so the ensemble provides no diversity and the final prediction is effectively just one metadata model. To move your score upward toward 0.9191 with minimal changes and the same core “blend three submissions” logic, I keep the exact blending in cell 2 but make each fallback slightly different by varying only LogisticRegression hyperparameters/feature variants in a deterministic way (still metadata-only, same training approach). This preserves the same DataFrame interfaces and evaluation semantics while creating a real ensemble from three related but non-identical metadata models, which should improve AUC without large code changes. I also keep strict `image_name` alignment and deterministic settings to avoid accidental score drops from misalignment.'
- What this solution (achieved 0.77387) has done: 'Your current gap to the target is large (0.65058 vs 0.9191), and the main limitation is that your “three-public-submissions blend” is effectively just three similar metadata-only models. To move the AUC upward without changing the blending core logic, I keep the same interfaces and LogisticRegression approach but add a few strong, competition-relevant tabular features that are already in `train.csv`/`test.csv` (notably `patient_id`, plus simple engineered age terms) to increase signal. I also make the three fallback variants genuinely different via feature subsets/hyperparameters while staying within the same training approach and ensuring strict `image_name` alignment. This should improve AUC materially while remaining lightweight and deterministic.'
- What this solution (achieved 0.73591) has done: 'Your current score (0.77387) is still well below the target (0.9191), so we should improve AUC with the smallest, safe changes while preserving the same “blend three submissions” core logic. The simplest high-impact upgrade is to make the metadata fallback models stronger without changing the modeling family: keep LogisticRegression, but use out-of-fold target encoding for `patient_id` (a very strong signal here) plus a couple of lightweight frequency/interaction features, all computed strictly within train data (no leakage). To preserve the ensemble diversity, each of the three fallbacks use the same pipeline style but slightly different feature sets/hyperparameters as before. We also keep strict `image_name` alignment and still write `submission.csv` with the required columns.'
- What this solution (achieved 0.79338) has done: 'Your current gap to the target is still large (0.73591 vs 0.9191), so we should cautiously increase AUC with minimal, low-risk changes while preserving the same overall “blend three submissions” logic and LogisticRegression metadata fallback. The biggest safe win is to fix a likely bug in `patient_count` for the test set (it’s currently computed from test-only counts, which is inconsistent with train) and to strengthen the very strong `patient_id` signal using leakage-safe out-of-fold *logit* target encoding with smoothing (still computed strictly within train folds). I keep the same three fallback variants and the same weighted blend in cell 2, but each variant benefits from the improved encoding and consistent count feature. These are small, deterministic changes that should move your score upward toward the target without altering the core architecture or submission semantics.'
- What this solution (achieved 0.79538) has done: 'Your current AUC (0.79338) is still well below the 0.9191 target, so we should increase score with the smallest, low-risk change while keeping the same “blend three submissions” core logic and LogisticRegression metadata fallback. The strongest remaining signal we can safely add without changing the model family is leakage-safe target encoding for `anatom_site_general_challenge` (computed out-of-fold on train, then applied to test with smoothing) and a consistent train-based frequency feature for that site; these typically help a lot in this competition’s tabular baseline. I add these features to all three fallback variants while leaving the rest of the pipeline, blending weights, and submission writing unchanged. This should move AUC upward toward the target with minimal additional complexity and within the runtime budget.'
- What this solution (achieved 0.79265) has done: 'Your current AUC (0.79538) is still far below the 0.9191 target, so we should make a small, low-risk improvement that strengthens the existing metadata+logistic-regression fallback without changing the ensemble/blending logic. The biggest remaining tabular signal in this competition is per-patient “how many lesions” and per-site frequency; you already add these, but the test-side defaults are currently inconsistent (unknown patients/sites default to 1, which is too arbitrary). I change only the fallback feature construction so that unseen `patient_id`/site in test get a stable, train-derived default frequency (median train frequency), and I add a simple log-count transform (monotonic, helps linear models) while keeping the same LogisticRegression family and the same downstream blend and submission format. These are minimal, deterministic feature tweaks that typically lift AUC without altering the core approach or runtime meaningfully.'
- What this solution (achieved 0.79279) has done: 'Your current AUC (0.79265) is well below the 0.9191 target, so we should make a small, low-risk improvement that keeps the same overall approach: three LogisticRegression metadata models blended with fixed weights. The minimal change with the best chance to lift AUC is to add one leakage-safe, competition-relevant feature: a smoothed out-of-fold target encoding for `sex`, computed exactly like your existing `patient_id`/`site` encodings (same semantics, just another categorical signal). I keep all existing features, variants, and blending logic unchanged, and only append `sex_te` as an additional numeric feature into each variant’s feature set. This is deterministic, uses only `train.csv`/`test.csv`, and should nudge score upward toward the target without changing the core logic.'
- What this solution (achieved 0.33864) has done: 'Your current AUC (0.79279) is far below the 0.9191 target, so we should make a small, low-risk improvement that keeps the exact same overall approach (three LogisticRegression metadata models + fixed weighted blend). The biggest safe boost without changing model family is to add one more leakage-safe categorical target-encoding feature: a smoothed out-of-fold logit target encoding for `diagnosis` (train-only column), then apply it to test using only fold-trained statistics. This preserves your pipeline and blending logic while adding a known-strong metadata signal from the provided CSV. I also keep strict `image_name` alignment and stable defaults for missing/unseen categories so the submission remains valid and deterministic.'
- What this solution (achieved 0.79252) has done: 'Your AUC collapsed to ~0.34 because the new `diagnosis_te` feature is effectively broken for test: `diagnosis` is set to NaN for all test rows, so the encoding becomes a constant and can distort the logistic model (especially with class_weight balanced), harming ranking. To move back upward toward the 0.9191 target with minimal change and preserved core logic (same logistic-regression + target-encoding approach + same 3-way blend), I switch `diagnosis_te` to a robust “text-present” surrogate available in both train and test: use `anatom_site_general_challenge` as the encoded source but keep the output column named `diagnosis_te` so the rest of the pipeline remains unchanged. I also ensure the encoding input columns are aligned and deterministic, and keep strict `image_name` alignment when writing predictions.'
- What this solution (achieved 0.79413) has done: 'We need to increase AUC from 0.79252 toward 0.9191 (higher is better), so we should strengthen ranking while keeping your core logic: three LogisticRegression metadata models (fallbacks) and the same fixed weighted blend in cell 2. The smallest high-impact fix is to make the patient-based signal leakage-safe and better aligned with the competition by computing *group-aware (patient_id) out-of-fold* target encodings (and counts) using GroupKFold, preventing the same patient from appearing in both train/val folds for the encoding. This preserves the same feature set and model family, but reduces optimistic bias and typically improves generalization/ranking on this dataset. I also ensure deterministic, train-derived defaults for unseen patients/sites and keep strict `image_name` alignment so submission order/rows remain correct.'
- What this solution (achieved 0.79046) has done: 'Your current AUC (0.79413) is far below the 0.9191 target, so we should make a small, low-risk improvement that strengthens ranking while keeping the same overall core logic (three LogisticRegression metadata fallbacks and the same fixed weighted blend). The biggest issue is that some target encodings (site/sex/“diagnosis_te”) are computed with StratifiedKFold, which can leak patient information across folds and hurt true generalization on this patient-grouped dataset. I switch those encodings to patient-group-aware out-of-fold computation using GroupKFold (same encoding formula, just safer splits), keeping all features, models, and blend weights intact. This should move AUC upward toward the target without changing architecture or output semantics, and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
import os

base_candidates = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "../input/siim-isic-melanoma-classification",
    "../data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
    "../data",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_paths = [os.path.join(b, "sample_submission.csv") for b in base_candidates]
sample_path = _first_existing(sample_paths)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {sample_paths}"
    )

sub = pd.read_csv(sample_path)

if "target" not in sub.columns:
    sub["target"] = 0.0
else:
    sub["target"] = (
        pd.to_numeric(sub["target"], errors="coerce").fillna(0.0).astype(float)
    )

train_paths = [os.path.join(b, "train.csv") for b in base_candidates]
train_path = _first_existing(train_paths)
if train_path is None:
    raise FileNotFoundError(f"Could not find train.csv in any of: {train_paths}")

test_paths = [os.path.join(b, "test.csv") for b in base_candidates]
test_path = _first_existing(test_paths)
if test_path is None:
    raise FileNotFoundError(f"Could not find test.csv in any of: {test_paths}")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _safe_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _add_oof_patient_target_encoding(
    train_df, test_df, y_col="target", n_splits=5, seed=0, smoothing=20.0
):
    tr = train_df.copy()
    te = test_df.copy()

    global_mean = float(tr[y_col].mean())
    global_mean = float(np.clip(global_mean, 1e-6, 1 - 1e-6))
    global_logit = float(_safe_logit(global_mean))

    tr["patient_te"] = np.nan

    groups = tr["patient_id"].astype(str).fillna("__MISSING__").values
    y = tr[y_col].astype(int).values

    order = np.argsort(groups, kind="mergesort")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))

    tr_ord = tr.iloc[order].reset_index(drop=True)
    y_ord = y[order]
    groups_ord = groups[order]

    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(tr_ord, y_ord, groups=groups_ord):
        fold_tr = tr_ord.iloc[tr_idx]
        grp = fold_tr.groupby("patient_id")[y_col].agg(["mean", "count"])
        smoothed_p = (grp["mean"] * grp["count"] + global_mean * smoothing) / (
            grp["count"] + smoothing
        )
        stats_logit = smoothed_p.map(_safe_logit)

        tr_ord.iloc[val_idx, tr_ord.columns.get_loc("patient_te")] = tr_ord.iloc[
            val_idx
        ]["patient_id"].map(stats_logit)

    tr_ord["patient_te"] = tr_ord["patient_te"].astype(float).fillna(global_logit)

    tr["patient_te"] = tr_ord["patient_te"].iloc[inv_order].values

    grp_full = tr.groupby("patient_id")[y_col].agg(["mean", "count"])
    smoothed_p_full = (
        grp_full["mean"] * grp_full["count"] + global_mean * smoothing
    ) / (grp_full["count"] + smoothing)
    stats_logit_full = smoothed_p_full.map(_safe_logit)
    te["patient_te"] = (
        te["patient_id"].map(stats_logit_full).astype(float).fillna(global_logit)
    )

    tr["patient_count"] = (
        tr.groupby("patient_id")["patient_id"].transform("size").astype(np.int16)
    )
    train_counts = tr["patient_id"].value_counts()
    default_patient_count = int(np.median(train_counts.values))
    te["patient_count"] = (
        te["patient_id"]
        .map(train_counts)
        .fillna(float(default_patient_count))
        .astype(np.int16)
    )

    tr["patient_count_log1p"] = np.log1p(tr["patient_count"].astype(float)).astype(
        float
    )
    te["patient_count_log1p"] = np.log1p(te["patient_count"].astype(float)).astype(
        float
    )

    return tr, te


def _add_oof_site_target_encoding(
    train_df,
    test_df,
    site_col="anatom_site_general_challenge",
    y_col="target",
    n_splits=5,
    seed=0,
    smoothing=50.0,
):
    tr = train_df.copy()
    te = test_df.copy()

    tr_site = tr[site_col].astype("object").fillna("__MISSING__")
    te_site = te[site_col].astype("object").fillna("__MISSING__")

    global_mean = float(tr[y_col].mean())
    global_mean = float(np.clip(global_mean, 1e-6, 1 - 1e-6))
    global_logit = float(_safe_logit(global_mean))

    tr["site_te"] = np.nan
    y = tr[y_col].astype(int).values
    groups = tr["patient_id"].astype(str).fillna("__MISSING__").values

    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(tr, y, groups=groups):
        fold_tr = tr.iloc[tr_idx]
        fold_site = fold_tr[site_col].astype("object").fillna("__MISSING__")
        grp = (
            fold_tr.assign(_site=fold_site)
            .groupby("_site")[y_col]
            .agg(["mean", "count"])
        )
        smoothed_p = (grp["mean"] * grp["count"] + global_mean * smoothing) / (
            grp["count"] + smoothing
        )
        stats_logit = smoothed_p.map(_safe_logit)

        val_site = tr_site.iloc[val_idx]
        tr.iloc[val_idx, tr.columns.get_loc("site_te")] = val_site.map(
            stats_logit
        ).values

    tr["site_te"] = tr["site_te"].astype(float).fillna(global_logit)

    grp_full = tr.assign(_site=tr_site).groupby("_site")[y_col].agg(["mean", "count"])
    smoothed_p_full = (
        grp_full["mean"] * grp_full["count"] + global_mean * smoothing
    ) / (grp_full["count"] + smoothing)
    stats_logit_full = smoothed_p_full.map(_safe_logit)
    te["site_te"] = te_site.map(stats_logit_full).astype(float).fillna(global_logit)

    site_counts = tr_site.value_counts()
    default_site_count = int(np.median(site_counts.values))
    tr["site_count"] = tr_site.map(site_counts).astype(np.int32)
    te["site_count"] = (
        te_site.map(site_counts).fillna(float(default_site_count)).astype(np.int32)
    )

    tr["site_count_log1p"] = np.log1p(tr["site_count"].astype(float)).astype(float)
    te["site_count_log1p"] = np.log1p(te["site_count"].astype(float)).astype(float)

    return tr, te


def _add_oof_cat_target_encoding_logit(
    train_df,
    test_df,
    cat_col,
    out_col,
    y_col="target",
    n_splits=5,
    seed=0,
    smoothing=50.0,
    missing_token="__MISSING__",
    group_col=None,
):
    tr = train_df.copy()
    te = test_df.copy()

    tr_cat = tr[cat_col].astype("object").fillna(missing_token)
    te_cat = te[cat_col].astype("object").fillna(missing_token)

    global_mean = float(tr[y_col].mean())
    global_mean = float(np.clip(global_mean, 1e-6, 1 - 1e-6))
    global_logit = float(_safe_logit(global_mean))

    tr[out_col] = np.nan
    y = tr[y_col].astype(int).values

    if group_col is not None and group_col in tr.columns:
        groups = tr[group_col].astype(str).fillna(missing_token).values
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(tr, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(tr, y)

    for tr_idx, val_idx in split_iter:
        fold_tr = tr.iloc[tr_idx]
        fold_cat = fold_tr[cat_col].astype("object").fillna(missing_token)

        grp = (
            fold_tr.assign(_cat=fold_cat).groupby("_cat")[y_col].agg(["mean", "count"])
        )
        smoothed_p = (grp["mean"] * grp["count"] + global_mean * smoothing) / (
            grp["count"] + smoothing
        )
        stats_logit = smoothed_p.map(_safe_logit)

        val_cat = tr_cat.iloc[val_idx]
        tr.iloc[val_idx, tr.columns.get_loc(out_col)] = val_cat.map(stats_logit).values

    tr[out_col] = tr[out_col].astype(float).fillna(global_logit)

    grp_full = tr.assign(_cat=tr_cat).groupby("_cat")[y_col].agg(["mean", "count"])
    smoothed_p_full = (
        grp_full["mean"] * grp_full["count"] + global_mean * smoothing
    ) / (grp_full["count"] + smoothing)
    stats_logit_full = smoothed_p_full.map(_safe_logit)
    te[out_col] = te_cat.map(stats_logit_full).astype(float).fillna(global_logit)

    return tr, te


def _fit_predict_metadata(train_csv, test_csv, sub_df, variant="base"):
    use_cols = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "diagnosis",
        "target",
    ]
    train_df = pd.read_csv(train_csv, usecols=use_cols)

    test_df = pd.read_csv(
        test_csv,
        usecols=[
            "image_name",
            "patient_id",
            "sex",
            "age_approx",
            "anatom_site_general_challenge",
        ],
    )
    test_df["diagnosis"] = np.nan

    train_df["age_approx"] = pd.to_numeric(train_df["age_approx"], errors="coerce")
    test_df["age_approx"] = pd.to_numeric(test_df["age_approx"], errors="coerce")

    for df in (train_df, test_df):
        df["age_missing"] = df["age_approx"].isna().astype(np.int8)
        df["age2"] = (df["age_approx"].fillna(df["age_approx"].median()) ** 2).astype(
            float
        )

    train_df, test_df = _add_oof_patient_target_encoding(
        train_df, test_df, y_col="target", n_splits=5, seed=0, smoothing=20.0
    )

    train_df, test_df = _add_oof_site_target_encoding(
        train_df,
        test_df,
        site_col="anatom_site_general_challenge",
        y_col="target",
        n_splits=5,
        seed=0,
        smoothing=50.0,
    )

    train_df, test_df = _add_oof_cat_target_encoding_logit(
        train_df,
        test_df,
        cat_col="sex",
        out_col="sex_te",
        y_col="target",
        n_splits=5,
        seed=0,
        smoothing=100.0,
        missing_token="__MISSING__",
        group_col="patient_id",
    )

    train_df, test_df = _add_oof_cat_target_encoding_logit(
        train_df,
        test_df,
        cat_col="anatom_site_general_challenge",
        out_col="diagnosis_te",
        y_col="target",
        n_splits=5,
        seed=0,
        smoothing=200.0,
        missing_token="__MISSING__",
        group_col="patient_id",
    )

    for df in (train_df, test_df):
        df["age_x_patient_te"] = df["age_approx"].fillna(
            train_df["age_approx"].median()
        ).astype(float) * df["patient_te"].astype(float)

    if variant == "base":
        feat_cols = [
            "patient_te",
            "patient_count",
            "patient_count_log1p",
            "site_te",
            "site_count",
            "site_count_log1p",
            "sex_te",
            "diagnosis_te",
            "sex",
            "age_approx",
            "age2",
            "age_missing",
            "age_x_patient_te",
            "anatom_site_general_challenge",
        ]
        C = 0.7
        class_weight = "balanced"
    elif variant == "strong_reg":
        feat_cols = [
            "patient_te",
            "patient_count",
            "patient_count_log1p",
            "site_te",
            "site_count",
            "site_count_log1p",
            "sex_te",
            "diagnosis_te",
            "sex",
            "age_approx",
            "age_missing",
            "anatom_site_general_challenge",
        ]
        C = 0.25
        class_weight = "balanced"
    elif variant == "no_age":
        feat_cols = [
            "patient_te",
            "patient_count",
            "patient_count_log1p",
            "site_te",
            "site_count",
            "site_count_log1p",
            "sex_te",
            "diagnosis_te",
            "sex",
            "anatom_site_general_challenge",
        ]
        C = 0.9
        class_weight = "balanced"
    else:
        feat_cols = [
            "patient_te",
            "patient_count",
            "patient_count_log1p",
            "site_te",
            "site_count",
            "site_count_log1p",
            "sex_te",
            "diagnosis_te",
            "sex",
            "age_approx",
            "age2",
            "age_missing",
            "age_x_patient_te",
            "anatom_site_general_challenge",
        ]
        C = 0.7
        class_weight = "balanced"

    X_train = train_df[feat_cols]
    y_train = train_df["target"].astype(int)
    X_test = test_df[feat_cols]

    numeric_features = [
        c
        for c in feat_cols
        if c
        in (
            "age_approx",
            "age2",
            "age_missing",
            "patient_te",
            "patient_count",
            "patient_count_log1p",
            "age_x_patient_te",
            "site_te",
            "site_count",
            "site_count_log1p",
            "sex_te",
            "diagnosis_te",
        )
    ]
    categorical_features = [c for c in feat_cols if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        C=C,
        class_weight=class_weight,
        random_state=0,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict_proba(X_test)[:, 1].astype(float)
    pred = np.clip(pred, 1e-6, 1 - 1e-6)

    pred_df = pd.DataFrame({"image_name": test_df["image_name"].values, "target": pred})

    pred_map = pred_df.set_index("image_name")["target"]
    out = sub_df[["image_name"]].copy()
    out["target"] = out["image_name"].map(pred_map).astype(float)

    prior = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    out["target"] = out["target"].fillna(prior).clip(0.0, 1.0)
    return out


ext_base = "../input/public-submission-melanoma-95"
mean_path = os.path.join(ext_base, "submission_mean.csv")
median_path = os.path.join(ext_base, "submission_median.csv")
meta_path = os.path.join(ext_base, "external_meta_ensembled.csv")


def _load_or_fallback(path, fallback_df, train_csv, test_csv, variant):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            return _fit_predict_metadata(
                train_csv, test_csv, fallback_df, variant=variant
            )
        df["target"] = (
            pd.to_numeric(df["target"], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .clip(0.0, 1.0)
        )
        df = fallback_df[["image_name"]].merge(
            df[["image_name", "target"]], on="image_name", how="left"
        )
        if df["target"].isna().any():
            meta_pred = _fit_predict_metadata(
                train_csv, test_csv, fallback_df, variant=variant
            ).set_index("image_name")["target"]
            df["target"] = df["target"].fillna(df["image_name"].map(meta_pred))
        df["target"] = df["target"].astype(float).clip(0.0, 1.0)
        return df

    return _fit_predict_metadata(train_csv, test_csv, fallback_df, variant=variant)


public_sub_mean_9533 = _load_or_fallback(
    mean_path, sub, train_path, test_path, variant="strong_reg"
)
public_sub_median_9533 = _load_or_fallback(
    median_path, sub, train_path, test_path, variant="base"
)
public_sub_meta_ens_9577 = _load_or_fallback(
    meta_path, sub, train_path, test_path, variant="no_age"
)



## === cell 2
sub.target = (
    public_sub_meta_ens_9577.target * 0.40
    + public_sub_median_9533.target * 0.30
    + public_sub_mean_9533.target * 0.30
)



## === cell 3
sub.head()
sub.to_csv("submission.csv", index=False)
