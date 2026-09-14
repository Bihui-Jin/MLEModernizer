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

0.8857038841136874

# 6. Current score

0.71928

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the internet `wget` dependency (Kaggle notebooks typically have no outbound network), and instead build the ensemble from locally-available files only. Because the two referenced `../input/...` submissions are not present in your provided filesystem, I fall back to the provided `sample_submission.csv` (valid format) and fill it with a safe constant probability so a valid `.csv` is always produced. I also add robust path checking and guaranteed column validation (`image_name`, `target`) to prevent the “Invalid submission” error. These changes are minimal and focused on producing a correct submission end-to-end in this environment.'
- What this solution (achieved 0.66776) has done: 'You’re currently scoring 0.5 because you submit a constant prediction, which yields near-random ranking and AUC≈0.5. To move toward the target AUC (~0.886) with minimal change and without adding any new modeling/training code, we replace the constant with a simple metadata-only probabilistic model trained on `train.csv` and applied to `test.csv`. This preserves the overall pipeline structure (read CSVs → build `sub` → write `submission.csv`) while producing non-trivial ranking. The model is a regularized logistic regression with one-hot encoding for categorical fields and imputation for missing values, which is fast and stable under the 600s constraint.'
- What this solution (achieved 0.64477) has done: 'Your current metadata-only logistic regression is valid but underpowered (AUC 0.6678 vs target 0.8857), so we keep the exact same pipeline/model family and only make small, metric-aligned tweaks that typically yield a noticeable lift without changing the “core logic.” Specifically: (1) add a few strong, still-metadata-only features (age missing indicator, age bins, and patient-level mean target encoding computed on train only) and (2) use a slightly better-calibrated solver/regularization setting for sparse one-hot features (`liblinear`, a bit less regularization) while keeping logistic regression and the same fit/predict flow. These changes should improve ranking (AUC) while staying well within Kaggle constraints and runtime, and the script still guarantees a valid `submission.csv` with correct columns/row alignment. If this overshoots the target band, the single knob to pull back is `C` (lower C = more regularization = slightly lower AUC).'
- What this solution (achieved 0.65775) has done: 'We keep your metadata-only logistic-regression pipeline intact, but fix the biggest score limiter: the `patient_target_mean` feature currently leaks the label for the same patient within training, which makes the model overfit and generalize poorly to the test set (hurting AUC). To preserve core logic while improving ranking, we replace that leaky patient mean with an out-of-fold (OOF) patient mean computed via a GroupKFold on `patient_id`, so each training row’s patient statistic is computed without using its own fold labels. We also add a very small, safe smoothing toward the global mean for patients with few images (still metadata-only), which typically improves generalization without changing the model family or training approach. Output writing/validation stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 0.74886) has done: 'Your current pipeline is already valid and non-leaky, but it’s likely underfitting because `patient_id` is not actually used as a feature (only indirectly via the precomputed patient mean), and the categorical space is missing a strong interaction signal between sex and anatomy site. To move AUC upward toward the 0.8857 target with minimal change, I (1) include `patient_id` itself as a one-hot categorical feature (safe because test contains many seen patients), and (2) add a single crossed categorical feature `sex_x_site` to capture a common interaction without changing the model family or training flow. I keep the same LogisticRegression setup and the same OOF patient-mean construction, only extending the feature set. The submission writing/validation remains unchanged.'
- What this solution (achieved 0.74888) has done: 'To move your AUC upward toward the 0.8857 target with minimal disruption, I keep the same metadata+LogisticRegression pipeline and only strengthen the weakest part: the patient-level target-encoding feature. Right now you compute an OOF patient mean, but then collapse it to a single value per patient, which blurs per-image OOF values and reduces signal; we instead compute a proper OOF patient mean per row (with count-based smoothing) and use that directly. I also add one more high-signal, still-metadata-only feature: an OOF target mean for `anatom_site_general_challenge` (also smoothed), computed with the same GroupKFold to avoid leakage. These are small, metric-aligned changes that usually improve ranking without changing the model family, training flow, or submission semantics.'
- What this solution (achieved 0.74885) has done: 'We keep the same metadata + LogisticRegression pipeline, but fix a subtle but important issue: your OOF target-encoding for `anatom_site_general_challenge` is currently computed with `GroupKFold` by `patient_id`, which means the same site appears in both train/val folds and the encoding becomes noisy/less stable than necessary. We instead compute the site OOF encoding with standard `KFold` (shuffled, fixed seed) while keeping the patient OOF encoding grouped (to avoid leakage across the same patient). Additionally, we slightly increase the smoothing strength for the site encoding only (more stable ranks on test) while leaving the rest unchanged, aiming to lift AUC toward your 0.8857 target with minimal risk and runtime impact. Submission writing and schema validation remain identical.'
- What this solution (achieved 0.74886) has done: 'You’re substantially below the target AUC (0.74885 vs 0.88570; higher-is-better), so we should improve ranking while keeping your LogisticRegression + metadata pipeline intact. The smallest high-impact change is to fix the fold strategy used for the OOF site target-encoding: we should use the same `GroupKFold(patient_id)` as training, so each fold’s site encoding is computed without seeing any labels from the validation patients (reduces cross-patient leakage via the encoding). We keep the exact same feature set and model, but recompute `oof_site_mean` with GroupKFold and also build the **test** site encoding from patient-level aggregated targets (i.e., per-site mean of per-patient means) to match the group-based semantics and improve generalization. Everything else (paths, preprocessing, solver, submission writing/validation) stays the same.'
- What this solution (achieved 0.74831) has done: 'Your current score (0.74886) is well below the target AUC (0.88570), so we should improve ranking while keeping the same metadata + LogisticRegression pipeline. The biggest likely limiter now is that `patient_id` one-hot can overfit and dominate (especially with `class_weight="balanced"`), hurting generalization; we keep `patient_id` available but downweight its impact by strengthening regularization slightly and adding a tiny, deterministic amount of noise to the patient-level OOF encodings to reduce tie/overconfidence artifacts (this preserves evaluation semantics and doesn’t change the model family). We also ensure the OOF target encodings use a fixed `random_state`-like determinism by explicitly sorting the training dataframe before fold generation (stable grouping order), which can modestly improve reproducibility and sometimes AUC. All changes are minimal, keep the same training flow, and still write a valid `submission.csv`.'
- What this solution (achieved 0.74835) has done: 'We’re well below the target AUC (0.74831 vs 0.88570; higher-is-better), so we should improve ranking with minimal disruption to your existing metadata + LogisticRegression pipeline. The biggest low-risk lift here is to add one more standard, high-signal metadata feature: a non-leaky out-of-fold (OOF) target mean for `sex_x_site` (sex-by-anatomy interaction), computed with the same GroupKFold-by-patient strategy used for other OOF encodings. We then apply the corresponding smoothed mapping to the test set from full training data, just like your patient/site encodings, and include it as an additional numeric feature. This preserves the same model family, same fit/predict flow, same loss/metric semantics, and keeps runtime small while typically yielding a meaningful AUC increase.'
- What this solution (achieved 0.7485) has done: 'Your current score (0.74835) is far below the target AUC (0.88570), so we should increase ranking quality while keeping your metadata + LogisticRegression pipeline intact. The smallest high-impact fix is to correct a train/test mismatch: `sex_x_site` is created using filled `unknown` values, but then your full-data `sexsite_mean_map_full` is computed from the raw `sex`/`site` columns without filling, so many test `sex_x_site` keys won’t match and that strong numeric feature degrades. I recompute the full `sex_x_site` mapping using the exact same fill logic as feature creation, and (to keep semantics consistent) also compute the full `sex_x_site` statistics at the patient level (mean per patient within `sex_x_site`) like you already do for the site encoding. Everything else (OOF encodings, model, preprocessing, submission writing) remains the same.'
- What this solution (achieved 0.72168) has done: 'We’re well below the target AUC (0.7485 vs 0.8857; higher-is-better), so we should improve ranking while keeping your exact metadata+LogisticRegression pipeline intact. The smallest likely lift is to stop forcing `class_weight="balanced"` (AUC is rank-based, and balancing can distort probability ordering when combined with strong patient_id one-hot and target encodings), and to slightly tune regularization upward to reduce overfitting on high-cardinality `patient_id`. I also make the smoothing consistent by using the same `m_smooth` for the test mappings as used for each OOF encoding (patient=5, site=20, sexsite=20), which avoids train/test feature distribution mismatch without changing the approach. Everything else (features, OOF construction, preprocessing, model family, and submission writing) remains the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.71806) has done: 'Your current AUC (0.72168) is far below the target (0.88570), so we should increase ranking quality while keeping the exact same metadata + LogisticRegression pipeline. The most likely limiter now is that the very high-cardinality `patient_id` one-hot is still dominating/overfitting; we keep using it (core logic unchanged) but reduce its impact by (a) lowering `C` slightly (stronger regularization) and (b) adding rare-category pruning via `OneHotEncoder(min_frequency=...)` to prevent thousands of near-singleton patient features from fitting noise. Additionally, we apply the *same* deterministic jitter to the corresponding numeric encodings on test (tiny, rank-preserving tie-breaker) to better match train/test behavior without changing semantics. These are minimal, safe changes that often lift generalization AUC without altering the overall approach or output format.'
- What this solution (achieved 0.71928) has done: 'We’re far below the target AUC (0.718 vs 0.886; higher-is-better), so we should improve ranking while keeping your exact metadata + LogisticRegression pipeline intact. The smallest likely lift is to stop injecting random jitter into the OOF encodings and test encodings: jitter is not guaranteed to help AUC and can easily harm ordering; removing it is deterministic and preserves semantics. Second, we slightly relax the rare-category pruning on `OneHotEncoder(min_frequency=10 -> 3)` so more patient/site categories are represented, which often improves ranking for this competition without changing the model family or training flow. Everything else (OOF target encodings, features, model, and submission writing/validation) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/data",
    "/kaggle/input",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input/siim-isic-melanoma-classification",
]


def first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = first_existing_path(DATA_DIR_CANDIDATES)
if BASE_DIR is None:
    raise FileNotFoundError(
        f"Could not find dataset directory from candidates: {DATA_DIR_CANDIDATES}"
    )


def read_first_existing_csv(relpaths, required_cols=None):
    for rp in relpaths:
        fp = os.path.join(BASE_DIR, rp) if not rp.startswith("/") else rp
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            if required_cols is not None:
                missing = set(required_cols) - set(df.columns)
                if missing:
                    raise ValueError(
                        f"CSV at {fp} missing required columns {missing}. Has columns: {list(df.columns)}"
                    )
            return df, fp
    return None, None


print("Using BASE_DIR:", BASE_DIR)

sample_df, sample_path = read_first_existing_csv(
    relpaths=[
        "sample_submission.csv",
        "siim-isic-melanoma-classification/sample_submission.csv",
    ],
    required_cols=["image_name", "target"],
)
test_df, test_path = read_first_existing_csv(
    relpaths=["test.csv", "siim-isic-melanoma-classification/test.csv"],
    required_cols=["image_name"],
)
train_df, train_path = read_first_existing_csv(
    relpaths=["train.csv", "siim-isic-melanoma-classification/train.csv"],
    required_cols=["image_name", "target", "patient_id"],
)

print("Loaded sample_submission from:", sample_path, "shape:", sample_df.shape)
print("Loaded test.csv from:", test_path, "shape:", test_df.shape)
print("Loaded train.csv from:", train_path, "shape:", train_df.shape)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold


def compute_oof_mean_smooth_groupkfold(
    df,
    group_col,
    target_col,
    groups,
    n_splits=5,
    m_smooth=5.0,
):
    global_mean = float(df[target_col].mean())
    oof = np.full(len(df), global_mean, dtype=float)

    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(df, df[target_col].values, groups=groups):
        tr = df.iloc[tr_idx]
        val = df.iloc[val_idx]

        stats = tr.groupby(group_col)[target_col].agg(["mean", "count"])
        val_keys = val[group_col]

        mean = val_keys.map(stats["mean"]).astype(float)
        cnt = val_keys.map(stats["count"]).astype(float)

        mean = mean.fillna(global_mean)
        cnt = cnt.fillna(0.0)

        smoothed = ((cnt * mean) + (m_smooth * global_mean)) / (cnt + m_smooth)
        oof[val_idx] = smoothed.values

    return oof, global_mean


def compute_oof_mean_smooth_kfold(
    df,
    group_col,
    target_col,
    n_splits=5,
    m_smooth=5.0,
    seed=42,
):
    global_mean = float(df[target_col].mean())
    oof = np.full(len(df), global_mean, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(df):
        tr = df.iloc[tr_idx]
        val = df.iloc[val_idx]

        stats = tr.groupby(group_col)[target_col].agg(["mean", "count"])
        val_keys = val[group_col]

        mean = val_keys.map(stats["mean"]).astype(float)
        cnt = val_keys.map(stats["count"]).astype(float)

        mean = mean.fillna(global_mean)
        cnt = cnt.fillna(0.0)

        smoothed = ((cnt * mean) + (m_smooth * global_mean)) / (cnt + m_smooth)
        oof[val_idx] = smoothed.values

    return oof, global_mean


def add_metadata_features(
    df,
    global_mean,
    patient_mean_map=None,
    patient_count_map=None,
    site_mean_map=None,
    site_count_map=None,
    sexsite_mean_map=None,
    sexsite_count_map=None,
    m_smooth=5.0,
):
    df = df.copy()

    df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
    df["age_missing"] = df["age_approx"].isna().astype(np.int8)

    age_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
    df["age_bin"] = pd.cut(df["age_approx"], bins=age_bins).astype(str)

    sex = df["sex"].fillna("unknown").astype(str)
    site = df["anatom_site_general_challenge"].fillna("unknown").astype(str)
    df["sex_x_site"] = sex + "__" + site

    if patient_mean_map is not None:
        base_mean = df["patient_id"].map(patient_mean_map).astype(float)
        if patient_count_map is not None:
            cnt = df["patient_id"].map(patient_count_map).astype(float).fillna(0.0)
            base_mean = base_mean.fillna(global_mean)
            df["patient_target_mean"] = (
                (cnt * base_mean) + (m_smooth * global_mean)
            ) / (cnt + m_smooth)
        else:
            df["patient_target_mean"] = base_mean.fillna(global_mean)
        df["patient_unseen"] = (
            df["patient_id"].map(patient_mean_map).isna().astype(np.int8)
        )
    else:
        df["patient_target_mean"] = global_mean
        df["patient_unseen"] = 1

    if site_mean_map is not None:
        site_base = df["anatom_site_general_challenge"].map(site_mean_map).astype(float)
        if site_count_map is not None:
            scnt = (
                df["anatom_site_general_challenge"]
                .map(site_count_map)
                .astype(float)
                .fillna(0.0)
            )
            site_base = site_base.fillna(global_mean)
            df["site_target_mean"] = ((scnt * site_base) + (m_smooth * global_mean)) / (
                scnt + m_smooth
            )
        else:
            df["site_target_mean"] = site_base.fillna(global_mean)
    else:
        df["site_target_mean"] = global_mean

    if sexsite_mean_map is not None:
        ss_base = df["sex_x_site"].map(sexsite_mean_map).astype(float)
        if sexsite_count_map is not None:
            sscnt = df["sex_x_site"].map(sexsite_count_map).astype(float).fillna(0.0)
            ss_base = ss_base.fillna(global_mean)
            df["sexsite_target_mean"] = (
                (sscnt * ss_base) + (m_smooth * global_mean)
            ) / (sscnt + m_smooth)
        else:
            df["sexsite_target_mean"] = ss_base.fillna(global_mean)
    else:
        df["sexsite_target_mean"] = global_mean

    return df


train_df = train_df.copy()
test_df = test_df.copy()

train_df["patient_id"] = train_df["patient_id"].astype(str)
test_df["patient_id"] = test_df["patient_id"].astype(str)

train_df = train_df.sort_values(["patient_id", "image_name"]).reset_index(drop=True)

global_mean = float(train_df["target"].mean())

groups = train_df["patient_id"].values
y_train = train_df["target"].astype(int).values

M_SMOOTH_PATIENT = 5.0
M_SMOOTH_SITE = 20.0
M_SMOOTH_SEXSITE = 20.0

oof_patient_mean, _ = compute_oof_mean_smooth_groupkfold(
    df=train_df,
    group_col="patient_id",
    target_col="target",
    groups=groups,
    n_splits=5,
    m_smooth=M_SMOOTH_PATIENT,
)

oof_site_mean, _ = compute_oof_mean_smooth_groupkfold(
    df=train_df,
    group_col="anatom_site_general_challenge",
    target_col="target",
    groups=groups,
    n_splits=5,
    m_smooth=M_SMOOTH_SITE,
)

train_meta_for_sexsite = train_df[
    ["sex", "anatom_site_general_challenge", "target", "patient_id"]
].copy()
sex = train_meta_for_sexsite["sex"].fillna("unknown").astype(str)
site = (
    train_meta_for_sexsite["anatom_site_general_challenge"]
    .fillna("unknown")
    .astype(str)
)
train_meta_for_sexsite["sex_x_site"] = sex + "__" + site

oof_sexsite_mean, _ = compute_oof_mean_smooth_groupkfold(
    df=train_meta_for_sexsite,
    group_col="sex_x_site",
    target_col="target",
    groups=train_meta_for_sexsite["patient_id"].values,
    n_splits=5,
    m_smooth=M_SMOOTH_SEXSITE,
)

patient_stats_full = train_df.groupby("patient_id")["target"].agg(["mean", "count"])
patient_mean_map_full = patient_stats_full["mean"]
patient_count_map_full = patient_stats_full["count"]

patient_level = (
    train_df.groupby(["patient_id", "anatom_site_general_challenge"])["target"]
    .mean()
    .reset_index()
)
site_stats_full = patient_level.groupby("anatom_site_general_challenge")["target"].agg(
    ["mean", "count"]
)
site_mean_map_full = site_stats_full["mean"]
site_count_map_full = site_stats_full["count"]

sexsite_patient_level = (
    train_meta_for_sexsite.groupby(["patient_id", "sex_x_site"])["target"]
    .mean()
    .reset_index()
)
sexsite_stats_full = sexsite_patient_level.groupby("sex_x_site")["target"].agg(
    ["mean", "count"]
)
sexsite_mean_map_full = sexsite_stats_full["mean"]
sexsite_count_map_full = sexsite_stats_full["count"]

X_train_raw = train_df[
    ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]
].copy()
X_test_raw = test_df[
    ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]
].copy()

X_train = add_metadata_features(
    X_train_raw,
    global_mean=global_mean,
    patient_mean_map=None,  # filled from OOF vectors
    site_mean_map=None,
    sexsite_mean_map=None,  # filled from OOF vector
)

X_train["patient_target_mean"] = np.clip(oof_patient_mean.astype(float), 0.0, 1.0)
X_train["patient_unseen"] = 0
X_train["site_target_mean"] = np.clip(oof_site_mean.astype(float), 0.0, 1.0)
X_train["sexsite_target_mean"] = np.clip(oof_sexsite_mean.astype(float), 0.0, 1.0)

X_test = add_metadata_features(
    X_test_raw,
    global_mean=global_mean,
    patient_mean_map=patient_mean_map_full,
    patient_count_map=patient_count_map_full,
    site_mean_map=site_mean_map_full,
    site_count_map=site_count_map_full,
    sexsite_mean_map=sexsite_mean_map_full,
    sexsite_count_map=sexsite_count_map_full,
    m_smooth=M_SMOOTH_SITE,  # default; we override patient/sexsite below for exact match
)

cnt_p = X_test_raw["patient_id"].map(patient_count_map_full).astype(float).fillna(0.0)
mean_p = (
    X_test_raw["patient_id"]
    .map(patient_mean_map_full)
    .astype(float)
    .fillna(global_mean)
)
X_test["patient_target_mean"] = (
    (cnt_p * mean_p) + (M_SMOOTH_PATIENT * global_mean)
) / (cnt_p + M_SMOOTH_PATIENT)

cnt_ss = X_test["sex_x_site"].map(sexsite_count_map_full).astype(float).fillna(0.0)
mean_ss = (
    X_test["sex_x_site"].map(sexsite_mean_map_full).astype(float).fillna(global_mean)
)
X_test["sexsite_target_mean"] = (
    (cnt_ss * mean_ss) + (M_SMOOTH_SEXSITE * global_mean)
) / (cnt_ss + M_SMOOTH_SEXSITE)

X_test["patient_target_mean"] = np.clip(
    X_test["patient_target_mean"].astype(float).values, 0.0, 1.0
)
X_test["site_target_mean"] = np.clip(
    X_test["site_target_mean"].astype(float).values, 0.0, 1.0
)
X_test["sexsite_target_mean"] = np.clip(
    X_test["sexsite_target_mean"].astype(float).values, 0.0, 1.0
)

numeric_features = [
    "age_approx",
    "patient_target_mean",
    "site_target_mean",
    "sexsite_target_mean",
    "age_missing",
    "patient_unseen",
]

categorical_features = [
    "sex",
    "anatom_site_general_challenge",
    "age_bin",
    "sex_x_site",
    "patient_id",
]

ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=3)

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
                    ("ohe", ohe),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

clf = LogisticRegression(
    solver="liblinear",
    max_iter=500,
    C=0.3,
    class_weight=None,
)

model = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
model.fit(X_train, y_train)

test_pred = model.predict_proba(X_test)[:, 1].astype(float)

sub = pd.DataFrame({"image_name": test_df["image_name"].astype(str)})
sub["target"] = np.clip(test_pred, 0.0, 1.0)

assert list(sub.columns) == ["image_name", "target"]
sub["target"] = sub["target"].astype(float)



## === cell 1
out_path = "submission.csv"
sub.to_csv(out_path, index=False)

check = pd.read_csv(out_path)
if "image_name" not in check.columns or "target" not in check.columns:
    raise ValueError(f"Invalid submission columns: {list(check.columns)}")
if len(check) != len(test_df):
    raise ValueError(
        f"Submission row count {len(check)} != test row count {len(test_df)}"
    )

print("Wrote submission:", out_path, "shape:", check.shape)
print(check.head())
print("Prediction summary:")
print(check["target"].describe())
