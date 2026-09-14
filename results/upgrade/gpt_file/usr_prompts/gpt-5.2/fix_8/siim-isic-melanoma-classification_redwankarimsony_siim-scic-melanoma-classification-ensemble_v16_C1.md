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

0.9091

# 6. Current score

0.78919

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because your notebook depends on an external Kaggle dataset (`public-submission-melanoma-95`) that is not available in this environment, so none of those CSVs can be loaded and the downstream cells fail. To keep the core “blend submissions” logic intact but make it runnable end-to-end, I add a small fallback that uses the provided `sample_submission.csv` and generates a deterministic, reasonable prior probability from `train.csv` (overall malignant rate) when the external files are missing. This produces a valid `submission.csv` with the correct columns and row order (matching `test.csv`/sample submission). If the external files do exist in some other environment, the original blend path still run unchanged.'
- What this solution (achieved 0.66776) has done: 'Your current 0.5 AUC comes from the fallback path predicting a constant prior for every test image, which yields random-ranking performance (AUC≈0.5). To move toward the 0.9091 target without changing the core “tabular-only fallback” approach, I keep the same simple metadata-only idea but replace the constant prediction with a minimal, deterministic model trained on `train.csv` metadata and applied to `test.csv`. Concretely, I use a scikit-learn logistic regression with straightforward preprocessing (impute + one-hot encode), and I only use this model when the external blend files are missing, preserving your original blend logic unchanged. This should materially improve AUC versus constant predictions while still being lightweight and fast, and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.72693) has done: 'To move your AUC up toward 0.9091 without changing the overall “blend if available, else metadata-only fallback” core logic, I keep the same logistic-regression fallback but make two minimal, score-relevant improvements: (1) add patient-level target encoding for `patient_id` using out-of-fold means (prevents leakage and typically boosts ranking), and (2) switch logistic regression to a slightly stronger, still-fast `saga` solver with mild L2 regularization tuning and more iterations for better convergence. I also ensure the submission row order always matches `test.csv` and that all feature columns are consistently typed/imputed, which avoids silent train/test mismatches that can hurt AUC. The blend path remains unchanged and still be used if the external CSVs exist.'
- What this solution (achieved 0.5153) has done: 'Your current score (0.72693) is well below the target (0.9091), so we should improve ranking quality while keeping the same “blend if available, else metadata-only fallback” structure and the same logistic-regression core. The biggest low-risk gain is to make the patient target-encoding more reliable by adding smoothing and a frequency feature (patient lesion count), which typically improves AUC without changing the overall approach. I also add one additional categorical feature (`diagnosis`, train-only but safely handled as missing in test) and tune logistic regression very slightly (still L2 + saga) to better use the richer feature set. The blend path remains unchanged; these changes only affect the fallback path when external submission files are unavailable.'
- What this solution (achieved 0.77826) has done: 'Your current gap to the target AUC is large (0.5153 → 0.9091), so the smallest legitimate improvement is to strengthen ranking signal while keeping the same fallback “metadata logistic regression” core logic unchanged. The biggest issue is that `diagnosis` is train-only, so in your current code it gets imputed to a constant in test and effectively adds noise; I drop it and instead add two safe, high-signal metadata features: one-hot encoded `patient_id` plus a smoothed target-encoded `anatom_site_general_challenge` (computed out-of-fold to avoid leakage). I also compute `pid_count` correctly out-of-fold (count within the training fold only) to prevent subtle leakage and improve generalization. The blend path remains identical and still takes priority if those external CSVs exist, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.77247) has done: 'To move your AUC up toward the 0.9091 target while keeping the same core “blend if available, else metadata logistic regression” logic, I make two minimal, score-relevant fixes in the fallback path: (1) avoid high-cardinality `patient_id` one-hot (which tends to overfit and hurts generalization) and replace it with a safe out-of-fold target encoding plus count (already present), and (2) add one more strong, competition-standard metadata signal: an out-of-fold target-encoded `age_approx` (binned) to capture the nonlinear age→risk relationship without changing the model class. The blend path remains unchanged and still takes priority when those external CSVs exist. Submission formatting and row alignment remain identical, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.78919) has done: 'Your current AUC (0.77247) is well below the target (0.9091), so we should improve ranking signal with the smallest safe changes while preserving the same fallback “metadata + target-encoding + logistic regression” core. The most impactful minimal fix is to compute `pid_count` out-of-fold (right now it’s effectively 0 for every training row, so the model can’t learn from that feature), and then standardize it with `log1p` to make it usable. I also add two lightweight interaction features (`age * site_target_mean` and `age * pid_target_mean`) that keep the same model and training approach but usually improve AUC for this competition’s metadata setting. The blend path remains unchanged, and the script still writes a valid `submission.csv` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input"
COMP_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "siim-isic-melanoma-classification"),
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/data",
]


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


comp_dir = _first_existing(*COMP_DIR_CANDIDATES)
if comp_dir is None:
    raise FileNotFoundError(
        "Could not locate competition data directory in expected paths."
    )

sample_path = os.path.join(comp_dir, "sample_submission.csv")
train_path = os.path.join(comp_dir, "train.csv")
test_path = os.path.join(comp_dir, "test.csv")

sub = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path)

if (
    "image_name" in test_df.columns
    and sub["image_name"].tolist() != test_df["image_name"].tolist()
):
    sub = sub.set_index("image_name").reindex(test_df["image_name"]).reset_index()

public_dir = os.path.join(BASE_INPUT, "public-submission-melanoma-95")


def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


public_sub_mean_9533 = safe_read_csv(os.path.join(public_dir, "submission_mean.csv"))
public_sub_median_9533 = safe_read_csv(
    os.path.join(public_dir, "submission_median.csv")
)
public_sub_meta_ens_9577 = safe_read_csv(
    os.path.join(public_dir, "external_meta_ensembled.csv")
)
public_sub_9581 = safe_read_csv(os.path.join(public_dir, "submission_9581.csv"))
public_sub_tabular = safe_read_csv(
    os.path.join(public_dir, "submission_tabular_only.csv")
)
public_sub_9619 = safe_read_csv(os.path.join(public_dir, "submission_9619.csv"))



## === cell 1
have_blend_inputs = (
    public_sub_9619 is not None
    and public_sub_median_9533 is not None
    and public_sub_mean_9533 is not None
    and public_sub_tabular is not None
)

if have_blend_inputs:
    sub["target"] = (
        public_sub_9619["target"] * 0.40
        + public_sub_median_9533["target"] * 0.20
        + public_sub_mean_9533["target"] * 0.20
        + public_sub_tabular["target"] * 0.20
    )
else:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    train_full = pd.read_csv(train_path)
    test_full = pd.read_csv(test_path)

    for c in ["sex", "anatom_site_general_challenge", "patient_id"]:
        if c in train_full.columns:
            train_full[c] = train_full[c].astype("object")
        if c in test_full.columns:
            test_full[c] = test_full[c].astype("object")

    y = train_full["target"].astype(int).values
    global_mean = float(np.mean(y))

    pid_tr = train_full["patient_id"].astype("object").fillna("__MISSING__")
    pid_te = test_full["patient_id"].astype("object").fillna("__MISSING__")
    site_tr = (
        train_full["anatom_site_general_challenge"]
        .astype("object")
        .fillna("__MISSING__")
    )
    site_te = (
        test_full["anatom_site_general_challenge"]
        .astype("object")
        .fillna("__MISSING__")
    )

    age_tr_num = pd.to_numeric(train_full["age_approx"], errors="coerce")
    age_te_num = pd.to_numeric(test_full["age_approx"], errors="coerce")
    age_tr_bin = (
        (np.floor(age_tr_num / 5) * 5)
        .astype("Int64")
        .astype("object")
        .fillna("__MISSING__")
    )
    age_te_bin = (
        (np.floor(age_te_num / 5) * 5)
        .astype("Int64")
        .astype("object")
        .fillna("__MISSING__")
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    oof_pid_mean_smooth = np.zeros(len(train_full), dtype=np.float32)
    oof_pid_count = np.zeros(len(train_full), dtype=np.float32)
    oof_site_mean_smooth = np.zeros(len(train_full), dtype=np.float32)
    oof_agebin_mean_smooth = np.zeros(len(train_full), dtype=np.float32)

    alpha_pid = 20.0
    alpha_site = 10.0
    alpha_age = 10.0

    for tr_idx, va_idx in skf.split(train_full, y):
        tr_y = y[tr_idx]

        tr_pid = pid_tr.iloc[tr_idx]
        pid_stats = (
            pd.DataFrame({"k": tr_pid.values, "target": tr_y})
            .groupby("k")["target"]
            .agg(["mean", "count"])
        )
        va_pid = pid_tr.iloc[va_idx]
        va_mean = va_pid.map(pid_stats["mean"])
        va_cnt = va_pid.map(pid_stats["count"])

        va_mean = va_mean.fillna(global_mean).astype(np.float32).values
        va_cnt = va_cnt.fillna(0).astype(np.float32).values

        oof_pid_mean_smooth[va_idx] = (va_cnt * va_mean + alpha_pid * global_mean) / (
            va_cnt + alpha_pid
        )

        oof_pid_count[va_idx] = va_cnt

        tr_site = site_tr.iloc[tr_idx]
        site_stats = (
            pd.DataFrame({"k": tr_site.values, "target": tr_y})
            .groupby("k")["target"]
            .agg(["mean", "count"])
        )
        va_site = site_tr.iloc[va_idx]
        va_smean = va_site.map(site_stats["mean"])
        va_scnt = va_site.map(site_stats["count"])
        va_smean = va_smean.fillna(global_mean).astype(np.float32).values
        va_scnt = va_scnt.fillna(0).astype(np.float32).values
        oof_site_mean_smooth[va_idx] = (
            va_scnt * va_smean + alpha_site * global_mean
        ) / (va_scnt + alpha_site)

        tr_ab = age_tr_bin.iloc[tr_idx]
        ab_stats = (
            pd.DataFrame({"k": tr_ab.values, "target": tr_y})
            .groupby("k")["target"]
            .agg(["mean", "count"])
        )
        va_ab = age_tr_bin.iloc[va_idx]
        va_ab_mean = va_ab.map(ab_stats["mean"])
        va_ab_cnt = va_ab.map(ab_stats["count"])
        va_ab_mean = va_ab_mean.fillna(global_mean).astype(np.float32).values
        va_ab_cnt = va_ab_cnt.fillna(0).astype(np.float32).values
        oof_agebin_mean_smooth[va_idx] = (
            va_ab_cnt * va_ab_mean + alpha_age * global_mean
        ) / (va_ab_cnt + alpha_age)

    full_pid_stats = (
        pd.DataFrame({"k": pid_tr.values, "target": y})
        .groupby("k")["target"]
        .agg(["mean", "count"])
    )
    te_mean = pid_te.map(full_pid_stats["mean"])
    te_cnt = pid_te.map(full_pid_stats["count"])
    te_mean = te_mean.fillna(global_mean).astype(np.float32).values
    te_cnt = te_cnt.fillna(0).astype(np.float32).values
    test_pid_mean_smooth = (te_cnt * te_mean + alpha_pid * global_mean) / (
        te_cnt + alpha_pid
    )
    test_pid_count = te_cnt.astype(np.float32)

    full_site_stats = (
        pd.DataFrame({"k": site_tr.values, "target": y})
        .groupby("k")["target"]
        .agg(["mean", "count"])
    )
    te_smean = site_te.map(full_site_stats["mean"])
    te_scnt = site_te.map(full_site_stats["count"])
    te_smean = te_smean.fillna(global_mean).astype(np.float32).values
    te_scnt = te_scnt.fillna(0).astype(np.float32).values
    test_site_mean_smooth = (te_scnt * te_smean + alpha_site * global_mean) / (
        te_scnt + alpha_site
    )

    full_ab_stats = (
        pd.DataFrame({"k": age_tr_bin.values, "target": y})
        .groupby("k")["target"]
        .agg(["mean", "count"])
    )
    te_ab_mean = age_te_bin.map(full_ab_stats["mean"])
    te_ab_cnt = age_te_bin.map(full_ab_stats["count"])
    te_ab_mean = te_ab_mean.fillna(global_mean).astype(np.float32).values
    te_ab_cnt = te_ab_cnt.fillna(0).astype(np.float32).values
    test_agebin_mean_smooth = (te_ab_cnt * te_ab_mean + alpha_age * global_mean) / (
        te_ab_cnt + alpha_age
    )

    train_full = train_full.copy()
    test_full = test_full.copy()
    train_full["pid_target_mean"] = oof_pid_mean_smooth
    test_full["pid_target_mean"] = test_pid_mean_smooth
    train_full["pid_count"] = oof_pid_count
    test_full["pid_count"] = test_pid_count
    train_full["site_target_mean"] = oof_site_mean_smooth
    test_full["site_target_mean"] = test_site_mean_smooth.astype(np.float32)
    train_full["agebin_target_mean"] = oof_agebin_mean_smooth
    test_full["agebin_target_mean"] = test_agebin_mean_smooth.astype(np.float32)

    train_full["pid_count_log1p"] = np.log1p(train_full["pid_count"].astype(np.float32))
    test_full["pid_count_log1p"] = np.log1p(test_full["pid_count"].astype(np.float32))

    age_tr_num_feat = pd.to_numeric(train_full["age_approx"], errors="coerce").astype(
        np.float32
    )
    age_te_num_feat = pd.to_numeric(test_full["age_approx"], errors="coerce").astype(
        np.float32
    )
    train_full["age_x_site"] = age_tr_num_feat * train_full["site_target_mean"].astype(
        np.float32
    )
    test_full["age_x_site"] = age_te_num_feat * test_full["site_target_mean"].astype(
        np.float32
    )
    train_full["age_x_pid"] = age_tr_num_feat * train_full["pid_target_mean"].astype(
        np.float32
    )
    test_full["age_x_pid"] = age_te_num_feat * test_full["pid_target_mean"].astype(
        np.float32
    )

    features = [
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "pid_target_mean",
        "pid_count_log1p",
        "site_target_mean",
        "agebin_target_mean",
        "age_x_site",
        "age_x_pid",
    ]
    X_train = train_full[features].copy()
    X_test = test_full[features].copy()

    numeric_features = [
        "age_approx",
        "pid_target_mean",
        "pid_count_log1p",
        "site_target_mean",
        "agebin_target_mean",
        "age_x_site",
        "age_x_pid",
    ]
    categorical_features = ["sex", "anatom_site_general_challenge"]

    preprocessor = ColumnTransformer(
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
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2500,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    )

    model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    model.fit(X_train, y)

    proba = model.predict_proba(X_test)[:, 1]
    sub["target"] = proba

sub["target"] = pd.to_numeric(sub["target"], errors="coerce").fillna(0.0).clip(0.0, 1.0)



## === cell 2
sub.head()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 3
sub.head()
