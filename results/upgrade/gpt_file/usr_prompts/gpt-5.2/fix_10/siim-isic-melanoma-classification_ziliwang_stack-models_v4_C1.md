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

0.9295456944914684

# 6. Current score

0.3127

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the internet `wget` dependency (it produces an empty file in this environment) and instead build a deterministic baseline submission from the provided `sample_submission.csv`. To keep your original ensemble structure without requiring unavailable Kaggle input datasets, I add a safe loader that uses those external submissions if present, otherwise falls back to the sample submission with constant probabilities. Finally, I ensure the output CSV is written as `submission.csv` with exactly the required columns (`image_name`, `target`) and correct row alignment with the official sample submission.'
- What this solution (achieved 0.64551) has done: 'Your current 0.5 AUC comes from submitting a constant prediction fallback because the external ensemble files are not present in this environment. To move the score toward the 0.9295 target without changing the overall “make a submission from available files” core logic, I replace the constant fallback with a simple, deterministic metadata-based probability model trained on `train.csv` and applied to `test.csv`. This keeps the same end-to-end structure (load sample → build predictions → write `submission.csv`) while producing non-constant, label-informed predictions that should materially improve AUC. I also keep your existing `safe_read_submission` mechanism and blend in any external submissions if they happen to exist, but otherwise the metadata model drive the output.'
- What this solution (achieved 0.64659) has done: 'Your current gap is large (0.6455 vs target 0.9295 AUC), so we should legitimately improve the predictive signal while keeping your “metadata-only model + optional blending with external submissions” core structure intact. The biggest low-risk win is to prevent patient leakage and better use the metadata by learning smoothed target means for *feature interactions* (sex×site, site×agebin, sex×agebin), which usually lifts AUC versus averaging three marginal encodings. I also add an out-of-fold (grouped by `patient_id`) encoding step so the learned encodings generalize better to test and avoid inflated/unstable mappings. Finally, I keep your existing optional blending logic and still write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.65486) has done: 'We keep your existing metadata-only target-encoding core logic, but fix a key correctness issue: you compute out-of-fold encodings yet never use them, so the model is effectively using leakage-prone full-data mappings and doesn’t benefit from the intended generalization. I integrate the OOF encodings to learn stable weights (via a tiny ridge regression on the OOF predictions) and then apply the same learned weights to the test predictions computed from full-data maps—this preserves the same features and approach but calibrates them better for AUC. I also add a deterministic patient-level prior feature (smoothed target mean by patient_id) which is legitimate and typically strong here, and include it in the same minimal linear blending step. The script still run end-to-end, keep optional external blending unchanged, and write a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.68849) has done: 'Your current score (0.65486 AUC) is far below the target (0.92955), so we should add legitimate predictive signal while keeping your core “OOF target-encoding components + ridge blend + optional external blending” approach intact. The biggest low-risk gain is to (1) stop using a patient_id mean as a test-time feature (it doesn’t generalize because patient_id is essentially unseen in test) and (2) add two strong, competition-standard metadata-derived features: per-patient lesion count and per-patient mean age (computed on train and mapped to test), both used via the same OOF target-encoding + ridge-blend mechanism you already have. These are minimal additions that preserve your training loop and model semantics (still just smoothed target encodings blended by ridge), but usually lift AUC materially for this dataset. I also keep your external submission blending unchanged and still write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.33208) has done: 'We keep your exact metadata target-encoding + OOF ridge-blend core logic, but add one missing strong metadata signal that is available in `train.csv` and correlates with malignancy: `benign_malignant` (train only). Using it only to compute smoothed target means by `diagnosis` and `benign_malignant` (and then mapping those onto test as priors) is a minimal extension of the same encoding approach and should materially lift AUC from ~0.69 toward your 0.93 target. To avoid leakage/overfit, we also include OOF encodings for these two new categorical priors in the same ridge weight fitting you already do. Submission writing, optional external blending, and ordering remain unchanged.'
- What this solution (achieved 0.31178) has done: 'Your score collapsed because `benign_malignant` is a train-only column and your current code effectively turns it into a constant feature at test time, while also letting the ridge fit over-weight it based on OOF signal that cannot transfer—this can severely hurt generalization and AUC. To move back toward the 0.9295 target with minimal logic change, I (1) remove `benign_malignant` from the ridge-blended feature set (keep it unused) and (2) add two safe, train→test transferable categorical encodings (`diagnosis` and `patient_id`) that are already present in both train and test. These are implemented with the same existing smoothed target encoding + patient-grouped OOF ridge blending you already use, so the approach stays identical, just with corrected/stronger features. The submission writing, ordering, and optional external blending remain unchanged.'
- What this solution (achieved 0.31359) has done: 'Your current AUC collapse is consistent with a leakage-heavy feature: the `patient_id` target-mean encoding, which cannot transfer because nearly all `patient_id`s in test are unseen, so it injects noise and can dominate the ridge blend. To move the score back up toward the 0.9295 target with minimal changes, I remove `patient_id` from both the OOF ridge-fitting matrix and the test-time metadata component matrix (keeping everything else identical). I also remove the unused `bm`/`benign_malignant` OOF feature plumbing to avoid accidental weighting of non-transferable signals, without changing the overall “target-encoded metadata components + ridge blend + optional external submission blending” approach. Submission writing, column order, and paths remain unchanged.'
- What this solution (achieved 0.3127) has done: 'We need to recover AUC from ~0.31 toward 0.93; the most likely cause is that the current ridge blend is trained with an MSE objective on probability-scale encodings, which can yield poor ranking (AUC) when weights become poorly calibrated/negative. To keep your core logic (same target-encoded metadata components + OOF training + linear blend + optional external blending), I change only the weight-fitting step to optimize *ranking-consistent* logit-space with the same ridge closed-form, and then convert back with a sigmoid—this typically improves AUC without changing features or introducing iterative training. I also standardize the OOF feature columns before fitting (and apply the same standardization to test) to prevent any single encoding from dominating due to scale/variance, which is a minimal and stable numerical fix. Everything else (data loading, encodings, OOF grouped-by-patient splitting, optional blending, and `submission.csv` format) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/data"
INPUT_DIR = "/kaggle/input"

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

sample = pd.read_csv(sample_path)
sample["target"] = sample["target"].astype(float)


def safe_read_submission(path, sample_df):
    """
    Try to read a submission-like CSV.
    If missing/invalid, return a fallback with the correct image_name order and NaN targets
    (so we can optionally fill from our metadata model rather than forcing a constant 0.5).
    """
    try:
        if path is None or (not os.path.exists(path)):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if not {"image_name", "target"}.issubset(df.columns):
            raise ValueError(
                f"Missing required columns in {path}: {df.columns.tolist()}"
            )
        df = df[["image_name", "target"]].copy()
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df = sample_df[["image_name"]].merge(df, on="image_name", how="left")
        return df
    except Exception:
        fb = sample_df[["image_name"]].copy()
        fb["target"] = np.nan
        return fb


a = safe_read_submission(os.path.join("/kaggle/working", "other1.csv"), sample)
b = safe_read_submission(
    os.path.join(INPUT_DIR, "stack-models", "submission.csv"), sample
)
c = safe_read_submission(
    os.path.join(INPUT_DIR, "melanoma-efficientnet-b6-tpu-tta", "submission.csv"),
    sample,
)



## === cell 2
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(INPUT_DIR, "train.csv")
if not os.path.exists(test_path):
    test_path = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sex"] = out["sex"].fillna("unknown").replace({"": "unknown"})
    out["anatom_site_general_challenge"] = (
        out["anatom_site_general_challenge"].fillna("unknown").replace({"": "unknown"})
    )
    out["age_approx"] = pd.to_numeric(out["age_approx"], errors="coerce")
    out["patient_id"] = out["patient_id"].fillna("unknown").replace({"": "unknown"})
    if "diagnosis" in out.columns:
        out["diagnosis"] = out["diagnosis"].fillna("unknown").replace({"": "unknown"})
    if "benign_malignant" in out.columns:
        out["benign_malignant"] = (
            out["benign_malignant"].fillna("unknown").replace({"": "unknown"})
        )
    return out


train = _prep(train)
test = _prep(test)

global_mean = float(train["target"].mean())


def _smoothed_map(train_df, key_col, y_col="target", alpha=20.0):
    stats = train_df.groupby(key_col, dropna=False)[y_col].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * alpha) / (
        stats["count"] + alpha
    )
    return smoothed.to_dict()


train["_pcount"] = (
    train.groupby("patient_id")["image_name"].transform("count").astype(int)
)
test["_pcount"] = (
    test.groupby("patient_id")["image_name"].transform("count").astype(int)
)

train["_page_mean"] = (
    train.groupby("patient_id")["age_approx"].transform("mean").astype(float)
)
test["_page_mean"] = (
    test.groupby("patient_id")["age_approx"].transform("mean").astype(float)
)


def _make_bins_from_train(x: pd.Series, q: int = 10):
    v = pd.to_numeric(x, errors="coerce")
    vv = v[~v.isna()]
    if len(vv) == 0:
        return np.array([0.0, 1.0])
    bins = np.nanquantile(vv, np.linspace(0, 1, q + 1))
    bins = np.unique(bins)
    if len(bins) < 3:
        mn = float(np.nanmin(vv))
        mx = float(np.nanmax(vv))
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            bins = np.array([0.0, 1.0, 2.0])
        else:
            bins = np.array([mn, (mn + mx) / 2.0, mx])
    return bins


age_train = train["age_approx"].copy()
age_bins = np.nanquantile(age_train, np.linspace(0, 1, 11))
age_bins = np.unique(age_bins)
if len(age_bins) < 3:
    age_bins = np.array([0.0, 50.0, 100.0])

train_age_bin = pd.cut(train["age_approx"], bins=age_bins, include_lowest=True)
test_age_bin = pd.cut(test["age_approx"], bins=age_bins, include_lowest=True)

train = train.assign(_age_bin=train_age_bin.astype(str))
test = test.assign(_age_bin=test_age_bin.astype(str))

pcount_bins = _make_bins_from_train(train["_pcount"], q=10)
page_bins = _make_bins_from_train(train["_page_mean"], q=10)

train = train.assign(
    _pcount_bin=pd.cut(
        train["_pcount"].astype(float), bins=pcount_bins, include_lowest=True
    ).astype(str),
    _page_bin=pd.cut(
        train["_page_mean"].astype(float), bins=page_bins, include_lowest=True
    ).astype(str),
)
test = test.assign(
    _pcount_bin=pd.cut(
        test["_pcount"].astype(float), bins=pcount_bins, include_lowest=True
    ).astype(str),
    _page_bin=pd.cut(
        test["_page_mean"].astype(float), bins=page_bins, include_lowest=True
    ).astype(str),
)

train = train.assign(
    _sex_site=(
        train["sex"].astype(str)
        + "||"
        + train["anatom_site_general_challenge"].astype(str)
    ),
    _site_age=(
        train["anatom_site_general_challenge"].astype(str)
        + "||"
        + train["_age_bin"].astype(str)
    ),
    _sex_age=(train["sex"].astype(str) + "||" + train["_age_bin"].astype(str)),
)
test = test.assign(
    _sex_site=(
        test["sex"].astype(str)
        + "||"
        + test["anatom_site_general_challenge"].astype(str)
    ),
    _site_age=(
        test["anatom_site_general_challenge"].astype(str)
        + "||"
        + test["_age_bin"].astype(str)
    ),
    _sex_age=(test["sex"].astype(str) + "||" + test["_age_bin"].astype(str)),
)


def _group_kfold_assignments(
    groups: pd.Series, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """
    Deterministic group split without sklearn dependency.
    Assign each unique group to a fold after shuffling with a fixed seed.
    """
    g = groups.fillna("__nan__").astype(str).to_numpy()
    uniq = np.unique(g)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {grp: i % n_splits for i, grp in enumerate(uniq)}
    return np.array([fold_of[gi] for gi in g], dtype=np.int16)


def _oof_target_encode(
    df: pd.DataFrame,
    key_col: str,
    group_col: str = "patient_id",
    y_col: str = "target",
    alpha: float = 20.0,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.Series:
    """
    Out-of-fold smoothed mean encoding by grouping on patient_id to reduce leakage.
    Returns an encoded Series aligned to df.index (train only).
    """
    folds = _group_kfold_assignments(df[group_col], n_splits=n_splits, seed=seed)
    enc = np.empty(len(df), dtype=float)
    for f in range(n_splits):
        trn_idx = folds != f
        val_idx = folds == f
        mp = _smoothed_map(df.loc[trn_idx], key_col=key_col, y_col=y_col, alpha=alpha)
        keys = df.loc[val_idx, key_col].astype(str)
        enc[val_idx] = keys.map(mp).fillna(global_mean).to_numpy(dtype=float)
    return pd.Series(enc, index=df.index, name=f"enc_{key_col}")


sex_map = _smoothed_map(train, "sex", alpha=50.0)
site_map = _smoothed_map(train, "anatom_site_general_challenge", alpha=50.0)
agebin_map = _smoothed_map(train, "_age_bin", alpha=50.0)

sex_site_map = _smoothed_map(train, "_sex_site", alpha=80.0)
site_age_map = _smoothed_map(train, "_site_age", alpha=80.0)
sex_age_map = _smoothed_map(train, "_sex_age", alpha=80.0)

pcount_map = _smoothed_map(train, "_pcount_bin", alpha=80.0)
page_map = _smoothed_map(train, "_page_bin", alpha=80.0)

dx_map = (
    _smoothed_map(train, "diagnosis", alpha=100.0)
    if "diagnosis" in train.columns
    else {}
)


def metadata_components(df: pd.DataFrame) -> dict:
    """Return per-feature probability components (used for both train OOF-weight fit and test inference)."""
    comps = {}
    comps["sex"] = df["sex"].map(sex_map).fillna(global_mean).to_numpy(dtype=float)
    comps["site"] = (
        df["anatom_site_general_challenge"]
        .map(site_map)
        .fillna(global_mean)
        .to_numpy(dtype=float)
    )
    comps["agebin"] = (
        df["_age_bin"].map(agebin_map).fillna(global_mean).to_numpy(dtype=float)
    )

    comps["sex_site"] = (
        df["_sex_site"].map(sex_site_map).fillna(global_mean).to_numpy(dtype=float)
    )
    comps["site_age"] = (
        df["_site_age"].map(site_age_map).fillna(global_mean).to_numpy(dtype=float)
    )
    comps["sex_age"] = (
        df["_sex_age"].map(sex_age_map).fillna(global_mean).to_numpy(dtype=float)
    )

    comps["pcount"] = (
        df["_pcount_bin"].map(pcount_map).fillna(global_mean).to_numpy(dtype=float)
    )
    comps["page"] = (
        df["_page_bin"].map(page_map).fillna(global_mean).to_numpy(dtype=float)
    )

    if "diagnosis" in df.columns and len(dx_map) > 0:
        comps["dx"] = (
            df["diagnosis"].map(dx_map).fillna(global_mean).to_numpy(dtype=float)
        )
    else:
        comps["dx"] = np.full(len(df), global_mean, dtype=float)

    return comps


oof_sex = _oof_target_encode(train, "sex", alpha=50.0)
oof_site = _oof_target_encode(train, "anatom_site_general_challenge", alpha=50.0)
oof_agebin = _oof_target_encode(train, "_age_bin", alpha=50.0)

oof_sex_site = _oof_target_encode(train, "_sex_site", alpha=80.0)
oof_site_age = _oof_target_encode(train, "_site_age", alpha=80.0)
oof_sex_age = _oof_target_encode(train, "_sex_age", alpha=80.0)

oof_pcount = _oof_target_encode(train, "_pcount_bin", alpha=80.0)
oof_page = _oof_target_encode(train, "_page_bin", alpha=80.0)

oof_dx = (
    _oof_target_encode(train, "diagnosis", alpha=100.0)
    if "diagnosis" in train.columns
    else pd.Series(
        np.full(len(train), global_mean, dtype=float),
        index=train.index,
        name="enc_diagnosis",
    )
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _fit_ridge_blend_logit(
    oof_matrix_prob: np.ndarray, y: np.ndarray, lam: float = 1e-2
):
    """
    Change is directly score-relevant for AUC: fit the linear blend in logit-space
    (ranking-friendly) and standardize features so no single encoding dominates.
    Core logic is unchanged: same components, same OOF, same closed-form ridge solve.
    Returns (weights, feature_mean, feature_std) to apply consistently at test time.
    """
    mu = oof_matrix_prob.mean(axis=0)
    sd = oof_matrix_prob.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xz = (oof_matrix_prob - mu) / sd

    yz = _logit(y.astype(float))

    X = np.c_[np.ones((Xz.shape[0], 1), dtype=float), Xz.astype(float)]
    yv = yz.reshape(-1, 1)
    XtX = X.T @ X
    reg = np.eye(XtX.shape[0], dtype=float) * lam
    reg[0, 0] = 0.0
    w = np.linalg.solve(XtX + reg, X.T @ yv).ravel()
    return w, mu, sd


oof_mat = np.column_stack(
    [
        oof_sex.to_numpy(dtype=float),
        oof_site.to_numpy(dtype=float),
        oof_agebin.to_numpy(dtype=float),
        oof_sex_site.to_numpy(dtype=float),
        oof_site_age.to_numpy(dtype=float),
        oof_sex_age.to_numpy(dtype=float),
        oof_pcount.to_numpy(dtype=float),
        oof_page.to_numpy(dtype=float),
        oof_dx.to_numpy(dtype=float),
    ]
)
y = train["target"].to_numpy(dtype=float)

w, oof_mu, oof_sd = _fit_ridge_blend_logit(oof_mat, y, lam=1e-2)

test_comps = metadata_components(test)
test_mat = np.column_stack(
    [
        test_comps["sex"],
        test_comps["site"],
        test_comps["agebin"],
        test_comps["sex_site"],
        test_comps["site_age"],
        test_comps["sex_age"],
        test_comps["pcount"],
        test_comps["page"],
        test_comps["dx"],
    ]
)

test_mat_z = (test_mat - oof_mu) / oof_sd

meta_logit = (w[0] + test_mat_z @ w[1:]).astype(float)
meta_pred = _sigmoid(meta_logit).astype(float)

meta_df = pd.DataFrame({"image_name": test["image_name"].values, "target": meta_pred})
meta_df = sample[["image_name"]].merge(meta_df, on="image_name", how="left")
meta_df["target"] = meta_df["target"].fillna(global_mean).astype(float)




## === cell 3
def _has_valid_preds(df: pd.DataFrame) -> bool:
    return df["target"].notna().any()


pred = meta_df["target"].to_numpy(dtype=float)

if _has_valid_preds(a):
    a_pred = a["target"].fillna(global_mean).to_numpy(dtype=float)
    pred = 0.80 * pred + 0.20 * a_pred

if _has_valid_preds(c):
    c_pred = c["target"].fillna(global_mean).to_numpy(dtype=float)
    pred = 0.70 * pred + 0.30 * c_pred

if _has_valid_preds(b):
    b_pred = b["target"].fillna(global_mean).to_numpy(dtype=float)
    pred = 0.85 * pred + 0.15 * b_pred

sub = pd.DataFrame(
    {"image_name": sample["image_name"].values, "target": np.clip(pred, 0.0, 1.0)}
)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Prediction summary:", sub["target"].describe().to_dict())
print(
    "Learned blend weights (intercept + standardized comps in logit-space):", w.tolist()
)
