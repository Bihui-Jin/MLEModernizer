# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def find_existing_path(rel_path: str):
    for base in DATA_DIR_CANDIDATES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None


train_path = find_existing_path("train.csv")
test_path = find_existing_path("test.csv")
sample_path = find_existing_path("sample_submission.csv")

if test_path is None or sample_path is None:
    raise FileNotFoundError(
        "Could not locate test.csv and/or sample_submission.csv in known Kaggle directories."
    )

train_df = pd.read_csv(train_path) if train_path is not None else None
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

assert "image_name" in test_df.columns, "test.csv must contain image_name"
assert list(sample_sub.columns) == [
    "image_name",
    "target",
], "Unexpected sample_submission.csv columns"



## === cell 1
if train_df is not None and "target" in train_df.columns:
    df_tr = train_df.copy()

    def _norm_cat(s: pd.Series) -> pd.Series:
        s = s.astype("string")
        s = s.fillna("unknown")
        s = s.str.strip()
        s = s.mask(s == "", "unknown")
        return s

    cat_cols_base = ["sex", "anatom_site_general_challenge", "patient_id"]
    for c in cat_cols_base:
        if c in df_tr.columns:
            df_tr[c] = _norm_cat(df_tr[c])

    if "diagnosis" in df_tr.columns:
        df_tr["diagnosis"] = _norm_cat(df_tr["diagnosis"])

    if "sex" in df_tr.columns and "anatom_site_general_challenge" in df_tr.columns:
        df_tr["sex_site"] = df_tr["sex"] + "||" + df_tr["anatom_site_general_challenge"]
    else:
        df_tr["sex_site"] = "unknown||unknown"

    if (
        "patient_id" in df_tr.columns
        and "anatom_site_general_challenge" in df_tr.columns
    ):
        df_tr["pid_site"] = (
            df_tr["patient_id"] + "||" + df_tr["anatom_site_general_challenge"]
        )
    else:
        df_tr["pid_site"] = "unknown||unknown"

    prior = float(np.clip(float(df_tr["target"].mean()), 1e-6, 1 - 1e-6))

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def make_oof_smoothed_rate(
        series: pd.Series,
        y: pd.Series,
        prior: float,
        m: float,
        n_splits: int = 5,
        seed: int = 42,
        group: pd.Series = None,
    ):
        series = series.astype("string").fillna("unknown")
        n = len(series)
        oof = np.empty(n, dtype="float64")

        rng = np.random.RandomState(seed)
        if group is None:
            fold_id = rng.randint(0, n_splits, size=n)
        else:
            group = group.astype("string").fillna("unknown")
            uniq = pd.Index(group.unique())
            perm = rng.permutation(len(uniq))
            fold_per_uniq = perm % n_splits
            fold_map = pd.Series(fold_per_uniq, index=uniq)
            fold_id = fold_map.loc[group].to_numpy(dtype=np.int16)

        for f in range(n_splits):
            tr_mask = fold_id != f
            te_mask = ~tr_mask

            stats = (
                y[tr_mask]
                .groupby(series[tr_mask])
                .agg(["sum", "count"])
                .rename(columns={"sum": "pos", "count": "cnt"})
            )
            rate = (stats["pos"] + m * prior) / (stats["cnt"] + m)
            mapped = series[te_mask].map(rate).astype("float64").fillna(prior).values
            oof[te_mask] = mapped

        return pd.Series(oof, index=series.index)

    def smoothed_map(series: pd.Series, y: pd.Series, prior: float, m: float):
        series = series.astype("string").fillna("unknown")
        stats = (
            y.groupby(series)
            .agg(["sum", "count"])
            .rename(columns={"sum": "pos", "count": "cnt"})
        )
        stats["rate"] = (stats["pos"] + m * prior) / (stats["cnt"] + m)
        return stats["rate"].to_dict()

    age_tr_num = pd.to_numeric(
        df_tr.get("age_approx", pd.Series([np.nan] * len(df_tr))), errors="coerce"
    )
    age_mean = (
        float(np.nanmean(age_tr_num)) if np.isfinite(np.nanmean(age_tr_num)) else 50.0
    )
    age_std = (
        float(np.nanstd(age_tr_num))
        if np.isfinite(np.nanstd(age_tr_num)) and float(np.nanstd(age_tr_num)) > 0
        else 10.0
    )

    age_fill = age_tr_num.fillna(age_mean)
    try:
        age_bin_tr = pd.qcut(age_fill, q=8, duplicates="drop")
        age_bin_edges = age_bin_tr.cat.categories
        age_bin_labels = list(range(len(age_bin_edges)))
        age_bin_tr = age_bin_tr.cat.rename_categories(age_bin_labels).astype(int)
        has_age_bins = True
    except Exception:
        age_bin_tr = pd.Series([0] * len(df_tr), index=df_tr.index)
        age_bin_edges = None
        has_age_bins = False

    if has_age_bins and "anatom_site_general_challenge" in df_tr.columns:
        df_tr["site_agebin"] = (
            df_tr["anatom_site_general_challenge"] + "||" + age_bin_tr.astype("string")
        )
    else:
        df_tr["site_agebin"] = "unknown||0"

    group_pid = df_tr["patient_id"] if "patient_id" in df_tr.columns else None

    oof_sex = (
        make_oof_smoothed_rate(
            df_tr["sex"], df_tr["target"], prior, m=200.0, group=group_pid
        )
        if "sex" in df_tr.columns
        else pd.Series([prior] * len(df_tr), index=df_tr.index, dtype="float64")
    )
    oof_site = (
        make_oof_smoothed_rate(
            df_tr["anatom_site_general_challenge"],
            df_tr["target"],
            prior,
            m=50.0,
            group=group_pid,
        )
        if "anatom_site_general_challenge" in df_tr.columns
        else pd.Series([prior] * len(df_tr), index=df_tr.index, dtype="float64")
    )
    oof_sexsite = make_oof_smoothed_rate(
        df_tr["sex_site"], df_tr["target"], prior, m=400.0, group=group_pid
    )

    oof_pid = (
        make_oof_smoothed_rate(
            df_tr["patient_id"], df_tr["target"], prior, m=800.0, group=None
        )
        if "patient_id" in df_tr.columns
        else pd.Series([prior] * len(df_tr), index=df_tr.index, dtype="float64")
    )

    oof_diag = (
        make_oof_smoothed_rate(
            df_tr["diagnosis"], df_tr["target"], prior, m=300.0, group=group_pid
        )
        if "diagnosis" in df_tr.columns
        else pd.Series([prior] * len(df_tr), index=df_tr.index, dtype="float64")
    )
    oof_agebin = (
        make_oof_smoothed_rate(
            age_bin_tr.astype("string"),
            df_tr["target"],
            prior,
            m=100.0,
            group=group_pid,
        )
        if has_age_bins
        else pd.Series([prior] * len(df_tr), index=df_tr.index, dtype="float64")
    )

    oof_siteage = (
        make_oof_smoothed_rate(
            df_tr["site_agebin"], df_tr["target"], prior, m=120.0, group=group_pid
        )
        if (has_age_bins and "anatom_site_general_challenge" in df_tr.columns)
        else pd.Series([prior] * len(df_tr), index=df_tr.index, dtype="float64")
    )

    oof_pidsite = (
        make_oof_smoothed_rate(
            df_tr["pid_site"], df_tr["target"], prior, m=500.0, group=group_pid
        )
        if ("pid_site" in df_tr.columns)
        else pd.Series([prior] * len(df_tr), index=df_tr.index, dtype="float64")
    )

    sex_map_full = (
        smoothed_map(df_tr["sex"], df_tr["target"], prior, m=200.0)
        if "sex" in df_tr.columns
        else {}
    )
    site_map_full = (
        smoothed_map(
            df_tr["anatom_site_general_challenge"], df_tr["target"], prior, m=50.0
        )
        if "anatom_site_general_challenge" in df_tr.columns
        else {}
    )
    sexsite_map_full = smoothed_map(df_tr["sex_site"], df_tr["target"], prior, m=400.0)
    pid_map_full = (
        smoothed_map(df_tr["patient_id"], df_tr["target"], prior, m=800.0)
        if "patient_id" in df_tr.columns
        else {}
    )
    diag_map_full = (
        smoothed_map(df_tr["diagnosis"], df_tr["target"], prior, m=300.0)
        if "diagnosis" in df_tr.columns
        else {}
    )

    siteage_map_full = (
        smoothed_map(df_tr["site_agebin"], df_tr["target"], prior, m=120.0)
        if (has_age_bins and "anatom_site_general_challenge" in df_tr.columns)
        else {}
    )

    pidsite_map_full = (
        smoothed_map(df_tr["pid_site"], df_tr["target"], prior, m=500.0)
        if ("pid_site" in df_tr.columns)
        else {}
    )

    if has_age_bins:
        agebin_map_full = smoothed_map(
            age_bin_tr.astype("string"), df_tr["target"], prior, m=100.0
        )
        low_bin = int(age_bin_tr.min())
        high_bin = int(age_bin_tr.max())
        p_low = float(np.clip(agebin_map_full.get(str(low_bin), prior), 1e-6, 1 - 1e-6))
        p_high = float(
            np.clip(agebin_map_full.get(str(high_bin), prior), 1e-6, 1 - 1e-6)
        )
        age_slope = float(
            np.clip(
                (logit(p_high) - logit(p_low)) / max(1.0, (high_bin - low_bin)),
                -0.35,
                0.35,
            )
        )
    else:
        agebin_map_full = {}
        age_slope = 0.0

    base = logit(prior)
    age_z_tr = (age_fill - age_mean) / age_std

    y_tr = df_tr["target"].values.astype(np.int8)

    d_sex = logit(oof_sex.values) - base
    d_site = logit(oof_site.values) - base
    d_sexsite = logit(oof_sexsite.values) - base
    d_agebin = logit(oof_agebin.values) - base
    d_pid = logit(oof_pid.values) - base
    d_diag = logit(oof_diag.values) - base
    d_agez = age_slope * age_z_tr.values
    d_siteage = logit(oof_siteage.values) - base

    d_pidsite = logit(oof_pidsite.values) - base

    def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.int8)
        y_score = np.asarray(y_score, dtype=np.float64)

        n_pos = int(y_true.sum())
        n = y_true.size
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        order = np.argsort(y_score, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)

        s_sorted = y_score[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and s_sorted[j] == s_sorted[i]:
                j += 1
            if j - i > 1:
                avg = (i + 1 + j) / 2.0
                ranks[order[i:j]] = avg
            i = j

        sum_ranks_pos = float(ranks[y_true == 1].sum())
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    w_sex_grid = np.array([0.45, 0.60, 0.75], dtype=np.float64)
    w_site_grid = np.array([0.45, 0.60, 0.75], dtype=np.float64)
    w_sexsite_grid = np.array([0.35, 0.55, 0.75], dtype=np.float64)
    w_agebin_grid = np.array([0.65, 0.90, 1.15], dtype=np.float64)
    w_pid_grid = np.array([0.25, 0.35, 0.50], dtype=np.float64)
    w_diag_grid = np.array([0.00, 0.05, 0.10], dtype=np.float64)
    w_agez_grid = np.array([0.15, 0.25, 0.35], dtype=np.float64)
    w_siteage_grid = np.array([0.15, 0.30, 0.45], dtype=np.float64)

    w_pidsite_grid = np.array([0.00, 0.15, 0.30], dtype=np.float64)

    scale_grid = np.array([0.8, 1.0, 1.2], dtype=np.float64)

    best_auc = -1.0
    best_params = (0.60, 0.60, 0.55, 0.90, 0.35, 0.05, 0.25, 0.30, 0.15, 1.0)

    for w_sex in w_sex_grid:
        for w_site in w_site_grid:
            for w_sexsite in w_sexsite_grid:
                for w_agebin in w_agebin_grid:
                    for w_pid in w_pid_grid:
                        for w_diag in w_diag_grid:
                            for w_agez in w_agez_grid:
                                for w_siteage in w_siteage_grid:
                                    for w_pidsite in w_pidsite_grid:
                                        score_oof = (
                                            base
                                            + w_sex * d_sex
                                            + w_site * d_site
                                            + w_sexsite * d_sexsite
                                            + w_agebin * d_agebin
                                            + w_pid * d_pid
                                            + w_diag * d_diag
                                            + w_agez * d_agez
                                            + w_siteage * d_siteage
                                            + w_pidsite * d_pidsite
                                        )
                                        for s in scale_grid:
                                            auc_s = fast_auc(
                                                y_tr, sigmoid(s * score_oof)
                                            )
                                            if auc_s > best_auc:
                                                best_auc = auc_s
                                                best_params = (
                                                    float(w_sex),
                                                    float(w_site),
                                                    float(w_sexsite),
                                                    float(w_agebin),
                                                    float(w_pid),
                                                    float(w_diag),
                                                    float(w_agez),
                                                    float(w_siteage),
                                                    float(w_pidsite),
                                                    float(s),
                                                )

    (
        w_sex,
        w_site,
        w_sexsite,
        w_agebin,
        w_pid,
        w_diag,
        w_agez,
        w_siteage,
        w_pidsite,
        best_scale,
    ) = best_params

else:
    prior = 0.5
    age_mean, age_std, age_slope = 50.0, 10.0, 0.0
    age_bin_edges = None
    has_age_bins = False
    sex_map_full = {}
    site_map_full = {}
    sexsite_map_full = {}
    pid_map_full = {}
    diag_map_full = {}
    agebin_map_full = {}
    siteage_map_full = {}
    pidsite_map_full = {}
    w_sex, w_site, w_sexsite, w_agebin, w_pid, w_diag, w_agez, w_siteage, w_pidsite = (
        0.60,
        0.60,
        0.55,
        0.90,
        0.35,
        0.05,
        0.25,
        0.30,
        0.15,
    )
    best_scale = 1.0



## === cell 2
submission = pd.DataFrame({"image_name": test_df["image_name"].values})

if train_df is None or "target" not in train_df.columns:
    submission["target"] = float(prior)
else:
    df_te = test_df.copy()

    def _norm_cat(s: pd.Series) -> pd.Series:
        s = s.astype("string")
        s = s.fillna("unknown")
        s = s.str.strip()
        s = s.mask(s == "", "unknown")
        return s

    for c in ["sex", "anatom_site_general_challenge", "patient_id"]:
        if c in df_te.columns:
            df_te[c] = _norm_cat(df_te[c])

    if "diagnosis" not in df_te.columns:
        df_te["diagnosis"] = "unknown"
    df_te["diagnosis"] = _norm_cat(df_te["diagnosis"])
    diag_missing_or_constant = bool(df_te["diagnosis"].nunique(dropna=False) <= 1)

    if "sex" in df_te.columns and "anatom_site_general_challenge" in df_te.columns:
        df_te["sex_site"] = df_te["sex"] + "||" + df_te["anatom_site_general_challenge"]
    else:
        df_te["sex_site"] = "unknown||unknown"

    if (
        "patient_id" in df_te.columns
        and "anatom_site_general_challenge" in df_te.columns
    ):
        df_te["pid_site"] = (
            df_te["patient_id"] + "||" + df_te["anatom_site_general_challenge"]
        )
    else:
        df_te["pid_site"] = "unknown||unknown"

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    base = logit(prior)

    sex_p = (
        df_te["sex"].map(sex_map_full).astype(float).fillna(prior)
        if "sex" in df_te.columns
        else prior
    )
    site_p = (
        df_te["anatom_site_general_challenge"]
        .map(site_map_full)
        .astype(float)
        .fillna(prior)
        if "anatom_site_general_challenge" in df_te.columns
        else prior
    )
    sexsite_p = df_te["sex_site"].map(sexsite_map_full).astype(float).fillna(prior)

    if "patient_id" in df_te.columns and len(pid_map_full) > 0:
        pid_seen_rate = float(
            df_te["patient_id"].isin(pd.Index(list(pid_map_full.keys()))).mean()
        )
    else:
        pid_seen_rate = 0.0
    pid_disabled_test = bool(pid_seen_rate < 0.05)

    pid_p = (
        df_te["patient_id"].map(pid_map_full).astype(float).fillna(prior)
        if ("patient_id" in df_te.columns and not pid_disabled_test)
        else prior
    )

    diag_p = df_te["diagnosis"].map(diag_map_full).astype(float).fillna(prior)

    if "pid_site" in df_te.columns and len(pidsite_map_full) > 0:
        pidsite_seen_rate = float(
            df_te["pid_site"].isin(pd.Index(list(pidsite_map_full.keys()))).mean()
        )
    else:
        pidsite_seen_rate = 0.0
    pidsite_disabled_test = bool(pidsite_seen_rate < 0.05)

    pidsite_p = (
        df_te["pid_site"].map(pidsite_map_full).astype(float).fillna(prior)
        if ("pid_site" in df_te.columns and not pidsite_disabled_test)
        else prior
    )

    age_te = pd.to_numeric(
        df_te.get("age_approx", pd.Series([np.nan] * len(df_te))), errors="coerce"
    ).fillna(age_mean)
    age_z = (age_te - age_mean) / age_std

    w_diag_eff = 0.0 if diag_missing_or_constant else float(w_diag)
    w_pid_eff = 0.0 if pid_disabled_test else float(w_pid)
    w_pidsite_eff = 0.0 if pidsite_disabled_test else float(w_pidsite)

    if (
        age_bin_edges is not None
        and len(age_bin_edges) > 0
        and len(agebin_map_full) > 0
    ):
        right_edges = [iv.right for iv in age_bin_edges]
        bin_edges = [-np.inf] + right_edges
        age_bin_te = pd.cut(age_te, bins=bin_edges, labels=False, include_lowest=True)
        age_bin_te = age_bin_te.fillna(0).astype(int)
        agebin_p = (
            age_bin_te.astype("string").map(agebin_map_full).astype(float).fillna(prior)
        )

        if (
            "anatom_site_general_challenge" in df_te.columns
            and len(siteage_map_full) > 0
        ):
            site_agebin_te = (
                df_te["anatom_site_general_challenge"]
                + "||"
                + age_bin_te.astype("string")
            )
            siteage_p = site_agebin_te.map(siteage_map_full).astype(float).fillna(prior)
        else:
            siteage_p = prior

        score = (
            base
            + float(w_sex) * (logit(sex_p) - base)
            + float(w_site) * (logit(site_p) - base)
            + float(w_sexsite) * (logit(sexsite_p) - base)
            + float(w_agebin) * (logit(agebin_p) - base)
            + float(w_pid_eff) * (logit(pid_p) - base)
            + float(w_diag_eff) * (logit(diag_p) - base)
            + float(w_agez) * age_slope * age_z.values
            + float(w_siteage) * (logit(siteage_p) - base)
            + float(w_pidsite_eff) * (logit(pidsite_p) - base)
        )
    else:
        score = (
            base
            + float(w_sex) * (logit(sex_p) - base)
            + float(w_site) * (logit(site_p) - base)
            + float(w_sexsite) * (logit(sexsite_p) - base)
            + float(w_pid_eff) * (logit(pid_p) - base)
            + float(w_diag_eff) * (logit(diag_p) - base)
            + float(w_agez) * age_slope * age_z.values
            + float(w_pidsite_eff) * (logit(pidsite_p) - base)
        )

    pred = sigmoid(float(best_scale) * score)
    submission["target"] = np.clip(pred.astype(float), 1e-6, 1 - 1e-6)

if submission.shape[0] != test_df.shape[0]:
    raise ValueError("Submission row count does not match test row count.")

if not np.array_equal(submission["image_name"].values, test_df["image_name"].values):
    submission = (
        submission.set_index("image_name").reindex(test_df["image_name"]).reset_index()
    )



## === cell 3
submission.to_csv("submission.csv", index=False, float_format="%.6f")

print("Wrote submission.csv")
print(submission.head())
print("prior used:", float(prior))
print("best_scale (OOF-fit):", float(best_scale))
print(
    "weights:",
    {
        "w_sex": float(w_sex),
        "w_site": float(w_site),
        "w_sexsite": float(w_sexsite),
        "w_agebin": float(w_agebin),
        "w_pid": float(w_pid),
        "w_diag": float(w_diag),
        "w_agez": float(w_agez),
        "w_siteage": float(w_siteage),
        "w_pidsite": float(w_pidsite),
        "w_pid_eff_test": float(
            0.0 if ("pid_disabled_test" in globals() and pid_disabled_test) else w_pid
        ),
        "w_pidsite_eff_test": float(
            0.0
            if ("pidsite_disabled_test" in globals() and pidsite_disabled_test)
            else w_pidsite
        ),
        "w_diag_eff_test": float(
            0.0
            if ("diag_missing_or_constant" in globals() and diag_missing_or_constant)
            else w_diag
        ),
    },
)
print(
    "pid_seen_rate_test:",
    float(pid_seen_rate) if "pid_seen_rate" in globals() else None,
    "pid_disabled_test:",
    bool(pid_disabled_test) if "pid_disabled_test" in globals() else None,
)
print(
    "pidsite_seen_rate_test:",
    float(pidsite_seen_rate) if "pidsite_seen_rate" in globals() else None,
    "pidsite_disabled_test:",
    bool(pidsite_disabled_test) if "pidsite_disabled_test" in globals() else None,
)
print(
    "target min/max:",
    float(submission["target"].min()),
    float(submission["target"].max()),
)
