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

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_DIR_CANDIDATES:
        nested = os.path.join(base, "siim-isic-melanoma-classification", filename)
        if os.path.exists(nested):
            return nested
    raise FileNotFoundError(
        f"Could not find {filename} in known Kaggle input locations: {DATA_DIR_CANDIDATES}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "target" in train.columns, "train.csv must contain 'target'"
assert (
    "image_name" in test.columns and "image_name" in sample_sub.columns
), "Missing 'image_name' column"
assert list(sample_sub.columns) == [
    "image_name",
    "target",
], "sample_submission.csv must have columns: image_name,target"



## === cell 1
feature_cols = ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]

X = train[feature_cols].copy()
y = train["target"].astype(int).values
groups = (
    train["patient_id"].values
    if "patient_id" in train.columns
    else np.arange(len(train))
)

X_test = test[feature_cols].copy()

for df_ in (X, X_test):
    for col in ["sex", "anatom_site_general_challenge", "patient_id"]:
        df_.loc[df_[col].eq(""), col] = np.nan

X["anatom_missing"] = X["anatom_site_general_challenge"].isna().astype(float)
X_test["anatom_missing"] = X_test["anatom_site_general_challenge"].isna().astype(float)

X["age_missing"] = X["age_approx"].isna().astype(float)
X_test["age_missing"] = X_test["age_approx"].isna().astype(float)

patient_counts = train["patient_id"].value_counts(dropna=False)
X["patient_lesion_count"] = train["patient_id"].map(patient_counts).astype(float)
X_test["patient_lesion_count"] = test["patient_id"].map(patient_counts).astype(float)
X_test["patient_lesion_count"] = X_test["patient_lesion_count"].fillna(1.0)

categorical_features = ["sex", "anatom_site_general_challenge"]


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _make_age_bin(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    b = (np.floor(s / 5.0) * 5.0).astype("float")
    return b.astype("Int64").astype("string")


X["age_bin"] = _make_age_bin(X["age_approx"])
X_test["age_bin"] = _make_age_bin(X_test["age_approx"])



## === cell 2
gkf = GroupKFold(n_splits=5)

test_pred_accum = np.zeros(len(test), dtype=float)
oof_pred = np.zeros(len(train), dtype=float)

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
ALPHA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
SITE_ALPHA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]

SEX_ALPHA_GRID = [0.5, 2.0, 10.0]
AGEBIN_ALPHA_GRID = [0.5, 2.0, 10.0]

fold_aucs = []

test_sites_all = test["anatom_site_general_challenge"].replace("", np.nan)
test_sex_all = test["sex"].replace("", np.nan)
test_pid_all = test["patient_id"]
test_agebin_all = X_test["age_bin"]

numeric_features_fold = [
    "age_approx",
    "patient_lesion_count",
    "patient_lesion_count_fold",
    "patient_lesion_count_log_fold",
    "age_missing",
    "anatom_missing",
    "patient_age_mean_fold",
    "patient_male_rate_fold",
    "patient_target_mean",
    "patient_target_logit",
    "site_target_mean",
    "site_target_logit",
    "sex_target_mean",
    "sex_target_logit",
    "agebin_target_mean",
    "agebin_target_logit",
]

preprocess_fold = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features_fold,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)


def _build_features_fast(
    X_base_local: pd.DataFrame,
    pids_local: pd.Series,
    sites_local: pd.Series,
    sex_local: pd.Series,
    agebin_local: pd.Series,
    smoothed_patient: pd.Series,
    smoothed_site: pd.Series,
    smoothed_sex: pd.Series,
    smoothed_agebin: pd.Series,
    global_mean_local: float,
) -> pd.DataFrame:
    X_out = X_base_local.copy()

    ptm = pids_local.map(smoothed_patient).to_numpy(dtype=float, na_value=np.nan)
    stm = sites_local.map(smoothed_site).to_numpy(dtype=float, na_value=np.nan)
    sxm = sex_local.map(smoothed_sex).to_numpy(dtype=float, na_value=np.nan)
    abm = agebin_local.map(smoothed_agebin).to_numpy(dtype=float, na_value=np.nan)

    ptm = np.where(np.isnan(ptm), global_mean_local, ptm)
    stm = np.where(np.isnan(stm), global_mean_local, stm)
    sxm = np.where(np.isnan(sxm), global_mean_local, sxm)
    abm = np.where(np.isnan(abm), global_mean_local, abm)

    X_out["patient_target_mean"] = ptm
    X_out["patient_target_logit"] = np.log(_clip_prob(ptm) / (1.0 - _clip_prob(ptm)))

    X_out["site_target_mean"] = stm
    X_out["site_target_logit"] = np.log(_clip_prob(stm) / (1.0 - _clip_prob(stm)))

    X_out["sex_target_mean"] = sxm
    X_out["sex_target_logit"] = np.log(_clip_prob(sxm) / (1.0 - _clip_prob(sxm)))

    X_out["agebin_target_mean"] = abm
    X_out["agebin_target_logit"] = np.log(_clip_prob(abm) / (1.0 - _clip_prob(abm)))

    return X_out


for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_tr_base = X.iloc[tr_idx].copy()
    X_va_base = X.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    tr_pids = train.iloc[tr_idx]["patient_id"]
    va_pids = train.iloc[va_idx]["patient_id"]

    tr_patient_counts = tr_pids.value_counts(dropna=False)
    X_tr_base["patient_lesion_count_fold"] = tr_pids.map(tr_patient_counts).astype(
        float
    )
    X_va_base["patient_lesion_count_fold"] = va_pids.map(tr_patient_counts).astype(
        float
    )

    X_test_fold = X_test.copy()
    X_test_fold["patient_lesion_count_fold"] = test_pid_all.map(
        tr_patient_counts
    ).astype(float)

    for df_ in (X_tr_base, X_va_base, X_test_fold):
        df_["patient_lesion_count_fold"] = df_["patient_lesion_count_fold"].fillna(1.0)
        df_["patient_lesion_count_log_fold"] = np.log1p(
            df_["patient_lesion_count_fold"].to_numpy()
        )

    global_mean_outer = float(np.mean(y_tr))

    tr_sites_outer = train.iloc[tr_idx]["anatom_site_general_challenge"].replace(
        "", np.nan
    )
    va_sites_outer = train.iloc[va_idx]["anatom_site_general_challenge"].replace(
        "", np.nan
    )

    patient_stats_outer = (
        pd.DataFrame({"patient_id": tr_pids.values, "y": y_tr})
        .groupby("patient_id")["y"]
        .agg(["mean", "count"])
    )
    site_stats_outer = (
        pd.DataFrame({"site": tr_sites_outer.values, "y": y_tr})
        .groupby("site")["y"]
        .agg(["mean", "count"])
    )

    tr_age_outer = pd.to_numeric(train.iloc[tr_idx]["age_approx"], errors="coerce")
    tr_sex_outer = train.iloc[tr_idx]["sex"].replace("", np.nan)

    pid_age_mean_outer = (
        pd.DataFrame({"patient_id": tr_pids.values, "age": tr_age_outer.values})
        .groupby("patient_id")["age"]
        .mean()
    )
    _tmp_med = np.nanmedian(tr_age_outer.values)
    global_age_median_outer = (
        float(_tmp_med)
        if np.isfinite(_tmp_med)
        else float(
            np.nanmedian(pd.to_numeric(train["age_approx"], errors="coerce").values)
        )
    )

    sex_bin_tr_outer = tr_sex_outer.map({"male": 1.0, "female": 0.0})
    pid_male_rate_outer = (
        pd.DataFrame({"patient_id": tr_pids.values, "male": sex_bin_tr_outer.values})
        .groupby("patient_id")["male"]
        .mean()
    )
    _tmp_male = np.nanmean(sex_bin_tr_outer.values)
    global_male_rate_outer = float(_tmp_male) if np.isfinite(_tmp_male) else 0.5

    X_tr_base["patient_age_mean_fold"] = (
        tr_pids.map(pid_age_mean_outer).astype(float).fillna(global_age_median_outer)
    )
    X_va_base["patient_age_mean_fold"] = (
        va_pids.map(pid_age_mean_outer).astype(float).fillna(global_age_median_outer)
    )
    X_test_fold["patient_age_mean_fold"] = (
        test_pid_all.map(pid_age_mean_outer)
        .astype(float)
        .fillna(global_age_median_outer)
    )

    X_tr_base["patient_male_rate_fold"] = (
        tr_pids.map(pid_male_rate_outer).astype(float).fillna(global_male_rate_outer)
    )
    X_va_base["patient_male_rate_fold"] = (
        va_pids.map(pid_male_rate_outer).astype(float).fillna(global_male_rate_outer)
    )
    X_test_fold["patient_male_rate_fold"] = (
        test_pid_all.map(pid_male_rate_outer)
        .astype(float)
        .fillna(global_male_rate_outer)
    )

    inner_splits = 3
    unique_groups = pd.Series(tr_pids.values).nunique(dropna=False)
    inner_n_splits = min(inner_splits, int(unique_groups)) if unique_groups >= 2 else 2
    inner_n_splits = max(2, inner_n_splits)
    inner_gkf = GroupKFold(n_splits=inner_n_splits)

    inner_cache = []
    outer_train_block = train.iloc[tr_idx]  # view for quicker repeated access

    for inner_tr_rel, inner_va_rel in inner_gkf.split(
        X_tr_base, y_tr, groups=tr_pids.values
    ):
        y_inner_tr = y_tr[inner_tr_rel]
        y_inner_va = y_tr[inner_va_rel]

        pids_inner_tr = tr_pids.iloc[inner_tr_rel]
        pids_inner_va = tr_pids.iloc[inner_va_rel]

        sites_inner_tr = tr_sites_outer.iloc[inner_tr_rel]
        sites_inner_va = tr_sites_outer.iloc[inner_va_rel]

        sex_inner_tr = outer_train_block.iloc[inner_tr_rel]["sex"].replace("", np.nan)
        sex_inner_va = outer_train_block.iloc[inner_va_rel]["sex"].replace("", np.nan)

        agebin_inner_tr = X_tr_base.iloc[inner_tr_rel]["age_bin"]
        agebin_inner_va = X_tr_base.iloc[inner_va_rel]["age_bin"]

        inner_pid_counts = pids_inner_tr.value_counts(dropna=False)

        X_inner_tr_base = X_tr_base.iloc[inner_tr_rel].copy()
        X_inner_va_base = X_tr_base.iloc[inner_va_rel].copy()

        X_inner_tr_base["patient_lesion_count_fold"] = (
            pids_inner_tr.map(inner_pid_counts).astype(float).fillna(1.0)
        )
        X_inner_va_base["patient_lesion_count_fold"] = (
            pids_inner_va.map(inner_pid_counts).astype(float).fillna(1.0)
        )
        X_inner_tr_base["patient_lesion_count_log_fold"] = np.log1p(
            X_inner_tr_base["patient_lesion_count_fold"].to_numpy()
        )
        X_inner_va_base["patient_lesion_count_log_fold"] = np.log1p(
            X_inner_va_base["patient_lesion_count_fold"].to_numpy()
        )

        global_mean_inner = float(np.mean(y_inner_tr))

        patient_stats_inner = (
            pd.DataFrame({"patient_id": pids_inner_tr.values, "y": y_inner_tr})
            .groupby("patient_id")["y"]
            .agg(["mean", "count"])
        )
        site_stats_inner = (
            pd.DataFrame({"site": sites_inner_tr.values, "y": y_inner_tr})
            .groupby("site")["y"]
            .agg(["mean", "count"])
        )
        sex_stats_inner = (
            pd.DataFrame({"sex": sex_inner_tr.values, "y": y_inner_tr})
            .groupby("sex")["y"]
            .agg(["mean", "count"])
        )
        agebin_stats_inner = (
            pd.DataFrame({"agebin": agebin_inner_tr.values, "y": y_inner_tr})
            .groupby("agebin")["y"]
            .agg(["mean", "count"])
        )

        inner_cache.append(
            dict(
                X_tr_base=X_inner_tr_base,
                X_va_base=X_inner_va_base,
                y_tr=y_inner_tr,
                y_va=y_inner_va,
                pids_tr=pids_inner_tr,
                pids_va=pids_inner_va,
                sites_tr=sites_inner_tr,
                sites_va=sites_inner_va,
                sex_tr=sex_inner_tr,
                sex_va=sex_inner_va,
                agebin_tr=agebin_inner_tr,
                agebin_va=agebin_inner_va,
                global_mean=global_mean_inner,
                patient_stats=patient_stats_inner,
                site_stats=site_stats_inner,
                sex_stats=sex_stats_inner,
                agebin_stats=agebin_stats_inner,
            )
        )

    best_params = None
    best_inner_auc = -1.0

    for alpha in ALPHA_GRID:
        for site_alpha in SITE_ALPHA_GRID:
            for sex_alpha in SEX_ALPHA_GRID:
                for agebin_alpha in AGEBIN_ALPHA_GRID:
                    split_mats = []
                    split_y = []
                    for pack in inner_cache:
                        ps = pack["patient_stats"]
                        ss = pack["site_stats"]
                        sxs = pack["sex_stats"]
                        abs_ = pack["agebin_stats"]
                        gm = pack["global_mean"]

                        smoothed_patient = (ps["mean"] * ps["count"] + gm * alpha) / (
                            ps["count"] + alpha
                        )
                        smoothed_site = (ss["mean"] * ss["count"] + gm * site_alpha) / (
                            ss["count"] + site_alpha
                        )
                        smoothed_sex = (sxs["mean"] * sxs["count"] + gm * sex_alpha) / (
                            sxs["count"] + sex_alpha
                        )
                        smoothed_agebin = (
                            abs_["mean"] * abs_["count"] + gm * agebin_alpha
                        ) / (abs_["count"] + agebin_alpha)

                        X_inner_tr = _build_features_fast(
                            pack["X_tr_base"],
                            pack["pids_tr"],
                            pack["sites_tr"],
                            pack["sex_tr"],
                            pack["agebin_tr"],
                            smoothed_patient,
                            smoothed_site,
                            smoothed_sex,
                            smoothed_agebin,
                            gm,
                        )
                        X_inner_va = _build_features_fast(
                            pack["X_va_base"],
                            pack["pids_va"],
                            pack["sites_va"],
                            pack["sex_va"],
                            pack["agebin_va"],
                            smoothed_patient,
                            smoothed_site,
                            smoothed_sex,
                            smoothed_agebin,
                            gm,
                        )

                        Xt_tr = preprocess_fold.fit_transform(X_inner_tr, pack["y_tr"])
                        Xt_va = preprocess_fold.transform(X_inner_va)
                        split_mats.append((Xt_tr, Xt_va))
                        split_y.append((pack["y_tr"], pack["y_va"]))

                    for C in C_GRID:
                        inner_aucs = []
                        for (Xt_tr, Xt_va), (y_inner_tr, y_inner_va) in zip(
                            split_mats, split_y
                        ):
                            clf = LogisticRegression(
                                solver="lbfgs",
                                max_iter=2000,
                                C=C,
                                random_state=RANDOM_STATE,
                            )
                            clf.fit(Xt_tr, y_inner_tr)
                            pred_inner = clf.predict_proba(Xt_va)[:, 1]
                            inner_aucs.append(roc_auc_score(y_inner_va, pred_inner))

                        mean_inner_auc = (
                            float(np.mean(inner_aucs)) if inner_aucs else -1.0
                        )
                        if mean_inner_auc > best_inner_auc:
                            best_inner_auc = mean_inner_auc
                            best_params = (
                                alpha,
                                site_alpha,
                                sex_alpha,
                                agebin_alpha,
                                C,
                            )

    best_alpha, best_site_alpha, best_sex_alpha, best_agebin_alpha, best_C = best_params

    tr_sex_outer_full = train.iloc[tr_idx]["sex"].replace("", np.nan)
    va_sex_outer_full = train.iloc[va_idx]["sex"].replace("", np.nan)

    tr_agebin_outer_full = X_tr_base["age_bin"]
    va_agebin_outer_full = X_va_base["age_bin"]
    te_agebin_outer_full = X_test_fold["age_bin"]

    sex_stats_outer = (
        pd.DataFrame({"sex": tr_sex_outer_full.values, "y": y_tr})
        .groupby("sex")["y"]
        .agg(["mean", "count"])
    )
    agebin_stats_outer = (
        pd.DataFrame({"agebin": tr_agebin_outer_full.values, "y": y_tr})
        .groupby("agebin")["y"]
        .agg(["mean", "count"])
    )

    smoothed_patient_outer = (
        patient_stats_outer["mean"] * patient_stats_outer["count"]
        + global_mean_outer * best_alpha
    ) / (patient_stats_outer["count"] + best_alpha)
    smoothed_site_outer = (
        site_stats_outer["mean"] * site_stats_outer["count"]
        + global_mean_outer * best_site_alpha
    ) / (site_stats_outer["count"] + best_site_alpha)
    smoothed_sex_outer = (
        sex_stats_outer["mean"] * sex_stats_outer["count"]
        + global_mean_outer * best_sex_alpha
    ) / (sex_stats_outer["count"] + best_sex_alpha)
    smoothed_agebin_outer = (
        agebin_stats_outer["mean"] * agebin_stats_outer["count"]
        + global_mean_outer * best_agebin_alpha
    ) / (agebin_stats_outer["count"] + best_agebin_alpha)

    X_tr_final = _build_features_fast(
        X_tr_base,
        tr_pids,
        tr_sites_outer,
        tr_sex_outer_full,
        tr_agebin_outer_full,
        smoothed_patient_outer,
        smoothed_site_outer,
        smoothed_sex_outer,
        smoothed_agebin_outer,
        global_mean_outer,
    )
    X_va_final = _build_features_fast(
        X_va_base,
        va_pids,
        va_sites_outer,
        va_sex_outer_full,
        va_agebin_outer_full,
        smoothed_patient_outer,
        smoothed_site_outer,
        smoothed_sex_outer,
        smoothed_agebin_outer,
        global_mean_outer,
    )
    X_te_final = _build_features_fast(
        X_test_fold,
        test_pid_all,
        test_sites_all,
        test_sex_all,
        te_agebin_outer_full,
        smoothed_patient_outer,
        smoothed_site_outer,
        smoothed_sex_outer,
        smoothed_agebin_outer,
        global_mean_outer,
    )

    best_model = Pipeline(
        steps=[
            ("preprocess", preprocess_fold),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    C=best_C,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    best_model.fit(X_tr_final, y_tr)

    va_pred = best_model.predict_proba(X_va_final)[:, 1]
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(fold_auc)
    oof_pred[va_idx] = va_pred
    test_pred_accum += best_model.predict_proba(X_te_final)[:, 1]

    print(
        f"Fold {fold}: best_alpha={best_alpha} best_site_alpha={best_site_alpha} "
        f"best_sex_alpha={best_sex_alpha} best_agebin_alpha={best_agebin_alpha} best_C={best_C} "
        f"(inner_auc={best_inner_auc:.6f}) val_auc={fold_auc:.6f}"
    )

test_pred = test_pred_accum / gkf.get_n_splits()
print(
    f"OOF AUC: {roc_auc_score(y, oof_pred):.6f} (mean fold AUC={np.mean(fold_aucs):.6f})"
)



## === cell 3
sub = sample_sub.copy()
pred_df = pd.DataFrame({"image_name": test["image_name"].values, "target": test_pred})
sub = sub.drop(columns=["target"]).merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(train["target"].mean())

sub["target"] = sub["target"].clip(0.0, 1.0).astype(float)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={sub.shape} and columns={list(sub.columns)}")
print(sub.head())
