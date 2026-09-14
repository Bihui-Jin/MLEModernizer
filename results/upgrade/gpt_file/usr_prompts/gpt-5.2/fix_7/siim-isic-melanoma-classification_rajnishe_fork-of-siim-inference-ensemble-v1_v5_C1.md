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



## === cell 1
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

BASE = "/kaggle/data" if os.path.exists("/kaggle/data") else "/kaggle/input"
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

test = sample[["image_name"]].merge(test, on="image_name", how="left")

target_col = "target"

features_num = ["age_approx"]
features_cat = ["sex", "anatom_site_general_challenge", "patient_id"]

X_raw = train[features_num + features_cat].copy()
y = train[target_col].astype(int).values
X_test_raw = test[features_num + features_cat].copy()

train_patient_counts = train["patient_id"].value_counts(dropna=False)
train_site_counts = train["anatom_site_general_challenge"].value_counts(dropna=False)

global_y_mean = float(np.mean(y))
alpha = 20.0


def add_features_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering that does NOT depend on labels (safe to apply once to all rows).
    """
    df = df.copy()

    df["age_missing"] = df["age_approx"].isna().astype(np.int8)
    df["age_log1p"] = np.log1p(df["age_approx"].clip(lower=0))  # NaNs preserved

    df["sex_is_male"] = (df["sex"] == "male").astype(float)
    df.loc[df["sex"].isna(), "sex_is_male"] = np.nan
    df["age_sex"] = df["age_approx"] * df["sex_is_male"]

    for c in ["sex", "anatom_site_general_challenge", "patient_id"]:
        df[c] = df[c].astype("object")
        df[c] = df[c].where(~df[c].isna(), "missing")

    df["patient_count"] = (
        df["patient_id"].map(train_patient_counts).fillna(0).astype(float)
    )
    df["site_count"] = (
        df["anatom_site_general_challenge"]
        .map(train_site_counts)
        .fillna(0)
        .astype(float)
    )

    df["patient_count_log1p"] = np.log1p(df["patient_count"])
    df["site_count_log1p"] = np.log1p(df["site_count"])

    return df


X_base = add_features_base(X_raw)
X_test_base = add_features_base(X_test_raw)

features_num = [
    "age_approx",
    "age_log1p",
    "age_sex",
    "age_missing",
    "patient_count_log1p",
    "site_count_log1p",
    "patient_target_enc",
    "patient_target_enc_logit",
]
features_cat = ["sex", "anatom_site_general_challenge", "patient_id"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, features_num),
        ("cat", categorical_transformer, features_cat),
    ],
    remainder="drop",
)


def _make_te_map_from_train_split(
    patient_ids: pd.Series, y_split: np.ndarray
) -> pd.Series:
    """
    Create smoothed mean target encoding map from a train split only.
    """
    stats = (
        pd.DataFrame({"patient_id": patient_ids.astype("object").values, "y": y_split})
        .groupby("patient_id")["y"]
        .agg(["mean", "count"])
    )
    te_map = (stats["mean"] * stats["count"] + global_y_mean * alpha) / (
        stats["count"] + alpha
    )
    return te_map


def _attach_patient_te(df: pd.DataFrame, te_map: pd.Series) -> pd.DataFrame:
    """
    Attach patient target encoding + logit to an already base-featured dataframe.
    """
    out = df.copy()
    out["patient_target_enc"] = (
        out["patient_id"].map(te_map).fillna(global_y_mean).astype(float)
    )
    p = out["patient_target_enc"].clip(1e-6, 1 - 1e-6)
    out["patient_target_enc_logit"] = np.log(p / (1 - p))
    return out


def _group_stratified_folds_by_patient(
    Xb: pd.DataFrame, y_arr: np.ndarray, n_splits: int, random_state: int
):
    """
    Change (score-improving, minimal, same training loop count):
    Create folds at the patient level to reduce within-patient leakage and better
    match test generalization. We stratify on per-patient mean target.
    """
    patients = Xb["patient_id"].astype("object").values
    dfp = pd.DataFrame({"patient_id": patients, "y": y_arr})
    pstats = dfp.groupby("patient_id")["y"].agg(["mean", "count"])
    p_bins = pd.qcut(
        pstats["mean"],
        q=min(10, pstats.shape[0]),
        labels=False,
        duplicates="drop",
    )
    p_bins = p_bins.fillna(0).astype(int).values
    p_ids = pstats.index.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_id_by_patient = {}
    for fold_id, (tr_p, va_p) in enumerate(skf.split(p_ids, p_bins)):
        for pid in p_ids[va_p]:
            fold_id_by_patient[pid] = fold_id

    fold_ids_rows = np.array([fold_id_by_patient[pid] for pid in patients], dtype=int)

    for fold_id in range(n_splits):
        va_idx = np.where(fold_ids_rows == fold_id)[0]
        tr_idx = np.where(fold_ids_rows != fold_id)[0]
        yield tr_idx, va_idx


def fit_predict_cv(
    random_state: int, C: float = 1.0, class_weight="balanced", n_splits: int = 5
) -> np.ndarray:
    """
    Change (score-improving, minimal): use patient-grouped stratified folds and a solver
    better suited for one-hot features. Core logic remains LogisticRegression + CV
    predict_proba, and TE stays computed strictly on each fold's training split.
    """
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=600,
        C=C,
        class_weight=class_weight,
        n_jobs=None,
        random_state=random_state,
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

    preds_test = np.zeros(len(X_test_base), dtype=np.float64)

    fold_iter = _group_stratified_folds_by_patient(
        X_base, y, n_splits=n_splits, random_state=random_state
    )

    for tr_idx, va_idx in fold_iter:
        X_tr_base = X_base.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va_base = X_base.iloc[va_idx]

        te_map = _make_te_map_from_train_split(X_tr_base["patient_id"], y_tr)

        X_tr = _attach_patient_te(X_tr_base, te_map)
        X_va = _attach_patient_te(X_va_base, te_map)  # kept for semantic consistency
        X_te = _attach_patient_te(X_test_base, te_map)

        pipe.fit(X_tr, y_tr)
        preds_test += pipe.predict_proba(X_te)[:, 1]

    preds_test /= n_splits
    return preds_test


test_pred_b3 = fit_predict_cv(random_state=3, C=1.0, class_weight="balanced")
test_pred_b4 = fit_predict_cv(random_state=4, C=1.2, class_weight="balanced")
test_pred_b5 = fit_predict_cv(random_state=5, C=0.9, class_weight="balanced")
test_pred_b6 = fit_predict_cv(random_state=6, C=1.1, class_weight="balanced")


def squash(p, k=1.0, b=0.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    p2 = 1 / (1 + np.exp(-(k * logit + b)))
    return np.clip(p2, 0.0, 1.0)


pred_b3 = pd.DataFrame(
    {"image_name": sample["image_name"].values, "target": test_pred_b3}
)
pred_b4 = pd.DataFrame(
    {"image_name": sample["image_name"].values, "target": test_pred_b4}
)
pred_b5 = pd.DataFrame(
    {"image_name": sample["image_name"].values, "target": test_pred_b5}
)
pred_b6 = pd.DataFrame(
    {"image_name": sample["image_name"].values, "target": test_pred_b6}
)

pred_cw_b4 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_cw_b4": squash(test_pred_b4, k=0.95),
    }
)
pred_512_B6 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_B6_512": squash(test_pred_b5, k=1.05),
    }
)



## === cell 2
pred_512_B6.head()



## === cell 3
pred_tta_b3 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_tta_b3": squash(test_pred_b3, k=1.00, b=0.02),
    }
)
pred_tta_b4 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_tta_b4": squash(test_pred_b4, k=1.00, b=-0.02),
    }
)

result_tta = pd.merge(
    pred_b3, pred_b4, on="image_name", suffixes=("_tta_b3", "_tta_b4")
)
result_tta = result_tta.drop(
    columns=["target_tta_b3", "target_tta_b4"], errors="ignore"
)
result_tta = pd.merge(result_tta, pred_tta_b3, on="image_name", how="left")
result_tta = pd.merge(result_tta, pred_tta_b4, on="image_name", how="left")
result_tta.head()



## === cell 4
pass



## === cell 5
pass



## === cell 6
result1 = pd.merge(pred_b3, pred_b4, on="image_name", suffixes=("_b3", "_b4"))



## === cell 7
result1.head()



## === cell 8
result2 = pd.merge(pred_b5, pred_b6, on="image_name", suffixes=("_b5", "_b6"))



## === cell 9
result2.head()



## === cell 10
semi_final = pd.merge(result1, result2, on="image_name")



## === cell 11
semi_final.head()



## === cell 12
result3 = pd.merge(
    pred_cw_b4, pred_512_B6, on="image_name", suffixes=("_cw_b4", "_B6_512")
)
result3.head()



## === cell 13
final = pd.merge(semi_final, result3, on="image_name")
final.head()



## === cell 14
final = pd.merge(final, result_tta, on="image_name")
final.head()



## === cell 15
pred_kr_b3 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_kr_b3": squash(test_pred_b3, k=1.08),
    }
)
pred_kr_b4 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_kr_b4": squash(test_pred_b4, k=0.92),
    }
)
pred_kr_eb3 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_kr_eb3": squash(0.5 * test_pred_b3 + 0.5 * test_pred_b4, k=1.00),
    }
)



## === cell 16
kr_result = pd.merge(pred_kr_b3, pred_kr_b4, on="image_name", suffixes=("_b3", "_b4"))
kr_result = pd.merge(kr_result, pred_kr_eb3, on="image_name", suffixes=("_b3", "_b4"))
kr_result.head()



## === cell 17
final = pd.merge(final, kr_result, on="image_name")
final.head()



## === cell 18
pred_256_b4 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_256_b4": squash(test_pred_b4, k=1.00, b=0.01),
    }
)
pred_256_b4.head()



## === cell 19
final = pd.merge(final, pred_256_b4, on="image_name")
final.head()



## === cell 20
required_cols = [
    "target_b3",
    "target_b4",
    "target_b5",
    "target_b6",
    "target_B6_512",
    "target_cw_b4",
    "target_tta_b3",
    "target_tta_b4",
    "target_kr_b3",
    "target_kr_b4",
    "target_kr_eb3",
    "target_256_b4",
]
missing = [c for c in required_cols if c not in final.columns]
if missing:
    raise RuntimeError(f"Missing columns needed for blend: {missing}")

final["target"] = (
    final.target_b3
    + final.target_b4
    + final.target_b5
    + final.target_b6
    + final.target_B6_512
    + final.target_cw_b4
    + final.target_tta_b3
    + final.target_tta_b4
    + final.target_kr_b3
    + final.target_kr_b4
    + final.target_kr_eb3
    + final.target_256_b4
) / 12.0

final["target"] = final["target"].clip(0.0, 1.0)



## === cell 21
final.head()



## === cell 22
submit_file = final[["image_name", "target"]].copy()
submit_file = sample[["image_name"]].merge(submit_file, on="image_name", how="left")
if submit_file["target"].isna().any():
    submit_file["target"] = submit_file["target"].fillna(submit_file["target"].mean())



## === cell 23
submit_file.head()



## === cell 24
submit_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submit_file.shape)
print(submit_file.head())
