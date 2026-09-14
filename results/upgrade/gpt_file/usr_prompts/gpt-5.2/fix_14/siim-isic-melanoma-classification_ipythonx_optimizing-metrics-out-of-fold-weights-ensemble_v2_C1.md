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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
import pandas as pd

"""
Same Seed: 42, 
Base Model: E6
Top Model - GAP, ATTENTION, GEM

This notebook originally depends on an external Kaggle Dataset:
'../input/efficientnetb6seed-42-oof-prediction/'.
In this environment that directory may not exist, so we:
- try to load the blend files if present
- otherwise fall back to a valid metadata-only submission to ensure a .csv is produced

Score-direction (current 0.68391 -> target 0.92556, higher is better):
- Keep the same metadata-only LogisticRegression + StratifiedGroupKFold loop fallback.
- Minimal, score-relevant improvements to increase AUC (ranking) without changing core logic:
  (1) Restrict numeric interaction features to continuous age-derived variables only.
      This preserves the same LR model family and the same CV loop, but avoids noisy
      interactions involving missingness indicator columns which often hurt AUC.
  (2) Slightly adjust the LR C micro-ensemble to focus around the previously good region,
      improving stability/generalization without changing modeling approach.
"""

BLEND_DIR = "../input/efficientnetb6seed-42-oof-prediction/"

BASE_DATA_DIR = "/kaggle/data"
TRAIN_CSV = os.path.join(BASE_DATA_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DATA_DIR, "sample_submission.csv")


def _safe_read_csv(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None


oof_one = _safe_read_csv(
    os.path.join(BLEND_DIR, "oof_e6_attn_gap_seed_42.csv")
)  # 0.912
test_one = _safe_read_csv(
    os.path.join(BLEND_DIR, "s_e6_attn_gap_seed_42.csv")
)  # 0.9422

oof_two = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_attn_seed_42.csv"))  # 0.918
test_two = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_attn_seed_42.csv"))  # 0.9431

oof_three = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_gem_seed_42.csv"))  # 0.9050
test_three = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_gem_seed_42.csv"))  # 0.9405

oof_four = _safe_read_csv(
    os.path.join(BLEND_DIR, "oof_e6_our_attn_seed_42.csv")
)  # 0.892
test_four = _safe_read_csv(
    os.path.join(BLEND_DIR, "s_e6_our_attn_seed_42.csv")
)  # 0.9445

oof_five = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_gap_seed_42.csv"))  # 0.904
test_five = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_gap_seed_42.csv"))  # 0.9454

blend_files_available = all(
    df is not None
    for df in [
        oof_one,
        test_one,
        oof_two,
        test_two,
        oof_three,
        test_three,
        oof_four,
        test_four,
        oof_five,
        test_five,
    ]
)

print("Blend files available:", blend_files_available)
if not blend_files_available:
    print(
        f"WARNING: Missing external blend files under {BLEND_DIR}. Will fit a metadata-only fallback model and write submission.csv."
    )



## === cell 1
if blend_files_available:
    print(oof_one.head())
else:
    print("Skipping oof_one.head() (blend files not available).")



## === cell 2
if blend_files_available:
    print(oof_two.head())
else:
    print("Skipping oof_two.head() (blend files not available).")



## === cell 3
if blend_files_available:
    print(oof_three.head())
else:
    print("Skipping oof_three.head() (blend files not available).")



## === cell 4
if blend_files_available:
    print(oof_four.head())
else:
    print("Skipping oof_four.head() (blend files not available).")



## === cell 5
if blend_files_available:
    print(oof_five.head())
else:
    print("Skipping oof_five.head() (blend files not available).")



## === cell 6
if blend_files_available:
    print(test_one.head())
else:
    print("Skipping test_one.head() (blend files not available).")



## === cell 7
if blend_files_available:
    print(test_two.head())
else:
    print("Skipping test_two.head() (blend files not available).")



## === cell 8
if blend_files_available:
    print(test_three.head())
else:
    print("Skipping test_three.head() (blend files not available).")



## === cell 9
if blend_files_available:
    print(test_four.head())
else:
    print("Skipping test_four.head() (blend files not available).")



## === cell 10
if blend_files_available:
    print(test_five.head())
else:
    print("Skipping test_five.head() (blend files not available).")



## === cell 11
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score


def _get_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


if blend_files_available:
    for _df in [oof_one, oof_two, oof_three, oof_four, oof_five]:
        _df.sort_values(by=["image_name"], ascending=True, inplace=True)
        _df.reset_index(drop=True, inplace=True)

    for _df in [test_one, test_two, test_three, test_four, test_five]:
        _df.sort_values(by=["image_name"], ascending=True, inplace=True)
        _df.reset_index(drop=True, inplace=True)

    y_col = _get_col(oof_one, ["target"])
    p1_col = _get_col(oof_one, ["pred", "oof", "prediction", "target_pred"])
    p2_col = _get_col(oof_two, ["pred", "oof", "prediction", "target_pred"])
    p3_col = _get_col(oof_three, ["pred", "oof", "prediction", "target_pred"])
    p4_col = _get_col(oof_four, ["pred", "oof", "prediction", "target_pred"])
    p5_col = _get_col(oof_five, ["pred", "oof", "prediction", "target_pred"])

    t1_col = _get_col(test_one, ["target", "pred", "prediction"])
    t2_col = _get_col(test_two, ["target", "pred", "prediction"])
    t3_col = _get_col(test_three, ["target", "pred", "prediction"])
    t4_col = _get_col(test_four, ["target", "pred", "prediction"])
    t5_col = _get_col(test_five, ["target", "pred", "prediction"])

    missing = [
        name
        for name, col in [
            ("y_col", y_col),
            ("p1_col", p1_col),
            ("p2_col", p2_col),
            ("p3_col", p3_col),
            ("p4_col", p4_col),
            ("p5_col", p5_col),
            ("t1_col", t1_col),
            ("t2_col", t2_col),
            ("t3_col", t3_col),
            ("t4_col", t4_col),
            ("t5_col", t5_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Required columns not found in blend files: {missing}. Available cols example: {oof_one.columns.tolist()}"
        )

    y_train = oof_one[y_col].to_numpy(dtype=float)

    blend_train = np.array(
        [
            oof_one[p1_col].to_numpy(dtype=float),
            oof_two[p2_col].to_numpy(dtype=float),
            oof_three[p3_col].to_numpy(dtype=float),
            oof_four[p4_col].to_numpy(dtype=float),
            oof_five[p5_col].to_numpy(dtype=float),
        ]
    )

    blend_test = np.array(
        [
            test_one[t1_col].to_numpy(dtype=float),
            test_two[t2_col].to_numpy(dtype=float),
            test_three[t3_col].to_numpy(dtype=float),
            test_four[t4_col].to_numpy(dtype=float),
            test_five[t5_col].to_numpy(dtype=float),
        ]
    )
else:
    y_train = None
    blend_train = None
    blend_test = None



## === cell 12
bestWght = None
bestSC = None

if blend_files_available:

    def roc_min_func(weights):
        final_prediction = np.zeros(blend_train.shape[1], dtype=float)
        for weight, prediction in zip(weights, blend_train):
            final_prediction += weight * prediction
        return -roc_auc_score(y_train, final_prediction)

    print("\n Finding Blending Weights ...")
    res_list = []
    weights_list = []

    rng = np.random.default_rng(42)
    n_models = blend_train.shape[0]
    bounds = [(0.0, 1.0)] * n_models

    for k in range(200):
        starting_values = rng.uniform(size=n_models)
        res = minimize(
            roc_min_func,
            starting_values,
            method="L-BFGS-B",
            bounds=bounds,
            options={"disp": False, "maxiter": 100000},
        )
        res_list.append(res["fun"])
        weights_list.append(res["x"])

        print(
            "{iter}\tScore: {score}\tWeights: {weights}".format(
                iter=(k + 1),
                score=-res["fun"],
                weights="\t".join([str(float(item)) for item in res["x"]]),
            )
        )

    best_idx = int(np.argmin(res_list))
    bestSC = -float(res_list[best_idx])
    bestWght = weights_list[best_idx]
    print("\n Ensemble Score: {best_score}".format(best_score=bestSC))
    print("\n Best Weights: {weights}".format(weights=bestWght))

    test_prices = np.zeros(blend_test.shape[1], dtype=float)
    for k in range(n_models):
        test_prices += blend_test[k] * bestWght[k]

    test_prices = np.clip(test_prices, 0.0, 1.0)

else:
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        _SKLEX_OK = True
    except Exception:
        _SKLEX_OK = False

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from scipy import sparse

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    feature_cols = [
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "patient_id",
        "age_bin",
        "age_missing",
        "age_scaled",
        "age_log",
        "age_sq",
        "sex_missing",
        "age_x_sex",
        "site_x_sex",
    ]

    for _df in (train_df, test_df):
        if "sex" in _df.columns:
            _df["sex"] = _df["sex"].replace("", np.nan)
        if "patient_id" in _df.columns:
            _df["patient_id"] = _df["patient_id"].astype(str)

        age = pd.to_numeric(_df.get("age_approx"), errors="coerce")
        _df["age_bin"] = pd.cut(
            age,
            bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
            labels=[
                "<=20",
                "21-30",
                "31-40",
                "41-50",
                "51-60",
                "61-70",
                "71-80",
                "81+",
            ],
        ).astype("object")

        _df["age_missing"] = age.isna().astype(int)
        _df["age_scaled"] = age  # scaled later
        _df["age_log"] = np.log1p(age.clip(lower=0))  # NaN stays NaN
        _df["age_sq"] = age * age
        _df["sex_missing"] = _df["sex"].isna().astype(int)

        sex_s = _df.get("sex").astype("object")
        _df["age_x_sex"] = (
            _df["age_bin"].astype("object").astype(str) + "_" + sex_s.astype(str)
        ).astype("object")
        _df["site_x_sex"] = (
            _df["anatom_site_general_challenge"].astype("object").astype(str)
            + "_"
            + sex_s.astype(str)
        ).astype("object")

        _df["age_x_sex"] = _df["age_x_sex"].replace(
            ["nan_nan", "nan_None", "None_nan"], np.nan
        )
        _df["site_x_sex"] = _df["site_x_sex"].replace(
            ["nan_nan", "nan_None", "None_nan"], np.nan
        )

    X_all = train_df[feature_cols].copy()
    y = train_df["target"].astype(int).to_numpy()
    X_test = test_df[feature_cols].copy()

    numeric_cont_features = ["age_approx", "age_scaled", "age_log", "age_sq"]
    numeric_indicator_features = ["age_missing", "sex_missing"]
    categorical_features = [
        c
        for c in feature_cols
        if c not in (numeric_cont_features + numeric_indicator_features)
    ]

    numeric_cont_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False, with_std=True)),
            (
                "poly",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
            ),
        ]
    )

    numeric_ind_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_cont", numeric_cont_transformer, numeric_cont_features),
            ("num_ind", numeric_ind_transformer, numeric_indicator_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    def _make_lr(C: float) -> LogisticRegression:
        return LogisticRegression(
            max_iter=1400,  # unchanged
            solver="saga",
            class_weight="balanced",
            n_jobs=None,
            random_state=42,
            C=C,
        )

    Cs = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    models = [_make_lr(C) for C in Cs]

    groups = train_df["patient_id"].astype(str).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    test_pred = np.zeros(len(test_df), dtype=float)
    oof_pred = np.zeros(len(train_df), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(X_all, y, groups=groups), start=1
    ):
        X_tr_df, y_tr = X_all.iloc[tr_idx], y[tr_idx]
        X_va_df, y_va = X_all.iloc[va_idx], y[va_idx]

        X_tr = preprocessor.fit_transform(X_tr_df, y_tr)
        X_va = preprocessor.transform(X_va_df)
        X_te = preprocessor.transform(X_test)

        if sparse.issparse(X_tr):
            X_tr = X_tr.tocsr()
            X_va = X_va.tocsr()
            X_te = X_te.tocsr()

        va_proba_ens = np.zeros(len(va_idx), dtype=float)
        te_proba_ens = np.zeros(len(test_df), dtype=float)

        for clf in models:
            clf.fit(X_tr, y_tr)
            va_proba_ens += clf.predict_proba(X_va)[:, 1].astype(float) / len(models)
            te_proba_ens += clf.predict_proba(X_te)[:, 1].astype(float) / len(models)

        oof_pred[va_idx] = va_proba_ens
        test_pred += te_proba_ens / sgkf.n_splits

        try:
            fold_auc = roc_auc_score(y_va, va_proba_ens)
        except Exception:
            fold_auc = float("nan")
        print(f"Fold {fold} AUC: {fold_auc:.5f}")

    try:
        oof_auc = roc_auc_score(y, oof_pred)
        print(f"OOF AUC (metadata fallback): {oof_auc:.5f}")
    except Exception:
        pass

    test_prices = np.nan_to_num(
        test_pred, nan=float(np.mean(y)), posinf=1.0, neginf=0.0
    )
    test_prices = np.clip(test_prices, 0.0, 1.0)



## === cell 13
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

if blend_files_available:
    pred_df = pd.DataFrame(
        {"image_name": test_one["image_name"].values, "target": test_prices}
    )
    sub = sample_sub[["image_name"]].merge(pred_df, on="image_name", how="left")
else:
    test_df = pd.read_csv(TEST_CSV)
    pred_df = pd.DataFrame(
        {"image_name": test_df["image_name"].values, "target": test_prices}
    )
    sub = sample_sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(float(sub["target"].mean()))

sub = sub.set_index("image_name").loc[sample_sub["image_name"].values].reset_index()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 14
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(sub["target"].values, bins=100)
plt.ylim((0, max(10, int(len(sub) * 0.05))))
plt.title("Submission target distribution")
plt.show()
