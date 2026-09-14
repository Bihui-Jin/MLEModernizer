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

0.9354084526526942

# 6. Current score

0.68347

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66764) has done: 'Your notebook fails because it depends on external OOF/submission CSVs from `../input/rcsiimpreds/` that do not exist in this environment, so none of the downstream merges/averaging can run. I keep the “blend multiple model predictions then average” core logic, but generate those prediction tables locally by training a lightweight metadata-only model (sklearn LogisticRegression) and creating multiple deterministic variants to stand in for the missing model files. Then I robustly merge everything onto `sample_submission.csv`’s `image_name`, compute the same 12-way average `target`, clip to `[0,1]`, and write a valid `submission.csv`.'
- What this solution (achieved 0.77059) has done: 'Your current score (0.66764) is far below the target (0.9354), so we should improve the model signal while keeping the same “metadata model → generate multiple variants → 12-way average blend” core logic intact. The biggest gain with minimal semantic change is to (1) add the strongest available tabular signal (`patient_id`) to the categorical features and (2) add a very small amount of benign feature engineering (missingness indicator + interaction) inside the same sklearn pipeline. This keeps the same LogisticRegression + CV predict_proba approach and the same blending/averaging, but typically lifts AUC substantially for this competition’s metadata baseline. I also keep the submission alignment logic unchanged and still write `submission.csv`.'
- What this solution (achieved 0.74277) has done: 'Your current AUC (0.77059) is far below the target (0.9354), so we should increase signal while keeping the same “metadata LogisticRegression with CV → generate multiple deterministic variants → 12-way average blend” core logic intact. The smallest high-impact fix is to add a few well-known strong metadata features for this competition (log-age, one-hot of `anatom_site_general_challenge` already present, plus patient-level frequency features computed from train only) without changing the model class or training loop. We also ensure categorical missing values are treated consistently (explicit “missing” token rather than relying on `most_frequent`) to reduce train/test mismatch. All blending/post-processing stays the same, and we still write a valid `submission.csv`.'
- What this solution (achieved 0.78191) has done: 'Your current score (0.74277) is far below the target (0.9354), so we need to increase AUC while keeping the same core “metadata LogisticRegression with CV → generate multiple deterministic variants → 12-way average blend” logic intact. The biggest minimal win is to fix a leakage-like inconsistency in your frequency features: for test you currently compute `patient_count/site_count` from the test distribution itself, which shifts feature meaning between train and test; we map test counts using *train* counts (unknowns→0) to align distributions. Next, we add one more high-signal but still “negligible” feature engineering step without changing the model/loop: patient-level target encoding computed out-of-fold (OOF) on train and then mapped to test by patient_id, which is a standard strong metadata-only boost for this competition. Finally, we include these new numeric features in the same preprocessing pipeline and keep the blending/post-processing/submission format unchanged.'
- What this solution (achieved 0.76647) has done: 'Your current AUC (0.7819) is far below the target (0.9354), so we should increase score with the smallest changes that keep the same “metadata LogisticRegression with CV → create multiple deterministic variants → 12-way average blend” core logic intact. The highest-impact minimal fix is to compute patient-level target encoding in a leakage-safe way that matches how you use it at test time: inside each CV fold, build the encoding map on that fold’s train split and apply it to both the fold’s validation and the test set, then average test predictions across folds. This preserves the same model class, preprocessing, and blending semantics, but removes the current train/test feature-generation mismatch (OOF on train vs full-map on test) that tends to hurt generalization. I also keep your existing frequency features mapped from train only and maintain the exact submission writing/alignment.'
- What this solution (achieved 0.65142) has done: 'The timeout is dominated by repeatedly fitting the same preprocessing (imputer/scaler/one-hot) inside every CV fold and for four different random states, plus using a dense one-hot matrix. I keep the exact same feature set, CV fold construction, target encoding semantics, and LogisticRegression core model, but pre-fit and reuse the base preprocessing once, switch OneHotEncoder back to sparse (mathematically identical), and cache per-fold transformed matrices so the four runs reuse them without redoing expensive encoding/scaling. I also avoid per-row Python loops when building patient fold IDs and reduce pandas copying in TE attachment while keeping identical computed columns. These changes keep the algorithm identical (same folds, same TE per fold, same model hyperparameters) but cut repeated preprocessing work enough to fit under 600 seconds.'
- What this solution (achieved 0.7132) has done: 'I fix the runtime error coming from `scipy.sparse.hstack` by ensuring every block passed to `hstack` is a 2D sparse matrix with consistent row counts; this is caused by `ColumnTransformer` returning a dense `ndarray` for the TE part, which then breaks `hstack`’s internal array conversion. I also keep the core training/blending logic identical, but make the TE transformer explicitly output sparse and avoid re-fitting it in a way that yields dense output. After that, the downstream blend/merge cells run again and the notebook write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.71244) has done: 'Your current AUC (0.7132) is far below the target (0.9354), so we should increase model signal while keeping the same core “metadata LogisticRegression with patient-grouped CV + per-fold patient TE → 4 runs → 12-way squash/average blend” intact. The biggest minimal fix is that you currently *do not train on `patient_id` as a feature at all* (it’s excluded from one-hot and only used indirectly for TE), so we add a leakage-safe numeric patient code per fold (and for test per fold) and scale it alongside the existing TE block; this preserves the same model class/training loop while adding a strong ID signal. Additionally, we expand the target encoding to include `anatom_site_general_challenge` and the `(patient_id, site)` pair, computed strictly within each fold’s train split and applied to that fold’s validation/test, which is a standard high-impact metadata boost for this competition and stays within the same TE mechanism you already use. All blending/post-processing/submission alignment stays identical, and the script still writes `submission.csv`.'
- What this solution (achieved 0.68347) has done: 'Your current AUC (0.712) is far below the target (0.935), so we should increase signal with minimal, metric-aligned changes while keeping the same “metadata LogisticRegression with patient-grouped CV + per-fold target encoding → 4 runs → 12-way squash/average blend” core logic intact. The largest low-risk gain here is to stop throwing away `patient_id` as a categorical feature: we include it in the OneHotEncoder (still with `handle_unknown="ignore"`), which is consistent with your earlier feature list and usually helps a lot on this competition. Separately, your target-encoding smoothing currently uses a global mean computed on the full dataset (including the fold’s validation), which is a small but real fold-inconsistency; we compute the prior mean from the fold’s train split inside each fold to make TE strictly fold-train-only (better generalization, same TE mechanism). Everything else (folding by patient, TE keys, LogisticRegression, squash variants, averaging, submission alignment) is kept the same, and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

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

    denom = float(len(train))
    df["patient_count_ratio"] = df["patient_count"] / denom
    df["site_count_ratio"] = df["site_count"] / denom

    df["patient_site"] = (
        df["patient_id"].astype(str)
        + "||"
        + df["anatom_site_general_challenge"].astype(str)
    ).astype("object")

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
    "patient_count_ratio",
    "site_count_ratio",
    "patient_target_enc",
    "patient_target_enc_logit",
    "site_target_enc",
    "site_target_enc_logit",
    "patient_site_target_enc",
    "patient_site_target_enc_logit",
    "patient_code",
]
features_cat = [
    "sex",
    "anatom_site_general_challenge",
    "patient_id",
]

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocess_base = ColumnTransformer(
    transformers=[
        (
            "num_base",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            ),
            [
                "age_approx",
                "age_log1p",
                "age_sex",
                "age_missing",
                "patient_count_log1p",
                "site_count_log1p",
                "patient_count_ratio",
                "site_count_ratio",
            ],
        ),
        ("cat", categorical_transformer, features_cat),
    ],
    remainder="drop",
)

preprocess_te = ColumnTransformer(
    transformers=[
        (
            "num_te",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            ),
            [
                "patient_target_enc",
                "patient_target_enc_logit",
                "site_target_enc",
                "site_target_enc_logit",
                "patient_site_target_enc",
                "patient_site_target_enc_logit",
                "patient_code",
            ],
        )
    ],
    remainder="drop",
    sparse_threshold=0.0,  # force sparse output even for small-dim numeric TE
)


def _make_te_map_from_train_split(
    key_series: pd.Series, y_split: np.ndarray, prior_mean: float
) -> pd.Series:
    """
    Create smoothed mean target encoding map from a train split only, for an arbitrary key.

    Change (score-improving, minimal): use fold-train prior_mean instead of a global mean
    computed on full data, to keep TE strictly fold-train-only and reduce CV mismatch.
    """
    stats = (
        pd.DataFrame({"key": key_series.astype("object").values, "y": y_split})
        .groupby("key")["y"]
        .agg(["mean", "count"])
    )
    te_map = (stats["mean"] * stats["count"] + prior_mean * alpha) / (
        stats["count"] + alpha
    )
    te_map.name = "te"
    return te_map


def _compute_te_arrays(key_series: pd.Series, te_map: pd.Series, prior_mean: float):
    te = key_series.map(te_map).fillna(prior_mean).astype(float).to_numpy()
    p = np.clip(te, 1e-6, 1 - 1e-6)
    te_logit = np.log(p / (1 - p))
    return te, te_logit


def _compute_patient_code_arrays(
    patient_id_series: pd.Series, code_map: pd.Series
) -> np.ndarray:
    return patient_id_series.map(code_map).fillna(-1).astype(float).to_numpy()


def _group_stratified_folds_by_patient(
    Xb: pd.DataFrame, y_arr: np.ndarray, n_splits: int, random_state: int
):
    """
    Create folds at the patient level; stratify on per-patient mean target.
    """
    patients = Xb["patient_id"].astype("object")
    dfp = pd.DataFrame({"patient_id": patients.values, "y": y_arr})
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

    fold_id_by_patient = np.empty(len(p_ids), dtype=np.int16)
    for fold_id, (_, va_p) in enumerate(skf.split(p_ids, p_bins)):
        fold_id_by_patient[va_p] = fold_id
    pid_to_fold = pd.Series(fold_id_by_patient, index=p_ids)

    fold_ids_rows = patients.map(pid_to_fold).astype(np.int16).to_numpy()

    for fold_id in range(n_splits):
        va_idx = np.where(fold_ids_rows == fold_id)[0]
        tr_idx = np.where(fold_ids_rows != fold_id)[0]
        yield tr_idx, va_idx


from scipy import sparse

_base_preprocess_fitted = False
_X_base_transformed = None
_X_test_base_transformed = None
_fold_cache = {}  # keyed by (n_splits, random_state, fold_id) -> (X_tr, y_tr, X_te)


def _ensure_base_preprocess_fitted():
    global _base_preprocess_fitted, _X_base_transformed, _X_test_base_transformed
    if _base_preprocess_fitted:
        return
    preprocess_base.fit(X_base)
    _X_base_transformed = preprocess_base.transform(X_base)
    _X_test_base_transformed = preprocess_base.transform(X_test_base)

    if not sparse.issparse(_X_base_transformed):
        _X_base_transformed = sparse.csr_matrix(_X_base_transformed)
    if not sparse.issparse(_X_test_base_transformed):
        _X_test_base_transformed = sparse.csr_matrix(_X_test_base_transformed)

    _base_preprocess_fitted = True


def fit_predict_cv(
    random_state: int, C: float = 1.0, class_weight="balanced", n_splits: int = 5
) -> np.ndarray:
    """
    LogisticRegression + CV predict_proba, TE computed strictly on each fold's train split.
    Reuses base preprocessing and caches transformed fold matrices.
    """
    _ensure_base_preprocess_fitted()

    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=600,
        C=C,
        class_weight=class_weight,
        n_jobs=None,
        random_state=random_state,
    )

    preds_test = np.zeros(len(X_test_base), dtype=np.float64)

    fold_iter = list(
        _group_stratified_folds_by_patient(
            X_base, y, n_splits=n_splits, random_state=random_state
        )
    )

    for fold_id, (tr_idx, va_idx) in enumerate(fold_iter):
        cache_key = (n_splits, random_state, fold_id)
        cached = _fold_cache.get(cache_key, None)

        if cached is None:
            X_tr_base_df = X_base.iloc[tr_idx]
            y_tr = y[tr_idx]

            prior_mean = float(np.mean(y_tr))

            te_map_patient = _make_te_map_from_train_split(
                X_tr_base_df["patient_id"], y_tr, prior_mean=prior_mean
            )
            te_map_site = _make_te_map_from_train_split(
                X_tr_base_df["anatom_site_general_challenge"],
                y_tr,
                prior_mean=prior_mean,
            )
            te_map_patient_site = _make_te_map_from_train_split(
                X_tr_base_df["patient_site"], y_tr, prior_mean=prior_mean
            )

            te_p_tr, te_p_tr_logit = _compute_te_arrays(
                X_tr_base_df["patient_id"], te_map_patient, prior_mean=prior_mean
            )
            te_p_te, te_p_te_logit = _compute_te_arrays(
                X_test_base["patient_id"], te_map_patient, prior_mean=prior_mean
            )

            te_s_tr, te_s_tr_logit = _compute_te_arrays(
                X_tr_base_df["anatom_site_general_challenge"],
                te_map_site,
                prior_mean=prior_mean,
            )
            te_s_te, te_s_te_logit = _compute_te_arrays(
                X_test_base["anatom_site_general_challenge"],
                te_map_site,
                prior_mean=prior_mean,
            )

            te_ps_tr, te_ps_tr_logit = _compute_te_arrays(
                X_tr_base_df["patient_site"], te_map_patient_site, prior_mean=prior_mean
            )
            te_ps_te, te_ps_te_logit = _compute_te_arrays(
                X_test_base["patient_site"], te_map_patient_site, prior_mean=prior_mean
            )

            uniq_p = pd.Index(pd.unique(X_tr_base_df["patient_id"].astype("object")))
            code_map = pd.Series(np.arange(len(uniq_p), dtype=np.int32), index=uniq_p)

            patient_code_tr = _compute_patient_code_arrays(
                X_tr_base_df["patient_id"], code_map
            )
            patient_code_te = _compute_patient_code_arrays(
                X_test_base["patient_id"], code_map
            )

            X_tr_te_df = pd.DataFrame(
                {
                    "patient_target_enc": te_p_tr,
                    "patient_target_enc_logit": te_p_tr_logit,
                    "site_target_enc": te_s_tr,
                    "site_target_enc_logit": te_s_tr_logit,
                    "patient_site_target_enc": te_ps_tr,
                    "patient_site_target_enc_logit": te_ps_tr_logit,
                    "patient_code": patient_code_tr,
                }
            )
            X_te_te_df = pd.DataFrame(
                {
                    "patient_target_enc": te_p_te,
                    "patient_target_enc_logit": te_p_te_logit,
                    "site_target_enc": te_s_te,
                    "site_target_enc_logit": te_s_te_logit,
                    "patient_site_target_enc": te_ps_te,
                    "patient_site_target_enc_logit": te_ps_te_logit,
                    "patient_code": patient_code_te,
                }
            )

            preprocess_te.fit(X_tr_te_df)

            X_tr_te = preprocess_te.transform(X_tr_te_df)
            X_te_te = preprocess_te.transform(X_te_te_df)

            if not sparse.issparse(X_tr_te):
                X_tr_te = sparse.csr_matrix(X_tr_te)
            if not sparse.issparse(X_te_te):
                X_te_te = sparse.csr_matrix(X_te_te)

            X_tr = sparse.hstack([_X_base_transformed[tr_idx], X_tr_te], format="csr")
            X_te = sparse.hstack([_X_test_base_transformed, X_te_te], format="csr")

            _fold_cache[cache_key] = (X_tr, y_tr, X_te)
        else:
            X_tr, y_tr, X_te = cached

        clf.fit(X_tr, y_tr)
        preds_test += clf.predict_proba(X_te)[:, 1]

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



## === cell 1
pred_512_B6.head()



## === cell 2
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



## === cell 3
pass



## === cell 4
pass



## === cell 5
result1 = pd.merge(pred_b3, pred_b4, on="image_name", suffixes=("_b3", "_b4"))
result1 = result1.rename(columns={"target_b3": "target_b3", "target_b4": "target_b4"})
result1.head()



## === cell 6
result2 = pd.merge(pred_b5, pred_b6, on="image_name", suffixes=("_b5", "_b6"))
result2.head()



## === cell 7
semi_final = pd.merge(result1, result2, on="image_name")
semi_final.head()



## === cell 8
result3 = pd.merge(
    pred_cw_b4, pred_512_B6, on="image_name", suffixes=("_cw_b4", "_B6_512")
)
result3.head()



## === cell 9
final = pd.merge(semi_final, result3, on="image_name")
final.head()



## === cell 10
final = pd.merge(final, result_tta, on="image_name")
final.head()



## === cell 11
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

kr_result = pd.merge(pred_kr_b3, pred_kr_b4, on="image_name", suffixes=("_b3", "_b4"))
kr_result = pd.merge(kr_result, pred_kr_eb3, on="image_name", suffixes=("_b3", "_b4"))
kr_result.head()



## === cell 12
final = pd.merge(final, kr_result, on="image_name")
final.head()



## === cell 13
pred_256_b4 = pd.DataFrame(
    {
        "image_name": sample["image_name"].values,
        "target_256_b4": squash(test_pred_b4, k=1.00, b=0.01),
    }
)
pred_256_b4.head()



## === cell 14
final = pd.merge(final, pred_256_b4, on="image_name")
final.head()



## === cell 15
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



## === cell 16
final.head()



## === cell 17
submit_file = final[["image_name", "target"]].copy()
submit_file = sample[["image_name"]].merge(submit_file, on="image_name", how="left")
if submit_file["target"].isna().any():
    submit_file["target"] = submit_file["target"].fillna(submit_file["target"].mean())
submit_file.head()



## === cell 18
submit_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submit_file.shape)
print(submit_file.head())
