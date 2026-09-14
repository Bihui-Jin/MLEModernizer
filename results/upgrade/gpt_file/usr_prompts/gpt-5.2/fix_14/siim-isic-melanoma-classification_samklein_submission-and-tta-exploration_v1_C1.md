# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.9338277214097706

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.66729) has done: 'I remove the hard dependency on the missing `../input/meta-384/*` files by instead generating predictions from the provided `train.csv`/`test.csv` metadata, ensuring the notebook runs end-to-end in this environment. To preserve the original “combine 5 predictions then take a geometric/average/median mean” core logic, I train 5 lightweight sklearn models on the same metadata features and treat their predicted probabilities as columns `1..5` (a drop-in replacement for `preds_all.csv`). I also make the submission creation robust by aligning to `sample_submission.csv` ordering and guaranteeing the output file is named `submission.csv` with `image_name,target`. These changes fix the runtime errors and should yield a non-trivial AUC (better than a constant prediction), moving toward the target score without changing the ensemble/mean-computation semantics.'
- What this solution (achieved 0.66729) has done: 'Your current score (0.66729) is far below the target (0.93383), so we should improve AUC while keeping your overall “5 predictions → (gmean/avg/median) → iterative compression → take component 0” ensemble logic intact. The biggest low-risk gain here is to fix train/validation leakage by generating out-of-fold (OOF) predictions for each of the 5 models (instead of fitting on all training data before predicting), then training each model on full data only for test predictions. Additionally, for ROC-AUC, any strictly monotonic transform preserves ranking, so we can safely replace the iterative compression output with a simple rank-based normalization (applied after your ensemble step) to stabilize probability spread without changing evaluation semantics (it won’t hurt AUC and often helps avoid degenerate distributions). These are minimal, sklearn-only changes and still write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.68416) has done: 'I fix the root runtime error by making the preprocessing output dense, because `HistGradientBoostingClassifier` cannot accept sparse matrices produced by `OneHotEncoder`. Then the downstream cells run because `all_preds`, `preds`, and `sub` be created successfully. I also keep the existing “5 models → (gmean/avg/median) → iterative compression → take component 0 → rank-normalize” ensemble logic unchanged, only adding small numerical safeguards (clipping) to avoid `gmean` issues with exact 0/1 probabilities. Finally, the script always write a valid `submission.csv` with `image_name,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.68798) has done: 'Your current score (0.68416) is far below the target (0.93383), so we should improve AUC while keeping your “5 models → (gmean/avg/median) → iterative compression → rank-normalize” ensemble semantics intact. The biggest low-risk gain within your exact model family is to make the CV and final fitting respect **grouping by patient_id**, because lesions from the same patient appear multiple times and random StratifiedKFold can leak patient-specific signals and harm generalization. I switch the out-of-fold generation to a **stratified GroupKFold** (implemented via StratifiedGroupKFold when available, otherwise a deterministic fallback), and I also add `class_weight="balanced"` to the classifier to better handle strong class imbalance without changing the learning algorithm. Everything else (features, 5-seed ensemble, mean-compression, rank normalization, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.674) has done: 'Your current AUC (0.68798) is far below the target (0.93383), so we should improve ranking quality while keeping your exact “metadata features → 5 HGB models → (gmean/avg/median) iterative compression → component 0 → rank-normalize” semantics intact. The lowest-risk gain is to remove the strongest leakage/overfitting source: using `patient_id` as a high-cardinality categorical feature can make models overfit to patient identities and hurt generalization to unseen patients in the test set. I keep `patient_id` only for GroupKFold splitting (to prevent leakage) but exclude it from the model’s feature set, leaving all other features and the ensemble/aggregation logic unchanged. This is a minimal, directly score-relevant change that typically improves public AUC for this competition with metadata-only approaches, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.6763) has done: 'We’re far below the target AUC, so the safest way to improve toward it while preserving your core “5 HGB models → (gmean/avg/median) iterative compression → rank-normalize” logic is to make the 5 base models meaningfully diverse in a way that often improves ranking. I keep the exact same model family and training loop, but (1) use the model-specific seed for fold shuffling so each model sees different folds, and (2) slightly vary a couple of HGB hyperparameters across the 5 models (still HGB, same loss/iterations) to reduce correlation between ensemble members. Everything else (features, preprocessing, aggregation/compression, and submission writing/alignment) stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.6763) has done: 'Your current AUC (0.6763) is far below the target (0.9338), so we should increase ranking quality while keeping the same metadata-only + 5x HGB + (gmean/avg/median) iterative compression + rank-normalize pipeline intact. The smallest high-impact fix is to ensure **consistent preprocessing categories across train/test** by fitting the `OneHotEncoder` on the concatenation of train+test features (unsupervised), then using that frozen preprocessor inside the same HGB pipelines; this often improves generalization and reduces train/test category-mismatch noise without changing the model family or training loop. Additionally, we compute and print the **mean OOF AUC across all 5 base models** (instead of only model 1) as a better sanity check, without affecting submissions. Everything else—including the ensemble aggregation/compression semantics and `submission.csv` writing/alignment—remains the same.'
- What this solution (achieved 0.67033) has done: 'Your current AUC (0.6763) is far below the target (0.9338), so we should improve ranking while keeping your exact metadata-only + 5x HistGradientBoosting + (gmean/avg/median) iterative compression + rank-normalize pipeline intact. The biggest score-relevant issue here is that `HistGradientBoostingClassifier` does **not** natively handle highly-informative categorical interactions; with your current preprocessing, the model mainly sees sparse one-hot signals and may underfit. A minimal but high-impact change that preserves your overall approach is to switch the categorical preprocessing from one-hot to **ordinal encoding** (with unknown handling), which works much better with histogram-based tree boosting and typically yields a large AUC jump on this competition’s metadata. Everything else (GroupKFold OOF, 5-model diversity, mean-compression, rank normalization, and `submission.csv` format/alignment) remains the same.'
- What this solution (achieved 0.67194) has done: 'We’re far below the target AUC, so we should improve ranking quality with minimal changes while preserving your exact “5 HGB models → (gmean/avg/median) iterative compression → component 0 → rank-normalize” ensemble semantics. The main score-limiting issue is that `StandardScaler` is not needed for tree-based HistGradientBoosting and can add noise/instability; removing it typically improves HGB performance without changing the overall approach. Additionally, I switch the numeric imputation from median to mean (more compatible with the removed scaling) and add a single, minimal monotonic post-calibration step: mixing a small amount of the mean base-model prediction into the compressed component before rank-normalization (AUC-friendly and low-risk). Everything else—including group-aware OOF generation, model family, number of models, aggregation/compression loop, and submission formatting—remains intact.'
- What this solution (achieved 0.6725) has done: 'Your current AUC is far below the target, so we should try to improve ranking with very small, low-risk tweaks while preserving your exact metadata-only + 5x HistGradientBoosting + (gmean/avg/median) iterative compression + rank-normalize pipeline. The two most score-relevant minimal changes are (1) add a couple of strong, competition-specific metadata features (log-age and missingness interactions) without changing the modeling approach, and (2) slightly strengthen the HGB base models’ capacity via `max_leaf_nodes` (still the same model, same loss/iterations) while keeping the 5-model ensemble structure and training loop unchanged. I also keep the group-aware CV logic exactly as-is, and I keep the submission alignment to `sample_submission.csv` unchanged to ensure a valid `submission.csv`. These changes should move AUC upward toward the target without altering your core ensemble/aggregation semantics.'
- What this solution (achieved 0.67169) has done: 'Your current AUC (0.6725) is far below the target (0.9338), so we should improve ranking with very small changes while preserving your exact metadata-only + 5×HistGradientBoosting + (gmean/avg/median) iterative compression + rank-normalize pipeline. The most score-relevant minimal fix is to use a loss that better matches AUC on imbalanced data: switch HGB to `loss="binary_crossentropy"` (still probabilistic classification, same model family/loop), which typically improves ranking vs log-loss. Additionally, we add one strong, competition-specific derived feature (`site_x_agebin`) to capture a simple interaction the trees can exploit without changing the modeling approach. Everything else (group-aware OOF, 5-model diversity, compression loop, rank normalization, and writing `submission.csv` aligned to `sample_submission.csv`) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing_file(rel_path: str) -> str:
    for root in DATA_ROOT_CANDIDATES:
        path = os.path.join(root, rel_path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {rel_path} in any of: {DATA_ROOT_CANDIDATES}"
    )


train_path = _first_existing_file("train.csv")
test_path = _first_existing_file("test.csv")
sample_sub_path = _first_existing_file("sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "target" in train_df.columns
assert "image_name" in test_df.columns
assert list(sample_sub.columns) == ["image_name", "target"]

train_df.head(), test_df.head(), sample_sub.head()




## === cell 1
def get_means(preds):
    preds = np.asarray(preds, dtype=np.float64)
    preds = np.clip(preds, 1e-12, 1 - 1e-12)
    gmean = scipy.stats.gmean(preds, axis=1)
    average = np.array(np.mean(preds, axis=1))
    median = np.median(preds, axis=1)
    return gmean, average, median




## === cell 2
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1

    _HAS_SGKF = True
except Exception:
    StratifiedGroupKFold = None
    _HAS_SGKF = False


def _normalize_str_to_nan(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip()
    s = s.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "UNKNOWN": pd.NA,
            "unknown": pd.NA,
        }
    )
    return s.astype(object)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["age_approx"] = pd.to_numeric(out["age_approx"], errors="coerce")
    out["age_isna"] = out["age_approx"].isna().astype(np.int8)

    out["age_filled"] = out["age_approx"].fillna(out["age_approx"].median())
    out["age_log1p"] = np.log1p(
        np.clip(out["age_filled"].astype(np.float64), 0, None)
    ).astype(np.float64)
    out["age_x_isna"] = (
        out["age_filled"].astype(np.float64) * out["age_isna"].astype(np.float64)
    ).astype(np.float64)

    out["age_bin"] = pd.cut(
        out["age_approx"], bins=[0, 30, 45, 60, 75, 120], include_lowest=True
    )
    out.loc[out["age_approx"].isna(), "age_bin"] = pd.NA

    out["patient_id"] = _normalize_str_to_nan(out["patient_id"]).fillna(
        "unknown_patient"
    )
    out["sex"] = _normalize_str_to_nan(out["sex"])
    out["anatom_site_general_challenge"] = _normalize_str_to_nan(
        out["anatom_site_general_challenge"]
    )

    out["sex_x_site"] = (
        out["sex"].astype("string").fillna("MISSING_SEX")
        + "__"
        + out["anatom_site_general_challenge"].astype("string").fillna("MISSING_SITE")
    ).astype(object)

    out["site_isna"] = out["anatom_site_general_challenge"].isna().astype(np.int8)

    out["site_x_agebin"] = (
        out["anatom_site_general_challenge"].astype("string").fillna("MISSING_SITE")
        + "__"
        + out["age_bin"].astype("string").fillna("MISSING_AGEBIN")
    ).astype(object)

    return out


def _iter_folds_stratified_group(X_df, y_arr, groups_arr, n_splits=5, seed=42):
    """
    Provides (train_idx, valid_idx) with approximate stratification and strict grouping.

    Uses StratifiedGroupKFold when available; otherwise falls back to:
    - group-level label = mean(y) per group (then binarized at 0.5)
    - StratifiedKFold over groups

    NOTE: To improve ensemble diversity (and thus AUC) without changing core logic,
    we allow the fold shuffling seed to vary per base model.
    """
    if _HAS_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in sgkf.split(X_df, y_arr, groups=groups_arr):
            yield tr_idx, va_idx
    else:
        grp = pd.Series(groups_arr, index=np.arange(len(groups_arr)))
        y_s = pd.Series(y_arr, index=np.arange(len(y_arr)))
        grp_target = y_s.groupby(grp).mean()
        grp_y = (grp_target >= 0.5).astype(int).values
        grp_ids = grp_target.index.to_numpy()

        skf_groups = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for grp_tr, grp_va in skf_groups.split(grp_ids, grp_y):
            tr_groups = set(grp_ids[grp_tr])
            va_groups = set(grp_ids[grp_va])
            tr_idx = np.where(grp.isin(tr_groups).to_numpy())[0]
            va_idx = np.where(grp.isin(va_groups).to_numpy())[0]
            yield tr_idx, va_idx


train_fe = add_features(train_df)
test_fe = add_features(test_df)

feature_cols = [
    "sex",
    "age_approx",
    "age_isna",
    "age_filled",
    "age_log1p",
    "age_x_isna",
    "age_bin",
    "anatom_site_general_challenge",
    "sex_x_site",
    "site_isna",
    "site_x_agebin",
]

X = train_fe[feature_cols].copy()
y = train_fe["target"].astype(int).values
groups = train_fe["patient_id"].astype(str).values  # keep for GroupKFold only
X_test = test_fe[feature_cols].copy()

numeric_features = [
    "age_approx",
    "age_isna",
    "age_filled",
    "age_log1p",
    "age_x_isna",
    "site_isna",
]
categorical_features = [
    "sex",
    "age_bin",
    "anatom_site_general_challenge",
    "sex_x_site",
    "site_x_agebin",
]

X_all_for_prep = pd.concat([X, X_test], axis=0, ignore_index=True)

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ord",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
    sparse_threshold=0.0,
)
preprocess.fit(X_all_for_prep)

seeds = [RANDOM_STATE + i for i in range(5)]

hgb_param_grid = [
    {
        "learning_rate": 0.06,
        "max_depth": 3,
        "min_samples_leaf": 30,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 25,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.07,
        "max_depth": 3,
        "min_samples_leaf": 35,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.06,
        "max_depth": 2,
        "min_samples_leaf": 30,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.06,
        "max_depth": 4,
        "min_samples_leaf": 30,
        "l2_regularization": 0.0,
    },
]

models = []
for sd, hp in zip(seeds, hgb_param_grid):
    models.append(
        Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        loss="binary_crossentropy",
                        learning_rate=hp["learning_rate"],
                        max_depth=hp["max_depth"],
                        max_leaf_nodes=63,
                        min_samples_leaf=hp["min_samples_leaf"],
                        l2_regularization=hp["l2_regularization"],
                        max_iter=300,
                        random_state=sd,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    )

oof_preds_5 = np.zeros((len(train_fe), len(models)), dtype=float)
oof_aucs = []
for mi, m in enumerate(models):
    oof_pred = np.zeros(len(train_fe), dtype=float)
    fold_seed = seeds[mi]
    for tr_idx, va_idx in _iter_folds_stratified_group(
        X, y, groups, n_splits=5, seed=fold_seed
    ):
        mm = clone(m)
        mm.fit(X.iloc[tr_idx], y[tr_idx])
        oof_pred[va_idx] = mm.predict_proba(X.iloc[va_idx])[:, 1]
    oof_preds_5[:, mi] = oof_pred
    oof_aucs.append(roc_auc_score(y, oof_pred))

print(
    "Sanity OOF AUCs (metadata-only, group OOF):",
    ", ".join([f"{a:.5f}" for a in oof_aucs]),
)
print(f"Mean OOF AUC: {float(np.mean(oof_aucs)):.5f}")

all_preds = pd.DataFrame({"image_name": test_df["image_name"].values})
for i, m in enumerate(models, start=1):
    m.fit(X, y)
    all_preds[str(i)] = m.predict_proba(X_test)[:, 1].astype(np.float64)

submission = sample_sub.copy()
all_preds.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/447203740.py in <cell line: 0>()
    189     sparse_threshold=0.0,
    190 )
--> 191 preprocess.fit(X_all_for_prep)
    192 
    193 seeds = [RANDOM_STATE + i for i in range(5)]

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in fit(self, X, y)
    692         # we use fit_transform to make sure to set sparse_output_ (for which we
    693         # need the transformed data) to have consistent output type in predict
--> 694         self.fit_transform(X, y=y)
    695         return self
    696 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in fit_transform(self, X, y)
    725         self._validate_remainder(X)
    726 
--> 727         result = self._fit_transform(X, y, _fit_transform_one)
    728 
    729         if not result:

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in _fit_transform(self, X, y, func, fitted, column_as_strings)
    656         )
    657         try:
--> 658             return Parallel(n_jobs=self.n_jobs)(
    659                 delayed(func)(
    660                     transformer=clone(trans) if not fitted else trans,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1
-> 1914                 res = func(*args, **kwargs)
   1915                 self.n_completed_tasks += 1
   1916                 self.print_progress()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
    121             config = {}
    122         with config_context(**config):
--> 123             return self.function(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
    435         """
    436         fit_params_steps = self._check_fit_params(**fit_params)
--> 437         Xt = self._fit(X, y, **fit_params_steps)
    438 
    439         last_step = self._final_estimator

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/impute/_base.py in fit(self, X, y)
    427 
    428         else:
--> 429             self.statistics_ = self._dense_fit(
    430                 X, self.strategy, self.missing_values, fill_value
    431             )

/usr/local/lib/python3.11/dist-packages/sklearn/impute/_base.py in _dense_fit(self, X, strategy, missing_values, fill_value)
    477     def _dense_fit(self, X, strategy, missing_values, fill_value):
    478         """Fit the transformer on dense data."""
--> 479         missing_mask = _get_mask(X, missing_values)
    480         masked_X = ma.masked_array(X, mask=missing_mask)
    481 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_mask.py in _get_mask(X, value_to_mask)
     51         # For all cases apart of a sparse input where we need to reconstruct
     52         # a sparse output
---> 53         return _get_dense_mask(X, value_to_mask)
     54 
     55     Xt = _get_dense_mask(X.data, value_to_mask)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_mask.py in _get_dense_mask(X, value_to_mask)
     24         else:
     25             # np.isnan does not work on object dtypes.
---> 26             Xt = _object_dtype_isnan(X)
     27     else:
     28         Xt = X == value_to_mask

/usr/local/lib/python3.11/dist-packages/sklearn/utils/fixes.py in _object_dtype_isnan(X)
     43 
     44 def _object_dtype_isnan(X):
---> 45     return X != X
     46 
     47 

missing.pyx in pandas._libs.missing.NAType.__bool__()

TypeError: boolean value of NA is ambiguous

## === cell 3
preds = all_preds[["1", "2", "3", "4", "5"]]
means = get_means(preds)
preds = np.transpose(means)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(preds[:, 0] - preds[:, 1], bins=100)
plt.title("Geometric - Average")
plt.subplot(1, 3, 2)
plt.hist(preds[:, 0] - preds[:, 2], bins=100)
plt.title("Geometric - Median")
plt.subplot(1, 3, 3)
plt.hist(preds[:, 1] - preds[:, 2], bins=100)
plt.title("Average - Median")
plt.show()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1462169990.py in <cell line: 0>()
----> 1 preds = all_preds[["1", "2", "3", "4", "5"]]
      2 means = get_means(preds)
      3 preds = np.transpose(means)
      4 
      5 plt.figure(figsize=(15, 5))

NameError: name 'all_preds' is not defined

## === cell 4
preds.shape



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1957060910.py in <cell line: 0>()
----> 1 preds.shape
      2 

NameError: name 'preds' is not defined

## === cell 5
preds = all_preds[["1", "2", "3", "4", "5"]]
n_repeat = 10
stds = []
mns = []
for _ in range(n_repeat):
    means = get_means(preds)
    stds += [np.std(means, axis=1)]
    mns += [np.mean(means, axis=1)]
    preds = np.transpose(means)

len(stds), len(mns)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3726792084.py in <cell line: 0>()
----> 1 preds = all_preds[["1", "2", "3", "4", "5"]]
      2 n_repeat = 10
      3 stds = []
      4 mns = []
      5 for _ in range(n_repeat):

NameError: name 'all_preds' is not defined

## === cell 6
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for i in range(3):
    plt.plot(np.stack(mns, axis=0)[:, i])
plt.title("Mean of (gmean, avg, median) over iterations")

plt.subplot(1, 2, 2)
for i in range(3):
    plt.plot(np.stack(stds, axis=0)[:, i])
plt.title("Std of (gmean, avg, median) over iterations")
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/328575318.py in <cell line: 0>()
      2 plt.subplot(1, 2, 1)
      3 for i in range(3):
----> 4     plt.plot(np.stack(mns, axis=0)[:, i])
      5 plt.title("Mean of (gmean, avg, median) over iterations")
      6 

NameError: name 'mns' is not defined

## === cell 7
for i in range(3):
    plt.hist(preds[:, i], bins=100, alpha=0.5)
    plt.title(f"Iteration-compressed preds component {i}")
    plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2791421670.py in <cell line: 0>()
      1 for i in range(3):
----> 2     plt.hist(preds[:, i], bins=100, alpha=0.5)
      3     plt.title(f"Iteration-compressed preds component {i}")
      4     plt.show()
      5 

NameError: name 'preds' is not defined

## === cell 8
test_pred_raw = preds[:, 0].astype(np.float64)
base_mean = all_preds[["1", "2", "3", "4", "5"]].mean(axis=1).to_numpy(dtype=np.float64)
alpha = 0.15
test_pred_raw = (1.0 - alpha) * test_pred_raw + alpha * base_mean

ranks = pd.Series(test_pred_raw).rank(method="average").to_numpy(dtype=np.float64)
test_pred = (ranks - 0.5) / len(ranks)

sub = pd.DataFrame({"image_name": all_preds["image_name"].values, "target": test_pred})
sub = sample_sub[["image_name"]].merge(sub, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(sub["target"].mean())

sub.to_csv("submission.csv", index=False)
sub.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3940490461.py in <cell line: 0>()
----> 1 test_pred_raw = preds[:, 0].astype(np.float64)
      2 base_mean = all_preds[["1", "2", "3", "4", "5"]].mean(axis=1).to_numpy(dtype=np.float64)
      3 alpha = 0.15
      4 test_pred_raw = (1.0 - alpha) * test_pred_raw + alpha * base_mean
      5 

NameError: name 'preds' is not defined

## === cell 9
plt.hist(sub.target, bins=100)
plt.title("Submission target distribution")
plt.show()

print("Wrote submission.csv with shape:", sub.shape)
print(sub.columns.tolist())
print(sub.head(3).to_string(index=False))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/115198725.py in <cell line: 0>()
----> 1 plt.hist(sub.target, bins=100)
      2 plt.title("Submission target distribution")
      3 plt.show()
      4 
      5 print("Wrote submission.csv with shape:", sub.shape)

NameError: name 'sub' is not defined
