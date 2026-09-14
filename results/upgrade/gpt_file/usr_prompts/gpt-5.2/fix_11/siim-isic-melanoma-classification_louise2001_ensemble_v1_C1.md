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

0.8992161873869151

# 6. Current score

0.65711

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code doesn’t yield a score because it tries to read OOF/submission files from `/kaggle/input/melanoma`, which doesn’t exist in your provided dataset paths, so it never create the `target_*` columns and error when taking their mean. To make it run end-to-end and produce a valid `submission.csv`, I (1) switch the blending source directory to the actual competition input folder, (2) automatically discover and load any valid submission-like CSVs (with `image_name` plus a probability column), and (3) add a safe fallback that outputs the sample submission (all zeros) if no blend files are found, ensuring a valid file is always created. This preserves your core logic (simple mean blending of multiple CSV predictions) while making it executable in this environment.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score is consistent with the fallback path that outputs a constant prediction (all zeros), which yields an AUC of 0.5. To move toward the target (~0.899), the smallest legitimate improvement while preserving your “blend CSVs and mean them” core logic is to (1) prioritize finding real model prediction CSVs inside the competition dataset directory (including common locations like `/kaggle/working`), (2) accept typical submission-like schemas (either `target` or a single non-`image_name` column), and (3) ensure robust alignment (deduplicate by `image_name`, keep test order) so the blended predictions aren’t accidentally broken. If no valid prediction files exist, it still safely output a valid submission (but that remain near 0.5). These changes don’t change the blending method; they just make it far more likely you actually blend real predictions instead of outputting zeros.'
- What this solution (achieved 0.66789) has done: 'Your 0.5 AUC indicates the pipeline is still falling back to constant predictions (no real prediction CSVs are being found/used). To move the score toward the ~0.899 target while preserving your “blend CSVs and mean them” core logic, the smallest effective change is to actually generate one legitimate set of predictions from the provided `train.csv`/`test.csv` metadata (a simple logistic regression baseline), write it as a candidate CSV, and then let your existing blending code pick it up and average it. This keeps the blending semantics identical (mean of CSV prediction columns), but ensures there is at least one non-trivial prediction source so AUC rises above 0.5. I’m also constraining the file search to avoid accidentally blending `submission.csv` with itself.'
- What this solution (achieved 0.66282) has done: 'Your current gap to the target AUC is large (0.66789 vs 0.8992), so we need a meaningful but still minimal change that preserves your overall approach (generate one prediction CSV, then mean-blend all candidate CSVs). The simplest lift is to make the metadata model a bit more expressive without changing the pipeline/training paradigm: switch logistic regression to use `class_weight="balanced"` (helps severe imbalance) and allow weak nonlinearity by adding `PolynomialFeatures` on `age_approx` only, while keeping the same sklearn Pipeline structure and predict_proba semantics. To prevent accidental self-blending or blending junk files, we also tighten CSV discovery to only include files that match the test set exactly (all `image_name`s), which reduces noise and should move AUC upward more reliably. The rest (merge/alignment/mean blend/submission writing) is unchanged.'
- What this solution (achieved 0.66613) has done: 'Your current score (0.66282) is far below the target (0.8992), so we should make a small, legitimate improvement to the metadata model while keeping your core “train a simple sklearn model → write a candidate CSV → mean-blend any found CSVs” logic unchanged. The biggest likely lift with minimal disruption is to let the linear model learn non-linear interactions between the categorical one-hots and age by moving `PolynomialFeatures` to act on the *full preprocessed feature space* (after one-hot), rather than only on `age_approx`. We keep the same model family (logistic regression), same predict_proba semantics, and the same blending code, only slightly increasing expressiveness to push AUC upward toward your target. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.66282) has done: 'Your current AUC (0.666) is far below the target (0.899), so we should improve the single metadata model that feeds your existing “write candidate CSV → discover CSVs → mean-blend” pipeline, without changing the blending logic. The smallest high-impact adjustment within the same LogisticRegression approach is to prevent the degree-2 PolynomialFeatures from exploding the one-hot space (which tends to overfit/noise and can hurt generalization) by applying polynomial expansion only to the numeric age feature, while keeping the categorical one-hot linear. Additionally, we add a minimal, leakage-safe grouped cross-validation (by `patient_id`) to learn a single blend weight between the raw model probability and the global prior; this is just calibration/shrinkage and often improves ROC-AUC stability on this dataset. Everything still trains one sklearn model, writes a candidate prediction CSV, then blends found CSVs exactly as before and writes `submission.csv`.'
- What this solution (achieved 0.65711) has done: 'Your current gap to the target AUC is large, so the most reliable way to move upward without changing your overall approach is to improve the *single metadata model* that feeds your existing “write candidate CSV → discover CSVs → mean-blend” pipeline. I keep the same LogisticRegression + preprocessing + GroupKFold shrinkage calibration, but add a minimal, leakage-safe patient-level aggregation of the same metadata (counts and malignant-rate per site/sex) computed on train and merged into both train/test; this often gives a meaningful AUC lift on this competition while preserving the same training paradigm. I also make the shrinkage weight selection properly out-of-fold by choosing `w` on OOF predictions only (already) and then applying it to the final model’s test probabilities (unchanged). Blending logic and submission writing stay the same, but now the candidate predictions should be stronger and pull your score closer to 0.899.'
- What this solution (achieved 0.65711) has done: 'Your current AUC (0.657) is far below the target (0.899), so we should improve the *single metadata model* that feeds your existing “write candidate CSV → discover CSVs → mean-blend” pipeline, without changing the blending logic. The biggest issue in your feature engineering is leakage/shift: the aggregate “malignant-rate per sex/site” is computed on the full training set, then used inside each fold, which inflates OOF selection and can hurt true generalization. I make those aggregated features strictly out-of-fold for training (GroupKFold by patient_id) and computed on the full train only for test, while keeping the same LogisticRegression + preprocessing + shrinkage calibration semantics. This is a minimal change localized to feature engineering/cross-validation consistency and should move AUC upward toward your target.'
- What this solution (achieved 0.65711) has done: 'We keep your overall pipeline (metadata LR → write a candidate CSV → discover CSVs → mean-blend → write submission) identical, but make two minimal fixes that should legitimately raise AUC toward the 0.899 target. First, we correct a key mismatch bug in the engineered feature names: your feature list expects `anatom_site_general_challenge_count/mal_rate`, but the feature engineering currently creates `anatom_site_general_challenge_mal_rate` without the `_challenge` part, effectively breaking those features; fixing this improves signal without changing the model family or training loop. Second, we make the aggregate “malignant-rate” features use simple count-based smoothing toward the global prior (still computed leakage-safe out-of-fold for train, full-train for test), which is a small regularization/calibration change that usually improves generalization while preserving the same semantics and predict_proba usage. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.65711) has done: 'We keep your end-to-end pipeline (metadata LR → write candidate CSV → discover CSVs → mean-blend → write submission) exactly the same, but make one minimal, high-impact correction that is currently breaking a key engineered feature. Right now `feature_cols_num` expects `anatom_site_general_challenge_count`, but `_add_agg_features_*` actually creates `anatom_site_general_challenge_count` only if the grouping key matches exactly; we ensure the feature engineering uses the exact same key strings and produced column names consistently, and we also ensure `age_approx` is numeric (some folds can end up with strings after CSV read/impute), which can silently degrade the LR signal. These are localized fixes (no new models, no new loops, no metric trickery) and should move AUC upward from ~0.657 toward your ~0.899 target. Submission writing and blend discovery remain unchanged and still produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

BASE_DIR = "/kaggle/input/siim-isic-melanoma-classification"

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
f = pd.read_csv(sample_path)[["image_name"]]
print("Sample submission shape:", f.shape)



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

for _df in (train_df, test_df):
    _df["age_approx"] = pd.to_numeric(_df["age_approx"], errors="coerce")


def _normalize_cats(tr: pd.DataFrame, te: pd.DataFrame, cols):
    tr = tr.copy()
    te = te.copy()
    for c in cols:
        tr[c] = (
            tr[c]
            .astype(str)
            .replace({"nan": "unknown", "None": "unknown"})
            .fillna("unknown")
        )
        te[c] = (
            te[c]
            .astype(str)
            .replace({"nan": "unknown", "None": "unknown"})
            .fillna("unknown")
        )
    return tr, te


def _add_agg_features_fulltrain(
    train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float = 50.0
):
    keys = ["sex", "anatom_site_general_challenge"]

    tr, te = _normalize_cats(train_df, test_df, keys)
    global_prior = float(tr["target"].mean())

    for key in keys:
        grp = (
            tr.groupby(key, dropna=False)["target"].agg(["count", "mean"]).reset_index()
        )

        grp["sm_mean"] = (grp["count"] * grp["mean"] + alpha * global_prior) / (
            grp["count"] + alpha
        )

        grp = grp.rename(
            columns={
                "count": f"{key}_count",
                "sm_mean": f"{key}_mal_rate",
            }
        )[[key, f"{key}_count", f"{key}_mal_rate"]]

        tr = tr.merge(grp, on=key, how="left", validate="many_to_one")
        te = te.merge(grp, on=key, how="left", validate="many_to_one")

        te[f"{key}_count"] = te[f"{key}_count"].fillna(0.0)
        te[f"{key}_mal_rate"] = te[f"{key}_mal_rate"].fillna(global_prior)

    return tr, te


def _add_agg_features_oof(
    train_df: pd.DataFrame, groups, n_splits=5, alpha: float = 50.0
):
    keys = ["sex", "anatom_site_general_challenge"]

    tr, _dummy = _normalize_cats(train_df, train_df.iloc[:0].copy(), keys)
    tr = tr.copy()
    global_prior = float(tr["target"].mean())

    for key in keys:
        tr[f"{key}_count"] = np.nan
        tr[f"{key}_mal_rate"] = np.nan

    gkf = GroupKFold(n_splits=n_splits)
    y = tr["target"].to_numpy()

    for fit_idx, val_idx in gkf.split(tr, y, groups=groups):
        fit = tr.iloc[fit_idx]

        for key in keys:
            grp = fit.groupby(key, dropna=False)["target"].agg(["count", "mean"])

            key_vals = tr.iloc[val_idx][key].values
            counts = pd.Series(key_vals).map(grp["count"]).to_numpy(dtype=float)
            means = pd.Series(key_vals).map(grp["mean"]).to_numpy(dtype=float)

            counts = np.where(np.isfinite(counts), counts, 0.0)
            means = np.where(np.isfinite(means), means, global_prior)

            sm_means = (counts * means + alpha * global_prior) / (counts + alpha)

            tr.iloc[val_idx, tr.columns.get_loc(f"{key}_count")] = counts
            tr.iloc[val_idx, tr.columns.get_loc(f"{key}_mal_rate")] = sm_means

    for key in keys:
        tr[f"{key}_count"] = tr[f"{key}_count"].fillna(0.0)
        tr[f"{key}_mal_rate"] = tr[f"{key}_mal_rate"].fillna(global_prior)

    return tr


groups = train_df["patient_id"].astype(str).fillna("unknown").to_numpy()

train_df_fe_oof = _add_agg_features_oof(train_df, groups=groups, n_splits=5, alpha=50.0)
train_df_fe_full, test_df_fe = _add_agg_features_fulltrain(
    train_df, test_df, alpha=50.0
)

feature_cols_num = [
    "age_approx",
    "sex_count",
    "sex_mal_rate",
    "anatom_site_general_challenge_count",
    "anatom_site_general_challenge_mal_rate",
]
feature_cols_cat = ["sex", "anatom_site_general_challenge"]

X_train_oof = train_df_fe_oof[feature_cols_num + feature_cols_cat].copy()
X_train_full = train_df_fe_full[feature_cols_num + feature_cols_cat].copy()
y_train = train_df_fe_full["target"].astype(int).to_numpy()  # labels unchanged
X_test = test_df_fe[feature_cols_num + feature_cols_cat].copy()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
)

base_model = LogisticRegression(
    max_iter=400, solver="lbfgs", class_weight="balanced", n_jobs=None
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", base_model),
    ]
)

prior = float(np.mean(y_train))
candidate_ws = np.linspace(0.60, 1.00, 21)  # conservative shrinkage range
gkf = GroupKFold(n_splits=5)

oof = np.zeros(len(train_df_fe_oof), dtype=float)
for tr_idx, va_idx in gkf.split(X_train_oof, y_train, groups=groups):
    clf_fold = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=400, solver="lbfgs", class_weight="balanced", n_jobs=None
                ),
            ),
        ]
    )
    clf_fold.fit(X_train_oof.iloc[tr_idx], y_train[tr_idx])
    oof[va_idx] = clf_fold.predict_proba(X_train_oof.iloc[va_idx])[:, 1].astype(float)

oof = np.clip(oof, 0.0, 1.0)

best_w = 1.0
best_auc = -np.inf
for w in candidate_ws:
    oof_cal = np.clip(w * oof + (1.0 - w) * prior, 0.0, 1.0)
    auc = roc_auc_score(y_train, oof_cal)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)

print(f"OOF AUC (raw)      : {roc_auc_score(y_train, oof):.6f}")
print(f"OOF AUC (shrinkage): {best_auc:.6f} at w={best_w:.3f}, prior={prior:.6f}")

clf.fit(X_train_full, y_train)
proba_test = clf.predict_proba(X_test)[:, 1].astype(float)
proba_test = np.clip(proba_test, 0.0, 1.0)
proba_test = np.clip(best_w * proba_test + (1.0 - best_w) * prior, 0.0, 1.0)

meta_pred_path = "/kaggle/working/meta_lr_predictions.csv"
pd.DataFrame(
    {"image_name": test_df_fe["image_name"].astype(str), "target": proba_test}
).to_csv(meta_pred_path, index=False)
print("Wrote candidate prediction file:", meta_pred_path)



## === cell 2
blend_dirs = [
    "/kaggle/input/melanoma",  # keep original intent (may not exist here)
    BASE_DIR,  # competition input directory (exists)
    "/kaggle/input",  # broader input search (some notebooks save outputs here)
    "/kaggle/working",  # common place for generated submissions during runs
]

cols = []
found_files = []

test_names = set(test_df_fe["image_name"].astype(str).tolist())
n_test = len(test_df_fe)


def _read_candidate_csv(full_path: str):
    try:
        ff = pd.read_csv(full_path)
    except Exception:
        return None

    if "image_name" not in ff.columns:
        return None

    pred_col = None
    if "target" in ff.columns:
        pred_col = "target"
    else:
        pred_cols = [c for c in ff.columns if c != "image_name"]
        if len(pred_cols) == 1:
            pred_col = pred_cols[0]
        else:
            return None

    tmp = ff[["image_name", pred_col]].copy()

    tmp = tmp.dropna(subset=["image_name"])
    tmp["image_name"] = tmp["image_name"].astype(str)
    tmp = tmp.drop_duplicates(subset=["image_name"], keep="last")

    if len(tmp) != n_test:
        return None
    if set(tmp["image_name"].tolist()) != test_names:
        return None

    tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
    if tmp[pred_col].notna().mean() < 0.999:
        return None
    tmp[pred_col] = tmp[pred_col].clip(0.0, 1.0)

    return tmp


for bdir in blend_dirs:
    if not os.path.isdir(bdir):
        continue
    for root, _, filenames in os.walk(bdir):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            full_path = os.path.join(root, filename)

            if os.path.abspath(full_path) == os.path.abspath(sample_path):
                continue
            if os.path.basename(full_path).lower() in (
                "train.csv",
                "test.csv",
                "submission.csv",
            ):
                continue

            tmp = _read_candidate_csv(full_path)
            if tmp is None:
                continue

            new_col = f"target_{len(cols)}"
            tmp = tmp.rename(
                columns={c: new_col for c in tmp.columns if c != "image_name"}
            )

            before_n = len(f)
            f = f.merge(tmp, on="image_name", how="left", validate="one_to_one")
            after_n = len(f)
            if after_n != before_n:
                f = f.drop_duplicates(subset=["image_name"], keep="first")

            cols.append(new_col)
            found_files.append(full_path)

print("Found blend files:", len(found_files))
for p in found_files[:20]:
    print(" -", p)
print("Current merged shape:", f.shape)
print("Blend columns:", cols[:10], ("..." if len(cols) > 10 else ""))



## === cell 3
if len(cols) > 0:
    for c in cols:
        if f[c].isna().any():
            col_mean = float(np.nanmean(f[c].to_numpy(dtype=float)))
            if not np.isfinite(col_mean):
                col_mean = 0.5
            f[c] = f[c].fillna(col_mean)
    f["target"] = f[cols].mean(axis=1)
    f.drop(columns=cols, inplace=True)
else:
    f["target"] = 0.0

f["target"] = (
    pd.to_numeric(f["target"], errors="coerce").fillna(0.5).astype(float).clip(0.0, 1.0)
)

print(f.head())
print("Final submission shape:", f.shape)
print("Target describe:\n", f["target"].describe())



## === cell 4
f.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
