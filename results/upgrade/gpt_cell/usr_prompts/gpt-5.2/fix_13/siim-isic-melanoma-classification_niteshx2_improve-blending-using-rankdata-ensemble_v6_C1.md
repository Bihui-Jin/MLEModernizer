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

0.8856

# 6. Current score

0.37978

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Cell 5 crashes because it assumes `dfs` contains at least 4 prediction DataFrames, but in this environment the `efficientnets` folder is not present so `dfs` is empty. The minimal fix is to make cell 5 robust: if 4+ DataFrames exist, keep the original averaging logic; otherwise, fall back to the competition’s `sample_submission.csv` so later cells still have a valid DataFrame to work with. This preserves the original ensemble semantics when possible and prevents an IndexError when the external inputs are missing. The output remains a DataFrame with `image_name` and `target`, matching what cell 6 expects.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score is coming from the fallback path outputting the unmodified `sample_submission.csv` (all targets identical), which yields ROC-AUC ≈ 0.5. To move toward the 0.8856 target while keeping the same “ensemble of external prediction CSVs + rank-averaging” core logic, I make the code load predictions from any available CSVs in the `efficientnets` directory (not only the first 4), align them by `image_name` before averaging, and only use the constant sample submission if no prediction files exist. This preserves the original rank-ensemble semantics but fixes a common silent issue (row-order mismatch) that can severely depress AUC. The script still write `sol.csv` with exactly `image_name,target`.'
- What this solution (achieved 0.66776) has done: 'I make the submission generation robust and score-improving when the external `efficientnets` predictions are missing by (1) guaranteeing the output is aligned to the exact `sample_submission.csv` order and row count, and (2) upgrading the fallback metadata model slightly (still a simple logistic regression) via cross-validated out-of-fold training with a patient-aware split to reduce leakage and improve ROC-AUC versus the single fit. I also fix a common AUC-killer in rank-ensembles: ensure ranking is applied after alignment and then scale ranks to (0,1) probabilities, which preserves ordering (thus AUC) but avoids pathological integer-valued outputs. These are minimal changes that preserve your core “load external preds → rank-average; else metadata LR fallback” logic while ensuring a valid `sol.csv` is always produced. The final cell always write `sol.csv` with exactly `image_name,target` and 4142 rows.'
- What this solution (achieved 0.37647) has done: 'Your current score (0.66776) is well below the 0.8856 target, so we should improve the metadata fallback without changing the overall “external preds → rank-average; else metadata LR fallback” core logic. The biggest AUC limiter in the fallback is that it ignores strong metadata signals (`diagnosis` and `benign_malignant` are train-only but can be used to learn better patterns via group-aware CV without leaking into test labels). I add safe, leakage-free target-encoding of `diagnosis` and `benign_malignant` computed out-of-fold with `GroupKFold(patient_id)` and then refit-fold-predict for test, then keep your existing alignment + rank-scaling submission semantics unchanged. This is a minimal, metric-aligned change that typically lifts ROC-AUC notably for this competition while staying within sklearn/pandas/scipy constraints.'
- What this solution (achieved 0.38754) has done: 'Your low AUC is mainly coming from the fallback metadata model producing weak, poorly-calibrated rankings; the biggest safe gain (without changing the overall “external preds → else metadata model → rank-scale submission” logic) is to make the fallback predict *a better ranking* by adding strong non-leaky metadata features and fixing category handling. I keep your ensemble/rank-averaging semantics identical, but in the fallback I (1) include `patient_id` as a categorical feature (it exists in both train/test and is very predictive for this competition), and (2) fix the target-encoding for train-only columns so test is encoded from train mappings (instead of passing all-missing, which currently nullifies the added TE). These are minimal changes that usually move AUC substantially upward toward your 0.8856 target while keeping the submission alignment + rank scaling exactly as you already do.'
- What this solution (achieved 0.38754) has done: 'I fix the biggest score-killer in your current fallback: the target-encoded features for `diagnosis` and `benign_malignant` are being mapped on test using an all-“missing” placeholder, so they carry almost no signal. I keep your core logic unchanged (external prediction rank-ensemble if available; otherwise metadata LogisticRegression with GroupKFold), but correctly apply each learned target-encoding mapping to the real test column. I also make the merge step keep all test rows (not inner-join shrink) while still aligning to `sample_submission.csv`, preventing accidental row loss/misalignment that can severely hurt AUC. The final output remains rank-scaled and written to `sol.csv` with exactly `image_name,target` and 4142 rows.'
- What this solution (achieved 0.37978) has done: 'I keep your core “external prediction CSVs → rank-average; else metadata LogisticRegression fallback → rank-scale submission” logic unchanged, but fix two score-killers in the fallback: (1) `age_approx` is being read as strings like `"45.0"`/`"nan"` so the numeric branch effectively becomes mostly missing after coercion, and (2) the patient-aware CV can silently error or degrade if `patient_id` has missing/blank values treated inconsistently between train/test. I minimally coerce `age_approx` to numeric in both train/test (preserving missingness) and standardize key categorical columns (`patient_id`, `sex`, `anatom_site_general_challenge`) with consistent string fill values before encoding. This should improve the ranking signal (AUC) of the metadata fallback toward your 0.8856 target without changing model type, loss, or the rank-based submission semantics. The script still always write a valid `sol.csv` with exactly 4142 rows aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
candidate_paths = [
    "/kaggle/input/efficientnets",
    "../input/efficientnets",
    "input/efficientnets",
    "/kaggle/input",
    "../input",
    "input",
    "/kaggle/data",
    "data",
    ".",
]

BASE_PATH = None
for p in candidate_paths:
    if os.path.isdir(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    BASE_PATH = "."


def find_dir_named(root, name):
    if os.path.isdir(os.path.join(root, name)):
        return os.path.join(root, name)
    try:
        for d in os.listdir(root):
            full = os.path.join(root, d)
            if os.path.isdir(full) and os.path.isdir(os.path.join(full, name)):
                return os.path.join(full, name)
    except Exception:
        pass
    return None


efficientnets_dir = None
for root in [BASE_PATH, "/kaggle/input", "/kaggle/data", "input", "data", "."]:
    if root is None or not os.path.isdir(root):
        continue
    candidate = find_dir_named(root, "efficientnets")
    if candidate is not None:
        efficientnets_dir = candidate
        break

EFFICIENTNETS_AVAILABLE = efficientnets_dir is not None and os.path.isdir(
    efficientnets_dir
)
print("BASE_PATH:", BASE_PATH)
print("efficientnets_dir:", efficientnets_dir)
print("EFFICIENTNETS_AVAILABLE:", EFFICIENTNETS_AVAILABLE)



## === cell 2
dfs = []
if EFFICIENTNETS_AVAILABLE:
    for df_loc in sorted(os.listdir(efficientnets_dir)):
        full_path = os.path.join(efficientnets_dir, df_loc)
        if not full_path.lower().endswith(".csv"):
            continue
        df = pd.read_csv(full_path)
        if {"image_name", "target"}.issubset(df.columns):
            dfs.append(df[["image_name", "target"]].copy())

print("Loaded prediction files:", len(dfs))



## === cell 3
from scipy.stats import rankdata

test_candidates = [
    os.path.join(BASE_PATH, "test.csv"),
    "/kaggle/data/test.csv",
    "data/test.csv",
    "/kaggle/input/test.csv",
    "input/test.csv",
    "/kaggle/input/siim-isic-melanoma-classification/test.csv",
    "input/siim-isic-melanoma-classification/test.csv",
    "data/siim-isic-melanoma-classification/test.csv",
]
test_path = next((p for p in test_candidates if os.path.isfile(p)), None)
test_df = pd.read_csv(test_path) if test_path is not None else None
if test_df is not None and "image_name" in test_df.columns:
    test_image_names = test_df[["image_name"]].drop_duplicates()
else:
    test_image_names = None

if len(dfs) > 0:
    if test_image_names is None:
        merged = None
        for j, df in enumerate(dfs):
            d = df.rename(columns={"target": f"target_{j}"}).copy()
            d = d.dropna(subset=["image_name"]).drop_duplicates(
                subset=["image_name"], keep="first"
            )
            merged = (
                d if merged is None else merged.merge(d, on="image_name", how="inner")
            )
    else:
        merged = test_image_names.copy()
        for j, df in enumerate(dfs):
            d = df.rename(columns={"target": f"target_{j}"}).copy()
            d = d.dropna(subset=["image_name"]).drop_duplicates(
                subset=["image_name"], keep="first"
            )
            merged = merged.merge(d, on="image_name", how="left")

    if merged is None or merged.shape[0] == 0:
        dfs = []
    else:
        target_cols = [c for c in merged.columns if c.startswith("target_")]
        if len(target_cols) == 0:
            dfs = []
        else:
            for c in target_cols:
                if merged[c].isna().any():
                    merged[c] = merged[c].fillna(merged[c].median())

            for c in target_cols:
                merged[c] = rankdata(merged[c].values, method="average")

            merged["target"] = merged[target_cols].mean(axis=1)
            dfs = [merged[["image_name", "target"]].copy()]

print("Post-merge dfs length:", len(dfs))
if len(dfs) > 0:
    print(dfs[0].head())



## === cell 4
if len(dfs) == 0:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    train_candidates = [
        os.path.join(BASE_PATH, "train.csv"),
        "/kaggle/data/train.csv",
        "data/train.csv",
        "/kaggle/input/train.csv",
        "input/train.csv",
        "/kaggle/input/siim-isic-melanoma-classification/train.csv",
        "input/siim-isic-melanoma-classification/train.csv",
        "data/siim-isic-melanoma-classification/train.csv",
    ]
    test_candidates = [
        os.path.join(BASE_PATH, "test.csv"),
        "/kaggle/data/test.csv",
        "data/test.csv",
        "/kaggle/input/test.csv",
        "input/test.csv",
        "/kaggle/input/siim-isic-melanoma-classification/test.csv",
        "input/siim-isic-melanoma-classification/test.csv",
        "data/siim-isic-melanoma-classification/test.csv",
    ]
    train_path = next((p for p in train_candidates if os.path.isfile(p)), None)
    test_path = next((p for p in test_candidates if os.path.isfile(p)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not locate train.csv/test.csv for metadata fallback model."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for _df in (train_df, test_df):
        if "age_approx" in _df.columns:
            _df["age_approx"] = pd.to_numeric(_df["age_approx"], errors="coerce")

    def _clean_cat(s, missing="missing"):
        s = s.astype("string")
        s = s.fillna(missing)
        s = s.str.strip()
        s = s.replace("", missing)
        return s

    for _df in (train_df, test_df):
        for c, miss in [
            ("patient_id", "unknown"),
            ("sex", "unknown"),
            ("anatom_site_general_challenge", "unknown"),
        ]:
            if c in _df.columns:
                _df[c] = _clean_cat(_df[c], missing=miss)

    def _oof_target_encode(train_series, y, groups, n_splits=5, smoothing=50.0):
        """Return (oof_encoded, map_test_fn) where encoding is computed OOF by groups."""
        gkf_local = GroupKFold(n_splits=n_splits)
        global_mean = float(np.mean(y))
        oof = np.empty(len(train_series), dtype=np.float64)
        oof[:] = np.nan

        s_all = train_series.astype("string").fillna("missing")

        for tr_idx, va_idx in gkf_local.split(s_all, y, groups=groups):
            s_tr = s_all.iloc[tr_idx]
            y_tr = y[tr_idx]

            stats = pd.DataFrame({"cat": s_tr.values, "y": y_tr})
            agg = stats.groupby("cat")["y"].agg(["mean", "count"])
            enc = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
                agg["count"] + smoothing
            )

            s_va = s_all.iloc[va_idx]
            oof[va_idx] = s_va.map(enc).astype(float).fillna(global_mean).values

        stats_full = pd.DataFrame({"cat": s_all.values, "y": y})
        agg_full = stats_full.groupby("cat")["y"].agg(["mean", "count"])
        enc_full = (agg_full["mean"] * agg_full["count"] + global_mean * smoothing) / (
            agg_full["count"] + smoothing
        )

        def map_test(test_series):
            s_test = test_series.astype("string").fillna("missing")
            return s_test.map(enc_full).astype(float).fillna(global_mean).values

        return oof, map_test

    feature_cols_base = [
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "patient_id",
    ]

    te_cols = []
    for c in ["diagnosis", "benign_malignant"]:
        if c in train_df.columns:
            te_cols.append(c)

    groups = (
        train_df["patient_id"].astype(str).fillna("unknown").values
        if "patient_id" in train_df.columns
        else np.arange(len(train_df))
    )
    y = train_df["target"].astype(int).values

    X_train = train_df.reindex(columns=feature_cols_base).copy()
    X_test = test_df.reindex(columns=feature_cols_base).copy()

    te_feature_names = []
    if len(te_cols) > 0:
        for c in te_cols:
            oof_enc, map_test = _oof_target_encode(
                train_df[c], y, groups=groups, n_splits=5, smoothing=50.0
            )
            new_col = f"te_{c}"
            X_train[new_col] = oof_enc
            if c in test_df.columns:
                X_test[new_col] = map_test(test_df[c])
            else:
                X_test[new_col] = map_test(pd.Series(["missing"] * len(test_df)))
            te_feature_names.append(new_col)

    numeric_features = ["age_approx"] + te_feature_names
    categorical_features = ["sex", "anatom_site_general_challenge", "patient_id"]

    preprocessor = ColumnTransformer(
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
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=600,
        C=1.0,
        class_weight="balanced",
        random_state=0,
    )

    model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    test_pred_folds = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y, groups=groups), 1):
        model.fit(X_train.iloc[tr_idx], y[tr_idx])
        test_pred_folds.append(model.predict_proba(X_test)[:, 1])

    test_pred = np.mean(np.vstack(test_pred_folds), axis=0)
    dfs = [
        pd.DataFrame({"image_name": test_df["image_name"].values, "target": test_pred})
    ]

dfs[0].head()



## === cell 5
sub_candidates = [
    os.path.join(BASE_PATH, "sample_submission.csv"),
    "/kaggle/data/sample_submission.csv",
    "data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "input/sample_submission.csv",
    "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv",
    "input/siim-isic-melanoma-classification/sample_submission.csv",
    "data/siim-isic-melanoma-classification/sample_submission.csv",
]
sub_path = next((p for p in sub_candidates if os.path.isfile(p)), None)
if sub_path is None:
    raise FileNotFoundError("Could not locate sample_submission.csv to align output.")
sample_sub = pd.read_csv(sub_path)[["image_name", "target"]].copy()

pred = dfs[0][["image_name", "target"]].copy()
pred = pred.dropna(subset=["image_name"]).drop_duplicates(
    subset=["image_name"], keep="first"
)

aligned = sample_sub[["image_name"]].merge(pred, on="image_name", how="left")
if aligned["target"].isna().any():
    aligned["target"] = aligned["target"].fillna(np.nanmedian(aligned["target"].values))

r = rankdata(aligned["target"].values, method="average")
n = len(r)
aligned["target"] = (r - 0.5) / n
aligned[["image_name", "target"]].to_csv("sol.csv", index=False)

print("Wrote sol.csv with shape:", aligned.shape)
print(aligned.head())
print("target min/max:", float(aligned["target"].min()), float(aligned["target"].max()))
