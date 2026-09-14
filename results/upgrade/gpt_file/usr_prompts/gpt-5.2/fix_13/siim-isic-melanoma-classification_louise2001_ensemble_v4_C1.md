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
sub_path = "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv"
f = pd.read_csv(sub_path, usecols=["image_name"], engine="c").copy()
print("Base submission shape:", f.shape)



## === cell 2
candidate_dirs = [
    "/kaggle/working",  # where users typically save inference outputs
    "/kaggle/input/siim-isic-melanoma-classification",  # competition dataset root (may contain notebooks' outputs)
    "/kaggle/input",  # kept for compatibility, but traversal is now tightly bounded and filename-filtered
]

ID_CANDIDATES = ("image_name", "id")
PRED_NAME_CANDIDATES = (
    "target",
    "pred",
    "preds",
    "prediction",
    "prob",
    "proba",
    "probability",
    "score",
    "oof",
)
SKIP_BASENAMES = {
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "folds.csv",
    "oof.csv",  # common but often not aligned with test set
}

SKIP_DIR_BASENAMES = {
    "jpeg",
    "tfrecords",
    "train",
    "test",
    ".ipynb_checkpoints",
    "__pycache__",
}


def _looks_like_prediction_csv(path: str) -> bool:
    base = os.path.basename(path).lower()
    if not base.endswith(".csv"):
        return False
    if base in SKIP_BASENAMES:
        return False
    return True


def _iter_candidate_csv_paths(roots, max_depth: int = 3):
    seen = set()

    def _scan_dir(root: str, cur: str, depth: int):
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            name = entry.name.lower()
                            if name in SKIP_DIR_BASENAMES:
                                continue
                            if depth < max_depth - 1:
                                _scan_dir(root, entry.path, depth + 1)
                        elif entry.is_file(follow_symlinks=False):
                            if not entry.name.lower().endswith(".csv"):
                                continue
                            p = entry.path
                            if p in seen:
                                continue
                            if not _looks_like_prediction_csv(p):
                                continue
                            seen.add(p)
                            yield p
                    except OSError:
                        continue
        except OSError:
            return

    for root in roots:
        if not os.path.isdir(root):
            continue
        root = os.path.abspath(root)
        yield from _scan_dir(root, root, 0)


def find_prediction_csvs(roots):
    return sorted(_iter_candidate_csv_paths(roots))


pred_csvs = find_prediction_csvs(candidate_dirs)
print("Found candidate prediction CSVs (bounded scan):", len(pred_csvs))
print("First few:", pred_csvs[:10])



## === cell 3
cols = []
used_files = 0

MIN_TEST_COVERAGE = 0.95


def _standardize_pred_df_from_columns(ff: pd.DataFrame):
    cols_lower = {c.lower(): c for c in ff.columns}

    id_col = None
    for cand in ID_CANDIDATES:
        if cand in ff.columns:
            id_col = cand
            break
        if cand in cols_lower:
            id_col = cols_lower[cand]
            break
    if id_col is None:
        return None

    pred_col = None
    if "target" in ff.columns:
        pred_col = "target"
    elif "target" in cols_lower:
        pred_col = cols_lower["target"]
    else:
        for cand in PRED_NAME_CANDIDATES:
            if cand in ff.columns:
                pred_col = cand
                break
            if cand in cols_lower:
                pred_col = cols_lower[cand]
                break

    if pred_col is None:
        other_cols = [c for c in ff.columns if c != id_col]
        if len(other_cols) != 1:
            return None
        pred_col = other_cols[0]

    return id_col, pred_col


f_idx = f.set_index("image_name", drop=True)
test_index = f_idx.index
n_test = len(test_index)

test_pos = pd.Series(np.arange(n_test, dtype=np.int32), index=test_index)

pred_matrix = None  # allocated on first accepted file

for path in pred_csvs:
    try:
        ff_head = pd.read_csv(path, nrows=0, engine="c")
    except Exception:
        continue

    std_cols = _standardize_pred_df_from_columns(ff_head)
    if std_cols is None:
        continue
    id_col, pred_col = std_cols

    try:
        preview = pd.read_csv(
            path,
            usecols=[id_col, pred_col],
            nrows=5000,
            engine="c",
            dtype={id_col: "string"},
        )
        if preview.empty:
            continue
        prev_ids = preview[id_col].astype("string")
        prev_codes = pd.Categorical(prev_ids, categories=test_index).codes
        if (prev_codes >= 0).sum() == 0:
            continue
    except Exception:
        try:
            probe = pd.read_csv(
                path,
                usecols=[id_col],
                nrows=5000,
                engine="c",
                dtype={id_col: "string"},
            )
            if probe.empty:
                continue
            probe_ids = probe[id_col].astype("string")
            probe_codes = pd.Categorical(probe_ids, categories=test_index).codes
            if (probe_codes >= 0).sum() == 0:
                continue
        except Exception:
            continue

    try:
        ff = pd.read_csv(
            path,
            usecols=[id_col, pred_col],
            engine="c",
            dtype={id_col: "string"},
        )
    except Exception:
        continue

    ff = ff.rename(columns={id_col: "image_name", pred_col: "target"})
    ff = ff.dropna(subset=["image_name"])
    if ff.empty:
        continue

    ff["target"] = pd.to_numeric(ff["target"], errors="coerce")
    ff["image_name"] = ff["image_name"].astype("string")

    codes = pd.Categorical(ff["image_name"], categories=test_index).codes
    mask = codes >= 0
    if not np.any(mask):
        continue
    ff_in = ff.loc[mask, ["image_name", "target"]]
    if ff_in.empty:
        continue

    coverage = ff_in["image_name"].nunique(dropna=True) / n_test
    if coverage < MIN_TEST_COVERAGE:
        continue

    ff_in = ff_in.drop_duplicates(subset=["image_name"], keep="first")

    pos = test_pos.reindex(ff_in["image_name"]).to_numpy()
    valid = pos.astype("float64")
    valid_mask = np.isfinite(valid)
    if not np.any(valid_mask):
        continue
    pos = pos[valid_mask].astype(np.int32, copy=False)
    vals = ff_in["target"].to_numpy()[valid_mask].astype(np.float64, copy=False)

    if pred_matrix is None:
        pred_matrix = np.full((n_test, 0), np.nan, dtype=np.float64)

    pred_matrix = np.concatenate(
        [pred_matrix, np.full((n_test, 1), np.nan, dtype=np.float64)], axis=1
    )
    pred_matrix[pos, -1] = vals

    new_col = f"target_{used_files}"
    cols.append(new_col)
    used_files += 1

print("Merged prediction files:", used_files)
print("Current shape after merges:", (n_test, 1 + used_files))
print("Merged columns:", cols[:10])

f = f_idx.reset_index()
if used_files > 0:
    for j, c in enumerate(cols):
        f[c] = pred_matrix[:, j]



## === cell 4
if len(cols) > 0:
    col_means = f[cols].mean(axis=0, skipna=True)
    col_means = col_means.where(np.isfinite(col_means), 0.5)

    f[cols] = f[cols].fillna(col_means.to_dict())

    f["target"] = f[cols].mean(axis=1).astype(float)
    f.drop(columns=cols, inplace=True)

    print("Using mean-ensemble of discovered prediction CSVs.")
    print(f.head())
    print("Final submission shape:", f.shape)
    print("Final submission columns:", f.columns.tolist())
else:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    train_path = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
    test_path = "/kaggle/input/siim-isic-melanoma-classification/test.csv"

    usecols_train = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "target",
    ]
    usecols_test = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
    ]
    train = pd.read_csv(train_path, usecols=usecols_train, engine="c")
    test = pd.read_csv(test_path, usecols=usecols_test, engine="c")

    if "image_name" in train.columns:
        before = len(train)
        train = train.drop_duplicates(subset=["image_name"], keep="first").reset_index(
            drop=True
        )
        after = len(train)
        print(f"Dropped duplicate train image_name rows: {before - after}")

    pid_train = train["patient_id"].astype("object")
    pid_codes, pid_uniques = pd.factorize(pid_train, sort=False)

    pid_map = {
        pid_uniques[i]: float(i)
        for i in range(len(pid_uniques))
        if pd.notna(pid_uniques[i])
    }
    unknown_pid_code = float(len(pid_map))

    def _add_age_bin_and_pid_features(
        df: pd.DataFrame, pid_map_local: dict
    ) -> pd.DataFrame:
        out = df.copy()

        age = pd.to_numeric(out["age_approx"], errors="coerce")
        out["age_bin"] = pd.cut(
            age,
            bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
            labels=["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"],
        ).astype("object")

        pid_obj = out["patient_id"].astype("object")
        out["patient_id_num"] = pid_obj.map(pid_map_local)
        out["patient_id_num"] = (
            out["patient_id_num"]
            .where(pid_obj.isna(), out["patient_id_num"].fillna(unknown_pid_code))
            .astype("float32")
        )

        nbuckets = 2048
        pid_str = pid_obj.where(pid_obj.notna(), "__MISSING__").astype(str)
        out["patient_id_hash"] = (
            (
                pd.util.hash_pandas_object(pid_str, index=False).astype(np.uint64)
                % nbuckets
            )
            .astype("int32")
            .astype(str)
        )
        return out

    feature_cols = ["patient_id", "sex", "age_approx", "anatom_site_general_challenge"]

    X_all = _add_age_bin_and_pid_features(train[feature_cols].copy(), pid_map)
    y_all = train["target"].astype(int).copy()
    X_test = _add_age_bin_and_pid_features(test[feature_cols].copy(), pid_map)

    pid = train["patient_id"]
    groups = np.where(pid.notna(), pid.values, train["image_name"].values)

    numeric_features = ["age_approx", "patient_id_num"]
    categorical_features = [
        "sex",
        "anatom_site_general_challenge",
        "age_bin",
        "patient_id_hash",
    ]

    pre = ColumnTransformer(
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
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    def make_clf():
        return LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.05,
            max_iter=1200,
            C=1.0,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        )

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    test_pred_folds = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]

        pipe = Pipeline(steps=[("pre", pre), ("clf", make_clf())])
        pipe.fit(X_tr, y_tr)

        proba_test = pipe.predict_proba(X_test)[:, 1].astype(float)
        test_pred_folds.append(proba_test)

        print(
            f"Trained fold {fold}/{n_splits}, train rows={len(tr_idx)}, val rows={len(va_idx)}"
        )

    proba_test_oofavg = np.mean(np.vstack(test_pred_folds), axis=0).astype(float)

    pipe_full = Pipeline(steps=[("pre", pre), ("clf", make_clf())])
    pipe_full.fit(X_all, y_all)
    proba_test_full = pipe_full.predict_proba(X_test)[:, 1].astype(float)

    proba = (0.5 * proba_test_oofavg + 0.5 * proba_test_full).astype(float)

    pred_df = pd.DataFrame({"image_name": test["image_name"].values, "target": proba})
    f = f.merge(pred_df, on="image_name", how="left")

    prior = float(y_all.mean())
    f["target"] = f["target"].fillna(prior)

    print(
        "No external prediction CSVs merged; using patient-wise GroupKFold OOF LogisticRegression fallback "
        "(age_bin + patient_id_num stable + patient_id_hash, elastic-net), averaged with full-data fit."
    )
    print("Train target prevalence (used only for any missing rows):", prior)
    print(f.head())
    print("Final submission shape:", f.shape)
    print("Final submission columns:", f.columns.tolist())



## === cell 5
f = f[["image_name", "target"]].copy()
f["target"] = pd.to_numeric(f["target"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

f.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
print("submission.csv rows:", len(f))
