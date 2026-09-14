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

0.9411769203044964

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the hard dependency on the missing `../input/ensemble-melanoma` directory (which causes the crash) and instead build a valid submission using the provided `sample_submission.csv` as the template. Since no model packages are available here and the current script is purely an ensemble of external submissions, the only safe end-to-end fix is to fall back to a deterministic baseline prediction when the ensemble files aren’t present. I also make sure the submission has the exact required columns (`image_name`, `target`), matches the test row order, and is written to a `.csv` file. If any compatible ensemble CSVs are found under the provided dataset folders, the script average them; otherwise it write the baseline.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC is caused by emitting a near-constant prediction (train prevalence), which yields random-ranking behavior on ROC-AUC. To move toward the 0.941 target without changing your “no-ML-packages ensemble” core logic, I keep the same submission-template + averaging structure but add a minimal, legitimate metadata-only model fallback when no external submission files are found. This uses scikit-learn logistic regression on `sex/age/anatom_site` (available in the provided CSVs), producing non-constant probabilities and a substantially better ranking than prevalence while staying fast and within constraints. If ensemble CSVs are found, the behavior remains the same (simple mean).'
- What this solution (achieved 0.5) has done: 'We keep your ensemble-first logic identical, but make the metadata fallback stronger (since your current 0.5 implies the fallback is effectively constant or not learning useful ranking). The minimal change is to (1) include `patient_id` as a categorical feature and (2) use a slightly more expressive but still lightweight/tabular-only model (`HistGradientBoostingClassifier`) with proper preprocessing, which typically yields a much better ROC-AUC ranking than plain logistic regression on this dataset’s metadata. We also harden alignment (ensure test rows match `sample_submission` order) and keep probabilities well-formed in [0,1]. This should move your score substantially upward toward the 0.941 target without changing the core “no image model; ensemble else fallback” approach.'
- What this solution (achieved 0.5) has done: 'I keep your ensemble-first logic unchanged, but strengthen the metadata fallback (the part currently responsible for the ~0.5 AUC when no external submission files are found). The main issue is that `HistGradientBoostingClassifier` is being fed a sparse one-hot matrix from `OneHotEncoder`, which it does not handle reliably; this can silently collapse signal and produce near-constant predictions. I switch the fallback model to `LogisticRegression` (still lightweight/tabular, same evaluation semantics) with proper preprocessing, plus add a safe out-of-fold style sanity check on train to ensure predictions are not constant (without changing the training loop structure). This should legitimately increase ranking quality and move AUC upward toward your 0.941 target while keeping changes minimal and ensuring a valid `submission_mean.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests the fallback predictions are effectively constant or nearly so, which breaks ranking for ROC-AUC. I keep the ensemble-first logic unchanged, but make the metadata fallback reliably non-constant by (1) using `LogisticRegression(saga)` which is better suited for high-cardinality sparse one-hot features like `patient_id`, and (2) adding a tiny, deterministic patient-level out-of-fold (OOF) blending step to reduce overfitting from `patient_id` while preserving the same “train once, predict test” approach. This stays purely metadata-based, uses only scikit-learn, and keeps I/O paths and submission format identical. The result should move your score meaningfully upward toward the 0.941 target without changing any image/model architecture (none is used).'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests the fallback is still effectively producing (near) constant or non-informative rankings, likely because the `patient_id` one-hot makes the sparse problem too high-dimensional and unstable for your current logistic setup. To move score upward toward the 0.941 target with minimal changes and keeping the same “ensemble if available else metadata fallback” core logic, I adjust only the fallback to (1) replace high-cardinality `patient_id` one-hot with a leakage-safe patient-level target-encoding computed via GroupKFold (OOF), and (2) keep a regularized LogisticRegression on the remaining one-hot metadata. I also add a deterministic standardization for numeric features and a final variance sanity check to prevent constant outputs. The ensemble path, I/O paths, and submission formatting remain unchanged.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC indicates the fallback is still effectively producing a near-constant or non-informative ranking, so we keep your “ensemble if found else metadata fallback” logic unchanged but make the fallback reliably learn signal from the metadata. The minimal score-directed change is to (1) use a stratified, patient-grouped cross-fit to generate an out-of-fold probability feature (OOF) from the same logistic model (so the model has a stronger, leakage-safe ranking signal), then (2) train a final logistic model on that OOF feature plus your existing metadata features. This preserves evaluation semantics (still just metadata probabilities), avoids the known sparse/HGB issues, and remains fast. We also harden splitting (StratifiedGroupKFold) and keep submission row order exactly matching `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_DIRS = [
    "/kaggle/input",
    "/kaggle/data",
    "../input",
    "../data",
]


def _safe_listdir(path):
    try:
        return os.listdir(path)
    except Exception:
        return []


def find_candidate_submission_csvs(base_dirs):
    candidates = []
    for base in base_dirs:
        if not os.path.exists(base):
            continue
        for name in _safe_listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p):
                for fn in _safe_listdir(p):
                    if fn.lower().endswith(".csv") and "submission" in fn.lower():
                        candidates.append(os.path.join(p, fn))
            elif (
                os.path.isfile(p)
                and p.lower().endswith(".csv")
                and "submission" in os.path.basename(p).lower()
            ):
                candidates.append(p)
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


candidate_csvs = find_candidate_submission_csvs(BASE_DIRS)
candidate_csvs[:10], len(candidate_csvs)



## === cell 2
POSSIBLE_SAMPLE_PATHS = [
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/siim-isic-melanoma-classification/sample_submission.csv",
    "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv",
]

sample_path = None
for p in POSSIBLE_SAMPLE_PATHS:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in the expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_path)
if list(sample_sub.columns) != ["image_name", "target"]:
    sample_sub = sample_sub.rename(
        columns={sample_sub.columns[0]: "image_name", sample_sub.columns[1]: "target"}
    )
sample_sub["image_name"] = sample_sub["image_name"].astype(str)

sample_sub.head(), sample_sub.shape, sample_path




## === cell 3
def load_submission_like(path):
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "image_name" in df.columns and "target" in df.columns:
        out = df[["image_name", "target"]].copy()
    elif (
        len(df.columns) >= 2
        and ("image_name" in cols[0] or "image" in cols[0])
        and ("target" in cols[1] or "prediction" in cols[1])
    ):
        out = df.iloc[:, :2].copy()
        out.columns = ["image_name", "target"]
    else:
        return None
    out["image_name"] = out["image_name"].astype(str)
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    if out["target"].isna().any():
        return None
    return out


usable = []
for p in candidate_csvs:
    df = load_submission_like(p)
    if df is None:
        continue
    overlap = df["image_name"].isin(sample_sub["image_name"]).mean()
    if overlap >= 0.95:
        usable.append((p, df))

len(usable), [os.path.basename(p) for p, _ in usable[:10]]



## === cell 4
POSSIBLE_TRAIN_PATHS = [
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/siim-isic-melanoma-classification/train.csv",
    "/kaggle/input/siim-isic-melanoma-classification/train.csv",
]

POSSIBLE_TEST_PATHS = [
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/siim-isic-melanoma-classification/test.csv",
    "/kaggle/input/siim-isic-melanoma-classification/test.csv",
]

train_path = None
for p in POSSIBLE_TRAIN_PATHS:
    if os.path.exists(p):
        train_path = p
        break

test_path = None
for p in POSSIBLE_TEST_PATHS:
    if os.path.exists(p):
        test_path = p
        break

train_prevalence = 0.02
if train_path is not None:
    tr_prev = pd.read_csv(train_path, usecols=["target"])
    train_prevalence = float(tr_prev["target"].mean())


def _metadata_fallback_predict(train_csv_path, test_csv_path, sample_df, prior):
    """
    Change (score-directed, minimal, preserves core ensemble-vs-fallback logic):
    - Your 0.5 AUC implies the fallback ranking is close to random/constant.
    - We keep the same metadata-only approach, but make it reliably informative by:
      (1) building a leakage-safe patient target encoding (as before), AND
      (2) adding a cross-fitted (OOF) probability feature from a simple logistic model
          using StratifiedGroupKFold (patient-grouped), then fitting a final logistic model.
    This usually improves ROC-AUC ranking materially without changing the overall approach.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    use_cols_train = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "target",
    ]
    use_cols_test = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
    ]

    tr = pd.read_csv(train_csv_path, usecols=use_cols_train)
    te = pd.read_csv(test_csv_path, usecols=use_cols_test)

    tr["image_name"] = tr["image_name"].astype(str)
    te["image_name"] = te["image_name"].astype(str)

    te = sample_df[["image_name"]].merge(te, on="image_name", how="left")

    tr_pid = tr["patient_id"].astype(str).fillna("NA")
    te_pid = te["patient_id"].astype(str).fillna("NA")
    y = tr["target"].astype(int).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    pid_oof = np.full(shape=(len(tr),), fill_value=prior, dtype=np.float64)

    alpha = 5.0  # deterministic smoothing

    for tr_idx, va_idx in sgkf.split(tr, y, groups=tr_pid.values):
        fold_tr = tr.iloc[tr_idx]
        cnt = fold_tr.groupby("patient_id")["target"].count()
        mean = fold_tr.groupby("patient_id")["target"].mean()
        smooth = ((mean * cnt) + (prior * alpha)) / (cnt + alpha)
        smooth = smooth.to_dict()
        pid_oof[va_idx] = (
            tr_pid.iloc[va_idx].map(smooth).astype(float).fillna(prior).to_numpy()
        )

    pid_oof = np.clip(pid_oof, 0.0, 1.0)

    cnt_full = tr.groupby("patient_id")["target"].count()
    mean_full = tr.groupby("patient_id")["target"].mean()
    smooth_full = ((mean_full * cnt_full) + (prior * alpha)) / (cnt_full + alpha)
    smooth_full = smooth_full.to_dict()
    pid_test = (
        te_pid.map(smooth_full).astype(float).fillna(prior).to_numpy(dtype=np.float64)
    )
    pid_test = np.clip(pid_test, 0.0, 1.0)

    tr_feat = tr[["sex", "age_approx", "anatom_site_general_challenge"]].copy()
    te_feat = te[["sex", "age_approx", "anatom_site_general_challenge"]].copy()
    tr_feat["patient_te"] = pid_oof
    te_feat["patient_te"] = pid_test

    numeric_features = ["age_approx", "patient_te"]
    categorical_features = ["sex", "anatom_site_general_challenge"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    base_clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=4000,
        random_state=0,
    )
    base_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", base_clf)])

    oof_pred = np.full(shape=(len(tr),), fill_value=prior, dtype=np.float64)
    test_pred_folds = []

    for tr_idx, va_idx in sgkf.split(tr_feat, y, groups=tr_pid.values):
        base_pipe.fit(tr_feat.iloc[tr_idx], y[tr_idx])
        oof_pred[va_idx] = base_pipe.predict_proba(tr_feat.iloc[va_idx])[:, 1].astype(
            np.float64
        )
        test_pred_folds.append(
            base_pipe.predict_proba(te_feat)[:, 1].astype(np.float64)
        )

    oof_pred = np.where(np.isfinite(oof_pred), oof_pred, prior)
    oof_pred = np.clip(oof_pred, 0.0, 1.0)

    test_pred_cv = np.mean(np.stack(test_pred_folds, axis=0), axis=0)
    test_pred_cv = np.where(np.isfinite(test_pred_cv), test_pred_cv, prior)
    test_pred_cv = np.clip(test_pred_cv, 0.0, 1.0)

    tr_feat2 = tr_feat.copy()
    te_feat2 = te_feat.copy()
    tr_feat2["oof_lr_pred"] = oof_pred
    te_feat2["oof_lr_pred"] = test_pred_cv

    numeric_features2 = ["age_approx", "patient_te", "oof_lr_pred"]
    preprocessor2 = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                numeric_features2,
            ),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    final_clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=4000,
        random_state=0,
    )
    final_pipe = Pipeline(steps=[("prep", preprocessor2), ("clf", final_clf)])
    final_pipe.fit(tr_feat2, y)
    proba = final_pipe.predict_proba(te_feat2)[:, 1].astype(np.float64)

    proba = np.where(np.isfinite(proba), proba, prior)
    proba = np.clip(proba, 0.0, 1.0)

    if float(np.std(proba)) < 1e-4:
        site_rate = (
            tr.groupby("anatom_site_general_challenge")["target"].mean().to_dict()
        )
        site_prior = te["anatom_site_general_challenge"].map(site_rate).astype(float)
        site_prior = site_prior.fillna(prior).to_numpy(dtype=np.float64)
        proba = 0.6 * proba + 0.4 * site_prior
        proba = np.clip(proba, 0.0, 1.0)

    return proba


if len(usable) == 0:
    submission = sample_sub.copy()
    if train_path is not None and test_path is not None:
        submission["target"] = _metadata_fallback_predict(
            train_path, test_path, sample_sub, train_prevalence
        )
    else:
        submission["target"] = train_prevalence
else:
    merged = sample_sub[["image_name"]].copy()
    pred_cols = []
    for i, (p, df) in enumerate(usable):
        col = f"pred_{i}"
        tmp = df.rename(columns={"target": col})
        merged = merged.merge(tmp[["image_name", col]], on="image_name", how="left")
        pred_cols.append(col)

    merged[pred_cols] = merged[pred_cols].fillna(train_prevalence)

    submission = sample_sub.copy()
    submission["target"] = merged[pred_cols].mean(axis=1).astype(np.float64)

submission["target"] = submission["target"].clip(0.0, 1.0)

submission.head(), submission.shape, float(submission["target"].min()), float(
    submission["target"].max()
)



## === cell 5
out_path = "submission_mean.csv"
submission[["image_name", "target"]].to_csv(out_path, index=False, float_format="%.6f")

print(f"Wrote submission to: {out_path}")
print(submission.describe(include="all"))
