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
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC strongly suggests the metadata fallback is not actually running (likely due to missing scikit-learn in your environment), so the script ends up writing (near) constant predictions. To move toward the 0.941 target with minimal changes and without altering the ensemble-vs-fallback core logic, I keep the exact same pipeline but replace the scikit-learn fallback with a pure pandas/numpy, leakage-safe patient target-encoding + simple site/sex/age aggregation that yields non-constant probabilities and a much better ROC-AUC ranking than a constant prior. The ensemble path (averaging any found submission CSVs) is unchanged. I also harden row alignment to the sample submission order and keep the output schema and file name identical.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC is consistent with effectively constant predictions; the biggest score-lever you can pull without changing the overall “ensemble if found else fallback” structure is to make the metadata fallback produce a stronger, stable ranking signal. I keep your ensemble path identical, but upgrade the fallback to use a leakage-safe out-of-fold (patient-grouped) target encoding for `patient_id` plus smoothed encodings for site/sex/age, which typically gives a much better ROC-AUC than the current per-row encoding. I also harden categorical missing handling (so `"nan"` strings don’t fragment groups) and ensure the test merge stays aligned to `sample_submission` order. These are minimal, purely tabular changes that should move you materially upward toward the 0.941 target while still producing a valid `submission_mean.csv`.'
- What this solution (achieved 0.5) has done: 'We keep your ensemble-first behavior exactly the same, and only strengthen the metadata fallback that is currently responsible for the ~0.5 AUC when no external submission CSVs are found. The main minimal change is to make the fallback produce a richer (non-constant) ranking signal by adding a leakage-safe, out-of-fold target encoding for `patient_id` plus an additional OOF target encoding for the *pair* `(patient_id, anatom_site_general_challenge)` and then blending these with your existing smoothed site/age/sex rates. This stays pure pandas/numpy (no new packages), preserves the “train once → predict test” semantics, and remains fast. We also add a deterministic sanity guard to ensure the output is not nearly constant, while keeping submission formatting and row order identical to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC suggests the fallback predictions are still effectively not producing a meaningful ranking signal on the leaderboard, so we keep the exact same “ensemble if found else metadata fallback” structure but make the fallback ranking stronger and more robust. The minimal score-directed change is to add one more leakage-safe out-of-fold target-encoding feature based on the high-signal `diagnosis` column (train-only), mapped to test as a prior (since test lacks diagnosis) and blended in a way that still preserves valid semantics (no leakage from test). We also slightly adjust the blend weights to rely a bit less on `patient_id` (which can overfit and generalize poorly to test patients) and a bit more on stable signals (`site`, `age`) to improve AUC ranking. The ensemble path, file paths, output columns/order, and CSV writing remain unchanged.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests the fallback is either not producing meaningful variation or (more likely) the “extra” features you added don’t transfer to test (e.g., `diagnosis` is train-only and currently contributes nothing because `dx_te` is just the prior). To move the score upward toward 0.941 with minimal change and keeping your ensemble-first + metadata-fallback core logic, I (1) remove the ineffective `diagnosis` contribution from the test blend and reallocate that weight to more stable signals (`site`, `age`, and patient/site pair), and (2) add a tiny, leakage-safe, OOF target-encoded feature for `anatom_site_general_challenge` (computed with the same patient-hash folds) to strengthen ranking without overfitting. I also add a deterministic “not-nearly-constant” guard that increases variance slightly only if needed, without changing semantics when predictions already vary. Paths, submission schema, and the ensemble averaging behavior remain unchanged.'

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


def _normalize_cat(s: pd.Series, unknown="unknown") -> pd.Series:
    s = s.astype("string")
    s = s.fillna(unknown)
    s = s.str.strip()
    s = s.replace({"": unknown, "nan": unknown, "None": unknown, "<NA>": unknown})
    return s.astype(str)


def _metadata_fallback_predict(train_csv_path, test_csv_path, sample_df, prior):
    """
    Score-directed minimal change (keeps ensemble-vs-fallback core logic):
    - Remove ineffective diagnosis contribution on test (test has no diagnosis -> dx_te == prior).
      Reallocate that weight to stable, test-available signals (pair/site/age).
    - Add leakage-safe OOF target encoding for anatom_site (same patient-hash folds) to strengthen ranking.
    - Keep deterministic behavior and strict [0,1] probabilities.
    """
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

    tr["patient_id"] = _normalize_cat(tr["patient_id"], unknown="NA")
    te["patient_id"] = _normalize_cat(te["patient_id"], unknown="NA")

    tr["sex"] = _normalize_cat(tr["sex"], unknown="unknown")
    te["sex"] = _normalize_cat(te["sex"], unknown="unknown")

    tr["anatom_site_general_challenge"] = _normalize_cat(
        tr["anatom_site_general_challenge"], unknown="unknown"
    )
    te["anatom_site_general_challenge"] = _normalize_cat(
        te["anatom_site_general_challenge"], unknown="unknown"
    )

    tr["age_approx"] = pd.to_numeric(tr["age_approx"], errors="coerce")
    te["age_approx"] = pd.to_numeric(te["age_approx"], errors="coerce")

    y = tr["target"].astype(float).to_numpy()
    n = len(tr)

    K = 5
    pid_codes, pid_uniques = pd.factorize(tr["patient_id"], sort=False)
    pid_hash = pd.util.hash_pandas_object(pd.Series(pid_uniques), index=False).to_numpy(
        dtype=np.uint64
    )
    pid_fold = (pid_hash % K).astype(int)
    folds = pid_fold[pid_codes]  # fold per row

    alpha_pid = 20.0
    pid_oof = np.empty(n, dtype=np.float64)
    for f in range(K):
        mask_tr = folds != f
        mask_va = folds == f
        tmp = pd.DataFrame(
            {
                "patient_id": tr.loc[mask_tr, "patient_id"].to_numpy(),
                "target": y[mask_tr],
            }
        )
        st = tmp.groupby("patient_id")["target"].agg(["sum", "count"]).astype(float)
        rate = (st["sum"] + alpha_pid * prior) / (st["count"] + alpha_pid)
        pid_oof[mask_va] = (
            tr.loc[mask_va, "patient_id"]
            .map(rate.to_dict())
            .astype(float)
            .fillna(prior)
            .to_numpy(dtype=np.float64)
        )
    pid_oof = np.clip(pid_oof, 0.0, 1.0)

    st_all = tr.groupby("patient_id")["target"].agg(["sum", "count"]).astype(float)
    pid_map = (
        (st_all["sum"] + alpha_pid * prior) / (st_all["count"] + alpha_pid)
    ).to_dict()
    pid_te = (
        te["patient_id"]
        .map(pid_map)
        .astype(float)
        .fillna(prior)
        .to_numpy(dtype=np.float64)
    )
    pid_te = np.clip(pid_te, 0.0, 1.0)

    tr_pair = (tr["patient_id"] + "||" + tr["anatom_site_general_challenge"]).astype(
        str
    )
    te_pair = (te["patient_id"] + "||" + te["anatom_site_general_challenge"]).astype(
        str
    )

    alpha_pair = 30.0
    pair_oof = np.empty(n, dtype=np.float64)
    for f in range(K):
        mask_tr = folds != f
        mask_va = folds == f
        tmp = pd.DataFrame({"pair": tr_pair[mask_tr].to_numpy(), "target": y[mask_tr]})
        st = tmp.groupby("pair")["target"].agg(["sum", "count"]).astype(float)
        rate = (st["sum"] + alpha_pair * prior) / (st["count"] + alpha_pair)
        pair_oof[mask_va] = (
            tr_pair[mask_va]
            .map(rate.to_dict())
            .astype(float)
            .fillna(prior)
            .to_numpy(dtype=np.float64)
        )
    pair_oof = np.clip(pair_oof, 0.0, 1.0)

    st_pair_all = (
        pd.DataFrame({"pair": tr_pair, "target": tr["target"].astype(float)})
        .groupby("pair")["target"]
        .agg(["sum", "count"])
        .astype(float)
    )
    pair_map = (
        (st_pair_all["sum"] + alpha_pair * prior) / (st_pair_all["count"] + alpha_pair)
    ).to_dict()
    pair_te = (
        te_pair.map(pair_map).astype(float).fillna(prior).to_numpy(dtype=np.float64)
    )
    pair_te = np.clip(pair_te, 0.0, 1.0)

    def _smooth_rate_map(col, alpha_col=200.0):
        st = tr.groupby(col)["target"].agg(["sum", "count"]).astype(float)
        rate = (st["sum"] + alpha_col * prior) / (st["count"] + alpha_col)
        return rate.to_dict()

    site_map = _smooth_rate_map("anatom_site_general_challenge", alpha_col=300.0)
    sex_map = _smooth_rate_map("sex", alpha_col=300.0)

    site_te = (
        te["anatom_site_general_challenge"]
        .map(site_map)
        .astype(float)
        .fillna(prior)
        .to_numpy(dtype=np.float64)
    )
    sex_te = (
        te["sex"].map(sex_map).astype(float).fillna(prior).to_numpy(dtype=np.float64)
    )

    alpha_site_oof = 150.0
    site_oof = np.empty(n, dtype=np.float64)
    for f in range(K):
        mask_tr = folds != f
        mask_va = folds == f
        tmp = pd.DataFrame(
            {
                "site": tr.loc[mask_tr, "anatom_site_general_challenge"].to_numpy(),
                "target": y[mask_tr],
            }
        )
        st = tmp.groupby("site")["target"].agg(["sum", "count"]).astype(float)
        rate = (st["sum"] + alpha_site_oof * prior) / (st["count"] + alpha_site_oof)
        site_oof[mask_va] = (
            tr.loc[mask_va, "anatom_site_general_challenge"]
            .map(rate.to_dict())
            .astype(float)
            .fillna(prior)
            .to_numpy(dtype=np.float64)
        )
    site_oof = np.clip(site_oof, 0.0, 1.0)

    age_bins = np.array([0, 30, 40, 50, 60, 70, 80, 200], dtype=float)
    tr_age = tr["age_approx"].fillna(tr["age_approx"].median())
    te_age = te["age_approx"].fillna(tr["age_approx"].median())
    tr_age_bin = pd.cut(tr_age, bins=age_bins, right=False, include_lowest=True).astype(
        str
    )
    te_age_bin = pd.cut(te_age, bins=age_bins, right=False, include_lowest=True).astype(
        str
    )

    tmp_age = pd.DataFrame(
        {"age_bin": tr_age_bin, "target": tr["target"].astype(float)}
    )
    st_age = tmp_age.groupby("age_bin")["target"].agg(["sum", "count"]).astype(float)
    alpha_age = 300.0
    age_rate = (st_age["sum"] + alpha_age * prior) / (st_age["count"] + alpha_age)
    age_map = age_rate.to_dict()
    age_te = (
        te_age_bin.map(age_map).astype(float).fillna(prior).to_numpy(dtype=np.float64)
    )

    proba = (
        0.52 * pid_te + 0.19 * pair_te + 0.17 * site_te + 0.10 * age_te + 0.02 * sex_te
    ).astype(np.float64)

    proba = np.where(np.isfinite(proba), proba, prior)
    proba = np.clip(proba, 0.0, 1.0)

    if float(np.std(proba)) < 1e-6:
        h = pd.util.hash_pandas_object(te["image_name"], index=False).to_numpy(
            dtype=np.uint64
        )
        jitter = (h % 1000) / 1000.0
        proba = np.clip(0.98 * prior + 0.02 * jitter, 0.0, 1.0).astype(np.float64)

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
