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
seaborn==0.12.2
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

0.9423

# 6. Current score

0.75332

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook is trying to ensemble a set of external “team submissions” that are not present in this Kaggle environment, so `glob()` finds no CSVs and everything downstream crashes. I keep the same rank-averaging ensemble core logic, but make it robust by (1) searching for any valid submission-like CSVs under the available `../input` tree and (2) falling back to a simple, deterministic baseline using `train.csv` target mean if none are found, so a valid `submission.csv` is always produced. I also remove notebook-only magic (`%matplotlib inline`) and fix pathing to the actual provided dataset locations. These changes are execution/stability focused; any score gain depends on whether compatible prediction CSVs exist in the input, otherwise the fallback be score-neutral (but valid).'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC is coming from the fallback path producing a constant prediction (train target mean), which yields ~random ranking and thus AUC≈0.5. To move toward the 0.9423 target without changing the “submission-ensemble” core logic, I keep the same rank-averaging flow but ensure we actually find usable prediction CSVs by expanding the search to `/kaggle/data` and `/kaggle/input` (your environment paths), and by explicitly excluding the competition’s own CSVs. If no external prediction CSVs exist, I keep the fallback (so it always produces a valid submission), but this change should allow the ensemble path to trigger when such files are present, improving AUC substantially toward your target.'
- What this solution (achieved 0.66789) has done: 'Your current 0.5 AUC comes from the constant-mean fallback, so the smallest legitimate way to move toward 0.9423 is to keep your ensemble core logic intact but replace the fallback with a simple metadata-only model trained on `train.csv` and predicting on `test.csv`. This preserves the overall pipeline (it still produces probabilistic `target` for each `image_name` and writes `submission.csv`) while creating non-constant, rankable predictions that should improve AUC substantially over 0.5. I’m also making the candidate-submission CSV scan a bit stricter so it’s less likely to accidentally pick up unrelated CSVs with a `target` column and silently hurt performance. If external submission-like CSVs are present, your original rank-averaging ensemble path is still used unchanged.'
- What this solution (achieved 0.66776) has done: 'Your current score (0.66789) is far below the target (0.9423), so we should improve ranking quality while keeping your ensemble core logic intact. The easiest legitimate gain without changing the approach is to make the metadata fallback stronger and better-calibrated: use patient-grouped CV out-of-fold predictions to reduce leakage and then train the same logistic regression on full data for test inference. I also harden candidate-submission CSV selection to avoid accidentally ensembling unrelated CSVs (which can silently hurt AUC). These are minimal, metric-aligned changes that keep your rank-averaging ensemble path unchanged and only improve the fallback path that is currently driving your score.'
- What this solution (achieved 0.66209) has done: 'Your score is far below the target (0.66776 vs 0.9423), and since your ensemble path usually won’t trigger (no external submissions found), the only way to move toward the target while preserving your overall pipeline is to strengthen the *fallback* metadata model without changing the overall training approach. I keep the same logistic-regression-on-metadata approach, but add a couple of competition-standard metadata features (`log1p(age)`, missing-age indicator, and per-category target-encoding computed out-of-fold with GroupKFold to avoid leakage). I also ensure deterministic behavior and keep the submission formatting/alignment identical. These are minimal, metric-aligned changes that should improve ranking quality and AUC without altering your ensemble logic.'
- What this solution (achieved 0.66182) has done: 'I fix the runtime error by ensuring the engineered feature list passed into the `ColumnTransformer` contains unique column names (the current code accidentally includes `age_bin` in both numeric and categorical lists, which makes columns non-unique and crashes). I keep the ensemble/rank-averaging path unchanged and only adjust the fallback metadata pipeline so it can train and produce predictions deterministically. I also add a small safety check to guarantee the generated submission aligns to the sample submission’s `image_name` order before writing `submission.csv`. These changes are execution/stability focused and should improve score versus the previous constant/failed fallback because the metadata model now run end-to-end.'
- What this solution (achieved 0.65194) has done: 'I fix the crash in the fallback metadata pipeline by ensuring the categorical columns contain no pandas `NA` scalars (which trigger the “boolean value of NA is ambiguous” error inside sklearn’s `SimpleImputer`). This is done by converting categorical features to plain `object` dtype and explicitly filling missing values with a sentinel string before the `ColumnTransformer`. I also make `age_log1p` safe for missing ages by computing it from an `age.fillna(0)` variant and keep all modeling/training logic unchanged. Finally, I keep the existing ensemble path intact and ensure a valid `submission.csv` is always written with correct ordering.'
- What this solution (achieved 0.65197) has done: 'Your current score (0.65194) is far below the target (0.9423), so we should improve the fallback path (since the ensemble usually won’t trigger) while keeping its core logic (metadata features + GroupKFold + LogisticRegression) intact. The smallest high-impact, metric-aligned change is to add a second LogisticRegression trained on the same engineered features but with a different regularization strength, then average their predicted probabilities; this keeps the same model family and training approach while usually improving ranking/AUC. I also ensure the test inference uses the exact same feature columns/order as training and keep the submission aligned to `sample_submission.csv` ordering. The ensemble/rank-averaging path remains unchanged.'
- What this solution (achieved 0.66045) has done: 'Your current score (0.65197) is far below the target (0.9423), so we should improve the *fallback* path (the metadata model) while leaving the ensemble/rank-averaging path unchanged. The smallest high-impact change that preserves the same model family/training approach is to add a couple of strong, competition-standard metadata priors using **out-of-fold target encoding**: one for `anatom_site_general_challenge` and one for `(sex, anatom_site_general_challenge)` interaction, both computed with `GroupKFold` to avoid patient leakage. We keep the same LogisticRegression pipelines and the same averaging of two C values, but these added features typically improve ranking/AUC substantially over the current feature set. I also keep the submission alignment to `sample_submission.csv` order as you already do.'
- What this solution (achieved 0.75332) has done: 'We need to move your AUC up toward 0.9423 (current 0.66045), and since no external submission CSVs are typically found, the fallback metadata model is what determines the score. I keep the exact same overall approach (metadata-only features + GroupKFold target encodings + two LogisticRegression models averaged), but fix a key issue: you compute OOF predictions in a loop and then discard them, so you never use them for any calibration/blending signal. I minimally add an OOF AUC check (for sanity) and then use the fold-OFF probabilities to fit a single scalar blending weight between the LR average and the strongest prior (`patient_prior`)—this preserves semantics (still probabilities, still LR-based) but improves ranking robustness with negligible extra complexity. I also keep submission alignment strict to `sample_submission.csv` order and ensure deterministic behavior.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import numpy as np
import pandas as pd

from scipy.stats import rankdata

warnings.filterwarnings("ignore")

INPUT_ROOTS = [
    "../input",  # kaggle notebook conventional
    "/kaggle/input",  # present in many environments
    "/kaggle/data",  # present in your provided tree
]

DATASET_SUBDIR = "siim-isic-melanoma-classification"

LABELS = ["target"]


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_comp_file(filename):
    candidates = []
    for r in INPUT_ROOTS:
        candidates.append(os.path.join(r, DATASET_SUBDIR, filename))
        candidates.append(os.path.join(r, filename))
        candidates.append(os.path.join(r, "data", filename))
        candidates.append(os.path.join(r, "data", DATASET_SUBDIR, filename))
    return _first_existing(*candidates)


train_csv_path = _find_comp_file("train.csv")
test_csv_path = _find_comp_file("test.csv")
sample_sub_path = _find_comp_file("sample_submission.csv")

if sample_sub_path is None or test_csv_path is None or train_csv_path is None:
    raise FileNotFoundError(
        "Required competition files not found. "
        f"train={train_csv_path}, test={test_csv_path}, sample_submission={sample_sub_path}"
    )

print("Using paths:")
print(" train:", train_csv_path)
print(" test :", test_csv_path)
print(" sub  :", sample_sub_path)




## === cell 1
def find_candidate_submission_csvs(max_files=50):
    candidates = []

    scan_dirs = []
    for r in INPUT_ROOTS:
        if r and os.path.exists(r):
            scan_dirs.append(r)
            ds_dir = os.path.join(r, DATASET_SUBDIR)
            if os.path.exists(ds_dir):
                scan_dirs.append(ds_dir)

    for d in scan_dirs:
        patterns = [
            os.path.join(d, "*.csv"),
            os.path.join(d, "**", "*.csv"),
        ]
        for pat in patterns:
            for f in glob.glob(pat, recursive=True):
                base = os.path.basename(f).lower()
                if base in {"train.csv", "test.csv", "sample_submission.csv"}:
                    continue
                candidates.append(f)

    seen = set()
    uniq = []
    for f in candidates:
        if f not in seen:
            seen.add(f)
            uniq.append(f)

    valid = []
    for f in uniq:
        try:
            head = pd.read_csv(f, nrows=50)
            cols = set(head.columns)

            if not ("image_name" in cols and "target" in cols):
                continue
            if len(head.columns) != 2:
                continue
            if head["image_name"].isna().any() or head["target"].isna().any():
                continue

            t = pd.to_numeric(head["target"], errors="coerce")
            if t.isna().any():
                continue

            if (t.min() < -1e-6) or (t.max() > 1.0 + 1e-6):
                continue

            valid.append(f)
        except Exception:
            continue

    return valid[:max_files]


all_files = find_candidate_submission_csvs()
print("Found candidate submission-like CSVs:", len(all_files))
for f in all_files[:10]:
    print(" ", f)




## === cell 2
def load_submission_as_series(path, test_image_names):
    df = pd.read_csv(path)

    if "image_name" not in df.columns:
        df2 = pd.read_csv(path, index_col=0)
        if df2.index.name != "image_name":
            df2.index.name = "image_name"
        df = df2.reset_index()

    if "target" not in df.columns or "image_name" not in df.columns:
        raise ValueError(f"Missing required columns in {path}")

    df = df[["image_name", "target"]].copy()
    df = df.set_index("image_name").reindex(test_image_names)
    if df["target"].isna().any():
        raise ValueError(f"Submission {path} does not cover the current test set.")
    return df["target"].astype(float).values


test_df = pd.read_csv(test_csv_path)
test_image_names = test_df["image_name"].tolist()

outs = []
used_files = []
for f in all_files:
    try:
        preds = load_submission_as_series(f, test_image_names)
        outs.append(pd.DataFrame({os.path.basename(f): preds}, index=test_image_names))
        used_files.append(f)
    except Exception:
        continue

print("Usable submission CSVs:", len(outs))
for f in used_files[:10]:
    print(" using:", f)

concat_sub = None
if len(outs) > 0:
    concat_sub = pd.concat(outs, axis=1)
    concat_sub.index.name = "image_name"
    concat_sub.reset_index(inplace=True)



## === cell 3
m_gmean = None
if concat_sub is not None:
    rank_mat = np.tril(concat_sub.iloc[:, 1:].corr().values, -1)
    m = (rank_mat > 0).sum()

    m_gmean_acc, s = 0.0, 0.0
    eps = 1e-12
    for n in range(min(rank_mat.shape[0], m)):
        mx = np.unravel_index(rank_mat.argmin(), rank_mat.shape)
        w = (m - n) / (m + n / 10) if (m + n / 10) != 0 else 1.0

        a = np.clip(concat_sub.iloc[:, mx[0] + 1].values.astype(float), eps, 1.0)
        b = np.clip(concat_sub.iloc[:, mx[1] + 1].values.astype(float), eps, 1.0)
        m_gmean_acc += w * (np.log(a) + np.log(b)) / 2.0
        s += w
        rank_mat[mx] = 1

    if s > 0:
        m_gmean = np.exp(m_gmean_acc / s).clip(0.0, 1.0)



## === cell 4
predict_list = []
if concat_sub is not None:
    for c in concat_sub.columns[1:]:
        predict_list.append(concat_sub[[c]].values)

print("Rank averaging on", len(predict_list), "files")

predictions = None
if len(predict_list) > 0:
    predictions = np.zeros_like(predict_list[0], dtype=float)
    for predict in predict_list:
        for i in range(1):
            predictions[:, i] = np.add(
                predictions[:, i], rankdata(predict[:, i]) / predictions.shape[0]
            )
    predictions = predictions / len(predict_list)

    if m_gmean is not None:
        predictions[:, 0] = (0.95 * predictions[:, 0] + 0.05 * m_gmean).clip(0.0, 1.0)




## === cell 5
def _baseline_metadata_predictions(train_path, test_path, n_splits=5, seed=42):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    base_num = ["age_approx"]
    base_cat = ["sex", "anatom_site_general_challenge"]

    y_train = train_df["target"].astype(int).values
    groups = train_df["patient_id"].values

    def _augment(df):
        out = df.copy()

        age = pd.to_numeric(out["age_approx"], errors="coerce")
        out["age_missing"] = age.isna().astype(np.int8)

        age_filled0 = age.fillna(0.0).astype(float)
        out["age_log1p"] = np.log1p(age_filled0)

        age_bin = pd.cut(
            age.fillna(-1),
            bins=[-2, 0, 20, 30, 40, 50, 60, 70, 80, 200],
            labels=False,
            include_lowest=True,
        )
        out["age_bin"] = age_bin.astype("Int64")

        out["age_x_missing"] = out["age_log1p"].astype(float) * out[
            "age_missing"
        ].astype(float)

        for c in base_cat:
            out[c] = out[c].astype("object").where(pd.notna(out[c]), "__MISSING__")
        out["age_bin"] = (
            out["age_bin"]
            .astype("object")
            .where(pd.notna(out["age_bin"]), "__MISSING__")
        )

        return out

    tr = _augment(train_df[base_num + base_cat])
    te = _augment(test_df[base_num + base_cat])

    gkf = GroupKFold(n_splits=n_splits)
    global_mean = float(np.mean(y_train))

    tr["patient_id"] = train_df["patient_id"].values
    te["patient_id"] = test_df["patient_id"].values

    def _oof_target_encode(series_tr, series_te, y, groups_arr, alpha):
        oof = np.zeros(len(series_tr), dtype=float)
        for tr_idx, va_idx in gkf.split(series_tr, y, groups=groups_arr):
            s_tr = series_tr.iloc[tr_idx]
            y_tr = y[tr_idx]
            stats = (
                pd.DataFrame({"k": s_tr.values, "y": y_tr})
                .groupby("k", dropna=False)["y"]
                .agg(["mean", "count"])
            )
            m = (
                series_tr.iloc[va_idx]
                .map(stats["mean"])
                .fillna(global_mean)
                .astype(float)
            )
            c = series_tr.iloc[va_idx].map(stats["count"]).fillna(0.0).astype(float)
            oof[va_idx] = ((m * c + global_mean * alpha) / (c + alpha)).values

        stats_full = (
            pd.DataFrame({"k": series_tr.values, "y": y})
            .groupby("k", dropna=False)["y"]
            .agg(["mean", "count"])
        )
        m2 = series_te.map(stats_full["mean"]).fillna(global_mean).astype(float)
        c2 = series_te.map(stats_full["count"]).fillna(0.0).astype(float)
        te_enc = ((m2 * c2 + global_mean * alpha) / (c2 + alpha)).values
        return oof, te_enc

    alpha_patient = 8.0
    oof_patient = np.zeros(len(tr), dtype=float)
    for tr_idx, va_idx in gkf.split(tr, y_train, groups=groups):
        fold_tr = tr.iloc[tr_idx]
        fold_y = y_train[tr_idx]
        stats = (
            pd.DataFrame({"patient_id": fold_tr["patient_id"].values, "y": fold_y})
            .groupby("patient_id")["y"]
            .agg(["mean", "count"])
        )
        mapped = (
            tr.iloc[va_idx]["patient_id"]
            .map(stats["mean"])
            .fillna(global_mean)
            .astype(float)
        )
        mapped_cnt = (
            tr.iloc[va_idx]["patient_id"].map(stats["count"]).fillna(0.0).astype(float)
        )
        oof_patient[va_idx] = (
            (mapped * mapped_cnt + global_mean * alpha_patient)
            / (mapped_cnt + alpha_patient)
        ).values

    stats_full = (
        pd.DataFrame({"patient_id": tr["patient_id"].values, "y": y_train})
        .groupby("patient_id")["y"]
        .agg(["mean", "count"])
    )
    te_m = te["patient_id"].map(stats_full["mean"]).fillna(global_mean).astype(float)
    te_c = te["patient_id"].map(stats_full["count"]).fillna(0.0).astype(float)
    te_patient = (
        (te_m * te_c + global_mean * alpha_patient) / (te_c + alpha_patient)
    ).values

    tr["patient_prior"] = oof_patient
    te["patient_prior"] = te_patient

    alpha_te = 20.0
    for col in base_cat + ["age_bin"]:
        oof_te = np.zeros(len(tr), dtype=float)
        for tr_idx, va_idx in gkf.split(tr, y_train, groups=groups):
            fold_tr = tr.iloc[tr_idx]
            fold_y = y_train[tr_idx]
            stats = (
                pd.DataFrame({col: fold_tr[col], "y": fold_y})
                .groupby(col, dropna=False)["y"]
                .agg(["mean", "count"])
            )
            m = (
                tr.iloc[va_idx][col]
                .map(stats["mean"])
                .fillna(global_mean)
                .astype(float)
            )
            c = tr.iloc[va_idx][col].map(stats["count"]).fillna(0.0).astype(float)
            oof_te[va_idx] = ((m * c + global_mean * alpha_te) / (c + alpha_te)).values

        stats_full = (
            pd.DataFrame({col: tr[col], "y": y_train})
            .groupby(col, dropna=False)["y"]
            .agg(["mean", "count"])
        )
        m2 = te[col].map(stats_full["mean"]).fillna(global_mean).astype(float)
        c2 = te[col].map(stats_full["count"]).fillna(0.0).astype(float)
        te_te = ((m2 * c2 + global_mean * alpha_te) / (c2 + alpha_te)).values

        tr[f"{col}__te"] = oof_te
        te[f"{col}__te"] = te_te

    anat = tr["anatom_site_general_challenge"].astype("object")
    anat_te = te["anatom_site_general_challenge"].astype("object")
    oof_anat, te_anat = _oof_target_encode(anat, anat_te, y_train, groups, alpha=12.0)
    tr["anat_prior"] = oof_anat
    te["anat_prior"] = te_anat

    sex_anat = (
        tr["sex"].astype("object")
        + "||"
        + tr["anatom_site_general_challenge"].astype("object")
    ).astype("object")
    sex_anat_te = (
        te["sex"].astype("object")
        + "||"
        + te["anatom_site_general_challenge"].astype("object")
    ).astype("object")
    oof_sex_anat, te_sex_anat = _oof_target_encode(
        sex_anat, sex_anat_te, y_train, groups, alpha=20.0
    )
    tr["sex_anat_prior"] = oof_sex_anat
    te["sex_anat_prior"] = te_sex_anat

    feature_cols_num = [
        "age_approx",
        "age_missing",
        "age_log1p",
        "age_x_missing",
        "patient_prior",
        "anat_prior",
        "sex_anat_prior",
    ] + [f"{c}__te" for c in (base_cat + ["age_bin"])]

    feature_cols_cat = base_cat + ["age_bin"]

    feature_cols_num = list(dict.fromkeys(feature_cols_num))
    feature_cols_cat = list(dict.fromkeys(feature_cols_cat))

    X_train = tr[feature_cols_num + feature_cols_cat].copy()
    X_test = te[feature_cols_num + feature_cols_cat].copy()

    for c in feature_cols_num:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype(float)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(float)

    for c in feature_cols_cat:
        X_train[c] = (
            X_train[c].astype("object").where(pd.notna(X_train[c]), "__MISSING__")
        )
        X_test[c] = X_test[c].astype("object").where(pd.notna(X_test[c]), "__MISSING__")

    numeric_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, feature_cols_num),
            ("cat", categorical_tf, feature_cols_cat),
        ],
        remainder="drop",
    )

    clf_1 = LogisticRegression(
        solver="lbfgs",
        max_iter=1200,
        n_jobs=1,
        class_weight="balanced",
        random_state=seed,
        C=1.0,
    )
    clf_2 = LogisticRegression(
        solver="lbfgs",
        max_iter=1200,
        n_jobs=1,
        class_weight="balanced",
        random_state=seed,
        C=0.3,
    )

    pipe_1 = Pipeline(steps=[("pre", pre), ("clf", clf_1)])
    pipe_2 = Pipeline(steps=[("pre", pre), ("clf", clf_2)])

    oof_1 = np.zeros(len(train_df), dtype=float)
    oof_2 = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups=groups):
        pipe_fold_1 = Pipeline(steps=pipe_1.steps)
        pipe_fold_2 = Pipeline(steps=pipe_2.steps)
        pipe_fold_1.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        pipe_fold_2.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_1[va_idx] = pipe_fold_1.predict_proba(X_train.iloc[va_idx])[:, 1].astype(
            float
        )
        oof_2[va_idx] = pipe_fold_2.predict_proba(X_train.iloc[va_idx])[:, 1].astype(
            float
        )

    oof_lr = 0.5 * oof_1 + 0.5 * oof_2
    oof_patient_prior = tr["patient_prior"].astype(float).values

    best_w = 1.0
    best_auc = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        blended = np.clip(w * oof_lr + (1.0 - w) * oof_patient_prior, 0.0, 1.0)
        auc = roc_auc_score(y_train, blended)
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)

    try:
        auc_lr = roc_auc_score(y_train, oof_lr)
        auc_pp = roc_auc_score(y_train, oof_patient_prior)
        print(
            f"OOF AUC: LR_avg={auc_lr:.5f}, patient_prior={auc_pp:.5f}, blended(w={best_w:.2f})={best_auc:.5f}"
        )
    except Exception:
        pass

    pipe_1.fit(X_train, y_train)
    pipe_2.fit(X_train, y_train)

    proba_1 = pipe_1.predict_proba(X_test)[:, 1].astype(float)
    proba_2 = pipe_2.predict_proba(X_test)[:, 1].astype(float)
    proba_lr = 0.5 * proba_1 + 0.5 * proba_2

    proba_patient_prior = te["patient_prior"].astype(float).values

    proba = np.clip(best_w * proba_lr + (1.0 - best_w) * proba_patient_prior, 0.0, 1.0)
    return proba


submission = pd.read_csv(sample_sub_path)

if predictions is None:
    proba = _baseline_metadata_predictions(train_csv_path, test_csv_path)
    sub_df = pd.DataFrame({"image_name": test_df["image_name"].values, "target": proba})

    submission = submission[["image_name"]].merge(sub_df, on="image_name", how="left")
    if submission["target"].isna().any():
        raise ValueError("Some test image_names did not receive predictions.")
    submission["target"] = np.clip(submission["target"].astype(float).values, 0.0, 1.0)
else:
    submission[LABELS] = predictions

submission = submission[["image_name", "target"]].copy()
submission["target"] = submission["target"].astype(float).clip(0.0, 1.0)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "with shape", submission.shape)
print(submission.head())
