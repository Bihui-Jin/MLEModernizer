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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook is trying to ensemble a set of external “team submissions” that are not present in this Kaggle environment, so `glob()` finds no CSVs and everything downstream crashes. I keep the same rank-averaging ensemble core logic, but make it robust by (1) searching for any valid submission-like CSVs under the available `../input` tree and (2) falling back to a simple, deterministic baseline using `train.csv` target mean if none are found, so a valid `submission.csv` is always produced. I also remove notebook-only magic (`%matplotlib inline`) and fix pathing to the actual provided dataset locations. These changes are execution/stability focused; any score gain depends on whether compatible prediction CSVs exist in the input, otherwise the fallback be score-neutral (but valid).'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC is coming from the fallback path producing a constant prediction (train target mean), which yields ~random ranking and thus AUC≈0.5. To move toward the 0.9423 target without changing the “submission-ensemble” core logic, I keep the same rank-averaging flow but ensure we actually find usable prediction CSVs by expanding the search to `/kaggle/data` and `/kaggle/input` (your environment paths), and by explicitly excluding the competition’s own CSVs. If no external prediction CSVs exist, I keep the fallback (so it always produces a valid submission), but this change should allow the ensemble path to trigger when such files are present, improving AUC substantially toward your target.'

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
            head = pd.read_csv(f, nrows=5)
            cols = set(head.columns)
            if "target" in cols and (
                "image_name" in cols or head.index.name == "image_name"
            ):
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
        df = pd.read_csv(path, index_col=0)
        if df.index.name != "image_name":
            df.index.name = "image_name"
        df = df.reset_index()

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
submission = pd.read_csv(sample_sub_path)

if predictions is None:
    train_df = pd.read_csv(train_csv_path)
    prior = float(train_df["target"].mean())
    submission["target"] = prior
else:
    submission[LABELS] = predictions

submission = submission[["image_name", "target"]].copy()
submission["target"] = submission["target"].astype(float).clip(0.0, 1.0)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "with shape", submission.shape)
print(submission.head())
