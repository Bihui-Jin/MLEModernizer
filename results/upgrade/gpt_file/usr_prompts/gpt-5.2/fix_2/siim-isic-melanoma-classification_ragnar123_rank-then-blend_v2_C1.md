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

0.929009075916602

# 6. Current score

0.58641

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.58641) has done: 'Your notebook fails because it tries to read five external submission CSVs from `../input/melanoma-dif-sub/`, which are not present in this environment. To make it run end-to-end and still produce a valid submission, I keep the same “rank + weighted blend” core logic but replace the missing inputs with a simple, deterministic baseline probability built from `train.csv` and the available test metadata (age/sex/site) using out-of-fold target encoding (no leakage). This yields a reasonable AUC-oriented submission without changing the submission schema, and it always write a `.csv` file (`blend_sub.csv`) with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(fname: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {fname} in any of: {DATA_DIR_CANDIDATES}")


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def _prep_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
    for c in ["sex", "anatom_site_general_challenge"]:
        if c in df.columns:
            df[c] = (
                df[c].fillna("unknown").astype(str).str.strip().replace({"": "unknown"})
            )
        else:
            df[c] = "unknown"
    return df


train = _prep_meta(train)
test = _prep_meta(test)

y = train["target"].astype(float).values
global_mean = float(np.mean(y))


def _make_folds(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


folds = _make_folds(len(train), k=5, seed=42)


def oof_target_encode(
    train_df, test_df, col, min_samples_leaf=50, smoothing=10.0, noise=0.0, seed=0
):
    """
    Out-of-fold smoothed mean encoding for binary target.
    Returns: oof_train_encoded, test_encoded
    """
    rng = np.random.RandomState(seed)
    oof = np.zeros(len(train_df), dtype=np.float64)
    test_enc = np.zeros(len(test_df), dtype=np.float64)

    stats_full = train_df.groupby(col)["target"].agg(["mean", "count"])
    smooth_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    test_enc = (
        test_df[col].map(smooth_full).fillna(global_mean).astype(np.float64).values
    )

    for val_idx in folds:
        tr_idx = np.setdiff1d(np.arange(len(train_df)), val_idx, assume_unique=False)
        tr = train_df.iloc[tr_idx]
        stats = tr.groupby(col)["target"].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        oof[val_idx] = (
            train_df.iloc[val_idx][col]
            .map(smooth)
            .fillna(global_mean)
            .astype(np.float64)
            .values
        )

    if noise > 0:
        oof = np.clip(oof + rng.normal(0, noise, size=oof.shape), 0, 1)
        test_enc = np.clip(test_enc + rng.normal(0, noise, size=test_enc.shape), 0, 1)

    return oof, test_enc


train_tmp = train[
    ["image_name", "target", "sex", "age_approx", "anatom_site_general_challenge"]
].copy()
test_tmp = test[
    ["image_name", "sex", "age_approx", "anatom_site_general_challenge"]
].copy()

age_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
train_tmp["age_bin"] = pd.cut(train_tmp["age_approx"], bins=age_bins).astype(str)
test_tmp["age_bin"] = pd.cut(test_tmp["age_approx"], bins=age_bins).astype(str)

train_tmp["sex_site"] = (
    train_tmp["sex"].astype(str)
    + "_"
    + train_tmp["anatom_site_general_challenge"].astype(str)
)
test_tmp["sex_site"] = (
    test_tmp["sex"].astype(str)
    + "_"
    + test_tmp["anatom_site_general_challenge"].astype(str)
)

train_tmp["site_age"] = (
    train_tmp["anatom_site_general_challenge"].astype(str)
    + "_"
    + train_tmp["age_bin"].astype(str)
)
test_tmp["site_age"] = (
    test_tmp["anatom_site_general_challenge"].astype(str)
    + "_"
    + test_tmp["age_bin"].astype(str)
)

_, te1 = oof_target_encode(
    train_tmp,
    test_tmp,
    "anatom_site_general_challenge",
    smoothing=5.0,
    noise=0.0,
    seed=1,
)
_, te2 = oof_target_encode(
    train_tmp, test_tmp, "sex", smoothing=20.0, noise=0.0, seed=2
)
_, te3 = oof_target_encode(
    train_tmp, test_tmp, "age_bin", smoothing=10.0, noise=0.0, seed=3
)
_, te4 = oof_target_encode(
    train_tmp, test_tmp, "sex_site", smoothing=30.0, noise=0.0, seed=4
)
_, te5 = oof_target_encode(
    train_tmp, test_tmp, "site_age", smoothing=30.0, noise=0.0, seed=5
)

sub1 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te1})
sub2 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te2})
sub3 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te3})
sub4 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te4})
sub5 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te5})


def rank_data(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    r = sub["target"].rank(method="average")
    sub["target"] = (r / r.max()).astype(np.float64)
    return sub


sub1 = rank_data(sub1)
sub2 = rank_data(sub2)
sub3 = rank_data(sub3)
sub4 = rank_data(sub4)
sub5 = rank_data(sub5)

sub1.columns = ["image_name", "target1"]
sub2.columns = ["image_name", "target2"]
sub3.columns = ["image_name", "target3"]
sub4.columns = ["image_name", "target4"]
sub5.columns = ["image_name", "target5"]

f_sub = (
    sub1.merge(sub2, on="image_name")
    .merge(sub3, on="image_name")
    .merge(sub4, on="image_name")
    .merge(sub5, on="image_name")
)

f_sub["target"] = (
    f_sub["target1"] * 0.3
    + f_sub["target2"] * 0.3
    + f_sub["target3"] * 0.05
    + f_sub["target4"] * 0.3
    + f_sub["target5"] * 0.05
)

f_sub = f_sub[["image_name", "target"]]

sample_path = _find_file("sample_submission.csv")
sample = pd.read_csv(sample_path)[["image_name"]]
f_sub = sample.merge(f_sub, on="image_name", how="left")
f_sub["target"] = f_sub["target"].fillna(global_mean).clip(0, 1)

f_sub.to_csv("blend_sub.csv", index=False)

print(
    f"Saved submission: blend_sub.csv with shape {f_sub.shape} and columns {list(f_sub.columns)}"
)
print(f_sub.head())
