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

DATA_DIR = "/kaggle/data"  # as listed in the provided paths

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_sub_path)

assert "target" in train.columns
assert "image_name" in test.columns and "image_name" in sub.columns
assert len(sub) == len(test)

JPEG_DIR_CANDIDATES = [
    f"{DATA_DIR}/jpeg",
    f"{DATA_DIR}/siim-isic-melanoma-classification/jpeg",
    "/kaggle/input/siim-isic-melanoma-classification/jpeg",
    "/kaggle/input/jpeg",
]
JPEG_DIR = next((p for p in JPEG_DIR_CANDIDATES if os.path.isdir(p)), None)
if JPEG_DIR is None:
    raise FileNotFoundError(
        "Could not find a valid JPEG directory in expected locations."
    )

TRAIN_JPEG_DIR = os.path.join(JPEG_DIR, "train")
TEST_JPEG_DIR = os.path.join(JPEG_DIR, "test")




## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

tr = train.copy(deep=False)
te = test.copy(deep=False)

for c in ["sex", "anatom_site_general_challenge"]:
    tr[c] = tr[c].fillna("unknown").astype(str).str.strip().str.lower()
    te[c] = te[c].fillna("unknown").astype(str).str.strip().str.lower()

tr["age_approx"] = pd.to_numeric(tr["age_approx"], errors="coerce")
te["age_approx"] = pd.to_numeric(te["age_approx"], errors="coerce")

age_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
age_labels = ["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"]

tr["age_bin"] = pd.cut(tr["age_approx"], bins=age_bins, labels=age_labels)
te["age_bin"] = pd.cut(te["age_approx"], bins=age_bins, labels=age_labels)

tr["age_bin"] = tr["age_bin"].astype(str).fillna("nan")
te["age_bin"] = te["age_bin"].astype(str).fillna("nan")

global_mean = float(tr["target"].mean())

tr["sex"] = tr["sex"].astype("category")
te["sex"] = te["sex"].astype("category")
tr["anatom_site_general_challenge"] = tr["anatom_site_general_challenge"].astype(
    "category"
)
te["anatom_site_general_challenge"] = te["anatom_site_general_challenge"].astype(
    "category"
)
tr["age_bin"] = tr["age_bin"].astype("category")
te["age_bin"] = te["age_bin"].astype("category")


def smoothed_rate(df, key, target_col="target", prior=None, m=200):
    if prior is None:
        prior = float(df[target_col].mean())
    stats = df.groupby(key, observed=True)[target_col].agg(["mean", "count"])
    stats["smooth"] = (stats["count"] * stats["mean"] + m * prior) / (
        stats["count"] + m
    )
    return stats["smooth"]


def build_te_maps(df_fit, prior, m_sex=500, m_site=500, m_age=500, m_pid=1000):
    sex_rate = smoothed_rate(df_fit, "sex", prior=prior, m=m_sex)
    site_rate = smoothed_rate(
        df_fit, "anatom_site_general_challenge", prior=prior, m=m_site
    )
    age_rate = smoothed_rate(df_fit, "age_bin", prior=prior, m=m_age)
    pid_rate = smoothed_rate(df_fit, "patient_id", prior=prior, m=m_pid)
    return sex_rate, site_rate, age_rate, pid_rate


def apply_te(df, sex_rate, site_rate, age_rate, pid_rate, fallback):
    x_sex = df["sex"].map(sex_rate).fillna(fallback).astype(float)
    x_site = (
        df["anatom_site_general_challenge"]
        .map(site_rate)
        .fillna(fallback)
        .astype(float)
    )
    x_age = df["age_bin"].map(age_rate).fillna(fallback).astype(float)
    x_pid = df["patient_id"].map(pid_rate).fillna(fallback).astype(float)
    X = np.vstack([x_sex.values, x_site.values, x_age.values, x_pid.values]).T
    X = np.nan_to_num(X, nan=fallback, posinf=fallback, neginf=fallback)
    return X




## === cell 2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

try:
    from multiprocessing import shared_memory
except Exception:
    shared_memory = None


def _img_stats_from_path(path):
    """
    Returns 5 deterministic features:
    - mean_gray, std_gray (computed from RGB -> gray by simple average)
    - mean_r, mean_g, mean_b
    If file missing/unreadable, returns NaNs (handled later).
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((128, 128))  # fixed small size for speed; deterministic
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        gray = arr.mean(axis=2)
        mean_gray = float(gray.mean())
        std_gray = float(gray.std())
        return np.array(
            [mean_gray, std_gray, mean_rgb[0], mean_rgb[1], mean_rgb[2]],
            dtype=np.float32,
        )
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)


def _img_chunk_worker_shared(args):
    """
    Worker processes a contiguous chunk [start:end) of image_names and writes to shared array.
    """
    (start, end, names, folder, shm_name, shape, dtype_str) = args
    dtype = np.dtype(dtype_str)
    shm = shared_memory.SharedMemory(name=shm_name)
    out = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    join = os.path.join
    stats_fn = _img_stats_from_path
    for i in range(start, end):
        name = names[i]
        out[i] = stats_fn(join(folder, f"{name}.jpg"))
    shm.close()
    return None


def _img_chunk_worker_return(args):
    """
    Fallback worker when shared_memory isn't available: returns (start, chunk_feats).
    """
    (start, end, names, folder) = args
    join = os.path.join
    stats_fn = _img_stats_from_path
    chunk = np.zeros((end - start, 5), dtype=np.float32)
    for j, i in enumerate(range(start, end)):
        chunk[j] = stats_fn(join(folder, f"{names[i]}.jpg"))
    return start, chunk


def build_image_features(
    image_names, folder, cache_path=None, max_workers=None, chunksize=512
):
    if cache_path is not None and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            feats = arr["feats"]
        else:
            feats = arr
        if feats.shape == (len(image_names), 5):
            return feats.astype(np.float32, copy=False)

    n = len(image_names)
    feats = np.zeros((n, 5), dtype=np.float32)

    if max_workers is None:
        max_workers = max(1, min(8, (os.cpu_count() or 2)))

    bounds = list(range(0, n, chunksize))
    chunks = [(s, min(s + chunksize, n)) for s in bounds]

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else None

    if shared_memory is not None:
        shm = shared_memory.SharedMemory(create=True, size=feats.nbytes)
        shm_arr = np.ndarray(feats.shape, dtype=feats.dtype, buffer=shm.buf)
        shm_arr[:] = 0.0

        tasks = [
            (s, e, image_names, folder, shm.name, feats.shape, feats.dtype.str)
            for (s, e) in chunks
        ]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            for _ in ex.map(_img_chunk_worker_shared, tasks, chunksize=1):
                pass

        feats[:] = shm_arr
        shm.close()
        shm.unlink()
    else:
        tasks = [(s, e, image_names, folder) for (s, e) in chunks]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            for start, chunk in ex.map(_img_chunk_worker_return, tasks, chunksize=1):
                feats[start : start + chunk.shape[0]] = chunk

    if cache_path is not None:
        np.save(cache_path, feats)
    return feats


tr_names = tr["image_name"].values
te_names = te["image_name"].values

tr_cache = os.path.join("/kaggle/working", "tr_img_feats_128.npy")
te_cache = os.path.join("/kaggle/working", "te_img_feats_128.npy")

tr_img = build_image_features(
    tr_names, TRAIN_JPEG_DIR, cache_path=tr_cache, chunksize=768
)
te_img = build_image_features(
    te_names, TEST_JPEG_DIR, cache_path=te_cache, chunksize=768
)

col_means = np.nanmean(tr_img, axis=0)
col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
tr_img = np.where(np.isfinite(tr_img), tr_img, col_means)
te_img = np.where(np.isfinite(te_img), te_img, col_means)




## === cell 3
gkf = GroupKFold(n_splits=5)
groups = tr["patient_id"].astype(str).values
y = tr["target"].values.astype(int)

oof_X = np.zeros((len(tr), 9), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(tr, y, groups=groups), 1):
    df_fit = tr.iloc[tr_idx]
    df_val = tr.iloc[va_idx]
    prior = float(df_fit["target"].mean())

    sex_rate, site_rate, age_rate, pid_rate = build_te_maps(df_fit, prior=prior)
    X_te_val = apply_te(df_val, sex_rate, site_rate, age_rate, pid_rate, fallback=prior)

    X_img_val = tr_img[va_idx].astype(np.float64)

    oof_X[va_idx] = np.hstack([X_te_val.astype(np.float64), X_img_val])

blender = LogisticRegression(
    solver="lbfgs",
    max_iter=400,  # unchanged
    C=1.0,
    class_weight=None,
    random_state=0,
)
blender.fit(oof_X, y)

oof_pred = blender.predict_proba(oof_X)[:, 1]
oof_auc = roc_auc_score(y, oof_pred)
print("OOF ROC-AUC (blender on TE+image features):", oof_auc)

sex_rate, site_rate, age_rate, pid_rate = build_te_maps(tr, prior=global_mean)
X_te_test = apply_te(te, sex_rate, site_rate, age_rate, pid_rate, fallback=global_mean)
X_test = np.hstack([X_te_test.astype(np.float64), te_img.astype(np.float64)])

pred = blender.predict_proba(X_test)[:, 1].astype(float)
pred = np.clip(pred, 0.0, 1.0)

pred_df = pd.DataFrame({"image_name": te["image_name"].values, "target": pred})
sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")
sub_out["target"] = sub_out["target"].fillna(global_mean).astype(float)

sub_out.to_csv("submission.csv", index=False)
sub_out.head()
