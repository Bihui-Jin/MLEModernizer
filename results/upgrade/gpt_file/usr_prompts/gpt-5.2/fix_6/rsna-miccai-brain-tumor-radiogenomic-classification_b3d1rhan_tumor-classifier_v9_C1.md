# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

WORK_DIR = "."
TESTI_DIR = os.path.join(WORK_DIR, "testi")



## === cell 2
os.makedirs(TESTI_DIR, exist_ok=True)



## === cell 3
import cv2
import pydicom
from pydicom.pixel_data_handlers import apply_voi_lut




## === cell 4
def _sorted_dicom_files(flair_dir):
    dcm_files = [f for f in os.listdir(flair_dir) if f.lower().endswith(".dcm")]
    keyed = []
    for f in dcm_files:
        p = os.path.join(flair_dir, f)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
                key = (0, float(ds.InstanceNumber))
            elif (
                hasattr(ds, "ImagePositionPatient")
                and ds.ImagePositionPatient is not None
            ):
                key = (1, float(ds.ImagePositionPatient[2]))
            else:
                key = (2, f)
        except Exception:
            key = (3, f)
        keyed.append((key, f))
    keyed.sort(key=lambda x: x[0])
    return [f for _, f in keyed]




## === cell 5
cases_list = sorted(
    [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
)

for case_id in cases_list:
    flair_dir = os.path.join(TEST_DIR, case_id, "FLAIR")
    if not os.path.isdir(flair_dir):
        continue

    out_case_dir = os.path.join(TESTI_DIR, case_id, "FLAIR")
    os.makedirs(out_case_dir, exist_ok=True)

    existing_pngs = [f for f in os.listdir(out_case_dir) if f.lower().endswith(".png")]
    if len(existing_pngs) > 0:
        continue

    liste_names = _sorted_dicom_files(flair_dir)

    for i, k in enumerate(liste_names):
        dcm_path = os.path.join(flair_dir, k)
        dicom = pydicom.dcmread(dcm_path, force=True)

        data = apply_voi_lut(dicom.pixel_array, dicom)
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            data = np.amax(data) - data

        data = data.astype(np.float32)
        data = data - np.min(data)
        mx = np.max(data)
        if mx > 0:
            data = data / mx
        data = (data * 255.0).astype(np.uint8)

        out_path = os.path.join(out_case_dir, f"{i:04d}.png")
        cv2.imwrite(out_path, data)



## === cell 6
from PIL import Image




## === cell 7
def _load_case_stack(case_id, max_slices=None):
    """Load case pngs from ./testi/<case>/FLAIR into a numpy stack of shape [N, 128, 128]."""
    flair_png_dir = os.path.join(TESTI_DIR, case_id, "FLAIR")
    if not os.path.isdir(flair_png_dir):
        return None

    files = sorted([f for f in os.listdir(flair_png_dir) if f.lower().endswith(".png")])
    if len(files) == 0:
        return None

    if max_slices is not None:
        files = files[:max_slices]

    stack = []
    for f in files:
        img = Image.open(os.path.join(flair_png_dir, f)).convert("L")
        img.thumbnail((128, 128), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)

        h, w = arr.shape
        out = np.zeros((128, 128), dtype=np.float32)
        out[: min(h, 128), : min(w, 128)] = arr[: min(h, 128), : min(w, 128)]
        stack.append(out)

    return np.stack(stack, axis=0)




## === cell 8
def _trimmed_mean(x, trim_frac=0.15):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    x = np.sort(x)
    k = int(trim_frac * x.size)
    if 2 * k >= x.size:
        return float(np.mean(x))
    return float(np.mean(x[k:-k]))


def _select_informative_slices(
    per_slice_contrast, min_center=24, center_frac=0.35, topk_frac=0.25
):
    """
    Why: keep the same aggregation idea but pick slices likely to contain brain/tumor signal.
    """
    n = int(len(per_slice_contrast))
    if n == 0:
        return np.array([], dtype=np.int32)

    if n >= min_center:
        center = n // 2
        half = max(10, int(center_frac * n))
        lo = max(0, center - half)
        hi = min(n, center + half)
        idx_pool = np.arange(lo, hi)
    else:
        idx_pool = np.arange(n)

    contrast_pool = per_slice_contrast[idx_pool]
    k = max(1, int(topk_frac * len(idx_pool)))
    top_local = np.argsort(contrast_pool)[-k:]
    top_idx = idx_pool[top_local]
    return np.asarray(top_idx, dtype=np.int32)




## === cell 9
def _iter_train_ids(base_train_dir):
    ids = sorted(
        [
            d
            for d in os.listdir(base_train_dir)
            if os.path.isdir(os.path.join(base_train_dir, d))
        ]
    )
    bad = set(["00109", "00123", "00709"])
    return [i for i in ids if i not in bad]


def _ensure_train_png_cache_exists(train_ids, limit=None):
    if limit is not None:
        train_ids = train_ids[:limit]

    for case_id in train_ids:
        flair_dir = os.path.join(TRAIN_DIR, case_id, "FLAIR")
        if not os.path.isdir(flair_dir):
            continue

        out_case_dir = os.path.join(TESTI_DIR, "_train_cache", case_id, "FLAIR")
        if (
            os.path.isdir(out_case_dir)
            and len([f for f in os.listdir(out_case_dir) if f.lower().endswith(".png")])
            > 0
        ):
            continue

        os.makedirs(out_case_dir, exist_ok=True)
        liste_names = _sorted_dicom_files(flair_dir)

        for i, k in enumerate(liste_names):
            dcm_path = os.path.join(flair_dir, k)
            try:
                dicom = pydicom.dcmread(dcm_path, force=True)
                data = apply_voi_lut(dicom.pixel_array, dicom)
                if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
                    data = np.amax(data) - data

                data = data.astype(np.float32)
                data = data - np.min(data)
                mx = np.max(data)
                if mx > 0:
                    data = data / mx
                data = (data * 255.0).astype(np.uint8)
                out_path = os.path.join(out_case_dir, f"{i:04d}.png")
                cv2.imwrite(out_path, data)
            except Exception:
                continue


def _load_train_case_stack(case_id):
    flair_png_dir = os.path.join(TESTI_DIR, "_train_cache", case_id, "FLAIR")
    if not os.path.isdir(flair_png_dir):
        return None
    files = sorted([f for f in os.listdir(flair_png_dir) if f.lower().endswith(".png")])
    if not files:
        return None
    stack = []
    for f in files:
        img = Image.open(os.path.join(flair_png_dir, f)).convert("L")
        img.thumbnail((128, 128), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape
        out = np.zeros((128, 128), dtype=np.float32)
        out[: min(h, 128), : min(w, 128)] = arr[: min(h, 128), : min(w, 128)]
        stack.append(out)
    return np.stack(stack, axis=0)


def _compute_feature_triplet_from_stack(stack):
    stack = stack / 255.0
    per_slice_median = np.median(stack, axis=(1, 2))
    per_slice_p95 = np.percentile(stack, 95, axis=(1, 2))
    per_slice_p05 = np.percentile(stack, 5, axis=(1, 2))
    per_slice_contrast = per_slice_p95 - per_slice_p05
    per_slice_bright_frac = np.mean(stack > 0.80, axis=(1, 2))

    top_idx = _select_informative_slices(
        per_slice_contrast, min_center=24, center_frac=0.33, topk_frac=0.22
    )
    if top_idx.size == 0:
        top_idx = np.arange(len(per_slice_contrast), dtype=np.int32)

    med_agg = _trimmed_mean(per_slice_median[top_idx], trim_frac=0.15)
    con_agg = _trimmed_mean(per_slice_contrast[top_idx], trim_frac=0.15)
    bright_agg = _trimmed_mean(per_slice_bright_frac[top_idx], trim_frac=0.10)
    return med_agg, con_agg, bright_agg


def _fit_normalization_constants(max_train_cases=80, seed=0):
    rng = np.random.default_rng(seed)
    train_ids = _iter_train_ids(TRAIN_DIR)
    if len(train_ids) == 0:
        return None

    if len(train_ids) > max_train_cases:
        sel = rng.choice(np.array(train_ids), size=max_train_cases, replace=False)
        train_ids = sorted(sel.tolist())

    _ensure_train_png_cache_exists(train_ids, limit=None)

    feats = []
    for cid in train_ids:
        st = _load_train_case_stack(cid)
        if st is None:
            continue
        try:
            feats.append(_compute_feature_triplet_from_stack(st))
        except Exception:
            continue

    if len(feats) < 10:
        return None

    F = np.asarray(feats, dtype=np.float32)
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (mu.astype(np.float32), sd.astype(np.float32))


NORM = _fit_normalization_constants(max_train_cases=80, seed=0)




## === cell 10
def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic_calibrator(max_train_cases=220, seed=0, iters=250, lr=0.15, l2=1e-2):
    labels = pd.read_csv(LABELS_PATH)
    labels["BraTS21ID"] = labels["BraTS21ID"].astype(str).str.zfill(5)

    train_ids = _iter_train_ids(TRAIN_DIR)
    train_ids = [cid for cid in train_ids if cid in set(labels["BraTS21ID"].values)]
    if len(train_ids) == 0:
        return None

    rng = np.random.default_rng(seed)
    if len(train_ids) > max_train_cases:
        sel = rng.choice(np.array(train_ids), size=max_train_cases, replace=False)
        train_ids = sorted(sel.tolist())

    _ensure_train_png_cache_exists(train_ids, limit=None)

    X_list, y_list = [], []
    y_map = dict(zip(labels["BraTS21ID"].values, labels["MGMT_value"].values))

    for cid in train_ids:
        st = _load_train_case_stack(cid)
        if st is None:
            continue
        try:
            med_agg, con_agg, bright_agg = _compute_feature_triplet_from_stack(st)
            if NORM is not None:
                mu, sd = NORM
                med_agg = float((med_agg - mu[0]) / sd[0])
                con_agg = float((con_agg - mu[1]) / sd[1])
                bright_agg = float((bright_agg - mu[2]) / sd[2])
            X_list.append([1.0, med_agg, con_agg, bright_agg])  # bias + 3 features
            y_list.append(float(y_map[cid]))
        except Exception:
            continue

    if len(X_list) < 40:
        return None

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)

    w = np.zeros(X.shape[1], dtype=np.float64)
    for _ in range(int(iters)):
        p = _sigmoid(X @ w)
        grad = (X.T @ (p - y)) / X.shape[0]
        grad[1:] += l2 * w[1:]  # no L2 on bias
        w -= lr * grad

    return w.astype(np.float32)


CAL_W = _fit_logistic_calibrator(
    max_train_cases=220, seed=0, iters=250, lr=0.15, l2=1e-2
)
print("Calibrator fitted:", CAL_W is not None)




## === cell 11
def predict_case_probability(case_id):
    stack = _load_case_stack(case_id)
    if stack is None:
        return 0.5

    med_agg, con_agg, bright_agg = _compute_feature_triplet_from_stack(stack)

    if NORM is not None:
        mu, sd = NORM
        med_agg = float((med_agg - mu[0]) / sd[0])
        con_agg = float((con_agg - mu[1]) / sd[1])
        bright_agg = float((bright_agg - mu[2]) / sd[2])

    if CAL_W is not None:
        score = float(
            CAL_W[0] + CAL_W[1] * med_agg + CAL_W[2] * con_agg + CAL_W[3] * bright_agg
        )
        prob = float(_sigmoid(score))
    else:
        score = float(med_agg + 0.55 * con_agg + 0.25 * bright_agg)
        prob = 1.0 / (1.0 + np.exp(-(score - 0.00) * 1.35))

    prob = float(np.clip(prob, 1e-4, 1.0 - 1e-4))
    return prob




## === cell 12
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)



## === cell 13
preds = []
missing_cases = 0

for case_id in sample_sub["BraTS21ID"].tolist():
    p = predict_case_probability(case_id)
    if p == 0.5:
        test_case_dir = os.path.join(TEST_DIR, case_id)
        if not os.path.isdir(test_case_dir):
            missing_cases += 1
    preds.append(p)



## === cell 14
final_submission = pd.DataFrame(
    {
        "BraTS21ID": sample_sub["BraTS21ID"].values,
        "MGMT_value": np.array(preds, dtype=np.float32),
    }
)

final_submission["BraTS21ID"] = final_submission["BraTS21ID"].astype(str).str.zfill(5)
final_submission["MGMT_value"] = final_submission["MGMT_value"].clip(0.0, 1.0)

print(final_submission.shape)
print(final_submission.head())
print("Missing case folders (should be 0):", missing_cases)
print(
    "Pred stats:",
    float(final_submission["MGMT_value"].min()),
    float(final_submission["MGMT_value"].mean()),
    float(final_submission["MGMT_value"].max()),
)



## === cell 15
final_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(final_submission))
