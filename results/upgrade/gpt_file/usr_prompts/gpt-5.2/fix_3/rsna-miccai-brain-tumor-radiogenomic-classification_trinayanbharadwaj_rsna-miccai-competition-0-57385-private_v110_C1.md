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

No external packages required in the script and installed.

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
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import pydicom as dicom  # type: ignore
except Exception:
    dicom = None

try:
    from skimage.transform import resize  # type: ignore
except Exception:
    resize = None

import struct


def _read_dicom_pixels_fallback(path):
    """
    Minimal DICOM reader for uncompressed, explicit VR little endian images.
    Returns float32 2D array or None on failure.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 132 or data[128:132] != b"DICM":
            return None

        def u16(off):
            return struct.unpack_from("<H", data, off)[0]

        def u32(off):
            return struct.unpack_from("<I", data, off)[0]

        off = 132
        rows = cols = None
        bits_alloc = None
        pixel_repr = 0
        samples_per_pixel = 1
        photometric = None
        pixel_data = None
        while off + 8 <= len(data):
            group = u16(off)
            elem = u16(off + 2)
            vr = data[off + 4 : off + 6]
            off_tag = off
            off += 6

            if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
                off += 2  # reserved
                if off + 4 > len(data):
                    break
                length = u32(off)
                off += 4
            else:
                if off + 2 > len(data):
                    break
                length = u16(off)
                off += 2

            value = data[off : off + length] if off + length <= len(data) else None

            if (group, elem) == (0x0028, 0x0010) and value is not None:  # Rows
                try:
                    rows = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                except Exception:
                    if length == 2:
                        rows = u16(off)
            elif (group, elem) == (0x0028, 0x0011) and value is not None:  # Columns
                try:
                    cols = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                except Exception:
                    if length == 2:
                        cols = u16(off)
            elif (group, elem) == (
                0x0028,
                0x0100,
            ) and value is not None:  # BitsAllocated
                try:
                    bits_alloc = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                except Exception:
                    if length == 2:
                        bits_alloc = u16(off)
            elif (group, elem) == (
                0x0028,
                0x0103,
            ) and value is not None:  # PixelRepresentation
                try:
                    pixel_repr = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                except Exception:
                    if length == 2:
                        pixel_repr = u16(off)
            elif (group, elem) == (
                0x0028,
                0x0002,
            ) and value is not None:  # SamplesPerPixel
                try:
                    samples_per_pixel = int(
                        value.rstrip(b"\x00 ").decode(errors="ignore")
                    )
                except Exception:
                    if length == 2:
                        samples_per_pixel = u16(off)
            elif (group, elem) == (
                0x0028,
                0x0004,
            ) and value is not None:  # PhotometricInterpretation
                photometric = value.rstrip(b"\x00 ").decode(errors="ignore")
            elif (group, elem) == (0x7FE0, 0x0010):  # PixelData
                if value is None:
                    break
                pixel_data = value
                break

            off += length
            if off <= off_tag:
                break

        if pixel_data is None or rows is None or cols is None or bits_alloc is None:
            return None
        if samples_per_pixel != 1:
            return None
        if bits_alloc not in (8, 16):
            return None

        if bits_alloc == 8:
            dt = np.int8 if pixel_repr == 1 else np.uint8
        else:
            dt = np.int16 if pixel_repr == 1 else np.uint16

        arr = np.frombuffer(pixel_data, dtype=dt)
        if arr.size < rows * cols:
            return None
        arr = arr[: rows * cols].reshape((rows, cols)).astype(np.float32)

        if photometric is not None and photometric.upper() == "MONOCHROME1":
            arr = np.max(arr) - arr

        return arr
    except Exception:
        return None




## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df.set_index("BraTS21ID")

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, "Test sample:", sample_df.shape)




## === cell 2
MRI_TYPE_INDEX_T2W = 3  # original code assumes mri_type[3] is T2w after sorted()
IMG_PX_SIZE = 150
SLICES_PER_CASE = 6

BAD_TRAIN_IDS = set(["00109", "00123", "00709"])  # as per competition note


def _resize2d(arr, out_hw):
    if resize is not None:
        r = resize(arr, out_hw, preserve_range=True, anti_aliasing=True).astype(
            np.float32
        )
        return r
    h, w = arr.shape
    oh, ow = out_hw
    y_idx = (np.linspace(0, h - 1, oh)).astype(np.int32)
    x_idx = (np.linspace(0, w - 1, ow)).astype(np.int32)
    return arr[np.ix_(y_idx, x_idx)].astype(np.float32)


def _safe_read_dcm(path):
    if dicom is not None:
        try:
            d = dicom.dcmread(path)
            arr = d.pixel_array.astype(np.float32)
            return arr
        except Exception:
            pass
    return _read_dicom_pixels_fallback(path)


def load_case_slices(
    case_dir,
    mri_type_index=MRI_TYPE_INDEX_T2W,
    img_px_size=IMG_PX_SIZE,
    slices_per_case=SLICES_PER_CASE,
):
    """
    Returns: np.ndarray shape (slices_per_case, img_px_size, img_px_size) normalized to [0,1]
    """
    mri_types = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    if len(mri_types) <= mri_type_index:
        t2_candidates = [p for p in mri_types if os.path.basename(p).lower() == "t2w"]
        if t2_candidates:
            mri_path = t2_candidates[0]
        else:
            return None
    else:
        mri_path = mri_types[mri_type_index]

    dcm_paths = sorted(
        [
            f.path
            for f in os.scandir(mri_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    selected = []
    for p in dcm_paths:
        arr = _safe_read_dcm(p)
        if arr is None:
            continue
        if float(np.nansum(arr)) > 100000:  # original heuristic
            r = _resize2d(arr, (img_px_size, img_px_size))
            mx = float(np.nanmax(r)) if np.isfinite(r).any() else 0.0
            if mx > 0:
                r = r / mx
            if float(np.nansum(r)) > 2000:  # original heuristic
                selected.append(r.astype(np.float32))
                if len(selected) >= slices_per_case:
                    break

    if len(selected) == 0:
        selected = [np.zeros((img_px_size, img_px_size), dtype=np.float32)]

    while len(selected) < slices_per_case:
        selected.append(selected[-1].copy())

    return np.stack(selected[:slices_per_case], axis=0)


def extract_slice_features(slice2d):
    """
    Very lightweight features from a normalized 2D slice.
    Returns 1D float32 feature vector.
    """
    x = slice2d.astype(np.float32)
    mean = float(np.mean(x))
    std = float(np.std(x))
    q10, q50, q90 = np.percentile(x, [10, 50, 90]).astype(np.float32)
    mx = float(np.max(x))
    mn = float(np.min(x))

    h, w = x.shape
    ch0, ch1 = int(0.25 * h), int(0.75 * h)
    cw0, cw1 = int(0.25 * w), int(0.75 * w)
    center = x[ch0:ch1, cw0:cw1]
    border_mask = np.ones_like(x, dtype=bool)
    border_mask[ch0:ch1, cw0:cw1] = False
    border = x[border_mask]
    cmean = float(np.mean(center))
    bmean = float(np.mean(border))
    cstd = float(np.std(center))
    bstd = float(np.std(border))

    hi = float(np.mean(x > 0.8))
    mid = float(np.mean(x > 0.5))

    feats = np.array(
        [mean, std, q10, q50, q90, mn, mx, cmean, bmean, cstd, bstd, hi, mid],
        dtype=np.float32,
    )
    return feats


def extract_subject_features(vol_slices):
    """
    vol_slices: (SLICES_PER_CASE, H, W)
    Subject-level feature: mean of per-slice features.
    """
    feats = np.stack(
        [extract_slice_features(vol_slices[i]) for i in range(vol_slices.shape[0])],
        axis=0,
    )
    return feats.mean(axis=0).astype(np.float32)


def load_subject_features(subject_ids, base_dir, labels_indexed=None):
    X_list, y_list, kept_ids = [], [], []
    for sid in subject_ids:
        case_dir = os.path.join(base_dir, sid)
        if not os.path.isdir(case_dir):
            continue
        vol = load_case_slices(case_dir)
        if vol is None:
            continue
        X_list.append(extract_subject_features(vol))
        kept_ids.append(sid)
        if labels_indexed is not None:
            y_list.append(float(labels_indexed.loc[sid, "MGMT_value"]))
    X = np.stack(X_list, axis=0) if len(X_list) else np.empty((0, 13), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if labels_indexed is not None else None
    return X, y, kept_ids




## === cell 3
all_train_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
all_train_ids = [sid for sid in all_train_ids if sid not in BAD_TRAIN_IDS]
all_train_ids = [sid for sid in all_train_ids if sid in labels_df.index]

MAX_TRAIN_SUBJECTS = 240  # pragmatic cap for runtime
if len(all_train_ids) > MAX_TRAIN_SUBJECTS:
    rng = np.random.default_rng(SEED)
    all_train_ids = sorted(
        rng.choice(all_train_ids, size=MAX_TRAIN_SUBJECTS, replace=False).tolist()
    )

rng = np.random.default_rng(SEED)
rng.shuffle(all_train_ids)

split = int(0.85 * len(all_train_ids))
train_ids = all_train_ids[:split]
val_ids = all_train_ids[split:]

print("Subjects train/val:", len(train_ids), len(val_ids))

X_train, y_train, kept_train_ids = load_subject_features(
    train_ids, TRAIN_DIR, labels_df
)
X_val, y_val, kept_val_ids = load_subject_features(val_ids, TRAIN_DIR, labels_df)

print("Loaded X_train:", X_train.shape, "y_train:", y_train.shape)
print("Loaded X_val:", X_val.shape, "y_val:", y_val.shape)




## === cell 4
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize_apply(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32)


def train_logreg_l2(X, y, lr=0.1, steps=800, l2=1.0):
    """
    X: (N, D) float32
    y: (N,) float32 in {0,1}
    """
    N, D = X.shape
    w = np.zeros((D,), dtype=np.float32)
    b = np.float32(0.0)

    for _ in range(steps):
        z = X @ w + b
        p = sigmoid(z).astype(np.float32)
        grad_w = (X.T @ (p - y)) / N + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w.astype(np.float32)
        b = np.float32(b - lr * grad_b)
    return w, b


def roc_auc_score_np(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)
    rank_sum_pos = np.sum(ranks[y_true_sorted == 1])
    auc = (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


mu, sigma = standardize_fit(X_train)
Xtr = standardize_apply(X_train, mu, sigma)
Xva = standardize_apply(X_val, mu, sigma)

w, b = train_logreg_l2(Xtr, y_train, lr=0.1, steps=800, l2=1.0)

val_pred = sigmoid(Xva @ w + b)
val_auc = roc_auc_score_np(y_val, val_pred)
print("Validation subject-level AUC:", val_auc)




## === cell 5
X_full, y_full, kept_full_ids = load_subject_features(
    all_train_ids, TRAIN_DIR, labels_df
)
mu_f, sigma_f = standardize_fit(X_full)
X_full_s = standardize_apply(X_full, mu_f, sigma_f)
w_f, b_f = train_logreg_l2(X_full_s, y_full, lr=0.1, steps=900, l2=1.0)

train_prior = float(np.mean(y_full)) if y_full is not None and len(y_full) else 0.5
print("Train prior:", train_prior, "Full train N:", X_full.shape[0])




## === cell 6
test_ids = sample_df["BraTS21ID"].tolist()
X_test, _, kept_test_ids = load_subject_features(
    test_ids, TEST_DIR, labels_indexed=None
)

test_pred_map = {}
if X_test.shape[0] > 0:
    X_test_s = standardize_apply(X_test, mu_f, sigma_f)
    preds = sigmoid(X_test_s @ w_f + b_f).astype(np.float32)
    for sid, p in zip(kept_test_ids, preds):
        test_pred_map[sid] = float(p)

sub = sample_df.copy()
sub["MGMT_value"] = (
    sub["BraTS21ID"].map(test_pred_map).fillna(train_prior).astype(np.float32)
)
sub["MGMT_value"] = sub["MGMT_value"].clip(0.0, 1.0)

print(sub.head())
print("Submission shape:", sub.shape)




## === cell 7
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.describe(include="all"))
