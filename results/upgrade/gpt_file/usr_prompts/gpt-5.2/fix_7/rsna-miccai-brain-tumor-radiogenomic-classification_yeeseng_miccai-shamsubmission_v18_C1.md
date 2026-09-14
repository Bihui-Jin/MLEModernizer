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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import io
import struct
from functools import lru_cache



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

train_labels_path = os.path.join(DATA_ROOT, "train_labels.csv")
if not os.path.exists(train_labels_path):
    train_labels_path = "/kaggle/input/train_labels.csv"

train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")

sample_df = pd.read_csv(sample_path)
assert (
    "BraTS21ID" in sample_df.columns and "MGMT_value" in sample_df.columns
), "Submission columns not found."
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

train_labels = pd.read_csv(train_labels_path)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
assert set(["BraTS21ID", "MGMT_value"]).issubset(train_labels.columns)

print("Loaded sample_submission:", sample_df.shape, "train_labels:", train_labels.shape)
print(
    "Train dir exists:",
    os.path.isdir(train_dir),
    "Test dir exists:",
    os.path.isdir(test_dir),
)



## === cell 2


def _read_dicom_pixel_array(path):
    """
    Minimal DICOM pixel reader for common RSNA-MICCAI BraTS DICOMs.
    Supports uncompressed little-endian syntaxes; returns float32 array or raises.

    Keeps prior correctness: applies RescaleSlope/RescaleIntercept when present.
    """

    def _read_exact(f, n):
        b = f.read(n)
        if len(b) != n:
            raise ValueError("Unexpected EOF.")
        return b

    def read_u16(f):
        return struct.unpack("<H", _read_exact(f, 2))[0]

    def read_u32(f):
        return struct.unpack("<I", _read_exact(f, 4))[0]

    def _parse_number(val_bytes):
        s = val_bytes.replace(b"\x00", b"").decode("ascii", errors="ignore").strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    with open(path, "rb") as f:
        pre = _read_exact(f, 132)
        if pre[128:132] != b"DICM":
            raise ValueError("Not a standard DICOM file with DICM marker.")

        transfer_syntax = None
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                raise ValueError("Unexpected EOF in meta header.")
            group, elem = struct.unpack("<HH", hdr)
            if group != 0x0002:
                f.seek(-4, io.SEEK_CUR)
                break

            vr = _read_exact(f, 2).decode("ascii", errors="ignore")
            if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                _read_exact(f, 2)  # reserved
                length = read_u32(f)
            else:
                length = read_u16(f)

            value = _read_exact(f, length)
            if (group, elem) == (0x0002, 0x0010):
                transfer_syntax = value.rstrip(b"\x00").decode("ascii", errors="ignore")

        if transfer_syntax is None:
            transfer_syntax = "1.2.840.10008.1.2"  # Implicit VR Little Endian

        if transfer_syntax not in (
            "1.2.840.10008.1.2",  # Implicit VR Little Endian
            "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
        ):
            raise ValueError(f"Unsupported TransferSyntaxUID: {transfer_syntax}")

        explicit = transfer_syntax == "1.2.840.10008.1.2.1"

        rows = None
        cols = None
        bits_alloc = None
        pixel_repr = 0
        samples_per_pixel = 1
        rescale_slope = 1.0
        rescale_intercept = 0.0

        while True:
            tag_bytes = f.read(4)
            if len(tag_bytes) < 4:
                break
            group, elem = struct.unpack("<HH", tag_bytes)

            if explicit:
                vr = _read_exact(f, 2).decode("ascii", errors="ignore")
                if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                    _read_exact(f, 2)  # reserved
                    length = read_u32(f)
                else:
                    length = read_u16(f)
            else:
                length = read_u32(f)
                vr = None  # not used

            tag = (group, elem)

            if tag == (0x0028, 0x0010):  # Rows
                v = _read_exact(f, length)
                rows = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0011):  # Columns
                v = _read_exact(f, length)
                cols = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0100):  # BitsAllocated
                v = _read_exact(f, length)
                bits_alloc = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0103):  # PixelRepresentation
                v = _read_exact(f, length)
                pixel_repr = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0002):  # SamplesPerPixel
                v = _read_exact(f, length)
                samples_per_pixel = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x1053):  # RescaleSlope
                v = _read_exact(f, length)
                num = _parse_number(v)
                if num is not None and np.isfinite(num) and num != 0:
                    rescale_slope = float(num)
            elif tag == (0x0028, 0x1052):  # RescaleIntercept
                v = _read_exact(f, length)
                num = _parse_number(v)
                if num is not None and np.isfinite(num):
                    rescale_intercept = float(num)
            elif tag == (0x7FE0, 0x0010):  # PixelData
                if rows is None or cols is None or bits_alloc is None:
                    raise ValueError(
                        "Missing Rows/Columns/BitsAllocated before PixelData."
                    )
                if samples_per_pixel != 1:
                    raise ValueError("Only SamplesPerPixel=1 supported.")

                pixel_bytes = _read_exact(f, length)

                if bits_alloc == 16:
                    dtype = np.int16 if pixel_repr == 1 else np.uint16
                elif bits_alloc == 8:
                    dtype = np.int8 if pixel_repr == 1 else np.uint8
                else:
                    raise ValueError(f"Unsupported BitsAllocated: {bits_alloc}")

                arr = np.frombuffer(pixel_bytes, dtype=dtype)
                expected = rows * cols
                if arr.size < expected:
                    raise ValueError("PixelData too short.")
                arr = arr[:expected].reshape(rows, cols).astype(np.float32)

                arr = arr * np.float32(rescale_slope) + np.float32(rescale_intercept)
                return arr
            else:
                if length:
                    f.seek(length, io.SEEK_CUR)

        raise ValueError("PixelData not found.")


@lru_cache(maxsize=200000)
def _try_read_instance_number(path):
    """
    Same semantics: return InstanceNumber (0020,0013) if found, else None.
    Speed fix: scan only a limited initial window; do not read entire file.
    """
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
            if len(pre) < 132 or pre[128:132] != b"DICM":
                return None

            def read_u16_from(buf, off):
                return int.from_bytes(buf[off : off + 2], "little", signed=False)

            def read_u32_from(buf, off):
                return int.from_bytes(buf[off : off + 4], "little", signed=False)

            window = f.read(256 * 1024)
            data = pre + window

        pos = 132
        transfer_syntax = None

        while pos + 8 <= len(data):
            group = read_u16_from(data, pos)
            elem = read_u16_from(data, pos + 2)
            if group != 0x0002:
                break
            vr = data[pos + 4 : pos + 6].decode("ascii", errors="ignore")
            if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                if pos + 12 > len(data):
                    return None
                length = read_u32_from(data, pos + 8)
                value_pos = pos + 12
            else:
                if pos + 8 > len(data):
                    return None
                length = read_u16_from(data, pos + 6)
                value_pos = pos + 8

            next_pos = value_pos + length
            if next_pos > len(data):
                return None

            if (group, elem) == (0x0002, 0x0010):
                transfer_syntax = (
                    data[value_pos : value_pos + length]
                    .rstrip(b"\x00")
                    .decode("ascii", errors="ignore")
                )
            pos = next_pos

        if transfer_syntax is None:
            transfer_syntax = "1.2.840.10008.1.2"
        if transfer_syntax not in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1"):
            return None
        explicit = transfer_syntax == "1.2.840.10008.1.2.1"

        while pos + 8 <= len(data):
            group = read_u16_from(data, pos)
            elem = read_u16_from(data, pos + 2)

            if explicit:
                if pos + 8 > len(data):
                    return None
                vr = data[pos + 4 : pos + 6].decode("ascii", errors="ignore")
                if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                    if pos + 12 > len(data):
                        return None
                    length = read_u32_from(data, pos + 8)
                    value_pos = pos + 12
                else:
                    length = read_u16_from(data, pos + 6)
                    value_pos = pos + 8
            else:
                length = read_u32_from(data, pos + 4)
                value_pos = pos + 8

            next_pos = value_pos + length
            if next_pos > len(data):
                return None

            if (group, elem) == (0x0020, 0x0013):  # InstanceNumber
                raw = data[value_pos : value_pos + length].replace(b"\x00", b"").strip()
                s = raw.decode("ascii", errors="ignore").strip()
                try:
                    return int(float(s))
                except Exception:
                    return None

            if (group, elem) == (0x7FE0, 0x0010):  # PixelData -> stop
                return None

            if next_pos <= pos:
                return None
            pos = next_pos

        return None
    except Exception:
        return None


def _list_dcm_files(series_dir):
    try:
        return [
            os.path.join(series_dir, n)
            for n in os.listdir(series_dir)
            if n.endswith(".dcm")
        ]
    except FileNotFoundError:
        return []


def _get_middle_slice_dcm(series_dir):
    files = _list_dcm_files(series_dir)
    if not files:
        return None

    inst = [(_try_read_instance_number(p), p) for p in files]
    if any(v is not None for v, _ in inst):
        files_sorted = [
            p
            for v, p in sorted(
                inst, key=lambda t: (t[0] if t[0] is not None else 10**9, t[1])
            )
        ]
    else:
        files_sorted = sorted(files)

    return files_sorted[len(files_sorted) // 2]


def subject_feature_mean_intensity(subject_dir, modality="FLAIR"):
    series_dir = os.path.join(subject_dir, modality)
    dcm_path = _get_middle_slice_dcm(series_dir)
    if dcm_path is None:
        return np.nan
    img = _read_dicom_pixel_array(dcm_path)

    lo, hi = np.percentile(img, [1.0, 99.0])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        img = np.clip(img, lo, hi)

    return float(np.mean(img))


def subject_feature_mean_intensity_multi(
    subject_dir, modalities=("FLAIR", "T1w", "T1wCE", "T2w")
):
    vals = []
    for m in modalities:
        try:
            v = subject_feature_mean_intensity(subject_dir, modality=m)
        except Exception:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))




## === cell 3
bad_ids = set(["00109", "00123", "00709"])


def _list_subject_ids(folder):
    if not os.path.isdir(folder):
        return []
    ids = []
    with os.scandir(folder) as it:
        for e in it:
            if e.is_dir():
                ids.append(str(e.name).zfill(5))
    return sorted(ids)


train_ids = _list_subject_ids(train_dir)
test_ids = _list_subject_ids(test_dir)

train_ids = [i for i in train_ids if i not in bad_ids]
train_labels_use = train_labels[train_labels["BraTS21ID"].isin(train_ids)].copy()

train_ids = train_labels_use["BraTS21ID"].tolist()

print(
    "Train subjects (after excluding bad):",
    len(train_ids),
    "Test subjects:",
    len(test_ids),
)

X_train = np.empty((len(train_ids), 1), dtype=np.float32)
y_train = train_labels_use["MGMT_value"].astype(int).to_numpy()

for idx, sid in enumerate(train_ids):
    subj_dir = os.path.join(train_dir, sid)
    try:
        X_train[idx, 0] = subject_feature_mean_intensity_multi(subj_dir)
    except Exception:
        X_train[idx, 0] = np.nan

X_test = np.empty((len(test_ids), 1), dtype=np.float32)
for idx, sid in enumerate(test_ids):
    subj_dir = os.path.join(test_dir, sid)
    try:
        X_test[idx, 0] = subject_feature_mean_intensity_multi(subj_dir)
    except Exception:
        X_test[idx, 0] = np.nan

med = np.nanmedian(X_train[:, 0])
if not np.isfinite(med):
    med = 0.0
X_train[:, 0] = np.where(np.isfinite(X_train[:, 0]), X_train[:, 0], med)
X_test[:, 0] = np.where(np.isfinite(X_test[:, 0]), X_test[:, 0], med)

print("Feature stats - train mean/std:", float(X_train.mean()), float(X_train.std()))
print("Feature stats - test  mean/std:", float(X_test.mean()), float(X_test.std()))



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="liblinear",
                max_iter=500,
                random_state=0,
                class_weight="balanced",
            ),
        ),
    ]
)

clf.fit(X_train, y_train)

test_proba = clf.predict_proba(X_test)[:, 1]
test_proba = np.clip(test_proba, 1e-6, 1 - 1e-6)

print(
    "Pred proba summary:",
    float(test_proba.min()),
    float(test_proba.mean()),
    float(test_proba.max()),
)



## === cell 5
pred_map = {sid: float(p) for sid, p in zip(test_ids, test_proba)}

sub = sample_df.copy()
sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).astype(float)
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 6
assert sub.columns.tolist() == ["BraTS21ID", "MGMT_value"]
assert len(sub) == len(sample_df)
assert sub["MGMT_value"].between(0.0, 1.0).all()

print("Unique IDs in submission:", sub["BraTS21ID"].nunique())
print(
    "Any missing preds filled with 0.5:",
    int(sub["BraTS21ID"].isin(pred_map.keys()).sum()),
    "/",
    len(sub),
)
