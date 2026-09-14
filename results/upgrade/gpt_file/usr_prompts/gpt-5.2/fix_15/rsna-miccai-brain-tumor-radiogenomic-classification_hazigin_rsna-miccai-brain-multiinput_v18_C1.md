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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import math
import struct
import zlib

print("Using pure-Python DICOM pixel decoder (no TensorFlow / no pydicom).")



## === cell 1
df_preds = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)

weightdatapath = Path("../input/weight-multi-20210919")

testdatapaht = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
if not os.path.isdir(testdatapaht):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
    if os.path.isdir(alt):
        testdatapaht = alt

print("Using test data path:", testdatapaht)



## === cell 2
height = 256
width = 256
channel = 3
batch_size = 32
epochs = 400
seed = 26
epoch = "0921"
views = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 3
def set_seed(seed=200):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)




## === cell 4
def _read_tag(fp):
    b = fp.read(4)
    if len(b) != 4:
        return None
    group, elem = struct.unpack("<HH", b)
    return group, elem


def _read_vr_len_explicit(fp):
    vr = fp.read(2)
    if len(vr) != 2:
        return None, None
    vr = vr.decode("ascii", errors="ignore")
    if vr in ("OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"):
        fp.read(2)  # reserved
        lbytes = fp.read(4)
        if len(lbytes) != 4:
            return vr, None
        length = struct.unpack("<I", lbytes)[0]
    else:
        lbytes = fp.read(2)
        if len(lbytes) != 2:
            return vr, None
        length = struct.unpack("<H", lbytes)[0]
    return vr, length


def _skip(fp, n):
    if n is None or n < 0:
        return
    fp.seek(n, 1)


def _dicom_read_uncompressed_pixel_array(path):
    with open(path, "rb") as fp:
        pre = fp.read(132)
        if len(pre) < 132 or pre[128:132] != b"DICM":
            raise ValueError("Not a DICOM file with DICM preamble")

        rows = cols = None
        bits_alloc = None
        pixel_repr = 0
        samples_per_pixel = 1
        photometric = None
        pixel_data = None

        while True:
            tag = _read_tag(fp)
            if tag is None:
                break
            group, elem = tag

            vr, length = _read_vr_len_explicit(fp)
            if vr is None or length is None:
                break

            if length == 0xFFFFFFFF:
                raise ValueError("Undefined length not supported")

            if (group, elem) == (0x0028, 0x0010):  # Rows
                v = fp.read(length)
                rows = (
                    int.from_bytes(v[:2], "little", signed=False)
                    if length >= 2
                    else None
                )
            elif (group, elem) == (0x0028, 0x0011):  # Columns
                v = fp.read(length)
                cols = (
                    int.from_bytes(v[:2], "little", signed=False)
                    if length >= 2
                    else None
                )
            elif (group, elem) == (0x0028, 0x0100):  # BitsAllocated
                v = fp.read(length)
                bits_alloc = (
                    int.from_bytes(v[:2], "little", signed=False)
                    if length >= 2
                    else None
                )
            elif (group, elem) == (0x0028, 0x0103):  # PixelRepresentation
                v = fp.read(length)
                pixel_repr = (
                    int.from_bytes(v[:2], "little", signed=False) if length >= 2 else 0
                )
            elif (group, elem) == (0x0028, 0x0002):  # SamplesPerPixel
                v = fp.read(length)
                samples_per_pixel = (
                    int.from_bytes(v[:2], "little", signed=False) if length >= 2 else 1
                )
            elif (group, elem) == (0x0028, 0x0004):  # PhotometricInterpretation
                v = fp.read(length)
                photometric = v.decode("ascii", errors="ignore").strip("\x00 ").upper()
            elif (group, elem) == (0x7FE0, 0x0010):  # PixelData
                pixel_data = fp.read(length)
                break
            else:
                _skip(fp, length)

        if rows is None or cols is None or bits_alloc is None or pixel_data is None:
            raise ValueError("Missing required DICOM tags")

        if samples_per_pixel != 1:
            raise ValueError("Only SamplesPerPixel=1 supported")

        if bits_alloc == 16:
            dtype = np.int16 if pixel_repr == 1 else np.uint16
        elif bits_alloc == 8:
            dtype = np.int8 if pixel_repr == 1 else np.uint8
        else:
            raise ValueError(f"Unsupported BitsAllocated={bits_alloc}")

        arr = np.frombuffer(pixel_data, dtype=dtype)
        if arr.size < rows * cols:
            raise ValueError("PixelData too short")
        arr = arr[: rows * cols].reshape(rows, cols)

        x = arr.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mn = float(x.min())
        mx = float(x.max())
        if mx > mn:
            x = (x - mn) * (255.0 / (mx - mn))
        else:
            x = np.zeros_like(x, dtype=np.float32)
        return x


def _area_resize_2x_downsample_np(img2d: np.ndarray) -> np.ndarray:
    if img2d.shape == (512, 512) and (height, width) == (256, 256):
        x = img2d.reshape(256, 2, 256, 2).mean(axis=(1, 3), dtype=np.float32)
        return x.astype(np.float32, copy=False)

    in_h, in_w = img2d.shape
    if in_h % height == 0 and in_w % width == 0:
        sh = in_h // height
        sw = in_w // width
        x = img2d.reshape(height, sh, width, sw).mean(axis=(1, 3), dtype=np.float32)
        return x.astype(np.float32, copy=False)

    ys = (np.linspace(0, in_h - 1, height)).astype(np.int32)
    xs = (np.linspace(0, in_w - 1, width)).astype(np.int32)
    return img2d[np.ix_(ys, xs)].astype(np.float32, copy=False)




## === cell 5
_DCM_LIST_CACHE = {}
_DCM_IMG_CACHE = {}  # cache decoded+resized slices within a run (path -> image)


def _list_dcm_files(idx: str, view: str):
    key = (idx, view)
    hit = _DCM_LIST_CACHE.get(key)
    if hit is not None:
        return hit

    base_dir = os.path.join(testdatapaht, idx, view)
    if not os.path.isdir(base_dir):
        files = []
    else:
        files = []
        with os.scandir(base_dir) as it:
            for e in it:
                if e.is_file():
                    n = e.name
                    if n.lower().endswith(".dcm"):
                        files.append(e.path)
        files.sort()
    _DCM_LIST_CACHE[key] = files
    return files


def _decode_resize_dicom_numpy(path: str) -> np.ndarray:
    cached = _DCM_IMG_CACHE.get(path)
    if cached is not None:
        return cached

    try:
        img2d = _dicom_read_uncompressed_pixel_array(path)
        out = _area_resize_2x_downsample_np(img2d)
    except Exception:
        out = np.zeros((height, width), dtype=np.float32)

    _DCM_IMG_CACHE[path] = out
    return out


def load_imgs(idx, view, ignore_zeros=True):
    files = _list_dcm_files(idx, view)
    if len(files) == 0:
        return np.zeros((1, height, width), dtype=np.float32)

    imgs = []
    dec = _decode_resize_dicom_numpy
    for p in files:
        im = dec(p)
        if ignore_zeros and (im.max() == 0.0):
            continue
        imgs.append(im)

    if len(imgs) == 0:
        return np.zeros((1, height, width), dtype=np.float32)

    return np.stack(imgs, axis=0).astype(np.float32, copy=False)




## === cell 6
dim = (height, width)
t_imgs = np.empty((channel, *dim), dtype=np.float32)


def data_generation(ID, view, is_Train=True):
    idx = str(ID).zfill(5)
    files = _list_dcm_files(idx, view)
    if len(files) == 0:
        t_imgs.fill(0.0)
        return t_imgs.transpose(1, 2, 0)

    n = len(files)
    t_a = math.ceil(n / 3)

    ranges = ((0, t_a), (t_a, n - t_a), (n - t_a, n))
    weights = (0.3, 0.4, 0.3)

    dec = _decode_resize_dicom_numpy

    for ch, ((lo, hi), w) in enumerate(zip(ranges, weights)):
        if hi <= lo:  # middle range can be empty for small n; match original fallback
            lo, hi = 0, n
        cnt = hi - lo
        if cnt <= 0:
            t_imgs[ch].fill(0.0)
            continue

        acc = None
        for p in files[lo:hi]:
            im = dec(p)
            if acc is None:
                acc = im.astype(np.float32, copy=True)
            else:
                acc += im
        acc *= w / float(cnt)
        t_imgs[ch] = acc

    img_ = t_imgs.transpose(1, 2, 0)
    return img_




## === cell 7
_CRC32C_TABLE = None


def _init_crc32c_table():
    global _CRC32C_TABLE
    if _CRC32C_TABLE is not None:
        return
    poly = 0x82F63B78
    table = np.empty(256, dtype=np.uint32)
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        table[i] = crc
    _CRC32C_TABLE = table


def _crc32c(data: bytes) -> int:
    _init_crc32c_table()
    crc = 0xFFFFFFFF
    for b in data:
        crc = (_CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)) & 0xFFFFFFFF
    return (~crc) & 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    x = _crc32c(data)
    return (((x >> 15) | (x << 17)) & 0xFFFFFFFF) + 0xA282EAD8 & 0xFFFFFFFF


def _write_tfrecord(path, records_iter):
    with open(path, "wb") as f:
        for rec in records_iter:
            if not isinstance(rec, (bytes, bytearray)):
                rec = bytes(rec)
            ln = struct.pack("<Q", len(rec))
            f.write(ln)
            f.write(struct.pack("<I", _masked_crc32c(ln)))
            f.write(rec)
            f.write(struct.pack("<I", _masked_crc32c(rec)))


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _len_delim(b: bytes) -> bytes:
    return _varint(len(b)) + b


def _field(tag: int, wire_type: int) -> bytes:
    return _varint((tag << 3) | wire_type)


def _make_example_with_image_bytes(img_bytes: bytes) -> bytes:
    value = _field(1, 2) + _len_delim(img_bytes)  # value field (1), len-delimited
    bytes_list = _field(1, 2) + _len_delim(value)  # bytes_list field (1) inside Feature

    feature = _field(1, 2) + _len_delim(bytes_list)  # Feature.bytes_list

    key = _field(1, 2) + _len_delim(b"image")
    val = _field(2, 2) + _len_delim(feature)
    entry = key + val
    features = _field(1, 2) + _len_delim(entry)

    example = _field(1, 2) + _len_delim(features)
    return example




## === cell 8
def serialize_example_test(feature0):
    feature0 = np.asarray(feature0, dtype=np.float32, order="C")
    return _make_example_with_image_bytes(feature0.tobytes())




## === cell 9
def _build_one_view_tfrec(view_name, ids, out_path):
    def _iter_records():
        for x in tqdm(ids, desc=f"TFREC {view_name}", leave=False):
            img = data_generation(x, view_name, False)
            yield serialize_example_test(img)

    _write_tfrecord(out_path, _iter_records())




## === cell 10
ids = df_preds["BraTS21ID"].tolist()




## === cell 11
def _import_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e:
        print("ERROR: TensorFlow import failed:", repr(e))
        return None


tf = _import_tensorflow()
if tf is None:
    print(
        "WARNING: Writing fallback submission (all 0.5) due to TensorFlow import failure."
    )
    df_preds["MGMT_value"] = 0.5
    subfilename = "submission.csv"
    df_preds.to_csv(subfilename, index=False)
    print(
        "Wrote:", subfilename, "rows:", len(df_preds), "cols:", list(df_preds.columns)
    )
    print(df_preds.head())
    raise SystemExit(0)

print("TF:", tf.__version__)

try:
    tf.random.set_seed(seed)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 12
@tf.keras.utils.register_keras_serializable(package="Custom")
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


@tf.keras.utils.register_keras_serializable(package="Custom")
def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


tf.keras.utils.get_custom_objects()["mish"] = mish
tf.keras.utils.get_custom_objects()["siren"] = siren
try:
    tf.keras.activations.mish = mish
except Exception:
    pass
try:
    tf.keras.activations.siren = siren
except Exception:
    pass
try:
    tf.keras.activations.get({"class_name": "siren", "config": {}})
except Exception:
    pass




## === cell 13
class Silu(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Silu, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.num_outputs],
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, input):
        return tf.nn.silu(input)




## === cell 14
def deserialize_example(serialized_string):
    image_feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed_record = tf.io.parse_single_example(
        serialized_string, image_feature_description
    )
    image = tf.io.decode_raw(parsed_record["image"], tf.float32)
    image = tf.reshape(image, [height, width, channel])
    return image




## === cell 15
def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 16
class RegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.rule1 = Silu(0)
        self.rule1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max1 = tf.keras.layers.MaxPooling2D(5)

        self.conv2 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.rule2 = Silu(0)
        self.max2 = tf.keras.layers.MaxPooling2D(5)

        self.conv3 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.rule3 = Silu(0)
        self.max3 = tf.keras.layers.MaxPooling2D(5)

        self.conv4 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.rule4 = Silu(0)
        self.max4 = tf.keras.layers.MaxPooling2D(5)

        para_relu = tf.keras.layers.LeakyReLU(alpha=0.5)
        self.dence256 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_2 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_2 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_2 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_3 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_3 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_3 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_4 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_4 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_4 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )

        self.dence32 = tf.keras.layers.Dense(
            32, kernel_initializer="he_normal", activation=siren
        )
        self.dence256_5 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_5 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_5 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation=para_relu
        )
        self.dropoup5 = tf.keras.layers.Dropout(0.2)
        self.dropoup4 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_2 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_3 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_4 = tf.keras.layers.Dropout(0.5)
        self.dropoup3 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_2 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_3 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_4 = tf.keras.layers.Dropout(0.2)
        self.dropoup2 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_2 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_3 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_4 = tf.keras.layers.Dropout(0.2)
        self.dropoup = tf.keras.layers.Dropout(0.1)
        self.dence1 = tf.keras.layers.Dense(1, activation="sigmoid")

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_tensor, training=True):
        x1 = input_tensor[0]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1, training=training)
        x1 = self.rule1(x1)
        x1 = self.max1(x1)
        x1 = self.dence256(x1)
        x1 = self.dropoup4(x1, training=training)
        x1 = self.dence128(x1)

        x2 = self.conv2(input_tensor[1])
        x2 = self.bn2(x2, training=training)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dence256_2(x2)
        x2 = self.dropoup4_2(x2, training=training)
        x2 = self.dence128_2(x2)

        x3 = self.conv3(input_tensor[2])
        x3 = self.bn3(x3, training=training)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dence256_3(x3)
        x3 = self.dropoup4_3(x3, training=training)
        x3 = self.dence128_3(x3)

        x4 = self.conv4(input_tensor[3])
        x4 = self.bn4(x4, training=training)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dence256_4(x4)
        x4 = self.dropoup4_4(x4, training=training)
        x4 = self.dence128_4(x4)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.dence256_5(x)
        x = self.dropoup3_4(x, training=training)
        x = self.dence128_5(x)
        x = self.dropoup5(x, training=training)
        x = self.flatten(x)
        x = self.dence64_5(x)
        x = self.dropoup(x, training=training)
        x = self.dence32(x)

        return self.dence1(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}




## === cell 17
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.SGD(
    learning_rate=1e-5, decay=1e-6, momentum=0.9, nesterov=True
)
f_pre = []

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True


def _make_view_tensor(ids_list, view_name):
    out = np.empty((len(ids_list), height, width, channel), dtype=np.float32)
    for i, x in enumerate(tqdm(ids_list, desc=f"Gen {view_name}", leave=False)):
        out[i] = data_generation(x, view_name, False)
    return out


view_arrays = {v: _make_view_tensor(ids, v) for v in views}

testset0 = tf.data.Dataset.from_tensor_slices(view_arrays["FLAIR"]).with_options(
    options
)
testset1 = tf.data.Dataset.from_tensor_slices(view_arrays["T1w"]).with_options(options)
testset2 = tf.data.Dataset.from_tensor_slices(view_arrays["T1wCE"]).with_options(
    options
)
testset3 = tf.data.Dataset.from_tensor_slices(view_arrays["T2w"]).with_options(options)

testset0 = testset0.map(
    argument_image_tw2_val, num_parallel_calls=AUTOTUNE, deterministic=True
)
testset1 = testset1.map(
    argument_image_tw2_val, num_parallel_calls=AUTOTUNE, deterministic=True
)
testset2 = testset2.map(
    argument_image_tw2_val, num_parallel_calls=AUTOTUNE, deterministic=True
)
testset3 = testset3.map(
    argument_image_tw2_val, num_parallel_calls=AUTOTUNE, deterministic=True
)

input_a = tf.keras.Input(shape=(height, width, channel), name="input_a")
input_b = tf.keras.Input(shape=(height, width, channel), name="input_b")
input_c = tf.keras.Input(shape=(height, width, channel), name="input_c")
input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")

model = RegressionModel()
model([input_a, input_b, input_c, input_d])

model.compile(
    optimizer=opt,
    loss=loss_func,
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="bce"),
    ],
)

test_ds = tf.data.Dataset.zip((testset0, testset1, testset2, testset3))
test_ds = test_ds.batch(10, drop_remainder=False).prefetch(AUTOTUNE)

loaded_any = False
for f in range(5):  # Fold
    t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
    w_path = Path(weightdatapath, t_weight)
    try:
        model.load_weights(w_path)
        loaded_any = True
    except Exception as e:
        print("WARNING: could not load weights:", str(w_path), "|", repr(e))
        continue

    preds = model.predict(test_ds, batch_size=batch_size, verbose=0)
    f_pre.append(preds)

if loaded_any and len(f_pre) > 0:
    pre = np.array(f_pre)  # (folds, N, 1)
    finpre = pre.mean(axis=0)
    finpre = np.asarray(finpre).reshape(-1)  # (N,)
else:
    print("WARNING: no weights loaded; using 0.5 predictions for all rows.")
    finpre = np.full((len(df_preds),), 0.5, dtype=np.float32)

if finpre.shape[0] != len(df_preds):
    print(
        "WARNING: prediction length mismatch:",
        finpre.shape[0],
        "vs",
        len(df_preds),
        "-> using 0.5 fallback.",
    )
    finpre = np.full((len(df_preds),), 0.5, dtype=np.float32)

df_preds["MGMT_value"] = finpre.astype(np.float32)

subfilename = "submission.csv"
df_preds.to_csv(subfilename, index=False)
print("Wrote:", subfilename, "rows:", len(df_preds), "cols:", list(df_preds.columns))
print(df_preds.head())
