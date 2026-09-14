# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import io
import glob
import struct
import zlib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
SAMPLE_CSV = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_TFRECORD_DIR = f"{DATA_ROOT}/train_tfrecords"
TEST_TFRECORD_DIR = f"{DATA_ROOT}/test_tfrecords"
SUBMISSION_PATH = "/kaggle/working/submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_CSV), f"Missing {SAMPLE_CSV}"
assert os.path.isdir(TRAIN_TFRECORD_DIR), f"Missing {TRAIN_TFRECORD_DIR}"
assert os.path.isdir(TEST_TFRECORD_DIR), f"Missing {TEST_TFRECORD_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_CSV)
if list(sample_df.columns) != ["image_id", "label"]:
    raise ValueError(
        f"Unexpected sample submission columns: {sample_df.columns.tolist()}"
    )

print(train_df.head())
print(sample_df.head())


## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

try:
    cpu_cnt = os.cpu_count() or 2
    torch.set_num_threads(max(1, min(8, cpu_cnt)))
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)


## === cell 2
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

num_classes = 5

img_size = 384
train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


## === cell 3
from sklearn.model_selection import StratifiedShuffleSplit


def _read_varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _skip_field(wire_type, buf, i):
    if wire_type == 0:  # varint
        _, i = _read_varint(buf, i)
        return i
    if wire_type == 1:  # 64-bit
        return i + 8
    if wire_type == 2:  # len-delimited
        l, i = _read_varint(buf, i)
        return i + l
    if wire_type == 5:  # 32-bit
        return i + 4
    raise ValueError(f"Unsupported wire type {wire_type}")


def _parse_example_name_only(serialized: bytes):
    buf = memoryview(serialized)
    n = len(buf)
    i = 0
    while i < n:
        tag, i = _read_varint(buf, i)
        field_num = tag >> 3
        wire_type = tag & 7
        if field_num != 1 or wire_type != 2:
            i = _skip_field(wire_type, buf, i)
            continue

        flen, i = _read_varint(buf, i)
        fend = i + flen

        j = i
        while j < fend:
            ftag, j = _read_varint(buf, j)
            fnum = ftag >> 3
            fwt = ftag & 7
            if fnum != 1 or fwt != 2:
                j = _skip_field(fwt, buf, j)
                continue

            ent_len, j = _read_varint(buf, j)
            ent_end = j + ent_len

            key = None
            k = j
            while k < ent_end:
                etag, k = _read_varint(buf, k)
                enum = etag >> 3
                ewt = etag & 7
                if enum == 1 and ewt == 2:  # key
                    key_len, k = _read_varint(buf, k)
                    key = bytes(buf[k : k + key_len])
                    k += key_len
                elif enum == 2 and ewt == 2:  # value (Feature message)
                    vlen, k = _read_varint(buf, k)
                    vend = k + vlen

                    if key == b"image_name":
                        kk = k
                        while kk < vend:
                            vtag, kk = _read_varint(buf, kk)
                            vnum = vtag >> 3
                            vwt = vtag & 7
                            if vnum == 1 and vwt == 2:  # BytesList
                                blen, kk = _read_varint(buf, kk)
                                bend = kk + blen
                                bkk = kk
                                while bkk < bend:
                                    btag, bkk = _read_varint(buf, bkk)
                                    bnum = btag >> 3
                                    bwt2 = btag & 7
                                    if bnum == 1 and bwt2 == 2:
                                        l, bkk = _read_varint(buf, bkk)
                                        return bytes(buf[bkk : bkk + l])
                                    bkk = _skip_field(bwt2, buf, bkk)
                                kk = bend
                            else:
                                kk = _skip_field(vwt, buf, kk)
                    k = vend
                else:
                    k = _skip_field(ewt, buf, k)

            j = ent_end

        i = fend
    return b""


def _parse_example_name_and_image_slice(serialized: bytes):
    buf = memoryview(serialized)
    n = len(buf)
    i = 0

    image_name = b""
    img_start = -1
    img_len = 0

    while i < n:
        tag, i = _read_varint(buf, i)
        field_num = tag >> 3
        wire_type = tag & 7
        if field_num != 1 or wire_type != 2:
            i = _skip_field(wire_type, buf, i)
            continue
        flen, i = _read_varint(buf, i)
        fend = i + flen

        j = i
        while j < fend:
            ftag, j = _read_varint(buf, j)
            fnum = ftag >> 3
            fwt = ftag & 7
            if fnum != 1 or fwt != 2:
                j = _skip_field(fwt, buf, j)
                continue
            ent_len, j = _read_varint(buf, j)
            ent_end = j + ent_len

            key = None
            k = j
            while k < ent_end:
                etag, k = _read_varint(buf, k)
                enum = etag >> 3
                ewt = etag & 7
                if enum == 1 and ewt == 2:  # key
                    key_len, k = _read_varint(buf, k)
                    key = bytes(buf[k : k + key_len])
                    k += key_len
                elif enum == 2 and ewt == 2:  # value Feature
                    vlen, k = _read_varint(buf, k)
                    vend = k + vlen

                    kk = k
                    while kk < vend:
                        vtag, kk = _read_varint(buf, kk)
                        vnum = vtag >> 3
                        vwt = vtag & 7
                        if vnum == 1 and vwt == 2:  # BytesList
                            blen, kk = _read_varint(buf, kk)
                            bend = kk + blen

                            bkk = kk
                            while bkk < bend:
                                btag, bkk = _read_varint(buf, bkk)
                                bnum = btag >> 3
                                bwt2 = btag & 7
                                if bnum == 1 and bwt2 == 2:
                                    l, bkk2 = _read_varint(buf, bkk)
                                    if key == b"image_name" and not image_name:
                                        image_name = bytes(buf[bkk2 : bkk2 + l])
                                    if key == b"image" and img_start < 0:
                                        img_start = bkk2
                                        img_len = int(l)
                                    bkk = bkk2 + l
                                else:
                                    bkk = _skip_field(bwt2, buf, bkk)
                            kk = bend
                        else:
                            kk = _skip_field(vwt, buf, kk)

                    k = vend
                else:
                    k = _skip_field(ewt, buf, k)

            j = ent_end

        i = fend

    return image_name, img_start, img_len


def _parse_example(serialized: bytes, want_image=True, want_label=True, want_name=True):
    buf = memoryview(serialized)
    n = len(buf)
    i = 0
    out = {}
    while i < n:
        tag, i = _read_varint(buf, i)
        field_num = tag >> 3
        wire_type = tag & 7
        if field_num != 1 or wire_type != 2:
            i = _skip_field(wire_type, buf, i)
            continue
        flen, i = _read_varint(buf, i)
        fend = i + flen

        j = i
        while j < fend:
            ftag, j = _read_varint(buf, j)
            fnum = ftag >> 3
            fwt = ftag & 7
            if fnum != 1 or fwt != 2:
                j = _skip_field(fwt, buf, j)
                continue
            ent_len, j = _read_varint(buf, j)
            ent_end = j + ent_len

            key = None
            val = None

            k = j
            while k < ent_end:
                etag, k = _read_varint(buf, k)
                enum = etag >> 3
                ewt = etag & 7
                if enum == 1 and ewt == 2:  # key
                    key_len, k = _read_varint(buf, k)
                    key = bytes(buf[k : k + key_len])
                    k += key_len
                elif enum == 2 and ewt == 2:  # value (Feature message)
                    vlen, k = _read_varint(buf, k)
                    vend = k + vlen

                    kk = k
                    parsed = None
                    while kk < vend:
                        vtag, kk = _read_varint(buf, kk)
                        vnum = vtag >> 3
                        vwt = vtag & 7
                        if vnum == 1 and vwt == 2:  # BytesList
                            blen, kk = _read_varint(buf, kk)
                            bend = kk + blen
                            bkk = kk
                            vals = []
                            while bkk < bend:
                                btag, bkk = _read_varint(buf, bkk)
                                bnum = btag >> 3
                                bwt2 = btag & 7
                                if bnum == 1 and bwt2 == 2:
                                    l, bkk = _read_varint(buf, bkk)
                                    vals.append(bytes(buf[bkk : bkk + l]))
                                    bkk += l
                                else:
                                    bkk = _skip_field(bwt2, buf, bkk)
                            parsed = vals
                            kk = bend
                        elif vnum == 3 and vwt == 2:  # Int64List
                            ilen, kk = _read_varint(buf, kk)
                            iend = kk + ilen
                            ikk = kk
                            vals = []
                            while ikk < iend:
                                itag, ikk = _read_varint(buf, ikk)
                                inum = itag >> 3
                                iwt2 = itag & 7
                                if inum == 1 and iwt2 == 0:
                                    v, ikk = _read_varint(buf, ikk)
                                    vals.append(int(v))
                                else:
                                    ikk = _skip_field(iwt2, buf, ikk)
                            parsed = vals
                            kk = iend
                        else:
                            kk = _skip_field(vwt, buf, kk)

                    val = parsed
                    k = vend
                else:
                    k = _skip_field(ewt, buf, k)

            if key == b"image" and want_image:
                out[key] = val
            elif key == b"label" and want_label:
                out[key] = val
            elif key == b"image_name" and want_name:
                out[key] = val

            j = ent_end

        i = fend

    img = out.get(b"image", [b""])[0] if want_image else b""
    lbl = out.get(b"label", [0])[0] if want_label else 0
    name = out.get(b"image_name", [b""])[0] if want_name else b""
    return img, int(lbl), name


def _build_or_load_needed_index(tfrec_paths, needed_names_bytes, cache_path):
    needed_set = set(needed_names_bytes)
    if os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=False)
            names = d["names"]
            file_ids = d["file_ids"]
            data_offs = d["data_offs"]
            lengths = d["lengths"]
            img_offs = d["img_offs"]
            img_lens = d["img_lens"]
            files = d["files"]
            idx = {
                names[i].tobytes() if hasattr(names[i], "tobytes") else names[i]: i
                for i in range(len(names))
            }
            ok = all((b in idx) for b in needed_set)
            if ok:
                out = {}
                for b in needed_set:
                    i = idx[b]
                    out[b] = (
                        str(files[file_ids[i]]),
                        int(data_offs[i]),
                        int(lengths[i]),
                        int(img_offs[i]),
                        int(img_lens[i]),
                    )
                return out
        except Exception:
            pass

    files = [str(p) for p in tfrec_paths]
    file_to_id = {p: i for i, p in enumerate(files)}

    names_list = []
    file_ids_list = []
    data_offs_list = []
    lengths_list = []
    img_offs_list = []
    img_lens_list = []

    remaining = len(needed_set)
    for p in files:
        if remaining == 0:
            break
        fid = file_to_id[p]
        with open(p, "rb", buffering=1024 * 1024) as f:
            while True:
                header = f.read(8)
                if not header:
                    break
                (length,) = struct.unpack("<Q", header)
                f.read(4)  # length crc
                data_off = f.tell()
                data = f.read(length)
                f.read(4)  # data crc

                name_b, img_start, img_len = _parse_example_name_and_image_slice(data)
                if name_b and (name_b in needed_set):
                    if img_start < 0 or img_len <= 0:
                        img_bytes, _, _ = _parse_example(
                            data, want_image=True, want_label=False, want_name=False
                        )
                        img_start = -1
                        img_len = 0

                    names_list.append(name_b)
                    file_ids_list.append(fid)
                    data_offs_list.append(data_off)
                    lengths_list.append(length)
                    img_offs_list.append(img_start)
                    img_lens_list.append(img_len)

                    needed_set.remove(name_b)
                    remaining -= 1
                    if remaining == 0:
                        break

    if remaining != 0:
        raise KeyError(
            f"Missing {remaining} records in TFRecords (some image_ids not found)."
        )

    names_arr = np.array(names_list, dtype=object)
    file_ids_arr = np.asarray(file_ids_list, dtype=np.int16)
    data_offs_arr = np.asarray(data_offs_list, dtype=np.int64)
    lengths_arr = np.asarray(lengths_list, dtype=np.int32)
    img_offs_arr = np.asarray(img_offs_list, dtype=np.int32)
    img_lens_arr = np.asarray(img_lens_list, dtype=np.int32)
    files_arr = np.array(files, dtype=object)

    try:
        np.savez_compressed(
            cache_path,
            names=names_arr,
            file_ids=file_ids_arr,
            data_offs=data_offs_arr,
            lengths=lengths_arr,
            img_offs=img_offs_arr,
            img_lens=img_lens_arr,
            files=files_arr,
        )
    except Exception:
        pass

    out = {}
    for i, b in enumerate(names_list):
        out[b] = (
            files[file_ids_list[i]],
            int(data_offs_list[i]),
            int(lengths_list[i]),
            int(img_offs_list[i]),
            int(img_lens_list[i]),
        )
    return out


train_tfrec_paths = sorted(glob.glob(os.path.join(TRAIN_TFRECORD_DIR, "*.tfrec")))
test_tfrec_paths = sorted(glob.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec")))
assert len(train_tfrec_paths) > 0, "No train tfrecords found"
assert len(test_tfrec_paths) > 0, "No test tfrecords found"

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(splitter.split(train_df["image_id"], train_df["label"]))

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_needed = np.unique(
    np.concatenate([tr_df["image_id"].values, va_df["image_id"].values])
)
train_needed_b = [s.encode("utf-8") for s in train_needed.tolist()]
test_needed = sample_df["image_id"].values
test_needed_b = [s.encode("utf-8") for s in test_needed.tolist()]

train_cache = "/kaggle/working/train_tfrecord_index_needed.npz"
test_cache = "/kaggle/working/test_tfrecord_index_needed.npz"

train_name_to_rec = _build_or_load_needed_index(
    train_tfrec_paths, train_needed_b, train_cache
)
test_name_to_rec = _build_or_load_needed_index(
    test_tfrec_paths, test_needed_b, test_cache
)

import multiprocessing as mp

batch_size = 32 if device.type == "cuda" else 16
cpu_cnt = os.cpu_count() or 2

if device.type == "cuda":
    num_workers = min(4, max(2, cpu_cnt // 4))
else:
    num_workers = min(2, max(1, cpu_cnt // 4))

prefetch_factor = 2 if num_workers > 0 else None


def _seed_worker(worker_id: int):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(SEED)

pin_mem = device.type == "cuda"
pin_mem_dev = "cuda" if pin_mem else ""


## === cell 4
from PIL import Image


class _TFRecordRandomAccessor:
    def __init__(self):
        self._files = {}  # path -> open file handle

    def get_record_bytes(self, path, data_off, length):
        f = self._files.get(path)
        if f is None:
            f = open(path, "rb", buffering=1024 * 1024)
            self._files[path] = f
        f.seek(int(data_off))
        return f.read(int(length))

    def close(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()


def _jpeg_bytes_to_pil_rgb_fast(img_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


class CassavaTrainDataset(Dataset):
    def __init__(self, df, tfrecord_offset_index, transform=None):
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(np.int64)
        self.tfrecord_offset_index = tfrecord_offset_index
        self.transform = transform

        self._name_bytes = np.fromiter(
            (s.encode("utf-8") for s in self.image_ids),
            count=len(self.image_ids),
            dtype=object,
        )
        self._rec = np.array(
            [self.tfrecord_offset_index[b] for b in self._name_bytes], dtype=object
        )
        self._acc = None  # lazily created per-process

    def __len__(self):
        return self.image_ids.shape[0]

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_acc"] = None  # don't pickle open file handles
        return d

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        path, data_off, length, img_off, img_len = self._rec[idx]

        if self._acc is None:
            self._acc = _TFRecordRandomAccessor()

        rec_bytes = self._acc.get_record_bytes(path, data_off, length)

        if img_off >= 0 and img_len > 0:
            img_bytes = rec_bytes[img_off : img_off + img_len]
        else:
            img_bytes, _, _ = _parse_example(
                rec_bytes, want_image=True, want_label=False, want_name=False
            )

        img = _jpeg_bytes_to_pil_rgb_fast(img_bytes)

        if self.transform is not None:
            img = self.transform(img)
        return img, label


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, tfrecord_offset_index, transform=None):
        self.image_ids = np.asarray(list(image_ids))
        self.tfrecord_offset_index = tfrecord_offset_index
        self.transform = transform

        self._name_bytes = np.fromiter(
            (str(s).encode("utf-8") for s in self.image_ids),
            count=len(self.image_ids),
            dtype=object,
        )
        self._rec = np.array(
            [self.tfrecord_offset_index[b] for b in self._name_bytes], dtype=object
        )
        self._acc = None  # lazily created per-process

    def __len__(self):
        return self.image_ids.shape[0]

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_acc"] = None
        return d

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx])
        path, data_off, length, img_off, img_len = self._rec[idx]

        if self._acc is None:
            self._acc = _TFRecordRandomAccessor()

        rec_bytes = self._acc.get_record_bytes(path, data_off, length)

        if img_off >= 0 and img_len > 0:
            img_bytes = rec_bytes[img_off : img_off + img_len]
        else:
            img_bytes, _, _ = _parse_example(
                rec_bytes, want_image=True, want_label=False, want_name=False
            )

        img = _jpeg_bytes_to_pil_rgb_fast(img_bytes)

        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


train_ds = CassavaTrainDataset(tr_df, train_name_to_rec, transform=train_tfms)
val_ds = CassavaTrainDataset(va_df, train_name_to_rec, transform=valid_tfms)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_mem,
    pin_memory_device=pin_mem_dev,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    pin_memory_device=pin_mem_dev,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

print("train/val sizes:", len(train_ds), len(val_ds))


## === cell 5
model = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
    except Exception:
        pass

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))




## === cell 6
@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs
    return total_loss / total_n, total_correct / total_n


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs

    return total_loss / total_n, total_correct / total_n




## === cell 7
epochs = 3

best_val_acc = -1.0
best_buf = None

for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    va_loss, va_acc = evaluate(model, val_loader)
    print(
        f"epoch {epoch}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}"
    )

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        buf = io.BytesIO()
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, buf)
        best_buf = buf.getvalue()

print("best_val_acc:", best_val_acc)

if best_buf is not None:
    model.load_state_dict(torch.load(io.BytesIO(best_buf), map_location=device))


## === cell 8
test_ids = sample_df["image_id"].values
test_ds = CassavaTestDataset(test_ids, test_name_to_rec, transform=valid_tfms)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    pin_memory_device=pin_mem_dev,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

model.eval()
all_preds = np.empty(len(test_ds), dtype=np.int64)

with torch.inference_mode():
    offset = 0
    for x, image_ids in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        bs = preds.shape[0]
        all_preds[offset : offset + bs] = preds
        offset += bs

all_ids = list(test_ids)
sub_df = pd.DataFrame({"image_id": all_ids, "label": all_preds.astype(int)})
assert sub_df.shape[0] == sample_df.shape[0]
assert list(sub_df.columns) == ["image_id", "label"]
assert (sub_df["image_id"].values == sample_df["image_id"].values).all()

sub_df.to_csv(SUBMISSION_PATH, index=False)
print("Wrote:", SUBMISSION_PATH)
print(sub_df.head())


## === cell 9
print(sub_df["label"].value_counts().sort_index())
print("submission rows:", len(sub_df))
print("unique ids:", sub_df["image_id"].nunique())
print("submission path exists:", os.path.exists(SUBMISSION_PATH))
print(
    "submission file size:",
    os.path.getsize(SUBMISSION_PATH) if os.path.exists(SUBMISSION_PATH) else None,
)
