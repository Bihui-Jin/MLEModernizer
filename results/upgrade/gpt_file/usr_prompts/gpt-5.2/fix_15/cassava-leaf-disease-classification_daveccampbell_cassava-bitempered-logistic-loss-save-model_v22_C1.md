# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

# 5. Target score

0.0991236022967664

# 6. Current score

0.2003

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17526) has done: 'I fix the runtime error caused by using `torch.cuda.amp.autocast` with the newer `device_type=` argument by switching to the correct `torch.amp.autocast` context manager while keeping AMP behavior identical on CUDA. I also make the EfficientNet head replacement robust across timm model variants (some use `classifier`, others `fc`/`head`) to avoid attribute errors. Finally, I keep training/inference logic the same and ensure the script always writes `/kaggle/working/submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.13341) has done: 'The timeout is dominated by training cost (EfficientNet-B4 at 512px for 10 epochs) and heavy CPU image augmentation/decoding overhead that starves the GPU. I keep the exact same model, loss, epochs, and data semantics, but speed things up by (1) switching to the dataset’s TFRecords (same images/labels, much faster sequential I/O than many JPEG opens), (2) optimizing the PyTorch input pipeline (larger prefetch, tuned workers, pinned memory, persistent workers, faster collation, and keeping CPU transforms off the critical path where possible), and (3) reducing per-iteration overhead inside the training loop (avoid repeated object creation and unnecessary Python work) while preserving identical training/eval behavior. These changes are runtime-focused and keep the core logic intact (same architecture, same bi-tempered loss, same loop structure, same fold split, same augmentations for train and resize+normalize for valid/test). If TFRecords aren’t present for some reason, it automatically fall back to the original JPEG loading.'
- What this solution (achieved 0.12706) has done: 'I fix the TensorFlow/protobuf import crash by removing the hard dependency on `tensorflow` (it isn’t needed because TFRecords can be parsed without it). Then I fix the DataLoader worker crash by removing the invalid attempt to assign attributes onto `WorkerInfo` and instead keep a per-process dataset cache in the dataset instance itself. Finally, I fix the test-TFRecord branch that accidentally overwrote the TFRecord dataset with the JPEG dataset, and make TFRecord parsing robust by matching the Cassava TFRecord schema (`image`, `image_name`/`image_id`, and optional `label`). These changes are execution/stability fixes and should also recover the intended faster TFRecord pipeline (which should help score by letting the full training run complete reliably).'
- What this solution (achieved 0.2003) has done: 'I fix the TFRecord label KeyError by making the TFRecord dataset filter its index to only examples that exist in the current fold’s `id_to_label` mapping (instead of trying to train on all TFRecord examples across folds). I also prevent the test-time TFRecord alignment dataset from pre-scanning and caching every record (which is causing a DataLoader worker `MemoryError`) by switching it to a streaming, one-time scan that builds the mapping without storing large intermediate data and by forcing `num_workers=0` for that scan-heavy dataset. These are stability/correctness fixes that preserve the same model, loss, epochs, and augmentations, and allow the pipeline to run end-to-end and write `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import copy
import random
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from torch.cuda import amp

from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm



## === cell 1
ROOT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_TFREC_DIR = "../input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "../input/cassava-leaf-disease-classification/test_tfrecords"

if not os.path.exists(ROOT_DIR):
    ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
    TRAIN_DIR = f"{ROOT_DIR}/train_images"
    TEST_DIR = f"{ROOT_DIR}/test_images"
    TRAIN_TFREC_DIR = f"{ROOT_DIR}/train_tfrecords"
    TEST_TFREC_DIR = f"{ROOT_DIR}/test_tfrecords"

assert os.path.exists(f"{ROOT_DIR}/train.csv"), f"train.csv not found under {ROOT_DIR}"




## === cell 2
class CFG:
    model_name = "tf_efficientnet_b4_ns"
    img_size = 512
    scheduler = "CosineAnnealingWarmRestarts"
    T_max = 10
    T_0 = 10
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-6
    seed = 42
    num_classes = 5
    num_epochs = 10
    n_fold = 5
    smoothing = 0.2
    t1 = 0.8
    t2 = 1.4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    savemodel = True
    savemodelpath = "/kaggle/working/bitemp-21.pth"
    dosubmission = True
    crop_size = 256
    crop_p = 0.9




## === cell 3
def set_seed(seed=42):
    """Reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



## === cell 4
df = pd.read_csv(f"{ROOT_DIR}/train.csv")
assert {"image_id", "label"}.issubset(df.columns)

skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
    df.loc[val_, "kfold"] = int(fold)
df["kfold"] = df["kfold"].astype(int)




## === cell 5
class CassavaLeafDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None, has_labels=True):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.has_labels = has_labels and ("label" in self.df.columns)

        if "image_id" in self.df.columns:
            image_ids = self.df["image_id"].astype(str).values
        else:
            image_ids = self.df.iloc[:, 0].astype(str).values

        self.img_paths = [os.path.join(self.root_dir, x) for x in image_ids]

        if self.has_labels:
            self.labels = self.df["label"].astype(np.int64).values
        else:
            self.labels = None

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _read_rgb(path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self._read_rgb(img_path)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        else:
            img = ToTensorV2()(image=img)["image"]

        if self.has_labels:
            return img, int(self.labels[index])
        return img, -1


class CassavaTFRecordDataset(Dataset):
    def __init__(self, tfrec_files, id_to_label=None, transforms=None, has_labels=True):
        self.tfrec_files = list(tfrec_files)
        self.id_to_label = id_to_label or {}
        self.transforms = transforms
        self.has_labels = has_labels

        self._file_cache = {}  # tfrec_path -> open file handle (per worker process)
        self._index = []
        for f in self.tfrec_files:
            self._index.extend(self._build_index_for_file(f))

        if self.has_labels and self.id_to_label:
            self._index = self._filter_index_to_known_ids(self._index)

    @staticmethod
    def _iter_record_offsets(tfrec_path: str):
        offsets = []
        with open(tfrec_path, "rb") as f:
            pos = 0
            while True:
                header = f.read(12)  # length (8) + length_crc (4)
                if len(header) < 12:
                    break
                length = int.from_bytes(header[:8], "little", signed=False)
                offsets.append(pos)
                f.seek(length + 4, os.SEEK_CUR)  # data + data_crc
                pos = f.tell()
        return offsets

    def _build_index_for_file(self, tfrec_path: str):
        return [(tfrec_path, off) for off in self._iter_record_offsets(tfrec_path)]

    def _read_record_at(self, tfrec_path: str, offset: int) -> bytes:
        fh = self._file_cache.get(tfrec_path)
        if fh is None or fh.closed:
            fh = open(tfrec_path, "rb")
            self._file_cache[tfrec_path] = fh
        fh.seek(offset)
        header = fh.read(12)
        if len(header) < 12:
            raise EOFError(f"Unexpected EOF reading TFRecord header: {tfrec_path}")
        length = int.from_bytes(header[:8], "little", signed=False)
        data = fh.read(length)
        _ = fh.read(4)  # data_crc (ignored)
        if len(data) != length:
            raise EOFError(f"Unexpected EOF reading TFRecord data: {tfrec_path}")
        return data

    @staticmethod
    def _read_uvarint(buf: bytes, i: int):
        x = 0
        shift = 0
        while True:
            if i >= len(buf):
                raise EOFError("Unexpected EOF while reading varint")
            b = buf[i]
            i += 1
            x |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
            if shift > 70:
                raise ValueError("varint too long")
        return x, i

    @staticmethod
    def _skip_field(wire_type: int, buf: bytes, i: int) -> int:
        if wire_type == 0:  # varint
            _, i = CassavaTFRecordDataset._read_uvarint(buf, i)
            return i
        if wire_type == 1:  # 64-bit
            return i + 8
        if wire_type == 2:  # len-delimited
            l, i = CassavaTFRecordDataset._read_uvarint(buf, i)
            return i + l
        if wire_type == 5:  # 32-bit
            return i + 4
        raise ValueError(f"Unsupported wire type: {wire_type}")

    @staticmethod
    def _parse_example_minimal(raw: bytes):
        """
        Parse tensorflow.train.Example without tensorflow/protobuf deps.
        We extract:
          - 'image': bytes (JPEG)
          - 'image_name' or 'image_id': bytes -> str (e.g., '123.jpg')
          - 'label': int64 (optional)
        """
        i = 0
        features_bytes = None
        while i < len(raw):
            key, i = CassavaTFRecordDataset._read_uvarint(raw, i)
            field = key >> 3
            wire = key & 7
            if field == 1 and wire == 2:
                l, i = CassavaTFRecordDataset._read_uvarint(raw, i)
                features_bytes = raw[i : i + l]
                i += l
            else:
                i = CassavaTFRecordDataset._skip_field(wire, raw, i)

        if features_bytes is None:
            raise RuntimeError("TFRecord Example parse failed: missing features")

        fb = features_bytes
        i = 0
        img_bytes = None
        img_id = None
        label = None

        while i < len(fb):
            key, i = CassavaTFRecordDataset._read_uvarint(fb, i)
            field = key >> 3
            wire = key & 7
            if field != 1 or wire != 2:
                i = CassavaTFRecordDataset._skip_field(wire, fb, i)
                continue

            entry_len, i = CassavaTFRecordDataset._read_uvarint(fb, i)
            entry = fb[i : i + entry_len]
            i += entry_len

            j = 0
            k = None
            v = None
            while j < len(entry):
                ekey, j = CassavaTFRecordDataset._read_uvarint(entry, j)
                ef = ekey >> 3
                ew = ekey & 7
                if ef == 1 and ew == 2:
                    kl, j = CassavaTFRecordDataset._read_uvarint(entry, j)
                    k = entry[j : j + kl].decode("utf-8", errors="ignore")
                    j += kl
                elif ef == 2 and ew == 2:
                    vl, j = CassavaTFRecordDataset._read_uvarint(entry, j)
                    v = entry[j : j + vl]
                    j += vl
                else:
                    j = CassavaTFRecordDataset._skip_field(ew, entry, j)

            if k is None or v is None:
                continue

            def _parse_feature_value(feature_msg: bytes):
                jj = 0
                out_bytes = []
                out_ints = []
                while jj < len(feature_msg):
                    fkey, jj = CassavaTFRecordDataset._read_uvarint(feature_msg, jj)
                    ff = fkey >> 3
                    fw = fkey & 7
                    if ff == 1 and fw == 2:
                        blen, jj = CassavaTFRecordDataset._read_uvarint(feature_msg, jj)
                        bytes_list_msg = feature_msg[jj : jj + blen]
                        jj += blen
                        kk = 0
                        while kk < len(bytes_list_msg):
                            bkey, kk = CassavaTFRecordDataset._read_uvarint(
                                bytes_list_msg, kk
                            )
                            bf = bkey >> 3
                            bw = bkey & 7
                            if bf == 1 and bw == 2:
                                vl2, kk = CassavaTFRecordDataset._read_uvarint(
                                    bytes_list_msg, kk
                                )
                                out_bytes.append(bytes_list_msg[kk : kk + vl2])
                                kk += vl2
                            else:
                                kk = CassavaTFRecordDataset._skip_field(
                                    bw, bytes_list_msg, kk
                                )
                    elif ff == 3 and fw == 2:
                        ilen, jj = CassavaTFRecordDataset._read_uvarint(feature_msg, jj)
                        int64_list_msg = feature_msg[jj : jj + ilen]
                        jj += ilen
                        kk = 0
                        while kk < len(int64_list_msg):
                            ikey, kk = CassavaTFRecordDataset._read_uvarint(
                                int64_list_msg, kk
                            )
                            inf = ikey >> 3
                            inw = ikey & 7
                            if inf == 1 and inw == 0:
                                ival, kk = CassavaTFRecordDataset._read_uvarint(
                                    int64_list_msg, kk
                                )
                                out_ints.append(int(ival))
                            else:
                                kk = CassavaTFRecordDataset._skip_field(
                                    inw, int64_list_msg, kk
                                )
                    else:
                        jj = CassavaTFRecordDataset._skip_field(fw, feature_msg, jj)
                return out_bytes, out_ints

            bvals, ivals = _parse_feature_value(v)
            if k == "image" and len(bvals) > 0:
                img_bytes = bvals[0]
            elif k in ("image_name", "image_id") and len(bvals) > 0:
                try:
                    img_id = bvals[0].decode("utf-8", errors="ignore")
                except Exception:
                    img_id = None
            elif k == "label" and len(ivals) > 0:
                label = int(ivals[0])

        if img_bytes is None:
            raise RuntimeError("TFRecord Example parse failed: missing image bytes")
        return img_id, img_bytes, label

    def _filter_index_to_known_ids(self, index_list):
        keep = []
        for tfrec_path, offset in index_list:
            raw = self._read_record_at(tfrec_path, offset)
            img_id, _, embedded_label = self._parse_example_minimal(raw)
            if img_id is not None and img_id in self.id_to_label:
                keep.append((tfrec_path, offset))
            elif embedded_label is not None:
                keep.append((tfrec_path, offset))
        if len(keep) == 0:
            return index_list
        return keep

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        tfrec_path, offset = self._index[idx]
        raw = self._read_record_at(tfrec_path, offset)
        img_id, img_bytes, embedded_label = self._parse_example_minimal(raw)

        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(
                f"Failed to decode image from TFRecord: {tfrec_path} @ {offset}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        else:
            img = ToTensorV2()(image=img)["image"]

        if self.has_labels:
            if img_id is not None and img_id in self.id_to_label:
                return img, int(self.id_to_label[img_id])
            if embedded_label is not None:
                return img, int(embedded_label)
            return img, 0

        return img, -1


class CassavaTFRecordAlignedTestDataset(Dataset):
    def __init__(self, tfrec_files, image_ids, transforms=None):
        self.image_ids = [str(x) for x in image_ids]
        self.transforms = transforms

        base = CassavaTFRecordDataset(
            tfrec_files=tfrec_files,
            id_to_label={},
            transforms=transforms,
            has_labels=False,
        )

        id_to_idx = {}
        for i in range(len(base)):
            tfrec_path, offset = base._index[i]
            raw = base._read_record_at(tfrec_path, offset)
            img_id, _, _ = base._parse_example_minimal(raw)
            if img_id is not None and img_id not in id_to_idx:
                id_to_idx[img_id] = i

        missing = [iid for iid in self.image_ids if iid not in id_to_idx]
        if len(missing) > 0:
            raise KeyError(
                f"Some test image_ids not found in TFRecords (n_missing={len(missing)}). Example: {missing[0]}"
            )

        self.base = base
        self.reindex = [id_to_idx[iid] for iid in self.image_ids]

    def __len__(self):
        return len(self.reindex)

    def __getitem__(self, idx):
        return self.base[self.reindex[idx]]




## === cell 6
data_transforms = {
    "train": A.Compose(
        [
            A.RandomResizedCrop(
                size=(CFG.img_size, CFG.img_size),
                p=1.0,
            ),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            A.CoarseDropout(p=0.5),
            ToTensorV2(),
        ],
        p=1.0,
    ),
    "valid": A.Compose(
        [
            A.Resize(height=CFG.img_size, width=CFG.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    ),
}




## === cell 7
def log_t(u, t):
    if t == 1.0:
        return u.log()
    else:
        inv_1mt = 1.0 / (1.0 - t)
        return (u.pow(1.0 - t) - 1.0) * inv_1mt


def exp_t(u, t):
    if t == 1:
        return u.exp()
    else:
        inv_1mt = 1.0 / (1.0 - t)
        return (1.0 + (1.0 - t) * u).relu().pow(inv_1mt)


def compute_normalization_fixed_point(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    pow_1mt = 1.0 - t
    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(
            pow_1mt
        )
    logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = -log_t(1.0 / logt_partition, t) + mu
    return normalization_constants


def compute_normalization_binary_search(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu
    inv_1mt = 1.0 / (1.0 - t)

    effective_dim = torch.sum(
        (normalized_activations > -inv_1mt).to(torch.int32),
        dim=-1,
        keepdim=True,
    ).to(activations.dtype)

    lower = torch.zeros_like(mu)
    upper = -log_t(1.0 / effective_dim, t)  # same shape as mu

    for _ in range(num_iters):
        logt_partition = (upper + lower) * 0.5
        sum_probs = torch.sum(
            exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True
        )
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = lower * update + (1.0 - update) * logt_partition
        upper = upper * (1.0 - update) + update * logt_partition

    logt_partition = (upper + lower) * 0.5
    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(
                activations, t, num_iters
            )
        else:
            normalization_constants = compute_normalization_fixed_point(
                activations, t, num_iters
            )
        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    if t == 1.0:
        return activations.softmax(dim=-1)
    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(
    activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction="mean"
):
    if len(labels.shape) < len(activations.shape):
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = (
            1 - label_smoothing * num_classes / (num_classes - 1)
        ) * labels_onehot + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    two_minus_t1 = 2.0 - t1
    loss_values = (
        labels_onehot * log_t(labels_onehot + 1e-10, t1)
        - labels_onehot * log_t(probabilities, t1)
        - labels_onehot.pow(two_minus_t1) / two_minus_t1
        + probabilities.pow(two_minus_t1) / two_minus_t1
    ).sum(dim=-1)

    if reduction == "none":
        return loss_values
    if reduction == "sum":
        return loss_values.sum()
    return loss_values.mean()




## === cell 8
class CUDAPrefetcher:
    def __init__(self, loader, device, use_channels_last: bool):
        self.loader = loader
        self.device = device
        self.use_channels_last = use_channels_last and (device.type == "cuda")
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        next_inputs, next_labels = None, None

        def _preload():
            nonlocal next_inputs, next_labels
            try:
                inputs, labels = next(it)
            except StopIteration:
                next_inputs, next_labels = None, None
                return
            with torch.cuda.stream(self.stream):
                inputs = inputs.to(self.device, non_blocking=True)
                if self.use_channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
            next_inputs, next_labels = inputs, labels

        _preload()
        while next_inputs is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            inputs, labels = next_inputs, next_labels
            _preload()
            yield inputs, labels

    def __len__(self):
        return len(self.loader)


def train_model(
    model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold
):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = defaultdict(list)
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    t1 = CFG.t1
    t2 = CFG.t2
    smoothing = CFG.smoothing
    use_cuda = device.type == "cuda"

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        prefetched_train = CUDAPrefetcher(
            dataloaders["train"], device=device, use_channels_last=True
        )
        prefetched_valid = CUDAPrefetcher(
            dataloaders["valid"], device=device, use_channels_last=True
        )

        for phase, prefetched in (
            ("train", prefetched_train),
            ("valid", prefetched_valid),
        ):
            is_train = phase == "train"
            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            ctx = torch.enable_grad() if is_train else torch.inference_mode()
            with ctx:
                for inputs, labels in prefetched:
                    if device.type != "cuda":
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                    if is_train:
                        optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                        outputs = model(inputs)
                        preds = outputs.argmax(dim=1)
                        loss = bi_tempered_logistic_loss(
                            outputs,
                            labels,
                            t1=t1,
                            t2=t2,
                            label_smoothing=smoothing,
                        )

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    bs = inputs.size(0)
                    running_loss += float(loss.detach()) * bs
                    running_corrects += int((preds == labels).sum().detach())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[phase + " loss"].append(epoch_loss)
            history[phase + " acc"].append(epoch_acc)

            if is_train and scheduler is not None:
                if isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step(epoch)
                else:
                    scheduler.step()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if (not is_train) and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Accuracy", best_acc)

    model.load_state_dict(best_model_wts)
    return model, history




## === cell 9
def seed_worker(worker_id):
    worker_seed = CFG.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)




## === cell 10
def _get_train_dataset(train_df, transforms):
    tfrec_files = sorted(glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    if len(tfrec_files) == 0:
        return CassavaLeafDataset(
            TRAIN_DIR, train_df, transforms=transforms, has_labels=True
        )

    id_to_label = dict(
        zip(
            train_df["image_id"].astype(str).values,
            train_df["label"].astype(int).values,
        )
    )
    return CassavaTFRecordDataset(
        tfrec_files=tfrec_files,
        id_to_label=id_to_label,
        transforms=transforms,
        has_labels=True,
    )


def _get_valid_dataset(valid_df, transforms):
    tfrec_files = sorted(glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    if len(tfrec_files) == 0:
        return CassavaLeafDataset(
            TRAIN_DIR, valid_df, transforms=transforms, has_labels=True
        )

    id_to_label = dict(
        zip(
            valid_df["image_id"].astype(str).values,
            valid_df["label"].astype(int).values,
        )
    )
    return CassavaTFRecordDataset(
        tfrec_files=tfrec_files,
        id_to_label=id_to_label,
        transforms=transforms,
        has_labels=True,
    )


def run_fold(model, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    train_df = df[df.kfold != fold].reset_index(drop=True)

    train_data = _get_train_dataset(train_df, transforms=data_transforms["train"])
    valid_data = _get_valid_dataset(valid_df, transforms=data_transforms["valid"])

    dataset_sizes = {"train": len(train_data), "valid": len(valid_data)}

    cpu_count = os.cpu_count() or 2
    num_workers = min(12, max(4, cpu_count - 2))

    g = torch.Generator()
    g.manual_seed(CFG.seed)

    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
    )
    if common_loader_kwargs["prefetch_factor"] is None:
        common_loader_kwargs.pop("prefetch_factor")

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=CFG.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    dataloaders = {"train": train_loader, "valid": valid_loader}

    model, history = train_model(
        model,
        optimizer,
        scheduler,
        num_epochs,
        dataloaders,
        dataset_sizes,
        device,
        fold,
    )
    return model, history




## === cell 11
model = timm.create_model(CFG.model_name, pretrained=True)

if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, CFG.num_classes)
elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, CFG.num_classes)
elif hasattr(model, "head") and isinstance(model.head, nn.Module):
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, CFG.num_classes)
else:
    raise AttributeError(
        "Could not locate classifier/fc/head to replace for this timm model."
    )

model.to(CFG.device)

optimizer = optim.Adam(
    model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False
)


def fetch_scheduler(optimizer):
    if CFG.scheduler == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr
        )
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr
        )
    return None


scheduler = fetch_scheduler(optimizer)

if CFG.device.type == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(f"torch.compile unavailable/failed, continuing eager. Reason: {e}")



## === cell 12
model, history = run_fold(
    model, optimizer, scheduler, device=CFG.device, fold=0, num_epochs=CFG.num_epochs
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
OverflowError                             Traceback (most recent call last)
/tmp/ipykernel_55/1664952551.py in <cell line: 0>()
----> 1 model, history = run_fold(
      2     model, optimizer, scheduler, device=CFG.device, fold=0, num_epochs=CFG.num_epochs
      3 )
      4 

/tmp/ipykernel_55/3879166357.py in run_fold(model, optimizer, scheduler, device, fold, num_epochs)
     84     dataloaders = {"train": train_loader, "valid": valid_loader}
     85 
---> 86     model, history = train_model(
     87         model,
     88         optimizer,

/tmp/ipykernel_55/3194588809.py in train_model(model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
     83             ctx = torch.enable_grad() if is_train else torch.inference_mode()
     84             with ctx:
---> 85                 for inputs, labels in prefetched:
     86                     if device.type != "cuda":
     87                         inputs = inputs.to(device, non_blocking=True)

/tmp/ipykernel_55/3194588809.py in __iter__(self)
     33             torch.cuda.current_stream().wait_stream(self.stream)
     34             inputs, labels = next_inputs, next_labels
---> 35             _preload()
     36             yield inputs, labels
     37 

/tmp/ipykernel_55/3194588809.py in _preload()
     18             nonlocal next_inputs, next_labels
     19             try:
---> 20                 inputs, labels = next(it)
     21             except StopIteration:
     22                 next_inputs, next_labels = None, None

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1453                 data = self._task_info.pop(self._rcvd_idx)[1]
   1454                 self._rcvd_idx += 1
-> 1455                 return self._process_data(data)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

OverflowError: Caught OverflowError in DataLoader worker process 3.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/1593524255.py", line 281, in __getitem__
    raw = self._read_record_at(tfrec_path, offset)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/1593524255.py", line 90, in _read_record_at
    data = fh.read(length)
           ^^^^^^^^^^^^^^^
OverflowError: cannot fit 'int' into an index-sized integer


## === cell 13
if CFG.savemodel:
    torch.save(
        {
            "model_name": CFG.model_name,
            "num_classes": CFG.num_classes,
            "state_dict": model.state_dict(),
        },
        CFG.savemodelpath,
    )



## === cell 14
if CFG.dosubmission:
    t_df = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")
    assert "image_id" in t_df.columns

    test_tfrec_files = sorted(glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
    if len(test_tfrec_files) > 0:
        t_data = CassavaTFRecordAlignedTestDataset(
            tfrec_files=test_tfrec_files,
            image_ids=t_df["image_id"].astype(str).values,
            transforms=data_transforms["valid"],
        )
    else:
        t_data = CassavaLeafDataset(
            TEST_DIR, t_df, transforms=data_transforms["valid"], has_labels=False
        )

    cpu_count = os.cpu_count() or 2
    if isinstance(t_data, CassavaTFRecordAlignedTestDataset):
        num_workers = 0
    else:
        num_workers = min(12, max(4, cpu_count - 2))

    g = torch.Generator()
    g.manual_seed(CFG.seed)

    loader_kwargs = dict(
        batch_size=32,  # unchanged
        num_workers=num_workers,
        pin_memory=(CFG.device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        shuffle=False,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    t_loader = DataLoader(dataset=t_data, **loader_kwargs)

    model.eval()
    if CFG.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    all_preds = []
    with torch.inference_mode():
        if CFG.device.type == "cuda":
            t_iter = CUDAPrefetcher(t_loader, device=CFG.device, use_channels_last=True)
            for inputs, _ in t_iter:
                with torch.amp.autocast(device_type=CFG.device.type, enabled=True):
                    outputs = model(inputs)
                all_preds.append(outputs.argmax(dim=1).detach().cpu().numpy())
        else:
            for inputs, _ in t_loader:
                inputs = inputs.to(CFG.device, non_blocking=True)
                outputs = model(inputs)
                all_preds.append(outputs.argmax(dim=1).detach().cpu().numpy())

    all_preds = np.concatenate(all_preds).astype(int)

    submit_df = pd.DataFrame({"image_id": t_df["image_id"].values, "label": all_preds})
    out_path = "/kaggle/working/submission.csv"
    submit_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} (rows={len(submit_df)})")
    print(submit_df.head())
    assert out_path.endswith(".csv")
    assert submit_df.shape[0] == t_df.shape[0]
    assert list(submit_df.columns) == ["image_id", "label"]
