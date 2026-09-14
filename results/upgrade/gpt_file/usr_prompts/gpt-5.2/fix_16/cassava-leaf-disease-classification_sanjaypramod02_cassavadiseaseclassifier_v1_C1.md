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

3.12

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
pillow==11.3.0
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.8331822302810517

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the environment crash caused by an incompatibility between `transformers` and `protobuf` by pinning the pure-Python protobuf implementation before importing `transformers`. Then I remove the dependency on missing Kaggle inputs (`/kaggle/input/google-vit` and `/kaggle/input/new-model/new_v2.pth`) by switching to a local, lightweight fallback that preserves the same “ViT-like classifier produces logits then argmax labels” semantics, so the notebook runs end-to-end. Finally, I ensure the test set is read from the provided `test_tfrecords`, predictions align with `sample_submission.csv`, and a valid `/kaggle/working/submission.csv` is always written.'
- What this solution (achieved 0.68797) has done: 'I fix the crash in the very first imports by removing the `transformers` dependency entirely (it’s incompatible with the provided `protobuf` in this environment and fails before your try/except can catch it). To move the accuracy score up toward the target while preserving the overall “image classifier produces logits then argmax” semantics, I minimally extend your fallback model to a small CNN (still taking 224×224 RGB and outputting 5-class logits) and train it on `train_tfrecords` for a few epochs, then run inference on `test_tfrecords`. I also add a deterministic train/val split and keep the submission alignment logic with `sample_submission.csv` to guarantee a valid `/kaggle/working/submission.csv`. All paths remain within the provided dataset structure and the script run end-to-end.'
- What this solution (achieved 0.48019) has done: 'I fix the crash happening before training by preventing TensorFlow from importing an incompatible `protobuf` C++ runtime (the error `MessageFactory.GetPrototype`), using the pure-Python protobuf implementation early in the script. I also make TFRecord reading robust and much faster by streaming TFRecords instead of materializing the full dataset into Python lists up front (this preserves the same image decoding/preprocessing and the same PyTorch training loop/model semantics). Finally, I keep the exact submission schema and alignment with `sample_submission.csv`, while ensuring deterministic behavior remains intact and the notebook always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.57399) has done: 'We fix the crash happening at the very first imports by avoiding TensorFlow/protobuf’s incompatible runtime path: set additional protobuf env flags early and, if needed, fall back to a safe TF import by forcing the pure-Python protobuf implementation consistently. Then we fix a major logic issue that is hurting accuracy: your train/val split currently uses the TFRecord stream enumeration index (TFRecord order), which does not correspond to `train.csv` and can mix labels/rows; we instead split by `image_id` and filter examples by `image_name`. Finally, we keep the same model and training loop semantics, but ensure deterministic, correct dataset filtering and always write a valid `/kaggle/working/submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.6846) has done: 'The timeout is dominated by TFRecord decoding: each example is converted to NumPy and decoded/resized via PIL in Python, which is extremely slow across ~18k train + ~2.7k test images and is done twice per epoch (train+val) for 4 epochs. I keep the exact same model/training loop semantics, but move decoding+resize into a TensorFlow `tf.data` pipeline (native JPEG decode + resize) and batch there, then convert the already-batched tensors to Torch—this preserves the same numeric range ([0,1]) and resize method (bilinear) while removing Python/PIL per-sample overhead. I also remove repeated CrossEntropyLoss instantiation (create once) and vectorize the split filtering mask to avoid Python loops per batch, keeping the split logic identical. Finally, I enable parallel reads and prefetching in `tf.data` to overlap I/O/CPU with GPU training without changing results.'
- What this solution (achieved 0.6846) has done: 'We need to fix the immediate runtime crash caused by a protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`), which currently happens before any training/inference. The minimal, robust fix is to ensure the protobuf Python implementation is used (and the C++ one avoided) *before* TensorFlow imports, and to fall back safely if TensorFlow import still fails. Once the pipeline runs end-to-end again, we keep your existing model/training semantics unchanged, but we also switch TF decoding from `decode_image` to `decode_jpeg` (the TFRecords contain JPEGs) to avoid slower/edge-case decoding and improve stability/accuracy slightly without changing the overall approach. The script still write `/kaggle/working/submission.csv` with the exact required columns and alignment to `sample_submission.csv`.'
- What this solution (achieved 0.6846) has done: 'I fix the immediate crash by making TensorFlow import robust to the protobuf C++ runtime issue in this Python 3.12 environment: force the pure-Python protobuf implementation early, and if the import still fails, automatically restart the process with environment flags applied before any TF/protobuf code loads. This is a correctness/stability fix and should not change the modeling/training logic or submission schema. After TF imports successfully, the rest of your pipeline (TFRecord tf.data decode/resize → Torch CNN training → argmax predictions → sample_submission alignment) is kept the same. The script always write a valid `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.6704) has done: 'We fix the immediate crash in cell 1 caused by the TensorFlow/protobuf incompatibility by forcing the pure-Python protobuf implementation and proactively importing `google.protobuf.message_factory` before TensorFlow so TensorFlow won’t see the incompatible C++ MessageFactory path. This is a stability-only change that keeps your TFRecord → tf.data decode/resize → Torch CNN training → argmax submission logic identical. Then, because your current score (0.6846) is far below the target (0.8332), we make a minimal, metric-aligned improvement without changing the model/training loop: apply standard ImageNet normalization (after scaling to [0,1]) consistently for both train and test, which typically improves transfer-style CNN performance while preserving the same architecture and loss. Finally, we keep submission alignment to `sample_submission.csv` and ensure `/kaggle/working/submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from PIL import Image
import io
import struct


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

print("torch:", torch.__version__)
print("Using PIL-based TFRecord reader (TensorFlow not imported).")




## === cell 1
def find_local_vit_dir(base="/kaggle/input/google-vit"):
    """
    Original intent: locate a local ViT directory under /kaggle/input/google-vit.
    This environment doesn't provide it; keep non-fatal behavior.
    """
    if not os.path.isdir(base):
        return None
    candidates = [base]
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            candidates.append(p)

    def looks_like_model_dir(p):
        return os.path.isfile(os.path.join(p, "config.json")) or os.path.isfile(
            os.path.join(p, "preprocessor_config.json")
        )

    for p in candidates:
        if looks_like_model_dir(p):
            return p

    for root, dirs, files in os.walk(base):
        if "config.json" in files or "preprocessor_config.json" in files:
            return root
    return None


VIT_DIR = find_local_vit_dir("/kaggle/input/google-vit")
print("Local ViT dir found:", VIT_DIR)



## === cell 2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _varint_read(buf, pos):
    """Read protobuf varint starting at pos; return (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("Too many bytes when decoding varint")


def _skip_field(wire_type, buf, pos):
    if wire_type == 0:  # varint
        _, pos = _varint_read(buf, pos)
        return pos
    if wire_type == 1:  # 64-bit
        return pos + 8
    if wire_type == 2:  # length-delimited
        l, pos = _varint_read(buf, pos)
        return pos + l
    if wire_type == 5:  # 32-bit
        return pos + 4
    raise ValueError(f"Unsupported wire type: {wire_type}")


def _parse_example_bytes(example_bytes):
    """
    Minimal protobuf parser for tf.train.Example that extracts bytes_list and int64_list
    for features used in this competition TFRecords:
      - image (bytes)
      - image_name (bytes)
      - target (int64) (train only)
    We avoid TensorFlow/protobuf dependency entirely.
    """
    want = {"image": None, "image_name": None, "target": None}
    buf = example_bytes
    pos = 0
    n = len(buf)

    while pos < n:
        tag, pos = _varint_read(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 7
        if field_num != 1 or wire_type != 2:
            pos = _skip_field(wire_type, buf, pos)
            continue

        flen, pos = _varint_read(buf, pos)
        features = buf[pos : pos + flen]
        pos += flen

        fpos = 0
        fn = len(features)
        while fpos < fn:
            ftag, fpos = _varint_read(features, fpos)
            f_field = ftag >> 3
            f_wire = ftag & 7
            if f_field != 1 or f_wire != 2:
                fpos = _skip_field(f_wire, features, fpos)
                continue

            elen, fpos = _varint_read(features, fpos)
            entry = features[fpos : fpos + elen]
            fpos += elen

            epos = 0
            en = len(entry)
            key = None
            feature_bytes = None
            while epos < en:
                etag, epos = _varint_read(entry, epos)
                e_field = etag >> 3
                e_wire = etag & 7
                if e_field == 1 and e_wire == 2:
                    klen, epos = _varint_read(entry, epos)
                    key = entry[epos : epos + klen].decode("utf-8", errors="ignore")
                    epos += klen
                elif e_field == 2 and e_wire == 2:
                    vlen, epos = _varint_read(entry, epos)
                    feature_bytes = entry[epos : epos + vlen]
                    epos += vlen
                else:
                    epos = _skip_field(e_wire, entry, epos)

            if key not in want or feature_bytes is None:
                continue

            vpos = 0
            vn = len(feature_bytes)
            value = None
            while vpos < vn:
                vtag, vpos = _varint_read(feature_bytes, vpos)
                v_field = vtag >> 3
                v_wire = vtag & 7
                if v_wire != 2:
                    vpos = _skip_field(v_wire, feature_bytes, vpos)
                    continue
                llen, vpos = _varint_read(feature_bytes, vpos)
                list_bytes = feature_bytes[vpos : vpos + llen]
                vpos += llen

                if v_field == 1:  # bytes_list
                    bpos = 0
                    bn = len(list_bytes)
                    vals = []
                    while bpos < bn:
                        btag, bpos = _varint_read(list_bytes, bpos)
                        b_field = btag >> 3
                        b_wire = btag & 7
                        if b_field == 1 and b_wire == 2:
                            blen, bpos = _varint_read(list_bytes, bpos)
                            vals.append(list_bytes[bpos : bpos + blen])
                            bpos += blen
                        else:
                            bpos = _skip_field(b_wire, list_bytes, bpos)
                    if vals:
                        value = vals[0]  # these features are single-valued
                    break

                if v_field == 3:  # int64_list
                    ipos = 0
                    in_ = len(list_bytes)
                    vals = []
                    while ipos < in_:
                        itag, ipos = _varint_read(list_bytes, ipos)
                        i_field = itag >> 3
                        i_wire = itag & 7
                        if i_field == 1 and i_wire == 0:
                            ival, ipos = _varint_read(list_bytes, ipos)
                            vals.append(int(ival))
                        else:
                            ipos = _skip_field(i_wire, list_bytes, ipos)
                    if vals:
                        value = vals[0]
                    break

            want[key] = value

    return want


def iter_tfrecord_examples(paths):
    """
    Iterate raw Example bytes from TFRecord files without TensorFlow.
    TFRecord format: uint64 length, uint32 masked_crc(length), data, uint32 masked_crc(data)
    """
    for path in paths:
        with open(path, "rb") as f:
            while True:
                header = f.read(12)
                if len(header) == 0:
                    break
                if len(header) < 12:
                    raise ValueError(f"Truncated TFRecord header in {path}")
                (length,) = struct.unpack("<Q", header[:8])
                data = f.read(length)
                if len(data) < length:
                    raise ValueError(f"Truncated TFRecord data in {path}")
                footer = f.read(4)  # data crc
                if len(footer) < 4:
                    raise ValueError(f"Truncated TFRecord footer in {path}")
                yield data


class CassavaLeafTFRecordIterableDataset(IterableDataset):
    """
    Streams TFRecords and yields (pixel_values, label_or_-1, image_id).

    TensorFlow-free implementation to avoid protobuf/TF import crash:
    - Parse tf.train.Example with a minimal protobuf decoder.
    - Decode JPEG via PIL, resize to 224x224 bilinear, scale to [0,1],
      apply ImageNet normalization, output CHW float32 torch tensor.
    """

    def __init__(
        self, tfrecord_files, has_label: bool, image_size=224, batch_size: int = 64
    ):
        super().__init__()
        self.tfrecord_files = list(tfrecord_files)
        self.has_label = bool(has_label)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)

    def __iter__(self):
        batch_imgs = []
        batch_labs = []
        batch_ids = []

        for ex_bytes in iter_tfrecord_examples(self.tfrecord_files):
            ex = _parse_example_bytes(ex_bytes)
            img_bytes = ex["image"]
            img_name = ex["image_name"]
            if img_bytes is None or img_name is None:
                continue

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize(
                (self.image_size, self.image_size), resample=Image.BILINEAR
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # normalize
            chw = np.transpose(arr, (2, 0, 1)).copy()

            if self.has_label:
                lab = int(ex["target"]) if ex["target"] is not None else -1
            else:
                lab = -1

            image_id = (
                img_name.decode("utf-8")
                if isinstance(img_name, (bytes, bytearray))
                else str(img_name)
            )

            batch_imgs.append(chw)
            batch_labs.append(lab)
            batch_ids.append(image_id)

            if len(batch_imgs) >= self.batch_size:
                x = torch.from_numpy(np.stack(batch_imgs, axis=0))
                y = torch.from_numpy(np.asarray(batch_labs, dtype=np.int64))
                ids = list(batch_ids)
                yield x, y, ids
                batch_imgs, batch_labs, batch_ids = [], [], []

        if batch_imgs:
            x = torch.from_numpy(np.stack(batch_imgs, axis=0))
            y = torch.from_numpy(np.asarray(batch_labs, dtype=np.int64))
            ids = list(batch_ids)
            yield x, y, ids




## === cell 3
class ViTForImageClassification(torch.nn.Module):
    """
    Keep interface (forward returns logits, loss) and overall semantics.
    Small CNN backbone -> 5-class logits.
    """

    def __init__(self, num_labels=5):
        super().__init__()
        self.num_labels = int(num_labels)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        x = self.features(pixel_values)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B,256)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss




## === cell 4
TRAIN_TFREC = sorted(
    glob.glob(
        "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/ld_train*.tfrec"
    )
)
if len(TRAIN_TFREC) == 0:
    TRAIN_TFREC = sorted(glob.glob("/kaggle/input/train_tfrecords/ld_train*.tfrec"))
if len(TRAIN_TFREC) == 0:
    raise FileNotFoundError("No train TFRecords found. Check the input path/pattern.")

TEST_TFREC = sorted(
    glob.glob(
        "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
    )
)
if len(TEST_TFREC) == 0:
    TEST_TFREC = sorted(glob.glob("/kaggle/input/test_tfrecords/ld_test*.tfrec"))
if len(TEST_TFREC) == 0:
    raise FileNotFoundError("No test TFRecords found. Check the input path/pattern.")

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = "/kaggle/input/train.csv"
train_df = pd.read_csv(train_csv_path)
n = len(train_df)

idx = np.arange(n)
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_frac = 0.1
n_val = int(n * val_frac)
val_rows = train_df.iloc[idx[:n_val]]
train_rows = train_df.iloc[idx[n_val:]]
val_ids = set(val_rows["image_id"].astype(str).tolist())
train_ids = set(train_rows["image_id"].astype(str).tolist())

print("Total train rows:", n, "Val rows:", len(val_ids), "Train rows:", len(train_ids))

BATCH_SIZE = 64


class SplitFilterIterable(IterableDataset):
    def __init__(self, base_iterable, want_val: bool, val_id_set):
        super().__init__()
        self.base = base_iterable
        self.want_val = bool(want_val)
        self.val_id_set = val_id_set

    def __iter__(self):
        want_val = self.want_val
        val_id_set = self.val_id_set
        for x_b, y_b, image_id_b in self.base:
            in_val = np.fromiter(
                (img_id in val_id_set for img_id in image_id_b),
                dtype=np.bool_,
                count=len(image_id_b),
            )
            keep_mask = in_val if want_val else ~in_val
            if not keep_mask.any():
                continue
            if keep_mask.all():
                yield x_b, y_b, image_id_b
            else:
                idx_t = torch.from_numpy(np.nonzero(keep_mask)[0]).to(dtype=torch.long)
                kept_ids = [image_id_b[i] for i in idx_t.tolist()]
                yield x_b.index_select(0, idx_t), y_b.index_select(0, idx_t), kept_ids


full_train_stream = CassavaLeafTFRecordIterableDataset(
    TRAIN_TFREC, has_label=True, image_size=224, batch_size=BATCH_SIZE
)
train_stream = SplitFilterIterable(
    full_train_stream, want_val=False, val_id_set=val_ids
)
val_stream = SplitFilterIterable(
    CassavaLeafTFRecordIterableDataset(
        TRAIN_TFREC, has_label=True, image_size=224, batch_size=BATCH_SIZE
    ),
    want_val=True,
    val_id_set=val_ids,
)

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

train_loader = DataLoader(
    train_stream,
    batch_size=None,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
    val_stream,
    batch_size=None,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification(num_labels=5).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


epochs = 4  # keep identical training length/semantics

for ep in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    tr_acc = 0.0
    tr_n = 0

    seen_train = 0
    max_train = len(train_ids)

    for pixel_values, labels, _ in train_loader:
        bs = labels.size(0)
        if seen_train + bs > max_train:
            keep = max_train - seen_train
            if keep <= 0:
                break
            pixel_values = pixel_values[:keep]
            labels = labels[:keep]
            bs = keep

        pixel_values = pixel_values.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(pixel_values, labels)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * bs
        tr_acc += accuracy_from_logits(logits.detach(), labels) * bs
        tr_n += bs
        seen_train += bs
        if seen_train >= max_train:
            break

    model.eval()
    va_loss = 0.0
    va_acc = 0.0
    va_n = 0

    seen_val = 0
    max_val = len(val_ids)
    with torch.no_grad():
        for pixel_values, labels, _ in val_loader:
            bs = labels.size(0)
            if seen_val + bs > max_val:
                keep = max_val - seen_val
                if keep <= 0:
                    break
                pixel_values = pixel_values[:keep]
                labels = labels[:keep]
                bs = keep

            pixel_values = pixel_values.to(
                device, dtype=torch.float32, non_blocking=True
            )
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            logits, loss = model(pixel_values, labels)
            va_loss += loss.item() * bs
            va_acc += accuracy_from_logits(logits, labels) * bs
            va_n += bs
            seen_val += bs
            if seen_val >= max_val:
                break

    print(
        f"Epoch {ep}/{epochs} "
        f"train_loss={tr_loss/max(1,tr_n):.4f} train_acc={tr_acc/max(1,tr_n):.4f} "
        f"val_loss={va_loss/max(1,va_n):.4f} val_acc={va_acc/max(1,va_n):.4f}"
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2436754316.py in <cell line: 0>()
     37         optimizer.zero_grad(set_to_none=True)
     38         logits, loss = model(pixel_values, labels)
---> 39         loss.backward()
     40         optimizer.step()
     41 

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    338 
    339     grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
--> 340     grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    341     if retain_graph is None:
    342         retain_graph = create_graph

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in _make_grads(outputs, grads, is_grads_batched)
    218                     assert isinstance(out, torch.Tensor)
    219                     new_grads.append(
--> 220                         torch.ones_like(out, memory_format=torch.preserve_format)
    221                     )
    222             else:

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## === cell 6
test_dataset = CassavaLeafTFRecordIterableDataset(
    TEST_TFREC, has_label=False, image_size=224, batch_size=64
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=None,
    shuffle=False,
    num_workers=0,  # avoid duplicate iteration over an iterable dataset
    pin_memory=torch.cuda.is_available(),
)

pred_map = {}  # image_id -> label (first occurrence kept for determinism)

model.eval()
with torch.no_grad():
    for pixel_values, _, ids in test_dataloader:
        pixel_values = pixel_values.to(device, dtype=torch.float32, non_blocking=True)
        logits, _ = model(pixel_values, None)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        for img_id, lab in zip(ids, preds):
            if img_id not in pred_map:
                pred_map[img_id] = int(lab)

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

fill_label = (
    int(pd.Series(list(pred_map.values())).mode().iloc[0]) if len(pred_map) else 0
)
labels_out = [
    pred_map.get(img_id, fill_label)
    for img_id in sample_sub["image_id"].astype(str).tolist()
]

submission_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].astype(str),
        "label": np.asarray(labels_out, dtype=int),
    }
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print(
    "Wrote:",
    out_path,
    "rows:",
    len(submission_df),
    "cols:",
    list(submission_df.columns),
)
sub = pd.read_csv(out_path)
print(sub.head())
print("shape:", sub.shape)
print("Unique labels:", sorted(sub["label"].unique().tolist()))
print("Nulls:", sub.isna().sum().to_dict())
assert len(submission_df) == len(
    sample_sub
), "Submission must match sample_submission length exactly."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns must be exactly: image_id,label"

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3864696288.py in <cell line: 0>()
     14 model.eval()
     15 with torch.no_grad():
---> 16     for pixel_values, _, ids in test_dataloader:
     17         pixel_values = pixel_values.to(device, dtype=torch.float32, non_blocking=True)
     18         logits, _ = model(pixel_values, None)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     96                 clone = copy.copy(data)  # type: ignore[arg-type]
     97                 for i, item in enumerate(data):
---> 98                     clone[i] = pin_memory(item, device)
     99                 return clone
    100             return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
