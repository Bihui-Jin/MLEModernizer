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

# 5. Target score

0.8443638561498942

# 6. Current score

0.58632

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the pipeline-breaking `FileNotFoundError`s by auto-detecting the available Kaggle input paths and falling back safely when the external tree-features CSV and the pretrained model files aren’t present. To preserve the original “stacked probs → decision tree” core logic, I keep the same feature construction (Keras probs + Torch probs) but make both model loaders robust; if a model can’t load, it outputs sensible probabilities/logits so inference and submission generation still work end-to-end. If the train-tree CSV isn’t available, I fit the same `DecisionTreeClassifier` on a small, deterministic subset of `train.csv` using the same combined-prob features computed by the (loaded or fallback) models. Finally, I ensure the submission matches `sample_submission.csv` order and is written as `submission.csv`.'
- What this solution (achieved 0.59118) has done: 'I fix the crash in the TensorFlow/Keras import that’s coming from a protobuf incompatibility (the `MessageFactory.GetPrototype` AttributeError) by making Keras usage robustly optional: if TensorFlow can’t import cleanly, we automatically fall back to a lightweight, deterministic “color-statistics → linear logits → softmax” probability generator instead of uniform 0.2. This keeps the core “(Keras-like 5 probs + Torch 5 logits) → DecisionTreeClassifier” stacking logic intact, but gives the tree much more informative features than the current uniform-prob fallback, which should move accuracy upward toward your target. I also harden the Torch forward pass to handle common saved-object formats (full model vs state_dict wrapper) without changing the overall approach. The rest of the pipeline (paths, decision tree training, submission formatting/order) is preserved and still writes `submission.csv`.'
- What this solution (achieved 0.59118) has done: 'I fix the TensorFlow/Keras import crash by making the protobuf “MessageFactory.GetPrototype” failure explicitly caught and ensuring the script cleanly falls back without stopping execution. I also correct a shape bug in the fallback training features matrix (it was allocated as 10 columns but filled with 10 concatenated values; the current code mistakenly used 10 while earlier versions sometimes used 8/9), and harden the Torch model loading so common checkpoint formats (state_dict) don’t break inference. These changes keep your core stacking logic intact (5 probs + 5 logits → DecisionTreeClassifier) while ensuring the pipeline runs end-to-end and produces a valid `submission.csv`. This should also improve score versus the current run because the Keras-failure path now execute as intended instead of crashing.'
- What this solution (achieved 0.59118) has done: 'I fix the runtime crash caused by the TensorFlow/protobuf incompatibility by isolating the TensorFlow import inside the probability function and catching the specific `MessageFactory.GetPrototype` failure so execution continues and the deterministic fallback is used. I also fix a shape bug in the fallback feature matrix (it must be 10 columns to hold 5 Keras-like probs + 5 Torch logits), which currently can raise a broadcast error once cell 1 stops crashing. Finally, I keep the stacking + decision tree logic unchanged but make the script robust to missing/unloadable Torch checkpoints and always write a correctly formatted `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.58894) has done: 'I fix the TensorFlow/protobuf crash by proactively disabling Keras in this environment (Python 3.13 commonly breaks TF import here) so the code always uses the existing deterministic fallback probability generator instead of crashing mid-feature-build. I also strengthen the exception handling around the Keras loader so that even unusual protobuf import errors are caught reliably and never abort the run. Finally, to move the score upward toward your target without changing the stacking+tree core logic, I replace the “zero torch logits” fallback with a deterministic, informative 5-logit generator derived from the same image statistics (so the tree gets meaningful 10-D features even when the torch checkpoint can’t load). All paths, the (5 probs + 5 logits) feature construction, the DecisionTreeClassifier usage, and submission formatting remain the same.'
- What this solution (achieved 0.59753) has done: 'We keep your “(5 probs + 5 logits) → DecisionTreeClassifier” stacking exactly the same, but make the features much more informative by actually using the TFRecords that are available in the dataset (instead of fragile external model files that currently trigger the fallback). Specifically, we replace the current PIL image-statistics fallback with a lightweight, deterministic “TFRecord-image → resize/normalize → simple conv-based features → linear logits/probs” generator that produces stable 5-class outputs and 5 logits per image without TensorFlow. We also train the decision tree on a slightly larger deterministic subset (still small enough to run fast) using the same feature construction, which should move the score upward toward your target while preserving the overall approach. Submission formatting/order and all paths remain the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.61211) has done: 'I keep your stacking pipeline unchanged (5 “Keras probs” + 5 “Torch logits” → DecisionTreeClassifier) but make the training of the decision tree less noisy and more representative by (1) using all TFRecord-available training examples (up to the full 18,721) instead of 7,000/9,000, and (2) using `train_label_map` from TFRecords when present so labels align with the exact images used for features. I also fix `_open_image_by_id` to actually fall back to disk for train/test when TFRecord bytes are missing (it currently returns `None` for disk paths), which improves feature completeness and reduces missing/uninformative rows. These are minimal, inference-safe changes that should move accuracy upward toward your target without changing model architecture, loss, or the core stacking logic. The script still write a valid `submission.csv` matching `sample_submission.csv` order.'
- What this solution (achieved 0.58632) has done: 'I keep your stacking pipeline exactly the same (5 “Keras-like” probs + 5 “Torch” logits → `DecisionTreeClassifier`) and only make small, score-relevant adjustments to reduce the current underfitting. Specifically, I (1) use a slightly stronger but still simple tree configuration (same model family) to better separate the 10-D stacked features, and (2) standardize the logits half of the feature vector using train-set statistics so the tree sees more comparable scales between probs (0–1) and logits (unbounded). I also ensure the same scaling is applied at test time and keep submission ordering identical to `sample_submission.csv`. These are minimal changes that should move accuracy upward toward your 0.844 target without changing the core approach.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from PIL import Image

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def _first_existing_dir(cands):
    for d in cands:
        if os.path.isdir(d):
            return d
    hits = glob.glob("/kaggle/**/cassava-leaf-disease-classification", recursive=True)
    for h in hits:
        if os.path.isdir(h):
            return h
    return None


DATA_DIR = _first_existing_dir(CANDIDATE_DATA_DIRS)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification data directory under /kaggle."
    )

TEST_DIR_CANDS = [
    os.path.join(DATA_DIR, "test_images"),
    os.path.join(DATA_DIR, "cassava-leaf-disease-classification", "test_images"),
]
TEST_DIR = None
for d in TEST_DIR_CANDS:
    if os.path.isdir(d):
        TEST_DIR = d
        break
if TEST_DIR is None:
    raise FileNotFoundError(f"Could not locate test_images under {DATA_DIR}")

SAMPLE_SUB_PATH_CANDS = [
    os.path.join(DATA_DIR, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
SAMPLE_SUB_PATH = None
for p in SAMPLE_SUB_PATH_CANDS:
    if os.path.exists(p):
        SAMPLE_SUB_PATH = p
        break
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected locations."
    )

TRAIN_CSV_CANDS = [
    os.path.join(DATA_DIR, "train.csv"),
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
TRAIN_CSV_PATH = None
for p in TRAIN_CSV_CANDS:
    if os.path.exists(p):
        TRAIN_CSV_PATH = p
        break
if TRAIN_CSV_PATH is None:
    raise FileNotFoundError("Could not locate train.csv in expected locations.")

TRAIN_TREE_PATH = "/kaggle/input/train-tree/train_tree_2.csv"
KERAS_MODEL_PATH = "/kaggle/input/densenet_70_512x512/keras/default/2/Densenet_70_512x512_weights (1).keras"
TORCH_MODEL_PATH = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

print("DATA_DIR:", DATA_DIR)
print("TEST_DIR:", TEST_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("CUDA:", torch.cuda.is_available())



## === cell 1
import struct
import zlib
import io
from typing import Dict, Tuple, Optional


def _masked_crc32c(data: bytes) -> int:
    x = zlib.crc32(data) & 0xFFFFFFFF  # not crc32c but we skip validation anyway
    return (((x >> 15) | (x << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _iter_tfrecord(path: str):
    with open(path, "rb") as f:
        while True:
            len_bytes = f.read(8)
            if not len_bytes:
                break
            (l,) = struct.unpack("<Q", len_bytes)
            f.read(4)  # len crc
            data = f.read(l)
            f.read(4)  # data crc
            if len(data) != l:
                break
            yield data


def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _read_len_delim(buf: bytes, i: int) -> Tuple[bytes, int]:
    l, i = _read_varint(buf, i)
    return buf[i : i + l], i + l


def _parse_example_get_fields(example_bytes: bytes) -> Dict[str, bytes]:
    i = 0
    features_msg = None
    n = len(example_bytes)
    while i < n:
        key, i = _read_varint(example_bytes, i)
        field = key >> 3
        wire = key & 7
        if field == 1 and wire == 2:
            features_msg, i = _read_len_delim(example_bytes, i)
            break
        if wire == 0:
            _, i = _read_varint(example_bytes, i)
        elif wire == 2:
            _, i = _read_len_delim(example_bytes, i)
        elif wire == 5:
            i += 4
        elif wire == 1:
            i += 8
        else:
            break
    if features_msg is None:
        return {}

    out = {}
    i = 0
    n = len(features_msg)
    while i < n:
        key, i = _read_varint(features_msg, i)
        field = key >> 3
        wire = key & 7
        if field != 1 or wire != 2:
            if wire == 0:
                _, i = _read_varint(features_msg, i)
            elif wire == 2:
                _, i = _read_len_delim(features_msg, i)
            elif wire == 5:
                i += 4
            elif wire == 1:
                i += 8
            else:
                break
            continue

        entry, i = _read_len_delim(features_msg, i)
        j = 0
        key_name = None
        value_msg = None
        m = len(entry)
        while j < m:
            k2, j = _read_varint(entry, j)
            f2 = k2 >> 3
            w2 = k2 & 7
            if f2 == 1 and w2 == 2:
                name_b, j = _read_len_delim(entry, j)
                key_name = name_b.decode("utf-8", errors="ignore")
            elif f2 == 2 and w2 == 2:
                value_msg, j = _read_len_delim(entry, j)
            else:
                if w2 == 0:
                    _, j = _read_varint(entry, j)
                elif w2 == 2:
                    _, j = _read_len_delim(entry, j)
                elif w2 == 5:
                    j += 4
                elif w2 == 1:
                    j += 8
                else:
                    break
        if key_name is not None and value_msg is not None:
            out[key_name] = value_msg
    return out


def _parse_feature_bytes_or_int(
    feature_msg: bytes,
) -> Tuple[Optional[bytes], Optional[int]]:
    i = 0
    n = len(feature_msg)
    bytes_value0 = None
    int_value0 = None
    while i < n:
        key, i = _read_varint(feature_msg, i)
        field = key >> 3
        wire = key & 7
        if wire != 2:
            if wire == 0:
                _, i = _read_varint(feature_msg, i)
            elif wire == 5:
                i += 4
            elif wire == 1:
                i += 8
            else:
                break
            continue

        submsg, i = _read_len_delim(feature_msg, i)

        if field == 1:
            j = 0
            m = len(submsg)
            while j < m:
                k2, j = _read_varint(submsg, j)
                f2 = k2 >> 3
                w2 = k2 & 7
                if f2 == 1 and w2 == 2:
                    b, j = _read_len_delim(submsg, j)
                    bytes_value0 = b
                    return bytes_value0, None
                else:
                    if w2 == 0:
                        _, j = _read_varint(submsg, j)
                    elif w2 == 2:
                        _, j = _read_len_delim(submsg, j)
                    elif w2 == 5:
                        j += 4
                    elif w2 == 1:
                        j += 8
                    else:
                        break
        elif field == 3:
            j = 0
            m = len(submsg)
            while j < m:
                k2, j = _read_varint(submsg, j)
                f2 = k2 >> 3
                w2 = k2 & 7
                if f2 == 1 and w2 == 0:
                    v, j = _read_varint(submsg, j)
                    int_value0 = int(v)
                    return None, int_value0
                else:
                    if w2 == 0:
                        _, j = _read_varint(submsg, j)
                    elif w2 == 2:
                        _, j = _read_len_delim(submsg, j)
                    elif w2 == 5:
                        j += 4
                    elif w2 == 1:
                        j += 8
                    else:
                        break
    return bytes_value0, int_value0


TEST_TFREC_DIR_CANDS = [
    os.path.join(DATA_DIR, "test_tfrecords"),
    os.path.join(DATA_DIR, "cassava-leaf-disease-classification", "test_tfrecords"),
]
TRAIN_TFREC_DIR_CANDS = [
    os.path.join(DATA_DIR, "train_tfrecords"),
    os.path.join(DATA_DIR, "cassava-leaf-disease-classification", "train_tfrecords"),
]
TEST_TFREC_DIR = next((d for d in TEST_TFREC_DIR_CANDS if os.path.isdir(d)), None)
TRAIN_TFREC_DIR = next((d for d in TRAIN_TFREC_DIR_CANDS if os.path.isdir(d)), None)

TEST_TFRECS = (
    sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))) if TEST_TFREC_DIR else []
)
TRAIN_TFRECS = (
    sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    if TRAIN_TFREC_DIR
    else []
)

print("TEST_TFRECS:", len(TEST_TFRECS), "from", TEST_TFREC_DIR)
print("TRAIN_TFRECS:", len(TRAIN_TFRECS), "from", TRAIN_TFREC_DIR)


def _softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(ex.sum())
    if not np.isfinite(s) or s <= 0:
        return np.full((len(x),), 1.0 / len(x), dtype=np.float32)
    return (ex / s).astype(np.float32)


def _img_to_small_tensor(image_pil, size=96) -> np.ndarray:
    img = image_pil.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x  # (H,W,3)


def _simple_conv_features(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=(0, 1))
    s = x.std(axis=(0, 1))
    gx = np.abs(x[:, 1:, :] - x[:, :-1, :]).mean(axis=(0, 1))
    gy = np.abs(x[1:, :, :] - x[:-1, :, :]).mean(axis=(0, 1))
    grad = 0.5 * (gx + gy)
    feats = np.concatenate([m, s, grad], axis=0).astype(np.float32)  # (9,)
    return feats


_FALLBACK_W = np.array(
    [
        [0.90, -0.25, 0.10, 0.05, -0.10, 0.08, 0.03, -0.02, 0.02],
        [-0.30, 0.85, -0.10, 0.10, 0.15, -0.05, -0.02, 0.03, 0.01],
        [0.10, -0.15, 0.75, -0.25, 0.05, 0.12, 0.02, -0.02, 0.03],
        [-0.10, 0.20, -0.30, 0.80, -0.05, 0.02, -0.03, 0.04, -0.01],
        [0.05, -0.10, 0.20, -0.15, 0.65, -0.08, 0.01, -0.01, 0.04],
    ],
    dtype=np.float32,
)
_FALLBACK_b = np.array([0.03, 0.00, 0.01, 0.00, 0.02], dtype=np.float32)

_FALLBACK_TORCH_W = np.array(
    [
        [0.65, -0.10, 0.06, 0.00, -0.05, 0.03, 0.01, -0.02, 0.00],
        [-0.20, 0.60, -0.05, 0.05, 0.10, -0.02, -0.01, 0.01, 0.00],
        [0.06, -0.06, 0.55, -0.15, 0.03, 0.05, 0.02, 0.00, 0.01],
        [-0.06, 0.12, -0.16, 0.60, -0.03, 0.00, -0.01, 0.02, -0.01],
        [0.03, -0.05, 0.10, -0.05, 0.50, -0.03, 0.00, -0.01, 0.02],
    ],
    dtype=np.float32,
)
_FALLBACK_TORCH_b = np.zeros((5,), dtype=np.float32)


def _fallback_probs_from_pil(image_pil) -> np.ndarray:
    x = _img_to_small_tensor(image_pil, size=96)
    feats = _simple_conv_features(x)
    logits = _FALLBACK_W @ feats + _FALLBACK_b
    return _softmax_np(logits)


def _fallback_torch_logits_from_pil(image_pil) -> np.ndarray:
    x = _img_to_small_tensor(image_pil, size=96)
    feats = _simple_conv_features(x)
    logits = _FALLBACK_TORCH_W @ feats + _FALLBACK_TORCH_b
    return logits.astype(np.float32)


FORCE_DISABLE_KERAS = True
use_keras = not FORCE_DISABLE_KERAS
model1 = None
keras_load_error = None
_tf = None


def _ensure_keras_loaded():
    global use_keras, model1, keras_load_error, _tf
    if (not use_keras) or (model1 is not None):
        return
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model  # noqa: F401

        _tf = tf
        if os.path.exists(KERAS_MODEL_PATH):
            model1 = load_model(KERAS_MODEL_PATH)
        else:
            raise FileNotFoundError(f"Keras model file not found: {KERAS_MODEL_PATH}")
    except Exception as e:
        use_keras = False
        keras_load_error = repr(e)


def get_keras_probs_pil(image_pil):
    _ensure_keras_loaded()
    if (not use_keras) or (model1 is None) or (_tf is None):
        return _fallback_probs_from_pil(image_pil)

    tf = _tf
    arr = np.array(image_pil, dtype=np.uint8)
    img = tf.convert_to_tensor(arr)
    img = tf.image.resize(img, [512, 512])
    img = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    std = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
    img = (img - mean) / std
    img = tf.expand_dims(img, axis=0)
    probs = model1.predict(img, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float32)
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0:
        return np.full((5,), 0.2, dtype=np.float32)
    return (probs / s).astype(np.float32)


model2 = None
use_torch = True
torch_load_error = None

try:
    if os.path.exists(TORCH_MODEL_PATH):
        loaded = torch.load(TORCH_MODEL_PATH, map_location=device)
        if isinstance(loaded, dict):
            if "model" in loaded:
                model2 = loaded["model"]
            elif "state_dict" in loaded:
                raise TypeError(
                    "Torch checkpoint contains only state_dict; model object not present."
                )
            else:
                raise TypeError("Unrecognized torch checkpoint dict format.")
        else:
            model2 = loaded
        if hasattr(model2, "to"):
            model2.to(device)
        if hasattr(model2, "eval"):
            model2.eval()
        else:
            raise TypeError("Loaded torch object is not a torch.nn.Module-like model.")
    else:
        raise FileNotFoundError(f"Torch model file not found: {TORCH_MODEL_PATH}")
except Exception as e:
    use_torch = False
    torch_load_error = repr(e)


def get_torch_logits_pil(image_pil):
    if (not use_torch) or (model2 is None):
        return _fallback_torch_logits_from_pil(image_pil)

    image2 = torch_transforms(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model2(image2)
    if isinstance(out, (tuple, list)):
        out = out[0]
    logits2 = out.detach().cpu().numpy()[0].astype(np.float32)
    if logits2.shape[0] != 5:
        logits2 = _fallback_torch_logits_from_pil(image_pil)
    return logits2


if not use_torch:
    print("WARNING: Torch model unavailable; using deterministic fallback logits.")
    print("Torch load error:", torch_load_error)
if (not use_keras) and (keras_load_error is not None):
    print("WARNING: Keras model unavailable; using deterministic fallback probs.")
    print("Keras load error:", keras_load_error)


def _build_tfrecord_image_map(tfrecs, need_labels=False, max_records=None):
    img_bytes = {}
    labels = {} if need_labels else None
    n_seen = 0
    for tfp in tfrecs:
        for rec in _iter_tfrecord(tfp):
            fields = _parse_example_get_fields(rec)
            if not fields:
                continue
            img_f = fields.get("image", None)
            id_f = fields.get("image_id", None)
            if img_f is None or id_f is None:
                continue
            b_img, _ = _parse_feature_bytes_or_int(img_f)
            b_id, _ = _parse_feature_bytes_or_int(id_f)
            if b_img is None or b_id is None:
                continue
            image_id = b_id.decode("utf-8", errors="ignore")
            img_bytes[image_id] = b_img
            if need_labels:
                lab_f = fields.get("label", None)
                if lab_f is not None:
                    _, lab = _parse_feature_bytes_or_int(lab_f)
                    if lab is not None:
                        labels[image_id] = int(lab)
            n_seen += 1
            if max_records is not None and n_seen >= max_records:
                return img_bytes, labels
    return img_bytes, labels


test_img_bytes_map = {}
if TEST_TFRECS:
    test_img_bytes_map, _ = _build_tfrecord_image_map(TEST_TFRECS, need_labels=False)
    print("Loaded test images from TFRecords:", len(test_img_bytes_map))
else:
    print(
        "TFRecords not found for test; will fall back to reading from test_images directory."
    )

train_img_bytes_map = {}
train_label_map = {}
if TRAIN_TFRECS:
    train_img_bytes_map, train_label_map = _build_tfrecord_image_map(
        TRAIN_TFRECS, need_labels=True, max_records=None
    )
    print(
        "Loaded train images from TFRecords:",
        len(train_img_bytes_map),
        "labels:",
        len(train_label_map),
    )
else:
    print(
        "TFRecords not found for train; will fall back to reading from train_images directory."
    )


TRAIN_IMG_DIR_CANDS = [
    os.path.join(DATA_DIR, "train_images"),
    os.path.join(DATA_DIR, "cassava-leaf-disease-classification", "train_images"),
]
TRAIN_IMG_DIR = next((d for d in TRAIN_IMG_DIR_CANDS if os.path.isdir(d)), None)
if TRAIN_IMG_DIR is None:
    raise FileNotFoundError(f"Could not locate train_images under {DATA_DIR}")


def _open_image_by_id(image_id: str, is_test: bool) -> Optional[Image.Image]:
    if is_test:
        if image_id in test_img_bytes_map:
            try:
                return Image.open(io.BytesIO(test_img_bytes_map[image_id])).convert(
                    "RGB"
                )
            except Exception:
                pass
        disk_path = os.path.join(TEST_DIR, image_id)
        if os.path.exists(disk_path):
            try:
                return Image.open(disk_path).convert("RGB")
            except Exception:
                return None
        return None
    else:
        if image_id in train_img_bytes_map:
            try:
                return Image.open(io.BytesIO(train_img_bytes_map[image_id])).convert(
                    "RGB"
                )
            except Exception:
                pass
        disk_path = os.path.join(TRAIN_IMG_DIR, image_id)
        if os.path.exists(disk_path):
            try:
                return Image.open(disk_path).convert("RGB")
            except Exception:
                return None
        return None




## === cell 2
LOGIT_MEAN = np.zeros((5,), dtype=np.float32)
LOGIT_STD = np.ones((5,), dtype=np.float32)


def _fit_logit_scaler_from_X(X: np.ndarray):
    global LOGIT_MEAN, LOGIT_STD
    logits = X[:, 5:10].astype(np.float32)
    m = logits.mean(axis=0)
    s = logits.std(axis=0)
    s = np.where(s < 1e-6, 1.0, s).astype(np.float32)
    LOGIT_MEAN = m.astype(np.float32)
    LOGIT_STD = s


def _apply_logit_scaler(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    X2 = X.copy()
    X2[:, 5:10] = (X2[:, 5:10] - LOGIT_MEAN) / LOGIT_STD
    return X2


decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=2,
    random_state=42,
)

train_labels = None
X_train_for_scaler = None

if os.path.exists(TRAIN_TREE_PATH):
    train_probs_df = pd.read_csv(TRAIN_TREE_PATH)
    if "label" not in train_probs_df.columns:
        raise ValueError("train_tree_2.csv must contain a 'label' column.")
    train_labels = train_probs_df["label"].values
    train_probs = train_probs_df.iloc[:, 1:-1].values.astype(np.float32)

    _fit_logit_scaler_from_X(train_probs)
    train_probs_scaled = _apply_logit_scaler(train_probs)

    decision_tree.fit(train_probs_scaled, train_labels)
    print(
        "Decision tree trained from external train_tree_2.csv. Features shape:",
        train_probs_scaled.shape,
    )
else:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    if not {"image_id", "label"}.issubset(train_df.columns):
        raise ValueError("train.csv must contain image_id and label.")

    if len(train_label_map) > 0:
        ids = []
        labs = []
        for k, v in train_label_map.items():
            if isinstance(k, str) and k.endswith(".jpg"):
                ids.append(k)
                labs.append(int(v))
        if len(ids) >= 1000:
            subset = pd.DataFrame({"image_id": ids, "label": labs})
        else:
            subset = train_df.copy()
    else:
        subset = train_df.copy()

    subset = subset.reset_index(drop=True)

    n_fit = len(subset)

    X = np.zeros((n_fit, 10), dtype=np.float32)
    y = subset["label"].astype(int).values

    for i, img_id in enumerate(subset["image_id"].values):
        img = _open_image_by_id(img_id, is_test=False)

        if img is None:
            p1 = np.full((5,), 0.2, dtype=np.float32)
            p2 = np.zeros((5,), dtype=np.float32)
        else:
            p1 = get_keras_probs_pil(img)
            p2 = get_torch_logits_pil(img)

        X[i, :] = np.concatenate((p1, p2), axis=0)

        if (i + 1) % 400 == 0 or (i + 1) == n_fit:
            print(f"Building fallback train features: {i+1}/{n_fit}", end="\r")

    print()

    _fit_logit_scaler_from_X(X)
    X_scaled = _apply_logit_scaler(X)

    decision_tree.fit(X_scaled, y)
    train_labels = y
    print(
        "Decision tree trained from fallback features. Features shape:", X_scaled.shape
    )



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image_id"].tolist()

combined_probs = np.zeros((len(test_images), 10), dtype=np.float32)
image_ids = []

length = len(test_images)
for idx, test_image in enumerate(test_images, start=1):
    image_ids.append(test_image)

    image = _open_image_by_id(test_image, is_test=True)

    if image is None:
        test_prediction_1 = np.full((5,), 0.2, dtype=np.float32)
        test_prediction_2 = np.zeros((5,), dtype=np.float32)
    else:
        test_prediction_1 = get_keras_probs_pil(image)
        test_prediction_2 = get_torch_logits_pil(image)

    combined_probs[idx - 1, :] = np.concatenate(
        (test_prediction_1, test_prediction_2), axis=0
    )

    if idx % 75 == 0 or idx == length:
        print(f"Count:{idx}/{length}", end="\r")

print()

combined_probs_scaled = _apply_logit_scaler(combined_probs)



## === cell 4
prediction = decision_tree.predict(combined_probs_scaled)

submission = pd.DataFrame(
    {
        "image_id": image_ids,
        "label": prediction.astype(int),
    }
)

submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")

if submission["label"].isna().any():
    majority = int(pd.Series(train_labels).value_counts().idxmax())
    submission["label"] = submission["label"].fillna(majority).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)

print("submission.csv written:", os.path.exists("submission.csv"))
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
print(submission.head())
print(submission.sample(5, random_state=42))
