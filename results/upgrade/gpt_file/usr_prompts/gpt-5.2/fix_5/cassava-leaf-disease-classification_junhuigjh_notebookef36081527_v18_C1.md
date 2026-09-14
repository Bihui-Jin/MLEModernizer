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

0.791779993955878

# 6. Current score

0.21188

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime crash caused by an incompatible `protobuf`/TensorFlow model loading path by avoiding `tf.keras.models.load_model()` for that external `.keras` artifact and instead using a built-in `tf.keras.applications.ResNet50` with ImageNet weights (same ResNet50 core family, so the overall “single-model + argmax” prediction logic remains the same). I also make the TFRecord iteration deterministic (sorted filenames) and parse both possible test TFRecord schemas (`image_name` vs `image_id`) so it runs across dataset variants. Finally, I ensure the submission uses the exact `sample_submission.csv` ordering to avoid any accidental misalignment and always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'You’re crashing before any submission is written due to an incompatibility between TensorFlow and the environment’s protobuf runtime (`MessageFactory.GetPrototype` AttributeError). The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation (which avoids that failing C++ path) before importing TensorFlow. After that, the rest of your pipeline can run unchanged, and it finally produce a valid `submission.csv` (so the score should move up from 0.0 toward the target simply by becoming a valid, non-empty submission). I’m also keeping TFRecord iteration deterministic and leaving your ResNet50/argmax logic intact.'
- What this solution (achieved 0.21973) has done: 'The crash happens before any submission is produced because TensorFlow can’t import cleanly in this environment due to a protobuf API mismatch; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone isn’t sufficient here. The minimal robust fix is to force-install compatible versions of `protobuf` and `tensorflow` at runtime (then restart imports within the same notebook run), which unblocks TFRecord reading and prediction and turns the 0.0 score (invalid/empty submission) into a real score. I also keep your core “ResNet50(ImageNet) + preprocess + argmax + sample_submission ordering” logic unchanged, but fix the label-space mismatch by mapping ImageNet’s 1000-class argmax into Cassava’s 5 classes deterministically so the submission labels are valid (0–4) and not mostly out-of-range. Finally, I make TFRecord parsing tolerant of both `image_name`/`image_id` keys and keep deterministic file iteration.'
- What this solution (achieved 0.21188) has done: 'I fix the TensorFlow import crash by avoiding the protobuf/TensorFlow runtime mismatch entirely, removing the “pip install different TF/protobuf mid-run” approach that can’t reliably work in Kaggle’s Python 3.13 environment. I keep your core inference logic (TFRecords → decode JPEG → ResNet50(ImageNet) → argmax → deterministic mapping to 0–4 → write submission in sample_submission order) intact, but implement it in PyTorch/torchvision which is available on Kaggle and stable here. I also make TFRecord parsing robust by supporting both `image_name` and `image_id` keys as you already intended, and ensure we always write a valid `submission.csv` with exactly the required columns. This should run end-to-end and typically improves over the current 0.21973 by producing consistent, non-degenerate predictions (still using the same “ImageNet → modulo 5” mapping semantics).'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_TFRECORD_DIR = f"{DATA_DIR}/test_tfrecords"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

print("Test tfrecords dir exists:", os.path.isdir(TEST_TFRECORD_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

weights = ResNet50_Weights.IMAGENET1K_V2
model2 = resnet50(weights=weights)
model2.eval()
model2.to(device)

preprocess = weights.transforms()

IMG_SIZE = (224, 224)


def imagenet_to_cassava_label(imagenet_class_idx: int) -> int:
    return int(imagenet_class_idx % 5)


print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Device:", device)



## === cell 2
from google.protobuf.message import DecodeError
from tensorflow.core.example import (
    example_pb2,
)  # protobuf-only module; does not require tensorflow runtime
from PIL import Image
import io

feature_keys = ("image", "image_name", "image_id")


def _get_bytes_feature(feat_map, key: str) -> bytes:
    if key not in feat_map:
        return b""
    f = feat_map[key]
    if f.bytes_list.value:
        return f.bytes_list.value[0]
    return b""


def parse_tfexample(record_bytes: bytes):
    ex = example_pb2.Example()
    try:
        ex.ParseFromString(record_bytes)
    except DecodeError:
        return None, None
    feats = ex.features.feature

    image_bytes = _get_bytes_feature(feats, "image")
    if not image_bytes:
        return None, None

    name = _get_bytes_feature(feats, "image_name")
    if not name:
        name = _get_bytes_feature(feats, "image_id")
    image_name = name.decode("utf-8") if name else ""

    return image_name, image_bytes


def second_model_preprocess_from_jpeg_bytes(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # add batch dim
    return x




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
import struct


def masked_crc32c(data: bytes) -> int:
    return 0


def tfrecord_iterator(path: str):
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) != 8:
                break
            (length,) = struct.unpack("<Q", header)
            f.read(4)  # length CRC (ignored)
            record = f.read(length)
            if len(record) != length:
                break
            f.read(4)  # data CRC (ignored)
            yield record


tfrecs = sorted([p for p in glob.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec"))])
print("Num tfrec files:", len(tfrecs))
print("First tfrec:", tfrecs[0] if tfrecs else None)

image_ids = []
prediction = []

with torch.inference_mode():
    for tfrec_path in tfrecs:
        for rec in tfrecord_iterator(tfrec_path):
            image_name, image_bytes = parse_tfexample(rec)
            if image_name is None or image_bytes is None:
                continue

            x = second_model_preprocess_from_jpeg_bytes(image_bytes).to(
                device, non_blocking=True
            )
            logits = model2(x)
            pred_imagenet = int(torch.argmax(logits, dim=1).item())
            pred = imagenet_to_cassava_label(pred_imagenet)

            image_ids.append(image_name)
            prediction.append(pred)

print("Predictions:", len(prediction), "Image IDs:", len(image_ids))
print("First 3:", list(zip(image_ids[:3], prediction[:3])))



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

pred_map = dict(zip(image_ids, prediction))
labels = [int(pred_map.get(img_id, 0)) for img_id in sample_sub["image_id"].tolist()]

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": labels})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Unique labels:", submission["label"].value_counts().to_dict())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
