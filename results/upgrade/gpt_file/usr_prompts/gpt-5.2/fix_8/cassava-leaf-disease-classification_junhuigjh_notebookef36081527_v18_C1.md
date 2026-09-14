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

0.791779993955878

# 6. Current score

0.41704

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime crash caused by an incompatible `protobuf`/TensorFlow model loading path by avoiding `tf.keras.models.load_model()` for that external `.keras` artifact and instead using a built-in `tf.keras.applications.ResNet50` with ImageNet weights (same ResNet50 core family, so the overall “single-model + argmax” prediction logic remains the same). I also make the TFRecord iteration deterministic (sorted filenames) and parse both possible test TFRecord schemas (`image_name` vs `image_id`) so it runs across dataset variants. Finally, I ensure the submission uses the exact `sample_submission.csv` ordering to avoid any accidental misalignment and always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'You’re crashing before any submission is written due to an incompatibility between TensorFlow and the environment’s protobuf runtime (`MessageFactory.GetPrototype` AttributeError). The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation (which avoids that failing C++ path) before importing TensorFlow. After that, the rest of your pipeline can run unchanged, and it finally produce a valid `submission.csv` (so the score should move up from 0.0 toward the target simply by becoming a valid, non-empty submission). I’m also keeping TFRecord iteration deterministic and leaving your ResNet50/argmax logic intact.'
- What this solution (achieved 0.21973) has done: 'The crash happens before any submission is produced because TensorFlow can’t import cleanly in this environment due to a protobuf API mismatch; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone isn’t sufficient here. The minimal robust fix is to force-install compatible versions of `protobuf` and `tensorflow` at runtime (then restart imports within the same notebook run), which unblocks TFRecord reading and prediction and turns the 0.0 score (invalid/empty submission) into a real score. I also keep your core “ResNet50(ImageNet) + preprocess + argmax + sample_submission ordering” logic unchanged, but fix the label-space mismatch by mapping ImageNet’s 1000-class argmax into Cassava’s 5 classes deterministically so the submission labels are valid (0–4) and not mostly out-of-range. Finally, I make TFRecord parsing tolerant of both `image_name`/`image_id` keys and keep deterministic file iteration.'
- What this solution (achieved 0.21188) has done: 'I fix the TensorFlow import crash by avoiding the protobuf/TensorFlow runtime mismatch entirely, removing the “pip install different TF/protobuf mid-run” approach that can’t reliably work in Kaggle’s Python 3.13 environment. I keep your core inference logic (TFRecords → decode JPEG → ResNet50(ImageNet) → argmax → deterministic mapping to 0–4 → write submission in sample_submission order) intact, but implement it in PyTorch/torchvision which is available on Kaggle and stable here. I also make TFRecord parsing robust by supporting both `image_name` and `image_id` keys as you already intended, and ensure we always write a valid `submission.csv` with exactly the required columns. This should run end-to-end and typically improves over the current 0.21973 by producing consistent, non-degenerate predictions (still using the same “ImageNet → modulo 5” mapping semantics).'
- What this solution (achieved 0.21188) has done: 'I fix the crash in TFRecord parsing by removing the protobuf-based `tensorflow.core.example` dependency that’s incompatible with this environment’s protobuf runtime, and replace it with a small, pure-Python TFRecord+Example parser that reads the minimal fields we need (`image` + `image_id`/`image_name`). This keeps your core pipeline intact: TFRecords → decode JPEG → ResNet50(ImageNet) → argmax → deterministic modulo-5 mapping → write `submission.csv` in `sample_submission.csv` order. I also add a safe fallback to load test images directly from `test_images/` if TFRecords aren’t found or parsing yields too few predictions, ensuring a valid submission is always produced. These changes are execution/stability fixes and should improve score versus the current low accuracy driven by missing/failed TFRecord decoding.'
- What this solution (achieved 0.61099) has done: 'Your current low score is mainly because the model is an ImageNet classifier and the `imagenet % 5` mapping is essentially arbitrary for Cassava labels, so we need a minimal but legitimate way to align predictions to the Cassava label space without changing the core “single ResNet50 + argmax” inference semantics. I keep your exact backbone (ResNet50 ImageNet weights) and the same prediction loop, but replace the modulo mapping with a deterministic, data-driven mapping learned from `train.csv` by running the same model over a small capped subset of training images and building an argmax→cassava lookup table. This is a minimal post-processing change (still `argmax` of the same logits, still single-pass inference), and it should move accuracy substantially toward your target. I also keep the sample_submission ordering and the TFRecord/image fallback logic unchanged, and ensure the script always writes a valid `submission.csv`.'
- What this solution (achieved 0.41704) has done: 'Your current gap to the target is large (0.61099 → 0.79178), so we need a real but still minimal accuracy gain without changing the core “ResNet50(ImageNet) + argmax + deterministic mapping” pipeline. The most direct improvement is to learn the ImageNet→Cassava lookup more reliably: use a stratified subset of training images (instead of purely random sampling) and increase the cap moderately so every Cassava class contributes, which usually improves the stability/quality of the mapping. I also compute the lookup in a more memory-safe way (no change in semantics) and keep inference/submission ordering exactly the same. These changes keep the same model, the same preprocessing, the same argmax-based prediction loop, and only strengthen the deterministic post-processing mapping to move accuracy toward your target.'

# 9. Code solution

## === cell 0
import os
import glob
import io
import struct
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_TFRECORD_DIR = f"{DATA_DIR}/test_tfrecords"
TEST_IMAGE_DIR = f"{DATA_DIR}/test_images"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TRAIN_IMAGE_DIR = f"{DATA_DIR}/train_images"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

print("Test tfrecords dir exists:", os.path.isdir(TEST_TFRECORD_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMAGE_DIR))
print("Train csv exists:", os.path.exists(TRAIN_CSV_PATH))
print("Train images dir exists:", os.path.isdir(TRAIN_IMAGE_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

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

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Device:", device)




## === cell 2
def _read_varint(buf: bytes, pos: int):
    shift = 0
    result = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 64:
            raise ValueError("Varint too long")


def _skip_field(wire_type: int, buf: bytes, pos: int):
    if wire_type == 0:  # varint
        _, pos = _read_varint(buf, pos)
        return pos
    if wire_type == 1:  # 64-bit
        return pos + 8
    if wire_type == 2:  # length-delimited
        ln, pos = _read_varint(buf, pos)
        return pos + ln
    if wire_type == 5:  # 32-bit
        return pos + 4
    raise ValueError(f"Unsupported wire type: {wire_type}")


def _parse_length_delimited_field(buf: bytes, pos: int):
    ln, pos = _read_varint(buf, pos)
    return buf[pos : pos + ln], pos + ln


def _parse_bytes_list_value(feature_msg: bytes):
    pos = 0
    out = []
    while pos < len(feature_msg):
        key, pos = _read_varint(feature_msg, pos)
        field_num = key >> 3
        wire_type = key & 7
        if field_num == 1 and wire_type == 2:  # bytes_list message
            bl_msg, pos = _parse_length_delimited_field(feature_msg, pos)
            p2 = 0
            while p2 < len(bl_msg):
                k2, p2 = _read_varint(bl_msg, p2)
                fn2 = k2 >> 3
                wt2 = k2 & 7
                if fn2 == 1 and wt2 == 2:
                    b, p2 = _parse_length_delimited_field(bl_msg, p2)
                    out.append(b)
                else:
                    p2 = _skip_field(wt2, bl_msg, p2)
        else:
            pos = _skip_field(wire_type, feature_msg, pos)
    return out


def _parse_feature_entry(entry_msg: bytes):
    pos = 0
    key_s = None
    feat_msg = None
    while pos < len(entry_msg):
        k, pos = _read_varint(entry_msg, pos)
        fn = k >> 3
        wt = k & 7
        if fn == 1 and wt == 2:
            kb, pos = _parse_length_delimited_field(entry_msg, pos)
            key_s = kb.decode("utf-8", errors="ignore")
        elif fn == 2 and wt == 2:
            feat_msg, pos = _parse_length_delimited_field(entry_msg, pos)
        else:
            pos = _skip_field(wt, entry_msg, pos)
    return key_s, feat_msg


def parse_tfexample(record_bytes: bytes):
    pos = 0
    features_msg = None
    while pos < len(record_bytes):
        k, pos = _read_varint(record_bytes, pos)
        fn = k >> 3
        wt = k & 7
        if fn == 1 and wt == 2:
            features_msg, pos = _parse_length_delimited_field(record_bytes, pos)
        else:
            pos = _skip_field(wt, record_bytes, pos)

    if not features_msg:
        return None, None

    feats = {}
    p = 0
    while p < len(features_msg):
        k, p = _read_varint(features_msg, p)
        fn = k >> 3
        wt = k & 7
        if fn == 1 and wt == 2:
            entry, p = _parse_length_delimited_field(features_msg, p)
            ek, ev = _parse_feature_entry(entry)
            if ek and ev is not None:
                feats[ek] = ev
        else:
            p = _skip_field(wt, features_msg, p)

    img_feat = feats.get("image")
    if img_feat is None:
        return None, None
    img_vals = _parse_bytes_list_value(img_feat)
    if not img_vals:
        return None, None
    image_bytes = img_vals[0]

    name_feat = feats.get("image_name") or feats.get("image_id")
    image_name = ""
    if name_feat is not None:
        name_vals = _parse_bytes_list_value(name_feat)
        if name_vals:
            image_name = name_vals[0].decode("utf-8", errors="ignore")

    return image_name, image_bytes


def second_model_preprocess_from_jpeg_bytes(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # add batch dim
    return x


def second_model_preprocess_from_path(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    return x




## === cell 3
def build_imagenet_to_cassava_lookup(
    train_csv_path: str,
    train_image_dir: str,
    max_images: int = 8000,  # increased from 2500 to reduce mapping noise (still fits time budget)
    batch_size: int = 64,
    seed: int = 42,
):
    df = pd.read_csv(train_csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError("train.csv must contain image_id and label columns")
    df = df.dropna(subset=["image_id", "label"]).copy()
    df["label"] = df["label"].astype(int)

    rng = np.random.default_rng(seed)
    per_class = max(1, max_images // 5)
    selected_idx = []
    for y in range(5):
        idx_y = df.index[df["label"] == y].to_numpy()
        if len(idx_y) == 0:
            continue
        rng.shuffle(idx_y)
        take = min(per_class, len(idx_y))
        selected_idx.append(idx_y[:take])
    if len(selected_idx) == 0:
        return np.arange(1000, dtype=np.int16) % 5
    idx = np.concatenate(selected_idx, axis=0)
    rng.shuffle(idx)
    df_sub = df.loc[idx].reset_index(drop=True)

    counts = np.zeros((1000, 5), dtype=np.int32)

    paths, labels = [], []
    for img_id, lab in zip(df_sub["image_id"].tolist(), df_sub["label"].tolist()):
        pth = os.path.join(train_image_dir, img_id)
        if os.path.exists(pth):
            paths.append(pth)
            labels.append(int(lab))

    if len(paths) == 0:
        return np.arange(1000, dtype=np.int16) % 5

    model2.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]

            xs = [second_model_preprocess_from_path(p) for p in batch_paths]
            x = torch.cat(xs, dim=0).to(device, non_blocking=True)

            logits = model2(x)
            pred_imagenet = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
            )

            for c1000, y5 in zip(pred_imagenet, batch_labels):
                if 0 <= c1000 < 1000 and 0 <= y5 < 5:
                    counts[c1000, y5] += 1

    global_majority = int(
        np.argmax(np.bincount(np.array(labels, dtype=int), minlength=5))
    )
    lookup = np.full((1000,), global_majority, dtype=np.int16)
    observed = counts.sum(axis=1) > 0
    lookup[observed] = np.argmax(counts[observed], axis=1).astype(np.int16)

    uniq, cnt = np.unique(lookup, return_counts=True)
    print("Lookup built from images:", len(paths))
    print(
        "Learned imagenet->cassava lookup label distribution:",
        dict(zip(uniq.tolist(), cnt.tolist())),
    )
    print("Global majority (fallback label):", global_majority)
    return lookup


lookup_1000_to_5 = build_imagenet_to_cassava_lookup(
    TRAIN_CSV_PATH, TRAIN_IMAGE_DIR, max_images=8000, batch_size=64, seed=42
)


def imagenet_to_cassava_label(imagenet_class_idx: int) -> int:
    if 0 <= imagenet_class_idx < 1000:
        return int(lookup_1000_to_5[imagenet_class_idx])
    return int(0)




## === cell 4
def tfrecord_iterator(path: str):
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) != 8:
                break
            (length,) = struct.unpack("<Q", header)
            _ = f.read(4)  # length CRC (ignored)
            record = f.read(length)
            if len(record) != length:
                break
            _ = f.read(4)  # data CRC (ignored)
            yield record


tfrecs = sorted(glob.glob(os.path.join(TEST_TFRECORD_DIR, "*.tfrec")))
print("Num tfrec files:", len(tfrecs))
print("First tfrec:", tfrecs[0] if tfrecs else None)

image_ids = []
prediction = []

with torch.inference_mode():
    if tfrecs:
        for tfrec_path in tfrecs:
            for rec in tfrecord_iterator(tfrec_path):
                image_name, image_bytes = parse_tfexample(rec)
                if image_bytes is None:
                    continue
                x = second_model_preprocess_from_jpeg_bytes(image_bytes).to(
                    device, non_blocking=True
                )
                logits = model2(x)
                pred_imagenet = int(torch.argmax(logits, dim=1).item())
                pred = imagenet_to_cassava_label(pred_imagenet)
                if image_name:
                    image_ids.append(image_name)
                    prediction.append(pred)

print("Predictions from TFRecords:", len(prediction), "Image IDs:", len(image_ids))
print("First 3:", list(zip(image_ids[:3], prediction[:3])))

if len(prediction) < 0.9 * 2676 and os.path.isdir(TEST_IMAGE_DIR):
    print(
        "TFRecord parsing incomplete; running fallback inference from test_images/ ..."
    )
    test_paths = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, "*.jpg")))
    image_ids = []
    prediction = []
    with torch.inference_mode():
        for pth in test_paths:
            img_id = os.path.basename(pth)
            x = second_model_preprocess_from_path(pth).to(device, non_blocking=True)
            logits = model2(x)
            pred_imagenet = int(torch.argmax(logits, dim=1).item())
            pred = imagenet_to_cassava_label(pred_imagenet)
            image_ids.append(img_id)
            prediction.append(pred)
    print("Predictions from test_images:", len(prediction))



## === cell 5
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
