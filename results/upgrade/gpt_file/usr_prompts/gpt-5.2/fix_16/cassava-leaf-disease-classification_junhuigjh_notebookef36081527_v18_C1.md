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

0.62145

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
- What this solution (achieved 0.52018) has done: 'Your current score (0.417) is far below the target (0.792), so we need a legitimate accuracy lift while keeping the same core pipeline: ResNet50(ImageNet) → argmax → learned deterministic ImageNet→Cassava lookup → submission in sample order. The smallest impactful change is to build the lookup table more reliably by using many more (but still time-safe) training images and doing the mapping accumulation in larger batches to reduce noise; this keeps identical semantics but improves the post-processing calibration. I’m also making the lookup computation deterministic and slightly more robust by ensuring each class contributes evenly and by using pinned-memory DataLoader-like batching (without changing the model or transforms). Everything else (TFRecord parsing, fallback to test_images, argmax inference, submission formatting) stays the same.'
- What this solution (achieved 0.6151) has done: 'Your current score (0.52018) is far below the target (0.79178), so we should improve accuracy while keeping your core pipeline unchanged: ResNet50(ImageNet) → argmax(1000) → learned deterministic 1000→5 lookup → submission in sample order. The smallest high-impact improvement is to make the lookup table less noisy by (1) using all available training images (still within time by batching) instead of a capped subset and (2) aggregating predictions in batches for both lookup-building and test inference (same semantics, just vectorized). I also add a tiny deterministic tie-break in the lookup (falls back to global majority when a class has a tie), which improves stability without changing the model or metric logic. Everything else—model, transforms, argmax prediction semantics, and submission formatting—stays the same.'
- What this solution (achieved 0.61809) has done: 'Your current gap to the target is still large, so the safest way to move accuracy upward without changing the model or training loop is to make the learned ImageNet→Cassava lookup less biased/noisy. I keep the exact same ResNet50(ImageNet) + preprocess + argmax(1000) semantics, but (1) build the lookup from **all** training images **without forcing equal-per-class sampling** (so the mapping reflects the true label distribution and avoids over-weighting rare classes), and (2) apply a small, deterministic Laplace smoothing + a “reject-to-global-majority” rule for weakly-supported ImageNet classes to reduce brittle mappings. Everything else (TFRecord parsing, batching, fallback to `test_images/`, submission ordering/format) stays the same.'
- What this solution (achieved 0.61435) has done: 'The timeout is dominated by repeated full-pass ResNet50 inference over the entire train split for every parameter-grid setting (6×), plus Python-level counting loops. I keep the exact mapping logic and grid search, but cache the expensive ImageNet argmax predictions for the train/val images once and reuse them across all grid points, making the grid search almost free. I also vectorize the count accumulation (using `np.add.at`) and the final mapping from ImageNet classes to cassava labels (using array indexing instead of Python loops), and enable faster dataloading/inference paths (channels-last, cuDNN benchmark, pinned memory) without changing the model or predictions’ semantics.'
- What this solution (achieved 0.61958) has done: 'Your current score (0.61435) is well below the target (0.79178), so we should improve accuracy while preserving your core pipeline (ResNet50(ImageNet) → argmax(1000) → deterministic 1000→5 lookup → argmax mapping on test). The biggest low-risk gain is to learn the 1000→5 lookup **using richer statistics than argmax only**, while still keeping inference as argmax: accumulate a small top‑k “vote” distribution from the same ResNet50 logits on train images, then choose the Cassava label per ImageNet class by aggregated votes (with the same Laplace/min_support/min_margin logic). This does not change the model, transforms, or prediction semantics on test (still argmax), but it makes the lookup less noisy and typically increases accuracy. I keep your caching/efficiency behavior and submission ordering unchanged, and ensure runtime stays within limits by batching and using top‑k=5.'
- What this solution (achieved 0.62145) has done: 'Your current pipeline is already stable and the main limitation is the noisiness of the learned ImageNet→Cassava lookup because it only uses train images and a small top‑k vote scheme. To move accuracy upward toward the 0.792 target without changing the model or inference semantics, I (1) fix a subtle but important bug in the top‑k vote accumulation (the `np.add.at` indexing currently votes into the wrong axis), and (2) learn the lookup with a slightly richer but still minimal/legit statistic: accumulate **softmax probability mass** for top‑k ImageNet classes per train image (still using the same ResNet50 logits), then derive the same deterministic lookup. Everything else (ResNet50 ImageNet backbone, preprocessing, argmax on test, submission ordering) stays the same, and runtime remains within limits via batching.'
- What this solution (achieved 0.62145) has done: 'Your current gap to the target is large, so we should make a small but high-impact correctness fix rather than tuning. The main issue is in `_accumulate_topk_votes_counts_for_paths`: it currently double-counts votes and also uses `np.add.at` with swapped axes, which corrupts the learned ImageNet→Cassava lookup; even if you’re currently using the prob-mass version, keeping the vote accumulator correct avoids accidental regressions and makes the mapping logic consistent. I also make the prob-mass accumulation slightly more stable (no semantic change) by doing the addition via `np.add.at` in a correctly vectorized way (same values, less Python-loop overhead), which helps ensure we fully finish within the time limit and thus reliably realize the intended mapping quality. Everything else (model, preprocess, argmax inference, lookup-building and submission ordering) stays the same.'

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

torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

weights = ResNet50_Weights.IMAGENET1K_V2
model2 = resnet50(weights=weights)
model2.eval()
model2.to(device)
model2 = model2.to(memory_format=torch.channels_last)

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


def second_model_preprocess_batch_from_paths(image_paths):
    imgs = []
    for p in image_paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    xs = [preprocess(im) for im in imgs]
    return torch.stack(xs, dim=0)


def second_model_preprocess_batch_from_jpeg_bytes_list(image_bytes_list):
    imgs = []
    for b in image_bytes_list:
        with Image.open(io.BytesIO(b)) as im:
            imgs.append(im.convert("RGB"))
    xs = [preprocess(im) for im in imgs]
    return torch.stack(xs, dim=0)




## === cell 3
def _stable_stratified_split(df, label_col="label", val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for lab, g in df.groupby(label_col):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(
        drop=True
    )


def _predict_imagenet_argmax_for_paths(paths, batch_size=128):
    preds = []
    model2.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            x = second_model_preprocess_batch_from_paths(batch_paths)
            x = x.contiguous(memory_format=torch.channels_last)
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
            logits = model2(x)
            pred_imagenet = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int32)
            )
            preds.append(pred_imagenet)
    if not preds:
        return np.array([], dtype=np.int32)
    return np.concatenate(preds, axis=0).astype(np.int32)


def _accumulate_topk_votes_counts_for_paths(paths, labels_arr, batch_size=128, topk=5):
    counts = np.zeros((1000, 5), dtype=np.int32)
    if len(paths) == 0:
        return counts

    model2.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            yb = labels_arr[start : start + batch_size].astype(np.int32, copy=False)

            x = second_model_preprocess_batch_from_paths(batch_paths)
            x = x.contiguous(memory_format=torch.channels_last)
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)

            logits = model2(x)
            k = int(min(int(topk), logits.shape[1]))
            top_idx = (
                torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
                .indices.detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

            weights = np.arange(k, 0, -1, dtype=np.int32)  # e.g., [5,4,3,2,1]
            for i in range(top_idx.shape[0]):
                yi = int(yb[i])
                if 0 <= yi < 5:
                    np.add.at(counts[:, yi], top_idx[i], weights)

    return counts


def _accumulate_topk_probmass_counts_for_paths(
    paths, labels_arr, batch_size=128, topk=5
):
    counts = np.zeros((1000, 5), dtype=np.float32)
    if len(paths) == 0:
        return counts

    model2.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            yb = labels_arr[start : start + batch_size].astype(np.int32, copy=False)

            x = second_model_preprocess_batch_from_paths(batch_paths)
            x = x.contiguous(memory_format=torch.channels_last)
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)

            logits = model2(x)
            probs = torch.softmax(logits, dim=1)
            k = int(min(int(topk), probs.shape[1]))
            top = torch.topk(probs, k=k, dim=1, largest=True, sorted=False)
            top_idx = top.indices.detach().cpu().numpy().astype(np.int32)
            top_val = top.values.detach().cpu().numpy().astype(np.float32)

            valid = (yb >= 0) & (yb < 5)
            if not np.any(valid):
                continue
            top_idx_v = top_idx[valid]
            top_val_v = top_val[valid]
            yb_v = yb[valid]

            ii = top_idx_v.reshape(-1)
            jj = np.repeat(yb_v, top_idx_v.shape[1]).astype(np.int32, copy=False)
            vv = top_val_v.reshape(-1)
            np.add.at(counts, (ii, jj), vv)

    return counts


def build_imagenet_to_cassava_lookup_from_counts(
    counts: np.ndarray,
    labels_arr: np.ndarray,
    laplace: float = 1.0,
    min_support: float = 3.0,
    min_margin: float = 0.05,
):
    if counts is None or counts.size == 0:
        lookup = np.arange(1000, dtype=np.int16) % 5
        global_majority = 0
        return lookup, global_majority

    global_majority = int(np.argmax(np.bincount(labels_arr, minlength=5)))

    lookup = np.full((1000,), global_majority, dtype=np.int16)

    support = counts.sum(axis=1)
    eligible = np.where(support >= float(min_support))[0]
    if eligible.size:
        sm = counts[eligible].astype(np.float64) + float(laplace)
        probs = sm / sm.sum(axis=1, keepdims=True)
        best = np.argmax(probs, axis=1).astype(np.int32)
        part = np.partition(probs, -2, axis=1)[:, -2:]
        margin = (part[:, 1] - part[:, 0]).astype(np.float64)
        use = margin >= float(min_margin)
        lookup[eligible[use]] = best[use].astype(np.int16)

    return lookup, global_majority


def score_lookup_on_preds(
    pred_imagenet: np.ndarray,
    y_true: np.ndarray,
    lookup: np.ndarray,
    global_majority: int,
):
    if pred_imagenet.size == 0:
        return 0.0, 0
    y_pred = np.full_like(y_true, global_majority)
    ok = (pred_imagenet >= 0) & (pred_imagenet < 1000)
    y_pred[ok] = lookup[pred_imagenet[ok]].astype(np.int32)
    acc = float((y_pred == y_true).mean())
    return acc, int(y_true.size)


def _existing_paths_and_labels(df: pd.DataFrame, image_dir: str):
    df = df.dropna(subset=["image_id", "label"]).copy()
    df["label"] = df["label"].astype(int)
    ids = df["image_id"].tolist()
    labs = df["label"].to_numpy(dtype=np.int32, copy=True)

    paths = [os.path.join(image_dir, img_id) for img_id in ids]
    exists = np.fromiter(
        (os.path.exists(p) for p in paths), dtype=np.bool_, count=len(paths)
    )
    paths_ok = [p for p, e in zip(paths, exists.tolist()) if e]
    labs_ok = labs[exists]
    return paths_ok, labs_ok


df_all = pd.read_csv(TRAIN_CSV_PATH)
df_all = df_all.dropna(subset=["image_id", "label"]).copy()
df_all["label"] = df_all["label"].astype(int)

df_tr, df_va = _stable_stratified_split(
    df_all, label_col="label", val_frac=0.1, seed=42
)
print("Train/val sizes:", len(df_tr), len(df_va))

tr_paths, tr_y = _existing_paths_and_labels(df_tr, TRAIN_IMAGE_DIR)
va_paths, va_y = _existing_paths_and_labels(df_va, TRAIN_IMAGE_DIR)

print("Existing train images:", len(tr_paths), "Existing val images:", len(va_paths))

pred_va_imagenet = _predict_imagenet_argmax_for_paths(va_paths, batch_size=128)

tr_counts_topk = _accumulate_topk_probmass_counts_for_paths(
    tr_paths, tr_y, batch_size=128, topk=5
)

param_grid = [
    {"laplace": 0.0, "min_support": 2.0, "min_margin": 0.00},
    {"laplace": 1.0, "min_support": 2.0, "min_margin": 0.00},
    {"laplace": 1.0, "min_support": 3.0, "min_margin": 0.03},
    {"laplace": 1.0, "min_support": 3.0, "min_margin": 0.05},
    {"laplace": 1.0, "min_support": 5.0, "min_margin": 0.03},
    {"laplace": 2.0, "min_support": 3.0, "min_margin": 0.05},
]

best = None
best_acc = -1.0
best_stats = None

for params in param_grid:
    lookup_tmp, gm_tmp = build_imagenet_to_cassava_lookup_from_counts(
        tr_counts_topk,
        tr_y,
        laplace=float(params["laplace"]),
        min_support=float(params["min_support"]),
        min_margin=float(params["min_margin"]),
    )
    acc, n_eval = score_lookup_on_preds(pred_va_imagenet, va_y, lookup_tmp, gm_tmp)
    print(
        "Val acc:",
        round(acc, 5),
        "n_eval:",
        n_eval,
        "params:",
        params,
        "global_majority:",
        gm_tmp,
    )
    if acc > best_acc:
        best_acc = acc
        best = params
        best_stats = (gm_tmp, lookup_tmp)

print("Selected params (by val acc):", best, "val acc:", round(best_acc, 5))

all_paths, all_y = _existing_paths_and_labels(df_all, TRAIN_IMAGE_DIR)

all_counts_topk = _accumulate_topk_probmass_counts_for_paths(
    all_paths, all_y, batch_size=128, topk=5
)

lookup_1000_to_5, global_majority_final = build_imagenet_to_cassava_lookup_from_counts(
    all_counts_topk,
    all_y,
    laplace=float(best["laplace"]),
    min_support=float(best["min_support"]),
    min_margin=float(best["min_margin"]),
)

uniq, cnt = np.unique(lookup_1000_to_5, return_counts=True)
print("Lookup built from ALL train images:", len(df_all))
print(
    "Learned imagenet->cassava lookup label distribution:",
    dict(zip(uniq.tolist(), cnt.tolist())),
)
print("Global majority (fallback label):", global_majority_final)
print("Final lookup params:", best)


def imagenet_to_cassava_label(imagenet_class_idx: int) -> int:
    if 0 <= imagenet_class_idx < 1000:
        return int(lookup_1000_to_5[imagenet_class_idx])
    return int(global_majority_final)




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

BATCH_SIZE_TEST = 128

with torch.inference_mode():
    if tfrecs:
        batch_names = []
        batch_bytes = []
        for tfrec_path in tfrecs:
            for rec in tfrecord_iterator(tfrec_path):
                image_name, image_bytes = parse_tfexample(rec)
                if image_bytes is None:
                    continue
                if not image_name:
                    continue
                batch_names.append(image_name)
                batch_bytes.append(image_bytes)

                if len(batch_names) >= BATCH_SIZE_TEST:
                    x = second_model_preprocess_batch_from_jpeg_bytes_list(batch_bytes)
                    x = x.contiguous(memory_format=torch.channels_last)
                    if device.type == "cuda":
                        x = x.pin_memory().to(device, non_blocking=True)
                    else:
                        x = x.to(device)
                    logits = model2(x)
                    pred_imagenet = (
                        torch.argmax(logits, dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )

                    preds5 = np.full(
                        pred_imagenet.shape, global_majority_final, dtype=np.int32
                    )
                    ok = (pred_imagenet >= 0) & (pred_imagenet < 1000)
                    preds5[ok] = lookup_1000_to_5[pred_imagenet[ok]].astype(np.int32)

                    image_ids.extend(batch_names)
                    prediction.extend(preds5.tolist())
                    batch_names, batch_bytes = [], []

        if batch_names:
            x = second_model_preprocess_batch_from_jpeg_bytes_list(batch_bytes)
            x = x.contiguous(memory_format=torch.channels_last)
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
            logits = model2(x)
            pred_imagenet = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int32)
            )

            preds5 = np.full(pred_imagenet.shape, global_majority_final, dtype=np.int32)
            ok = (pred_imagenet >= 0) & (pred_imagenet < 1000)
            preds5[ok] = lookup_1000_to_5[pred_imagenet[ok]].astype(np.int32)

            image_ids.extend(batch_names)
            prediction.extend(preds5.tolist())

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
        for start in range(0, len(test_paths), BATCH_SIZE_TEST):
            batch_paths = test_paths[start : start + BATCH_SIZE_TEST]
            batch_ids = [os.path.basename(p) for p in batch_paths]

            x = second_model_preprocess_batch_from_paths(batch_paths)
            x = x.contiguous(memory_format=torch.channels_last)
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
            logits = model2(x)
            pred_imagenet = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int32)
            )

            preds5 = np.full(pred_imagenet.shape, global_majority_final, dtype=np.int32)
            ok = (pred_imagenet >= 0) & (pred_imagenet < 1000)
            preds5[ok] = lookup_1000_to_5[pred_imagenet[ok]].astype(np.int32)

            image_ids.extend(batch_ids)
            prediction.extend(preds5.tolist())
    print("Predictions from test_images:", len(prediction))



## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

pred_map = dict(zip(image_ids, prediction))
labels = [
    int(pred_map.get(img_id, global_majority_final))
    for img_id in sample_sub["image_id"].tolist()
]

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
