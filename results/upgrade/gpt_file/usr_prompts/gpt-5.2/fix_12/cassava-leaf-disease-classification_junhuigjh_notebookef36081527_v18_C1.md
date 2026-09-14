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


def second_model_preprocess_batch_from_paths(image_paths):
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    xs = [preprocess(im) for im in imgs]
    return torch.stack(xs, dim=0)


def second_model_preprocess_batch_from_jpeg_bytes_list(image_bytes_list):
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
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
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
            logits = model2(x)
            pred_imagenet = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
            )
            preds.append(pred_imagenet)
    if not preds:
        return np.array([], dtype=np.int32)
    return np.concatenate(preds, axis=0).astype(np.int32)


def build_imagenet_to_cassava_lookup_from_df(
    df: pd.DataFrame,
    train_image_dir: str,
    batch_size: int = 128,
    laplace: int = 1,
    min_support: int = 3,
    min_margin: float = 0.05,
):
    df = df.dropna(subset=["image_id", "label"]).copy()
    df["label"] = df["label"].astype(int)

    paths, labels = [], []
    for img_id, lab in zip(df["image_id"].tolist(), df["label"].tolist()):
        pth = os.path.join(train_image_dir, img_id)
        if os.path.exists(pth):
            paths.append(pth)
            labels.append(int(lab))

    if len(paths) == 0:
        lookup = np.arange(1000, dtype=np.int16) % 5
        global_majority = 0
        return lookup, global_majority

    labels_arr = np.array(labels, dtype=np.int32)
    global_majority = int(np.argmax(np.bincount(labels_arr, minlength=5)))

    pred_imagenet = _predict_imagenet_argmax_for_paths(paths, batch_size=batch_size)

    counts = np.zeros((1000, 5), dtype=np.int32)
    for c1000, y5 in zip(pred_imagenet.tolist(), labels_arr.tolist()):
        if 0 <= c1000 < 1000 and 0 <= y5 < 5:
            counts[c1000, y5] += 1

    lookup = np.full((1000,), global_majority, dtype=np.int16)
    for c in range(1000):
        row = counts[c]
        support = int(row.sum())
        if support < min_support:
            continue
        sm = row.astype(np.float64) + float(laplace)
        probs = sm / sm.sum()
        best = int(np.argmax(probs))
        top2 = np.partition(probs, -2)[-2:]
        margin = float(top2.max() - top2.min())
        if margin < min_margin:
            continue
        lookup[c] = best

    return lookup, global_majority


def score_lookup_on_df(
    df_val: pd.DataFrame,
    lookup: np.ndarray,
    global_majority: int,
    train_image_dir: str,
    batch_size: int = 128,
):
    df_val = df_val.dropna(subset=["image_id", "label"]).copy()
    df_val["label"] = df_val["label"].astype(int)

    paths, y_true = [], []
    for img_id, lab in zip(df_val["image_id"].tolist(), df_val["label"].tolist()):
        pth = os.path.join(train_image_dir, img_id)
        if os.path.exists(pth):
            paths.append(pth)
            y_true.append(int(lab))

    if len(paths) == 0:
        return 0.0, 0

    y_true = np.array(y_true, dtype=np.int32)
    pred_imagenet = _predict_imagenet_argmax_for_paths(paths, batch_size=batch_size)
    y_pred = np.full_like(y_true, global_majority)
    ok = (pred_imagenet >= 0) & (pred_imagenet < 1000)
    y_pred[ok] = lookup[pred_imagenet[ok]].astype(np.int32)
    acc = float((y_pred == y_true).mean())
    return acc, len(paths)


df_all = pd.read_csv(TRAIN_CSV_PATH)
df_all = df_all.dropna(subset=["image_id", "label"]).copy()
df_all["label"] = df_all["label"].astype(int)

df_tr, df_va = _stable_stratified_split(
    df_all, label_col="label", val_frac=0.1, seed=42
)
print("Train/val sizes:", len(df_tr), len(df_va))

param_grid = [
    {"laplace": 0, "min_support": 2, "min_margin": 0.00},
    {"laplace": 1, "min_support": 2, "min_margin": 0.00},
    {"laplace": 1, "min_support": 3, "min_margin": 0.03},
    {"laplace": 1, "min_support": 3, "min_margin": 0.05},
    {"laplace": 1, "min_support": 5, "min_margin": 0.03},
    {"laplace": 2, "min_support": 3, "min_margin": 0.05},
]

best = None
best_acc = -1.0
best_stats = None

for params in param_grid:
    lookup_tmp, gm_tmp = build_imagenet_to_cassava_lookup_from_df(
        df_tr,
        TRAIN_IMAGE_DIR,
        batch_size=128,
        laplace=int(params["laplace"]),
        min_support=int(params["min_support"]),
        min_margin=float(params["min_margin"]),
    )
    acc, n_eval = score_lookup_on_df(
        df_va, lookup_tmp, gm_tmp, TRAIN_IMAGE_DIR, batch_size=128
    )
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

lookup_1000_to_5, global_majority_final = build_imagenet_to_cassava_lookup_from_df(
    df_all,
    TRAIN_IMAGE_DIR,
    batch_size=128,
    laplace=int(best["laplace"]),
    min_support=int(best["min_support"]),
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
                    if device.type == "cuda":
                        x = x.pin_memory().to(device, non_blocking=True)
                    else:
                        x = x.to(device)
                    logits = model2(x)
                    pred_imagenet = (
                        torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
                    )
                    preds5 = [imagenet_to_cassava_label(int(c)) for c in pred_imagenet]
                    image_ids.extend(batch_names)
                    prediction.extend(preds5)
                    batch_names, batch_bytes = [], []

        if batch_names:
            x = second_model_preprocess_batch_from_jpeg_bytes_list(batch_bytes)
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
            logits = model2(x)
            pred_imagenet = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
            )
            preds5 = [imagenet_to_cassava_label(int(c)) for c in pred_imagenet]
            image_ids.extend(batch_names)
            prediction.extend(preds5)

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
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
            logits = model2(x)
            pred_imagenet = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
            )
            preds5 = [imagenet_to_cassava_label(int(c)) for c in pred_imagenet]

            image_ids.extend(batch_ids)
            prediction.extend(preds5)
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
