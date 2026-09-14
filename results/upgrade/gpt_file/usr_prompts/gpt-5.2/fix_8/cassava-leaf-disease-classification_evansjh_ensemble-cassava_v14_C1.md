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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["PYTHONHASHSEED"] = "0"

import sys
from collections import Counter

import numpy as np
import pandas as pd

np.random.seed(0)

for k in list(sys.modules.keys()):
    if k == "google" or k.startswith("google."):
        sys.modules.pop(k, None)

TF_AVAILABLE = False
tf = None
load_model = None
load_img = None
img_to_array = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model  # noqa: F401

    tf.random.set_seed(0)

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

    TF_AVAILABLE = True
    print("TensorFlow import: OK")
except Exception as e:
    TF_AVAILABLE = False
    print("TensorFlow import: FAILED; will run fallback predictor.")
    print("Import error:", repr(e))




## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

OUT_SUB_PATH = "/kaggle/working/submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

majority_label = int(train_df["label"].value_counts().idxmax())
label_counts = train_df["label"].value_counts().to_dict()
print("Train label distribution:", label_counts)
print("Majority label:", majority_label)




## === cell 2
model_candidates = [
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5",
        (550, 550),
    ),
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5",
        (512, 512),
    ),
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5",
        (448, 448),
    ),
    (
        "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5",
        (512, 512),
    ),
    (
        "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5",
        (550, 550),
    ),
    (
        "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5",
        (512, 512),
    ),
    ("/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5", (512, 512)),
]

available_models_info = [(p, sz) for (p, sz) in model_candidates if os.path.exists(p)]
print(f"Found {len(available_models_info)} available .h5 model(s).")
for p, sz in available_models_info:
    print(" -", p, "input:", sz)




## === cell 3
models = []
if TF_AVAILABLE and len(available_models_info) > 0:
    for path, input_size in available_models_info:
        try:
            m = load_model(path, compile=False)
            models.append((m, input_size))
        except Exception as e:
            print(f"Warning: failed to load model {path}: {e}")
else:
    if not TF_AVAILABLE:
        print("Skipping model loading because TensorFlow is unavailable.")
    else:
        print(
            "Skipping model loading because no candidate models exist in this environment."
        )

print(f"Successfully loaded {len(models)} model(s).")




## === cell 4
def _tf_build_base_decode_dataset(file_paths, base_size, batch_size=32):
    base_size = tuple(map(int, base_size))

    def _decode_and_base_resize(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img,
            base_size,
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(file_paths).with_options(opts)
    ds = ds.map(_decode_and_base_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _tf_build_resized_from_base_dataset(base_batched_ds, input_size):
    size = tuple(map(int, input_size))

    def _resize_batch(xb):
        return tf.image.resize(
            xb, size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )

    return base_batched_ds.map(
        _resize_batch, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


def predict_with_models_for_paths(image_paths, models_with_sizes, batch_size=32):
    n = len(image_paths)
    if n == 0:
        return []

    max_side = 0
    for _, sz in models_with_sizes:
        max_side = max(max_side, int(sz[0]), int(sz[1]))
    base_size = (max_side, max_side)

    base_ds = _tf_build_base_decode_dataset(
        image_paths, base_size=base_size, batch_size=batch_size
    )

    all_preds = []
    all_confs = []

    for model, input_size in models_with_sizes:
        ds = _tf_build_resized_from_base_dataset(base_ds, input_size)
        preds = model.predict(ds, verbose=0)  # (n, num_classes)
        preds = np.asarray(preds)
        top1 = np.argmax(preds, axis=1).astype(np.int32)
        top1_conf = preds[np.arange(preds.shape[0]), top1].astype(np.float32)
        all_preds.append(top1)
        all_confs.append(top1_conf)

    all_preds = np.stack(all_preds, axis=1)  # (n, m)
    all_confs = np.stack(all_confs, axis=1)  # (n, m)

    n, m = all_preds.shape
    num_classes = int(all_preds.max()) + 1 if all_preds.size else 5

    counts = np.zeros((n, num_classes), dtype=np.int16)
    rows = np.repeat(np.arange(n, dtype=np.int32), m)
    cols = all_preds.reshape(-1)
    np.add.at(counts, (rows, cols), 1)

    max_counts = counts.max(axis=1, keepdims=True)
    tied_mask = counts == max_counts  # (n, C), True for tied winners

    sum_conf = np.zeros((n, num_classes), dtype=np.float32)
    np.add.at(sum_conf, (rows, cols), all_confs.reshape(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_conf = sum_conf / np.maximum(counts.astype(np.float32), 1.0)

    avg_conf_masked = np.where(tied_mask, avg_conf, -np.inf)
    final = np.argmax(avg_conf_masked, axis=1).astype(np.int32)

    return final.tolist()




## === cell 5
TORCH_AVAILABLE = False
torch = None
torchvision = None
PIL = None

try:
    import torch
    import torchvision
    from PIL import Image

    TORCH_AVAILABLE = True
    print("PyTorch/torchvision import: OK")
except Exception as e:
    TORCH_AVAILABLE = False
    print("PyTorch/torchvision import: FAILED; will use majority class fallback only.")
    print("Import error:", repr(e))




## === cell 6
torch_model = None
torch_transform = None
torch_device = None
torch_label_head = None
use_torch_fallback = False

if (len(models) == 0) and TORCH_AVAILABLE:
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    torch_transform = weights.transforms()

    backbone = torchvision.models.efficientnet_b0(weights=weights)
    in_features = backbone.classifier[1].in_features
    backbone.classifier[1] = torch.nn.Identity()

    torch_label_head = torch.nn.Linear(in_features, 5)

    torch_model = torch.nn.Sequential(backbone, torch_label_head).to(torch_device)

    for p in backbone.parameters():
        p.requires_grad = False

    class CassavaDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            image_id = self.df.loc[idx, "image_id"]
            label = int(self.df.loc[idx, "label"])
            path = os.path.join(self.img_dir, image_id)
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
            y = torch.tensor(label, dtype=torch.long)
            return x, y

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    train_img_dir = os.path.join(DATA_ROOT, "train_images")
    assert os.path.isdir(train_img_dir), f"Missing: {train_img_dir}"

    ds = CassavaDataset(train_df, train_img_dir, torch_transform)

    num_workers = min(4, (os.cpu_count() or 2))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    opt = torch.optim.AdamW(torch_label_head.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    torch_model.train()
    EPOCHS = 2
    for epoch in range(EPOCHS):
        running = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(torch_device, non_blocking=True)
            yb = yb.to(torch_device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = torch_model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        print(
            f"Torch head training epoch {epoch+1}/{EPOCHS} - loss: {running/max(n,1):.4f}"
        )

    torch_model.eval()
    use_torch_fallback = True
    print("Torch fallback: ENABLED (TF models unavailable).")
else:
    if len(models) > 0:
        print("Torch fallback: not used because TF models are loaded.")
    else:
        print("Torch fallback: not used (torch/torchvision unavailable).")




## === cell 7
def predict_with_torch_for_paths(image_paths, batch_size=64):
    class _TestDataset(torch.utils.data.Dataset):
        def __init__(self, paths, transform):
            self.paths = list(paths)
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            p = self.paths[idx]
            img = Image.open(p).convert("RGB")
            return self.transform(img)

    ds = _TestDataset(image_paths, torch_transform)
    num_workers = min(4, (os.cpu_count() or 2))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    preds_out = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(torch_device, non_blocking=True)
            logits = torch_model(xb)
            preds_out.extend(torch.argmax(logits, dim=1).to("cpu").numpy().tolist())
    return [int(x) for x in preds_out]




## === cell 8
image_ids = sample_df["image_id"].tolist()
use_tf_models = TF_AVAILABLE and (len(models) > 0)

test_paths = [os.path.join(TEST_IMG_DIR, image_id) for image_id in image_ids]
exists_mask = np.fromiter(
    (os.path.exists(p) for p in test_paths), dtype=np.bool_, count=len(test_paths)
)

pred_labels = [majority_label] * len(image_ids)

if use_tf_models:
    present_idx = np.flatnonzero(exists_mask)
    present_paths = [test_paths[i] for i in present_idx]

    batch_size = 16 if any(sz[0] >= 550 for _, sz in models) else 32
    present_preds = predict_with_models_for_paths(
        present_paths, models, batch_size=batch_size
    )

    for j, i in enumerate(present_idx.tolist()):
        pred_labels[i] = int(present_preds[j])

elif use_torch_fallback:
    present_idx = np.flatnonzero(exists_mask)
    present_paths = [test_paths[i] for i in present_idx]
    present_preds = predict_with_torch_for_paths(present_paths, batch_size=64)
    for j, i in enumerate(present_idx.tolist()):
        pred_labels[i] = int(present_preds[j])
else:
    pass

submission_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv(OUT_SUB_PATH, index=False)
print(f"Wrote submission to: {OUT_SUB_PATH}")
print(submission_df.head())




## === cell 9
assert os.path.exists(OUT_SUB_PATH), "submission.csv was not created."
chk = pd.read_csv(OUT_SUB_PATH)
assert list(chk.columns) == [
    "image_id",
    "label",
], f"Wrong columns: {chk.columns.tolist()}"
assert len(chk) == len(sample_df), f"Row count mismatch: {len(chk)} vs {len(sample_df)}"
assert chk["label"].dtype.kind in (
    "i",
    "u",
), f"Label dtype should be int, got {chk['label'].dtype}"
chk
