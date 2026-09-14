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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np

from collections import Counter
from PIL import Image

TF_AVAILABLE = False
tf = None
load_model = None

print("TF_AVAILABLE:", TF_AVAILABLE)



## === cell 1
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"



## === cell 2
sample_csv = pd.read_csv(sample)
print(sample_csv.head())
print("sample_csv shape:", sample_csv.shape)

train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("train_df shape:", train_df.shape)

majority_label = int(train_df["label"].value_counts().idxmax())
label_dist = train_df["label"].value_counts().sort_index()
print("Train label distribution:\n", label_dist.to_string())
print("Majority label:", majority_label)




## === cell 3
def discover_models_under_kaggle_input(root="/kaggle/input", max_models=2):
    h5_paths = []
    savedmodel_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")):
                h5_paths.append(os.path.join(dirpath, fn))
        if "saved_model.pb" in filenames:
            savedmodel_dirs.append(dirpath)

    h5_paths = sorted(h5_paths)
    savedmodel_dirs = sorted(savedmodel_dirs)

    candidates = h5_paths + savedmodel_dirs
    return candidates[:max_models], {
        "h5_found": len(h5_paths),
        "savedmodel_found": len(savedmodel_dirs),
    }


def try_load_model(path):
    try:
        m = load_model(path, compile=False)
        return m, None
    except Exception as e:
        return None, e


preferred_paths = [
    model_path_5,
    model_path_6,
    model_path_4,
    model_path_1,
    model_path_2,
    model_path_3,
]
existing_preferred = [p for p in preferred_paths if os.path.exists(p)]

if len(existing_preferred) < 2:
    discovered, stats = discover_models_under_kaggle_input(
        "/kaggle/input", max_models=10
    )
    print("Preferred existing:", existing_preferred)
    print("Discovered models stats:", stats)
    for p in discovered:
        if p not in existing_preferred:
            existing_preferred.append(p)

print("Model candidates (first 10):")
for p in existing_preferred[:10]:
    print(" -", p)




## === cell 4
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _extra_color_features(arr_rgb01: np.ndarray) -> np.ndarray:
    r = arr_rgb01[..., 0]
    g = arr_rgb01[..., 1]
    b = arr_rgb01[..., 2]

    s = r + g + b + 1e-6
    r_ch = r / s
    g_ch = g / s
    b_ch = b / s

    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    exb = 1.4 * b - g

    feats = np.array(
        [
            float(exg.mean()),
            float(exg.std()),
            float(exr.mean()),
            float(exr.std()),
            float(exb.mean()),
            float(exb.std()),
            float(r_ch.mean()),
            float(g_ch.mean()),
            float(b_ch.mean()),
            float(r_ch.std()),
            float(g_ch.std()),
            float(b_ch.std()),
        ],
        dtype=np.float32,
    )
    return feats


def _block_means(arr_rgb01: np.ndarray, grid: int = 6) -> np.ndarray:
    h, w, _ = arr_rgb01.shape
    gh = h // grid
    gw = w // grid
    if gh <= 0 or gw <= 0:
        return np.zeros((3 * grid * grid,), dtype=np.float32)

    blocks = []
    for yy in range(grid):
        y0 = yy * gh
        y1 = (yy + 1) * gh if yy < grid - 1 else h
        for xx in range(grid):
            x0 = xx * gw
            x1 = (xx + 1) * gw if xx < grid - 1 else w
            patch = arr_rgb01[y0:y1, x0:x1, :]
            blocks.append(patch.mean(axis=(0, 1)))
    blocks = np.stack(blocks, axis=0).reshape(-1).astype(np.float32)  # (grid*grid*3,)
    return blocks


def _apply_brightness_contrast(
    arr_rgb01: np.ndarray, brightness_delta: float, contrast: float
) -> np.ndarray:
    x = arr_rgb01 + brightness_delta
    x = (x - 0.5) * contrast + 0.5
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _feats_from_arr_and_blocks(
    arr_rgb01: np.ndarray, blocks: np.ndarray, size: int, block_grid: int
) -> np.ndarray:
    gray = (
        0.2989 * arr_rgb01[..., 0]
        + 0.5870 * arr_rgb01[..., 1]
        + 0.1140 * arr_rgb01[..., 2]
    ).astype(np.float32)

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_mean = float((gx.mean() + gy.mean()) * 0.5)
    edge_std = float((gx.std() + gy.std()) * 0.5)

    g_mean = float(gray.mean())
    g_std = float(gray.std())
    g_q25 = float(np.percentile(gray, 25, method="linear"))
    g_q75 = float(np.percentile(gray, 75, method="linear"))

    r_mean = float(arr_rgb01[..., 0].mean())
    gch_mean = float(arr_rgb01[..., 1].mean())
    b_mean = float(arr_rgb01[..., 2].mean())
    r_std = float(arr_rgb01[..., 0].std())
    gch_std = float(arr_rgb01[..., 1].std())
    b_std = float(arr_rgb01[..., 2].std())

    flat_rgb = arr_rgb01.reshape(-1).astype(np.float32)
    extra = _extra_color_features(arr_rgb01)

    feats = np.concatenate(
        [
            flat_rgb,
            np.array(
                [
                    g_mean,
                    g_std,
                    g_q25,
                    g_q75,
                    edge_mean,
                    edge_std,
                    r_mean,
                    gch_mean,
                    b_mean,
                    r_std,
                    gch_std,
                    b_std,
                ],
                dtype=np.float32,
            ),
            extra,
            blocks.astype(np.float32, copy=False),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return feats


def build_dataset_triple_aug(
    df,
    img_dir,
    size=72,
    block_grid=6,
    brightness_delta=0.06,
    contrast=1.10,
    resize_resample=Image.Resampling.BICUBIC,
):
    d = 3 * size * size + 12 + 12 + 3 * block_grid * block_grid
    n = len(df)
    X = np.zeros((n * 3, d), dtype=np.float32)
    y = None
    if "label" in df.columns:
        y0 = df["label"].to_numpy(dtype=np.int64)
        y = np.concatenate([y0, y0, y0], axis=0)

    image_ids = df["image_id"].to_numpy()
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(img_dir, str(image_id))
        base = i
        flip = i + n
        bc = i + 2 * n
        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
            img = img.resize((size, size), resample=resize_resample)
            arr = (np.asarray(img, dtype=np.float32) / 255.0).astype(
                np.float32, copy=False
            )

            blocks0 = _block_means(arr, grid=block_grid)
            X[base] = _feats_from_arr_and_blocks(
                arr, blocks0, size=size, block_grid=block_grid
            )

            arr_f = arr[:, ::-1, :].copy()  # ensure contiguous for downstream ops
            blocks_f = _block_means(arr_f, grid=block_grid)
            X[flip] = _feats_from_arr_and_blocks(
                arr_f, blocks_f, size=size, block_grid=block_grid
            )

            arr_bc = _apply_brightness_contrast(
                arr, brightness_delta=brightness_delta, contrast=contrast
            )
            blocks_bc = _block_means(arr_bc, grid=block_grid)
            X[bc] = _feats_from_arr_and_blocks(
                arr_bc, blocks_bc, size=size, block_grid=block_grid
            )

        except Exception:
            pass

    return X, y


def build_dataset_single(
    df,
    img_dir,
    size=72,
    block_grid=6,
    resize_resample=Image.Resampling.BICUBIC,
):
    d = 3 * size * size + 12 + 12 + 3 * block_grid * block_grid
    X = np.zeros((len(df), d), dtype=np.float32)
    y = None
    if "label" in df.columns:
        y = df["label"].to_numpy(dtype=np.int64)

    image_ids = df["image_id"].to_numpy()
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(img_dir, str(image_id))
        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
            img = img.resize((size, size), resample=resize_resample)
            arr = (np.asarray(img, dtype=np.float32) / 255.0).astype(
                np.float32, copy=False
            )
            blocks = _block_means(arr, grid=block_grid)
            X[i] = _feats_from_arr_and_blocks(
                arr, blocks, size=size, block_grid=block_grid
            )
        except Exception:
            pass
    return X, y


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-6, 1.0, sig)
    return mu.astype(np.float32), sig.astype(np.float32)


def standardize_apply(X, mu, sig):
    return ((X - mu) / sig).astype(np.float32)


def train_multinomial_logreg(
    X, y, n_classes=5, lr=0.15, reg=1e-3, epochs=60, batch_size=256, seed=123
):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    y_onehot = np.eye(n_classes, dtype=np.float32)[y]

    for ep in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            j = idx[start : start + batch_size]
            Xb = X[j]
            yb = y_onehot[j]

            logits = Xb @ W + b
            P = softmax(logits)

            grad_logits = (P - yb) / Xb.shape[0]
            gW = Xb.T @ grad_logits + reg * W
            gb = grad_logits.sum(axis=0)

            W -= lr * gW
            b -= lr * gb

    return W, b


def predict_logreg(X, W, b):
    logits = X @ W + b
    return np.argmax(logits, axis=1).astype(np.int64)


SEED = 123

MAX_TRAIN_SAMPLES = 14000
FEAT_SIZE = 72

BLOCK_GRID = 8

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
if len(train_df_shuf) > MAX_TRAIN_SAMPLES:
    train_used = train_df_shuf.iloc[:MAX_TRAIN_SAMPLES].copy()
else:
    train_used = train_df_shuf

print("Using train samples:", len(train_used), "of", len(train_df))

X_train, y_train = build_dataset_triple_aug(
    train_used,
    train_image_dir,
    size=FEAT_SIZE,
    block_grid=BLOCK_GRID,
    brightness_delta=0.06,
    contrast=1.10,
)

mu, sig = standardize_fit(X_train)
X_train = standardize_apply(X_train, mu, sig)

W, b = train_multinomial_logreg(
    X_train,
    y_train,
    n_classes=5,
    lr=0.12,
    reg=5e-3,
    epochs=60,
    batch_size=256,
    seed=SEED,
)

X_test, _ = build_dataset_single(
    sample_csv, test_image_dir, size=FEAT_SIZE, block_grid=BLOCK_GRID
)
X_test = standardize_apply(X_test, mu, sig)
test_pred = predict_logreg(X_test, W, b)

test_pred = np.clip(test_pred, 0, 4).astype(int)



## === cell 5
image_predictions = []
test_ids = sample_csv["image_id"].tolist()

if TF_AVAILABLE:
    raise RuntimeError(
        "TF_AVAILABLE unexpectedly True in this environment; this script expects TF import to fail. "
        "If TF becomes available, re-enable the original model inference code path."
    )
else:
    image_predictions = [
        {"image_id": iid, "label": int(p)} for iid, p in zip(test_ids, test_pred)
    ]

submission_df = pd.DataFrame(image_predictions)
submission_df = submission_df[["image_id", "label"]]
assert submission_df.shape[0] == sample_csv.shape[0], (
    submission_df.shape,
    sample_csv.shape,
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission_df.shape)



## === cell 6
print(submission_df.head())
print(submission_df["label"].value_counts().sort_index())
print("Submission columns:", submission_df.columns.tolist())
print("Submission preview saved at /kaggle/working/submission.csv")
