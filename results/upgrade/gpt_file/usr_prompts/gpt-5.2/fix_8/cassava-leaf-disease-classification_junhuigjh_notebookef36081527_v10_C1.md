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
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
rng = np.random.default_rng(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"

print("Python:", os.sys.version)
print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("Working dir:", os.getcwd())




## === cell 1
def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def _compute_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Keep the same handcrafted global-feature approach; no architectural/training changes.
    """
    x = img_rgb.astype(np.float32) / 255.0  # HWC in [0,1]
    flat = x.reshape(-1, 3)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-12

    small = Image.fromarray(img_rgb, mode="RGB").resize(
        (32, 32), resample=Image.BILINEAR
    )
    s = (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1, 3)
    s_mean = s.mean(axis=0)

    r, g, b = mean
    greenness = g - (r + b) / 2.0
    yellowness = (r + g) / 2.0 - b

    hsv_small = small.convert("HSV")
    hsv = (np.asarray(hsv_small, dtype=np.float32) / 255.0).reshape(-1, 3)
    hsv_mean = hsv.mean(axis=0)
    hsv_std = hsv.std(axis=0)

    z = (flat - mean[None, :]) / std[None, :]
    skew = (z**3).mean(axis=0)

    gray = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    gsmall = (
        np.asarray(
            Image.fromarray((gray * 255.0).astype(np.uint8), mode="L").resize(
                (32, 32), resample=Image.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    gx = np.diff(gsmall, axis=1, append=gsmall[:, -1:])
    gy = np.diff(gsmall, axis=0, append=gsmall[-1:, :])
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(grad.mean())
    grad_std = float(grad.std())

    feat = np.concatenate(
        [
            mean,
            std - 1e-12,
            s_mean,
            np.array([greenness, yellowness], dtype=np.float32),
            hsv_mean,
            hsv_std,
            skew,
            np.array([grad_mean, grad_std], dtype=np.float32),
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def _zscore_fit(X: np.ndarray, eps: float = 1e-6):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


def _kmeans_cosine(
    X_unit: np.ndarray, k: int, iters: int = 12, seed: int = 42
) -> np.ndarray:
    """
    Minimal score-improving change:
    - still using cosine similarity to representative vectors
    - but allow multiple prototypes per class (k-means on unit vectors).
    """
    n = X_unit.shape[0]
    if n == 0:
        raise ValueError("Empty X for kmeans.")
    if n <= k:
        centers = X_unit.copy()
        while centers.shape[0] < k:
            centers = np.vstack([centers, centers[: (k - centers.shape[0])]])
        return _l2_normalize(centers[:k])

    local_rng = np.random.default_rng(seed)
    init_idx = local_rng.choice(n, size=k, replace=False)
    centers = X_unit[init_idx].copy()

    for _ in range(iters):
        sims = X_unit @ centers.T  # (n,k) cosine sims because both unit
        assign = np.argmax(sims, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = assign == j
            if np.any(mask):
                new_centers[j] = X_unit[mask].mean(axis=0)
            else:
                new_centers[j] = X_unit[local_rng.integers(0, n)]
        centers = _l2_normalize(new_centers)
    return centers.astype(np.float32)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_image_ids = sample_sub["image_id"].astype(str).tolist()

PER_CLASS_CAP = 1200  # keep same cap to preserve runtime/behavior envelope

train_df = train_df.copy()
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

selected_idx = []
for c in sorted(train_df["label"].unique()):
    idx = train_df.index[train_df["label"] == c].to_numpy()
    if len(idx) > PER_CLASS_CAP:
        idx = rng.choice(idx, size=PER_CLASS_CAP, replace=False)
    selected_idx.append(idx)
selected_idx = np.concatenate(selected_idx, axis=0)
train_sub = train_df.loc[selected_idx].reset_index(drop=True)

print("Train total:", len(train_df), "Train used:", len(train_sub))
print("Train used per class:", train_sub["label"].value_counts().sort_index().to_dict())

X_list = []
y_list = []
missing_train = 0

for image_id, label in zip(train_sub["image_id"].tolist(), train_sub["label"].tolist()):
    p = os.path.join(TRAIN_IMG_DIR, image_id)
    if not os.path.exists(p):
        missing_train += 1
        continue
    img = _read_rgb(p)
    feat = _compute_features(img)
    X_list.append(feat)
    y_list.append(label)

if missing_train:
    print("WARNING: missing train images:", missing_train)

X_train = np.stack(X_list, axis=0).astype(np.float32)
y_train = np.array(y_list, dtype=np.int64)

mu, sd = _zscore_fit(X_train)
X_train_z = _zscore_apply(X_train, mu, sd)
X_train_unit = _l2_normalize(X_train_z)

num_classes = 5

K_PROTOS = 2
protos = np.zeros((num_classes, K_PROTOS, X_train_unit.shape[1]), dtype=np.float32)
counts = np.zeros((num_classes,), dtype=np.int64)

for c in range(num_classes):
    mask = y_train == c
    Xc = X_train_unit[mask]
    counts[c] = int(Xc.shape[0])
    if Xc.shape[0] == 0:
        global_center = _l2_normalize(X_train_unit.mean(axis=0, keepdims=True))[0]
        protos[c, :, :] = global_center[None, :]
    elif Xc.shape[0] == 1:
        protos[c, :, :] = Xc[0][None, :]
    else:
        protos[c] = _kmeans_cosine(Xc, k=K_PROTOS, iters=12, seed=SEED + 1000 * c)

print("Prototype counts per class:", counts.tolist())

majority_label = int(train_df["label"].mode().iloc[0])
print("Majority label:", majority_label)



## === cell 3
image_ids_out = []
preds_out = []
missing_test = 0
unreadable_test = 0

for image_name in test_image_ids:
    img_path = os.path.join(TEST_IMG_DIR, image_name)
    if not os.path.exists(img_path):
        missing_test += 1
        image_ids_out.append(image_name)
        preds_out.append(majority_label)
        continue

    try:
        img = _read_rgb(img_path)

        feat1 = _compute_features(img).astype(np.float32)

        img_flip = img[:, ::-1, :]
        feat2 = _compute_features(img_flip).astype(np.float32)

        feat = 0.5 * (feat1 + feat2)
        feat = _zscore_apply(feat[None, :], mu, sd)[0]
        feat = _l2_normalize(feat[None, :])[0]

        sims = np.einsum("ckd,d->ck", protos, feat)  # (C,K)
        sims_class = sims.max(axis=1)  # (C,)
        pred = int(np.argmax(sims_class))
    except Exception:
        unreadable_test += 1
        pred = majority_label

    image_ids_out.append(image_name)
    preds_out.append(pred)

if missing_test:
    print(
        f"WARNING: {missing_test} test images listed in sample_submission not found on disk."
    )
if unreadable_test:
    print(
        f"WARNING: {unreadable_test} test images could not be read; used majority fallback."
    )

pred_df = pd.DataFrame(
    {"image_id": image_ids_out, "label": np.array(preds_out, dtype=np.int64)}
)

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    missing = int(submission["label"].isna().sum())
    print(
        f"WARNING: {missing} missing predictions after merge; filling with majority_label={majority_label}"
    )
    submission["label"] = submission["label"].fillna(majority_label)

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
