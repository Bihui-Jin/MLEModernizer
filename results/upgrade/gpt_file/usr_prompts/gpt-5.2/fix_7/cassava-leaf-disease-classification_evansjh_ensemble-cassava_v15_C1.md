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

0.8987609549712904

# 6. Current score

0.26457

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15433) has done: 'I remove the failing TensorFlow Hub dependency (it’s triggering the `MessageFactory.GetPrototype` protobuf error in this environment and also relies on an unavailable cached input folder), and replace it with a stable, built-in TensorFlow/Keras model available offline. I also fix the missing-path issues by resolving the correct competition input directory and ensuring `test_image_dir`/`sample` are always defined. Because your external `.h5` models are not present in the provided filesystem, I keep the ensemble/voting logic but load a single lightweight baseline model (MobileNetV2) so the notebook runs end-to-end and writes a valid `submission.csv` with the correct columns. Finally, I ensure predictions align exactly to `sample_submission.csv` order to avoid submission formatting/index mismatches.'
- What this solution (achieved 0.61099) has done: 'The crash happens before any data/model code runs because TensorFlow’s import triggers a protobuf API mismatch (`MessageFactory.GetPrototype`) in this environment. To make the notebook run end-to-end and produce a valid `submission.csv`, I remove the TensorFlow/Keras dependency entirely and replace the failing image-model section with a stable, offline baseline: a simple majority-class predictor learned from `train.csv`. This is minimal, fixes the runtime error deterministically, preserves correct submission ordering/format, and should substantially improve accuracy over near-random predictions (moving your score toward the target). Paths are kept compatible with both `/kaggle/input/...` and `/kaggle/data/...` layouts.'
- What this solution (achieved 0.17937) has done: 'To move accuracy upward from the current majority-class baseline (0.61099) toward your target (~0.8988), the smallest legitimate improvement is to stop predicting a single class for every image and instead use the provided test images with a lightweight, offline classifier. Since TensorFlow currently fails to import in this environment, this patch keeps the same simple “train once → predict labels” approach but upgrades the model to a CPU-friendly nearest-centroid classifier using raw pixel features (downsampled grayscale), implemented only with NumPy/Pandas (no new dependencies). This should materially increase accuracy versus majority class while staying simple and deterministic. The submission is still written in exactly `sample_submission.csv` order with the correct columns and `submission.csv` filename.'
- What this solution (achieved 0.17676) has done: 'Your current nearest-centroid baseline is likely underperforming because it uses only the first 6000 training images (sorted by `image_id`), which can bias class coverage and centroid quality. To move accuracy upward toward the 0.8988 target while preserving the same “nearest centroid on downsampled grayscale pixels” core logic, I (1) change the training subset selection to a deterministic **stratified** per-class sample so every class contributes, and (2) increase the number of training images used (still bounded for runtime) to produce more reliable centroids. I also compute centroids for all 5 classes (falling back to global mean if a class has zero loaded images) to avoid missing-class behavior. Submission formatting and ordering remain exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.32549) has done: 'We keep your nearest-centroid-on-downsampled-grayscale core logic unchanged, but fix the main accuracy bottleneck: raw grayscale pixels are very sensitive to lighting/background, so we add a tiny, deterministic normalization step (per-image standardization) before flattening, which usually improves centroid separability without changing the model family. We also compute centroids in a numerically stable, streaming way (sums/counts) to avoid any subtle bias from stacking order and reduce memory pressure, while still using the same subset and distance rule. Finally, we slightly increase `IMG_SIZE` (still lightweight) to retain more leaf structure, which should move the score upward toward your target without altering the overall approach or runtime envelope.'
- What this solution (achieved 0.26457) has done: 'Your current gap to the target is large (0.32549 → 0.89876), so we need a meaningful but still minimal, same-family improvement. We keep the exact “nearest-centroid on downsampled grayscale pixels + L2 distance” core logic, but fix the biggest accuracy limiter: using a single global centroid per class is too crude. The smallest upgrade within the same centroid framework is to use multiple centroids per class (a tiny, fixed k-means per class implemented in NumPy), then predict by nearest centroid among all class-centroids. This remains deterministic, lightweight, uses no new packages, keeps the same submission semantics/order, and should move accuracy upward toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",  # fallback (some environments mount files directly here)
    "/kaggle/data",
]


def _resolve_base():
    for p in BASE_INPUT_CANDIDATES:
        if os.path.isdir(p):
            if os.path.isfile(os.path.join(p, "sample_submission.csv")):
                return p
            nested = os.path.join(p, "cassava-leaf-disease-classification")
            if os.path.isfile(os.path.join(nested, "sample_submission.csv")):
                return nested
    return None


BASE_INPUT = _resolve_base()
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not find competition directory containing sample_submission.csv in: "
        + ", ".join(BASE_INPUT_CANDIDATES)
    )

train_image_dir = os.path.join(BASE_INPUT, "train_images")
test_image_dir = os.path.join(BASE_INPUT, "test_images")
sample = os.path.join(BASE_INPUT, "sample_submission.csv")
train_csv_path = os.path.join(BASE_INPUT, "train.csv")

if not os.path.isfile(sample):
    raise FileNotFoundError(f"Missing sample submission: {sample}")
if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(f"Missing train.csv: {train_csv_path}")
if not os.path.isdir(train_image_dir):
    raise FileNotFoundError(f"Missing train_images directory: {train_image_dir}")
if not os.path.isdir(test_image_dir):
    raise FileNotFoundError(f"Missing test_images directory: {test_image_dir}")

print("Resolved paths:")
print("BASE_INPUT:", BASE_INPUT)
print("train_image_dir:", train_image_dir)
print("test_image_dir:", test_image_dir)
print("sample:", sample)
print("train_csv:", train_csv_path)



## === cell 1
sample_csv = pd.read_csv(sample)
assert list(sample_csv.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv columns"
print(sample_csv.head())
print("Test rows:", len(sample_csv))

train_df = pd.read_csv(train_csv_path)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"
print(train_df.head())
print("Train rows:", len(train_df))

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())
print("Train label distribution:")
print(label_counts.sort_index())
print("Majority label:", majority_label, "count:", int(label_counts.max()))




## === cell 2
def _try_import_pil():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:
        return None


Image = _try_import_pil()
if Image is None:
    print(
        "WARNING: Pillow (PIL) not available; falling back to majority-class predictions."
    )
    USE_IMAGES = False
else:
    USE_IMAGES = True
    print("Pillow available; using image-based nearest-centroid classifier.")

IMG_SIZE = 96

MAX_TRAIN_IMAGES = 15000
N_CLASSES = 5

K_PER_CLASS = 6  # small, fixed to keep runtime bounded
KMEANS_ITERS = (
    6  # small, deterministic; avoids changing training "approach" beyond centroids
)


def load_image_feature(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    gray = gray / 255.0

    m = float(gray.mean())
    s = float(gray.std())
    if s < 1e-6:
        s = 1e-6
    gray = (gray - m) / s

    return gray.reshape(-1)


def _stratified_subset(df, max_images, seed=42):
    rng = np.random.RandomState(seed)
    per_class = max(1, max_images // N_CLASSES)
    parts = []
    for cls in range(N_CLASSES):
        cls_df = df[df["label"] == cls]
        if len(cls_df) == 0:
            continue
        n = min(per_class, len(cls_df))
        idx = rng.choice(cls_df.index.values, size=n, replace=False)
        parts.append(df.loc[idx])
    if len(parts) == 0:
        return df.head(0)
    sub = pd.concat(parts, axis=0)
    if len(sub) < max_images:
        remaining = df.drop(index=sub.index)
        need = min(max_images - len(sub), len(remaining))
        if need > 0:
            perm = rng.permutation(remaining.index.values)
            fill_idx = perm[:need]
            sub = pd.concat([sub, df.loc[fill_idx]], axis=0)
    return sub.sort_values("image_id").reset_index(drop=True)


def _kmeans_lloyd(X, k, iters, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    k_eff = int(min(k, n))
    init_idx = rng.choice(n, size=k_eff, replace=False)
    C = X[init_idx].astype(np.float32, copy=True)  # (k, D)

    for _ in range(int(iters)):
        C_norm = (C * C).sum(axis=1, keepdims=True)  # (k,1)
        X_norm = (X * X).sum(axis=1, keepdims=True)  # (n,1)
        d2 = X_norm + C_norm.T - 2.0 * (X @ C.T)  # (n,k)
        a = np.argmin(d2, axis=1)  # (n,)

        newC = np.empty_like(C)
        for j in range(k_eff):
            mask = a == j
            if np.any(mask):
                newC[j] = X[mask].mean(axis=0)
            else:
                far_idx = int(np.argmax(d2.min(axis=1)))
                newC[j] = X[far_idx]
        C = newC
    return C.astype(np.float32)


def build_centroids(train_df, train_dir, max_images=MAX_TRAIN_IMAGES):
    sub = _stratified_subset(train_df, max_images=max_images, seed=42)

    feats_by_class = {c: [] for c in range(N_CLASSES)}
    failures = 0
    global_sum = None
    global_count = 0
    feat_dim = None

    for img_id, y in zip(sub["image_id"].values, sub["label"].values):
        p = os.path.join(train_dir, img_id)
        try:
            x = load_image_feature(p).astype(np.float32, copy=False)
            if feat_dim is None:
                feat_dim = int(x.shape[0])
                global_sum = np.zeros((feat_dim,), dtype=np.float64)
            cls = int(y)
            feats_by_class[cls].append(x)
            global_sum += x.astype(np.float64, copy=False)
            global_count += 1
        except Exception:
            failures += 1
            continue

    if global_count == 0 or global_sum is None or feat_dim is None:
        raise RuntimeError("Could not load any training images to build centroids.")

    global_mean = (global_sum / max(1, global_count)).astype(np.float32)

    centroids = {}
    per_class_counts = []
    for cls in range(N_CLASSES):
        X_list = feats_by_class.get(cls, [])
        per_class_counts.append(len(X_list))
        if len(X_list) == 0:
            centroids[int(cls)] = global_mean.reshape(1, -1)
            continue
        X = np.stack(X_list, axis=0)  # (n,D)
        C = _kmeans_lloyd(X, k=K_PER_CLASS, iters=KMEANS_ITERS, seed=42 + cls)
        if C.shape[0] == 0:
            C = global_mean.reshape(1, -1)
        centroids[int(cls)] = C

    total_centroids = int(sum(centroids[c].shape[0] for c in centroids))
    print(
        f"Built centroids using {int(global_count)} images (failures={failures}). "
        f"Feature dim={feat_dim}. Total centroids={total_centroids} ({K_PER_CLASS}/class target)."
    )
    print("Per-class images used:", per_class_counts)
    print(
        "Per-class centroids:",
        {c: int(centroids[c].shape[0]) for c in sorted(centroids)},
    )
    return centroids


def predict_nearest_centroid(
    centroids, test_ids, test_dir, fallback_label=majority_label
):
    classes = sorted(centroids.keys())
    C_list = []
    L_list = []
    for c in classes:
        Ci = centroids[c]
        C_list.append(Ci)
        L_list.extend([c] * Ci.shape[0])

    C = np.vstack(C_list).astype(np.float32)  # (M,D)
    L = np.asarray(L_list, dtype=np.int64)  # (M,)
    C_norm = (C * C).sum(axis=1)  # (M,)

    preds = []
    failures = 0
    for img_id in test_ids:
        p = os.path.join(test_dir, img_id)
        try:
            x = load_image_feature(p).astype(np.float32, copy=False)  # (D,)
            x_norm = float((x * x).sum())
            dots = C @ x  # (M,)
            d2 = x_norm + C_norm - 2.0 * dots
            pred = int(L[int(np.argmin(d2))])
            preds.append(pred)
        except Exception:
            failures += 1
            preds.append(int(fallback_label))

    print(
        f"Predicted {len(preds)} test images (failures={failures}, used fallback={fallback_label} for those)."
    )
    return preds


if USE_IMAGES:
    centroids = build_centroids(train_df, train_image_dir, max_images=MAX_TRAIN_IMAGES)
    pred_labels = predict_nearest_centroid(
        centroids=centroids,
        test_ids=sample_csv["image_id"].values.tolist(),
        test_dir=test_image_dir,
        fallback_label=majority_label,
    )
else:
    pred_labels = np.full(
        shape=(len(sample_csv),), fill_value=majority_label, dtype=np.int64
    ).tolist()

print("Predictions:", len(pred_labels))



## === cell 3
submission_df = sample_csv.copy()
submission_df["label"] = pred_labels

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)

assert submission_df.shape[0] == sample_csv.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]
assert submission_path.endswith(".csv")
