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

0.17676

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15433) has done: 'I remove the failing TensorFlow Hub dependency (it’s triggering the `MessageFactory.GetPrototype` protobuf error in this environment and also relies on an unavailable cached input folder), and replace it with a stable, built-in TensorFlow/Keras model available offline. I also fix the missing-path issues by resolving the correct competition input directory and ensuring `test_image_dir`/`sample` are always defined. Because your external `.h5` models are not present in the provided filesystem, I keep the ensemble/voting logic but load a single lightweight baseline model (MobileNetV2) so the notebook runs end-to-end and writes a valid `submission.csv` with the correct columns. Finally, I ensure predictions align exactly to `sample_submission.csv` order to avoid submission formatting/index mismatches.'
- What this solution (achieved 0.61099) has done: 'The crash happens before any data/model code runs because TensorFlow’s import triggers a protobuf API mismatch (`MessageFactory.GetPrototype`) in this environment. To make the notebook run end-to-end and produce a valid `submission.csv`, I remove the TensorFlow/Keras dependency entirely and replace the failing image-model section with a stable, offline baseline: a simple majority-class predictor learned from `train.csv`. This is minimal, fixes the runtime error deterministically, preserves correct submission ordering/format, and should substantially improve accuracy over near-random predictions (moving your score toward the target). Paths are kept compatible with both `/kaggle/input/...` and `/kaggle/data/...` layouts.'
- What this solution (achieved 0.17937) has done: 'To move accuracy upward from the current majority-class baseline (0.61099) toward your target (~0.8988), the smallest legitimate improvement is to stop predicting a single class for every image and instead use the provided test images with a lightweight, offline classifier. Since TensorFlow currently fails to import in this environment, this patch keeps the same simple “train once → predict labels” approach but upgrades the model to a CPU-friendly nearest-centroid classifier using raw pixel features (downsampled grayscale), implemented only with NumPy/Pandas (no new dependencies). This should materially increase accuracy versus majority class while staying simple and deterministic. The submission is still written in exactly `sample_submission.csv` order with the correct columns and `submission.csv` filename.'
- What this solution (achieved 0.17676) has done: 'Your current nearest-centroid baseline is likely underperforming because it uses only the first 6000 training images (sorted by `image_id`), which can bias class coverage and centroid quality. To move accuracy upward toward the 0.8988 target while preserving the same “nearest centroid on downsampled grayscale pixels” core logic, I (1) change the training subset selection to a deterministic **stratified** per-class sample so every class contributes, and (2) increase the number of training images used (still bounded for runtime) to produce more reliable centroids. I also compute centroids for all 5 classes (falling back to global mean if a class has zero loaded images) to avoid missing-class behavior. Submission formatting and ordering remain exactly aligned to `sample_submission.csv`.'

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

IMG_SIZE = 64  # keep same core feature extraction resolution

MAX_TRAIN_IMAGES = 15000

N_CLASSES = 5


def load_image_feature(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    gray = gray / 255.0
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


def build_centroids(train_df, train_dir, max_images=MAX_TRAIN_IMAGES):
    sub = _stratified_subset(train_df, max_images=max_images, seed=42)

    X_list = []
    y_list = []
    failures = 0
    for img_id, y in zip(sub["image_id"].values, sub["label"].values):
        p = os.path.join(train_dir, img_id)
        try:
            X_list.append(load_image_feature(p))
            y_list.append(int(y))
        except Exception:
            failures += 1
            continue

    if len(X_list) == 0:
        raise RuntimeError("Could not load any training images to build centroids.")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int64)

    centroids = {}
    global_mean = X.mean(axis=0)

    for cls in range(N_CLASSES):
        m = y == cls
        if np.any(m):
            centroids[int(cls)] = X[m].mean(axis=0)
        else:
            centroids[int(cls)] = global_mean

    print(
        f"Built centroids using {len(X_list)} images (failures={failures}). Feature dim={X.shape[1]}"
    )
    return centroids


def predict_nearest_centroid(
    centroids, test_ids, test_dir, fallback_label=majority_label
):
    classes = sorted(centroids.keys())
    C = np.vstack([centroids[c] for c in classes])  # (K, D)
    C_norm = (C * C).sum(axis=1, keepdims=True)  # (K,1)

    preds = []
    failures = 0
    for img_id in test_ids:
        p = os.path.join(test_dir, img_id)
        try:
            x = load_image_feature(p)  # (D,)
            x_norm = (x * x).sum()
            dots = C @ x  # (K,)
            d2 = x_norm + C_norm[:, 0] - 2.0 * dots
            pred = classes[int(np.argmin(d2))]
            preds.append(int(pred))
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
