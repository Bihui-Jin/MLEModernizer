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

2.7

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

0.8634028407373829

# 6. Current score

0.18348

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the environment/runtime failures first by removing the TensorFlow import from the initial “version print” cell (it triggers the `MessageFactory.GetPrototype` crash) and making the script self-contained using only the local Cassava dataset. Then I fix the data generator error by converting `train_df["label"]` to string for `class_mode="categorical"`, and I ensure `NUM_CLASSES` is defined before it’s used so the fallback model path runs. Finally, I make inference robust by always producing a prediction array aligned to `sample_submission.csv` and writing `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.1293) has done: 'We fix the TensorFlow import/runtime crash (`MessageFactory.GetPrototype`) by avoiding TF entirely and switching to a minimal, self-contained image pipeline using only built-in/standard Kaggle-available libraries (PIL + numpy + pandas), while keeping the “train a small CNN then predict” core intent intact. To ensure the notebook always produces a valid `submission.csv`, we make inference robust and guaranteed to align to `sample_submission.csv` row order. Since your current run yields no submission, the primary goal is to unblock end-to-end execution; the simple model be trained deterministically and produce plausible predictions (better than random) without external models. All paths remain the same and the output format is validated.'
- What this solution (achieved 0.18348) has done: 'The timeout is dominated by repeatedly opening/decoding/resizing 18.7k JPEGs multiple times (train split, val split, full train again, then test), plus expensive PIL↔NumPy roundtrips inside feature extraction. I preserve the exact feature definition and centroid-classifier logic, but make extraction single-pass and cached so each train image is decoded only once, then reused for both validation and final centroid fitting. I also remove the redundant PIL reconstruction by computing the 16×16 thumbnail directly from the already-loaded PIL image, and speed up I/O via list-based iteration and optional multiprocessing with deterministic ordering. Finally, I compute centroids via vectorized `np.add.at` (exact sums/means) and compute distances using the algebraic expansion to reduce temporary allocations.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

print(
    "Running self-contained Cassava pipeline (train small classifier from train_images and infer on test_images)."
)



## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T

NOTE: Preserved for original logic context. Not executed in this runtime.
"""
pass



## === cell 2
"""Gambler's loss helpers (not executed in this runtime)."""
pass



## === cell 3
model_v1_path = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
model_v2_path = "/kaggle/input/gambler-s-loss-cassava/saved-model-05-0.860"
model_v3_path = (
    "/kaggle/input/bitempered-loss-only-xception-with-cropping/saved-model-15-0.839"
)

all_model_paths = [model_v1_path, model_v2_path, model_v3_path]
have_external_models = False
print("External models disabled; training an in-notebook model instead.")



## === cell 4
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())

train_df["label"] = train_df["label"].astype(str)
NUM_CLASSES = train_df["label"].nunique()
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 5
from PIL import Image

IMG_SIZE = 224
EPOCHS = 3  # preserved variable for core "train loop" semantics (not used by this classifier)
val_frac = 0.1

train_df_shuffled = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
val_size = int(len(train_df_shuffled) * val_frac)
val_df = train_df_shuffled.iloc[:val_size].copy()
tr_df = train_df_shuffled.iloc[val_size:].copy()

classes_sorted = sorted(train_df["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(classes_sorted)}
index_to_class = {i: c for c, i in class_to_index.items()}


def _safe_open_resize_pil(path, img_size):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    return img


def _extract_feature_vector_from_pil(img_pil, thumb_size=16):
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0  # (H,W,3), H=W=IMG_SIZE
    flat = arr.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)

    thumb = img_pil.resize((thumb_size, thumb_size), resample=Image.NEAREST)
    thumb_arr = (np.asarray(thumb, dtype=np.float32) / 255.0).reshape(-1)  # 16*16*3

    feat = np.concatenate([mean_rgb, std_rgb, thumb_arr], axis=0).astype(np.float32)
    return feat


def _extract_features_cached(df, img_dir, img_size, cache, with_labels=True):
    n = len(df)
    X = None
    y = None
    if with_labels:
        y = np.zeros((n,), dtype=np.int64)

    if with_labels:
        rows = df[["image_id", "label"]].values
    else:
        rows = df[["image_id"]].values

    for i in range(n):
        image_id = rows[i][0]
        key = str(image_id)
        feat = cache.get(key)
        if feat is None:
            path = os.path.join(img_dir, key)
            img_pil = _safe_open_resize_pil(path, img_size)
            feat = _extract_feature_vector_from_pil(img_pil, thumb_size=16)
            cache[key] = feat

        if X is None:
            X = np.zeros((n, feat.shape[0]), dtype=np.float32)
        X[i, :] = feat

        if y is not None:
            label_str = rows[i][1]
            y[i] = class_to_index[str(label_str)]

        if (i + 1) % 2000 == 0:
            print("Processed {}/{} images for features...".format(i + 1, n))
    return X, y


def _compute_centroids(X, y, num_classes):
    d = X.shape[1]
    sums = np.zeros((num_classes, d), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int64)
    np.add.at(sums, y, X)
    np.add.at(counts, y, 1)
    global_mean = X.mean(axis=0)
    centroids = np.empty((num_classes, d), dtype=np.float32)
    for k in range(num_classes):
        if counts[k] > 0:
            centroids[k] = sums[k] / float(counts[k])
        else:
            centroids[k] = global_mean
    return centroids


def _predict_nearest_centroid(X, centroids):
    x2 = (X * X).sum(axis=1, keepdims=True)  # (N,1)
    c2 = (centroids * centroids).sum(axis=1, keepdims=True).T  # (1,K)
    d2 = x2 + c2 - 2.0 * np.dot(X, centroids.T)
    return np.argmin(d2, axis=1)


train_feature_cache = {}

print("Extracting train features (single pass; cached)...")
X_all, y_all = _extract_features_cached(
    train_df, TRAIN_IMG_DIR, IMG_SIZE, train_feature_cache, with_labels=True
)

val_idx = val_df.index.values
tr_idx = tr_df.index.values

id_to_pos = {str(img_id): i for i, img_id in enumerate(train_df["image_id"].values)}

val_positions = np.fromiter(
    (id_to_pos[str(x)] for x in val_df["image_id"].values),
    dtype=np.int64,
    count=len(val_df),
)
tr_positions = np.fromiter(
    (id_to_pos[str(x)] for x in tr_df["image_id"].values),
    dtype=np.int64,
    count=len(tr_df),
)

X_val = X_all[val_positions]
y_val = y_all[val_positions]
X_tr = X_all[tr_positions]
y_tr = y_all[tr_positions]

centroids_valfit = _compute_centroids(X_tr, y_tr, NUM_CLASSES)
pred_val = _predict_nearest_centroid(X_val, centroids_valfit)
val_acc = float((pred_val == y_val).mean()) if len(y_val) else 0.0
print("Validation accuracy (centroid classifier):", val_acc)

print("Fitting final centroids on ALL train features...")
centroids = _compute_centroids(X_all, y_all, NUM_CLASSES)

test_df = sample_sub[["image_id"]].copy()


def _extract_test_features_multiproc(df, img_dir, img_size, workers=4):
    rows = df[["image_id"]].values
    image_ids = [str(rows[i][0]) for i in range(len(rows))]

    def _worker(image_id):
        path = os.path.join(img_dir, image_id)
        img_pil = _safe_open_resize_pil(path, img_size)
        return _extract_feature_vector_from_pil(img_pil, thumb_size=16)

    try:
        import multiprocessing as mp

        workers_eff = max(1, min(int(workers), mp.cpu_count()))
        if workers_eff == 1:
            raise Exception("force single")
        pool = mp.Pool(processes=workers_eff)
        feats = pool.map(_worker, image_ids, chunksize=64)
        pool.close()
        pool.join()
        X = np.vstack(feats).astype(np.float32, copy=False)
        return X
    except Exception:
        X, _ = _extract_features_cached(
            df, img_dir, img_size, cache={}, with_labels=False
        )
        return X


print("Extracting test features...")
X_test = _extract_test_features_multiproc(test_df, TEST_IMG_DIR, IMG_SIZE, workers=4)

pred_idx = _predict_nearest_centroid(X_test, centroids)

pred_labels = np.array([int(index_to_class[int(i)]) for i in pred_idx], dtype=np.int64)
predicted_class_indices_new = pred_labels

print(
    "Pred label distribution (test):",
    pd.Series(predicted_class_indices_new).value_counts().sort_index().to_dict(),
)



## === cell 6
submission = sample_sub[["image_id"]].copy()

assert "predicted_class_indices_new" in globals()
assert len(predicted_class_indices_new) == len(submission)

submission["label"] = predicted_class_indices_new.astype(int)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(submission.head())
print("Rows:", len(submission))
assert out_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
