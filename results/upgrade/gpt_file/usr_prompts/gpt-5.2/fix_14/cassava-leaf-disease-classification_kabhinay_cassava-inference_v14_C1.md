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

0.60389

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the environment/runtime failures first by removing the TensorFlow import from the initial “version print” cell (it triggers the `MessageFactory.GetPrototype` crash) and making the script self-contained using only the local Cassava dataset. Then I fix the data generator error by converting `train_df["label"]` to string for `class_mode="categorical"`, and I ensure `NUM_CLASSES` is defined before it’s used so the fallback model path runs. Finally, I make inference robust by always producing a prediction array aligned to `sample_submission.csv` and writing `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.1293) has done: 'We fix the TensorFlow import/runtime crash (`MessageFactory.GetPrototype`) by avoiding TF entirely and switching to a minimal, self-contained image pipeline using only built-in/standard Kaggle-available libraries (PIL + numpy + pandas), while keeping the “train a small CNN then predict” core intent intact. To ensure the notebook always produces a valid `submission.csv`, we make inference robust and guaranteed to align to `sample_submission.csv` row order. Since your current run yields no submission, the primary goal is to unblock end-to-end execution; the simple model be trained deterministically and produce plausible predictions (better than random) without external models. All paths remain the same and the output format is validated.'
- What this solution (achieved 0.18348) has done: 'The timeout is dominated by repeatedly opening/decoding/resizing 18.7k JPEGs multiple times (train split, val split, full train again, then test), plus expensive PIL↔NumPy roundtrips inside feature extraction. I preserve the exact feature definition and centroid-classifier logic, but make extraction single-pass and cached so each train image is decoded only once, then reused for both validation and final centroid fitting. I also remove the redundant PIL reconstruction by computing the 16×16 thumbnail directly from the already-loaded PIL image, and speed up I/O via list-based iteration and optional multiprocessing with deterministic ordering. Finally, I compute centroids via vectorized `np.add.at` (exact sums/means) and compute distances using the algebraic expansion to reduce temporary allocations.'
- What this solution (achieved 0.1861) has done: 'The timeout is dominated by PIL decoding/resizing and per-image thumbnail extraction across ~21k images; we keep the exact same feature vector and nearest-centroid classifier but reduce overhead around those operations. Specifically, we (1) avoid slow EXIF parsing by using `ImageOps.exif_transpose` directly, (2) ensure images are closed promptly to prevent file-handle churn, (3) reduce multiprocessing IPC overhead by sending only `image_id` strings (not `(i,id)` tuples) and returning fixed-size arrays that we place by deterministic index, and (4) use a faster, deterministic worker-count selection and larger chunk sizes to cut scheduling overhead. All math (features, standardization, centroids, distance, predictions) remains identical aside from negligible floating-point ordering differences from parallelism.'
- What this solution (achieved 0.60389) has done: 'Your current score (0.1861) is far below the target (0.8634), so we should improve the predictive signal without changing the overall “fast handcrafted features + simple classifier” core logic. The biggest quality issue is that the current nearest-centroid classifier on raw thumbnail/mean/std features is too weak; we can keep the same features and training flow but swap the classifier to a closed-form regularized multinomial logistic regression (softmax regression) trained on the standardized features. This is still the same overall approach (extract features → standardize → train lightweight classifier → predict), but it typically boosts accuracy substantially on this dataset while staying CPU-only and under the time budget. We keep the same single-pass cached feature extraction and submission alignment; only the classifier step changes.'

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
from PIL import Image, ImageFile, ImageOps

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

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    ImageFile.MAXBLOCK = 2**20  # larger blocks can speed some decodes
except Exception:
    pass


def _safe_open_resize_pil(path, img_size):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
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


def _fit_standardizer(X, eps=1e-6):
    mu = X.mean(axis=0).astype(np.float32, copy=False)
    sigma = X.std(axis=0).astype(np.float32, copy=False)
    sigma = np.maximum(sigma, eps).astype(np.float32, copy=False)
    return mu, sigma


def _apply_standardizer(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32, copy=False)


def _softmax_stable(logits):
    m = logits.max(axis=1, keepdims=True)
    exps = np.exp(logits - m)
    return exps / exps.sum(axis=1, keepdims=True)


def _train_softmax_regression(
    X, y, num_classes, epochs=120, lr=0.25, l2=1e-3, batch_size=1024, seed=0
):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    W = np.zeros((d, num_classes), dtype=np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)

    for ep in range(int(epochs)):
        idx = rng.permutation(n)
        for s in range(0, n, int(batch_size)):
            batch = idx[s : s + int(batch_size)]
            Xb = X[batch]
            yb = y[batch]

            logits = np.dot(Xb, W) + b  # (B,K)
            P = _softmax_stable(logits)  # (B,K)

            P[np.arange(len(batch)), yb] -= 1.0
            P /= float(len(batch))

            gW = np.dot(Xb.T, P) + l2 * W
            gb = P.sum(axis=0)

            W -= lr * gW.astype(np.float32, copy=False)
            b -= lr * gb.astype(np.float32, copy=False)

        if (ep + 1) in (40, 80, 110):
            lr *= 0.5

    return W, b


def _predict_softmax_regression(X, W, b):
    logits = np.dot(X, W) + b
    return np.argmax(logits, axis=1)


_WORKER_IMG_DIR = None
_WORKER_IMG_SIZE = None
_WORKER_THUMB = None


def _mp_init(img_dir, img_size, thumb_size):
    global _WORKER_IMG_DIR, _WORKER_IMG_SIZE, _WORKER_THUMB
    _WORKER_IMG_DIR = img_dir
    _WORKER_IMG_SIZE = img_size
    _WORKER_THUMB = thumb_size


def _mp_worker_with_index(i_imageid):
    i, image_id = i_imageid
    path = os.path.join(_WORKER_IMG_DIR, image_id)
    img_pil = _safe_open_resize_pil(path, _WORKER_IMG_SIZE)
    feat = _extract_feature_vector_from_pil(img_pil, thumb_size=_WORKER_THUMB)
    return i, feat


def _extract_features_multiproc_prealloc(
    image_ids, img_dir, img_size, workers=4, chunksize=1024, thumb_size=16
):
    n = len(image_ids)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    path0 = os.path.join(img_dir, image_ids[0])
    img0 = _safe_open_resize_pil(path0, img_size)
    feat0 = _extract_feature_vector_from_pil(img0, thumb_size=thumb_size)
    d = int(feat0.shape[0])

    X = np.empty((n, d), dtype=np.float32)
    X[0, :] = feat0

    try:
        import multiprocessing as mp

        cpu = mp.cpu_count()
        workers_eff = int(max(1, min(int(workers), cpu)))
        if workers_eff <= 1 or n < 512:
            raise Exception("use single-process path")

        pool = mp.Pool(
            processes=workers_eff,
            initializer=_mp_init,
            initargs=(img_dir, img_size, thumb_size),
            maxtasksperchild=0,
        )
        try:
            it = pool.imap_unordered(
                _mp_worker_with_index,
                ((i, image_ids[i]) for i in range(1, n)),
                chunksize=int(chunksize),
            )
            done = 1
            for i, feat in it:
                X[i, :] = feat
                done += 1
                if done % 4000 == 0:
                    print("Processed {}/{} images for features...".format(done, n))
        finally:
            pool.close()
            pool.join()
        return X
    except Exception:
        for i in range(1, n):
            path = os.path.join(img_dir, image_ids[i])
            img_pil = _safe_open_resize_pil(path, img_size)
            X[i, :] = _extract_feature_vector_from_pil(img_pil, thumb_size=thumb_size)
            if (i + 1) % 4000 == 0:
                print("Processed {}/{} images for features...".format(i + 1, n))
        return X


print("Extracting train features (single pass)...")
train_image_ids = train_df["image_id"].astype(str).values.tolist()

X_all = _extract_features_multiproc_prealloc(
    train_image_ids, TRAIN_IMG_DIR, IMG_SIZE, workers=4, chunksize=1024, thumb_size=16
)

y_all = np.asarray(
    [class_to_index[str(s)] for s in train_df["label"].values], dtype=np.int64
)

train_ids = train_df["image_id"].astype(str).values
id_index = pd.Index(train_ids)

val_positions = id_index.get_indexer(val_df["image_id"].astype(str).values).astype(
    np.int64, copy=False
)
tr_positions = id_index.get_indexer(tr_df["image_id"].astype(str).values).astype(
    np.int64, copy=False
)

X_val = X_all[val_positions]
y_val = y_all[val_positions]
X_tr = X_all[tr_positions]
y_tr = y_all[tr_positions]

mu_tr, sigma_tr = _fit_standardizer(X_tr)
X_tr_s = _apply_standardizer(X_tr, mu_tr, sigma_tr)
X_val_s = _apply_standardizer(X_val, mu_tr, sigma_tr)

print("Training softmax regression on train split...")
W_tr, b_tr = _train_softmax_regression(
    X_tr_s, y_tr, NUM_CLASSES, epochs=120, lr=0.25, l2=1e-3, batch_size=1024, seed=0
)
pred_val = _predict_softmax_regression(X_val_s, W_tr, b_tr)
val_acc = float((pred_val == y_val).mean()) if len(y_val) else 0.0
print("Validation accuracy (softmax regression):", val_acc)

print("Training final softmax regression on ALL train features...")
mu_all, sigma_all = _fit_standardizer(X_all)
X_all_s = _apply_standardizer(X_all, mu_all, sigma_all)

W_all, b_all = _train_softmax_regression(
    X_all_s, y_all, NUM_CLASSES, epochs=140, lr=0.25, l2=1e-3, batch_size=1024, seed=0
)

test_df = sample_sub[["image_id"]].copy()

print("Extracting test features...")
test_image_ids = test_df["image_id"].astype(str).values.tolist()
X_test = _extract_features_multiproc_prealloc(
    test_image_ids, TEST_IMG_DIR, IMG_SIZE, workers=4, chunksize=1024, thumb_size=16
)
X_test_s = _apply_standardizer(X_test, mu_all, sigma_all)

pred_idx = _predict_softmax_regression(X_test_s, W_all, b_all)

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
