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

0.8686914475672408

# 6. Current score

0.27728

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The immediate crash comes from importing TensorFlow in an environment where protobuf/TensorFlow are incompatible, plus the script also references three external `/kaggle/input/...` model datasets that are not present, so inference can never run. To make the notebook run end-to-end and still produce a valid `submission.csv`, I remove the TensorFlow dependency entirely and replace the missing-model ensemble with a deterministic, legitimate baseline that predicts the most frequent class from `train.csv` (this preserves correct submission semantics for an accuracy metric). I also update the cells to be Python 2.7 compatible (no `tf.random.set_seed`, no TF/Keras generators), keep the original cell order, and ensure the output file is written to `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.11883) has done: 'Your current score (0.61099) is far below the target (0.86869), so we should improve accuracy while keeping your “no-TensorFlow” baseline structure. The smallest legitimate upgrade that preserves your overall approach (train.csv-driven inference and producing `submission.csv`) is to replace the constant majority-class predictor with a simple image-driven heuristic: compute per-class mean RGB “prototype” vectors from the training images, then classify each test image by nearest prototype in RGB space. This avoids any new ML libraries, keeps runtime under the limit by downsampling images, and should move accuracy substantially upward compared with a constant guess while keeping the rest of the pipeline intact. I also keep your existing cells (bi-tempered/gambler placeholders) and only add minimal image reading logic using PIL (with a safe fallback if PIL is unavailable).'
- What this solution (achieved 0.23019) has done: 'Your current score (0.11883) is far below the target (0.86869), so we should increase accuracy with the smallest changes that keep your no-TensorFlow, prototype-based inference core intact. The main issue is that a single mean-RGB prototype per class is too weak; we can significantly improve it by computing multiple prototypes per class (K-means in RGB-mean space implemented with NumPy only) and classifying by nearest prototype, while still using the same “downsample image -> mean RGB feature -> nearest prototype” logic. To keep runtime safe under 600s, we (1) avoid recomputing train features repeatedly, (2) cap per-class samples, and (3) compute test features once then vectorize distance computation per chunk. We also keep the majority-label fallback to guarantee a valid submission if PIL isn’t available or images fail to load.'
- What this solution (achieved 0.27728) has done: 'Your current score (0.23019) is far below the target (0.86869), so we need a legitimate accuracy jump while keeping the same “downsample image → simple feature → nearest prototype” core logic. The smallest high-impact change is to enrich the feature from mean RGB (3D) to a tiny “color+texture” descriptor (mean+std per channel = 6D), and to make prototype matching more robust by whitening (z-scoring) features using train statistics before K-means and nearest-prototype search. This keeps the same pipeline structure (feature extraction + per-class K-means prototypes + nearest prototype inference), but usually improves separability a lot for cassava leaves. I also make the training subset selection deterministic but more representative via stratified sampling without changing caps or adding new dependencies, and keep the same submission writing logic.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

np.random.seed(42)

print("Python:", sys.version)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.isdir(DATA_DIR):
    alt = "/kaggle/data/cassava-leaf-disease-classification"
    if os.path.isdir(alt):
        DATA_DIR = alt

print("Using DATA_DIR:", DATA_DIR)



## === cell 1
"""
Bi-Tempered loss implementation (kept for minimal structural change from the original notebook).

Not used in this no-TensorFlow inference-only baseline because TensorFlow cannot be imported
in this environment (protobuf incompatibility). Kept to preserve original cell structure.
"""
pass



## === cell 2
"""
Gambler's loss helpers (kept for minimal structural change from the original notebook).

Not used in this no-TensorFlow inference-only baseline because TensorFlow cannot be imported.
"""
pass



## === cell 3
"""
Original notebook tried to load external SavedModels via TFSMLayer from datasets that are not present.

Change (score-improving, minimal core-logic impact):
- Keep the same image -> downsample -> simple feature -> prototype classifier approach.
- Upgrade feature from mean RGB (3D) to mean+std RGB (6D) to add lightweight texture/contrast info.
- Normalize (z-score) features using train subset stats before K-means and inference.
- Keep multiple prototypes per class via tiny NumPy-only K-means and nearest-prototype prediction.
This preserves evaluation semantics (predict class labels) and stays within Kaggle constraints.
"""
train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_img_dir = os.path.join(DATA_DIR, "train_images")
test_img_dir = os.path.join(DATA_DIR, "test_images")

for p in [train_csv_path, sample_path]:
    if not os.path.exists(p):
        raise IOError("Missing required file at: {}".format(p))

if not os.path.isdir(train_img_dir):
    alt_train = os.path.join(
        DATA_DIR, "cassava-leaf-disease-classification", "train_images"
    )
    if os.path.isdir(alt_train):
        train_img_dir = alt_train
if not os.path.isdir(test_img_dir):
    alt_test = os.path.join(
        DATA_DIR, "cassava-leaf-disease-classification", "test_images"
    )
    if os.path.isdir(alt_test):
        test_img_dir = alt_test

if not os.path.isdir(train_img_dir):
    raise IOError("Missing train_images directory at: {}".format(train_img_dir))
if not os.path.isdir(test_img_dir):
    raise IOError("Missing test_images directory at: {}".format(test_img_dir))

train_df = pd.read_csv(train_csv_path)
if "label" not in train_df.columns or "image_id" not in train_df.columns:
    raise ValueError("train.csv must contain columns: image_id, label")

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())

print("Train rows:", train_df.shape[0])
print("Label distribution:\n{}".format(label_counts.to_string()))
print("Majority label (fallback):", majority_label)



## === cell 4
"""
Compute features and multiple prototypes (K-means) per class.

Why this should move score toward target (minimal change while keeping the same core logic):
- Mean RGB alone is too weak; mean+std RGB (6D) is still extremely cheap but more informative.
- Z-scoring (whitening) prevents any single dimension from dominating distance/K-means.
- Still: per-class K-means prototypes + nearest-prototype inference (same classifier family).
"""
try:
    from PIL import Image

    PIL_OK = True
except Exception as e:
    PIL_OK = False
    pil_err = str(e)


def image_feat_mean_std_rgb(path, size=(96, 96)):
    """
    Returns 6D feature: [meanR,meanG,meanB,stdR,stdG,stdB]
    """
    if not PIL_OK:
        return None
    try:
        im = Image.open(path).convert("RGB")
        if size is not None:
            im = im.resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32).reshape((-1, 3))
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        return np.concatenate([mu, sd], axis=0).astype(np.float32)
    except Exception:
        return None


def kmeans_np(X, k, n_iter=25, seed=42):
    """
    Tiny NumPy-only K-means for low-dimensional data.
    X: (N,D)
    Returns centers: (k,D)
    """
    rs = np.random.RandomState(seed)
    N = X.shape[0]
    if N == 0:
        return None
    if N <= k:
        centers = X.copy()
        if centers.shape[0] < k:
            pad = centers[rs.randint(0, centers.shape[0], size=(k - centers.shape[0],))]
            centers = np.vstack([centers, pad])
        return centers.astype(np.float32)

    idx = rs.choice(N, size=k, replace=False)
    centers = X[idx].astype(np.float32)

    for _ in range(n_iter):
        d = X[:, None, :] - centers[None, :, :]
        dist2 = (d * d).sum(axis=2)
        assign = dist2.argmin(axis=1)

        new_centers = np.empty_like(centers)
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = X[rs.randint(0, N)]
        centers = new_centers.astype(np.float32)

    return centers.astype(np.float32)


if not PIL_OK:
    print(
        "WARNING: PIL not available ({}). Falling back to majority-class predictions.".format(
            pil_err
        )
    )
    proto_centers = None
    proto_labels = None
    feat_mu = None
    feat_sd = None
else:
    img_size = (96, 96)

    per_class_cap = 1400

    k_per_class = 4
    kmeans_iters = 25

    class_ids = sorted(train_df["label"].unique().tolist())

    parts = []
    for c in class_ids:
        dfc = train_df[train_df["label"] == c]
        n = int(min(per_class_cap, dfc.shape[0]))
        if n < dfc.shape[0]:
            dfc = dfc.sample(frac=1.0, random_state=42).head(n)
        parts.append(dfc)
    sub_train = pd.concat(parts, axis=0).reset_index(drop=True)
    print("Prototype training subset size:", sub_train.shape[0])

    feats = []
    labs = []
    missing = 0
    for i in range(sub_train.shape[0]):
        img_id = sub_train.at[i, "image_id"]
        c = int(sub_train.at[i, "label"])
        p = os.path.join(train_img_dir, img_id)
        f = image_feat_mean_std_rgb(p, size=img_size)
        if f is None:
            missing += 1
            continue
        feats.append(f)
        labs.append(c)

    if len(feats) == 0:
        print(
            "WARNING: no training features could be read; falling back to majority-class."
        )
        proto_centers = None
        proto_labels = None
        feat_mu = None
        feat_sd = None
    else:
        X = np.stack(feats, axis=0).astype(np.float32)  # (N,6)
        y = np.array(labs, dtype=np.int64)
        print("Training features shape:", X.shape, "Unreadable/missing:", missing)

        feat_mu = X.mean(axis=0).astype(np.float32)
        feat_sd = X.std(axis=0).astype(np.float32)
        feat_sd = np.maximum(feat_sd, 1e-6).astype(np.float32)

        Xn = (X - feat_mu[None, :]) / feat_sd[None, :]

        centers_list = []
        labels_list = []
        for c in class_ids:
            Xc = Xn[y == int(c)]
            if Xc.shape[0] == 0:
                continue
            centers = kmeans_np(
                Xc, k=k_per_class, n_iter=kmeans_iters, seed=42 + int(c)
            )
            if centers is None:
                continue
            centers_list.append(centers)
            labels_list.append(np.full((centers.shape[0],), int(c), dtype=np.int64))

        if len(centers_list) == 0:
            proto_centers = None
            proto_labels = None
        else:
            proto_centers = np.vstack(centers_list).astype(
                np.float32
            )  # (M,6) in normalized space
            proto_labels = np.concatenate(labels_list).astype(np.int64)  # (M,)
            print(
                "Total prototypes:",
                proto_centers.shape[0],
                "Prototype labels:",
                np.unique(proto_labels),
            )



## === cell 5
"""
Load test list from sample_submission.csv (same as before).
"""
sample_sub = pd.read_csv(sample_path)
if "image_id" not in sample_sub.columns or "label" not in sample_sub.columns:
    raise ValueError("sample_submission.csv must contain columns: image_id, label")

test_df = sample_sub[["image_id"]].copy()
print("Test images:", len(test_df))



## === cell 6
"""
Generate predictions for each test image.

Why this should improve score (minimal change):
- Same nearest-prototype decision rule, but done in a better feature space (mean+std, normalized).
- Vectorized nearest-prototype assignment in chunks as before.
"""
if proto_centers is None or proto_labels is None or feat_mu is None or feat_sd is None:
    predicted_class_indices_new = np.full(
        (len(test_df),), majority_label, dtype=np.int64
    )
else:
    img_size = (96, 96)
    test_feats = np.empty((len(test_df), 6), dtype=np.float32)
    ok_mask = np.ones((len(test_df),), dtype=np.bool_)
    unreadable = 0

    for i in range(len(test_df)):
        img_id = test_df.at[i, "image_id"]
        p = os.path.join(test_img_dir, img_id)
        f = image_feat_mean_std_rgb(p, size=img_size)
        if f is None:
            ok_mask[i] = False
            test_feats[i] = 0.0
            unreadable += 1
        else:
            test_feats[i] = f

    test_feats = (test_feats - feat_mu[None, :]) / feat_sd[None, :]

    preds = np.full((len(test_df),), majority_label, dtype=np.int64)

    chunk = 512
    for s in range(0, len(test_df), chunk):
        e = min(len(test_df), s + chunk)
        Xb = test_feats[s:e]  # (B,6)
        maskb = ok_mask[s:e]
        if not maskb.any():
            continue
        d = Xb[:, None, :] - proto_centers[None, :, :]
        dist2 = (d * d).sum(axis=2)
        nn = dist2.argmin(axis=1)
        pb = proto_labels[nn]
        out = preds[s:e]
        out[maskb] = pb[maskb]
        preds[s:e] = out

    predicted_class_indices_new = preds
    print(
        "Unreadable/missing test images (fell back to majority for those):", unreadable
    )

print(
    "Predictions generated. Unique predicted labels:",
    np.unique(predicted_class_indices_new),
)



## === cell 7
"""
Write submission.csv in the required format (unchanged).
"""
submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_new}
)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
