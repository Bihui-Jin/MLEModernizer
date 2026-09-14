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

3.9

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

0.7518887881535207

# 6. Current score

0.571

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow imports and the missing model file, replace them with a lightweight baseline that predicts the most frequent class from the training set, and ensure the script correctly creates and saves `submission.csv` with the required columns.'
- What this solution (achieved 0.43909) has done: 'I replace the constant‑label baseline with a lightweight nearest‑neighbor classifier that uses each image’s average RGB colour as a feature. By comparing test images to the labelled training images we can assign more informative labels, which should raise accuracy toward the target while keeping the overall logic simple and preserving the required CSV output. The changes only add feature extraction and nearest‑neighbor prediction; all other pipeline steps and file paths remain unchanged.'
- What this solution (achieved 0.54036) has done: 'The changes parallelize image feature extraction and replace the pure‑Python distance loop with scikit‑learn’s exact nearest‑neighbour search, which runs in optimized C code. This keeps the feature definition, voting, and tie‑breaking logic identical while dramatically cutting runtime. No algorithmic approximations are added and the output format stays unchanged.'
- What this solution (achieved 0.6136) has done: 'I add feature scaling with StandardScaler so that the Euclidean distances used by the nearest‑neighbour model are more meaningful, and I increase k to 7 (neighbour count) which usually gives a slightly better voting decision. These minimal tweaks keep the original pipeline intact while moving the validation accuracy upward toward the target score.'
- What this solution (achieved 0.60052) has done: 'I slightly adjust the k‑Nearest‑Neighbour classifier to use fewer neighbours (k = 5) and replace the simple majority vote with a weighted‑vote that gives each neighbour a weight proportional to the inverse of its distance. This keeps the overall pipeline unchanged while making the distance information more influential, which should raise the validation accuracy toward the target.'
- What this solution (achieved 0.5654) has done: 'The script was slowed mainly by the heavy use of a limited‑size `ProcessPoolExecutor`, which incurs process start‑up and data‑pickling costs for each image. We expand the pool to use all available CPUs and switch to a `ThreadPoolExecutor`, which avoids the inter‑process pickling overhead (PIL releases the GIL during I/O). These changes keep the exact feature‑extraction logic, ordering, and neighbour voting unchanged, so results are identical while runtime drops well below the 600 s limit.'
- What this solution (achieved 0.60239) has done: 'I adjust the neighbour search to use a smaller k and a Euclidean metric, and replace the exponential‑decay weighting with a more standard inverse‑distance weighting. These tweaks keep the overall pipeline unchanged while giving the classifier a sharper, more locally‑focused decision making, which should raise validation accuracy toward the target.'
- What this solution (achieved 0.571) has done: 'I enhance the feature representation by adding a low‑resolution (32×32) colour flattening to each image, which gives the classifier more visual detail while keeping the existing histogram‑based features. I also increase the neighbour count to 7, a value that past trials showed a modest gain with inverse‑distance voting. These adjustments stay within the same K‑NN pipeline and are expected to raise validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor  # use threads instead of processes
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
DEBUG = False
np.random.seed(SEED)




## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = "data/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)

id_to_label = dict(zip(train_df["image_id"], train_df["label"]))
train_id_set = set(id_to_label.keys())  # fast membership test




## === cell 2
MAX_WORKERS = os.cpu_count() or 1
CHUNKSIZE = 5000  # unchanged – suitable for both executors


def _list_image_paths(root):
    return [
        entry.path
        for entry in os.scandir(root)
        if entry.is_file()
        and entry.name.lower().endswith(".jpg")
        and entry.name in train_id_set
    ]


train_images = sorted(
    _list_image_paths("../input/cassava-leaf-disease-classification/train_images")
)
if not train_images:
    train_images = sorted(
        _list_image_paths("data/cassava-leaf-disease-classification/train_images")
    )


def _hist16(channel_arr):
    """Return a 16‑bin density histogram for a single channel."""
    bins = (channel_arr // 16).ravel()
    hist = np.bincount(bins, minlength=16).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _extract_features(img):
    """Compute colour statistics, histograms and a low‑res flattening."""
    arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
    mean_rgb = arr.mean(axis=(0, 1), dtype=np.float32)
    std_rgb = arr.std(axis=(0, 1), dtype=np.float32)
    hist = [_hist16(arr[:, :, ch]) for ch in range(3)]
    low_res = img.resize((32, 32), Image.BILINEAR)
    low_res_arr = np.array(low_res, dtype=np.uint8).ravel().astype(np.float32) / 255.0
    feat = np.concatenate([mean_rgb, std_rgb, np.concatenate(hist), low_res_arr])
    return feat  # shape (54 + 3072,) = (3126,)


def _process_train(path):
    img_id = os.path.basename(path)
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
        feat = _extract_features(img)
        return (img_id, feat.astype(np.float32))
    except Exception:
        return None


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    results = list(exe.map(_process_train, train_images, chunksize=CHUNKSIZE))

train_ids = []
train_features = []
for res in results:
    if res is not None:
        img_id, feat = res
        train_ids.append(img_id)
        train_features.append(feat)

train_features = np.vstack(train_features)  # (N_train, 3126), float32
train_labels = np.array([id_to_label[i] for i in train_ids])

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features.astype(np.float32))




## === cell 3
test_images = sorted(
    [
        entry.path
        for entry in os.scandir(
            "../input/cassava-leaf-disease-classification/test_images"
        )
        if entry.is_file() and entry.name.lower().endswith(".jpg")
    ]
)
if not test_images:
    test_images = sorted(
        [
            entry.path
            for entry in os.scandir(
                "data/cassava-leaf-disease-classification/test_images"
            )
            if entry.is_file() and entry.name.lower().endswith(".jpg")
        ]
    )


def _process_test(path):
    img_id = os.path.basename(path)
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
        feat = _extract_features(img)
        return (img_id, feat.astype(np.float32))
    except Exception:
        return (img_id, np.full(train_features.shape[1], np.nan, dtype=np.float32))


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    test_results = list(exe.map(_process_test, test_images, chunksize=CHUNKSIZE))

test_ids = [img_id for img_id, _ in test_results]
test_features = np.vstack([feat for _, feat in test_results])  # (N_test, 3126)




## === cell 4
most_common_label = train_df["label"].mode()[0]

k = 7  # increased neighbour count for a slightly richer vote
nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
nn.fit(train_features)

nan_mask = np.isnan(test_features).any(axis=1)

valid_features = test_features[~nan_mask]
valid_features = scaler.transform(valid_features.astype(np.float32))

distances, indices = nn.kneighbors(valid_features, return_distance=True)

pred_labels = []


def _vote_inverse_distance(neigh_idx, neigh_dists):
    """Weighted vote where weight = 1 / (distance + eps)."""
    eps = 1e-8
    neigh_labels = train_labels[neigh_idx]
    weights = 1.0 / (neigh_dists + eps)
    label_weights = {}
    for lbl, w in zip(neigh_labels, weights):
        label_weights[lbl] = label_weights.get(lbl, 0.0) + w

    max_weight = max(label_weights.values())
    candidates = [lbl for lbl, w in label_weights.items() if w == max_weight]
    if len(candidates) == 1:
        return candidates[0]
    for lbl in neigh_labels:
        if lbl in candidates:
            return lbl
    return candidates[0]


for neigh_idx_row, neigh_dist_row in zip(indices, distances):
    pred_labels.append(_vote_inverse_distance(neigh_idx_row, neigh_dist_row))

pred_labels.extend([most_common_label] * nan_mask.sum())




## === cell 5
submission = pd.DataFrame({"image_id": test_ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)




## === cell 6
print(submission.head())
