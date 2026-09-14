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

0.8819885161680266

# 6. Current score

0.62332

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.21637) has done: 'The script is rewritten to remove the failing TensorFlow imports and unavailable saved‑model files, replace them with a lightweight dummy model that generates random predictions, and correctly build the submission CSV using the test image filenames. This fixes the import and loading errors, ensures the variables referenced later exist, and writes a valid `submission.csv` required by Kaggle.'
- What this solution (achieved 0.61099) has done: 'I replace the random‑prediction dummy models with a very simple, deterministic baseline: read the training labels, find the most frequent class, and assign that class to every test image. This keeps the overall pipeline (loading test filenames and writing a CSV) unchanged while improving expected accuracy from ≈0.2 to the majority‑class frequency, moving the score closer to the target.'
- What this solution (achieved 0.43871) has done: 'I replace the naive majority‑class prediction with a lightweight image‑based nearest‑neighbor heuristic: compute the average RGB colour of each training image, then assign each test image the label of the training image whose colour mean is closest. This uses only Pillow (available in the environment) and modest NumPy operations, keeping the overall pipeline unchanged while likely improving accuracy toward the target score.'
- What this solution (achieved 0.51233) has done: 'The changes introduce fast feature extraction using Pillow’s `ImageStat` (avoiding full NumPy array conversion) and parallelize both train‑ and test‑image processing with a thread pool, which dramatically cuts I/O time while keeping the exact mean/std calculations and the original K‑NN voting logic unchanged.'
- What this solution (achieved 0.47384) has done: 'I keep the overall pipeline unchanged but add a simple feature standard‑scaling step (subtract train mean, divide by train std) so that the Euclidean distance is more meaningful, and I set the neighbour count k to 1 (pure nearest‑neighbor) which typically gives a stronger signal for this limited 6‑dim colour feature. These tiny tweaks should raise the validation accuracy toward the target without altering any core logic.'
- What this solution (achieved 0.56577) has done: 'I increase the neighbor count from 1 to 5 and use distance‑weighted voting for the K‑NN classifier. This keeps the original feature extraction and scaling unchanged while providing a modest, well‑behaved boost in accuracy, moving the validation score closer to the target without altering the core pipeline.'
- What this solution (achieved 0.59865) has done: 'I enhance the image feature extractor to include per‑channel colour histograms (8 bins each) in addition to the existing mean / std RGB values, giving a richer 30‑dimensional descriptor. The rest of the pipeline (scaling, weighted k‑NN, fallback handling and CSV output) stays unchanged, so the core logic is preserved while the added information should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.57586) has done: 'I keep the overall pipeline unchanged but make three small, targeted tweaks that are likely to raise the validation accuracy: (1) increase the colour‑histogram resolution from 8 to 16 bins (richer descriptor), (2) use Manhattan (L1) distance instead of Euclidean for the nearest‑neighbour search (often better for histogram‑type features), and (3) shrink the neighbourhood size to k=3 so the prediction relies on the most similar images. These changes preserve the core logic while providing a modest boost toward the target score.'
- What this solution (achieved 0.60912) has done: 'The changes focus on speeding up image feature extraction, which is the main bottleneck. By replacing per‑channel `np.histogram` calls with Pillow’s fast `histogram()` method and reshaping to 32‑bin histograms, we eliminate many Python loops. The same optimization is applied to both training and test images. The rest of the pipeline—including the k‑NN classification and submission generation—remains unchanged, preserving exact model behavior and results.'
- What this solution (achieved 0.62182) has done: 'I increase the colour‑histogram resolution from 32 bins to 64 bins (giving a richer descriptor) and raise the neighbour count to 7 so the voting is a bit more stable. These tweaks keep the overall pipeline unchanged while providing more discriminative features that should raise the validation accuracy toward the target.'
- What this solution (achieved 0.60837) has done: 'I keep the overall pipeline and k‑NN logic but switch the distance metric from Euclidean to Manhattan (L1) and reduce the neighbour count from 7 to 5. Both changes are tiny, preserve the core architecture, and are expected to make the similarity measure more appropriate for the colour‑histogram descriptors, thus nudging the validation accuracy upward toward the target score.'
- What this solution (achieved 0.62332) has done: 'I slightly enrich the colour descriptor by adding a tiny 8×8 grayscale thumbnail (flattened) to each feature vector, keeping the original mean/std and 64‑bin RGB/HSV histograms. This adds spatial information without altering the core k‑NN pipeline. I also raise the neighbour count from 5 to 7, which previously gave a modest boost. These minimal tweaks preserve the overall logic while expected to move the validation accuracy closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageStat  # use ImageStat for fast mean/std
import concurrent.futures  # parallel image processing

np.random.seed(42)



## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)



## === cell 2
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"


def _extract_feature(img_path):
    """Return a richer colour + texture feature.

    Features:
    - mean and std of RGB channels (6 values)
    - 64‑bin RGB histograms (3×64)
    - 64‑bin HSV histograms (3×64)
    - 8×8 grayscale thumbnail (64 values) to capture coarse texture/layout
    """
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            mean_std = stat.mean + stat.stddev  # 6 floats

            rgb_full_hist = np.array(img.histogram(), dtype=np.float32).reshape(3, 256)
            rgb_hist = rgb_full_hist.reshape(3, 64, 4).sum(axis=2) / (
                img.width * img.height
            )

            hsv_img = img.convert("HSV")
            hsv_full_hist = np.array(hsv_img.histogram(), dtype=np.float32).reshape(
                3, 256
            )
            hsv_hist = hsv_full_hist.reshape(3, 64, 4).sum(axis=2) / (
                img.width * img.height
            )

            gray = img.convert("L").resize((8, 8), Image.BILINEAR)
            gray_thumb = np.array(gray, dtype=np.float32).ravel() / 255.0

            feature = np.concatenate(
                [mean_std, rgb_hist.ravel(), hsv_hist.ravel(), gray_thumb]
            ).astype(np.float32)
            return feature
    except Exception:
        return None


train_paths_labels = [
    (os.path.join(train_images_dir, row["image_id"]), int(row["label"]))
    for _, row in train_df.iterrows()
]

train_features = []
train_labels = []
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for feat, (path, label) in zip(
        executor.map(lambda pl: _extract_feature(pl[0]), train_paths_labels),
        train_paths_labels,
    ):
        if feat is not None:
            train_features.append(feat)
            train_labels.append(label)

train_features_arr = np.stack(train_features)  # (N_train, 390+64 = 454)
train_labels_arr = np.array(train_labels, dtype=np.int32)  # (N_train,)

feat_mean = train_features_arr.mean(axis=0, keepdims=True)
feat_std = train_features_arr.std(axis=0, keepdims=True)
feat_std[feat_std == 0] = 1.0
train_features_arr = (train_features_arr - feat_mean) / feat_std



## === cell 3
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_filenames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
num_test = len(test_filenames)

fallback_label = int(train_df["label"].value_counts().idxmax())


def _extract_test_feature(filename):
    img_path = os.path.join(test_dir, filename)
    return _extract_feature(img_path)


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    test_features_list = list(executor.map(_extract_test_feature, test_filenames))

test_features_arr = np.array(
    [
        (
            feat
            if feat is not None
            else np.full(train_features_arr.shape[1], np.nan, dtype=np.float32)
        )
        for feat in test_features_list
    ]
)

test_features_arr = (test_features_arr - feat_mean) / feat_std

k = 7  # slightly larger neighbourhood than before

predictions = []

for idx, test_feat in enumerate(test_features_arr):
    if np.isnan(test_feat).any():
        pred_label = fallback_label
    else:
        dists = np.sum(np.abs(train_features_arr - test_feat), axis=1)
        kn = min(k, len(dists))
        nearest_idxs = np.argpartition(dists, kn - 1)[:kn]
        nearest_labels = train_labels_arr[nearest_idxs]
        nearest_dists = dists[nearest_idxs]
        weights = 1.0 / (nearest_dists + 1e-8)
        counts = np.bincount(nearest_labels, weights=weights, minlength=5)
        pred_label = int(np.argmax(counts))
    predictions.append(pred_label)



## === cell 4
submission_path = "/kaggle/working/submission.csv"
results = pd.DataFrame({"image_id": test_filenames, "label": predictions})
results.to_csv(submission_path, index=False)
