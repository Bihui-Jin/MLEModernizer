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

3.14

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

0.7417648836506497

# 6. Current score

0.43797

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script now safely handles missing TensorFlow model files and protobuf issues by falling back to a simple baseline that predicts the most frequent label from the training set. It loads the test image filenames, creates a constant prediction array, merges it with the sample submission, and writes a correctly‑named `submission.csv` file, ensuring the notebook runs end‑to‑end without errors. This minimal change fixes the runtime failures while keeping the original workflow structure.'
- What this solution (achieved 0.60874) has done: 'I keep the existing workflow but add a tiny heuristic: compute the most common label for each image‑id prefix (first 4 characters) in the training set and use that as the prediction for test images with the same prefix, falling back to the overall most‑frequent label when the prefix was never seen. This requires only a few extra lines and is expected to raise accuracy toward the target without changing the overall logic.'
- What this solution (achieved 0.60874) has done: 'I add a direct image‑id lookup: if a test image name appears in the training set we use its true label, otherwise we fall back to the existing prefix‑based heuristic (and finally to the overall most‑common label). This tiny lookup often captures a few duplicate images and improves accuracy with virtually no extra computation, moving the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 0.43871) has done: 'I add a cheap image‑based nearest‑neighbor fallback: compute the mean RGB colour of every training image, do the same for each test image, and assign the label of the closest training image. This uses only Pillow (standard in Kaggle), keeps the original file‑handling logic, and is expected to raise accuracy toward the target without altering the overall workflow.'
- What this solution (achieved 0.43797) has done: 'I keep the existing workflow but add cheap, deterministic heuristics before falling back to the mean‑RGB nearest‑neighbor predictions:  
1. Use the true label when a test image id appears in the training set.  
2. If not, use the most frequent label for the image‑id prefix (first 4 characters) learned from the training data.  
3. Otherwise keep the original nearest‑neighbor label.  
These lightweight rules are added after the NN step and are expected to raise the accuracy toward the target while preserving the core logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image

TRAIN_CSV = Path("/kaggle/input/cassava-leaf-disease-classification/train.csv")
TRAIN_IMG_DIR = Path("/kaggle/input/cassava-leaf-disease-classification/train_images")
TEST_DIR = Path("/kaggle/input/cassava-leaf-disease-classification/test_images")
SAMPLE_SUB = Path(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

train_df = pd.read_csv(TRAIN_CSV)

most_common_label = int(Counter(train_df["label"]).most_common(1)[0][0])
print(f"Most common label in training set: {most_common_label}")

train_id_label_map = dict(zip(train_df["image_id"], train_df["label"]))

train_df["prefix"] = train_df["image_id"].str.slice(0, 4)
prefix_label_map = (
    train_df.groupby("prefix")["label"]
    .agg(lambda x: x.value_counts().idxmax())
    .to_dict()
)
print(f"Built prefix label map with {len(prefix_label_map)} entries.")


def mean_rgb(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return arr.mean(axis=(0, 1))


print("Computing mean RGB for training images (may take a few seconds)...")
train_means = np.vstack(
    [mean_rgb(TRAIN_IMG_DIR / img_id) for img_id in train_df["image_id"]]
)  # shape (N_train, 3)
train_labels_array = train_df["label"].values
print(f"Computed mean RGB for {train_means.shape[0]} training images.")



## === cell 1
test_files = sorted(TEST_DIR.glob("*.jpg"))
test_image_ids = [p.name for p in test_files]
print(f"Number of test images found: {len(test_image_ids)}")



## === cell 2
print("Computing mean RGB for test images...")
test_means = np.vstack([mean_rgb(TEST_DIR / img_id) for img_id in test_image_ids])

print("Performing nearest‑neighbor search (mean colour)...")
diff = (
    train_means[:, np.newaxis, :] - test_means[np.newaxis, :, :]
)  # shape (N_train, N_test, 3)
dists = np.linalg.norm(diff, axis=2)  # shape (N_train, N_test)
nn_indices = np.argmin(
    dists, axis=0
)  # index of closest training image for each test image
nn_labels = train_labels_array[nn_indices]

final_labels = nn_labels.copy()

for i, img_id in enumerate(test_image_ids):
    if img_id in train_id_label_map:
        final_labels[i] = train_id_label_map[img_id]

for i, img_id in enumerate(test_image_ids):
    if final_labels[i] == nn_labels[i]:
        prefix = img_id[:4]
        if prefix in prefix_label_map:
            final_labels[i] = prefix_label_map[prefix]

final_labels = np.where(pd.isnull(final_labels), most_common_label, final_labels)

labels = final_labels.astype(int)
print(f"Generated predictions for {len(labels)} test images after heuristics.")



## === cell 3
sub = pd.read_csv(SAMPLE_SUB)
sub["label"] = labels[: len(sub)]  # ensure length matches



## === cell 4
output_path = Path("submission.csv")
sub.to_csv(output_path, index=False)
print(f"submission.csv saved to {output_path.resolve()}")
print(sub.head())
