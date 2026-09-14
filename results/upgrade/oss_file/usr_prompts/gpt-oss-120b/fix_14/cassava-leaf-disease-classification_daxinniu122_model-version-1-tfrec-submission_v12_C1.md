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

0.8541855545482019

# 6. Current score

0.6136

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.67526) has done: 'Implemented batch loading and prediction for the test set to eliminate the per‑image `model.predict` call, which was the primary cause of the timeout. Images are now read in chunks (default batch = 32), stacked once per batch, and passed to the model in a single forward pass, preserving identical predictions while dramatically reducing Python‑level overhead. Minor clean‑up (pre‑allocation of result list and clearing Keras session) keeps memory usage predictable without altering model architecture or training logic.'
- What this solution (achieved 0.61734) has done: 'The fix replaces the TensorFlow pipeline (which caused import and API errors) with a lightweight scikit‑learn RandomForest model that loads images via Pillow, extracts simple pixel‑level features, and trains on the integer labels directly. The data generators are removed, the label column is kept as integers, and the prediction step no longer passes unsupported arguments. The script now writes a correctly formatted CSV submission file, ensuring end‑to‑end execution without runtime errors while preserving the original task logic.'
- What this solution (achieved 0.6136) has done: 'I keep the overall pipeline (loading images, preprocessing, training a tree‑based model, predicting and writing a CSV) unchanged but improve the feature representation and use a stronger ensemble. The image loader now adds per‑channel mean and standard‑deviation to the flattened pixel vector, giving the model extra colour statistics. I also switch to an `ExtraTreesClassifier` with more trees (500) and enable class‑weight balancing, which typically yields higher accuracy for high‑dimensional image data while staying within the same core‑logic family.'
- What this solution (achieved 0.6136) has done: 'The changes keep the same model and feature extraction but speed up the heavy parts: the image‑loading pool now uses a larger `chunksize` and limits workers to avoid overhead, and the `ExtraTreesClassifier` keeps all trees while using the default `max_features='sqrt'` (much faster than 0.5 of the features) which does not alter the algorithmic approach. These tweaks reduce CPU work and I/O bottlenecks, fitting the 600‑second limit without affecting prediction semantics.'

# 9. Code solution

## === cell 0
import os, random, warnings, multiprocessing, gc
import numpy as np, pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
OUTPUT_SUB = "/kaggle/working/submission.csv"

IMG_SIZE = (96, 96)  # width, height
BATCH_SIZE = 32




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)


def load_and_preprocess(img_path):
    """Load an image, resize, and return an enriched feature vector:
    flattened pixel values + per‑channel mean, std, min, max."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
        flat = arr.reshape(-1).astype(np.float32)  # pixel intensities
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        min_rgb = arr.min(axis=(0, 1))
        max_rgb = arr.max(axis=(0, 1))
        return np.concatenate([flat, mean_rgb, std_rgb, min_rgb, max_rgb], axis=0)


train_image_paths = [
    os.path.join(TRAIN_IMG_DIR, fname) for fname in train_df["image_id"]
]

num_workers = min(multiprocessing.cpu_count(), 4)
chunks = 256
with multiprocessing.Pool(processes=num_workers) as pool:
    first_feat = load_and_preprocess(train_image_paths[0])
    feat_len = first_feat.shape[0]
    X_train = np.empty((len(train_image_paths), feat_len), dtype=np.float32)
    for idx, feat in enumerate(
        pool.imap(load_and_preprocess, train_image_paths, chunksize=chunks)
    ):
        X_train[idx] = feat

y_train = train_df["label"].astype(int).values

rf_clf = ExtraTreesClassifier(
    n_estimators=800,
    max_depth=None,
    max_features="sqrt",  # faster split search, algorithmic approach unchanged
    random_state=SEED,
    n_jobs=multiprocessing.cpu_count(),
    class_weight="balanced",
    bootstrap=False,
)
rf_clf.fit(X_train, y_train)

del X_train
gc.collect()




## === cell 2
test_df = pd.read_csv(SAMPLE_SUB)[["image_id"]]

test_image_paths = [os.path.join(TEST_IMG_DIR, fname) for fname in test_df["image_id"]]

with multiprocessing.Pool(processes=num_workers) as pool:
    X_test = np.empty((len(test_image_paths), feat_len), dtype=np.float32)
    for idx, feat in enumerate(
        pool.imap(load_and_preprocess, test_image_paths, chunksize=chunks)
    ):
        X_test[idx] = feat

pred_labels = rf_clf.predict(X_test).astype(int)




## === cell 3
submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
submission.to_csv(OUTPUT_SUB, index=False)
print(f"Submission saved to {OUTPUT_SUB}")
