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

0.8661226956784527

# 6. Current score

0.59081

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix execution by (1) addressing the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, and (2) removing the dependency on the missing `/kaggle/input/f-models` models by switching to a minimal, local baseline that still produces a valid `submission.csv`. To keep changes minimal and stable, the baseline predict the majority class from `train.csv`, which is score-reasonable for Cassava and avoids training time. I also make the data path resolution robust to either `/kaggle/input/...` or `/kaggle/data/...` layouts while keeping the existing intended paths. Finally, I ensure the submission file is written with the exact required columns and row alignment.'
- What this solution (achieved 0.41741) has done: 'The crash happens before any training/prediction because importing TensorFlow triggers a protobuf incompatibility (`MessageFactory.GetPrototype` missing). Since this baseline doesn’t actually use TensorFlow/Keras, the smallest and most stable fix is to remove the TensorFlow/Keras imports entirely so the pipeline runs end-to-end and writes `submission.csv`. To move the score toward your target without changing the “no-training baseline” approach, I replace the pure majority-class guess with a slightly better prior-based predictor: sample labels according to the empirical class distribution from `train.csv` (seeded for determinism). This keeps the same simple semantics (no image usage, no model training) but should improve expected accuracy vs always predicting the single majority class. The script still validate paths and submission format and write a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current solution samples labels from the global class prior, which has expected accuracy equal to the sum of squared class frequencies and is typically low for Cassava—explaining the 0.417 score. To move toward the 0.866 target with minimal change and without introducing training loops or image modeling, I keep the “no-training baseline” approach but switch to a deterministic, higher-accuracy heuristic: always predict the majority class from `train.csv`. This preserves the same overall semantics (no image use, no model), is stable/deterministic, and should substantially increase accuracy on this dataset relative to random prior sampling. I also keep the same path checks and ensure the submission format stays identical.'
- What this solution (achieved 0.59081) has done: 'To move your score up toward 0.866 with minimal disruption, the smallest legitimate step is to keep your current “no heavy training loop” structure but switch from a constant-majority guess to a lightweight image-based classifier using the provided `train_images/` and `test_images/`. This preserves the simple single-pass pipeline (no TensorFlow, no deep model), stays within the package constraints, and should substantially improve accuracy because Cassava classes have strong color/texture cues that a basic linear model can pick up. Concretely, we extract a compact RGB color histogram feature per image (fast), train a multinomial logistic regression (implemented in NumPy with a fixed number of iterations), and predict labels for the test set in the exact sample_submission order. The output remains a valid `submission.csv` with identical schema and row alignment.'

# 9. Code solution

## === cell 0
import os
from PIL import Image

import pandas as pd
import numpy as np

np.random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/data/cassava-leaf-disease-classification"
    if os.path.exists(alt):
        DATA_DIR = alt

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "sample_submission.csv must have image_id and label columns"

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert {"image_id", "label"}.issubset(
    train_df.columns
), "train.csv must have image_id and label columns"

print("DATA_DIR:", DATA_DIR)
print("sample rows:", len(sample_sub), "train rows:", len(train_df))



## === cell 1


def image_to_feature(img_path, size=(96, 96), bins=16):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)

    feat = []
    for c in range(3):
        h, _ = np.histogram(arr[..., c], bins=bins, range=(0, 256))
        feat.append(h.astype(np.float32))
    feat = np.concatenate(feat, axis=0)
    feat /= feat.sum() + 1e-8  # normalize to reduce lighting/size effects
    return feat


def build_feature_matrix(image_ids, img_dir, size=(96, 96), bins=16):
    X = np.zeros((len(image_ids), 3 * bins), dtype=np.float32)
    missing = 0
    for i, img_id in enumerate(image_ids):
        p = os.path.join(img_dir, img_id)
        if not os.path.exists(p):
            missing += 1
            continue
        X[i] = image_to_feature(p, size=size, bins=bins)
    if missing:
        print(
            f"Warning: {missing} images missing in {img_dir}. Their features are zeros."
        )
    return X


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)


def train_softmax_regression(X, y, num_classes, lr=0.5, l2=1e-3, iters=200):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    n, d = Xb.shape
    W = np.zeros((d, num_classes), dtype=np.float32)

    y_onehot = np.zeros((n, num_classes), dtype=np.float32)
    y_onehot[np.arange(n), y] = 1.0

    for t in range(iters):
        logits = Xb @ W
        P = softmax(logits)
        grad = (Xb.T @ (P - y_onehot)) / n
        reg = l2 * W
        reg[-1, :] = 0.0
        grad += reg
        W -= lr * grad

        if (t + 1) % 50 == 0:
            pred = np.argmax(P, axis=1)
            acc = (pred == y).mean()
            print(f"iter {t+1:>3d}/{iters} train-acc={acc:.4f}")
    return W


def predict_softmax_regression(X, W):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    logits = Xb @ W
    P = softmax(logits)
    return np.argmax(P, axis=1).astype(int)


train_image_ids = train_df["image_id"].values
y = train_df["label"].astype(int).values
num_classes = int(train_df["label"].nunique())

print("Extracting train features...")
X_train = build_feature_matrix(train_image_ids, TRAIN_IMG_DIR, size=(96, 96), bins=16)

mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-6
X_train_std = (X_train - mu) / sigma

print("Training linear softmax classifier...")
W = train_softmax_regression(
    X_train_std, y, num_classes=num_classes, lr=0.5, l2=1e-3, iters=200
)

test_image_ids = sample_sub["image_id"].values
print("Extracting test features...")
X_test = build_feature_matrix(test_image_ids, TEST_IMG_DIR, size=(96, 96), bins=16)
X_test_std = (X_test - mu) / sigma

preds = predict_softmax_regression(X_test_std, W)

preds = np.clip(preds, 0, 4).astype(int)

my_submission = pd.DataFrame({"image_id": test_image_ids, "label": preds})

assert len(my_submission) == len(
    sample_sub
), "Submission row count must match sample_submission.csv"
assert my_submission["image_id"].isnull().sum() == 0, "image_id must be non-null"
assert my_submission["label"].isnull().sum() == 0, "label must be non-null"

out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(my_submission.head())
print("Rows:", len(my_submission))
print("Unique predicted labels:", sorted(my_submission["label"].unique().tolist()))
