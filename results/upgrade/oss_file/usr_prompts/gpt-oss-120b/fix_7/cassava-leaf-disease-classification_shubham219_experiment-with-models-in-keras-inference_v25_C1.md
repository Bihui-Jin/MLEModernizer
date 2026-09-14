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

0.6941674221819281

# 6. Current score

0.61173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The timeout is caused by using an infinite generator for `my_model.predict`, which makes the prediction loop run forever. The fix replaces the generator call with a direct NumPy batch sized to the number of test images, allowing `DummyModel.predict` to compute the required prediction array in one fast step while keeping the same output shape and logic.'
- What this solution (achieved 0.61099) has done: 'I replace the placeholder model with a simple “most‑common‑label” model that learns the majority class from the training CSV and always predicts that class. This keeps the overall pipeline unchanged while giving a much higher expected accuracy, moving the score toward the target. The rest of the code (data loading, CSV creation) remains the same.'
- What this solution (achieved 0.61099) has done: 'I replace the constant‑majority dummy model with a very lightweight “mean‑color” classifier: it reads each training image, computes its average RGB values, learns a linear mapping to one‑hot labels via a closed‑form least‑squares solution, and then applies the same mean‑color feature to the test images to produce class probabilities. This keeps the original `predict` interface while giving a modest accuracy boost toward the target score.'
- What this solution (achieved 0.61173) has done: 'The change introduces parallel image feature extraction using a thread pool, which dramatically reduces the I/O‑bound time spent reading and processing thousands of images while keeping exactly the same feature vector and linear‑model logic. The core model, loss, and prediction steps remain unchanged; only the way the (mean, std) features are computed is accelerated, preserving deterministic ordering and numerical results.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import glob
from PIL import Image
import os
import concurrent.futures

SEED = 42
DEBUG = False

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)




## === cell 1
class SimpleMeanModel:
    """
    Linear model trained on per‑image mean RGB values **and** per‑channel standard deviations.
    Fits a least‑squares solution to map (R, G, B, stdR, stdG, stdB, bias) → one‑hot label.
    Provides a ``predict`` method returning class probabilities, matching the original interface.
    """

    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.weights = None  # shape (7, n_classes)

    @staticmethod
    def _rgb_mean_std(img_path):
        """Return concatenated mean and std of RGB channels (6‑dim feature)."""
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))  # (R, G, B)
        std = arr.std(axis=(0, 1))  # (stdR, stdG, stdB)
        return np.concatenate([mean, std])  # (6,)

    @staticmethod
    def _compute_features_parallel(paths, max_workers=None):
        """
        Compute the 6‑dim feature for each image path in parallel,
        preserving the order of ``paths``.
        """
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            features = list(executor.map(SimpleMeanModel._rgb_mean_std, paths))
        return np.asarray(features, dtype=np.float32)

    def fit(self, image_paths, labels):
        feats = self._compute_features_parallel(image_paths)  # (N, 6)
        bias = np.ones((feats.shape[0], 1), dtype=np.float32)
        X = np.hstack([feats, bias])  # (N, 7)

        Y = np.zeros((len(labels), self.n_classes), dtype=np.float32)
        Y[np.arange(len(labels)), labels] = 1.0

        XtX = X.T @ X
        ridge = 1e-4 * np.eye(XtX.shape[0])
        self.weights = np.linalg.solve(XtX + ridge, X.T @ Y)  # (7, n_classes)

    def predict(self, data, verbose=0):
        """
        ``data`` can be a NumPy array of shape (N, 7) where the last column is bias,
        or any iterable that yields such rows. Returns class probabilities.
        """
        if isinstance(data, np.ndarray):
            X = data
        else:
            X = np.array(list(data), dtype=np.float32)

        logits = X @ self.weights  # (N, n_classes)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs


train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
train_image_paths = [
    f"{train_images_dir}/{img_id}" for img_id in train_df["image_id"].values
]
train_labels = train_df["label"].astype(int).values

my_model = SimpleMeanModel()
my_model.fit(train_image_paths, train_labels)




## === cell 2
test_images = glob.glob(
    "../input/cassava-leaf-disease-classification/test_images/*.jpg"
)
df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_gen(batch_size=64):
    while True:
        for i in range(0, len(df_test), batch_size):
            batch_paths = df_test["path"].iloc[i : i + batch_size].values
            dummy_batch = np.zeros((len(batch_paths), 300, 300, 3), dtype=np.float32)
            yield dummy_batch




## === cell 3
test_feats = SimpleMeanModel._compute_features_parallel(df_test["path"].values)

bias = np.ones((test_feats.shape[0], 1), dtype=np.float32)
test_input = np.hstack([test_feats, bias])  # shape (N, 7)

pred_test = my_model.predict(test_input, verbose=False)
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)




## === cell 4
final_csv.head()
