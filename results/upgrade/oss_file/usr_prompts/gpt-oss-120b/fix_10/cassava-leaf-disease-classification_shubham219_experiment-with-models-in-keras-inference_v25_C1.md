# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
import glob
from PIL import Image
import os
import concurrent.futures

SEED = 42
DEBUG = False
np.random.seed(SEED)

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)




## === cell 1
def _rgb_stats_path(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))
    return np.concatenate([mean, std, mins, maxs])  # (12,)


class SimpleMeanModel:
    """
    Linear model trained on per‑image statistics:
    mean, std, min, max of each RGB channel (12‑dim feature).
    Features are standardised (zero‑mean, unit‑variance) before fitting a
    ridge‑regularised least‑squares solution. A tiny validation split
    selects the best ridge coefficient from a few candidates, which often
    improves accuracy without changing the model’s overall design.
    """

    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.weights = None  # shape (13, n_classes)
        self.mean_feat = None  # mean of the raw 12‑dim features
        self.std_feat = None  # std of the raw 12‑dim features
        self.ridge = 1e-4  # default ridge (may be overridden)

    @staticmethod
    def _rgb_stats(img_path):
        """Return concatenated mean, std, min, max of RGB channels (12‑dim)."""
        return _rgb_stats_path(img_path)

    @staticmethod
    def _compute_features_parallel(paths, max_workers=None):
        """Parallel computation of the 12‑dim feature for each image using processes."""
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            features = list(executor.map(_rgb_stats_path, paths))
        return np.asarray(features, dtype=np.float32)

    def _standardise(self, feats):
        """Standardise raw features using stored mean/std."""
        return (feats - self.mean_feat) / self.std_feat

    def fit(self, image_paths, labels):
        """Fit a ridge‑linear model, selecting the ridge parameter via a quick hold‑out."""
        feats = self._compute_features_parallel(image_paths)  # (N, 12)

        N = len(labels)
        indices = np.arange(N)
        np.random.shuffle(indices)
        split = int(0.8 * N)
        train_idx, val_idx = indices[:split], indices[split:]

        train_feats = feats[train_idx]
        val_feats = feats[val_idx]

        self.mean_feat = train_feats.mean(axis=0, keepdims=True)  # (1, 12)
        self.std_feat = train_feats.std(axis=0, keepdims=True) + 1e-8

        train_std = self._standardise(train_feats)  # (N_train, 12)
        bias_train = np.ones((train_std.shape[0], 1), dtype=np.float32)
        X_train = np.hstack([train_std, bias_train])  # (N_train, 13)

        Y_train = np.zeros((len(train_idx), self.n_classes), dtype=np.float32)
        Y_train[np.arange(len(train_idx)), labels[train_idx]] = 1.0

        ridge_candidates = [1e-4, 1e-3, 1e-2]
        best_ridge = ridge_candidates[0]
        best_acc = -1.0

        for r in ridge_candidates:
            w = np.linalg.solve(
                X_train.T @ X_train + r * np.eye(X_train.shape[1]), X_train.T @ Y_train
            )

            val_std = (val_feats - self.mean_feat) / self.std_feat
            bias_val = np.ones((val_std.shape[0], 1), dtype=np.float32)
            X_val = np.hstack([val_std, bias_val])
            logits_val = X_val @ w
            preds_val = np.argmax(logits_val, axis=1)
            acc = (preds_val == labels[val_idx]).mean()
            if acc > best_acc:
                best_acc = acc
                best_ridge = r

        self.ridge = best_ridge
        feats_std_all = self._standardise(feats)  # (N, 12)
        bias_all = np.ones((feats_std_all.shape[0], 1), dtype=np.float32)
        X_all = np.hstack([feats_std_all, bias_all])  # (N, 13)

        Y_all = np.zeros((N, self.n_classes), dtype=np.float32)
        Y_all[np.arange(N), labels] = 1.0

        self.weights = np.linalg.solve(
            X_all.T @ X_all + self.ridge * np.eye(X_all.shape[1]), X_all.T @ Y_all
        )  # (13, n_classes)

    def predict(self, data, verbose=0):
        """
        ``data``: NumPy array of shape (N, 13) where the last column is bias,
                 or any iterable yielding such rows.
        Returns class‑probability matrix (N, n_classes).
        """
        if isinstance(data, np.ndarray):
            X = data
        else:
            X = np.array(list(data), dtype=np.float32)

        feats_raw = X[:, :-1]  # (N, 12)
        bias = X[:, -1:]  # (N, 1)
        feats_std = (feats_raw - self.mean_feat) / self.std_feat
        X_std = np.hstack([feats_std, bias])  # (N, 13)

        logits = X_std @ self.weights
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
test_input = np.hstack([test_feats, bias])  # shape (N, 13)

pred_test = my_model.predict(test_input, verbose=False)
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)




## === cell 4
final_csv.head()
