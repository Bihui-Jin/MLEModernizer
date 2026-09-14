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

0.6151

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The timeout is caused by using an infinite generator for `my_model.predict`, which makes the prediction loop run forever. The fix replaces the generator call with a direct NumPy batch sized to the number of test images, allowing `DummyModel.predict` to compute the required prediction array in one fast step while keeping the same output shape and logic.'
- What this solution (achieved 0.61099) has done: 'I replace the placeholder model with a simple “most‑common‑label” model that learns the majority class from the training CSV and always predicts that class. This keeps the overall pipeline unchanged while giving a much higher expected accuracy, moving the score toward the target. The rest of the code (data loading, CSV creation) remains the same.'
- What this solution (achieved 0.61099) has done: 'I replace the constant‑majority dummy model with a very lightweight “mean‑color” classifier: it reads each training image, computes its average RGB values, learns a linear mapping to one‑hot labels via a closed‑form least‑squares solution, and then applies the same mean‑color feature to the test images to produce class probabilities. This keeps the original `predict` interface while giving a modest accuracy boost toward the target score.'
- What this solution (achieved 0.61173) has done: 'The change introduces parallel image feature extraction using a thread pool, which dramatically reduces the I/O‑bound time spent reading and processing thousands of images while keeping exactly the same feature vector and linear‑model logic. The core model, loss, and prediction steps remain unchanged; only the way the (mean, std) features are computed is accelerated, preserving deterministic ordering and numerical results.'
- What this solution (achieved 0.61173) has done: 'I add feature standardisation inside `SimpleMeanModel` so the linear regression is trained on zero‑mean, unit‑variance RGB statistics. This small change often raises validation accuracy without altering the overall pipeline or model architecture, moving the score toward the target. The rest of the code remains unchanged, and the script still writes a proper `submission.csv`.'
- What this solution (achieved 0.61173) has done: 'The changes replace the heavy NumPy‑based image statistics computation with Pillow’s native `ImageStat`, which directly provides mean, std, min, and max per channel without creating a full image array. This dramatically reduces per‑image processing time while yielding identical normalized statistics, so the model’s behavior and accuracy remain unchanged. The rest of the pipeline, including parallel execution and the linear model logic, stays the same.'
- What this solution (achieved 0.6151) has done: 'I expand the feature set by adding HSV channel mean and std values to the existing RGB statistics, and make the model adapt to the new feature dimension dynamically. This small enrichment keeps the linear‑ridge framework unchanged while giving the classifier more discriminative information, which should raise validation accuracy and move the score nearer the target.'
- What this solution (achieved 0.6151) has done: 'I enrich the colour statistics by adding HSV channel minima and maxima (expanding the feature vector from 18 to 24 dimensions) and broaden the ridge‑regularisation search to include a few extra candidates (including zero regularisation). These lightweight changes keep the original linear‑ridge workflow intact while giving the model more discriminative information, which should raise validation accuracy and move the Kaggle score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import glob
from PIL import Image, ImageStat
import os
import concurrent.futures

SEED = 42
DEBUG = False
np.random.seed(SEED)

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)


def _rgb_hsv_stats_path(img_path):
    """
    Compute per‑channel RGB and HSV statistics (mean & std, plus min & max for each).
    Returns a 24‑dim feature vector:
    [RGB mean(3), RGB std(3), RGB min(3), RGB max(3),
     HSV mean(3), HSV std(3), HSV min(3), HSV max(3)].
    """
    with Image.open(img_path).convert("RGB") as img:
        rgb_stat = ImageStat.Stat(img)
        rgb_mean = np.array(rgb_stat.mean, dtype=np.float32) / 255.0
        rgb_std = np.array(rgb_stat.stddev, dtype=np.float32) / 255.0
        rgb_mins = np.array([e[0] for e in rgb_stat.extrema], dtype=np.float32) / 255.0
        rgb_maxs = np.array([e[1] for e in rgb_stat.extrema], dtype=np.float32) / 255.0

        hsv_img = img.convert("HSV")
        hsv_stat = ImageStat.Stat(hsv_img)
        hsv_mean = np.array(hsv_stat.mean, dtype=np.float32) / 255.0
        hsv_std = np.array(hsv_stat.stddev, dtype=np.float32) / 255.0
        hsv_mins = np.array([e[0] for e in hsv_stat.extrema], dtype=np.float32) / 255.0
        hsv_maxs = np.array([e[1] for e in hsv_stat.extrema], dtype=np.float32) / 255.0

        return np.concatenate(
            [
                rgb_mean,
                rgb_std,
                rgb_mins,
                rgb_maxs,
                hsv_mean,
                hsv_std,
                hsv_mins,
                hsv_maxs,
            ]
        )  # (24,)


class SimpleMeanModel:
    """
    Linear ridge‑regression model trained on per‑image colour statistics.
    Feature dimension is inferred at fit‑time, allowing easy extension
    (e.g., adding HSV stats) without changing the core algorithm.
    """

    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.weights = None  # shape (n_features+1, n_classes)
        self.mean_feat = None  # mean of raw features, shape (1, n_features)
        self.std_feat = None  # std of raw features, shape (1, n_features)
        self.ridge = 1e-4  # will be selected via validation
        self.n_features = None  # set during fit

    @staticmethod
    def _rgb_hsv_stats(img_path):
        """Wrapper that returns the extended 24‑dim feature."""
        return _rgb_hsv_stats_path(img_path)

    @staticmethod
    def _compute_features_parallel(paths, max_workers=None):
        """Parallel extraction of colour statistics for a list of image paths."""
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        chunksize = max(1, len(paths) // (max_workers * 4))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            features = list(
                executor.map(_rgb_hsv_stats_path, paths, chunksize=chunksize)
            )
        return np.asarray(features, dtype=np.float32)

    def _standardise(self, feats):
        """Standardise using stored mean / std (broadcast over rows)."""
        return (feats - self.mean_feat) / self.std_feat

    def fit(self, image_paths, labels):
        """Fit ridge‑linear model; ridge strength chosen on a hold‑out split."""
        feats = self._compute_features_parallel(image_paths)  # (N, n_features)
        self.n_features = feats.shape[1]

        N = len(labels)
        indices = np.arange(N)
        np.random.shuffle(indices)
        split = int(0.8 * N)
        train_idx, val_idx = indices[:split], indices[split:]

        train_feats = feats[train_idx]
        val_feats = feats[val_idx]

        self.mean_feat = train_feats.mean(axis=0, keepdims=True)  # (1, n_features)
        self.std_feat = train_feats.std(axis=0, keepdims=True) + 1e-8

        train_std = self._standardise(train_feats)  # (N_train, n_features)
        bias_train = np.ones((train_std.shape[0], 1), dtype=np.float32)
        X_train = np.hstack([train_std, bias_train])  # (N_train, n_features+1)

        Y_train = np.zeros((len(train_idx), self.n_classes), dtype=np.float32)
        Y_train[np.arange(len(train_idx)), labels[train_idx]] = 1.0

        ridge_candidates = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        best_ridge = ridge_candidates[0]
        best_acc = -1.0

        for r in ridge_candidates:
            w = np.linalg.solve(
                X_train.T @ X_train + r * np.eye(X_train.shape[1]),
                X_train.T @ Y_train,
            )
            val_std = self._standardise(val_feats)
            bias_val = np.ones((val_std.shape[0], 1), dtype=np.float32)
            X_val = np.hstack([val_std, bias_val])
            logits_val = X_val @ w
            preds_val = np.argmax(logits_val, axis=1)
            acc = (preds_val == labels[val_idx]).mean()
            if acc > best_acc:
                best_acc = acc
                best_ridge = r

        self.ridge = best_ridge

        feats_std_all = self._standardise(feats)
        bias_all = np.ones((feats_std_all.shape[0], 1), dtype=np.float32)
        X_all = np.hstack([feats_std_all, bias_all])  # (N, n_features+1)

        Y_all = np.zeros((N, self.n_classes), dtype=np.float32)
        Y_all[np.arange(N), labels] = 1.0

        self.weights = np.linalg.solve(
            X_all.T @ X_all + self.ridge * np.eye(X_all.shape[1]),
            X_all.T @ Y_all,
        )  # (n_features+1, n_classes)

    def predict(self, data, verbose=0):
        """
        ``data``: NumPy array of shape (N, n_features+1) where the last column is bias,
                  or any iterable yielding such rows.
        Returns class‑probability matrix (N, n_classes).
        """
        if isinstance(data, np.ndarray):
            X = data
        else:
            X = np.array(list(data), dtype=np.float32)

        feats_raw = X[:, :-1]  # (N, n_features)
        bias = X[:, -1:]  # (N, 1)
        feats_std = (feats_raw - self.mean_feat) / self.std_feat
        X_std = np.hstack([feats_std, bias])  # (N, n_features+1)

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




## === cell 1
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




## === cell 2
test_feats = SimpleMeanModel._compute_features_parallel(df_test["path"].values)

bias = np.ones((test_feats.shape[0], 1), dtype=np.float32)
test_input = np.hstack([test_feats, bias])  # shape (N, n_features+1)

pred_test = my_model.predict(test_input, verbose=False)
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)




## === cell 3
final_csv.head()
