# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7985796313085525

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I bypass TensorFlow entirely to avoid import and fit errors by forcing `tf_available` to False and simplifying the prediction step to use the most frequent label from the training set. This ensures `pred_labels` is always defined, produces a correctly‑formatted CSV submission, and eliminates the runtime failures that prevented any output file from being created. No core modeling logic is retained, but the change is minimal and guarantees a valid submission file.'
- What this solution (achieved 0.61846) has done: 'We replace the naive “most‑frequent label” prediction with a tiny image‑based model that extracts simple RGB statistics from each leaf picture and trains a RandomForest classifier. This keeps the overall pipeline (CSV handling, train/validation split, submission writing) intact while providing a modest‑yet‑real improvement in accuracy, moving the score closer to the target. The changes are limited to feature extraction, model training, and using the model’s predictions instead of the constant mode label.'
- What this solution (achieved 0.6151) has done: 'Implemented several runtime‑focused tweaks while keeping the exact model, feature design, and training procedure unchanged.

Key changes  
* Avoid unnecessary division by 255 during image loading – read raw uint8 pixels and cast to float only once, cutting CPU work and memory pressure.  
* Use a slightly smaller process pool (`cpu_count()-1`) so the main process isn’t starved.  
* Increase the pool `chunksize` to 500 to reduce inter‑process dispatch overhead.  
* Add lightweight timing prints (no effect on results) to monitor the major steps.  

These adjustments preserve the original feature vector format, RandomForest settings, and output semantics, but markedly speed up feature extraction and overall runtime, keeping the script well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os, json, random, time
import numpy as np, pandas as pd

tf_available = False

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)




## === cell 1
train_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
test_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

numeric_mode_label = int(train_df["label"].mode()[0])

train_df["label"] = train_df["label"].astype(str)
train_df["filepath"] = train_dir + train_df["image_id"].astype(str)
test_df["filepath"] = test_dir + test_df["image_id"].astype(str)

train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import multiprocessing as mp


def extract_features(img_path, size=(128, 128)):
    """Resize image, flatten RGB pixels, append channel stats, and colour histograms."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.NEAREST)  # fast resize, same as original intent
        arr_uint8 = np.asarray(img, dtype=np.uint8)  # (H, W, 3) uint8

        flat = arr_uint8.flatten().astype(np.float32)

        channel_means = arr_uint8.mean(axis=(0, 1)).astype(np.float32)
        channel_stds = arr_uint8.std(axis=(0, 1)).astype(np.float32)

        hist_bins = 16
        hist_range = (0, 256)
        hist_r, _ = np.histogram(arr_uint8[:, :, 0], bins=hist_bins, range=hist_range)
        hist_g, _ = np.histogram(arr_uint8[:, :, 1], bins=hist_bins, range=hist_range)
        hist_b, _ = np.histogram(arr_uint8[:, :, 2], bins=hist_bins, range=hist_range)
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)

        return np.concatenate([flat, channel_means, channel_stds, hist])
    except Exception:
        dim = size[0] * size[1] * 3 + 2 + 48
        return np.zeros(dim, dtype=np.float32)


def _extract_indexed(args):
    """Helper for parallel processing that keeps original index."""
    idx, path = args
    return idx, extract_features(path)


def parallel_features(paths):
    """Extract features in parallel using a shared pool, pre‑allocating the result array."""
    n = len(paths)
    dim = 128 * 128 * 3 + 2 + 48
    result = np.empty((n, dim), dtype=np.float32)

    indexed_paths = list(enumerate(paths))

    proc_cnt = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=proc_cnt) as pool:
        for idx, feat in pool.imap_unordered(
            _extract_indexed, indexed_paths, chunksize=500
        ):
            result[idx] = feat
    return result


def main():
    start_total = time.time()

    train_paths = train_df["filepath"].tolist()
    train_labels = train_df["label"].astype(int).values

    t0 = time.time()
    X_train = parallel_features(train_paths)
    print(f"Feature extraction (train) took {time.time() - t0:.2f}s")
    y_train = np.array(train_labels)

    rf = RandomForestClassifier(
        n_estimators=800,  # more trees for better stability
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        class_weight="balanced",  # mitigate class imbalance
    )
    t1 = time.time()
    rf.fit(X_train, y_train)
    print(f"RandomForest training took {time.time() - t1:.2f}s")

    test_paths = test_df["filepath"].tolist()
    t2 = time.time()
    X_test = parallel_features(test_paths)
    print(f"Feature extraction (test) took {time.time() - t2:.2f}s")

    pred_labels = rf.predict(X_test).astype(int)

    submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
    submission_path = "/kaggle/working/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")

    print("Full run time:", time.time() - start_total)
    print(pd.read_csv(submission_path).head())


if __name__ == "__main__":
    main()

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2577159126.py in <cell line: 0>()
    116 
    117 if __name__ == "__main__":
--> 118     main()

/tmp/ipykernel_11/2577159126.py in main()
     83 
     84     t0 = time.time()
---> 85     X_train = parallel_features(train_paths)
     86     print(f"Feature extraction (train) took {time.time() - t0:.2f}s")
     87     y_train = np.array(train_labels)

/tmp/ipykernel_11/2577159126.py in parallel_features(paths)
     72             _extract_indexed, indexed_paths, chunksize=500
     73         ):
---> 74             result[idx] = feat
     75     return result
     76 

ValueError: could not broadcast input array from shape (49206,) into shape (49202,)
