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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9628698141306518

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Implemented a shape‑check in the ensembling logic to skip any CSV files (e.g., the full training set) whose row count doesn’t match the reference submission. This prevents broadcasting errors when averaging predictions and ensures the resulting array aligns with the test set size, allowing a valid `submission.csv` to be written.'
- What this solution (achieved 0.5) has done: 'I add a lightweight image‑feature model that is used only when the ensemble fallback is triggered (i.e., no valid CSV submissions). The model extracts mean and standard‑deviation of RGB channels from each image, trains a simple One‑Vs‑Rest logistic regression on the training set, and predicts probabilities for the test set. This provides a more informative baseline than constant class means, moving the ROC‑AUC score closer to the target while keeping the original ensemble logic intact.'
- What this solution (achieved 0.5) has done: 'I enhance the fallback image‑based model so it can reach the target AUC.  
First, the image feature extractor now returns a richer 8‑dimensional vector (RGB mean & std, grayscale mean & std, plus image width and height).  
Second, the classifier is switched from a simple LogisticRegression to a stronger RandomForest (wrapped in One‑Vs‑Rest), which works better with these cues.  
These changes stay within the original pipeline (same ensembling logic, same CSV handling) but provide a more discriminative model, moving the score from ~0.5 toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image


def _collect_submissions():
    all_paths = [
        p
        for p in glob.glob("**/*.csv", recursive=True)
        if os.path.basename(p) not in ("submission.csv",)
    ]
    sample_paths = [
        p for p in all_paths if "sample_submission" in os.path.basename(p).lower()
    ]
    if sample_paths:
        template = sample_paths[0]
        other_paths = [p for p in all_paths if p != template]
        return [template] + other_paths
    return all_paths


submissions_all = _collect_submissions()
if not submissions_all:
    possible = glob.glob("**/sample_submission.csv", recursive=True)
    if possible:
        submissions_all = [possible[0]]
    else:
        raise FileNotFoundError("No sample submission CSV found to use as a template.")




## === cell 1
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average selected submissions with given weights.
    Only submissions whose row count matches the reference (first) submission
    are used. Missing indices or submissions lacking required columns are ignored;
    if weights is None, equal weighting is used for the provided indices.
    """
    required = ["healthy", "multiple_diseases", "rust", "scab"]
    template_path = submissions_all[0]
    template_len = pd.read_csv(template_path).shape[0]

    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError("weights and sub_idx must have the same length.")

    weighted_sum = None
    total_w = 0.0

    for idx, w in zip(sub_idx, weights):
        if idx >= len(submissions_all):
            print(f"Skipping missing submission index {idx}")
            continue
        sub_path = submissions_all[idx]
        print(f"Loading submission {sub_path} with raw weight {w:.3f}")
        sub = pd.read_csv(sub_path)

        if not set(required).issubset(sub.columns):
            print(f"Skipping {sub_path} – missing required columns.")
            continue

        if sub.shape[0] != template_len:
            print(
                f"Skipping {sub_path} – row count {sub.shape[0]} does not match template {template_len}."
            )
            continue

        sub_vals = sub.loc[:, required].values
        if weighted_sum is None:
            weighted_sum = np.zeros_like(sub_vals, dtype=float)

        weighted_sum += sub_vals * w
        total_w += w

    if total_w == 0.0 or weighted_sum is None:
        raise RuntimeError("No valid submissions were loaded for ensembling.")
    submission_avg = weighted_sum / total_w
    return submission_avg




## === cell 2
def _extract_image_features(image_path):
    """
    Return a feature vector:
    - RGB channel means (3) and stds (3)
    - HSV channel means (3) and stds (3)
    - Grayscale mean & std (2)
    - Image width & height (2)
    Total length = 16.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb).astype(np.float32) / 255.0  # (H, W, 3)

        img_hsv = img.convert("HSV")
        hsv_arr = np.array(img_hsv).astype(np.float32) / 255.0

    rgb_means = arr.mean(axis=(0, 1))
    rgb_stds = arr.std(axis=(0, 1))

    hsv_means = hsv_arr.mean(axis=(0, 1))
    hsv_stds = hsv_arr.std(axis=(0, 1))

    gray = np.dot(arr, [0.2989, 0.5870, 0.1140])
    gray_mean = gray.mean()
    gray_std = gray.std()

    height, width = arr.shape[:2]

    return np.concatenate(
        [
            rgb_means,
            rgb_stds,
            hsv_means,
            hsv_stds,
            [gray_mean, gray_std, width, height],
        ]
    )


def _train_image_model(train_df, train_img_dir):
    """
    Train a One‑Vs‑Rest RandomForest on the enriched image features.
    """
    feature_list = []
    for img_id in train_df["image_id"]:
        img_path = os.path.join(train_img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            feature_list.append(np.zeros(16))
        else:
            feature_list.append(_extract_image_features(img_path))
    X = np.stack(feature_list, axis=0)

    y = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=5,
        class_weight="balanced",
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", OneVsRestClassifier(rf)),
        ]
    )
    model.fit(X, y)
    return model


def _predict_image_model(model, test_df, test_img_dir):
    """
    Generate probability predictions for the test set using the trained model.
    """
    test_features = []
    for img_id in test_df["image_id"]:
        img_path = os.path.join(test_img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            test_features.append(np.zeros(16))
        else:
            test_features.append(_extract_image_features(img_path))
    X_test = np.stack(test_features, axis=0)
    probs = model.predict_proba(X_test)  # list of arrays per class
    prob_array = np.column_stack([p[:, 1] for p in probs])  # prob of positive class
    return prob_array


def make_submission_file(submission_avg, submissions_all):
    """
    Write the averaged predictions to submission.csv.
    If the shape does not match the reference file (e.g., only one submission
    was available) **or** the predictions have near‑zero variance (typical of
    placeholder CSVs), we fall back to the improved image‑based model.
    """
    template_df = pd.read_csv(submissions_all[0])
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if not all(col in template_df.columns for col in required_cols):
        raise ValueError("Template submission missing required columns.")

    if submission_avg.shape != (len(template_df), 4) or np.std(submission_avg) < 1e-6:
        print(
            "Shape mismatch or low‑variance predictions – falling back to image‑based model."
        )
        train_path = os.path.abspath(
            os.path.join(os.path.dirname(submissions_all[0]), "..", "train.csv")
        )
        train_df = pd.read_csv(train_path)
        train_img_dir = os.path.abspath(
            os.path.join(os.path.dirname(train_path), "images")
        )
        test_path = os.path.abspath(
            os.path.join(os.path.dirname(submissions_all[0]), "..", "test.csv")
        )
        test_df = pd.read_csv(test_path)
        test_img_dir = os.path.abspath(
            os.path.join(os.path.dirname(test_path), "images")
        )
        try:
            model = _train_image_model(train_df, train_img_dir)
            submission_avg = _predict_image_model(model, test_df, test_img_dir)
        except Exception as e:
            print(f"Image model failed ({e}); falling back to mean‑label baseline.")
            means = (
                train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
            )
            submission_avg = np.tile(means, (len(template_df), 1))

    template_df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = (
        submission_avg
    )
    template_df.to_csv("submission.csv", index=False)
    print("submission.csv written with shape:", template_df.shape)




## === cell 3
indices = list(range(min(2, len(submissions_all))))  # use up to first two submissions
weights = [0.8, 0.2] if len(indices) == 2 else [1.0]
submission_avg = ensemble(submissions_all, indices, weights)
make_submission_file(submission_avg, submissions_all)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3781073912.py in <cell line: 0>()
      2 weights = [0.8, 0.2] if len(indices) == 2 else [1.0]
      3 submission_avg = ensemble(submissions_all, indices, weights)
----> 4 make_submission_file(submission_avg, submissions_all)

/tmp/ipykernel_11/1843745278.py in make_submission_file(submission_avg, submissions_all)
    111             os.path.join(os.path.dirname(submissions_all[0]), "..", "train.csv")
    112         )
--> 113         train_df = pd.read_csv(train_path)
    114         train_img_dir = os.path.abspath(
    115             os.path.join(os.path.dirname(train_path), "images")

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train.csv'
