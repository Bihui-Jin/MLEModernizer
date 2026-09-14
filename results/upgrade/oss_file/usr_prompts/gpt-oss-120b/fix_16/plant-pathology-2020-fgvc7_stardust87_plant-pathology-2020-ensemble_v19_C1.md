# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
from PIL import Image
import numpy as np
import concurrent.futures  # parallel image processing
import multiprocessing



## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Combine existing submission files with given weights.
    If submissions_all is empty this function will simply return None.
    """
    if not submissions_all:
        return None
    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    return sum(submission_with_weight)


def make_submission_file_from_ensemble(submission_avg, submissions_all):
    """Create submission.csv from an ensemble matrix."""
    submission_df = pd.read_csv(submissions_all[0])
    submission_df.iloc[:, 1:] = 0
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)




## === cell 3
SUBMISSIONS_PATH = "/kaggle/input/submissions"
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            submissions_all.append(os.path.join(dirname, filename))
    submissions_all.sort()
print("Found submissions:", submissions_all)



## === cell 4
submission_avg = None
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.05, 0.9, 0.05])
    if submission_avg is not None:
        make_submission_file_from_ensemble(submission_avg, submissions_all)



## === cell 5
if submission_avg is None:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    ALPHA = 1.2  # stretch factor to increase prediction variance

    def extract_id_num(img_id):
        nums = re.findall(r"\d+", str(img_id))
        return int(nums[0]) if nums else 0

    def digit_sum(n):
        return sum(int(d) for d in str(abs(n)))

    def compute_rgb_stats(img_id):
        """Return mean, std and bright/dark proportions for each RGB channel."""
        img_path = os.path.join(DATA_ROOT, "images", f"{img_id}.jpg")
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, 3)

                mean_vals = arr.mean(axis=(0, 1))  # (R, G, B)
                std_vals = arr.std(axis=(0, 1))  # (R, G, B)

                bright = (arr > 0.5).mean(axis=(0, 1))  # (R, G, B)
                dark = (arr <= 0.5).mean(axis=(0, 1))  # (R, G, B)

                return list(mean_vals) + list(std_vals) + list(bright) + list(dark)
        except Exception:
            return [0.0] * 12

    def add_features(df):
        df["id_num"] = df["image_id"].apply(extract_id_num)
        df["id_mod_10"] = df["id_num"] % 10
        df["id_mod_100"] = df["id_num"] % 100
        df["id_mod_5"] = df["id_num"] % 5
        df["id_mod_20"] = df["id_num"] % 20
        df["id_digit_sum"] = df["id_num"].apply(digit_sum)

        img_ids = df["image_id"].tolist()
        max_workers = min(5, multiprocessing.cpu_count())
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            rgb_stats_list = list(executor.map(compute_rgb_stats, img_ids))

        rgb_stats_arr = np.array(rgb_stats_list, dtype=np.float32)  # (n, 12)
        (
            df["mean_r"],
            df["mean_g"],
            df["mean_b"],
            df["std_r"],
            df["std_g"],
            df["std_b"],
            df["bright_r"],
            df["bright_g"],
            df["bright_b"],
            df["dark_r"],
            df["dark_g"],
            df["dark_b"],
        ) = rgb_stats_arr.T
        return df

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    feature_cols = [
        "id_num",
        "id_mod_10",
        "id_mod_100",
        "id_mod_5",
        "id_mod_20",
        "id_digit_sum",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "bright_r",
        "bright_g",
        "bright_b",
        "dark_r",
        "dark_g",
        "dark_b",
    ]

    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_train_poly = poly.fit_transform(train_df[feature_cols].values).astype(np.float32)
    X_test_poly = poly.transform(test_df[feature_cols].values).astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_poly).astype(np.float32)
    X_test = scaler.transform(X_test_poly).astype(np.float32)

    preds = pd.DataFrame()
    preds["image_id"] = test_df["image_id"]

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for col in label_cols:
        y_train = train_df[col].values

        if y_train.min() == y_train.max():
            preds[col] = y_train.mean()
            continue

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            C=1000.0,
            class_weight="balanced",
            n_jobs=5,
        )
        model.fit(X_train, y_train)
        prob_pos = model.predict_proba(X_test)[:, 1]

        calibrated = np.clip(0.5 + ALPHA * (prob_pos - 0.5), 0.0, 1.0)

        preds[col] = calibrated

    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    preds = preds[sample_sub.columns]

    preds.to_csv("submission.csv", index=False)
    print(
        "Added bright/dark pixel features and applied ALPHA calibration; "
        "submission.csv written."
    )
