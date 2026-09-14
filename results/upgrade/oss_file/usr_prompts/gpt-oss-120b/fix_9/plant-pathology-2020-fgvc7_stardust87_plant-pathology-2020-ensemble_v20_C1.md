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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted average of the probabilities from a list of submission files.
    If the requested indices are out of range we raise a clear error.
    """
    if not submissions_all:
        raise ValueError("No submission files available for ensembling.")
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")
    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}] == {idx} is out of bounds for submissions_all of length {len(submissions_all)}"
            )
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        sub = pd.read_csv(submissions_all[idx])
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(sub_vals * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def model_based_submission():
    """
    Light model using several cheap image‑derived features plus 8‑bin RGB histograms:
      1. File size (bytes)
      2. Numeric part of the image_id
      3. Image width & height (after a fast resize)
      4. Mean R, G, B channel values (after resize to 64×64)
      5. Std R, G, B channel values
      6. Grayscale mean intensity
      7. Grayscale variance
      8. 8‑bin histograms for each RGB channel (24 features)
      9. Flattened 64×64 RGB pixels (12 288 features) – new visual signal
    A GradientBoostingClassifier (increased capacity) is trained per target column.
    """

    def get_features(df):
        sizes = []
        ids = []
        widths = []
        heights = []
        mean_r = []
        mean_g = []
        mean_b = []
        std_r = []
        std_g = []
        std_b = []
        gray_means = []
        gray_vars = []
        hist_r = []
        hist_g = []
        hist_b = []
        flat_pixels = []  # new list for flattened pixel vectors

        for img_id in df["image_id"]:
            path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
            try:
                sizes.append(os.path.getsize(path))
            except Exception:
                sizes.append(0)

            num_str = "".join(filter(str.isdigit, str(img_id)))
            ids.append(int(num_str) if num_str else 0)

            if _PIL_AVAILABLE and os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    img_resized = img.resize((64, 64))
                    img_array = np.array(img_resized)  # (64, 64, 3)

                    widths.append(img_resized.width)
                    heights.append(img_resized.height)

                    r = img_array[:, :, 0]
                    g = img_array[:, :, 1]
                    b = img_array[:, :, 2]

                    mean_r.append(r.mean())
                    mean_g.append(g.mean())
                    mean_b.append(b.mean())

                    std_r.append(r.std())
                    std_g.append(g.std())
                    std_b.append(b.std())

                    h_r, _ = np.histogram(r, bins=8, range=(0, 255), density=True)
                    h_g, _ = np.histogram(g, bins=8, range=(0, 255), density=True)
                    h_b, _ = np.histogram(b, bins=8, range=(0, 255), density=True)
                    hist_r.append(h_r)
                    hist_g.append(h_g)
                    hist_b.append(h_b)

                    gray = img.convert("L")
                    gray_arr = np.array(gray.resize((64, 64)))
                    gray_means.append(gray_arr.mean())
                    gray_vars.append(gray_arr.var())

                    flat_pixels.append(img_array.ravel())
                except Exception:
                    widths.append(0)
                    heights.append(0)
                    mean_r.append(0.0)
                    mean_g.append(0.0)
                    mean_b.append(0.0)
                    std_r.append(0.0)
                    std_g.append(0.0)
                    std_b.append(0.0)
                    hist_r.append(np.zeros(8))
                    hist_g.append(np.zeros(8))
                    hist_b.append(np.zeros(8))
                    gray_means.append(0.0)
                    gray_vars.append(0.0)
                    flat_pixels.append(np.zeros(64 * 64 * 3))
            else:
                widths.append(0)
                heights.append(0)
                mean_r.append(0.0)
                mean_g.append(0.0)
                mean_b.append(0.0)
                std_r.append(0.0)
                std_g.append(0.0)
                std_b.append(0.0)
                hist_r.append(np.zeros(8))
                hist_g.append(np.zeros(8))
                hist_b.append(np.zeros(8))
                gray_means.append(0.0)
                gray_vars.append(0.0)
                flat_pixels.append(np.zeros(64 * 64 * 3))

        hist_r_arr = np.vstack(hist_r)
        hist_g_arr = np.vstack(hist_g)
        hist_b_arr = np.vstack(hist_b)
        flat_arr = np.vstack(flat_pixels)

        return np.column_stack(
            [
                sizes,
                ids,
                widths,
                heights,
                mean_r,
                mean_g,
                mean_b,
                std_r,
                std_g,
                std_b,
                gray_means,
                gray_vars,
                hist_r_arr,
                hist_g_arr,
                hist_b_arr,
                flat_arr,
            ]
        )

    prob_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = get_features(train_df)
    X_test = get_features(test_df)

    preds = np.empty((len(test_df), len(prob_cols)), dtype=float)

    for i, col in enumerate(prob_cols):
        y = train_df[col].values
        try:
            model = GradientBoostingClassifier(
                n_estimators=1200,  # increased capacity
                learning_rate=0.05,
                max_depth=6,  # deeper trees
                random_state=42,
            )
            model.fit(X_train, y)
            preds[:, i] = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(
                f"GradientBoosting failed for {col} ({e}), falling back to LogisticRegression."
            )
            lr = LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
            )
            lr.fit(X_train, y)
            preds[:, i] = lr.predict_proba(X_test)[:, 1]

    sub_df = test_df.copy()
    sub_df[prob_cols] = preds
    sub_df.to_csv("submission.csv", index=False)
    print("Model‑based submission written to submission.csv")




## === cell 4
def make_submission_file(submission_avg, reference_csv_path):
    """
    Write the averaged probabilities to a CSV file using the structure of a reference submission.
    """
    ref_df = pd.read_csv(reference_csv_path)
    prob_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    ref_df.loc[:, prob_cols] = submission_avg
    ref_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")


def baseline_submission():
    """
    Simple baseline: predict each class probability as the mean label frequency
    observed in the training data.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    prob_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    means = train_df[prob_cols].mean().values.reshape(1, -1)
    test_df = pd.read_csv(TEST_PATH)
    baseline_preds = np.repeat(means, repeats=len(test_df), axis=0)
    sub_df = test_df.copy()
    sub_df[prob_cols] = baseline_preds
    sub_df.to_csv("submission.csv", index=False)
    print("Baseline submission written to submission.csv")




## === cell 5
import numpy as np

if submissions_all:
    n_subs = min(3, len(submissions_all))
    idxs = list(range(n_subs))
    weights = [1.0 / n_subs] * n_subs
    submission_avg = ensemble(submissions_all, idxs, weights)
    make_submission_file(submission_avg, submissions_all[0])
else:
    try:
        model_based_submission()
    except Exception as e:
        print(f"Model‑based submission failed ({e}), falling back to baseline.")
        baseline_submission()
