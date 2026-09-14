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

0.9699976501828376

# 6. Current score

0.49317

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.49317) has done: 'I fix the runtime error by removing the hard dependency on a non-existent `/kaggle/input/submissions/submissions/` dataset and instead generate a valid prediction file directly from the provided competition data. To preserve the “ensemble submissions” core logic, the script still average/weight multiple “submissions”, but when none are available it create two simple baseline submissions (image-based brightness heuristics) and ensemble them. This guarantees an end-to-end run that writes `submission.csv` with the exact required columns and row alignment to `test.csv`. The approach is fully self-contained in the given environment and avoids changing the competition’s evaluation semantics (still outputs probabilities per class).'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
]


def _find_file(filename):
    for base in BASE_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in BASE_CANDIDATES:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if filename in files:
                    return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not locate {filename} in known Kaggle input paths.")


TRAIN_CSV = _find_file("train.csv")
TEST_CSV = _find_file("test.csv")
SAMPLE_SUB = _find_file("sample_submission.csv")


def _find_images_dir():
    candidates = []
    for base in BASE_CANDIDATES:
        candidates.extend(
            [
                os.path.join(base, "images"),
                os.path.join(base, "plant-pathology-2020-fgvc7", "images"),
            ]
        )
    for c in candidates:
        if os.path.isdir(c):
            return c
    for base in BASE_CANDIDATES:
        if os.path.exists(base):
            for root, dirs, _ in os.walk(base):
                if "images" in dirs:
                    cand = os.path.join(root, "images")
                    try:
                        if any(f.lower().endswith(".jpg") for f in os.listdir(cand)):
                            return cand
                    except Exception:
                        pass
    raise FileNotFoundError("Could not locate images directory.")


IMAGES_DIR = _find_images_dir()

print("Using:")
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV :", TEST_CSV)
print("SAMPLE   :", SAMPLE_SUB)
print("IMAGES   :", IMAGES_DIR)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")
    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values.astype(np.float64)
        submission_with_weight.append(submission * float(weights[i]))
    submission_sum = np.sum(submission_with_weight, axis=0)
    weight_sum = float(np.sum(weights))
    submission_avg = submission_sum / weight_sum if weight_sum != 0 else submission_sum
    return submission_avg


def make_submission_file(submission_avg, template_csv_path, out_path="submission.csv"):
    submission_df = pd.read_csv(template_csv_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_df[
        ["healthy", "multiple_diseases", "rust", "scab"]
    ].clip(0.0, 1.0)
    submission_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
    )




## === cell 4


test_df = pd.read_csv(TEST_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

test_ids = test_df["image_id"].astype(str).tolist()
sample_df["image_id"] = sample_df["image_id"].astype(str)
sample_df = sample_df.set_index("image_id").loc[test_ids].reset_index()

try:
    from PIL import Image

    def _load_gray(path):
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

except Exception:
    import matplotlib.image as mpimg

    def _load_gray(path):
        arr = mpimg.imread(path)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return arr


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_baseline_submission(mode=1):
    feats = []
    for image_id in test_ids:
        img_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_DIR, image_id)
        arr = _load_gray(img_path)
        m = float(arr.mean())
        s = float(arr.std())
        feats.append((m, s))
    feats = np.array(feats, dtype=np.float64)
    m = feats[:, 0]
    s = feats[:, 1]

    if mode == 1:
        healthy = _sigmoid((m - 0.50) * 8.0)
        rust = _sigmoid((s - 0.18) * 10.0) * (1.0 - 0.35 * healthy)
        scab = _sigmoid((s - 0.16) * 9.0) * (1.0 - 0.30 * healthy)
        multiple = _sigmoid((rust + scab - 0.90) * 6.0)
    else:
        dark = _sigmoid((0.48 - m) * 9.0)
        healthy = _sigmoid((m - 0.52) * 9.0) * (1.0 - 0.35 * dark)
        rust = _sigmoid((s - 0.17) * 11.0) * (0.6 + 0.4 * dark)
        scab = _sigmoid((s - 0.15) * 10.0) * (0.6 + 0.4 * dark)
        multiple = _sigmoid((rust + scab - 1.05) * 7.0)

    mat = np.vstack([healthy, multiple, rust, scab]).T
    mat = np.clip(mat, 1e-6, 1.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.2, 0.8])
    make_submission_file(submission_avg, submissions_all[0], out_path="submission.csv")
elif len(submissions_all) == 1:
    one = pd.read_csv(submissions_all[0])
    submission_avg = one[
        ["healthy", "multiple_diseases", "rust", "scab"]
    ].values.astype(np.float64)
    make_submission_file(submission_avg, submissions_all[0], out_path="submission.csv")
else:
    sub1 = build_baseline_submission(mode=1)
    sub2 = build_baseline_submission(mode=2)
    w1, w2 = 0.5, 0.5
    submission_avg = (sub1 * w1 + sub2 * w2) / (w1 + w2)

    make_submission_file(submission_avg, SAMPLE_SUB, out_path="submission.csv")

out = pd.read_csv("submission.csv")
assert out.shape[0] == len(test_df), "Submission row count must match test.csv"
assert list(out.columns) == [
    "image_id",
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], "Wrong submission columns/order"
print(out.head())
