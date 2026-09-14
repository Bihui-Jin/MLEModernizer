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

0.9699

# 6. Current score

0.52875

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: `submissions_all` is empty because `SUBMISSIONS_PATH` points to `/kaggle/input/submissions/`, which does not exist in this environment (the available submissions/sample files are under `/kaggle/data/...`). As a result, `ensemble(submissions_all,[0,1,2], ...)` tries to index into an empty list and crashes with `IndexError`. The minimal fix is to update cell 5 to gracefully handle missing/insufficient submission files by falling back to the provided `sample_submission.csv` (which matches the expected schema) and using only the available indices. This preserves the ensemble and output-file creation logic without changing model/metric semantics.

Patch summary: In cell 5 only, detect if `submissions_all` has fewer than 3 files; if so, set `submissions_all` to a list containing the known `sample_submission.csv` path and adjust `sub_idx/weights` to valid lengths before calling `ensemble`.

Updated cells:'
- What this solution (achieved 0.49301) has done: 'Your current 0.5 score comes from submitting essentially constant/placeholder probabilities (the fallback to `sample_submission.csv` has all 0.25s), so ROC AUC stays near random. To move toward the 0.9699 target with minimal logic change, I keep your ensemble structure but add a deterministic “fallback model” that creates non-constant predictions from the training labels’ class priors, so the submission is valid and non-degenerate. I also make the code robust to missing `/kaggle/input/submissions/` by searching known data locations and ensuring weights are renormalized to sum to 1 for proper averaging (same ensemble semantics, just correct scaling). This should increase score from 0.5 toward the target without changing the overall approach (still producing a weighted average submission).'
- What this solution (achieved 0.52875) has done: 'Your current score is low because the “prior fallback” produces almost-constant predictions (just tiny noise around class priors), which yields near-random ROC AUC. To move toward the 0.9699 target while keeping the ensemble core logic intact, I keep your ensemble/make-submission flow but replace the fallback predictions with a lightweight, label-derived heuristic: infer each test image’s label by matching its `image_id` to the closest train `image_id` in pixel-space (1-NN on downsampled grayscale), then output a smoothed one-hot probability vector. This stays within the same evaluation semantics (still probabilities per class), avoids changing any model/training loop (there is none), and should substantially improve AUC versus priors-only. I also make path resolution robust to your provided `/kaggle/data/...` locations and ensure the output submission rows align exactly with `test.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"



## === cell 2
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
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
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all):
    submission_df = pd.read_csv(submissions_all[0])
    submission_df.iloc[:, 1:] = 0
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
def build_prior_fallback_submission(
    train_csv_path="/kaggle/data/train.csv",
    test_csv_path="/kaggle/data/test.csv",
    sample_sub_path="/kaggle/data/sample_submission.csv",
    out_path="prior_fallback_submission.csv",
):
    import numpy as np
    from PIL import Image

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    sample_df = pd.read_csv(sample_sub_path)

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    def _find_images_dir(base_csv_path):
        base_dir = os.path.dirname(os.path.abspath(base_csv_path))
        candidates = [
            os.path.join(base_dir, "images"),
            os.path.join(base_dir, "plant-pathology-2020-fgvc7", "images"),
            "/kaggle/data/images",
            "/kaggle/data/plant-pathology-2020-fgvc7/images",
            "/kaggle/input/plant-pathology-2020-fgvc7/images",
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return None

    images_dir = _find_images_dir(train_csv_path)
    if images_dir is None:
        raise FileNotFoundError(
            "Could not locate images directory for fallback heuristic."
        )

    def _img_feature(img_path, size=(32, 32)):
        im = Image.open(img_path).convert("L").resize(size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.reshape(-1)

    train_ids = train_df["image_id"].astype(str).tolist()
    X_train = np.zeros((len(train_ids), 32 * 32), dtype=np.float32)
    for i, img_id in enumerate(train_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, img_id)
        X_train[i] = _img_feature(img_path)

    y_train = train_df[target_cols].values.astype(np.float32)

    test_ids = test_df["image_id"].astype(str).tolist()
    preds = np.zeros((len(test_ids), len(target_cols)), dtype=np.float32)

    eps = 0.02  # small smoothing toward uniform
    uniform = np.full((len(target_cols),), 1.0 / len(target_cols), dtype=np.float32)

    for j, img_id in enumerate(test_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, img_id)
        x = _img_feature(img_path)

        x2 = float(np.dot(x, x))
        train_norms = np.einsum("ij,ij->i", X_train, X_train)
        dists = train_norms + x2 - 2.0 * (X_train @ x)
        nn = int(np.argmin(dists))

        onehot = y_train[nn]
        s = float(onehot.sum())
        if s <= 0:
            onehot = uniform
        else:
            onehot = onehot / s

        p = (1.0 - eps) * onehot + eps * uniform
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        preds[j] = p

    sub = sample_df[["image_id"]].copy()
    sub = sub.merge(test_df[["image_id"]], on="image_id", how="right")
    sub[target_cols] = preds

    sub.to_csv(out_path, index=False)
    return out_path




## === cell 6
if len(submissions_all) == 0:
    candidates = [
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
    ]
    sample_path = None
    for p in candidates:
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        raise FileNotFoundError(
            "Could not find sample_submission.csv in expected Kaggle paths."
        )

    train_candidates = [
        "/kaggle/data/train.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/train.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/train.csv",
    ]
    test_candidates = [
        "/kaggle/data/test.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/test.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/test.csv",
    ]
    train_path = next((p for p in train_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not find train.csv/test.csv in expected Kaggle paths."
        )

    prior_sub_path = build_prior_fallback_submission(
        train_csv_path=train_path,
        test_csv_path=test_path,
        sample_sub_path=sample_path,
        out_path="prior_fallback_submission.csv",
    )
    submissions_all = [prior_sub_path]

sub_idx = [0, 1, 2]
weights = [0.15, 0.8, 0.05]

valid = [(i, w) for i, w in zip(sub_idx, weights) if i < len(submissions_all)]
sub_idx = [i for i, _ in valid]
weights = [w for _, w in valid]

if len(weights) == 0:
    raise ValueError("No valid submissions available to ensemble.")
wsum = sum(weights)
if wsum == 0:
    weights = [1.0 / len(weights)] * len(weights)
else:
    weights = [w / wsum for w in weights]

submission_avg = ensemble(submissions_all, sub_idx, weights)
make_submission_file(submission_avg, submissions_all)
print("Wrote submission.csv")
