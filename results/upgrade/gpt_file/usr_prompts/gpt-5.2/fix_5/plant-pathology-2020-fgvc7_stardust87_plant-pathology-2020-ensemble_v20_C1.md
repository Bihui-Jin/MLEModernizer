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

0.9694615556143084

# 6. Current score

0.47103

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code crashes because it assumes an `/kaggle/input/submissions/` directory containing at least 3 submission files, but that folder doesn’t exist in this environment, so `submissions_all` is empty. I fix this by (1) switching to a safe, existing input path, (2) making the ensembling robust to “no external submissions found” by falling back to a valid baseline submission built from `sample_submission.csv`, and (3) ensuring the output is a correctly formatted `submission.csv` with the required columns and aligned `image_id`s from `test.csv`. This run end-to-end and produce a valid submission file; score be a simple baseline (since no model code is provided here).'
- What this solution (achieved 0.51861) has done: 'Your current 0.5 score comes from writing the sample submission probabilities (uniform 0.25), which has essentially no ranking signal for ROC AUC. To move toward the 0.969 target without changing the “no training” core approach, I replace the uniform fallback with a minimal heuristic that uses the provided train label priors (class prevalences) and a tiny per-image deterministic jitter derived from `image_id` to create consistent, non-constant rankings per class (ROC AUC needs ranking, not perfect calibration). The external-submission ensembling path remains intact; only the fallback is improved. This should increase the score above 0.5 while staying lightweight and within the same end-to-end submission-writing logic.'
- What this solution (achieved 0.41667) has done: 'Your current score is low because the fallback predictions are almost constant per class (priors + tiny jitter), which gives weak ranking signal for ROC AUC. Without changing the overall “no training / heuristic fallback” core approach, I increase the deterministic per-image variation and make it class-specific using multiple independent hash-based features, then combine them with fixed weights so each class gets a stronger, more diverse ranking. I also normalize each class’s raw scores via rank-to-uniform mapping on the test set to guarantee non-constant, well-spread predictions (good for AUC, which only cares about ranking), while keeping outputs clipped to [0,1] and submission formatting identical. The external-submission ensembling path stays intact; only the fallback is strengthened to move the score upward toward your target.'
- What this solution (achieved 0.47103) has done: 'Your current score is far below the target, so we should increase it with minimal, safe changes that keep your “no training / heuristic fallback” core approach intact. The biggest issue is that the fallback produces essentially random rankings, which won’t correlate with true labels and therefore can’t yield high ROC AUC. I change the fallback to use **image-derived features** (mean/std of RGB channels) from the provided JPGs, then apply the same **rank-to-uniform** step per class to maximize AUC-relevant ranking spread; this stays lightweight and deterministic while adding real signal. The external-submission ensembling path and submission formatting/alignment remain unchanged.'

# 9. Code solution

## === cell 0
import os
import hashlib
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input"

DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename, roots):
    for r in roots:
        p = os.path.join(r, filename)
        if os.path.exists(p):
            return p
    for r in roots:
        if os.path.exists(r):
            for dirpath, _, filenames in os.walk(r):
                if filename in filenames:
                    return os.path.join(dirpath, filename)
    return None


sample_path = _find_file("sample_submission.csv", DATA_DIR_CANDIDATES)
test_path = _find_file("test.csv", DATA_DIR_CANDIDATES)
train_path = _find_file("train.csv", DATA_DIR_CANDIDATES)

if sample_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not locate required files. sample_path={sample_path}, test_path={test_path}"
    )

print("Using sample_submission:", sample_path)
print("Using test.csv:", test_path)
print("Using train.csv:", train_path)



## === cell 2
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if not filename.lower().endswith(".csv"):
            continue
        full = os.path.join(dirname, filename)
        base = os.path.basename(full).lower()
        if base in {"sample_submission.csv", "train.csv", "test.csv"}:
            continue
        submissions_all.append(full)

submissions_all.sort()
print("Found candidate external submissions:", len(submissions_all))
print(submissions_all[:20])




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted sum ensemble of provided submission files.
    Fixes:
      - validates indices and weights
      - always returns an array shaped (n_test, 4)
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} vs {len(weights)}"
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {idx} but only {len(submissions_all)} files available."
            )
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        df = pd.read_csv(path)
        needed = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Submission file {path} is missing columns: {missing}")
        arr = df.loc[:, needed].to_numpy(dtype="float64")
        submission_with_weight.append(arr * w)

    submission_sum = submission_with_weight[0]
    for arr in submission_with_weight[1:]:
        submission_sum = submission_sum + arr
    return submission_sum




## === cell 4
def make_submission_file(
    submission_avg, sample_submission_path, test_csv_path, out_path="submission.csv"
):
    """
    Fix: ensure correct columns/order and that image_id matches test.csv.
    Also clip predictions to [0,1] to be safe.
    """
    sample_df = pd.read_csv(sample_submission_path)
    test_df = pd.read_csv(test_csv_path)

    out = sample_df.copy()
    out = out.set_index("image_id")
    out = out.reindex(test_df["image_id"].values)
    out = out.reset_index()

    needed = ["healthy", "multiple_diseases", "rust", "scab"]
    if submission_avg is None:
        raise ValueError("submission_avg cannot be None")

    pred = pd.DataFrame(submission_avg, columns=needed)
    pred = pred.clip(0.0, 1.0)

    if len(pred) != len(out):
        raise ValueError(
            f"Prediction rows ({len(pred)}) do not match test rows ({len(out)})"
        )

    out[needed] = pred[needed].values
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {out.shape} and columns {list(out.columns)}")




## === cell 5
def _id_to_u01_salt(image_id: str, salt: str) -> float:
    """
    Deterministic pseudo-random in [0,1) from image_id + salt (stable across runs).
    """
    h = hashlib.md5((salt + "::" + image_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(16**8)


def _rank_to_uniform(x):
    """
    Map scores to (0,1) via ranks so each class has a full spread; AUC depends on ordering.
    """
    s = pd.Series(x)
    r = s.rank(method="average").to_numpy(dtype="float64")
    n = float(len(r))
    return r / (n + 1.0)


def _find_images_dir(roots):
    candidates = []
    for r in roots:
        p = os.path.join(r, "images")
        if os.path.isdir(p):
            candidates.append(p)
    if candidates:
        return candidates[0]

    for r in roots:
        if os.path.exists(r):
            for dirpath, dirnames, _ in os.walk(r):
                if "images" in dirnames:
                    return os.path.join(dirpath, "images")
    return None


def _image_rgb_stats(image_path):
    """
    Minimal, fast feature extraction: mean and std of RGB channels.
    Uses PIL only if available; otherwise returns None (and fallback continues).
    """
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize((128, 128))
            import numpy as np

            arr = np.asarray(im, dtype="float32") / 255.0
            means = arr.mean(axis=(0, 1))
            stds = arr.std(axis=(0, 1))
            return (
                float(means[0]),
                float(means[1]),
                float(means[2]),
                float(stds[0]),
                float(stds[1]),
                float(stds[2]),
            )
    except Exception:
        return None


def fallback_predictions_from_priors_and_images(
    train_csv_path, test_csv_path, data_roots
):
    """
    Improvement toward target score (still no training; stronger heuristic fallback):
    - Hash-only ranks are essentially random -> low AUC.
    - Add simple image-derived features (RGB mean/std) to create meaningful rankings.
    - Keep the rank-to-uniform mapping per class (AUC focuses on ranking).
    """
    test_df = pd.read_csv(test_csv_path)
    needed = ["healthy", "multiple_diseases", "rust", "scab"]

    if train_csv_path is not None and os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        priors = train_df[needed].mean().to_dict()
    else:
        priors = {c: 0.25 for c in needed}

    images_dir = _find_images_dir(data_roots)
    print("Using images_dir:", images_dir)

    image_ids = test_df["image_id"].astype(str).tolist()

    raw = {c: [] for c in needed}
    for image_id in image_ids:
        u0 = _id_to_u01_salt(image_id, "pp2020_s0")
        u1 = _id_to_u01_salt(image_id, "pp2020_s1")
        z0, z1 = (u0 - 0.5, u1 - 0.5)

        r_mean = g_mean = b_mean = r_std = g_std = b_std = 0.0
        if images_dir is not None:
            img_path = os.path.join(images_dir, f"{image_id}.jpg")
            stats = _image_rgb_stats(img_path) if os.path.exists(img_path) else None
            if stats is not None:
                r_mean, g_mean, b_mean, r_std, g_std, b_std = stats

        raw["healthy"].append(
            0.10 * priors["healthy"] + 2.0 * (g_mean - r_mean) - 1.2 * r_std + 0.4 * z0
        )
        raw["multiple_diseases"].append(
            0.10 * priors["multiple_diseases"]
            + 1.0 * (r_std + g_std + b_std)
            - 0.8 * (g_mean + b_mean)
            + 0.4 * z1
        )
        raw["rust"].append(
            0.10 * priors["rust"]
            + 2.2 * (r_mean - g_mean)
            + 0.6 * r_std
            - 0.3 * b_mean
            - 0.2 * z0
        )
        raw["scab"].append(
            0.10 * priors["scab"]
            + 1.6 * (b_mean - g_mean)
            + 0.8 * b_std
            - 0.4 * r_mean
            - 0.2 * z1
        )

    preds = []
    for c in needed:
        u = _rank_to_uniform(raw[c])
        p = 0.90 * u + 0.10 * float(priors[c])
        preds.append(p)

    pred = pd.DataFrame({c: preds[i] for i, c in enumerate(needed)}).clip(0.0, 1.0)
    return pred.to_numpy(dtype="float64")




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.04, 0.2, 0.04])
else:
    print(
        "No (or insufficient) external submission files found to ensemble; using image-stat + rank-based fallback to improve ROC AUC signal."
    )
    submission_avg = fallback_predictions_from_priors_and_images(
        train_path, test_path, DATA_DIR_CANDIDATES
    )

make_submission_file(submission_avg, sample_path, test_path, out_path="submission.csv")
