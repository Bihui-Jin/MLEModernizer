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

0.9700224130896464

# 6. Current score

0.58121

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The error happens because the notebook expects pre-made submission files under `/kaggle/input/submissions/submissions/`, but that folder doesn’t exist here so `submissions_all` is empty and indexing `[0,1]` crashes. I keep the ensemble logic intact, but add a safe fallback: if no external submissions are found, create a valid submission using `sample_submission.csv` with uniform probabilities (0.25 each) so you always get a `submission.csv`. I also add minimal input validation (weights length, index bounds) and ensure the output columns/order exactly match the competition format. This run end-to-end and produce a valid `.csv` submission file.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score is coming from the uniform fallback (0.25 for every class), which is effectively random and caps ROC AUC near 0.5. To move toward the 0.97 target while keeping changes minimal and preserving the “no training / ensemble submissions” core logic, I replace the uniform fallback with a simple label-prior fallback computed from `train.csv` (mean of each target column) and applied to every test row. This is still a constant-prediction baseline (so same evaluation semantics and no model/feature changes), but it usually improves ROC AUC above 0.5 by reflecting true class imbalance. I also ensure we always align the submission rows to `test.csv` image_id order and keep columns exactly `image_id + TARGET_COLS`.'
- What this solution (achieved 0.65466) has done: 'Your current 0.5 score is coming from the constant “class prior” fallback, which still behaves close to random under mean column-wise ROC AUC. To move toward the 0.97 target while keeping the core “no training / submission-ensemble” logic, I keep the external-submission ensemble path unchanged, but improve the fallback to a minimal, fully classical baseline: train a one-vs-rest logistic regression on simple image statistics (color means/stds) extracted from the provided JPGs, then predict probabilities for the test set. This preserves evaluation semantics (proper probabilities per class), uses only the provided data, and should materially increase AUC versus constant priors, while staying lightweight and within time. I also enforce exact `test.csv` order and required column names when writing `submission.csv`.'
- What this solution (achieved 0.64637) has done: 'Your current 0.65466 score comes from a very weak image-statistics + logistic regression fallback; to move closer to the 0.97 target without changing the overall approach, I keep the same feature extraction and OneVsRest LogisticRegression pipeline but make it behave better for ROC AUC. Specifically, I (1) use `class_weight="balanced"` to reduce bias from label imbalance, (2) increase `max_iter` to ensure convergence, and (3) apply a tiny amount of probability smoothing (mixing model probabilities with train label priors) to stabilize predictions without changing semantics. I also add lightweight caching for extracted features so the run stays well within the time limit and remains deterministic. Submission formatting and external-ensemble behavior remain unchanged.'
- What this solution (achieved 0.58121) has done: 'We need to move the ROC AUC score up from 0.646 toward 0.97 while keeping your core approach (external-submission ensemble else fallback) intact. The current fallback uses only very weak global image statistics; the minimal way to improve materially without changing the overall training approach is to extract slightly richer-but-still-simple features (downsampled RGB pixels + the existing stats) and use the same OneVsRest LogisticRegression pipeline. I also switch the fallback to a deterministic stratified split for calibration-free validation sanity (no metric tuning) and keep the same probability smoothing and submission formatting/alignment. This stays lightweight, uses only provided data, and should significantly improve separability versus mean/std-only features.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _find_first_existing_file(rel_path):
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, rel_path)
        if os.path.exists(p):
            return p
    return None


sample_sub_path = _find_first_existing_file("sample_submission.csv")
test_path = _find_first_existing_file("test.csv")
train_path = _find_first_existing_file("train.csv")


def _find_first_existing_dir(rel_path):
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, rel_path)
        if os.path.isdir(p):
            return p
    return None


images_dir = _find_first_existing_dir("images")



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; cannot ensemble zero submissions.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise FileNotFoundError(f"No submission files found under {SUBMISSIONS_PATH}")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        if sub_idx[i] < 0 or sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={sub_idx[i]} is out of range for submissions_all of length {len(submissions_all)}"
            )
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[:, TARGET_COLS].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all):
    if test_path is None:
        raise FileNotFoundError("Could not locate test.csv in known data roots.")
    test_df = pd.read_csv(test_path)
    if "image_id" not in test_df.columns:
        raise ValueError("test.csv must contain 'image_id' column.")

    if len(submissions_all) > 0:
        template_path = submissions_all[0]
        template_df = pd.read_csv(template_path)
    else:
        if sample_sub_path is None:
            raise FileNotFoundError(
                "Could not locate sample_submission.csv in known data roots."
            )
        template_df = pd.read_csv(sample_sub_path)

    if "image_id" not in template_df.columns:
        raise ValueError("Template submission must contain 'image_id' column.")

    submission_df = pd.DataFrame({"image_id": test_df["image_id"].values})
    for c in TARGET_COLS:
        submission_df[c] = 0.0

    if len(submission_df) != len(submission_avg):
        raise ValueError(
            f"submission_avg rows ({len(submission_avg)}) must match test rows ({len(submission_df)})."
        )

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.loc[:, TARGET_COLS] = submission_df.loc[:, TARGET_COLS].clip(0.0, 1.0)

    submission_df = submission_df[["image_id"] + TARGET_COLS]

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("Columns:", submission_df.columns.tolist())




## === cell 5
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.22, 0.78])
    make_submission_file(submission_avg, submissions_all)
else:
    if train_path is None:
        raise FileNotFoundError(
            "No external submissions found and train.csv is missing; cannot train fallback."
        )
    if test_path is None:
        raise FileNotFoundError(
            "No external submissions found and test.csv is missing; cannot size submission."
        )
    if images_dir is None:
        raise FileNotFoundError(
            "No external submissions found and images/ directory is missing; cannot extract image features."
        )

    import numpy as np

    try:
        from PIL import Image
    except Exception as e:
        raise ImportError(
            "PIL is required to read jpg images for the fallback baseline."
        ) from e

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    missing = [c for c in TARGET_COLS if c not in train_df.columns]
    if missing:
        raise ValueError(f"train.csv missing required target columns: {missing}")

    y_prior = train_df[TARGET_COLS].mean(axis=0).values.astype(np.float64)

    def _img_path(image_id: str) -> str:
        return os.path.join(images_dir, f"{image_id}.jpg")

    _feat_cache = {}

    def _extract_features(image_id: str) -> np.ndarray:
        if image_id in _feat_cache:
            return _feat_cache[image_id]
        p = _img_path(image_id)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image file: {p}")

        im = Image.open(p).convert("RGB")

        thumb = im.resize((32, 32), resample=Image.BILINEAR)
        arr_t = np.asarray(thumb, dtype=np.float32) / 255.0  # (32,32,3)
        thumb_flat = arr_t.reshape(-1).astype(np.float32)  # 3072 dims

        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        std_rgb = arr.reshape(-1, 3).std(axis=0)
        gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        mean_gray = np.array([gray.mean()], dtype=np.float32)
        std_gray = np.array([gray.std()], dtype=np.float32)

        feat = np.concatenate(
            [
                thumb_flat,
                mean_rgb.astype(np.float32),
                std_rgb.astype(np.float32),
                mean_gray,
                std_gray,
            ],
            axis=0,
        ).astype(np.float32)

        _feat_cache[image_id] = feat
        return feat

    X_train = np.vstack([_extract_features(i) for i in train_df["image_id"].values])
    y_train = train_df[TARGET_COLS].values.astype(int)
    X_test = np.vstack([_extract_features(i) for i in test_df["image_id"].values])

    base_lr = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=0,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ovr", OneVsRestClassifier(base_lr)),
        ]
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test).astype(np.float64)

    alpha = 0.03  # 3% prior mix
    proba = (1.0 - alpha) * proba + alpha * y_prior[None, :]
    proba = np.clip(proba, 0.0, 1.0)

    submission_avg = np.asarray(proba, dtype=float)
    make_submission_file(submission_avg, submissions_all)
