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

3.10

# 3. Installed packages

No external packages required in the script and installed.

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

0.91331

# 6. Current score

0.51663

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the file‑path errors that prevent the script from loading the CSV files. The code now searches common Kaggle data locations (./data, ./input, /kaggle/input) and picks the first existing directory, guaranteeing that `train.csv`, `test.csv` and `sample_submission.csv` are found. No other logic is changed, so the model‑free baseline still runs and writes a valid `submission.csv` file.'
- What this solution (achieved 0.5) has done: 'I add a small heuristic that boosts predictions for any test images that also appear in the training set by copying their true label values, while keeping the original class‑mean baseline for all other images. This leverages available ground‑truth information without changing the overall model logic and is expected to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I keep the existing data loading and submission‑writing logic, but replace the uniform class‑mean predictions with a lightweight heuristic that parses each image filename for disease keywords (healthy, rust, scab, multiple) and assigns a high probability (0.9) to the detected class and distributes the remaining probability among the others. This adds meaningful ranking variation, which should raise the ROC‑AUC toward the target score while preserving the original overlap‑copying step and all file‑path handling.'
- What this solution (achieved 0.5) has done: 'I keep the existing data handling and filename‑based heuristic, but blend its predictions with the overall class‑mean baseline (70 % heuristic + 30 % class means). This adds a small amount of calibrated information while preserving the core logic, and should increase the ROC‑AUC from the current ~0.5 toward the target score. The script still writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I increase the influence of the filename‑based heuristic by raising its blend weight (from 0.70 to 0.90) and giving the detected class a slightly higher probability (0.95 instead of 0.90). This keeps the core logic unchanged while making the predictions more discriminative, which should lift the ROC‑AUC toward the target score.'
- What this solution (achieved 0.47271) has done: 'Implemented a lightweight numeric‑similarity heuristic: each test image’s ID number is matched to the nearest training image ID and its label vector is used as a prediction fallback. The `predict_from_filename` function now returns keyword‑based probabilities when disease keywords are present, otherwise it returns the nearest‑training‑label vector. These richer predictions are blended with the overall class‑means (still using the existing blend weight). This adds discriminative information while keeping the original pipeline and submission format unchanged.'
- What this solution (achieved 0.49243) has done: 'I improve the numeric‑ID fallback by averaging the labels of the k nearest training IDs (k = 3) instead of using only the single closest one, and I increase the heuristic blend weight to 0.95 so the richer heuristic influences the final predictions more. These minimal tweaks keep the overall pipeline unchanged while providing more discriminative information, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.48716) has done: 'I keep the overall pipeline unchanged but make the numeric‑ID fallback more informative by using a small number of nearest training IDs (K = 5) and weighting their label vectors inversely by distance rather than a simple mean. This provides smoother, more discriminative predictions while still preserving the original logic. I also lower the blend weight to 0.85 so the calibrated class‑means help regularize the heuristic, which should raise the ROC‑AUC toward the target without changing the core approach.'
- What this solution (achieved 0.49371) has done: 'We increase the influence of the filename‑based heuristic, raise its confidence for detected disease keywords, and use a slightly larger neighbourhood for the numeric‑ID fallback. These tweaks keep the overall pipeline unchanged but should make the predictions more discriminative, moving the ROC‑AUC closer to the target score.'
- What this solution (achieved 0.51663) has done: 'I keep the overall pipeline unchanged but make the heuristic more decisive: increase the confidence given to keyword matches, raise the blend weight toward the heuristic, and improve the numeric‑ID fallback by using a soft‑max‑style distance weighting (exponential decay) with a larger neighbourhood. These modest tweaks should raise the ROC‑AUC toward the target without altering the core logic.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
import numpy as np

_possible_base_dirs = [
    "./data/plant-pathology-2020-fgvc7",
    "./input/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
]
BASE_DIR = next((d for d in _possible_base_dirs if os.path.isdir(d)), None)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate the dataset directory. Checked paths: "
        + ", ".join(_possible_base_dirs)
    )

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_DIR, "sample_submission.csv")
SUBMISSION_OUT = "submission.csv"




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
class_means = train_df[target_cols].mean().values.astype(np.float32)  # shape (4,)


def _extract_num(image_id: str) -> int | None:
    m = re.search(r"(\d+)", image_id)
    return int(m.group(1)) if m else None


train_nums = train_df["image_id"].apply(_extract_num).values
train_labels = train_df[target_cols].values.astype(np.float32)  # shape (N_train, 4)




## === cell 2
test_df = pd.read_csv(TEST_CSV)
submission_df = pd.read_csv(SAMPLE_SUBMISSION)

submission_df = (
    submission_df.set_index("image_id").loc[test_df["image_id"]].reset_index()
)

K_NEAREST = 15  # larger neighbourhood for numeric fallback
_EPS = 1e-6  # avoid division‑by‑zero
HEURISTIC_MAX_PROB = 0.99  # give almost all mass to detected keywords
BLEND_WEIGHT = 0.98  # rely more on the heuristic
_DISTANCE_BETA = 0.01  # exponential decay factor for distance weighting


def predict_from_filename(fname: str) -> np.ndarray:
    """Return a probability vector for a given image id."""
    lower_name = fname.lower()
    matches = [col for col in target_cols if col in lower_name]
    if "multiple" in lower_name and "multiple_diseases" not in matches:
        matches.append("multiple_diseases")
    if matches:
        prob = np.full(
            len(target_cols),
            (1.0 - HEURISTIC_MAX_PROB) / (len(target_cols) - 1),
            dtype=np.float32,
        )
        for match in matches:
            idx = target_cols.index(match)
            prob[idx] = HEURISTIC_MAX_PROB / len(matches)
        return prob
    else:
        num = _extract_num(fname)
        if num is None or len(train_nums) == 0:
            return class_means
        diffs = np.abs(train_nums - num).astype(np.float32)
        if len(train_nums) <= K_NEAREST:
            nearest_idxs = np.arange(len(train_nums))
        else:
            nearest_idxs = np.argpartition(diffs, K_NEAREST)[:K_NEAREST]
        nearest_diffs = diffs[nearest_idxs]

        weights = np.exp(-_DISTANCE_BETA * nearest_diffs) + _EPS
        weighted_sum = (train_labels[nearest_idxs].T * weights).T.sum(axis=0)
        prob = weighted_sum / weights.sum()
        return prob.astype(np.float32)


heuristic_preds = np.vstack(
    [predict_from_filename(img_id) for img_id in test_df["image_id"]]
)

pred_matrix = BLEND_WEIGHT * heuristic_preds + (1 - BLEND_WEIGHT) * class_means

submission_df[target_cols] = pred_matrix

overlap_ids = set(train_df["image_id"]).intersection(set(test_df["image_id"]))
if overlap_ids:
    train_labels_subset = train_df.set_index("image_id")[target_cols]
    overlap_mask = submission_df["image_id"].isin(overlap_ids)
    submission_df.loc[overlap_mask, target_cols] = train_labels_subset.loc[
        submission_df.loc[overlap_mask, "image_id"]
    ].values




## === cell 3
submission_df.to_csv(SUBMISSION_OUT, index=False)
print(f"Submission file written to {SUBMISSION_OUT} with shape {submission_df.shape}")
