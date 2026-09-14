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
numpy==1.26.4
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

0.9675421146508574

# 6. Current score

0.52452

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix removes the unavailable external CSV files, computes per‑class average probabilities from the provided training data, and fills the sample submission with these constants so a valid `submission.csv` is written. This restores execution, eliminates the NameError, and creates a sensible baseline submission.'
- What this solution (achieved 0.5) has done: 'I keep the original data loading and mean‑baseline calculations, but add a merge with the training labels for any test image IDs that also appear in the training set. For those matches we use the exact training label values (0/1) as predictions, which provides real signal instead of constant averages. Unmatched test rows keep the global mean probabilities, preserving a valid fallback. This small augmentation should raise the ROC‑AUC well above the baseline 0.5 and move the score toward the target while leaving the core logic unchanged.'
- What this solution (achieved 0.47271) has done: 'I add a lightweight numeric‑ID similarity heuristic to give the model a non‑random signal for test images that do not appear in the training set. After loading the data I extract the numeric part of each `image_id` and, for any test rows still missing labels after the exact‑match merge, I fill them with the labels of the nearest training image (based on the numeric ID). Remaining missing values fall back to the global class means. This small change keeps the original workflow intact while providing extra information that should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.47271) has done: 'The fix resolves the missing “num” column caused by duplicate column names after merging the test and train data. After the merge we create a single “num” column from the available “num_x”/“num_y” columns (or recompute it if needed) and drop the duplicates. This allows the subsequent modulo‑based feature engineering to run without errors, producing a valid `submission.csv`. No other logic is changed, preserving the original modeling approach.'
- What this solution (achieved 0.47271) has done: 'I replace the single‑nearest‑match heuristic with a forward‑ and backward‑nearest average, which gives each test image a smoother label estimate based on its surrounding training IDs. This minor change keeps the overall workflow unchanged while providing richer information to the ROC‑AUC calculation and is expected to raise the score toward the target.'
- What this solution (achieved 0.47271) has done: 'I replace the simple forward‑/backward‑average with a distance‑weighted average of the nearest training rows, giving more influence to the closer neighbour. This keeps the overall pipeline unchanged while providing smarter predictions that should raise the ROC‑AUC toward the target. The rest of the code, including exact matches, modulo‑group means, and global fallback, remains the same.'
- What this solution (achieved 0.47271) has done: 'I replace the weighted‑average nearest‑neighbour fill with a simpler nearest‑neighbour rule: for each test image that is not in the training set, the prediction copy the label of the closest training image based on the numeric part of the ID (forward or backward whichever is nearer). This gives more varied predictions than the previous smooth average while preserving all later fallback steps, so the ROC‑AUC should move upward toward the target.'
- What this solution (achieved 0.47271) has done: 'I replace the forward‑ and backward‑nearest‑neighbor logic with a single “nearest” merge‑asof, which provides a more direct estimate for each test image and should improve the ROC‑AUC while keeping the rest of the pipeline unchanged. The fallback steps (mod‑group means and global means) remain the same, ensuring a valid submission file.'
- What this solution (achieved 0.47271) has done: 'I replace the single‑nearest‑neighbor fill with a forward‑and‑backward‑nearest‑neighbor average, giving each prediction a fractional probability (0, 0.5, 1) instead of a hard 0/1 label. This richer signal should improve the ROC‑AUC and move the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.47271) has done: 'I replace the simple un‑weighted forward/backward average with a distance‑weighted nearest‑neighbor estimate. After the forward and backward “asof” merges I compute the numeric distance to each neighbour, give the closer neighbour more weight, and use the weighted mean to fill the missing predictions. This adds a modest amount of signal while preserving the overall pipeline, so the ROC‑AUC should move upward toward the target score.'
- What this solution (achieved 0.47271) has done: 'I add a lightweight hierarchical fallback that groups images by the integer division of their numeric ID (num // 10). After the existing modulo‑group means are applied, the script compute per‑group class averages from the training data and use them to fill any remaining missing predictions before falling back to the global means. This keeps the original workflow intact while providing extra signal to improve the ROC‑AUC toward the target score.'
- What this solution (achieved 0.53841) has done: 'I add a small numeric‑ID window‑averaging step that replaces the earlier forward/backward weighted‑average – it computes the mean label of all training images whose numeric ID is within ±5 of each test image. This gives a richer, distance‑based signal while keeping the rest of the pipeline unchanged, and it should move the ROC‑AUC closer to the target. The rest of the fallback hierarchy (mod‑group, group, global mean) remains the same, ensuring a valid submission file is still written.'
- What this solution (achieved 0.52452) has done: 'I increase the numeric‑ID window size from 5 to 10 in the window‑mean fallback (cell 4). A larger window provides more neighboring training samples, giving a richer probability estimate for test images that lack exact or nearest‑neighbor matches. This small adjustment keeps the overall pipeline unchanged while adding extra signal, which should raise the ROC AUC and move the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import re




## === cell 1
train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
train_df = pd.read_csv(train_path)

mean_healthy = train_df["healthy"].mean()
mean_multiple = train_df["multiple_diseases"].mean()
mean_rust = train_df["rust"].mean()
mean_scab = train_df["scab"].mean()


def _num_from_id(img_id):
    m = re.search(r"\d+", str(img_id))
    return int(m.group()) if m else -1


train_df["num"] = train_df["image_id"].apply(_num_from_id)




## === cell 2
test_path = "../input/plant-pathology-2020-fgvc7/test.csv"
test_df = pd.read_csv(test_path)

sample_sub_path = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"
sub = pd.read_csv(sample_sub_path)

test_df["num"] = test_df["image_id"].apply(_num_from_id)




## === cell 3
preds = test_df.merge(
    train_df[["image_id", "healthy", "multiple_diseases", "rust", "scab", "num"]],
    on="image_id",
    how="left",
)

if "num_x" in preds.columns:
    preds["num"] = preds["num_x"]
elif "num_y" in preds.columns:
    preds["num"] = preds["num_y"]
else:
    preds["num"] = preds["image_id"].apply(_num_from_id)
preds.drop(columns=[c for c in ["num_x", "num_y"] if c in preds.columns], inplace=True)

cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 4
train_sorted = train_df.sort_values("num").reset_index(drop=True)
train_nums = train_sorted["num"].values
train_vals = {col: train_sorted[col].values.astype(float) for col in cols}
window = 10  # increased window to look ±10 numeric IDs for a richer local estimate


def _window_mean(test_num, col_vals):
    lo = np.searchsorted(train_nums, test_num - window, side="left")
    hi = np.searchsorted(train_nums, test_num + window, side="right")
    if lo < hi:
        return np.mean(col_vals[lo:hi])
    else:
        return np.nan  # caller will handle fallback


for col in cols:
    missing_mask = preds[col].isna()
    if missing_mask.any():
        test_missing_nums = preds.loc[missing_mask, "num"].values
        window_means = [_window_mean(num, train_vals[col]) for num in test_missing_nums]
        preds.loc[missing_mask, col] = window_means




## === cell 5
test_df["_orig_idx"] = np.arange(len(test_df))
test_sorted = test_df.sort_values("num").reset_index(drop=True)

train_nearest = (
    train_df[["image_id", "healthy", "multiple_diseases", "rust", "scab", "num"]]
    .sort_values("num")
    .reset_index(drop=True)
)

forward = pd.merge_asof(test_sorted, train_nearest, on="num", direction="forward")
backward = pd.merge_asof(test_sorted, train_nearest, on="num", direction="backward")

dist_f = forward["num"] - test_sorted["num"]
dist_b = test_sorted["num"] - backward["num"]

weight_f = np.where(dist_f == 0, 1.0, 1.0 / (dist_f + 1e-6))
weight_b = np.where(dist_b == 0, 1.0, 1.0 / (dist_b + 1e-6))
norm = weight_f + weight_b

weighted_avg = pd.DataFrame(index=forward.index)
for col in cols:
    weighted_avg[col] = (forward[col] * weight_f + backward[col] * weight_b) / norm

weighted_avg["_orig_idx"] = test_sorted["_orig_idx"]
weighted_avg.set_index("_orig_idx", inplace=True)

for col in cols:
    preds[col].fillna(weighted_avg[col].reindex(preds.index), inplace=True)




## === cell 6
train_df["num_mod"] = train_df["num"] % 10
mod_means = (
    train_df.groupby("num_mod")[["healthy", "multiple_diseases", "rust", "scab"]]
    .mean()
    .reset_index()
)

preds["num_mod"] = preds["num"] % 10

preds = preds.merge(
    mod_means,
    on="num_mod",
    how="left",
    suffixes=("", "_mod"),
)

for col in cols:
    preds[col].fillna(preds[f"{col}_mod"], inplace=True)
    preds.drop(columns=[f"{col}_mod"], inplace=True)




## === cell 7
train_df["num_group"] = train_df["num"] // 10
group_means = (
    train_df.groupby("num_group")[["healthy", "multiple_diseases", "rust", "scab"]]
    .mean()
    .reset_index()
)

preds["num_group"] = preds["num"] // 10

preds = preds.merge(
    group_means,
    on="num_group",
    how="left",
    suffixes=("", "_group"),
)

for col in cols:
    preds[col].fillna(preds[f"{col}_group"], inplace=True)
    preds.drop(columns=[f"{col}_group"], inplace=True)




## === cell 8
preds["healthy"].fillna(mean_healthy, inplace=True)
preds["multiple_diseases"].fillna(mean_multiple, inplace=True)
preds["rust"].fillna(mean_rust, inplace=True)
preds["scab"].fillna(mean_scab, inplace=True)

submission = preds[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
submission.to_csv("submission.csv", index=False)
