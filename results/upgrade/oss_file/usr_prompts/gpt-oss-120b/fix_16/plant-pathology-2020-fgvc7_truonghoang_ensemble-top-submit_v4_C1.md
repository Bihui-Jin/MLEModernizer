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

0.9667180849324454

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix adds robust handling for missing external prediction files, corrects the data path, ensures the sample submission is loaded safely, averages any available predictions, and always writes a valid `submission.csv` file. This eliminates the `FileNotFoundError` and `NameError` while keeping the original ensembling logic unchanged.'
- What this solution (achieved 0.53466) has done: 'I broaden the search paths so the script can actually find the external prediction CSVs (adding the typical Kaggle working directories), guard against missing prediction columns, and fall back to a simple class‑prior baseline when no external files are loaded. This keeps the original averaging logic intact while ensuring a non‑constant prediction vector, which should move the ROC‑AUC from the current 0.5 toward the target score.'
- What this solution (achieved 0.52356) has done: 'I make the script locate the external prediction CSVs correctly by searching in both the base directories and their common subfolder (`plant-pathology-2020-fgvc7`). I also align predictions on `image_id` when averaging, which ensures the right probabilities are summed even if row orders differ. These small fixes let the ensemble of strong external models be used, moving the ROC‑AUC score upward toward the target.'
- What this solution (achieved 0.5289) has done: 'I expand the search paths so the script can actually find the external prediction CSVs and the training file, compute class‑prior probabilities from the training set, and use those priors to fill missing predictions instead of zero. This keeps the original averaging logic but gives more sensible values for missing entries, which should raise the ROC‑AUC toward the target while preserving the core workflow.'
- What this solution (achieved 0.5) has done: 'I broaden the external‑prediction loading: after the original search, if no CSVs were successfully read, the script scan the data directory (and its immediate sub‑folder) for any `.csv` files that contain the required prediction columns and include them in the ensemble. This keeps the original averaging logic unchanged but makes it far more likely to use the strong external models that already exist, raising the ROC‑AUC toward the target.'
- What this solution (achieved 0.5) has done: 'I broaden the external‑prediction loading so that any CSV found (including in sub‑folders) that contains at least one of the required label columns is used, filling missing columns with class‑prior values. This increases the chance of leveraging useful predictions, raising the ROC‑AUC toward the target while keeping the original ensembling logic unchanged.'
- What this solution (achieved 0.5) has done: 'I add a lightweight heuristic that creates gently varying predictions when no external models are loaded. Instead of a constant prior or tiny random noise, each label gets a small offset based on the test row’s rank, producing non‑constant probabilities that are still centered around the class prior. This simple variation keeps the core ensembling logic unchanged while giving the ROC‑AUC a chance to move above the 0.5 baseline toward the target score.'
- What this solution (achieved 0.5) has done: 'I keep the existing ensemble logic but add a deterministic, hash‑based offset to the fallback predictions when no external CSVs are found. This introduces per‑image variation that can capture any hidden correlation between `image_id` and the targets, moving the ROC‑AUC slightly above the constant‑baseline score while preserving the core workflow.'
- What this solution (achieved 0.5) has done: 'I add a lightweight validation step that scores each external prediction file against the training labels using ROC‑AUC, then combine the predictions with weights proportional to their observed AUC (shifted so that a baseline of 0.5 receives no weight). This keeps the original ensembling logic but gives more influence to stronger external models, moving the score toward the target without changing the core workflow. If no external files are found, the fallback prior‑based predictions remain unchanged.'
- What this solution (achieved 0.5) has done: 'I broaden the automatic discovery of external prediction CSVs by comparing only the file names (instead of the whole path) when excluding the known list, and by loading any CSV that contains the required label columns. This small fix lets the script actually find and use the provided external model predictions, turning the constant‑baseline score of 0.5 into a much higher ROC‑AUC that moves toward the target.'
- What this solution (achieved 0.5) has done: 'I broaden the directory search for external prediction CSVs so that all possible data locations are scanned (not just the first found folder). This ensures that any available model predictions are loaded, allowing the existing weighted‑averaging logic to produce much better than the constant‑baseline score and move the evaluation toward the target.'
- What this solution (achieved 0.5) has done: 'I replace the AUC‑based weighting with simple equal weighting of all loaded external predictions (since the current AUC computation aligns predictions to train IDs and ends up giving zero or near‑zero weights, resulting in a constant‑baseline score). Equal weighting keeps the original ensembling logic but lets each external model contribute, which should raise the ROC‑AUC toward the target while preserving the core workflow. The fallback hash‑based predictions remain unchanged for the case where no external files are found.'
- What this solution (achieved 0.5) has done: 'I add a lightweight, deterministic offset based on the numeric part of each `image_id` for the fallback case (when no external predictions are found). This keeps the original ensemble logic unchanged but replaces the purely hash‑based random offset with a systematic one that can capture a weak signal from the image‑id numbers, helping the ROC‑AUC move above the 0.5 baseline toward the target. The rest of the script remains the same.'
- What this solution (achieved 0.5) has done: 'I expand the search for external prediction CSVs so that any file containing the required label columns **and** an `image_id` column is loaded, even when the known filenames are not found. This increases the chance of using real model predictions, moving the ROC‑AUC from the constant‑baseline towards the target. The fallback logic remains unchanged if no external files are discovered.'
- What this solution (achieved 0.5) has done: 'I keep the overall workflow unchanged but replace the constant‑plus‑small‑offset fallback with a simple, data‑driven heuristic: the numeric part of each image_id (e.g., “Train_370.jpg” → 370) is binned, and the average label values for those bins are computed from the training set. Test images receive the mean probabilities of their corresponding bin (or the overall class prior if the bin is unseen). This small, deterministic change uses existing information to create per‑image variations that are correlated with the true labels, moving the ROC‑AUC toward the target without altering the core ensembling logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import re
from sklearn.metrics import roc_auc_score

_possible_paths = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../input/plantpathology",  # legacy typo used in original code
    "/kaggle/working",
    os.getcwd(),
    os.path.join(os.getcwd(), "data", "plant-pathology-2020-fgvc7"),
    os.path.join(os.getcwd(), "data"),
    os.path.join("..", "working", "plant-pathology-2020-fgvc7"),
]


def _locate_file(fname):
    for root in _possible_paths:
        direct = os.path.join(root, fname)
        if os.path.isfile(direct):
            return direct
        subfolder = os.path.join(root, "plant-pathology-2020-fgvc7")
        inside = os.path.join(subfolder, fname)
        if os.path.isfile(inside):
            return inside
    return None


base_path = next((p for p in _possible_paths if os.path.isdir(p)), None)
if base_path is None:
    raise FileNotFoundError("Could not locate the competition data directory.")

external_csvs = [
    "effnet-fastai-folds-x5_version3.csv",
    "fork-of-plant-2020-tpu-915e9c_version1.csv",
    "plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv",
    "public-first-score-tpu-incepresnetv2-enb7_version8.csv",
    "classification-densenet201-efficientnetb7.csv",
    "plant-pathology-2020-efficientnetb7-0-980-score.csv",
    "tf-zoo-models-on-tpu.csv",
]

pred_cols = ["healthy", "multiple_diseases", "rust", "scab"]
dsub = []  # will hold dataframes of external predictions

for fname in external_csvs:
    fpath = _locate_file(fname)
    if fpath is not None:
        try:
            df = pd.read_csv(fpath)
            if "image_id" in df.columns and any(col in df.columns for col in pred_cols):
                dsub.append(df)
            else:
                print(f"Warning: {fname} missing image_id or target columns; skipped.")
        except Exception as e:
            print(f"Warning: failed to read {fpath}: {e}")
    else:
        print(f"Info: {fname} not found in any searched location.")

known_set = set(external_csvs)
search_dirs = [p for p in _possible_paths if os.path.isdir(p)]

for root_dir in search_dirs:
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            if f in known_set:
                continue
            full_path = os.path.join(root, f)
            try:
                df = pd.read_csv(full_path)
                if "image_id" in df.columns and any(
                    col in df.columns for col in pred_cols
                ):
                    print(
                        f"Info: using additional external prediction file {full_path}"
                    )
                    dsub.append(df)
            except Exception:
                continue

n = len(dsub)  # number of successfully loaded external submissions

train_path = _locate_file("train.csv")
priors = None
train_df = None
if train_path is not None:
    try:
        train_df = pd.read_csv(train_path)
        priors = train_df[pred_cols].mean()
    except Exception as e:
        print(f"Warning: could not read train.csv for priors: {e}")

bin_means = None
if train_df is not None:

    def _numeric_part(s):
        m = re.search(r"\d+", str(s))
        return int(m.group()) if m else 0

    train_df = train_df.copy()
    train_df["img_bin"] = train_df["image_id"].apply(_numeric_part) % 10
    bin_means = train_df.groupby("img_bin")[pred_cols].mean()
    if priors is not None:
        bin_means = bin_means.fillna(priors)




## === cell 1
sample_sub_path = _locate_file("sample_submission.csv")
if sample_sub_path is None:
    sample_sub_path = "sample_submission.csv"
sub = pd.read_csv(sample_sub_path)

if "image_id" not in sub.columns:
    raise KeyError("sample submission must contain 'image_id' column.")
sub = sub.set_index("image_id")

for col in pred_cols:
    sub[col] = 0.0

if n > 0:
    weights = np.ones(n) / n

    for w, df_ext in zip(weights, dsub):
        df_ext_idx = df_ext.set_index("image_id")
        for col in pred_cols:
            if col in df_ext_idx.columns:
                vals = df_ext_idx[col]
            else:
                fill_val = priors[col] if priors is not None else 0.0
                vals = pd.Series(fill_val, index=df_ext_idx.index)
            fill_value = priors[col] if priors is not None else 0.0
            sub[col] += w * vals.reindex(sub.index).fillna(fill_value)

else:
    def _numeric_part(s):
        m = re.search(r"\d+", str(s))
        return int(m.group()) if m else 0

    img_bins = sub.index.to_series().apply(_numeric_part) % 10
    for col in pred_cols:
        if bin_means is not None and col in bin_means.columns:
            preds = img_bins.map(bin_means[col]).fillna(
                priors[col] if priors is not None else 0.5
            )
        else:
            preds = pd.Series(
                priors[col] if priors is not None else 0.5, index=sub.index
            )
        sub[col] = np.clip(preds, 0.0, 1.0)

sub = sub.reset_index()

output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
