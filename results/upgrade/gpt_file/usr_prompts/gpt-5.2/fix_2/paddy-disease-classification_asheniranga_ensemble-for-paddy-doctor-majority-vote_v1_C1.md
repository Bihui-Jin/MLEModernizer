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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.976958525345622

# 6. Current score

0.17487

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.17487) has done: 'I remove the broken TensorFlow/Keras imports that are triggering the protobuf `MessageFactory` error, since they are unused in your current ensemble-by-mode approach. I also fix the missing input files by switching to the competition’s provided `sample_submission.csv` as the base, and fall back gracefully if any of the three external submission paths don’t exist. The ensemble logic (row-wise mode vote with a tie-breaker) is preserved; when there’s no majority we fall back to the first available submission (or keep the sample file’s labels if none exist) so the notebook always produces a valid `.csv` submission.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import scipy.stats as ss



## === cell 1
BASE_DIR = "/kaggle/input/paddy-disease-classification"

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample = pd.read_csv(sample_path)

candidate_paths = [
    "../input/paddydocoutputs/model_submission_v5.csv",
    "../input/paddy-doctor-training/model_submission_v5.csv",
    "../input/k/kasunpramodya/paddy-doctor-training/model_submission_v5.csv",
]

subs = []
loaded_paths = []
for p in candidate_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        if {"image_id", "label"}.issubset(df.columns):
            subs.append(df[["image_id", "label"]].copy())
            loaded_paths.append(p)

if len(subs) == 0:
    train_path = os.path.join(BASE_DIR, "train.csv")
    if os.path.exists(train_path):
        train = pd.read_csv(train_path)
        default_label = train["label"].mode().iloc[0]
    else:
        default_label = "normal"
    s = sample.copy()
    s["label"] = default_label
else:
    aligned = []
    for df in subs:
        df2 = sample[["image_id"]].merge(df, on="image_id", how="left")
        aligned.append(df2["label"].fillna("normal").to_frame())

    s_full = pd.concat([sample[["image_id"]]] + aligned, axis=1)
    s_full = s_full.set_index("image_id")

    s_full_mode = ss.mode(s_full.values, axis=1, keepdims=False)

    most = pd.Series(s_full_mode.mode, index=s_full.index, name="most")
    counts = pd.Series(s_full_mode.count, index=s_full.index, name="count")

    n_models = s_full.shape[1]
    majority_needed = int(np.floor(n_models / 2) + 1)

    s_final = most.to_frame()
    if len(subs) > 0:
        fallback = aligned[-1].iloc[
            :, 0
        ]  # last loaded submission labels aligned to sample
        weak_mask = counts < majority_needed
        s_final.loc[weak_mask, "most"] = fallback.loc[weak_mask].values

    s = s_final.rename(columns={"most": "label"}).reset_index()

s = sample[["image_id"]].merge(s[["image_id", "label"]], on="image_id", how="left")
s["label"] = s["label"].fillna(sample["label"])

print("Loaded external submissions:", loaded_paths)
print("Submission shape:", s.shape)
s.head()



## === cell 2
out_path = "model_submission_v10.csv"
s.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
