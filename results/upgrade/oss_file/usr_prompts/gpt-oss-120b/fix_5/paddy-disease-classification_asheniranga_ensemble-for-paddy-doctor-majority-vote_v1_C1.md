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

# 8. Previous improvement plans

- What this solution (achieved 0.17487) has done: 'The fix removes the unnecessary TensorFlow import that caused a protobuf error, adds safe loading of any existing submission files (ignoring missing ones), and replaces the broken ensemble logic with a simple fallback that predicts the overall most common label from the training data for every test image. This ensures the script runs end‑to‑end, writes a correctly formatted CSV, and produces a valid submission file.'
- What this solution (achieved 0.17487) has done: 'We relax the strict requirement that *all* three external submission files must exist before using them. Instead, we gather whichever of the files are present, combine their predictions by row‑wise mode, and fall back to the most‑common training label only when none are available. This small change lets the script leverage any better predictions that happen to be present, moving the validation score toward the target while preserving the overall workflow and output format.'
- What this solution (achieved 0.17487) has done: 'I broaden the file‑search logic so the script can actually locate any existing model_submission_v5.csv files that may be stored in alternative folders, enabling the ensemble step to run on real predictions instead of falling back to a constant label. This small adjustment keeps the overall workflow unchanged while giving the model a chance to achieve a much higher accuracy, moving the score toward the target.'
- What this solution (achieved 0.17487) has done: 'I adjust the script to automatically discover any existing submission CSVs that match the pattern `model_submission_*.csv` across the usual Kaggle input directories, load all of them, and ensemble their predictions via mode. This broader search is likely to find a high‑accuracy pre‑computed submission (if present) instead of falling back to the single most‑common label, moving the validation score much closer to the target while keeping the original workflow unchanged. If no such files are found, the original fallback remains unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import scipy.stats as ss
import os
from pathlib import Path
import glob


def try_load(filename):
    """
    Search common Kaggle directories for the given filename and return the first
    successfully read CSV, or None if not found.
    """
    search_dirs = [
        "../input/paddydocoutputs/",
        "../input/paddy-doctor-training/",
        "../input/k/kasunpramodya/paddy-doctor-training/",
        "./",
        "../working/paddy-disease-classification/",
        "../input/paddy-disease-classification/",
        "../input/",
        "/kaggle/input/",
    ]
    for d in search_dirs:
        potential_path = Path(d) / filename
        if potential_path.is_file():
            try:
                return pd.read_csv(potential_path)
            except Exception:
                continue
    if Path(filename).is_file():
        try:
            return pd.read_csv(filename)
        except Exception:
            pass
    return None


def load_all_submissions():
    """
    Locate every CSV file whose name matches 'model_submission_*.csv' in the
    standard input locations and return a list of DataFrames.
    """
    search_dirs = [
        "../input/paddydocoutputs/",
        "../input/paddy-doctor-training/",
        "../input/k/kasunpramodya/paddy-doctor-training/",
        "./",
        "../working/paddy-disease-classification/",
        "../input/paddy-disease-classification/",
        "../input/",
        "/kaggle/input/",
    ]
    dfs = []
    seen_paths = set()
    for d in search_dirs:
        pattern = str(Path(d) / "model_submission_*.csv")
        for path_str in glob.glob(pattern):
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            try:
                df = pd.read_csv(path_str)
                if "image_id" in df.columns:
                    dfs.append(df)
            except Exception:
                continue
    return dfs


submission_dfs = load_all_submissions()
if not submission_dfs:
    s1 = try_load("model_submission_v5.csv")
    s2 = try_load("model_submission_v5.csv")
    s3 = try_load("model_submission_v5.csv")
    for s in (s1, s2, s3):
        if s is not None and "image_id" in s.columns:
            submission_dfs.append(s)



## === cell 1
prediction_dfs = []
for s in submission_dfs:
    if s is not None and "image_id" in s.columns:
        prediction_dfs.append(s.set_index("image_id"))

if prediction_dfs:
    combined = pd.concat(prediction_dfs, axis=1)
    mode_labels = combined.apply(lambda row: row.mode().iloc[0], axis=1)
    s_final = pd.DataFrame({"image_id": combined.index, "label": mode_labels.values})
else:
    train_path = "../input/paddy-disease-classification/train.csv"
    if not os.path.exists(train_path):
        train_path = "../input/train.csv"
    train_df = pd.read_csv(train_path)
    most_common_label = train_df["label"].mode()[0]

    sample_sub_path = "../input/paddy-disease-classification/sample_submission.csv"
    if not os.path.exists(sample_sub_path):
        sample_sub_path = "../input/sample_submission.csv"
    sample_sub = pd.read_csv(sample_sub_path)

    s_final = pd.DataFrame(
        {"image_id": sample_sub["image_id"], "label": most_common_label}
    )



## === cell 2
submission = s_final[["image_id", "label"]]
submission.to_csv("model_submission_v10.csv", index=False)
