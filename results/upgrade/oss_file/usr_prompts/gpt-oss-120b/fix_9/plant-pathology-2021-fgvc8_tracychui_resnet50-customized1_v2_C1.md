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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.3850230840258536

# 6. Current score

0.30565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'Implemented a minimal, end‑to‑end pipeline that avoids the missing model and problematic imports. The script now:
- Imports only the needed libraries.
- Loads the training CSV, computes the most frequent label (“healthy” in this dataset) and uses it as a default prediction.
- Lists all test image filenames from the input folder.
- Builds the required submission DataFrame with columns `image` and `labels`.
- Writes a valid `submission.csv` file.'
- What this solution (achieved 0.24791) has done: 'The fix removes the faulty TensorFlow import that caused the runtime error and adds a lightweight heuristic: it builds a mapping from filename prefixes to the most common label seen in the training set for that prefix. Test images are then predicted using this prefix‑based label when available, otherwise falling back to the overall most frequent label. This change keeps the original simple baseline while providing a modest, score‑boosting improvement and ensures a proper `submission.csv` is written.'
- What this solution (achieved 0.31987) has done: 'I keep the overall workflow but add a small but effective change: for each filename prefix I store the full set of labels seen in the training data and use that exact set as the prediction when the prefix occurs in the test set. If a prefix is unseen I fall back to the overall most‑common label. This preserves the original simple heuristic while giving the model the ability to output multiple correct labels, which should raise the mean F1 toward the target.'
- What this solution (achieved 0.30794) has done: 'I keep the overall pipeline unchanged but improve the prediction rule: instead of outputting every label that ever appeared for a filename prefix, I predict only the most frequent one or, if several labels are tied, the top two. This reduces noisy extra tags, increasing precision and therefore the mean F1‑Score, moving the result closer to the target while preserving the original structure.'
- What this solution (achieved 0.30565) has done: 'I add a quick validation step that tries a few prefix lengths (2‑5) and two label‑selection strategies (all observed labels vs top‑2 most frequent). The script pick the combination that gives the highest mean F1 on a held‑out split of the training data, then rebuild the prefix counters with the full training set using that chosen configuration before generating the final submission. This keeps the original heuristic‑based approach while making it slightly more tailored, which should lift the score toward the target.'
- What this solution (achieved 0.30565) has done: 'I load the training CSV into `train_df`, fix the variable name errors, recompute the prefix counters on the full training set after selecting the best validation configuration, and then generate predictions for the test images and write a proper `submission.csv`. These minimal fixes resolve the NameErrors and ensure a valid submission file is created, while using the same simple prefix‑based heuristic that was already tuned on a validation split, moving the score toward the target.'
- What this solution (achieved 0.30565) has done: 'I add a lightweight “top‑1” prediction mode and extend the prefix‑length search up to 7 characters. This keeps the original prefix‑based heuristic while giving a higher‑precision option that often boosts macro F1, and it lets the validation loop choose the best configuration automatically. The rest of the pipeline stays unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import random



## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"

train_path = os.path.join(DATA_ROOT, "train.csv")
train_df = pd.read_csv(train_path)




## === cell 2
def build_prefix_counters(df, prefix_len):
    """
    Build:
    - overall_top: list of labels ordered by decreasing frequency in the whole dataframe.
    - prefixes: dict mapping each filename prefix to a Counter of label occurrences.
    """
    overall = Counter()
    prefixes = {}
    for _, row in df.iterrows():
        img = row["image"]
        labs = row["labels"].split()
        for l in labs:
            overall[l] += 1
        pref = os.path.splitext(img)[0][:prefix_len]
        if pref not in prefixes:
            prefixes[pref] = Counter()
        for l in labs:
            prefixes[pref][l] += 1
    overall_top = [lbl for lbl, _ in overall.most_common()] if overall else ["healthy"]
    return overall_top, prefixes


def predict(overall_top, prefix_counters, files, prefix_len, mode):
    """
    Predict labels for a list of image filenames.
    - mode can be "all", "top2", or "top1".
    - If the prefix exists, return the requested set of labels.
    - If the prefix is absent, fall back to the most common global labels:
        * "top2" -> overall_top[:2]
        * "top1" -> overall_top[:1]
        * "all"  -> overall_top[:2] (same fallback as before)
    """
    if mode == "top1":
        fallback = overall_top[:1]
    else:
        fallback = overall_top[:2]  # default for "all" and "top2"
    preds = []
    for img in files:
        pref = os.path.splitext(img)[0][:prefix_len]
        if pref in prefix_counters and prefix_counters[pref]:
            if mode == "top2":
                most_common = prefix_counters[pref].most_common(2)
                top = [lbl for lbl, _ in most_common]
            elif mode == "top1":
                top = [prefix_counters[pref].most_common(1)[0][0]]
            else:  # mode == "all"
                top = list(prefix_counters[pref].keys())
            pred = " ".join(sorted(top))
        else:
            pred = " ".join(sorted(fallback))
        preds.append(pred)
    return preds




## === cell 3
random.seed(42)
indices = list(train_df.index)
random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

train_split = train_df.loc[train_idx].reset_index(drop=True)
val_split = train_df.loc[val_idx].reset_index(drop=True)

best_f1 = -1.0
best_cfg = None

for plen in [2, 3, 4, 5, 6, 7]:
    overall_top, pref_counters = build_prefix_counters(train_split, plen)
    val_files = val_split["image"].tolist()
    val_true = [row.split() for row in val_split["labels"]]

    for mode in ["all", "top2", "top1"]:
        val_pred_str = predict(overall_top, pref_counters, val_files, plen, mode)
        val_pred = [p.split() for p in val_pred_str]

        mlb = MultiLabelBinarizer()
        all_labels = list(set([l for sub in (val_true + val_pred) for l in sub]))
        mlb.fit([all_labels])
        y_true = mlb.transform(val_true)
        y_pred = mlb.transform(val_pred)

        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = (plen, mode, overall_top, pref_counters)

print(
    f"Best validation F1: {best_f1:.5f} using prefix_len={best_cfg[0]}, mode={best_cfg[1]}"
)



## === cell 4
chosen_prefix_len, chosen_mode, _, _ = best_cfg
overall_top_full, prefix_counters_full = build_prefix_counters(
    train_df, chosen_prefix_len
)



## === cell 5
test_dir = os.path.join(DATA_ROOT, "test_images")
test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
print("Number of test images:", len(test_files))



## === cell 6
pred_labels = predict(
    overall_top_full,
    prefix_counters_full,
    test_files,
    chosen_prefix_len,
    chosen_mode,
)



## === cell 7
submission_df = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
