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

0.7448199445983383

# 6. Current score

0.30565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.35916) has done: 'I remove the failing imports and the unavailable model loading, then replace the inference step with a simple baseline that predicts the most frequent disease(s) observed in the training metadata. This fixes the runtime errors, ensures a correctly‑formatted `submission.csv` is written, and provides a reasonable baseline score that moves toward the target.'
- What this solution (achieved 0.28656) has done: 'I reduce the prediction to only the single most frequent disease label (TOP_K = 1) instead of the top two. Predicting fewer labels generally raises precision for the mean F1‑score, moving the current 0.35916 result closer to the target 0.7448 while keeping the core logic unchanged.'
- What this solution (achieved 0.3327) has done: 'I increase the number of most‑frequent disease labels predicted for every test image from 1 to 5. By outputting a few common labels we raise recall while still keeping precision reasonable, which should move the mean F1‑score noticeably closer to the target 0.7448. The rest of the pipeline and file handling remain unchanged.'
- What this solution (achieved 0.30565) has done: 'I increase the number of most‑frequent disease labels predicted for each test image from 5 to 10. By adding a few extra common labels we raise recall while keeping precision acceptable, which should move the mean F1‑score nearer to the target 0.7448 without altering the core baseline logic.'
- What this solution (achieved 0.35916) has done: 'I reduce the number of globally‑predicted labels from the ten most frequent diseases to the two most frequent ones. Predicting fewer common labels improves precision while keeping enough recall, which raises the mean F1‑score and moves the current 0.30565 result closer to the target 0.7448. This change is minimal, keeps the overall pipeline unchanged, and still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.38173) has done: 'The update adds a quick validation step that tests different values of `TOP_K` on a held‑out slice of the training data and selects the one that gives the highest mean sample‑wise F1‑score. This keeps the same simple “global‑most‑frequent‑labels” approach but chooses the number of labels to predict in a data‑driven way, which is expected to move the score closer to the target while preserving the original pipeline and output format.'
- What this solution (achieved 0.35308) has done: 'I keep the original global‑most‑frequent‑label baseline but add a tiny heuristic that always includes the “complex” label (if it exists in the training set) in every prediction. This modest extra label can improve recall for the few images that contain “complex” without heavily harming precision, moving the mean‑F1 score closer to the target while preserving the core logic and file handling.'
- What this solution (achieved 0.35308) has done: 'I keep the original simple “global‑most‑frequent‑labels” baseline but add a tiny data‑driven heuristic: after picking the best TOP_K labels on a validation split, I look at the training rows that contain any of those top labels and add the most common additional label that co‑occurs with them (if it improves the validation mean F1). This modest extra label can raise recall without hurting precision much, moving the score closer to the target while preserving the core logic and output format.'
- What this solution (achieved 0.35308) has done: 'I expand the search for the optimal number of global labels by allowing up to all classes and then iteratively add the most helpful co‑occurring labels as long as they improve validation F1. This keeps the same simple “global‑most‑frequent‑labels” logic but tunes it more thoroughly, which should raise the mean F1 toward the target while still writing a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.30565) has done: 'I keep the original “global‑most‑frequent‑labels” pipeline but add a simple fallback: if the validation‑based F1 is noticeably below the target (here < 0.5), we predict **all** observed labels (plus “complex” if present). This dramatically raises recall for every image while preserving the core logic and still writing a correctly formatted `submission.csv`, moving the score closer to the target without altering the overall architecture.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd



## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SUBMISSION_PATH = "submission.csv"



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)

all_labels = train_df["labels"].str.split(" ").explode()
label_counts = all_labels.value_counts()

val_frac = 0.2
val_df = train_df.sample(frac=val_frac, random_state=42).reset_index(drop=True)


def mean_sample_f1(pred_labels, true_labels):
    f1s = []
    for pred, true in zip(pred_labels, true_labels):
        pred_set = set(pred)
        true_set = set(true)
        if not pred_set and not true_set:
            f1s.append(1.0)
            continue
        intersect = pred_set.intersection(true_set)
        if len(pred_set) + len(true_set) == 0:
            f1 = 0.0
        else:
            f1 = 2 * len(intersect) / (len(pred_set) + len(true_set))
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


max_k = len(label_counts)  # allow any number of labels
best_k = 1
best_score = 0.0
true_labels_val = val_df["labels"].str.split(" ").tolist()

for k in range(1, max_k + 1):
    top_k_labels = label_counts.head(k).index.tolist()
    pred_labels_val = [top_k_labels] * len(val_df)
    score = mean_sample_f1(pred_labels_val, true_labels_val)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Selected TOP_K={best_k} with validation mean F1={best_score:.5f}")

TARGET_SCORE = 0.7448199445983383
if best_score < 0.5:  # heuristic threshold to trigger the fallback
    best_k = max_k
    best_score = mean_sample_f1(
        [label_counts.index.tolist()] * len(val_df), true_labels_val
    )
    print(
        f"Fallback triggered: using all {best_k} labels (validation mean F1≈{best_score:.5f})"
    )

top_labels = label_counts.head(best_k).index.tolist()

if "complex" in label_counts.index and "complex" not in top_labels:
    top_labels.append("complex")
    print('Added "complex" label to predictions for better recall.')

train_label_sets = train_df["labels"].str.split(" ").tolist()
cooccur_counter = {}

for label_set in train_label_sets:
    if any(lbl in top_labels for lbl in label_set):
        for lbl in label_set:
            if lbl not in top_labels:
                cooccur_counter[lbl] = cooccur_counter.get(lbl, 0) + 1

sorted_candidates = sorted(cooccur_counter.items(), key=lambda x: x[1], reverse=True)

added_extra = []
for candidate, _ in sorted_candidates:
    trial_labels = top_labels + [candidate]
    pred_labels_val_ext = [trial_labels] * len(val_df)
    ext_score = mean_sample_f1(pred_labels_val_ext, true_labels_val)
    if ext_score > best_score:
        top_labels.append(candidate)
        added_extra.append(candidate)
        best_score = ext_score
        print(
            f'Added co‑occurring label "{candidate}" improving validation F1 to {best_score:.5f}'
        )
    if len(added_extra) >= 5:
        break

if not added_extra:
    print("No co‑occurring label improved validation F1.")

print("Final label set to predict:", top_labels)



## === cell 3
test_filenames = sorted(
    [
        f
        for f in os.listdir(TEST_IMG_DIR)
        if re.search(
            r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$",
            f,
            re.IGNORECASE,
        )
    ]
)
print(f"Found {len(test_filenames)} test images.")



## === cell 4
submission_df = pd.DataFrame(
    {
        "image": test_filenames,
        "labels": [" ".join(top_labels) for _ in range(len(test_filenames))],
    }
)
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
display(submission_df.head())
