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

0.4357340720221596

# 6. Current score

0.38173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.3327) has done: 'I correct the data paths so the script can locate train.csv and the test image folder, add a fallback lookup, and slightly increase the number of frequent tags predicted (TOP_N = 5) to raise the F1 score while keeping the original logic unchanged. The updated cells also ensure the submission DataFrame is created and saved as submission.csv with the required columns.'
- What this solution (achieved 0.30565) has done: 'I increase the number of most‑frequent tags that are predicted for every test image (TOP_N) from 5 to 12. This adds more likely disease labels to each prediction, raising recall while keeping precision reasonable, which should lift the mean F1‑Score toward the target without changing the overall modelling approach.'
- What this solution (achieved 0.28656) has done: 'I lower the number of globally‑predicted tags from 12 to 1 so that every test image receives only the most frequent label (likely “healthy”). This reduces false‑positives and should raise the mean F1‑Score, moving the result closer to the target without altering the overall pipeline.'
- What this solution (achieved 0.3327) has done: 'I raise the number of globally‑predicted tags from 1 to 5 (TOP_N = 5), which previously boosted the mean F1 from ~0.29 to ~0.33, moving the score closer to the target while keeping the original simple baseline unchanged. No other logic is altered.'
- What this solution (achieved 0.35308) has done: 'I replace the fixed‑TOP_N selection with a frequency‑threshold based tag list: tags are added in order of overall frequency until a chosen cumulative proportion (e.g., 85 %) of all occurrences is reached, then capped at a reasonable maximum (5). This keeps the original “global‑frequency” idea but usually yields fewer, more representative tags per image, improving precision and thus moving the mean F1‑Score closer to the target. The rest of the pipeline (path handling, CSV writing) remains unchanged.'
- What this solution (achieved 0.38173) has done: 'I add a lightweight hyper‑parameter search that evaluates several cumulative‑frequency thresholds and max‑tag limits on the training data itself, picking the combination that yields the highest mean F1 using the same “global‑frequency” logic. This keeps the core approach unchanged while tuning the TOP_N and frequency cut‑off to move the score upward toward the target. The rest of the pipeline (path handling and CSV writing) stays the same.'
- What this solution (achieved 0.38173) has done: 'I broaden the tag‑selection search to consider larger prefix lengths of the most‑frequent tags (up to 30) instead of limiting the search by cumulative frequency ratios and a small max‑tag list. By evaluating every prefix size we can find a globally‑best tag set that improves the mean F1 on the training data, moving the score closer to the target while preserving the original simple “global‑frequency” prediction logic.'
- What this solution (achieved 0.38173) has done: 'I add a lightweight search over cumulative‑frequency thresholds (in addition to the existing top‑k loop) to pick the globally‑predicted tag set that yields the highest mean F1 on the training data. This keeps the original “same tags for every image” logic while giving a slightly better tag list, moving the score upward toward the target. All other parts of the script remain unchanged.'
- What this solution (achieved 0.38173) has done: 'I expand the search for the best global tag list by allowing more candidate tags (up to 100 instead of 30) and by testing additional cumulative‑frequency thresholds (0.95 and 0.97). This keeps the original “same tags for every image” logic but gives the optimizer a larger space, which should modestly raise the mean F1 and move the score closer to the target. No other logic is altered.'
- What this solution (achieved 0.38173) has done: 'I add a deterministic train/validation split and use the validation set to choose the global tag list instead of evaluating on the whole training data. This keeps the same “global‑frequency” logic but avoids over‑fitting the tag selection, which should raise the test F1 toward the target. The rest of the pipeline (path handling, CSV creation) remains unchanged.'
- What this solution (achieved 0.38173) has done: 'I compute tag frequencies on the full training set (instead of only the training split) before searching for the best global tag list. This gives a more representative tag distribution, which should improve the validation F1 and move the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.38173) has done: 'I add a small hyper‑parameter sweep over several random seeds and compute tag frequencies only on the training split for each seed. This keeps the original “global‑frequency” prediction logic but lets us pick the tag list that gives the highest validation F1 across seeds, moving the mean F1 closer to the target. The rest of the pipeline (path handling, CSV creation) remains unchanged.'
- What this solution (achieved 0.38173) has done: 'I compute tag frequencies on the full training set (instead of each random split) so the global‑frequency tag list is based on all available data, which typically gives a more representative candidate set and improves validation F1, moving the score nearer the target. I also broaden the cumulative‑frequency thresholds slightly to explore a few more tag‑list sizes with negligible extra cost.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

candidate_paths = [
    "/kaggle/input/plant-pathology-2021-fgvgc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "./input/plant-pathology-2021-fgvc8",
    "./data/plant-pathology-2021-fgvc8",
    "./input",
    "./data",
]
BASE_INPUT = next((p for p in candidate_paths if os.path.isdir(p)), None)
if BASE_INPUT is None:
    raise FileNotFoundError("Base input directory for the competition not found.")

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

print("Resolved paths ->", TRAIN_CSV, TEST_IMG_DIR)




## === cell 1
import numpy as np

train_df = pd.read_csv(TRAIN_CSV)


def mean_f1(pred_tags_set, true_tags_list):
    """Calculate the mean F1 over all images."""
    total_f1 = 0.0
    n = len(true_tags_list)
    for pred_tags, true_tags in zip(pred_tags_set, true_tags_list):
        tp = len(pred_tags & true_tags)
        fp = len(pred_tags - true_tags)
        fn = len(true_tags - pred_tags)
        if tp + fp + fn == 0:
            f1 = 1.0
        else:
            f1 = (2 * tp) / (2 * tp + fp + fn)
        total_f1 += f1
    return total_f1 / n


seeds = [0, 42, 123, 2021, 7]
overall_best_f1 = -1.0
overall_best_tags = []

all_tags_full = []
for lbls in train_df["labels"]:
    all_tags_full.extend(lbls.split())
tag_counts_full = pd.Series(all_tags_full).value_counts()
max_k = min(100, len(tag_counts_full))

thresholds = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

for seed in seeds:
    np.random.seed(seed)
    perm = np.random.permutation(len(train_df))
    split_idx = int(0.8 * len(train_df))
    train_idx, val_idx = perm[:split_idx], perm[split_idx:]

    val_split = train_df.iloc[val_idx].reset_index(drop=True)
    true_tag_sets_val = [set(lbl.split()) for lbl in val_split["labels"]]

    best_f1_seed = -1.0
    best_tags_seed = []

    for k in range(1, max_k + 1):
        selected = tag_counts_full.index[:k].tolist()
        pred_sets = [set(selected) for _ in true_tag_sets_val]
        f1 = mean_f1(pred_sets, true_tag_sets_val)
        if f1 > best_f1_seed:
            best_f1_seed = f1
            best_tags_seed = selected

    cum_freq = tag_counts_full.cumsum() / tag_counts_full.sum()
    for th in thresholds:
        selected = tag_counts_full.index[cum_freq <= th].tolist()
        if not selected:
            continue
        pred_sets = [set(selected) for _ in true_tag_sets_val]
        f1 = mean_f1(pred_sets, true_tag_sets_val)
        if f1 > best_f1_seed:
            best_f1_seed = f1
            best_tags_seed = selected

    if best_f1_seed > overall_best_f1:
        overall_best_f1 = best_f1_seed
        overall_best_tags = best_tags_seed

print(
    f"Best configuration across seeds → {len(overall_best_tags)} tags, Validation F1 ≈ {overall_best_f1:.5f}"
)
print("Selected tags for submission:", overall_best_tags)

top_tags = overall_best_tags  # use this tag list for the test predictions




## === cell 2
test_filenames = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)

submission_df = pd.DataFrame(
    {
        "image": test_filenames,
        "labels": [" ".join(top_tags) for _ in test_filenames],
    }
)

print("Submission preview (first 5 rows):")
print(submission_df.head())




## === cell 3
output_path = "./submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
