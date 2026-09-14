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

0.7917820867959388

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I remove the failing TensorFlow model loading and replace it with a lightweight baseline that predicts the most common disease labels based on their prevalence in the training data. This avoids the protobuf import error, eliminates the need for external model files, and ensures a valid `submission.csv` is written. The new code computes label frequencies, selects those occurring in more than 50% of the training samples (or falls back to the single most frequent label), and assigns these labels to every test image, keeping the original data paths and output format.'
- What this solution (achieved 0.28656) has done: 'I replace the fixed 0.5‑prevalence threshold with a tiny heuristic that predicts the N most common disease classes, where N is set to the average number of labels per training image (rounded). This keeps the deterministic baseline while giving each image a more realistic multilabel set, which should raise the mean F1‑score toward the target without changing the core workflow.'
- What this solution (achieved 0.28656) has done: 'I add a deterministic “hash‑based” variation to the baseline predictor so that different images receive a different number of the most frequent classes (instead of the same fixed set for every image). This small diversification keeps the same core logic and still uses the most common labels, but should raise the mean F1‑score toward the target without over‑complicating the model.'
- What this solution (achieved 0.26421) has done: 'I lower the prevalence threshold so that more of the common disease classes are included, and I let the hash‑based diversification choose any number of those classes up to the full set. This keeps the same deterministic baseline while giving each image a richer multilabel prediction, which should raise the mean F1‑Score toward the target.'
- What this solution (achieved 0.22145) has done: 'I replace the hash‑based selection of the most frequent classes with a deterministic lookup that assigns each test image the exact multilabel tag of a training image chosen by the same hash value. This keeps the baseline deterministic, removes the overly restrictive “most common only” rule, and provides realistic label combinations, which should raise the mean F1‑Score toward the target. The rest of the pipeline (paths, CSV writing) stays unchanged.'
- What this solution (achieved 0.30949) has done: 'I lower the prevalence threshold to include more common classes, compute the average number of labels per training image and add that many top‑prevalence classes to every prediction, and filter the hash‑selected training label set to keep only the selected prevalent classes (fall‑back to the most common class if empty). This keeps the deterministic baseline while enriching predictions, which should raise the mean F1‑Score toward the target. I also ensure the output directory exists before writing the CSV.'
- What this solution (achieved 0.22145) has done: 'I raise the prevalence threshold to keep only the truly frequent disease classes, drop the unconditional addition of all top‑k frequent labels, and instead pad each prediction only up to the average number of labels per training image using the most common classes. This reduces noisy false positives while still providing enough labels, which should lift the mean F1 score toward the target.'
- What this solution (achieved 0.21672) has done: 'The changes lower the prevalence threshold to include more disease classes, remove the restrictive filtering step, and ensure each prediction contains exactly the average number of labels (padding with the most common classes when needed). This keeps the deterministic hash‑based lookup of a training image while providing richer, more realistic multilabel predictions, which should raise the mean F1‑Score toward the target.'
- What this solution (achieved 0.28656) has done: 'The update replaces the hash‑based lookup with a deterministic baseline that predicts the `avg_label_cnt` most frequent disease classes for every test image. This removes the noisy random training labels, reduces false positives, and better aligns predictions with the label distribution, which should raise the mean F1‑Score toward the target while keeping the overall workflow unchanged. The script still writes a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import hashlib

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"

train_df = pd.read_csv(train_csv_path)

one_hot = train_df["labels"].str.get_dummies(sep=" ")
class_prevalence = one_hot.mean()

prevalence_threshold = 0.01  # 1% instead of 5%
selected_classes = class_prevalence[
    class_prevalence >= prevalence_threshold
].index.tolist()
if not selected_classes:
    selected_classes = [class_prevalence.idxmax()]

avg_label_cnt = max(1, int(round(train_df["labels"].str.split().apply(len).mean())))

most_common_classes = class_prevalence.sort_values(ascending=False).index.tolist()


def predict_labels(_image_path: str) -> str:
    """
    Predict labels for a test image by returning the `avg_label_cnt`
    most frequent disease classes. This deterministic baseline avoids
    random hash‑based selection and therefore reduces noisy false positives,
    moving the mean F1‑Score toward the target.
    """
    top_k = avg_label_cnt
    preds = most_common_classes[:top_k]
    return " ".join(preds)




## === cell 1
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    test_images = sorted(
        [
            f
            for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
    )

    submission_rows = []
    for img_name in test_images:
        preds = predict_labels(os.path.join(test_dir, img_name))
        submission_rows.append([img_name, preds])

    submission_df = pd.DataFrame(submission_rows, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(
        f"Submission file written to {submission_path} with {len(submission_df)} rows."
    )
