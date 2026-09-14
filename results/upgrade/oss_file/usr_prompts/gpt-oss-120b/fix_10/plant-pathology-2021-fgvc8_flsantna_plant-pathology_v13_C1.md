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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.6434533702677737

# 6. Current score

0.38173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'The changes remove the problematic protobuf environment setting that caused an import error and filter the test directory contents so only image files are processed, preventing TensorFlow from trying to read a sub‑directory as an image. This fixes the runtime failures and ensures a valid submission.csv is written.'
- What this solution (achieved 0.38173) has done: 'Implemented a lightweight, TensorFlow‑free pipeline that avoids the protobuf import error by dropping the TF dependency.  
Added NumPy‑based prior probabilities computed from the training labels and a simple `PriorModel` that returns these priors for every test image.  
Predictions now use a modest threshold (0.2) on class priors (with a fallback to the most common label), yielding richer multilabel outputs and improving the F1 score toward the target.  
All original steps (reading CSVs, listing test images, building the submission file) are retained, and the script now reliably writes a valid `submission.csv`.'
- What this solution (achieved 0.38173) has done: 'We add a quick validation split to tune how many top‑prior labels to output for each image (instead of using a fixed probability threshold). By testing k = 1 … 5 on a held‑out part of the training data and picking the k that gives the highest sample‑wise F1, we can raise the expected F1 score while keeping the same “prior‑only” model architecture. The rest of the pipeline (reading CSVs, listing test images, writing `submission.csv`) stays unchanged.'
- What this solution (achieved 0.38173) has done: 'Improved the validation step to search for the optimal probability threshold instead of a fixed top‑k count, keeping the prior‑only model unchanged. The code now evaluates several thresholds on the validation split, selects the one giving the highest sample‑wise F1, and uses this threshold to generate predictions for the test set, falling back to the most common label when no class exceeds the threshold. This small change is expected to raise the F1 score toward the target while preserving the original pipeline.'
- What this solution (achieved 0.38173) has done: 'I replace the threshold‑based selection with a tiny top‑k search on the validation split, keeping the prior‑only model unchanged. By evaluating k = 1…5 (or up to the number of classes) we pick the k that gives the highest sample‑wise F1 on the held‑out data, then use that k to output the k most probable labels for every test image. This small change is expected to raise the F1 score toward the target while preserving the core logic.'
- What this solution (achieved 0.38173) has done: 'The update adds a lightweight validation step that searches for the best global probability threshold (in addition to the existing top‑k search) and selects whichever rule gives the higher validation F1. The chosen rule (threshold or top‑k) is then used to build the submission, keeping the PriorModel unchanged while aiming to improve the score toward the target.'
- What this solution (achieved 0.38173) has done: 'Implemented a small but effective enhancement to the prior‑only solution:

* Added computation of the average number of labels per training image and included it as an additional candidate k when searching for the best top‑k rule.  
* Updated the validation loop to evaluate this extra k and select the method (global threshold, original top‑k, or the new average‑based k) that yields the highest sample‑wise F1 on the held‑out split.  
* Kept the original lightweight pipeline and model unchanged; only the post‑processing logic was refined, which safely moves the validation F1 closer to the target without altering the core architecture.'
- What this solution (achieved 0.38173) has done: 'The update adds a greedy validation‑based rule that iteratively adds the class whose inclusion most improves the sample‑averaged F1 on the held‑out split. This rule is compared with the previous top‑k and global‑threshold methods, and the best‑performing one is used for the final predictions. The core prior‑only model and data handling remain unchanged, while the new post‑processing step is expected to move the validation F1 (and thus the competition score) closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")

train_oh, val_oh = train_test_split(
    one_hot,
    test_size=0.2,
    random_state=42,
)

class_priors = train_oh.mean().values.astype(np.float32)  # (num_classes,)
dataset_labels = one_hot.columns.tolist()
num_classes = len(dataset_labels)

most_common_label = one_hot.sum().idxmax()
if most_common_label is None:
    most_common_label = "healthy"


def eval_f1(preds: np.ndarray) -> float:
    """Sample‑averaged F1 for a binary prediction matrix."""
    return f1_score(val_oh.values, preds, average="samples")


sorted_class_idxs = np.argsort(-class_priors)  # descending order
max_k = min(5, num_classes)

best_k = 1
best_f1_topk = 0.0
for k in range(1, max_k + 1):
    preds = np.zeros((val_oh.shape[0], num_classes), dtype=int)
    preds[:, sorted_class_idxs[:k]] = 1
    f1 = eval_f1(preds)
    if f1 > best_f1_topk:
        best_f1_topk = f1
        best_k = k

thresholds = np.linspace(0.0, class_priors.max(), 101)
best_thr = 0.0
best_f1_thr = 0.0
for thr in thresholds:
    mask = (class_priors >= thr).astype(int)
    preds = np.tile(mask, (val_oh.shape[0], 1))
    f1 = eval_f1(preds)
    if f1 > best_f1_thr:
        best_f1_thr = f1
        best_thr = thr

best_f1_greedy = 0.0
selected_idxs = []
preds_greedy = np.zeros((val_oh.shape[0], num_classes), dtype=int)
current_f1 = eval_f1(preds_greedy)

for idx in sorted_class_idxs:
    temp_preds = preds_greedy.copy()
    temp_preds[:, idx] = 1
    f1 = eval_f1(temp_preds)
    if f1 > current_f1 + 1e-6:
        preds_greedy = temp_preds
        current_f1 = f1
        selected_idxs.append(idx)

best_f1_greedy = current_f1

if best_f1_thr > best_f1_topk and best_f1_thr > best_f1_greedy:
    chosen_method = "threshold"
    chosen_value = best_thr
    print(
        f"Chosen global threshold {chosen_value:.4f} (validation F1≈{best_f1_thr:.4f})"
    )
elif best_f1_topk > best_f1_greedy:
    chosen_method = "topk"
    chosen_value = best_k
    print(f"Chosen top‑k {chosen_value} (validation F1≈{best_f1_topk:.4f})")
else:
    chosen_method = "greedy"
    chosen_value = selected_idxs  # list of class indexes
    print(
        f"Chosen greedy set of {len(chosen_value)} classes (validation F1≈{best_f1_greedy:.4f})"
    )




## === cell 1
if __name__ == "__main__":

    class PriorModel:
        def __call__(self, _):
            return np.expand_dims(class_priors, axis=0)

    model = PriorModel()

    images_path_list = sorted(
        [
            f
            for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    rows = []

    for name in images_path_list:
        if chosen_method == "topk":
            top_k_labels = [
                dataset_labels[idx] for idx in sorted_class_idxs[:chosen_value]
            ]
            labels_str = " ".join(top_k_labels) if top_k_labels else most_common_label
        elif chosen_method == "threshold":
            mask = class_priors >= chosen_value
            thresh_labels = [lbl for lbl, keep in zip(dataset_labels, mask) if keep]
            labels_str = " ".join(thresh_labels) if thresh_labels else most_common_label
        else:  # greedy method
            greedy_labels = [dataset_labels[idx] for idx in chosen_value]
            labels_str = " ".join(greedy_labels) if greedy_labels else most_common_label

        rows.append([name, labels_str])

    submission_df = pd.DataFrame(rows, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
