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

0.6406278855032318

# 6. Current score

0.18891

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'The fixes remove the faulty model loading and broken image handling, replace them with a lightweight baseline that selects the most common disease label from the training data and assigns it to every test image. This ensures the script runs end‑to‑end, correctly gathers the test filenames, builds the required DataFrame with the “image, labels” columns, and writes a proper `submission.csv` file.'
- What this solution (achieved 0.3327) has done: 'The plan is to replace the single‑label baseline with a small multi‑label baseline that chooses the k most frequent disease labels, where k is selected by evaluating macro‑F1 on the training set itself. This keeps the core logic (reading CSVs, building a submission DataFrame) unchanged while adding a lightweight, data‑driven step that should raise the score toward the target.'
- What this solution (achieved 0.34732) has done: 'I keep the existing data loading and frequency‑based label ordering, but instead of assigning the same set of top‑k labels to every test image I vary the number of labels per image deterministically using a hash of the filename. This adds modest diversity, which should raise the macro F1 toward the target while preserving the overall baseline logic. The change only touches the label‑construction step and adds an inexpensive `hashlib` import.'
- What this solution (achieved 0.18891) has done: 'We keep the overall baseline but replace the uniform random‑k assignment with a distribution‑matched one: we compute how many labels each training image actually has, turn that into a probability distribution over possible k, and use the image‑specific hash to sample k accordingly. We also rotate the start position in the ordered‑by‑frequency label list so each image gets a different subset of the top labels rather than always the same prefix. This modest change respects the existing logic while better mirroring the true label‑count pattern, which should raise the macro‑F1 toward the target.'
- What this solution (achieved 0.22392) has done: 'The update replaces the uniform top‑k label strategy with a frequency‑aware selection of whole label‑sets observed in the training data. By hashing each test filename we now sample a realistic multi‑label combination (proportional to its occurrence in the training set), preserving the deterministic, lightweight baseline while providing far more diverse and representative predictions, which should raise the macro F1 toward the target. The rest of the pipeline and file handling remain unchanged.'
- What this solution (achieved 0.29487) has done: 'I replace the combo‑sampling baseline with a lightweight multi‑label baseline that respects the observed distribution of how many labels each training image has. For each test image we hash its filename to a deterministic uniform value, map that to a sampled label‑count k using the training‑derived distribution, and then assign the top‑k most frequent disease labels. This keeps the overall structure unchanged while yielding more realistic multi‑label predictions and should move the macro‑F1 closer to the target.'
- What this solution (achieved 0.18891) has done: 'Implemented a deterministic offset for label selection per image, introducing diversity while preserving the original lightweight baseline. By rotating the start position in the ordered‑by‑frequency label list based on a hash of the filename, each test image now receives a varied subset of the top‑k frequent labels, which is expected to boost the macro‑F1 score and move it closer to the target.'
- What this solution (achieved 0.22392) has done: 'I keep the overall pipeline intact but replace the per‑image label generation with a deterministic sampling of whole label‑sets observed in the training data. By building a frequency‑based cumulative distribution of these exact multi‑label combinations and selecting one for each test filename using a hash‑derived random value, the predictions become far more representative of the true label distribution, which should raise the macro F1 toward the target while preserving the original lightweight baseline. The rest of the script (data loading, submission writing) remains unchanged.'
- What this solution (achieved 0.18891) has done: 'The change updates the label‑generation logic to sample a realistic number of labels per image (based on the training distribution) and rotate through the most frequent labels using a hash‑derived offset. This adds diversity while staying lightweight, which is expected to raise the macro‑F1 toward the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import pathlib
import pandas as pd
from collections import Counter
import hashlib

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

DATA_ROOT = pathlib.Path("/kaggle/input/plant-pathology-2021-fgvc8")

TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_IMAGES_DIR = DATA_ROOT / "test_images"



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

all_labels = train_df["labels"].dropna().str.split(" ").explode()
label_counts = Counter(all_labels)

label_list = [lbl for lbl, _ in label_counts.most_common()]

train_label_sets = train_df["labels"].dropna().str.split(" ").tolist()

mlb = MultiLabelBinarizer(classes=label_list)
y_true = mlb.fit_transform(train_label_sets)

best_k = 1
best_f1 = 0.0
max_k = min(5, len(label_list))
for k in range(1, max_k + 1):
    top_k_labels = label_list[:k]
    y_pred = mlb.transform([top_k_labels] * len(train_label_sets))
    f1 = f1_score(y_true, y_pred, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_k = k

default_labels = label_list[:best_k]
print(f"Selected top-{best_k} labels for baseline: {default_labels}")
print(f"Training macro‑F1 of this baseline: {best_f1:.5f}")

label_counts_per_image = [len(s) for s in train_label_sets]
count_counter = Counter(label_counts_per_image)
max_count = max(label_counts_per_image)

count_probs = [
    count_counter[i] / len(train_label_sets) for i in range(1, max_count + 1)
]

count_cum = []
cum = 0.0
for p in count_probs:
    cum += p
    count_cum.append(cum)

label_set_counter = Counter([" ".join(s) for s in train_label_sets])
total_sets = sum(label_set_counter.values())
label_set_list = list(label_set_counter.keys())
label_set_probs = [label_set_counter[ls] / total_sets for ls in label_set_list]

label_set_cum = []
cum = 0.0
for p in label_set_probs:
    cum += p
    label_set_cum.append(cum)



## === cell 2
test_image_paths = sorted(TEST_IMAGES_DIR.rglob("*.jpg"))
filenames = [p.name for p in test_image_paths]
print(f"Found {len(filenames)} test images.")




## === cell 3
def labels_for_image(fname: str) -> str:
    """
    Deterministically select a realistic number of top‑frequency labels for each image.
    The number of labels (k) follows the empirical distribution of label counts in the
    training set, and the starting position in the sorted‑by‑frequency label list is
    offset by a hash of the filename to introduce diversity while remaining reproducible.
    """
    h = int(hashlib.md5(fname.encode()).hexdigest(), 16)
    rnd = (h % 1_000_000) / 1_000_000

    k = 1
    for idx, threshold in enumerate(count_cum):
        if rnd <= threshold:
            k = idx + 1  # idx starts at 0, representing count = 1
            break

    start = h % len(label_list)

    selected = [label_list[(start + i) % len(label_list)] for i in range(k)]
    return " ".join(selected)


submission_df = pd.DataFrame(
    {"image": filenames, "labels": [labels_for_image(fname) for fname in filenames]}
)



## === cell 4
submission_path = pathlib.Path("submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")
