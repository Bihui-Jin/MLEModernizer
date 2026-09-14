# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, re
import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception as e:
    Image = None
    print("Pillow import failed, proceeding without image‑based fallback:", e)


def get_data_path(*parts):
    path = os.path.abspath(os.path.join(*parts))
    if os.path.exists(path):
        return path
    path = os.path.abspath(os.path.join("..", *parts))
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Path not found: {'/'.join(parts)}")


try:
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
except Exception as e:
    tf = None
    print("TensorFlow import failed, proceeding without TF:", e)




## === cell 1
def decode_image(filename, label=None, image_size=(512, 512)):
    """Decode an image file to a normalized tensor. Returns None if TF is unavailable."""
    if tf is None:
        return None if label is None else (None, label)
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    return image if label is None else (image, label)




## === cell 2
BATCH_SIZE = 32

source = get_data_path("input", "plant-pathology-2021-fgvc8", "test_images")
IMAGE_PATHS = [
    os.path.join(source, f)
    for f in os.listdir(source)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f)
]

if tf is not None:
    AUTO = tf.data.experimental.AUTOTUNE
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
    )
else:
    test_dataset = None




## === cell 3
import itertools
import random

train_path = get_data_path("input", "plant-pathology-2021-fgvc8", "train.csv")
train_df = pd.read_csv(train_path)


def sample_f1(true_str, pred_str):
    true_set = set(true_str.split())
    pred_set = set(pred_str.split())
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    inter = len(true_set & pred_set)
    return 2 * inter / (len(true_set) + len(pred_set))


def mean_f1(df, pred_label):
    return np.mean([sample_f1(t, pred_label) for t in df["labels"]])


random_state = 42
val_frac = 0.2
val_df = train_df.sample(frac=val_frac, random_state=random_state)


candidate_full = train_df["labels"].value_counts().idxmax()

all_labels = train_df["labels"].str.split().explode()
top1 = all_labels.value_counts().nlargest(1).index.tolist()
candidate_top1 = " ".join(top1)

top2 = all_labels.value_counts().nlargest(2).index.tolist()
candidate_top2 = " ".join(top2)

top3 = all_labels.value_counts().nlargest(3).index.tolist()
candidate_top3 = " ".join(top3)

topK_fixed_n = 5
topK = all_labels.value_counts().nlargest(topK_fixed_n).index.tolist()
candidate_topK = " ".join(topK)

top7 = all_labels.value_counts().nlargest(7).index.tolist()
candidate_top7 = " ".join(top7)

top10 = all_labels.value_counts().nlargest(10).index.tolist()
candidate_top10 = " ".join(top10)

top12 = all_labels.value_counts().nlargest(12).index.tolist()
candidate_top12 = " ".join(top12)

top15 = all_labels.value_counts().nlargest(15).index.tolist()
best_subset = None
best_subset_score = -1.0
for r in range(1, len(top15) + 1):
    for combo in itertools.combinations(top15, r):
        cand = " ".join(combo)
        score = mean_f1(val_df, cand)  # evaluate on validation split
        if score > best_subset_score:
            best_subset_score = score
            best_subset = cand

freq_thresh = 0.03
min_count = max(1, int(freq_thresh * len(train_df)))
freq_labels = all_labels.value_counts()
freq_candidates = freq_labels[freq_labels >= min_count].index.tolist()
candidate_freq = " ".join(freq_candidates)

median_label_cnt = int(train_df["labels"].str.split().apply(len).median())
median_top = all_labels.value_counts().nlargest(median_label_cnt).index.tolist()
candidate_median = " ".join(median_top)

candidates = {
    "full": candidate_full,
    "top1": candidate_top1,
    "top2": candidate_top2,
    "top3": candidate_top3,
    "topK": candidate_topK,
    "top7": candidate_top7,
    "top10": candidate_top10,
    "top12": candidate_top12,
    "combo_best": best_subset,
    "freq": candidate_freq,
    "median": candidate_median,
}

best_name = None
best_score = -1.0
for name, pred in candidates.items():
    if pred is None:
        continue
    score = mean_f1(val_df, pred)
    print(f"Candidate '{name}': mean F1 on validation = {score:.5f}")
    if score > best_score:
        best_score = score
        best_name = name

baseline_label = candidates.get(best_name, candidate_full)
print(
    f'Selected baseline ({best_name}) -> "{baseline_label}" with validation F1 {best_score:.5f}'
)




## === cell 4
sample_sub_path = get_data_path(
    "input", "plant-pathology-2021-fgvc8", "sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
test_images_ordered = sample_sub["image"].tolist()

train_label_lookup = dict(zip(train_df["image"], train_df["labels"]))

if Image is not None:
    train_img_dir = get_data_path("input", "plant-pathology-2021-fgvc8", "train_images")
    train_colors = []
    train_labels = []
    train_names = []

    for img_name, lbl in zip(train_df["image"], train_df["labels"]):
        img_path = os.path.join(train_img_dir, img_name)
        if os.path.exists(img_path):
            try:
                im = Image.open(img_path).convert("RGB")
                im = im.resize((64, 64))  # downscale for speed
                arr = np.array(im)
                avg = arr.mean(axis=(0, 1))  # shape (3,)
                train_colors.append(avg)
                train_labels.append(lbl)
                train_names.append(img_name)
            except Exception:
                continue

    train_colors = np.array(train_colors)  # (N, 3)

    def nearest_label(test_img_path):
        try:
            im = Image.open(test_img_path).convert("RGB")
            im = im.resize((64, 64))
            arr = np.array(im)
            avg = arr.mean(axis=(0, 1))
            dists = np.linalg.norm(train_colors - avg, axis=1)
            idx = np.argmin(dists)
            return train_labels[idx]
        except Exception:
            return None

else:
    nearest_label = lambda x: None  # dummy

pred_strings = []
for img_name in test_images_ordered:
    label = train_label_lookup.get(img_name)
    if label is None:
        test_path = os.path.join(source, img_name)
        label = nearest_label(test_path)
    if label is None:
        label = baseline_label
    pred_strings.append(label)

df = pd.DataFrame({"image": test_images_ordered, "labels": pred_strings})
submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(df.head())
