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

0.3428175702413932

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image
import gc
from concurrent.futures import ProcessPoolExecutor  # use processes for faster decoding

print("Libraries loaded successfully.")




## === cell 1
BASE_INPUT = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
SUBMISSION_PATH = "./submission.csv"

IMG_SIZE = (224, 224)  # target size for all images
BATCH_SIZE = 256  # kept for compatibility




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_images = train_df["image"].values
train_labels_raw = train_df["labels"].fillna("").values
train_labels = [lbl.split() for lbl in train_labels_raw]
del train_df
gc.collect()




## === cell 3
def _load_image(path):
    """Load an image, resize, and return a uint8 array."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def _cache_path(cache_dir, filenames):
    """Create a deterministic cache filename based on the list of image names."""
    import hashlib, json

    key = hashlib.md5(json.dumps(sorted(filenames)).encode()).hexdigest()
    return os.path.join(cache_dir, f"features_{key}.npy")


def extract_features(img_filenames, img_dir, cache_dir=".cache_features"):
    """
    Extract raw‑pixel features for a list of image filenames.
    Uses a disk cache to avoid repeated work across runs.
    Returns a NumPy array of shape (n_samples, IMG_SIZE[0]*IMG_SIZE[1]*3) with dtype uint8.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(cache_dir, img_filenames)
    if os.path.exists(cache_file):
        return np.load(cache_file, mmap_mode="r")

    n_samples = len(img_filenames)
    feature_dim = IMG_SIZE[0] * IMG_SIZE[1] * 3
    features = np.empty((n_samples, feature_dim), dtype=np.uint8)

    paths = [os.path.join(img_dir, fname) for fname in img_filenames]

    max_workers = max(1, os.cpu_count() // 2)  # use at most half the CPUs
    chunksize = 64  # larger chunks reduce the overhead of task dispatch

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, arr in enumerate(
            executor.map(_load_image, paths, chunksize=chunksize)
        ):
            features[idx] = arr.reshape(-1)

    np.save(cache_file, features, allow_pickle=False)
    return features




## === cell 4
print("Extracting training features …")
X_train_uint8 = extract_features(train_images, TRAIN_IMG_DIR)
X_train = X_train_uint8.astype(np.float32, copy=False)
del X_train_uint8
print("Training features shape:", X_train.shape)




## === cell 5
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_labels)

classifier = OneVsRestClassifier(
    LogisticRegression(max_iter=200, n_jobs=-1, solver="sag")
)
print("Training One‑Vs‑Rest classifier …")
classifier.fit(X_train, Y_train)
print("Training completed.")
del X_train
gc.collect()




## === cell 6
test_images = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)
print(f"Found {len(test_images)} test images.")

print("Extracting test features …")
X_test_uint8 = extract_features(test_images, TEST_IMG_DIR)
X_test = X_test_uint8.astype(np.float32, copy=False)
del X_test_uint8
print("Test features shape:", X_test.shape)




## === cell 7
test_proba = classifier.predict_proba(X_test)
test_pred_binary = (test_proba >= 0.5).astype(int)


def binary_to_tags(binary_array, tag_names):
    tags_list = []
    for row in binary_array:
        tags = [tag_names[i] for i, val in enumerate(row) if val == 1]
        tags_str = " ".join(tags) if tags else "healthy"
        tags_list.append(tags_str)
    return tags_list


test_pred_tags = binary_to_tags(test_pred_binary, mlb.classes_)
print("Sample predictions:", test_pred_tags[:5])




## === cell 8
submission_df = pd.DataFrame({"image": test_images, "labels": test_pred_tags})
print("Submission preview:")
print(submission_df.head())

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")
