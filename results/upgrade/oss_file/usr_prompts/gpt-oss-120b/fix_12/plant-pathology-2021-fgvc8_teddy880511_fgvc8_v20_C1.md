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
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import gc

try:
    import cv2

    _cv2_available = True
except ImportError:
    from PIL import Image

    _cv2_available = False


def locate_base():
    """
    Return the first directory that contains a train.csv file.
    Checks common Kaggle folder layouts and also searches recursively
    as a fallback.
    """
    candidates = [
        "./data/plant-pathology-2021-fgvc8",
        "./input/plant-pathology-2021-fgvc8",
        "./working/plant-pathology-2021-fgvc8",
        "./data/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "./input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "./working/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "/kaggle/working/plant-pathology-2021-fgvc8",
    ]
    for p in candidates:
        csv_path = os.path.join(p, "train.csv")
        if os.path.isdir(p) and os.path.isfile(csv_path):
            return p
    for root, dirs, files in os.walk("."):
        if (
            "train.csv" in files
            and os.path.basename(root) == "plant-pathology-2021-fgvc8"
        ):
            return root
    raise FileNotFoundError(
        "Base data directory not found among candidates with a train.csv file."
    )


BASE_PATH = locate_base()
train_imgpath = os.path.join(BASE_PATH, "train_images")
train_csvpath = os.path.join(BASE_PATH, "train.csv")


def load_image_normalized(path, dtype=np.float32):
    """Read an image, resize to 240x160, convert to the given float dtype and scale to [0,1]."""
    if _cv2_available:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Unable to read image {path}")
        if img.shape != (160, 240, 3):
            img = cv2.resize(img, (240, 160))
        return img.astype(dtype) / 255.0
    else:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (240, 160):
                im = im.resize((240, 160))
            arr = np.array(im, dtype=dtype) / 255.0
            return arr


def load_images_parallel_flat(img_dir, files, dtype=np.float32):
    """
    Load images in the same order as *files* using a thread pool,
    writing directly into a pre‑allocated flat array (n_samples, features).
    """
    n = len(files)
    height, width, channels = 160, 240, 3
    feature_len = height * width * channels
    out = np.empty((n, feature_len), dtype=dtype)

    max_workers = min(os.cpu_count() or 4, 32)

    def _loader(idx_fname):
        idx, fname = idx_fname
        img = load_image_normalized(os.path.join(img_dir, fname), dtype=dtype)
        out[idx] = img.ravel()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(_loader, enumerate(files))
    return out


y_train_df = pd.read_csv(train_csvpath)
train_files = y_train_df["image"].tolist()
X_flat = load_images_parallel_flat(train_imgpath, train_files, dtype=np.float16)




## === cell 1
label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "rust",  # <-- added
    "scab frog_eye_leaf_spot",
    "scab frog_eye_leaf_spot complex",
    "frog_eye_leaf_spot complex",
    "rust frog_eye_leaf_spot",
    "rust complex",
    "powdery_mildew complex",
]


def primary_label(label_str):
    """Extract the first token of the space‑delimited label string."""
    return label_str.split()[0]


y_train_df["primary"] = y_train_df["labels"].apply(primary_label)
label_to_idx = {label: idx for idx, label in enumerate(label_class)}
y_train_df["label_num"] = y_train_df["primary"].map(label_to_idx).fillna(-1).astype(int)

if (y_train_df["label_num"] == -1).any():
    unmapped = y_train_df[y_train_df["label_num"] == -1]["primary"].unique()
    raise ValueError(f"Found unmapped labels: {unmapped}")

y_train = y_train_df["label_num"].values  # integer class labels

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=200,
    n_jobs=-1,  # use all cores for faster fitting
    multi_class="multinomial",
    solver="saga",
    random_state=42,
)

model.fit(X_flat, y_train)

del X_flat, y_train, y_train_df
gc.collect()




## === cell 2
test_imgpath = os.path.join(BASE_PATH, "test_images")
test_files = sorted(
    [
        f
        for f in os.listdir(test_imgpath)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
)

X_test_flat = load_images_parallel_flat(test_imgpath, test_files, dtype=np.float32)

pred_idxs = model.predict(X_test_flat).tolist()

sub = pd.DataFrame(
    {
        "image": test_files,
        "labels": [label_class[idx] for idx in pred_idxs],
    }
)

sub.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
