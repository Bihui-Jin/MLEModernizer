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

0.7849676823638048

# 6. Current score

0.35176

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I remove the TensorFlow‑related code that triggers import errors and replace the model‑based prediction with a simple baseline that assigns the most common label from the training set to every test image. This guarantees a valid CSV submission with the required columns and avoids the earlier protobuf/TensorFlow crash while keeping the core data‑handling logic intact.'
- What this solution (achieved 0.35176) has done: 'The update speeds up image loading by replacing the NumPy‑based mean calculation with Pillow’s C‑level `ImageStat` (which computes the mean directly in compiled code) and switches the pool to `ProcessPoolExecutor` so that image decoding runs in parallel without the GIL bottleneck. Worker counts are set to the available CPU cores (capped at 16) to fully utilize the hardware while keeping deterministic ordering. No algorithmic logic, model, or data paths are changed, so the predictions remain identical.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_path = os.path.join(load_dir, "train.csv")
train_img_dir = os.path.join(load_dir, "train_images")
test_img_dir = os.path.join(load_dir, "test_images")

train_df = pd.read_csv(train_path)

train_df["primary_label"] = train_df.labels.apply(lambda x: str(x).split()[0])

le = LabelEncoder()
y = le.fit_transform(train_df["primary_label"])


def mean_rgb(image_path):
    """Return mean RGB (float32 in [0,1]) for an image; fallback to 0.5 if error."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            return np.array(stat.mean, dtype=np.float32) / 255.0
    except Exception:
        return np.full(3, 0.5, dtype=np.float32)


train_image_paths = [os.path.join(train_img_dir, fname) for fname in train_df["image"]]

max_workers = min(16, os.cpu_count() or 1)  # use available cores, capped for safety
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    train_features = list(executor.map(mean_rgb, train_image_paths))

X_train = np.stack(train_features)

model = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000,
    n_jobs=5,
    solver="lbfgs",
    random_state=42,
)
model.fit(X_train, y)




## === cell 1
test_images = sorted(
    [
        f
        for f in os.listdir(test_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)
test_df = pd.DataFrame({"image": test_images})

test_image_paths = [os.path.join(test_img_dir, fname) for fname in test_df["image"]]

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    test_features = list(executor.map(mean_rgb, test_image_paths))

X_test = np.stack(test_features)

test_pred_idx = model.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred_idx)

test_df["labels"] = test_pred_labels




## === cell 2
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(test_df.head())
