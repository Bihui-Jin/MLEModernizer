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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

possible_paths = glob.glob("/kaggle/**/train.csv", recursive=True)
if not possible_paths:
    raise FileNotFoundError("train.csv not found in the environment.")
DATA_ROOT = os.path.dirname(possible_paths[0])

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
images_dir = os.path.join(DATA_ROOT, "images")

train_df = pd.read_csv(train_path)
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
y_train = train_df[target_cols].values.astype(np.float32)

IMG_SIZE = (128, 128)  # higher resolution for more detail
CHANNELS = 3
HIST_BINS = 8


def load_image_array(img_id, size=IMG_SIZE):
    """Load an image, resize, normalise to [0,1]; missing files become zeros."""
    img_file = os.path.join(images_dir, f"{img_id}.jpg")
    if not os.path.isfile(img_file):
        return np.zeros((size[0], size[1], CHANNELS), dtype=np.float32)
    with Image.open(img_file) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        return arr / 255.0


def channel_histograms(arr, bins=HIST_BINS):
    """Concatenated normalised histograms for each colour channel."""
    hist_features = []
    for ch in range(CHANNELS):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 1), density=True)
        hist_features.append(hist.astype(np.float32))
    return np.concatenate(hist_features)


def texture_features(arr):
    """Simple texture: mean and std of gradient magnitude on a grayscale version."""
    gray = arr.mean(axis=2)  # approximate grayscale
    gx = np.diff(gray, axis=1, prepend=0)
    gy = np.diff(gray, axis=0, prepend=0)
    grad = np.sqrt(gx**2 + gy**2)
    return np.array([grad.mean(), grad.std()], dtype=np.float32)


def flatten_with_stats(arr):
    """Flatten image and append channel stats, histograms and texture features."""
    flat = arr.ravel()
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    medians = np.median(arr, axis=(0, 1))
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))
    stats = np.concatenate([means, stds, medians, mins, maxs]).astype(np.float32)
    hist = channel_histograms(arr)
    tex = texture_features(arr)
    return np.concatenate([flat, stats, hist, tex])


def horizontal_flip(arr):
    return np.fliplr(arr)


def vertical_flip(arr):
    return np.flipud(arr)


def rotate90(arr):
    """Rotate 90° clockwise."""
    return np.rot90(arr, k=-1)


def rotate180(arr):
    """Rotate 180°."""
    return np.rot90(arr, k=2)


base_imgs = np.stack([load_image_array(img_id) for img_id in train_df["image_id"]])

X_train_orig = np.stack([flatten_with_stats(img) for img in base_imgs])
X_train_hflip = np.stack(
    [flatten_with_stats(horizontal_flip(img)) for img in base_imgs]
)
X_train_vflip = np.stack([flatten_with_stats(vertical_flip(img)) for img in base_imgs])
X_train_rot90 = np.stack([flatten_with_stats(rotate90(img)) for img in base_imgs])
X_train_rot180 = np.stack([flatten_with_stats(rotate180(img)) for img in base_imgs])

X_train = np.vstack(
    [X_train_orig, X_train_hflip, X_train_vflip, X_train_rot90, X_train_rot180]
)
y_train_aug = np.vstack([y_train] * 5)  # repeat labels for each augmentation

clf = make_pipeline(
    StandardScaler(),
    PCA(
        n_components=0.999,  # retain a bit more variance
        random_state=42,
    ),
    OneVsRestClassifier(
        MLPClassifier(
            hidden_layer_sizes=(512, 128),
            activation="relu",
            solver="adam",
            alpha=1e-5,  # weaker regularisation
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=500,  # allow more iterations
            early_stopping=False,
            random_state=42,
            verbose=False,
        ),
        n_jobs=-1,
    ),
)

clf.fit(X_train, y_train_aug)




## === cell 1
test_df = pd.read_csv(test_path)

test_imgs = np.stack([load_image_array(img_id) for img_id in test_df["image_id"]])

X_test_orig = np.stack([flatten_with_stats(img) for img in test_imgs])
X_test_hflip = np.stack([flatten_with_stats(horizontal_flip(img)) for img in test_imgs])
X_test_vflip = np.stack([flatten_with_stats(vertical_flip(img)) for img in test_imgs])
X_test_rot90 = np.stack([flatten_with_stats(rotate90(img)) for img in test_imgs])
X_test_rot180 = np.stack([flatten_with_stats(rotate180(img)) for img in test_imgs])

pred_orig = clf.predict_proba(X_test_orig)
pred_hflip = clf.predict_proba(X_test_hflip)
pred_vflip = clf.predict_proba(X_test_vflip)
pred_rot90 = clf.predict_proba(X_test_rot90)
pred_rot180 = clf.predict_proba(X_test_rot180)

y_pred = (pred_orig + pred_hflip + pred_vflip + pred_rot90 + pred_rot180) / 5.0

submission_df = pd.DataFrame()
submission_df["image_id"] = test_df["image_id"]
for idx, col in enumerate(target_cols):
    submission_df[col] = y_pred[:, idx]




## === cell 2
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
