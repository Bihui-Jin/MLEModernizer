# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5740535549399802

# 6. Current score

0.43339

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.38173) has done: 'I remove the failing TensorFlow import and model loading, and replace them with a simple baseline that predicts the most common disease labels from the training set for every test image. This fixes the protobuf error, ensures a valid submission.csv is written, and should give a reasonable F1 score by leveraging label frequencies without altering the core competition logic.'
- What this solution (achieved 0.28656) has done: 'The baseline was predicting three of the most frequent labels for every image, which lowers precision and hurts the mean F 1‑Score. By predicting only the single most common label (the top‑1 label) we increase precision while still keeping reasonable recall for the dominant class, moving the expected score upward toward the target. The change is limited to how the baseline string is built and does not alter any core modeling logic.'
- What this solution (achieved 0.28656) has done: 'I compute the average number of labels per training image and use that count (rounded to at least 1) to select the same number of most‑frequent labels for every test image. Predicting a few top labels rather than a single label should boost recall while keeping precision reasonable, moving the mean F1‑Score closer to the target.'
- What this solution (achieved 0.28656) has done: 'I tighten the baseline to predict only the single most‑frequent label for every test image (top_k = 1). This raises precision and is expected to improve the mean F1‑Score, moving the current 0.286 score upward toward the target 0.574 while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.38173) has done: 'I increase the number of most‑frequent labels predicted for every test image from 1 to 3. Predicting a few top labels improves recall while keeping precision reasonable, which should raise the mean F1‑Score toward the target without changing any core modeling logic. The rest of the pipeline (reading data, building the submission file) stays unchanged.'
- What this solution (achieved 0.35916) has done: 'I lower the number of most‑frequent labels predicted for every test image from three to two. Using fewer common labels improves precision while still keeping reasonable recall, which should raise the mean F1‑Score toward the target (higher is better). The rest of the pipeline stays unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.38173) has done: 'We replace the fixed `top_k = 2` baseline with a frequency‑threshold based selection: compute each label’s relative frequency, keep the most common labels whose cumulative share reaches ≈ 80 % of all occurrences (ensuring at least one label). This adds a few more relevant frequent classes, improving recall while keeping precision reasonable, which should move the mean F1 score closer to the target without altering any core pipeline logic.'
- What this solution (achieved 0.35308) has done: 'I slightly raise the cumulative‑frequency threshold used to choose the frequent labels (from 80 % to 90 %) and cap the number of selected labels to five. This adds a few more common classes, improving recall while limiting excess labels, which should raise the mean F1‑Score toward the target without altering the overall baseline approach.'
- What this solution (achieved 0.3327) has done: 'I add a small evaluation loop that tests k = 1…5 most‑frequent labels on the training set (using a macro F1 computed with sklearn’s MultiLabelBinarizer) and automatically pick the k that gives the highest macro F1.  The chosen top‑k labels are then used as the constant prediction for every test image, keeping the overall pipeline unchanged while improving precision/recall balance and moving the score toward the target.'
- What this solution (achieved 0.11339) has done: 'I replace the constant‑frequency baseline with a lightweight transfer‑learning model (EfficientNet‑B0) that learns multi‑label predictions from the training images.  The script now creates a train/validation split, builds a tf.data pipeline that loads and resizes images, trains a sigmoid‑output network for a few epochs, and uses a 0.5 probability threshold to generate space‑separated label strings for every test image.  This modest model addition is allowed because the current score is far below the target, and it keeps the overall pipeline structure while aiming to raise the macro F1 toward the desired value.'
- What this solution (achieved 0.43339) has done: 'The changes switch the image‑feature extraction to TensorFlow’s fast JPEG decoder (returning the same 6‑dim mean & std vector) and use a process‑based pool for parallelism, which speeds up the I/O‑bound decoding without altering the feature definition or model training logic.  All other steps—including data splitting, model fitting, and submission creation—remain unchanged, preserving exact semantics and results.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_imgs_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
test_imgs_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
output_dir = "./"

train_df = pd.read_csv(train_csv_path)
train_df["labels_list"] = train_df["labels"].str.split()

mlb = MultiLabelBinarizer()
y_all = mlb.fit_transform(train_df["labels_list"])
class_names = mlb.classes_
num_classes = len(class_names)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df["labels_list"].apply(lambda x: x[0] if x else "none"),
)

train_files = train_df.iloc[train_idx]["image"].values
val_files = train_df.iloc[val_idx]["image"].values
y_train = y_all[train_idx]
y_val = y_all[val_idx]


def extract_features_tf(img_path):
    """Fast 6‑dim feature (mean & std of RGB) using TensorFlow’s JPEG decoder."""
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # shape H, W, 3
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0,1]
    mean = tf.reduce_mean(img, axis=[0, 1])  # per‑channel mean
    if hasattr(tf.math, "reduce_std"):
        std = tf.math.reduce_std(img, axis=[0, 1])
    else:
        var = tf.math.reduce_variance(img, axis=[0, 1])
        std = tf.sqrt(var)
    feats = tf.concat([mean, std], axis=0)  # shape (6,)
    return feats.numpy()


def build_feature_matrix(file_names, base_dir):
    """Parallel extraction of 6‑dim features using a process pool (CPU‑bound decoding)."""
    n = len(file_names)
    feats = np.empty((n, 6), dtype=np.float32)
    paths = [os.path.join(base_dir, fname) for fname in file_names]

    max_workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, feat in enumerate(executor.map(extract_features_tf, paths, chunksize=1)):
            feats[i] = feat
    return feats


X_train = build_feature_matrix(train_files, train_imgs_dir)
X_val = build_feature_matrix(val_files, train_imgs_dir)

base_clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=200)
model = OneVsRestClassifier(base_clf)

model.fit(X_train, y_train)

val_pred = (model.predict_proba(X_val) >= 0.5).astype(int)
val_f1 = f1_score(y_val, val_pred, average="macro")
print(f"Validation macro F1: {val_f1:.4f}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_files = sorted(
    [
        f
        for f in os.listdir(test_imgs_dir)
        if os.path.isfile(os.path.join(test_imgs_dir, f))
    ]
)

X_test = build_feature_matrix(test_files, test_imgs_dir)

test_pred_binary = (model.predict_proba(X_test) >= 0.5).astype(int)


def decode_prediction(row):
    idx = np.where(row == 1)[0]
    if len(idx) == 0:
        return class_names[0]
    return " ".join(class_names[idx])


pred_labels = [decode_prediction(row) for row in test_pred_binary]

submission_df = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path} with {len(submission_df)} rows.")
