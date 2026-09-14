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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from concurrent.futures import ThreadPoolExecutor

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def locate_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


base_input = locate_path(
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8",
    "./input/plant-pathology-2021-fgvc8",
)

train_csv_path = os.path.join(base_input, "train.csv")
train_images_dir = os.path.join(base_input, "train_images")
test_dir = os.path.join(base_input, "test_images")

output_dir = "./"
submission_path = os.path.join(output_dir, "submission.csv")

EfficientNet_image_size = (224, 224)  # model default
efficient_model = EfficientNetB0(
    include_top=False,
    pooling="avg",
    input_shape=EfficientNet_image_size + (3,),
)


def extract_features(image_path):
    """Return a rich vector: EfficientNet‑B0 embedding + mean/std of RGB & HSV."""
    try:
        img_raw = tf.io.read_file(image_path)
        img = tf.image.decode_image(img_raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img_resized = tf.image.resize(img, EfficientNet_image_size)

        img_pre = preprocess_input(
            img_resized * 255.0
        )  # preprocess_input expects 0‑255
        embedding = efficient_model(
            img_pre[tf.newaxis, ...], training=False
        )  # (1,1280)
        embed_np = embedding.numpy().squeeze()  # (1280,)

        mean_rgb = tf.reduce_mean(img_resized, axis=[0, 1]).numpy()
        std_rgb = tf.math.reduce_std(img_resized, axis=[0, 1]).numpy()
        hsv = tf.image.rgb_to_hsv(img_resized)
        mean_hsv = tf.reduce_mean(hsv, axis=[0, 1]).numpy()
        std_hsv = tf.math.reduce_std(hsv, axis=[0, 1]).numpy()

        return np.concatenate([embed_np, mean_rgb, std_rgb, mean_hsv, std_hsv])
    except Exception:
        return np.zeros(1280 + 12, dtype=np.float32)


train_df = pd.read_csv(train_csv_path)
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split(" "))

mlb = MultiLabelBinarizer()
mlb.fit(train_df["label_list"])
label_names = mlb.classes_.tolist()

train_image_paths = [
    os.path.join(train_images_dir, img_name) for img_name in train_df["image"]
]
with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
    train_features = list(executor.map(extract_features, train_image_paths))

X = np.vstack(train_features)
y = mlb.transform(train_df["label_list"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_clf = LogisticRegression(
    penalty="l2",
    C=2.0,  # slightly stronger regularisation
    solver="lbfgs",
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
)
clf = OneVsRestClassifier(base_clf, n_jobs=-1)
clf.fit(X_train, y_train)

threshold_candidates = np.arange(0.10, 0.91, 0.01)
best_thresh = 0.5
best_f1 = 0.0
val_probs = clf.predict_proba(X_val)
for t in threshold_candidates:
    preds_binary = (val_probs >= t).astype(int)
    f1 = f1_score(y_val, preds_binary, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
print(f"Best threshold on validation: {best_thresh:.2f} with macro‑F1: {best_f1:.4f}")

label_counts = y.sum(axis=0)
most_common_label = label_names[np.argmax(label_counts)]



## === cell 1
test_image_names = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
)
test_image_paths = [os.path.join(test_dir, name) for name in test_image_names]

with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
    test_features = list(executor.map(extract_features, test_image_paths))

predictions = []
for img_name, feats in zip(test_image_names, test_features):
    probs = clf.predict_proba(feats.reshape(1, -1))[0]  # (n_labels,)

    selected_idx = np.where(probs > best_thresh)[0]
    if selected_idx.size == 0:
        selected_labels = [most_common_label]
    else:
        selected_labels = [label_names[i] for i in selected_idx]

    predictions.append([img_name, " ".join(selected_labels).strip()])

submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
submission_df.to_csv(submission_path, index=False)

print(f"Submission file written to: {submission_path}")
