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

0.8117451523545736

# 6. Current score

0.56755

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I set the protobuf implementation before importing TensorFlow to avoid the MessageFactory error, replace the faulty model loading with a simple placeholder that assigns the “healthy” label to every test image (ensuring a valid CSV submission), and clean up the inference loop accordingly. This fixes the runtime crashes while still producing a correctly‑formatted submission file.'
- What this solution (achieved 0.58833) has done: 'The fix adds a lightweight transfer‑learning pipeline using TensorFlow’s built‑in MobileNetV2 to generate image embeddings, creates class centroids from a small sampled subset of the training set, and assigns each test image the label of the nearest centroid. This replaces the “always‑healthy” placeholder with a data‑driven prediction, keeps the original file‑handling logic, and still writes a correctly formatted `submission.csv`. The changes also keep the protobuf environment setting and guard against GPU use.'
- What this solution (achieved 0.56755) has done: 'I keep the original pipeline but fix the protobuf import issue by setting the environment variable before importing TensorFlow (as already done) and make two small, score‑oriented tweaks: increase the number of sampled training images per class to improve centroid quality, and change the inference step to output the two most similar disease labels per image (space‑delimited) instead of a single label. These adjustments stay within the existing logic, avoid any major architectural changes, and should raise the F1 score toward the target while still writing a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_images_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

default_label = "healthy"
if default_label not in dataset_labels:
    default_label = dataset_labels[0]




## === cell 2
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_SAMPLES_PER_CLASS = 100  # increased from 30

feature_extractor = MobileNetV2(
    weights="imagenet", include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)
)


def load_and_preprocess(img_path):
    """Load an image file and resize / preprocess for MobileNetV2."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return arr


embeddings = []
labels = []

for cls in dataset_labels:
    mask = data_set["labels"].str.contains(rf"\b{cls}\b")
    cls_rows = data_set[mask]
    sampled = cls_rows.sample(
        n=min(len(cls_rows), MAX_SAMPLES_PER_CLASS), random_state=42, replace=False
    )
    for img_name in sampled["image"]:
        img_path = os.path.join(train_images_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        try:
            arr = load_and_preprocess(img_path)
        except Exception:
            continue
        embeddings.append(arr)
        labels.append(cls)

if len(embeddings) == 0:
    raise RuntimeError("No training embeddings could be loaded.")

embeddings = np.stack(embeddings, axis=0)
train_features = feature_extractor.predict(embeddings, batch_size=BATCH_SIZE, verbose=0)

centroids = {}
for cls in dataset_labels:
    cls_feat = train_features[np.array(labels) == cls]
    if cls_feat.size == 0:
        continue
    centroid = np.mean(cls_feat, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
    centroids[cls] = centroid_norm

centroid_matrix = np.stack(list(centroids.values()))
centroid_labels = list(centroids.keys())
centroid_matrix_norm = centroid_matrix / (
    np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10
)




## === cell 3
if __name__ == "__main__":
    test_images = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    predictions = []
    batch_imgs = []
    batch_names = []

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        try:
            arr = load_and_preprocess(img_path)
        except Exception:
            predictions.append([img_name, default_label])
            continue
        batch_imgs.append(arr)
        batch_names.append(img_name)

        if len(batch_imgs) == BATCH_SIZE:
            batch_arr = np.stack(batch_imgs)
            feats = feature_extractor.predict(
                batch_arr, batch_size=BATCH_SIZE, verbose=0
            )
            feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10)
            sim = np.dot(feats_norm, centroid_matrix_norm.T)  # (batch, n_classes)

            top_k = 2
            best_idxs = np.argpartition(-sim, range(top_k), axis=1)[:, :top_k]

            for name, idxs in zip(batch_names, best_idxs):
                pred_labels = " ".join([centroid_labels[i] for i in idxs])
                predictions.append([name, pred_labels])

            batch_imgs, batch_names = [], []

    if batch_imgs:
        batch_arr = np.stack(batch_imgs)
        feats = feature_extractor.predict(
            batch_arr, batch_size=len(batch_imgs), verbose=0
        )
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10)
        sim = np.dot(feats_norm, centroid_matrix_norm.T)
        top_k = 2
        best_idxs = np.argpartition(-sim, range(top_k), axis=1)[:, :top_k]

        for name, idxs in zip(batch_names, best_idxs):
            pred_labels = " ".join([centroid_labels[i] for i in idxs])
            predictions.append([name, pred_labels])

    submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
