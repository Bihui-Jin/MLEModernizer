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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.9926

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'We remove the eager TensorFlow import that crashes with the current protobuf version, lazily import it inside the training function, and add a safe fallback for the missing test folder by reading the sample submission IDs and creating dummy image data. The cells are renumbered starting at 1, and the CNN is trained for a few more epochs to stay close to the target AUC while preserving the original model design.'
- What this solution (achieved 0.5) has done: 'The fix adds a protobuf compatibility flag `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'` before importing TensorFlow (preventing the “MessageFactory” error) and extends the default training epochs to give the CNN enough learning capacity to reach a validation AUC close to the target. No other logic is altered, so the original data handling, model architecture, and submission format stay the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

possible_paths = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/working/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
]
BASE_PATH = next((p for p in possible_paths if os.path.isdir(p)), None)
if BASE_PATH is None:
    raise FileNotFoundError("Base data directory not found among expected locations.")

TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")


def process_picture():
    """Read the CSV and build full image paths + labels."""
    data = pd.read_csv(TRAIN_CSV)
    image_files = []
    labels = []
    for img_id, label in zip(data["id"], data["has_cactus"]):
        img_path = os.path.join(TRAIN_DIR, img_id)
        image_files.append(img_path)
        labels.append(label)
    return image_files, np.array(labels, dtype=np.int32)


def _read_image(path):
    """Read an image, resize to 32×32, return as float32 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (32, 32))
    return img.astype(np.float32) / 255.0


def get_images_labels():
    """Load all training images as (N,32,32,3) tensors and labels."""
    image_files, labels = process_picture()
    images = np.stack([_read_image(f) for f in image_files])
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=7, stratify=labels
    )
    print("Train/val shapes (images):", X_train.shape, X_val.shape)
    return X_train, X_val, y_train, y_val




## === cell 1
X_train, X_val, y_train, y_val = get_images_labels()




## === cell 2
def train_cnn(epochs=20, batch_size=64):
    """Train a small CNN and report validation AUC."""
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    import tensorflow as tf  # lazy import to avoid protobuf incompatibility

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(32, 32, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    val_probs = model.predict(X_val).ravel()
    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation ROC‑AUC (sklearn): {auc:.6f}")
    return model, auc




## === cell 3
model, val_auc = train_cnn()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def get_test_images():
    """Load and preprocess test images; fallback to dummy data if test folder missing."""
    if os.path.isdir(TEST_DIR):
        ids = []
        imgs = []
        for fname in sorted(os.listdir(TEST_DIR)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ids.append(fname)
            fpath = os.path.join(TEST_DIR, fname)
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((32, 32, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (32, 32))
            imgs.append(img.astype(np.float32) / 255.0)
        imgs_arr = (
            np.stack(imgs) if imgs else np.empty((0, 32, 32, 3), dtype=np.float32)
        )
        print("Test images shape:", imgs_arr.shape)
        return imgs_arr, ids
    else:
        sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
        ids = pd.read_csv(sample_path)["id"].tolist()
        dummy_imgs = np.zeros((len(ids), 32, 32, 3), dtype=np.float32)
        print("Test directory missing – using dummy images for", len(ids), "ids.")
        return dummy_imgs, ids




## === cell 5
def predict_and_submit(model, out_path="submission.csv"):
    test_X, test_ids = get_test_images()
    probs = model.predict(test_X).ravel()
    sub_df = pd.DataFrame({"id": test_ids, "has_cactus": probs})
    sub_df.to_csv(out_path, index=False)
    print(f"Submission written to {out_path} ({sub_df.shape[0]} rows)")




## === cell 6
predict_and_submit(model)
