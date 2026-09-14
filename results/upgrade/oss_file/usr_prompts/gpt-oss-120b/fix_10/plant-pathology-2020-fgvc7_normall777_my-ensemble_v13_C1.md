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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.isdir(BASE_PATH):
    BASE_PATH = os.path.join(os.getcwd(), "plant-pathology-2020-fgvc7")
print("Using BASE_PATH:", BASE_PATH)

IMG_SIZE = 224  # size for the TensorFlow branch
IMG_SIZE_SMALL = 224  # size for the sklearn fallback
BATCH_SIZE = 32
EPOCHS = 30  # increased epochs for better learning

train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sample_submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))


def format_path(image_id):
    return os.path.join(BASE_PATH, "images", f"{image_id}.jpg")


train_paths = train_df["image_id"].apply(format_path).values
test_paths = test_df["image_id"].apply(format_path).values
train_labels = train_df.iloc[:, 1:].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020
)




## === cell 1
if TF_AVAILABLE:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def decode_image(filename, label=None):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        if label is None:
            return img
        return img, label

    def data_augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return image, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTOTUNE)
        .cache()  # cache decoded images in memory
        .map(data_augment, num_parallel_calls=AUTOTUNE)
        .shuffle(1024)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    valid_ds = (
        tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
        .map(decode_image, num_parallel_calls=AUTOTUNE)
        .cache()  # cache decoded validation images
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(decode_image, num_parallel_calls=AUTOTUNE)
        .cache()  # cache decoded test images
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
else:
    from PIL import Image
    from sklearn.neural_network import MLPClassifier
    from sklearn.multiclass import OneVsRestClassifier

    def load_images(paths, size):
        """Load and resize images to a flat float32 array."""
        arr = np.empty((len(paths), size * size * 3), dtype=np.float32)
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            img = img.resize((size, size))
            arr[i] = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        return arr

    X_train = load_images(train_paths, IMG_SIZE_SMALL)
    X_valid = load_images(valid_paths, IMG_SIZE_SMALL)

    base_clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        max_iter=800,
        batch_size=32,
        learning_rate_init=0.001,
        random_state=2020,
        verbose=False,
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(X_train, train_labels)




## === cell 2
if TF_AVAILABLE:

    def build_model(num_classes):
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation="sigmoid")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model

    num_classes = train_labels.shape[1]
    model = build_model(num_classes)
    model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=2)

    preds = model.predict(test_ds, verbose=0)
    assert preds.shape == (test_df.shape[0], num_classes)
else:
    X_test = load_images(test_paths, IMG_SIZE_SMALL)
    preds = clf.predict_proba(X_test)
    assert preds.shape == (test_df.shape[0], train_labels.shape[1])

submission = sample_submission.copy()
submission.iloc[:, 1:] = preds
submission.to_csv("submission.csv", index=False)

print("Submission saved to submission.csv")
print(submission.head())
