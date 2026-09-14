# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 5. Code solution

## === cell 0
import os, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory as mf

    if not hasattr(mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return descriptor._concrete_class

        mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # protobuf optional; TensorFlow will raise later if needed

import pandas as pd, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

candidates = [
    os.path.abspath(
        os.path.join(os.getcwd(), "..", "input", "aerial-cactus-identification")
    ),
    "/kaggle/input/aerial-cactus-identification",
    "/working/aerial-cactus-identification",
    "/kaggle/working/aerial-cactus-identification",
    os.path.abspath(os.path.join(os.getcwd(), "aerial-cactus-identification")),
]

BASE_PATH = None
for p in candidates:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.csv")):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError("Could not locate the competition data directory.")

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

meta_data = pd.read_csv(TRAIN_CSV)
meta_data["has_cactus"] = meta_data["has_cactus"].astype(np.float32)

meta_data["filepath"] = meta_data["id"].apply(
    lambda img_id: os.path.join(TRAIN_DIR, img_id)
)




## === cell 1
train_df, val_df = train_test_split(
    meta_data,
    test_size=0.1,
    random_state=42,
    stratify=meta_data["has_cactus"],
)


def make_dataset(df, shuffle=True, augment=False):
    paths = df["filepath"].values
    labels = df["has_cactus"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_image(path, label):
        def _py_load(p):
            p = p.decode("utf-8")
            if not tf.io.gfile.exists(p):
                img = np.zeros((32, 32, 3), dtype=np.float32)
            else:
                img_bytes = tf.io.read_file(p)
                img = tf.image.decode_image(
                    img_bytes, channels=3, expand_animations=False
                ).numpy()
                img = tf.image.resize(img, [32, 32]).numpy()
                img = img.astype(np.float32) / 255.0
            return img

        img = tf.numpy_function(_py_load, [path], tf.float32)
        img.set_shape([32, 32, 3])
        return img, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:

        def _augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            return img, label

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_dataset(train_df, shuffle=True, augment=True)
val_dataset = make_dataset(val_df, shuffle=False, augment=False)




## === cell 2
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models, optimizers, metrics

base_model = VGG19(input_shape=(32, 32, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # initial frozen training

x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy", metrics.AUC(name="auc")],
)

model.summary()

labels = train_df["has_cactus"].values
neg = np.sum(labels == 0)
pos = np.sum(labels == 1)
total = len(labels)
weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1.0
weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1.0
class_weight = {0: weight_for_0, 1: weight_for_1}

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weight,
    verbose=1,
)

base_model.trainable = True
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.Adam(learning_rate=5e-6),
    metrics=["accuracy", metrics.AUC(name="auc")],
)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", factor=0.5, patience=3, verbose=1, min_lr=1e-8
)

fine_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,  # extended fine‑tuning
    class_weight=class_weight,
    callbacks=[lr_reduce],
    verbose=1,
)

if "history" in locals():
    for key in fine_history.history:
        history.history.setdefault(key, []).extend(fine_history.history[key])

final_auc = history.history["val_auc"][-1]
print(f"\nFinal validation AUC: {final_auc:.5f}")




## === cell 3
if os.path.isdir(TEST_DIR):
    test_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))]
    )
    test_paths = [os.path.join(TEST_DIR, f) for f in test_files]

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    def _load_test_image(path):
        def _py_load(p):
            p = p.decode("utf-8")
            if not tf.io.gfile.exists(p):
                img = np.zeros((32, 32, 3), dtype=np.float32)
            else:
                img_bytes = tf.io.read_file(p)
                img = tf.image.decode_image(
                    img_bytes, channels=3, expand_animations=False
                ).numpy()
                img = tf.image.resize(img, [32, 32]).numpy()
                img = img.astype(np.float32) / 255.0
            return img

        img = tf.numpy_function(_py_load, [path], tf.float32)
        img.set_shape([32, 32, 3])
        return img

    test_ds = test_ds.map(_load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    preds = model.predict(test_ds).flatten()
else:
    sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    test_files = sample_sub["id"].tolist()
    preds = np.full(len(test_files), 0.5, dtype=np.float32)  # fallback baseline




## === cell 4
submission = pd.DataFrame({"id": test_files, "has_cactus": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 5
if "history" in globals():
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
