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

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

mixed_precision.set_global_policy("mixed_float16")



## === cell 1
train_dir = "/kaggle/input/aerial-cactus-identification/train"
test_dir = "/kaggle/input/aerial-cactus-identification/test"



## === cell 2
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
train_df["has_cactus"] = train_df["has_cactus"].astype(int)
train_df.head()




## === cell 3
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


print(f"Train images found: {count_files(train_dir)}")
print(f"Test  images found: {count_files(test_dir)}")



## === cell 4
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 5
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = {int(i): w for i, w in enumerate(class_weights)}
print("Class weights:", class_weights_dict)




## === cell 6
def tf_augment(image):
    """Random 0‑3 90° rotations and optional flips, applied with TF ops."""
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_up_down(image)
    return image


def tf_scale(image):
    """Rescale uint8 image to float32 in [0,1]."""
    return tf.cast(image, tf.float32) / 255.0


def tf_decode_image(path):
    """Read a JPEG file and return a uint8 tensor of shape (32,32,3)."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # already 32x32
    return img




## === cell 7
train_filenames = train_df["id"].values.astype(str)
train_labels = train_df["has_cactus"].values

train_fnames, val_fnames, y_train, y_val = train_test_split(
    train_filenames,
    train_labels,
    test_size=0.10,
    stratify=train_labels,
    random_state=42,
)

batch_size = 256

train_paths = tf.strings.join([train_dir, train_fnames], separator="/")
val_paths = tf.strings.join([train_dir, val_fnames], separator="/")

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, y_train))
    .shuffle(buffer_size=len(train_fnames), seed=42, reshuffle_each_iteration=True)
    .map(
        lambda p, l: (tf_decode_image(p), l),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(lambda img, l: (tf_scale(img), l), num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .map(lambda img, l: (tf_augment(img), l), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((val_paths, y_val))
    .map(
        lambda p, l: (tf_scale(tf_decode_image(p)), l),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .cache()
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 8
test_filenames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_paths = tf.strings.join([test_dir, test_filenames], separator="/")

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(
        lambda p: tf_scale(tf_decode_image(p)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 9
efficient_net = EfficientNetB0(
    weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
)
efficient_net.trainable = False  # <<< added freeze

model = Sequential(
    [
        efficient_net,
        Dense(120, activation="relu"),
        Dense(120, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.summary()



## === cell 10
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 11
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    class_weight=class_weights_dict,
    verbose=1,
)



## === cell 12
preds = model.predict(test_dataset, verbose=1)



## === cell 13
image_ids = test_filenames  # filenames serve as IDs
predictions = preds.ravel()
submission = pd.DataFrame({"id": image_ids, "has_cactus": predictions})
submission.head()



## === cell 14
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print("Finished.")
