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

3.14

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import zipfile
import warnings

warnings.filterwarnings("ignore")



## === cell 1
train_dir = "/kaggle/working/aerial-cactus-identification/train"
test_dir = "/kaggle/working/aerial-cactus-identification/test"
if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/train.zip", "r"
    ) as z:
        z.extractall("/kaggle/working")
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/test.zip", "r"
    ) as z:
        z.extractall("/kaggle/working")



## === cell 2
base_path = "/kaggle/working/aerial-cactus-identification"
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
test_filenames = sorted(
    [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
)
test_df = pd.DataFrame({"id": test_filenames})



## === cell 3
img_list = [os.path.join(train_path, img_id) for img_id in train_df["id"]]
label_list = train_df["has_cactus"].astype(int).tolist()
df = pd.DataFrame({"image": img_list, "label": label_list})



## === cell 4
from sklearn.utils import class_weight

numeric_labels = train_df["has_cactus"].astype(int)
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(numeric_labels),
    y=numeric_labels,
)
weights_dict = dict(enumerate(class_weights))



## === cell 5
import tensorflow as tf

tf.random.set_seed(42)  # ensure reproducibility
tf.config.optimizer.set_jit(True)

cpu_cores = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)
tf.config.threading.set_inter_op_parallelism_threads(cpu_cores)

IMG_SIZE = (64, 64)
BATCH_SIZE = 256
NUM_WORKERS = cpu_cores
AUTOTUNE = tf.data.AUTOTUNE

from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)


def _tf_load_and_preprocess(path, label):
    """Read JPEG file, resize, and normalize to float16."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float16) / tf.constant(255.0, dtype=tf.float16)
    return image, tf.cast(label, tf.int32)


def _tf_load_test(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float16) / tf.constant(255.0, dtype=tf.float16)
    return image


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_split["image"].values, train_split["label"].values)
)
train_ds = (
    train_ds.shuffle(buffer_size=len(train_split), seed=42)
    .map(_tf_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_split["image"].values, val_split["label"].values)
)
val_ds = (
    val_ds.map(_tf_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 6
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=weights_dict,
    verbose=1,
)



## === cell 7
model.save("cactus.h5")



## === cell 8
test_image_paths = [os.path.join(test_path, fname) for fname in test_filenames]

test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = (
    test_ds.map(_tf_load_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

predictions = model.predict(test_ds, verbose=1)

submission_df = pd.DataFrame(
    {"id": test_filenames, "has_cactus": predictions.flatten()}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", submission_df.shape)
