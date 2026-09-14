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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
from sklearn.model_selection import train_test_split

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)




## === cell 1
def auto_select_accelerator():
    """
    Detects TPU (if available) else falls back to the default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:  # any issue → use default strategy
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


def _load_and_preprocess(path):
    """Read image file, decode, resize, and apply EfficientNet preprocessing."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [300, 300])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img




## === cell 2
load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

class_name = df.labels.unique().tolist()
print("Classes:", class_name)
print("Number of classes:", len(class_name))

df.labels = df.labels.astype(str)
n_labels = len(class_name)

strategy = auto_select_accelerator()
BATCH_SIZE = min(256, strategy.num_replicas_in_sync * 128)

test_root = os.path.join(load_dir, "test_images")
image_files = [
    f
    for f in os.listdir(test_root)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.isfile(os.path.join(test_root, f))
]
if not image_files:
    nested_dir = os.path.join(test_root, "test_images")
    if os.path.isdir(nested_dir):
        image_files = [
            f
            for f in os.listdir(nested_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.isfile(os.path.join(nested_dir, f))
        ]
        test_root = nested_dir

test_df = pd.DataFrame({"image": image_files})
test_paths = [os.path.join(test_root, fname) for fname in test_df["image"].tolist()]

test_set = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 3
with strategy.scope():
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=(300, 300, 3)
    )
    base.trainable = False  # frozen backbone for fast training
    model = Sequential(
        [base, GlobalMaxPooling2D(), Dense(n_labels, activation="softmax")]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )
    model.summary()



## === cell 4
train_root = os.path.join(load_dir, "train_images")
df["primary_label"] = df["labels"].apply(lambda x: str(x).split()[0])
label_to_idx = {label: i for i, label in enumerate(class_name)}
df["label_idx"] = df["primary_label"].map(label_to_idx)

train_paths = [os.path.join(train_root, fname) for fname in df["image"]]
train_labels = df["label_idx"].values

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(
        lambda p, l: (
            _load_and_preprocess(p),
            tf.one_hot(l, n_labels, dtype=tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .cache()  # Cache pre‑processed images to avoid repeated I/O
    .shuffle(1000)  # Shuffle after caching for per‑epoch randomness
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

model.fit(train_ds, epochs=3, verbose=1)



## === cell 5
weight_path = "/kaggle/input/modelplant1/bestmodel_tpu_aug.h5"
if os.path.exists(weight_path):
    try:
        model.load_weights(weight_path)
        print("Loaded pretrained weights.")
    except Exception as e:
        print("Failed to load weights:", e)
else:
    print("Weight file not found – proceeding with trained model.")



## === cell 6
pred = model.predict(test_set, verbose=0)
argpred = np.argmax(pred, axis=1)
test_df["labels"] = np.take(class_name, argpred)

submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
