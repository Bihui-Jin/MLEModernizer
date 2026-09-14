# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.6415143120960296

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.75058) has done: 'I remove `tensorflow_addons` (it fails to import in this environment due to protobuf incompatibility) and keep the rest of the TensorFlow pipeline intact. Since the referenced pre-trained model file does not exist in your input, I add a minimal “train-once then predict” fallback using a standard Keras ImageNet backbone so the notebook runs end-to-end and produces a valid `submission.csv`. I also fix path handling, ensure the labels are space-delimited strings, and make sure the submission uses only the basename for `image`. These changes are required for correctness and to yield a non-empty submission file.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    import pkgutil
    import importlib.metadata as importlib_metadata

    pb_ver = importlib_metadata.version("protobuf")
    if pb_ver.startswith("6."):
        print(
            f"Detected protobuf=={pb_ver}; installing protobuf==4.25.3 for TF compatibility..."
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        print(
            "Restarting Python process to reload TensorFlow with compatible protobuf..."
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception as e:
    print("Protobuf compatibility precheck warning:", repr(e))



## === cell 1
import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 2
class_name = [
    "healthy",
    "scab frog_eye_leaf_spot complex",
    "scab",
    "complex",
    "rust",
    "frog_eye_leaf_spot",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
    "frog_eye_leaf_spot complex",
    "rust frog_eye_leaf_spot",
    "powdery_mildew complex",
    "rust complex",
]

num_classes = 12
image_size = 224




## === cell 3
@tf.function
def parse_function(filename):
    image_string = tf.io.read_file(filename)
    image = tf.io.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image




## === cell 4
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert tf.io.gfile.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert tf.io.gfile.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert tf.io.gfile.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert tf.io.gfile.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

label_to_idx = {s: i for i, s in enumerate(class_name)}
train_df = train_df[train_df["labels"].isin(label_to_idx)].copy()
train_df["y"] = train_df["labels"].map(label_to_idx).astype("int32")
train_df["filepath"] = train_df["image"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

exists_mask = train_df["filepath"].apply(tf.io.gfile.exists)
train_df = train_df[exists_mask].reset_index(drop=True)

len_train = len(train_df)
len_test = len(sample_df)
print("Filtered train rows:", len_train, "Test rows:", len_test)



## === cell 5
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def decode_with_label(filename, label):
    image = parse_function(filename)
    return image, tf.one_hot(label, depth=num_classes)


idx = np.arange(len_train)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
split = int(0.9 * len_train)
tr_idx, va_idx = idx[:split], idx[split:]

tr_files = train_df.loc[tr_idx, "filepath"].values
tr_labels = train_df.loc[tr_idx, "y"].values
va_files = train_df.loc[va_idx, "filepath"].values
va_labels = train_df.loc[va_idx, "y"].values

train_ds = (
    tf.data.Dataset.from_tensor_slices((tr_files, tr_labels))
    .shuffle(min(len(tr_files), 2048), seed=SEED, reshuffle_each_iteration=True)
    .map(decode_with_label, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((va_files, va_labels))
    .map(decode_with_label, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

model_path = "../input/bestmodel/bestmodel_tpu.h5"
if tf.io.gfile.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    base.trainable = False  # keep runtime bounded and stable

    inp = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inp * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=2)



## === cell 6
filenames = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
filenames = sorted(filenames)

test_ds = (
    tf.data.Dataset.from_tensor_slices(filenames)
    .map(parse_function, num_parallel_calls=AUTOTUNE)
    .batch(8)
    .prefetch(AUTOTUNE)
)

pred_proba = model.predict(test_ds, verbose=0)
prediction = tf.argmax(pred_proba, axis=1).numpy().astype(int)

print("Pred shape:", pred_proba.shape, "Pred classes:", prediction.shape)



## === cell 7
submission_pred = pd.DataFrame(
    {"image": [os.path.basename(f) for f in filenames], "labels": prediction}
)
submission_pred["labels"] = (
    submission_pred["labels"].apply(lambda x: class_name[int(x)]).astype(str)
)

submission = sample_df[["image"]].merge(submission_pred, on="image", how="left")
submission["labels"] = submission["labels"].fillna("healthy").astype(str)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
