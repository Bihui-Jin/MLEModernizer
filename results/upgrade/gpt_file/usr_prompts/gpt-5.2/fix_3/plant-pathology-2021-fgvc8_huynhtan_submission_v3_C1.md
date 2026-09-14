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

0.5674053554939973

# 6. Current score

0.29602

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.30228) has done: 'I remove the hard dependency on `tensorflow_addons` because it’s crashing due to an incompatibility in this environment, and it’s not actually used in your inference-only pipeline. Since the referenced pretrained `.h5` file isn’t available, I keep the same overall approach (TF dataset → Keras model → predict → threshold → submission) but train a small standard Keras image classifier from `train.csv` and `train_images/` so the notebook can run end-to-end and generate predictions for the test set. I also fix the filename/length mismatch by ensuring we build `labels` for exactly the same ordered list of `filenames` and by using the official `sample_submission.csv` ordering for the output. Finally, I keep your thresholding and “default to healthy” behavior (score-neutral to your intent) while making sure the submission file is written as `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.29602) has done: 'I keep the exact same model, loss, training loop, and thresholding logic, but remove avoidable Python overhead and make the input pipeline faster. The main speedups come from (1) building labels and filepaths with vectorized pandas/numpy instead of per-row Python `apply`, (2) using a fully TensorFlow-native decode path (`decode_jpeg`) to reduce `decode_image` overhead, and (3) adding deterministic dataset options + caching of decoded/resized images (per split) so repeated epoch reads don’t re-decode every time. I also eliminate the slow per-file `os.path.exists` scan (not needed for training) and vectorize submission label creation to avoid Python loops over every prediction.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

class_name = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
num_classes = len(class_name)
image_size = 224

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

print(train_df.head())
print(sub_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sub_df))




## === cell 2
class_to_idx = {c: i for i, c in enumerate(class_name)}

labels_series = train_df["labels"].astype(str)
y_mat = np.zeros((len(train_df), num_classes), dtype=np.float32)
for j, c in enumerate(class_name):
    y_mat[:, j] = labels_series.str.contains(
        rf"(?:^|\s){c}(?:\s|$)", regex=True
    ).to_numpy(dtype=np.float32)

train_df["filepath"] = (TRAIN_IMG_DIR + "/" + train_df["image"].astype(str)).to_numpy()
train_df["y"] = list(y_mat)


idx = np.arange(len(train_df))
np.random.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)
print("Train/Val:", len(tr_df), len(va_df))




## === cell 3
@tf.function
def parse_function(filename):
    image_string = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(
        image_string, channels=3
    )  # faster than decode_image for JPEGs
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def make_ds(df: pd.DataFrame, batch_size=16, training=False, cache_name=None):
    paths = df["filepath"].to_numpy()
    ys = np.stack(df["y"].to_numpy()).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: (parse_function(p), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache_name is not None:
        ds = ds.cache(cache_name)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


batch_size = 16
train_ds = make_ds(
    tr_df,
    batch_size=batch_size,
    training=True,
    cache_name="/kaggle/working/cache_train.tfdata",
)
val_ds = make_ds(
    va_df,
    batch_size=batch_size,
    training=False,
    cache_name="/kaggle/working/cache_val.tfdata",
)




## === cell 4
inputs = tf.keras.Input(shape=(image_size, image_size, 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
)

model.summary()




## === cell 5
epochs = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2,
)




## === cell 6
test_images = sub_df["image"].to_numpy()
test_paths = (TEST_IMG_DIR + "/" + sub_df["image"].astype(str)).to_numpy()

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
options = tf.data.Options()
options.experimental_deterministic = True
test_ds = test_ds.with_options(options)

test_ds = (
    test_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(8, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

prediction = model.predict(test_ds, verbose=1)
print("Pred shape:", prediction.shape, "Expected:", (len(test_images), num_classes))




## === cell 7
pred_bin = prediction > 0.5  # strictly equivalent to per-element float(value) > 0.5
labels_out = []
for row in pred_bin:
    idxs = np.flatnonzero(row)
    if idxs.size == 0:
        labels_out.append("healthy")
    else:
        labels_out.append(" ".join([class_name[i] for i in idxs]))

print(labels_out[:10], " ... total:", len(labels_out))




## === cell 8
submission = pd.DataFrame({"image": test_images, "labels": labels_out})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
