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

0.6655216989843012

# 6. Current score

0.27058

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.27058) has done: 'Your timeout is dominated by repeatedly decoding/resizing large JPEGs on-the-fly while the model trains, plus some extra overhead from determinism and Python-side loops. I keep the exact same model, loss, optimizer, epochs, and thresholding logic, but speed up the input pipeline by enabling dataset caching (to RAM/disk) so images are decoded/resized once instead of every epoch, and by adding a few deterministic tf.data options that reduce overhead while preserving semantics. I also remove the expensive, unnecessary protobuf downgrade step (TensorFlow 2.18 is compatible with protobuf 6.x here), and I vectorize the submission label formatting to avoid per-row Python work. These changes are all equivalence-preserving for outputs (aside from negligible FP noise) and should bring runtime under 600 seconds.'
- What this solution (achieved 0.27058) has done: 'I remove the protobuf-related crash by eliminating the unnecessary import/subprocess logic and forcing a clean TensorFlow import (this error is coming from an incompatible protobuf runtime path). Then I keep the exact same model/training/thresholding core logic, but ensure the input pipeline and label formatting run correctly end-to-end in Kaggle and always write a valid `submission.csv`. Finally, I make the submission label generation robust and faster (vectorized where possible) without changing the decision rule, so runtime stays under the limit and the current score can improve toward the target primarily by actually completing training/inference successfully.'
- What this solution (achieved 0.27058) has done: 'I remove the protobuf auto-downgrade/restart logic because it can prevent a submission from being produced (and TF 2.18 here is compatible with protobuf 6.x), which is why your score is currently “Not yielded.” Then I keep your exact model/training loop/loss and prediction thresholding, but make two minimal, score-relevant fixes: ensure the output label list is always in a stable sorted order (prevents accidental label-order mismatch across runs) and vectorize the label-string creation to avoid Python loops that can time out before writing `submission.csv`. These changes preserve the same evaluation semantics and should reliably finish end-to-end within the time limit, yielding a valid submission and typically improving score simply by successfully training/inferencing on all test images.'
- What this solution (achieved 0.27058) has done: 'The timeout is dominated by repeatedly decoding/resizing ~15k high‑resolution JPEGs on the CPU during training; we keep the exact same model and training loop but remove avoidable overhead in the input pipeline. I (1) enable dataset caching to disk so each image is decoded/resized only once across epochs, (2) switch the map to a stateless function (no label default/branch) and set `deterministic=True` for stable results, (3) precompute constant tensors and reduce Python↔TF crossings, and (4) vectorize the submission label formatting to avoid per-row Python loops. These changes preserve training semantics (same data, same augmentations = none, same epochs/batches/loss/model), but significantly reduce wall-clock time.'
- What this solution (achieved 0.27058) has done: 'I fix the crash in the first cell by removing the protobuf environment override that forces the pure-Python protobuf implementation (it is what triggers the `MessageFactory.GetPrototype` error with TF 2.18 + protobuf 6.x). Then I keep your exact data processing/model/training/inference logic, but make path handling robust for both Kaggle `/kaggle/input/...` and the provided relative paths so it always finds the CSVs/images. Finally, I ensure the submission label strings are produced deterministically (sorted class order is already enforced) and that `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.27058) has done: 'I remove the line that unsets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` because it triggers the TensorFlow/protobuf incompatibility causing the `MessageFactory.GetPrototype` crash at import time. Then I keep your exact data pipeline, model, training loop, and thresholding logic, but make one minimal score-relevant fix: after thresholding, if any disease labels are predicted we explicitly exclude `healthy` from the output string (and only output `healthy` when nothing else is predicted), which matches the competition’s label semantics and typically improves mean F1. I also add small path/type robustness (ensuring `test_paths` is a NumPy object array) so the tf.data pipeline always builds correctly. The script still writes a valid `submission.csv` with `image,labels` columns.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf


print("Python:", sys.version)
print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"

base_input_candidates = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8",
]
base_input = None
for p in base_input_candidates:
    if os.path.exists(p):
        base_input = p
        break
if base_input is None:
    base_input = "../input/plant-pathology-2021-fgvc8"

train_csv_path = os.path.join(base_input, "train.csv")
sample_sub_path = os.path.join(base_input, "sample_submission.csv")
train_dir = os.path.join(base_input, "train_images")
test_dir = os.path.join(base_input, "test_images")

assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.exists(train_dir), f"Missing train_images dir at {train_dir}"
assert os.path.exists(test_dir), f"Missing test_images dir at {test_dir}"

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
one_hot = one_hot.reindex(sorted(one_hot.columns), axis=1)

dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)

print("Base input:", base_input)
print("Train rows:", len(data_set), "Num classes:", num_classes)
print("First classes:", dataset_labels[:10])



## === cell 2
train_paths = (train_dir + "/" + data_set["image"].astype(str)).to_numpy(dtype=object)
y = one_hot.to_numpy(dtype=np.float32)

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

SIZE_T = tf.constant([image_dims[0], image_dims[1]], dtype=tf.int32)


@tf.function
def decode_and_resize_xy(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, SIZE_T)
    return img, label


@tf.function
def decode_and_resize_x(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, SIZE_T)
    return img


idx = np.arange(len(train_paths))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_paths = train_paths[tr_idx]
tr_y = y[tr_idx]
val_paths = train_paths[val_idx]
val_y = y[val_idx]

options = tf.data.Options()
options.deterministic = True

cache_dir = os.path.join(output_dir, "tf_cache_pp2021")
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(
    cache_dir, f"train_cache_{image_dims[0]}x{image_dims[1]}_bs{BATCH_SIZE}"
)
val_cache_path = os.path.join(
    cache_dir, f"val_cache_{image_dims[0]}x{image_dims[1]}_bs{BATCH_SIZE}"
)

train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_y)).with_options(options)
train_ds = train_ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(
    decode_and_resize_xy, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(options)
val_ds = val_ds.map(
    decode_and_resize_xy, num_parallel_calls=AUTOTUNE, deterministic=True
)
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.prefetch(AUTOTUNE)

print("Train batches:", int(tf.data.experimental.cardinality(train_ds).numpy()))
print("Val batches:", int(tf.data.experimental.cardinality(val_ds).numpy()))



## === cell 3
inputs = tf.keras.Input(shape=image_dims, name="image")
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
)

EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## === cell 4
sub_df = pd.read_csv(sample_sub_path)
test_images = sub_df["image"].astype(str).to_numpy()

test_paths = (test_dir + "/" + pd.Series(test_images)).to_numpy(dtype=object)

test_cache_path = os.path.join(
    cache_dir, f"test_cache_{image_dims[0]}x{image_dims[1]}_bs{BATCH_SIZE}"
)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = test_ds.map(
    decode_and_resize_x, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

pred = model.predict(test_ds, verbose=0)
pred = np.asarray(pred, dtype=np.float32)

THRESH = 0.60
above = pred > THRESH

labels_arr = np.asarray(dataset_labels, dtype=object)
class_str = labels_arr.astype(str)

healthy_idx = dataset_labels.index("healthy") if "healthy" in dataset_labels else None
if healthy_idx is not None:
    above_nonhealthy = above.copy()
    above_nonhealthy[:, healthy_idx] = False
else:
    above_nonhealthy = above

out_labels = np.full(len(sub_df), "healthy", dtype=object)
rows_with_any = np.where(above_nonhealthy.any(axis=1))[0]
if rows_with_any.size:
    for r in rows_with_any:
        out_labels[r] = " ".join(class_str[np.flatnonzero(above_nonhealthy[r])])

sub_df["labels"] = out_labels

csv_path = os.path.join(output_dir, "submission.csv")
sub_df[["image", "labels"]].to_csv(csv_path, index=False)
print(f"Wrote submission to: {csv_path} (rows={len(sub_df)})")
print(sub_df.head())
