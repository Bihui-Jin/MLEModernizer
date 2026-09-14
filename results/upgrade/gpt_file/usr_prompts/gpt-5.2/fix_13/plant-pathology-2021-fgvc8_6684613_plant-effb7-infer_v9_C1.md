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
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy. Kept as-is logically, just made exception broader.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # 600 as in original code

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num classes:", n_labels)
print("First 10 classes:", class_name[:10])



## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = 32

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

print("Test images:", len(test_df))
test_df.head()



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_and_cast(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        (im_size, im_size),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    return tf.cast(img, tf.float32)


def _preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

test_files = tf.constant(test_df["image"].values)
test_paths = tf.strings.join([tf.constant(test_dir), test_files])

base_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
base_ds = base_ds.map(_decode_resize_and_cast, num_parallel_calls=AUTOTUNE)
base_ds = base_ds.prefetch(AUTOTUNE)



## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalMaxPooling2D(),
            Dense(n_labels, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
        jit_compile=True,
    )

model.summary()



## === cell 6
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if tf.io.gfile.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights not found:", weights_path)
    print("Proceeding with untrained model to produce a valid submission.csv.")



## === cell 7
import math

TTA = 6


@tf.function
def _augment_with_idx(image_idx, tta_idx, img):
    seed = tf.stack(
        [
            tf.cast(SEED, tf.int64),
            tf.cast(tta_idx, tf.int64) * tf.cast(1000003, tf.int64)
            + tf.cast(image_idx, tf.int64),
        ]
    )

    img2 = tf.image.stateless_random_flip_left_right(
        img, seed=tf.random.experimental.stateless_fold_in(seed, 1)
    )
    img2 = tf.image.stateless_random_flip_up_down(
        img2, seed=tf.random.experimental.stateless_fold_in(seed, 2)
    )

    scale = tf.random.stateless_uniform(
        [],
        seed=tf.random.experimental.stateless_fold_in(seed, 3),
        minval=0.9,
        maxval=1.1,
    )
    new_size = tf.cast(tf.round(scale * tf.cast(im_size, tf.float32)), tf.int32)
    img2 = tf.image.resize(
        img2,
        (new_size, new_size),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, im_size, im_size)

    factor = tf.random.stateless_uniform(
        [],
        seed=tf.random.experimental.stateless_fold_in(seed, 4),
        minval=0.9,
        maxval=1.1,
    )
    img2 = tf.clip_by_value(img2 * factor, 0.0, 255.0)

    img2 = _preprocess(img2)
    return img2


def make_tta_dataset(base, tta, n_items):
    n_items = int(n_items)
    total = n_items * int(tta)

    image_idx = tf.range(total, dtype=tf.int64) % tf.cast(n_items, tf.int64)
    tta_idx = tf.range(total, dtype=tf.int64) // tf.cast(n_items, tf.int64)

    ds_img = base.repeat(tta)
    ds_idx = tf.data.Dataset.from_tensor_slices((image_idx, tta_idx))
    ds = tf.data.Dataset.zip((ds_idx, ds_img)).with_options(options)

    ds = ds.map(
        lambda idx, img: _augment_with_idx(idx[0], idx[1], img),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    try:
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/device:GPU:0"))
    except Exception:
        ds = ds.prefetch(AUTOTUNE)
    return ds


n_test = len(test_df)
tta_ds = make_tta_dataset(base_ds, TTA, n_test)

steps = int(math.ceil((n_test * TTA) / BATCH_SIZE))
pred_all = model.predict(tta_ds, steps=steps, verbose=1)

pred_all = pred_all[: (n_test * TTA)]
pred_all = pred_all.reshape((TTA, n_test, n_labels))
pred = pred_all.mean(axis=0)

argpred = np.argmax(pred, axis=1)

class_name_arr = np.array(class_name, dtype=object)
test_df["labels"] = class_name_arr[argpred]

submission = test_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
