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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TF version:", tf.__version__)
print("Num GPUs available:", len(tf.config.list_physical_devices("GPU")))

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
        print("Running on default strategy (CPU/GPU).")
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num unique label-strings (as used by this model):", n_labels)
print("Example label-strings:", class_name[:10])



## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")

sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image"]].copy()

test_images_in_dir = None
try:
    test_images_in_dir = set(tf.io.gfile.listdir(test_dir))
except Exception:
    test_images_in_dir = None

first_missing = None
if test_images_in_dir is not None:
    for img in test_df["image"].tolist():
        if img not in test_images_in_dir:
            first_missing = img
            break
else:
    for img in test_df["image"].tolist():
        if not tf.io.gfile.exists(os.path.join(test_dir, img)):
            first_missing = img
            break

if first_missing is not None:
    print(
        "Warning: at least 1 image from sample_submission not found in test_images folder "
        f"(e.g., {first_missing}). Falling back to folder listing."
    )
    if test_images_in_dir is None:
        test_df = pd.DataFrame({"image": sorted(tf.io.gfile.listdir(test_dir))})
    else:
        test_df = pd.DataFrame({"image": sorted(test_images_in_dir)})

print("Test images:", len(test_df))



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _preprocess_base(img):
    img = img * (1.0 / 255.0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _apply_tta(img, idx, tta_round):
    seed_h = tf.stack([tf.cast(tta_round, tf.int32), tf.cast(idx, tf.int32)])
    seed_v = tf.stack([tf.cast(tta_round + 1337, tf.int32), tf.cast(idx, tf.int32)])
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_h)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed_v)
    return img


def make_test_tta_ds(images, tta=5):
    paths = tf.constant([os.path.join(test_dir, x) for x in images], dtype=tf.string)
    idxs = tf.range(tf.shape(paths)[0], dtype=tf.int32)

    base_ds = tf.data.Dataset.from_tensor_slices((paths, idxs))

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune = True
        opts.experimental_optimization.map_vectorization.enabled = True
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    base_ds = base_ds.with_options(opts)

    base_ds = base_ds.map(
        lambda p, i: (_preprocess_base(_read_decode_resize(p)), i),
        num_parallel_calls=AUTOTUNE,
    ).cache()

    base_rep = base_ds.repeat(tta)

    counter = tf.data.Dataset.range(
        tf.cast(tf.shape(paths)[0], tf.int64) * tf.cast(tta, tf.int64)
    )

    def _attach_round(img_idx, c):
        img, idx = img_idx
        r = tf.cast(tf.math.floormod(c, tf.cast(tta, tf.int64)), tf.int32)
        img = _apply_tta(img, idx, r)
        return img

    ds = tf.data.Dataset.zip((base_rep, counter)).map(
        _attach_round, num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_images_list = test_df["image"].tolist()
n_test = len(test_images_list)



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
    )

model.summary()



## === cell 6
weights_path = "/kaggle/input/modelplant1/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print(
        f"Warning: weights file not found at {weights_path}. Proceeding with random weights (score will be poor)."
    )



## === cell 7
import math

TTA = 5

tta_ds = make_test_tta_ds(test_images_list, tta=TTA)

n_pred = n_test * TTA
steps = int(math.ceil(n_pred / BATCH_SIZE))

preds_all = model.predict(tta_ds, verbose=0, steps=steps)
preds_all = preds_all[:n_pred]
if preds_all.shape[0] != n_pred:
    raise RuntimeError(
        f"Prediction count mismatch: got {preds_all.shape[0]}, expected {n_pred}"
    )

preds_all = preds_all.reshape((n_test, TTA, n_labels))
pred = preds_all.mean(axis=1)

argpred = np.argmax(pred, axis=1)

test_df = test_df.copy()
test_df["labels"] = [class_name[i] for i in argpred]

submission = test_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
