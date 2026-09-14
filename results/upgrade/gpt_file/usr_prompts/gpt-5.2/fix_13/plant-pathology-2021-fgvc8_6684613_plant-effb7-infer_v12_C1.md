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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.keras.backend.clear_session()
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy (GPU/CPU).
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
im_size = IMSIZES[7]

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

df["labels"] = df["labels"].astype(str)

all_tokens = set()
for s in df["labels"].values:
    for t in str(s).split():
        if t.strip():
            all_tokens.add(t.strip())

class_name = sorted(all_tokens)
n_labels = len(class_name)

print("classes:", class_name)
print("n_classes:", n_labels)




## === cell 3
strategy = auto_select_accelerator()

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"

test_images = sorted(tf.io.gfile.listdir(test_dir))
test_df = pd.DataFrame({"image": test_images})

print("n_test_images:", len(test_df))
print("example test image:", test_df["image"].iloc[0] if len(test_df) else None)




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

DATA_OPTS = tf.data.Options()
try:
    DATA_OPTS.experimental_deterministic = True
except Exception:
    pass
try:
    DATA_OPTS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    DATA_OPTS.experimental_slack = True
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_INTRA_OP", "0"))
        or tf.config.threading.get_intra_op_parallelism_threads()
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_INTER_OP", "0"))
        or tf.config.threading.get_inter_op_parallelism_threads()
    )
except Exception:
    pass


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.resize(img, [im_size, im_size], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)  # EfficientNet preprocess expects float
    return img


def _preprocess_effnet(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


test_paths = tf.strings.join([tf.constant(test_dir), tf.constant(test_images)])
base_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(DATA_OPTS)
    .map(_decode_resize, num_parallel_calls=AUTOTUNE)
    .map(_preprocess_effnet, num_parallel_calls=AUTOTUNE)
)

batched_base_ds = base_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)


@tf.function
def _tta_augment_batch(images, seed2):
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(seed2, tf.int32)], axis=0)  # [2]
    images = tf.image.stateless_random_flip_left_right(images, seed=seed)
    images = tf.image.stateless_random_flip_up_down(
        images, seed=seed + tf.constant([1, 1], tf.int32)
    )
    deltas = tf.random.stateless_uniform(
        [tf.shape(images)[0], 1, 1, 1],
        seed=seed + tf.constant([2, 2], tf.int32),
        minval=-0.2,
        maxval=0.2,
        dtype=images.dtype,
    )
    images = images + deltas
    return images


def make_single_pass_tta_ds(batched_ds, tta):
    tta_ids = tf.data.Dataset.from_tensor_slices(tf.range(tta, dtype=tf.int32))

    def _expand_batch(batch):
        return tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensors(batch).repeat(tta), tta_ids)
        )

    ds = batched_ds.flat_map(_expand_batch)

    def _apply_tta(batch, tta_idx):
        seed2 = tf.cast(tta_idx * 100000 + 7, tf.int32)
        return _tta_augment_batch(batch, seed2)

    ds = ds.map(_apply_tta, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)




## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential()
    model.add(base)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(n_labels, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )

model.summary()




## === cell 6
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights file not found at:", weights_path)
    print(
        "Proceeding with untrained weights (submission will be low-scoring but valid)."
    )




## === cell 7
TTA = 6

n_test = len(test_df)

tta_ds = make_single_pass_tta_ds(batched_base_ds, TTA)
pred_all = model.predict(tta_ds, verbose=0)  # shape: (n_test*TTA, n_labels)

pred_all = pred_all.reshape((TTA, n_test, n_labels))
pred = pred_all.mean(axis=0)

class_name_arr = np.asarray(class_name, dtype=object)

top1 = np.argmax(pred, axis=1)
top1_prob = pred[np.arange(pred.shape[0]), top1]

REL = float(os.environ.get("REL_THR", "0.55"))  # include if p >= REL * p_top1

thr = (REL * top1_prob)[:, None]
keep_mask = pred >= thr
keep_mask[np.arange(pred.shape[0]), top1] = True  # ensure top1 kept

labels_out = []
for i in range(pred.shape[0]):
    idx = np.flatnonzero(keep_mask[i])
    idx = idx[np.argsort(-pred[i, idx])]
    lab = " ".join(class_name_arr[idx].tolist()).strip()
    if not lab:
        lab = "healthy"
    labels_out.append(lab)

test_df["labels"] = labels_out

submission_path = "submission.csv"
test_df[["image", "labels"]].to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(test_df.head())
print("Submission rows:", len(test_df))
print("Submission columns:", list(test_df[["image", "labels"]].columns))
