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

os.environ["TF_XLA_FLAGS"] = os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_auto_jit=0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)




## === cell 1
def auto_select_accelerator():
    """
    TPU on Kaggle if available, else default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("TPU not found, using default strategy. Reason:", repr(e))
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # keep original core choice

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

df["labels"] = df["labels"].astype(str)

all_tokens = sorted({t for s in df["labels"].values for t in s.split(" ") if t})
class_name = all_tokens
n_labels = len(class_name)

print("Num classes (tokens):", n_labels)
print("Classes:", class_name)



## === cell 3
strategy = auto_select_accelerator()

BASE_BATCH_SIZE = 32
BATCH_SIZE = BASE_BATCH_SIZE * max(1, getattr(strategy, "num_replicas_in_sync", 1))

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

print("Test images:", len(test_df))
print("Batch size:", BATCH_SIZE)



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, (im_size, im_size), method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


ds_options = tf.data.Options()
ds_options.experimental_deterministic = True
try:
    ds_options.threading.private_threadpool_size = 16
except Exception:
    pass
try:
    ds_options.threading.max_intra_op_parallelism = 0
except Exception:
    pass

test_paths = tf.constant([os.path.join(test_dir, f) for f in test_df["image"].values])

ds_base = tf.data.Dataset.from_tensor_slices(test_paths).with_options(ds_options)
ds_base = ds_base.map(_read_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
ds_base = ds_base.cache()
ds_base = ds_base.batch(BATCH_SIZE, drop_remainder=False)
ds_base = ds_base.prefetch(AUTOTUNE)



## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet",  # fallback weights; may be overridden by loading custom checkpoint below
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
        jit_compile=False,
        run_eagerly=False,
    )

model.summary()




## === cell 6
def _find_file_under_kaggle_input(target_filename: str):
    base_dir = "/kaggle/input"
    for root, _, files in os.walk(base_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
resolved_path = (
    weights_path
    if tf.io.gfile.exists(weights_path)
    else _find_file_under_kaggle_input("bestmodel_tpu_aug.h5")
)

if resolved_path is not None and tf.io.gfile.exists(resolved_path):
    model.load_weights(resolved_path)
    print("Loaded weights:", resolved_path)
else:
    print(
        "WARNING: Custom weights not found (searched default path and /kaggle/input). "
        "Proceeding with ImageNet-initialized backbone weights."
    )



## === cell 7
TTA = 6


@tf.function
def _tta_mean_probs_batch(imgs_b):
    t0 = imgs_b
    t1 = tf.image.flip_left_right(imgs_b)
    t2 = tf.image.flip_up_down(imgs_b)
    t3 = tf.image.flip_up_down(tf.image.flip_left_right(imgs_b))
    t4 = tf.image.rot90(imgs_b, k=1)
    t5 = tf.image.rot90(imgs_b, k=3)

    imgs6 = tf.concat([t0, t1, t2, t3, t4, t5], axis=0)  # [6B,H,W,3]

    probs = model(imgs6, training=False)  # [6B,n_labels]
    b = tf.shape(imgs_b)[0]
    probs = tf.reshape(probs, (6, b, n_labels))  # [6,B,n_labels]
    probs = tf.reduce_mean(probs, axis=0)  # [B,n_labels]
    return probs


@tf.function
def _predict_all(ds):
    ta = tf.TensorArray(
        tf.float32, size=0, dynamic_size=True, element_shape=(None, n_labels)
    )
    i = tf.constant(0)
    for batch in ds:
        p = _tta_mean_probs_batch(batch)
        ta = ta.write(i, p)
        i += 1
    return ta.concat()


tta_pred_tf = _predict_all(ds_base)
tta_pred = tta_pred_tf.numpy()

argpred = np.argmax(tta_pred, axis=1)

class_name_arr = np.asarray(class_name, dtype=object)
sub = test_df.copy()
sub["labels"] = class_name_arr[argpred]
sub = sub[["image", "labels"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Submission path:", os.path.abspath("submission.csv"))
