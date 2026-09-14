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

# 5. Target score

0.78016620498615

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the protobuf/TensorFlow crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting, which triggers the `MessageFactory.GetPrototype` failure in this environment. Then I fix the tf.data TTA pipeline error by replacing the invalid `parallel_iterations=tf.data.AUTOTUNE` (must be a positive int) with a safe integer, keeping the same TTA logic and model inference semantics. Finally, I add a robust fallback to generate a valid `submission.csv` in case the external weights file is missing, ensuring end-to-end execution and a correctly formatted submission file.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash happening before any training/inference by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow (this avoids the `MessageFactory.GetPrototype` mismatch in this environment). I keep the rest of the pipeline (EfficientNetB7 + softmax over unique label strings + TTA inference) unchanged to preserve core logic and semantics. I also make the import order deterministic and ensure the submission is always written as `submission.csv` with the required `image,labels` columns. This should both unblock execution and restore the intended (weight-loaded) scoring behavior toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

os.environ.setdefault("PYTHONHASHSEED", "42")
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as _:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    TPU if available, otherwise default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("TPU not available, using default strategy. Reason:", repr(e))
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # keep original choice: 600

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num classes (unique label strings):", n_labels)
print("First 10 classes:", class_name[:10])



## === cell 3
strategy = auto_select_accelerator()

BATCH_SIZE = 64

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image"]].copy()
assert test_df["image"].nunique() == len(
    test_df
), "Duplicate images in sample submission?"

print("Test rows:", len(test_df))
print("Test dir exists:", os.path.isdir(test_dir))



## === cell 4
from tensorflow.keras.applications.efficientnet import preprocess_input

test_paths = tf.constant(
    [os.path.join(test_dir, fn) for fn in test_df["image"].tolist()]
)

AUTOTUNE = tf.data.AUTOTUNE

_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.deterministic = True


@tf.function
def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function
def _tta_stack(img, tta):
    img0 = img
    img1 = tf.image.flip_up_down(img)
    img2 = tf.image.flip_left_right(img)
    img3 = tf.image.flip_up_down(img2)
    base4 = tf.stack([img0, img1, img2, img3], axis=0)  # [4,H,W,3]

    tta = tf.cast(tta, tf.int32)
    idx = tf.math.mod(tf.range(tta, dtype=tf.int32), 4)  # [tta]
    return tf.gather(base4, idx, axis=0)


def make_base_test_ds_cached():
    ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(_DATA_OPTIONS)
    ds = ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.cache()  # in-memory cache; avoids expensive filesystem I/O
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_tta_batched_ds(base_ds, tta):
    tta = int(tta)
    ds = base_ds.with_options(_DATA_OPTIONS)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    _PAR_ITERS = 16

    ds = ds.map(
        lambda x: tf.map_fn(
            lambda img: _tta_stack(img, tf.constant(tta, tf.int32)),
            x,
            fn_output_signature=tf.TensorSpec(
                shape=(tta, None, None, 3), dtype=tf.float32
            ),
            parallel_iterations=_PAR_ITERS,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.prefetch(AUTOTUNE)
    return ds




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
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
    _HAS_WEIGHTS = True
else:
    print("WARNING: Weights not found:", weights_path)
    print(
        "Proceeding with a deterministic fallback to generate a valid submission.csv."
    )
    _HAS_WEIGHTS = False

try:
    model.run_eagerly = False
except Exception:
    pass



## === cell 7
TTA = 6

if _HAS_WEIGHTS:
    base_ds = make_base_test_ds_cached()
    tta_batched_ds = make_tta_batched_ds(base_ds, TTA)

    pred_chunks = []
    for x in tta_batched_ds:
        b = tf.shape(x)[0]
        x_flat = tf.reshape(x, [b * TTA, im_size, im_size, 3])
        p = model(x_flat, training=False)  # [B*TTA, n_labels]
        p = tf.reshape(p, [b, TTA, n_labels])
        pred_chunks.append(p)

    p_all = tf.concat(pred_chunks, axis=0).numpy()  # [n_test, TTA, n_labels]

    n_test = len(test_df)
    if p_all.shape[0] != n_test or p_all.shape[1] != TTA:
        raise RuntimeError(
            f"TTA prediction shape mismatch: got {p_all.shape}, expected ({n_test},{TTA},{n_labels})"
        )

    pred = p_all.mean(axis=1)  # [n_test, n_labels]
    argpred = np.argmax(pred, axis=1)

    if len(argpred) != len(test_df):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(argpred)} preds, expected {len(test_df)}"
        )

    test_df["labels"] = argpred.astype(np.int32)
    test_df["labels"] = np.take(
        np.array(class_name, dtype=object), test_df["labels"].values
    )
else:
    fallback_label = pd.read_csv(sample_sub_path)["labels"].mode().iloc[0]
    test_df["labels"] = fallback_label

submission = test_df[["image", "labels"]].copy()
assert list(submission.columns) == ["image", "labels"]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("Submission path:", os.path.abspath("submission.csv"))
