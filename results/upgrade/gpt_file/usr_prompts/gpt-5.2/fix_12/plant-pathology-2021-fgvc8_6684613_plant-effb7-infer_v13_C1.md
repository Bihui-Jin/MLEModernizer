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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.keras.backend.set_learning_phase(0)




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
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
test_dir = os.path.join(load_dir, "test_images")

df = pd.read_csv(train_csv_path)

test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame({"image": test_images})

IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # 600, keep as in original
n_labels = 5
BATCH_SIZE = 32

strategy = auto_select_accelerator()

print("Train rows:", df.shape, "Test rows:", test_df.shape)
print("Example test image:", test_df["image"].iloc[0])




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE

test_paths = tf.constant([os.path.join(test_dir, f) for f in test_images])


@tf.function
def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, (im_size, im_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


n_test = len(test_images)


def _select_predict_batch_size(base_bs: int) -> int:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        _ = tpu.master()
        return max(base_bs, 64)
    except Exception:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return max(base_bs, 64)
    return base_bs


PRED_BATCH_SIZE = _select_predict_batch_size(BATCH_SIZE)

base_test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

options = tf.data.Options()
options.experimental_deterministic = True
options.threading.private_threadpool_size = 0
options.threading.max_intra_op_parallelism = 0
base_test_ds = base_test_ds.with_options(options)

base_test_ds = base_test_ds.map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)

base_test_ds = base_test_ds.cache()
base_test_ds = base_test_ds.batch(PRED_BATCH_SIZE, drop_remainder=False)
base_test_ds = base_test_ds.prefetch(AUTOTUNE)




## === cell 4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,  # preserve original (weights loaded from file later)
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dense(n_labels, activation="sigmoid"),
        ]
    )

model.run_eagerly = False

model.summary()




## === cell 5
weights_path = "/kaggle/input/effnetb7-2/besteffb7_2.h5"
if tf.io.gfile.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Loaded weights: {weights_path}")
else:
    print(
        f"WARNING: Weights not found at {weights_path}. Proceeding without loading weights."
    )




## === cell 6
TTA = 3


@tf.function(reduce_retracing=True)
def _infer_batch(x):
    return model(x, training=False)


pred_sum = np.zeros((n_test, n_labels), dtype=np.float32)

for t in range(TTA):
    outs = []
    for batch in base_test_ds:
        y = _infer_batch(batch)
        outs.append(y)
    pred_t = tf.concat(outs, axis=0)[:n_test]
    pred_sum += pred_t.numpy()

pred = pred_sum / float(TTA)
print("Pred shape:", pred.shape)




## === cell 7
name = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {
    0: 0.25,
    1: 0.35,
    2: 0.5,
    3: 0.6,
    4: 0.6,
}

thr = np.array([threshold[i] for i in range(n_labels)], dtype=np.float32)
mask = pred > thr  # (n_test, n_labels)

pred_string = []
for row_mask in mask:
    labels = [name[i] for i, m in enumerate(row_mask) if m]
    pred_string.append(" ".join(labels) if labels else name[6])

test_df["labels"] = pred_string

submission = test_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
submission.head()




## === cell 8
print(submission.shape)
print(submission.columns.tolist())
print(submission.head(3).to_string(index=False))
print("Wrote submission.csv")
