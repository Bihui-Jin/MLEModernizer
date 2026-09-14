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

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass
try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

print("TF version:", tf.__version__)




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
im_size = IMSIZES[7]  # keep original choice

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

strategy = auto_select_accelerator()
BATCH_SIZE = 32

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

n_labels = 5  # ['complex','scab','frog_eye_leaf_spot','rust','powdery_mildew']

print("Train rows:", len(df), "Test rows:", len(test_df))
print("Example test image:", test_df["image"].iloc[0])



## === cell 3
AUTO = tf.data.AUTOTUNE

test_paths = tf.constant([os.path.join(test_dir, fn) for fn in test_df["image"].values])
num_test = int(test_paths.shape[0])


@tf.function
def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


@tf.function
def _apply_tta_batch(imgs, idxs, tta_pass_idx):
    """
    FIX: stateless_* ops require seed shape (2,), not (B,2).
    We keep deterministic, per-image augmentation by mapping each image with its own seed.
    """
    tta_pass_idx = tf.cast(tta_pass_idx, tf.int32)
    idxs = tf.cast(idxs, tf.int32)

    def _augment_one(x, idx):
        seed0 = tf.stack([42 + tta_pass_idx, idx], axis=0)  # (2,)
        x = tf.image.stateless_random_flip_left_right(x, seed=seed0)

        seed1 = seed0 + tf.constant([1, 0], dtype=tf.int32)
        x = tf.image.stateless_random_flip_up_down(x, seed=seed1)

        seed2 = seed0 + tf.constant([2, 0], dtype=tf.int32)
        factor = tf.random.stateless_uniform(
            shape=[],
            seed=seed2,
            minval=0.8,
            maxval=1.2,
            dtype=x.dtype,
        )
        x = x * factor
        return x

    return tf.map_fn(
        lambda t: _augment_one(t[0], t[1]),
        (imgs, idxs),
        fn_output_signature=imgs.dtype,
        parallel_iterations=32,
        back_prop=False,
    )


options = tf.data.Options()
options.experimental_deterministic = True

base_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, tf.range(num_test, dtype=tf.int32))
).with_options(options)

base_ds = base_ds.map(
    lambda path, idx: (_decode_resize_preprocess(path), idx),
    num_parallel_calls=AUTO,
    deterministic=True,
)

base_ds = base_ds.cache()
base_batched = base_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)


def make_test_ds(tta_pass_idx):
    tta_pass_idx = tf.cast(tta_pass_idx, tf.int32)

    def _map_batch(imgs, idxs):
        return _apply_tta_batch(imgs, idxs, tta_pass_idx)

    ds = base_batched.map(
        _map_batch, num_parallel_calls=AUTO, deterministic=True
    ).prefetch(AUTO)
    return ds




## === cell 4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
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

model(tf.zeros([1, im_size, im_size, 3], dtype=tf.float32))
print("Model built. Output shape:", model.output_shape)



## === cell 5
weights_path = "/kaggle/input/effnetb7-2/besteffb7_2.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights file not found at:", weights_path)
    print(
        "Proceeding without pretrained weights (submission will be valid but score will be low)."
    )



## === cell 6
TTA = 3


@tf.function(jit_compile=True)
def _infer_batch(x):
    return model(x, training=False)


pred_sum = np.zeros((num_test, n_labels), dtype=np.float32)

for i in range(TTA):
    ds = make_test_ds(i)
    write_pos = 0
    for batch in ds:
        out = _infer_batch(batch)
        out_np = out.numpy()
        bs = out_np.shape[0]
        pred_sum[write_pos : write_pos + bs] += out_np
        write_pos += bs
    if write_pos != num_test:
        raise RuntimeError(f"Dataset yielded {write_pos} samples, expected {num_test}")

pred = pred_sum / float(TTA)
print("Pred shape:", pred.shape)



## === cell 7
idx_to_name = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
}
healthy_label = "healthy"

threshold = {
    0: 0.25,
    1: 0.35,
    2: 0.60,
    3: 0.80,
    4: 0.80,
}

pred_string = []
for line in pred:
    labels = []
    for i in range(n_labels):
        if float(line[i]) > threshold[i]:
            labels.append(idx_to_name[i])
    if not labels:
        labels = [healthy_label]
    pred_string.append(" ".join(labels))

sub = pd.DataFrame({"image": test_df["image"].values, "labels": pred_string})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with rows:", len(sub))



## === cell 8
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image", "labels"]
assert len(check) == len(test_df)
print("Submission OK. Unique label strings (sample):", check["labels"].unique()[:10])
