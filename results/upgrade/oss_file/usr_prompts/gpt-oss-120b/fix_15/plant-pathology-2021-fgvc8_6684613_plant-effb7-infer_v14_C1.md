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
import random
import sys
import multiprocessing

try:
    import tensorflow as tf

    tf_available = True
except Exception as e:
    print("TensorFlow import failed:", e)
    tf_available = False
    tf = None

if tf_available:
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)

    tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())




## === cell 1
def auto_select_accelerator():
    """
    Detect and initialize TPU if available; otherwise fall back to
    MirroredStrategy (uses all available devices/CPU cores).
    """
    if not tf_available:
        return None
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.MirroredStrategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[5]  # 456 instead of 600
BATCH_SIZE = (
    64  # reduced batch size to lower memory pressure and speed up CPU inference
)
n_labels = 5  # number of target disease classes (healthy handled separately)

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": os.listdir(test_dir)})

if not tf_available:
    test_df["labels"] = "healthy"
    submission_path = "submission.csv"
    test_df.to_csv(submission_path, index=False)
    print(f"TensorFlow unavailable – wrote default submission to {submission_path}")
    skip_remaining = True
else:
    skip_remaining = False
    strategy = auto_select_accelerator()



## === cell 2
if not skip_remaining:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    test_image_paths = [os.path.join(test_dir, fname) for fname in test_df["image"]]

    def _load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [im_size, im_size])
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    def _augment(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image

    base_test_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)
    base_test_dataset = base_test_dataset.map(
        _load_and_preprocess, num_parallel_calls=AUTOTUNE
    ).cache(
        filename="/tmp/test_image_cache"
    )  # disk‑based cache

    options = tf.data.Options()
    options.experimental_deterministic = False
    base_test_dataset = base_test_dataset.with_options(options)

    def make_tta_dataset(TTA, base_dataset):
        """
        Creates a dataset that repeats each image TTA times with fresh random augmentations.
        Decoding is cached, so only augmentation is repeated.
        """
        tta_dataset = base_dataset.repeat(TTA)  # repeat after caching
        tta_dataset = tta_dataset.map(
            _augment, num_parallel_calls=AUTOTUNE
        )  # then augment
        tta_dataset = tta_dataset.batch(BATCH_SIZE, drop_remainder=False)
        tta_dataset = tta_dataset.prefetch(AUTOTUNE)
        return tta_dataset




## === cell 3
if not skip_remaining:
    with strategy.scope():
        base = tf.keras.applications.EfficientNetB7(
            weights=None,
            include_top=False,
            input_shape=(im_size, im_size, 3),
        )
        model = Sequential(
            [base, GlobalAveragePooling2D(), Dense(n_labels, activation="sigmoid")]
        )



## === cell 4
if not skip_remaining:
    weights_path = "/kaggle/input/effnetb7-2/besteffb7_2.h5"
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Pretrained weights loaded.")
        except Exception as e:
            print("Error loading weights:", e)
    else:
        print("Weights file not found – using random initialization.")



## === cell 5
if not skip_remaining:
    TTA = 3
    tta_dataset = make_tta_dataset(TTA, base_test_dataset)
    all_preds = model.predict(tta_dataset, verbose=0)
    num_test = test_df.shape[0]
    preds = all_preds.reshape((TTA, num_test, n_labels))
    pred = np.mean(preds, axis=0)  # shape (num_test, n_labels)



## === cell 6
if not skip_remaining:
    name = {
        0: "complex",
        1: "scab",
        2: "frog_eye_leaf_spot",
        3: "rust",
        4: "powdery_mildew",
        6: "healthy",  # used when no disease thresholds are met
    }
    threshold = {0: 0.25, 1: 0.35, 2: 0.8, 3: 0.8, 4: 0.8}

    pred_strings = []
    for line in pred:
        tags = ""
        for i in range(n_labels):
            if line[i] > threshold.get(i, 0.5):
                tags += name[i] + " "
        tags = tags.strip()
        if not tags:
            tags = name[6]  # default to 'healthy'
        pred_strings.append(tags)

    test_df["labels"] = pred_strings
    submission_path = "submission.csv"
    test_df.to_csv(submission_path, index=False)
    print(f"Submission file written to {submission_path}")
