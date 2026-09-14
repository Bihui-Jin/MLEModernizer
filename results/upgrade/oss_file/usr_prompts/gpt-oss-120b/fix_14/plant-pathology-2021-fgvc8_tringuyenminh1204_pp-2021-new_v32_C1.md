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
import numpy as np
import pandas as pd
import os
import re, math, random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    pass

print(tf.__version__)
print(tf.keras.__version__)



## === cell 1
import pathlib



## === cell 2
USE_FP16 = tf.keras.mixed_precision.global_policy().compute_dtype == tf.float16


def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float16 if USE_FP16 else tf.float32)
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 128



## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"
IMAGE_PATHS = [
    str(p)
    for p in pathlib.Path(source).glob("*")
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
]



## === cell 5
IMAGE_PATHS[:5]



## === cell 6
AUTO = tf.data.experimental.AUTOTUNE



## === cell 7
options = tf.data.Options()
options.experimental_deterministic = False

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .map(
        decode_image, num_parallel_calls=AUTO, deterministic=False
    )  # parallel, non‑deterministic decode
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    .with_options(options)
)




## === cell 8
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 9
model_path = "../input/smresnet50/SMResNet50.h5"
try:
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    print("Custom model loaded.")
except Exception as e:
    print(f"Custom model not loaded ({e}); building fallback model.")
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(512, 512, 3), pooling="avg"
    )
    outputs = Dense(5, activation="sigmoid")(base.output)
    model = Model(inputs=base.input, outputs=outputs)



## === cell 10
probs = model.predict(test_dataset, verbose=0)



## === cell 11
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}
threshold = np.array([0.27, 0.5, 0.3, 0.5, 0.5], dtype=np.float32)

probs_arr = np.asarray(probs)
above_thr = probs_arr > threshold  # shape (N,5)


def build_label(row_mask, row_probs):
    selected = [name[i] for i, flag in enumerate(row_mask) if flag]
    if len(selected) >= 2:
        if "complex" in selected:
            if all(row_mask) == False:  # not needed but keeps logic
                selected = [lbl for lbl in selected if lbl != "complex"]
        else:
            selected.append("complex")
    if not selected:
        selected = [name[6]]
    return " ".join(selected)


pred_string = [build_label(mask, prob) for mask, prob in zip(above_thr, probs_arr)]



## === cell 12
image_names = [os.path.basename(p) for p in IMAGE_PATHS]
df = pd.DataFrame({"image": image_names, "labels": pred_string})
print(df.head())



## === cell 13
submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
