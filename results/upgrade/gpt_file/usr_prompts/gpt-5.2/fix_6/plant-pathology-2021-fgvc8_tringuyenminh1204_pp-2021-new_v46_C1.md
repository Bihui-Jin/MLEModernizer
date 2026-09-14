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
import os, re, math, random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import tensorflow as tf
import tensorflow.keras.backend as K

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)



## === cell 1
import pathlib

CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8",
    "../data/plant-pathology-2021-fgvc8",
]
DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError("Could not find dataset root under expected Kaggle paths.")

source = os.path.join(DATA_ROOT, "test_images")
if not os.path.isdir(source):
    raise FileNotFoundError(f"test_images directory not found at: {source}")

sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    sample_sub_path = "/kaggle/input/sample_submission.csv"

print("DATA_ROOT:", DATA_ROOT)
print("test_images:", source)
print("sample_submission.csv:", sample_sub_path)




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)

    image = tf.io.decode_image(bits, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size, method="bilinear", antialias=False)

    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
img_re = re.compile(
    r"([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$", flags=re.IGNORECASE
)

IMAGE_FILES = sorted([f for f in os.listdir(source) if img_re.search(f)])

if len(IMAGE_FILES) == 0:
    raise RuntimeError(f"No image files found in {source}")

IMAGE_PATHS = [os.path.join(source, f) for f in IMAGE_FILES]
print("Num test images:", len(IMAGE_PATHS))
print("First 3:", IMAGE_FILES[:3])



## === cell 5
_ = IMAGE_PATHS[:5]



## === cell 6
AUTO = tf.data.experimental.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True  # preserve determinism

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 7
from tensorflow import keras
from tensorflow.keras import layers


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




## === cell 8
IMG_SIZE = (512, 512)

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
base.trainable = False

x = base.output
x = layers.Dense(256, activation="relu")(x)
x = FixedDropout(0.2)(x)
outputs = layers.Dense(6, activation="sigmoid")(x)  # multi-label style probabilities

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()



## === cell 9
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs

print("probs shape:", probs.shape)
print("probs min/max:", float(probs.min()), float(probs.max()))



## === cell 10
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
threshold2 = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}


def get_key(val):
    for key, value in name.items():
        if val == value:
            return key
    return "key doesn't exist"


thr = np.array([threshold[i] for i in range(6)], dtype=np.float32)
thr2 = np.array([threshold2[i] for i in range(6)], dtype=np.float32)

above_thr = temp_probs > thr[None, :]
above_thr2 = temp_probs > thr2[None, :]

count2 = above_thr2.sum(axis=1)
complex_idx = get_key("complex")
has_complex = above_thr[:, complex_idx]
need_add_complex = (count2 >= 2) & (~has_complex)

label_names = [name[i] for i in range(6)]
pred_string = []
for j in range(temp_probs.shape[0]):
    parts = [label_names[i] for i in range(6) if above_thr[j, i]]
    if need_add_complex[j]:
        parts.append("complex")
    if len(parts) == 0:
        parts = [name[5]]
    pred_string.append(" ".join(parts))

print("Num predictions:", len(pred_string))
print("First 5 preds:", pred_string[:5])



## === cell 11
sample_sub = pd.read_csv(sample_sub_path)
expected_cols = list(sample_sub.columns)
if expected_cols != ["image", "labels"]:
    print("Warning: unexpected sample_submission columns:", expected_cols)

pred_map = dict(zip(IMAGE_FILES, pred_string))
ordered_images = sample_sub["image"].tolist()

missing = [im for im in ordered_images if im not in pred_map]
if len(missing) > 0:
    raise RuntimeError(
        f"Missing predictions for {len(missing)} images. Example: {missing[:3]}"
    )

sub = pd.DataFrame(
    {
        "image": ordered_images,
        "labels": [pred_map[im] for im in ordered_images],
    }
)

sub = sub[["image", "labels"]]
assert len(sub) == len(sample_sub)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
print(sub.tail())
