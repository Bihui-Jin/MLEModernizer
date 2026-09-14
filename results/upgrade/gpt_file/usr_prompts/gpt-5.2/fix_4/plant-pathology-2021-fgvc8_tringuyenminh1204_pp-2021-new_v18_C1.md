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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import random, re, math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

print(tf.__version__)
print(tf.keras.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.config.experimental.enable_op_determinism()




## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
if not os.path.exists(path):
    path = "/kaggle/input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "sample_submission.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train.head(), test.head()




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE




## === cell 3
pass




## === cell 4
import pathlib




## === cell 5
train_paths = []

test_paths = tf.io.gfile.glob(os.path.join(path, "test_images", "*.jpg"))
test_paths = sorted(test_paths)

len(train_paths), len(test_paths), test_paths[:3]




## === cell 6
ALL_CLASSES = [
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
    "healthy",
]
ALL_CLASSES




## === cell 7
def encode_labels(label_str: str):
    items = str(label_str).split()
    return [1.0 if c in items else 0.0 for c in ALL_CLASSES]


labels_mat = np.array(
    [encode_labels(s) for s in train["labels"].values], dtype=np.float32
)
new_train = pd.concat(
    [train[["image"]], pd.DataFrame(labels_mat, columns=ALL_CLASSES)], axis=1
)

new_train.head()




## === cell 8
new_train




## === cell 9
IMG_SIZE = (512, 512)


@tf.function
def decode_image(filename, label=None, image_size=IMG_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_paths[:5]




## === cell 11
BATCH_SIZE = 64




## === cell 12
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(
        (
            lambda: (
                setattr((opt := tf.data.Options()), "experimental_deterministic", True)
                or opt
            )
        )()
    )
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 13
import numpy as np
import tensorflow as tf
from tensorflow import keras




## === cell 14
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"swish": keras.layers.Activation(tf.nn.swish)})




## === cell 15
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




## === cell 16
base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = base.output
out = Dense(len(ALL_CLASSES), activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=out)

model.summary()




## === cell 17
probs = model.predict(test_dataset, verbose=1)
probs.shape




## === cell 18
probs.shape




## === cell 19
probs[:2]




## === cell 20
temp_probs = probs




## === cell 21
temp_probs[:2]




## === cell 22
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.25, 1: 0.50, 2: 0.30, 3: 0.50, 4: 0.50, 5: 0.50}

test_files = [os.path.basename(p) for p in test_paths]
pred_df = pd.DataFrame(temp_probs, columns=[name[i] for i in range(len(ALL_CLASSES))])
pred_df["image"] = test_files

out_df = sub[["image"]].merge(pred_df, on="image", how="left")
out_df[[name[i] for i in range(len(ALL_CLASSES))]] = out_df[
    [name[i] for i in range(len(ALL_CLASSES))]
].fillna(0.0)

prob_cols = [name[i] for i in range(len(ALL_CLASSES))]
probs_np = out_df[prob_cols].to_numpy(dtype=np.float32, copy=False)
thr_np = np.array([threshold[i] for i in range(len(ALL_CLASSES))], dtype=np.float32)

mask = probs_np > thr_np[None, :]

pred_string = []
for r in range(mask.shape[0]):
    idx = np.flatnonzero(mask[r])
    if idx.size == 0:
        pred_string.append("healthy")
    else:
        pred_string.append(" ".join([prob_cols[i] for i in idx]))

out_sub = out_df[["image"]].copy()
out_sub["labels"] = pred_string

out_sub.to_csv("submission.csv", index=False)
out_sub.head()
