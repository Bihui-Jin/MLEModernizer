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

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import re, math, random, pathlib
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
BASE_INPUT = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print(train_df.shape, sample_df.shape)
print(train_df.columns.tolist(), sample_df.columns.tolist())




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size)
    if label is None:
        return img
    else:
        return img, label




## === cell 3
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)

AUTO = tf.data.AUTOTUNE




## === cell 4
def list_image_files(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(tf.io.gfile.glob(os.path.join(folder, ext)))
    paths = sorted(paths)
    files = [os.path.basename(p) for p in paths]
    return files


test_files = list_image_files(TEST_DIR)
IMAGE_PATHS = [os.path.join(TEST_DIR, f) for f in test_files]

print("Num test images:", len(IMAGE_PATHS))
print("First 3:", test_files[:3])




## === cell 5
test_options = tf.data.Options()
test_options.deterministic = False

TEST_CACHE = os.path.join("/kaggle/working", "test_cache_512x512")

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(test_options)
    .map(
        lambda x: decode_image(x, label=None, image_size=IMAGE_SIZE),
        num_parallel_calls=AUTO,
    )
    .cache(TEST_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 6
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




## === cell 7
MODEL_PATH = "../input/2smresnet50/2SMResNet50.h5"

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    print("Loaded model from:", MODEL_PATH)
else:
    print("Pretrained model not found at:", MODEL_PATH)
    print("Training a fallback model to ensure a valid submission is produced...")

    CLASSES = [
        "scab",
        "frog_eye_leaf_spot",
        "complex",
        "rust",
        "powdery_mildew",
        "healthy",
    ]
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def labels_to_vec(label_str):
        vec = np.zeros(len(CLASSES), dtype=np.float32)
        for lab in str(label_str).split():
            if lab in class_to_idx:
                vec[class_to_idx[lab]] = 1.0
        return vec

    train_files = train_df["image"].astype(str).tolist()
    train_paths = [os.path.join(TRAIN_DIR, f) for f in train_files]
    y = np.stack(
        [labels_to_vec(s) for s in train_df["labels"].astype(str).tolist()], axis=0
    )

    idx = np.arange(len(train_paths))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_paths = np.array(train_paths, dtype=object)[tr_idx].tolist()
    va_paths = np.array(train_paths, dtype=object)[va_idx].tolist()
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_options = tf.data.Options()
    train_options.deterministic = True

    train_ds = (
        tf.data.Dataset.from_tensor_slices((tr_paths, y_tr))
        .with_options(train_options)
        .map(
            lambda x, yb: decode_image(x, yb, image_size=IMAGE_SIZE),
            num_parallel_calls=AUTO,
        )
        .cache()
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((va_paths, y_va))
        .with_options(train_options)
        .map(
            lambda x, yb: decode_image(x, yb, image_size=IMAGE_SIZE),
            num_parallel_calls=AUTO,
        )
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=Input(shape=(*IMAGE_SIZE, 3)),
    )
    x = base.output
    x = GlobalAveragePooling2D()(x)
    out = Dense(len(CLASSES), activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy"
    )
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)




## === cell 8
probs = model.predict(test_dataset, verbose=1)
temp_probs = np.asarray(probs)
print("probs shape:", temp_probs.shape)




## === cell 9
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
threshold2 = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}

temp_probs = np.asarray(temp_probs)
p5 = temp_probs[:, :5]

thr = np.array([threshold[i] for i in range(5)], dtype=p5.dtype)
thr2 = np.array([threshold2[i] for i in range(5)], dtype=p5.dtype)

mask_thr = p5 > thr[None, :]
mask_thr2 = p5 > thr2[None, :]
count2 = mask_thr2.sum(axis=1)

labels_0_4 = []
names_0_4 = [name[i] for i in range(5)]
for row_mask in mask_thr:
    labs = [names_0_4[i] for i in range(5) if bool(row_mask[i])]
    labels_0_4.append(labs)

complex_idx = 2
append_complex = (count2 >= 2) & (~mask_thr[:, complex_idx])

pred_string = []
n = temp_probs.shape[0]
for i in range(n):
    labs = labels_0_4[i]
    if bool(append_complex[i]):
        labs = labs + ["complex"]
    if len(labs) == 0:
        labs = ["healthy"]
    pred_string.append(" ".join(labs))

print("Num predictions:", len(pred_string))




## === cell 10
if "image" in sample_df.columns and len(sample_df) == len(test_files):
    out_images = sample_df["image"].astype(str).tolist()
else:
    out_images = test_files

assert len(out_images) == len(pred_string), (len(out_images), len(pred_string))

submission = pd.DataFrame({"image": out_images, "labels": pred_string})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
