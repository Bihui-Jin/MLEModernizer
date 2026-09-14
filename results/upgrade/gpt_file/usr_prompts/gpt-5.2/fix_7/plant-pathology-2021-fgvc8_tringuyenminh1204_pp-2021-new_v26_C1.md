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
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf

print("TF:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_img_dir = os.path.join(path, "train_images")
test_img_dir = os.path.join(path, "test_images")

print(train.shape, sub.shape)
train.head()



## === cell 2
AUTO = tf.data.AUTOTUNE



## === cell 3
from matplotlib import pyplot as plt

sample_path = os.path.join(train_img_dir, "800113bb65efe69e.jpg")
if os.path.exists(sample_path):
    img = plt.imread(sample_path)
    print(img.shape)
    plt.imshow(img)
    plt.axis("off")



## === cell 4
import pathlib



## === cell 5
train_paths = [os.path.join(train_img_dir, img) for img in train["image"].values]
test_paths = [os.path.join(test_img_dir, img) for img in sub["image"].values]

missing_train = sum(0 if os.path.exists(p) else 1 for p in train_paths[:100])
missing_test = sum(0 if os.path.exists(p) else 1 for p in test_paths[:100])
print("Missing (first 100) train/test:", missing_train, missing_test)



## === cell 6
ALL_CLASSES = [
    "scab",
    "frog_eye_leaf_spot",
    "rust",
    "complex",
    "powdery_mildew",
    "healthy",
]
DISEASE_CLASSES_5 = [
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
]  # used by your thresholding loop


def split_labels(s):
    if pd.isna(s) or s.strip() == "":
        return []
    return s.split()


labels_list = train["labels"].fillna("").astype(str).str.split().to_list()
y = np.zeros((len(labels_list), len(DISEASE_CLASSES_5)), dtype=np.float32)
class_to_idx = {c: j for j, c in enumerate(DISEASE_CLASSES_5)}
for i, labs in enumerate(labels_list):
    for lab in labs:
        j = class_to_idx.get(lab)
        if j is not None:
            y[i, j] = 1.0

print("y shape:", y.shape, "positive rates:", y.mean(axis=0))



## === cell 7
new_train = pd.concat(
    [train[["image"]], pd.DataFrame(y, columns=DISEASE_CLASSES_5)], axis=1
)
new_train.head()



## === cell 8
new_train



## === cell 9
BATCH_SIZE = 32  # keep as provided
IMG_SIZE = (512, 512)


@tf.function(reduce_retracing=True)
def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(
        bits, channels=3, dct_method="INTEGER_FAST", fancy_upscaling=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image,
        IMG_SIZE,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
        preserve_aspect_ratio=False,
    )
    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_paths[:5]



## === cell 11
test_options = tf.data.Options()
test_options.experimental_deterministic = True
test_options.experimental_optimization.apply_default_optimizations = True
test_options.experimental_optimization.map_parallelization = True
test_options.experimental_optimization.parallel_batch = True

test_cache_path = os.path.join("/kaggle/working", "cache_test_ds")
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(test_options)
    .map(lambda x: decode_image(x, None), num_parallel_calls=AUTO, deterministic=True)
    .cache(test_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 12
from tensorflow import keras



## === cell 13
idx = np.arange(len(train_paths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)

val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_paths = [train_paths[i] for i in trn_idx]
val_paths = [train_paths[i] for i in val_idx]
y_trn = y[trn_idx]
y_val = y[val_idx]

print("Train/Val:", len(trn_paths), len(val_paths))

train_options = tf.data.Options()
train_options.experimental_deterministic = False
train_options.experimental_optimization.apply_default_optimizations = True
train_options.experimental_optimization.map_parallelization = True
train_options.experimental_optimization.parallel_batch = True

val_options = tf.data.Options()
val_options.experimental_deterministic = True
val_options.experimental_optimization.apply_default_optimizations = True
val_options.experimental_optimization.map_parallelization = True
val_options.experimental_optimization.parallel_batch = True

train_cache_path = os.path.join("/kaggle/working", "cache_train_ds")
val_cache_path = os.path.join("/kaggle/working", "cache_val_ds")

train_ds = (
    tf.data.Dataset.from_tensor_slices((trn_paths, y_trn))
    .with_options(train_options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(
        lambda x, yy: decode_image(x, yy), num_parallel_calls=AUTO, deterministic=False
    )
    .cache(train_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, y_val))
    .with_options(val_options)
    .map(lambda x, yy: decode_image(x, yy), num_parallel_calls=AUTO, deterministic=True)
    .cache(val_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

try:
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        train_ds = train_ds.apply(
            tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
        )
        val_ds = val_ds.apply(
            tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
        )
        test_dataset = test_dataset.apply(
            tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTO)
        )
except Exception:
    pass



## === cell 14
"""from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'swish': keras.layers.Activation(tf.nn.swish)})"""



## === cell 15
"""class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
"""



## === cell 16
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False  # keep unchanged

inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inp * 255.0)
x = base(x, training=False)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(len(DISEASE_CLASSES_5), activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=10,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1,
)



## === cell 17
probs = model.predict(test_dataset, verbose=1, steps=None)



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
}

threshold = {
    0: 0.10,
    1: 0.15,
    2: 0.15,
    3: 0.15,
    4: 0.15,
}

thr = np.array([threshold[i] for i in range(5)], dtype=temp_probs.dtype)
pred_mask = temp_probs > thr  # (N,5) boolean

count = pred_mask.sum(axis=1)
need_add_complex = (count >= 2) & (~pred_mask[:, 2])
pred_mask2 = pred_mask.copy()
pred_mask2[need_add_complex, 2] = True

labels_by_idx = np.array([name[i] for i in range(5)], dtype=object)

rows_any = pred_mask2.any(axis=1)
pred_string = np.empty(pred_mask2.shape[0], dtype=object)
pred_string[~rows_any] = "healthy"
pred_string[rows_any] = [" ".join(labels_by_idx[row]) for row in pred_mask2[rows_any]]

submission = sub.copy()
submission["labels"] = pred_string

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
