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
import os, random, re, math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

print("tf:", tf.__version__)
print("keras:", tf.keras.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_experimental_options({"layout_optimizer": True})
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
sub = pd.read_csv(path + "sample_submission.csv")

train.head(), sub.head()




## === cell 2
AUTO = tf.data.AUTOTUNE
CPU_COUNT = os.cpu_count() or 4
MAP_PARALLEL = max(2, min(16, CPU_COUNT))




## === cell 3
img_path = os.path.join(path, "train_images", train.iloc[0]["image"])
print("Example image path:", img_path)




## === cell 4
import pathlib




## === cell 5
train_img_dir = os.path.join(path, "train_images")
test_img_dir = os.path.join(path, "test_images")

train_paths = [
    os.path.join(train_img_dir, img_id) for img_id in train["image"].tolist()
]
test_paths = [os.path.join(test_img_dir, img_id) for img_id in sub["image"].tolist()]

assert len(train_paths) == len(train)
assert len(test_paths) == len(sub)
assert os.path.exists(train_paths[0]), f"Missing train image: {train_paths[0]}"
assert os.path.exists(test_paths[0]), f"Missing test image: {test_paths[0]}"




## === cell 6
all_classes = sorted(
    {c for s in train["labels"].astype(str).tolist() for c in s.split() if c}
)
print("num_classes:", len(all_classes))
print(all_classes)




## === cell 7
primary_labels = train["labels"].astype(str).str.split().str[0]
label_to_idx = {c: i for i, c in enumerate(all_classes)}
y = primary_labels.map(label_to_idx).astype(int).values

new_train = pd.DataFrame(
    {"image": train["image"].values, "primary_label": primary_labels.values, "y": y}
)
new_train.head()




## === cell 8
new_train.describe(include="all")




## === cell 9
def decode_image_only(filename, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image.set_shape((image_size[0], image_size[1], 3))
    return image


def decode_image_with_label(filename, label, image_size=(224, 224)):
    image = decode_image_only(filename, image_size)
    return image, label


def _map_train(p, lab):
    return decode_image_with_label(p, lab, IMAGE_SIZE)


def _map_test(p):
    return decode_image_only(p, IMAGE_SIZE)




## === cell 10
print("First 5 test paths:", test_paths[:5])




## === cell 11
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)




## === cell 12
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_slack = True
except Exception:
    pass

USE_DISK_CACHE = True
CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)




## === cell 13
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)

test_ds = test_ds.map(
    _map_test,
    num_parallel_calls=MAP_PARALLEL,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())

test_dataset = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)




## === cell 14
idx = np.arange(len(train_paths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_paths = np.array(train_paths)[tr_idx]
va_paths = np.array(train_paths)[va_idx]
tr_y = y[tr_idx].astype(np.int32, copy=False)
va_y = y[va_idx].astype(np.int32, copy=False)

train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_y)).with_options(options)
val_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_y)).with_options(options)

train_ds = train_ds.map(
    _map_train,
    num_parallel_calls=MAP_PARALLEL,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())

val_ds = val_ds.map(
    _map_train,
    num_parallel_calls=MAP_PARALLEL,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())

if USE_DISK_CACHE:
    train_ds = train_ds.cache(_cache_path(f"train_img_224_n{len(tr_paths)}.cache"))
    val_ds = val_ds.cache(_cache_path(f"val_img_224_n{len(va_paths)}.cache"))
else:
    val_ds = val_ds.cache()

train_dataset = (
    train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .repeat()
    .prefetch(AUTO)
)

val_dataset = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)




## === cell 15
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"swish": keras.layers.Activation(tf.nn.swish)})




## === cell 16
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




## === cell 17
num_classes = len(all_classes)

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = inputs
x = base(x, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()




## === cell 18
EPOCHS = 2

steps_per_epoch = len(tr_paths) // BATCH_SIZE  # drop_remainder=True
validation_steps = int(math.ceil(len(va_paths) / BATCH_SIZE))  # drop_remainder=False

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 19
probs = model.predict(test_dataset, verbose=1)
probs.shape




## === cell 20
result = np.argmax(probs, axis=1)
result.shape, result[:10]




## === cell 21
result_labels = np.array(all_classes, dtype=str)
pred_strings = result_labels[result]
pred_strings[:10], len(pred_strings)




## === cell 22
assert len(pred_strings) == len(
    sub
), f"Pred length {len(pred_strings)} != sub length {len(sub)}"
sub_out = sub.copy()
sub_out["labels"] = pred_strings

sub_out.to_csv("submission.csv", index=False)
sub_out.head()




## === cell 23
print("Wrote submission.csv with shape:", sub_out.shape)
print("Columns:", sub_out.columns.tolist())
print(sub_out["labels"].value_counts().head(10))
