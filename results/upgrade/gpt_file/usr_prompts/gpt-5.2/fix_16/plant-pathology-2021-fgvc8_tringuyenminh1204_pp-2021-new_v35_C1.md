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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

import re, math, random
import numpy as np
import pandas as pd

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as _e:
    pass

import tensorflow as tf
import tensorflow.keras.backend as K

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("enable_op_determinism failed (ignored):", repr(e))

try:
    import multiprocessing as _mp

    _cpu = _mp.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(4, _cpu))
except Exception as e:
    print("thread config failed (ignored):", repr(e))

try:
    gpus = tf.config.list_physical_devices("GPU")
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g, True)
    if gpus:
        print("GPUs:", gpus)
except Exception as e:
    print("GPU memory growth config failed (ignored):", repr(e))




## === cell 1
import pathlib




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32




## === cell 4
candidate_sources = [
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images",
    "../input/plant-pathology-2021-fgvc8/test_images",
    "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images",
]
source = None
for s in candidate_sources:
    if os.path.isdir(s):
        source = s
        break
if source is None:
    raise FileNotFoundError(
        f"Could not find test_images in any of: {candidate_sources}"
    )

valid_ext = (".jpg", ".jpeg", ".png")
patterns = [os.path.join(source, "**", f"*{ext}") for ext in valid_ext]
IMAGE_PATHS = []
for pat in patterns:
    IMAGE_PATHS.extend(tf.io.gfile.glob(pat))
IMAGE_PATHS = sorted(set(IMAGE_PATHS))

TEST_IMAGE_IDS = [os.path.basename(pp) for pp in IMAGE_PATHS]

print("Using test_images:", source)
print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", TEST_IMAGE_IDS[:3])




## === cell 5
IMAGE_PATHS[:5]




## === cell 6
AUTO = tf.data.AUTOTUNE




## === cell 7
_opts = tf.data.Options()
_opts.experimental_deterministic = True
_opts.experimental_optimization.apply_default_optimizations = True
_opts.experimental_optimization.map_parallelization = True

test_dataset = tf.data.Dataset.from_tensor_slices(IMAGE_PATHS).with_options(_opts)
test_dataset = test_dataset.map(
    decode_image, num_parallel_calls=AUTO, deterministic=True
)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.cache()
test_dataset = test_dataset.prefetch(AUTO)




## === cell 8
from tensorflow import keras




## === cell 9
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




## === cell 10
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.resnet.preprocess_input(inputs * 255.0)

backbone = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling="avg",
)
x = backbone(x, training=False)

x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(5, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


def _try_load_weights(m):
    candidates = [
        "/kaggle/input/model/model.h5",
        "/kaggle/input/model/best.h5",
        "/kaggle/input/model/weights.h5",
        "/kaggle/input/model/model.weights.h5",
        "/kaggle/input/model/best.weights.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/model.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/best.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/weights.h5",
        "/kaggle/working/model.h5",
        "/kaggle/working/best.h5",
        "/kaggle/working/weights.h5",
        "./model.h5",
        "./best.h5",
        "./weights.h5",
        "./model.weights.h5",
        "./best.weights.h5",
    ]
    for path in candidates:
        if os.path.isfile(path):
            print("Loading model weights from:", path)
            m.load_weights(path)
            return path
    return None


loaded_from = _try_load_weights(model)
if loaded_from is None:
    print(
        "No trained weights found. Will train the existing architecture on train.csv to move score toward target."
    )
else:
    print("Using provided trained weights:", loaded_from)

print(model.output_shape)




## === cell 11
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}
idx_to_name = [name[i] for i in range(5)]

train_csv_candidates = [
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "../input/train.csv",
]
train_csv_path = None
for tp in train_csv_candidates:
    if os.path.isfile(tp):
        train_csv_path = tp
        break
if train_csv_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv in any of: {train_csv_candidates}"
    )

train_img_candidates = [
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/train_images",
    "../input/plant-pathology-2021-fgvc8/train_images",
    "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/train_images",
]
train_img_dir = None
for d in train_img_candidates:
    if os.path.isdir(d):
        train_img_dir = d
        break
if train_img_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images in any of: {train_img_candidates}"
    )

tr = pd.read_csv(train_csv_path)
tr["image"] = tr["image"].astype(str)
tr["labels"] = tr["labels"].fillna("").astype(str)

_labels_norm = " " + tr["labels"].str.replace(r"\s+", " ", regex=True).str.strip() + " "
y_cols = []
for cls in idx_to_name:
    y_cols.append(
        _labels_norm.str.contains(f" {re.escape(cls)} ", regex=True).to_numpy(
            np.float32
        )
    )
y_all = np.stack(y_cols, axis=1).astype(np.float32)

x_paths_all = [os.path.join(train_img_dir, fn) for fn in tr["image"].tolist()]

n = len(x_paths_all)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.10
n_val = int(round(n * val_frac))
val_idx = idx[:n_val]
train_idx = idx[n_val:]

x_train = [x_paths_all[i] for i in train_idx]
y_train = y_all[train_idx]
x_val = [x_paths_all[i] for i in val_idx]
y_val = y_all[val_idx]

print("Train size:", len(x_train), "Val size:", len(x_val))

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).with_options(_opts)
train_ds = train_ds.map(
    lambda f, y: decode_image(f, y), num_parallel_calls=AUTO, deterministic=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).with_options(_opts)
val_ds = val_ds.map(
    lambda f, y: decode_image(f, y), num_parallel_calls=AUTO, deterministic=True
)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTO)




## === cell 12
default_threshold = {0: 0.5, 1: 0.4, 2: 0.35, 3: 0.35, 4: 0.6}
threshold = dict(default_threshold)

if loaded_from is None:
    backbone.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    EPOCHS_HEAD = 3
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2)

    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
    )
    EPOCHS_FT = 1
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, verbose=2)

    val_probs = model.predict(val_ds, verbose=0)
    yv = y_val.astype(np.int32)

    def best_f1_threshold(y_true, p, grid):
        y_true = y_true.astype(np.int32, copy=False)
        p = p.astype(np.float32, copy=False)
        grid = grid.astype(np.float32, copy=False)

        pred = (p[None, :] > grid[:, None]).astype(np.int32)  # (T, N)
        yt = y_true[None, :]  # (1, N)

        tp = np.sum((pred == 1) & (yt == 1), axis=1).astype(np.int64)
        fp = np.sum((pred == 1) & (yt == 0), axis=1).astype(np.int64)
        fn = np.sum((pred == 0) & (yt == 1), axis=1).astype(np.int64)

        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, (2.0 * tp) / denom, 0.0).astype(np.float32)

        best_k = int(np.argmax(f1))
        return float(grid[best_k]), float(f1[best_k])

    grid = np.linspace(0.05, 0.95, 37, dtype=np.float32)
    for i in range(5):
        thr_i, f1_i = best_f1_threshold(yv[:, i], val_probs[:, i], grid)
        threshold[i] = thr_i
    print("Using validation-tuned thresholds (trained in-notebook):", threshold)
else:
    print("Using default thresholds (weights loaded):", threshold)




## === cell 13
model.trainable = False
model.compile(run_eagerly=False)

probs = model.predict(test_dataset, verbose=1)
temp_probs = probs

print("probs shape:", probs.shape)




## === cell 14
thr = np.array([threshold[i] for i in range(5)], dtype=temp_probs.dtype)
above = temp_probs > thr  # shape (N, 5)

complex_idx = 2  # fixed by mapping above

label_names = np.array(idx_to_name, dtype=object)
pred_string = []
for row_mask in above:
    selected = label_names[row_mask].tolist()
    if len(selected) >= 2 and (not row_mask[complex_idx]):
        selected.append("complex")
    pred_string.append(name[6] if not selected else " ".join(selected))

print("Num predictions:", len(pred_string))
print(pred_string[:5])




## === cell 15
pred_string[:10]




## === cell 16
sub_candidates = [
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/plant-pathology-2021-fgvcvc8/sample_submission.csv".replace(
        "fgvcvc8", "fgvc8"
    ),
    "../input/sample_submission.csv",
]
sub_path = None
for sp in sub_candidates:
    if os.path.isfile(sp):
        sub_path = sp
        break

pred_map = dict(zip(TEST_IMAGE_IDS, pred_string))

if sub_path is not None:
    sub_df = pd.read_csv(sub_path)
    df = pd.DataFrame(
        {"image": sub_df["image"].astype(str), "labels": sub_df["image"].map(pred_map)}
    )
    df["labels"] = df["labels"].fillna(name[6])
else:
    df = pd.DataFrame({"image": TEST_IMAGE_IDS, "labels": pred_string})

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Submission path:", os.path.abspath("submission.csv"))
