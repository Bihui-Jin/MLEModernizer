# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.4426223453370249

# 6. Current score

0.60078

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The timeout is dominated by (1) building and iterating the `tf.data` pipelines with disk-cache files, plus the extra `take(1)`/printing that forces an additional full pipeline instantiation, and (2) slower input throughput due to non-fused JPEG decode/resize and conservative dataset options. I keep the exact same preprocessing, model, training (2 epochs), and prediction semantics, but speed up I/O by switching to `tf.image.decode_and_crop_jpeg`-equivalent fast path (`tf.io.decode_jpeg` remains), enabling deterministic-safe parallelism, removing on-disk caching (which is slower than streaming here and adds filesystem overhead), and ensuring the dataset is built once without extra warmup iterations. I also avoid expensive Python-side list creation where possible and reduce redundant work/objects, while preserving identical outputs up to negligible float differences.'
- What this solution (achieved 0.60024) has done: 'The timeout is dominated by image input throughput (JPEG decode + resize + preprocess) and by `model.predict` over the full test set; the model itself is small/frozen so we focus on making the `tf.data` pipeline and batching as efficient as possible without changing the model or training semantics. I remove retracing overhead by giving the decode function a fixed input signature, enable TF Data autotuned threading, cache the small training subset in-memory to avoid re-decoding across epochs, and set dataset options to avoid unnecessary determinism constraints while keeping seeds as-is. I also increase test batch size adaptively for GPU/CPU to reduce Python/TF call overhead while preserving identical predictions (only negligible float-order differences). All paths, preprocessing, architecture, loss, and training loop (2 epochs) remain unchanged.'
- What this solution (achieved 0.60031) has done: 'I fix the runtime crash occurring at import-time (the `MessageFactory.GetPrototype` protobuf issue) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which is the standard Kaggle-safe workaround and is score-neutral. I also make the path selection more robust (fall back across the provided dataset locations) to avoid file-not-found issues without changing any modeling logic. Finally, I keep the exact same model/training/prediction logic and only add small safety checks so the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.60078) has done: 'I fix the import-time protobuf crash by applying a robust, Kaggle-safe protobuf/TensorFlow compatibility workaround before importing TensorFlow (and fall back to the pure-Python protobuf implementation if needed). I also ensure the environment variable setup happens before any TensorFlow/Keras import to prevent the `MessageFactory.GetPrototype` error from occurring. These changes are score-neutral (model, preprocessing, training, and thresholding remain identical) and are only to make the notebook run end-to-end and reliably write `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

import glob
import gc
import random
import numpy as np
import pandas as pd

try:
    import google.protobuf as _pb  # noqa: F401
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype") and hasattr(
        _mf.MessageFactory, "GetMessageClass"
    ):
        _mf.MessageFactory.GetPrototype = _mf.MessageFactory.GetMessageClass  # type: ignore[attr-defined]
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.applications.resnet50 as resnet

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

_CANDIDATE_DATA_DIRS = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../data/plant-pathology-2021-fgvc8",
]
DATA_DIR = None
for d in _CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "sample_submission.csv")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = _CANDIDATE_DATA_DIRS[0]

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("Using DATA_DIR:", DATA_DIR)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))



## === cell 2
imglist_test = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
imglist_test = sorted(imglist_test)
len(imglist_test)




## === cell 3
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return img


options = tf.data.Options()
options.experimental_deterministic = False
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass

TEST_BATCH = 256

test_files_ds = tf.data.Dataset.from_tensor_slices(imglist_test).with_options(options)

test_ds = (
    test_files_ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    .batch(TEST_BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 4
print(
    "Num test images:",
    len(imglist_test),
    "Example path:",
    imglist_test[0] if imglist_test else None,
)



## === cell 5
training_csv = pd.read_csv(TRAIN_CSV)

tagnames = np.sort(
    pd.Series(training_csv["labels"].astype(str).str.split())
    .explode()
    .dropna()
    .unique()
)
NUM_CLASSES = len(tagnames)

print("Num classes:", NUM_CLASSES)
print("Classes:", tagnames)



## === cell 6
tag2idx = {t: i for i, t in enumerate(tagnames)}

labels_series = training_csv["labels"].astype(str).str.get_dummies(sep=" ")
y = labels_series.reindex(columns=tagnames, fill_value=0).to_numpy(
    dtype=np.float32, copy=False
)

MAX_TRAIN = 2048
train_df = training_csv.iloc[:MAX_TRAIN].reset_index(drop=True)
y_train = y[:MAX_TRAIN]

train_paths = np.array(
    [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].astype(str).values],
    dtype=str,
)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
    ]
)
def _train_map(p, yy):
    return _decode_resize_preprocess(p), yy


TRAIN_BATCH = 32

train_base_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train)).with_options(
    options
)

train_ds = (
    train_base_ds.map(_train_map, num_parallel_calls=AUTOTUNE, deterministic=False)
    .cache()
    .batch(TRAIN_BATCH, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

print("Train size:", len(train_df), "Train y:", y_train.shape)



## === cell 7
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
)
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model_f = keras.Model(inputs, outputs)

base.trainable = False

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model_f.fit(
    train_ds,
    epochs=2,
    verbose=1,
)

X_test = model_f.predict(test_ds, verbose=1)
print(X_test.shape)



## === cell 8
print(
    "Pred stats: min/mean/max =",
    float(X_test.min()),
    float(X_test.mean()),
    float(X_test.max()),
)




## === cell 9
def class2tags(classes, tagnames, fallback_label="healthy"):
    tagnames = np.asarray(tagnames)
    idxs = [np.flatnonzero(r) for r in classes]
    out = []
    for i in idxs:
        if i.size == 0:
            out.append(fallback_label)
        else:
            out.append(" ".join(tagnames[i]))
    return out


test_predclass = X_test > 0.2
test_predtags = class2tags(test_predclass, tagnames, fallback_label="healthy")



## === cell 10
del test_predclass
gc.collect()



## === cell 11
sub = pd.read_csv(SAMPLE_SUB)

pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}
sub["labels"] = sub["image"].map(pred_map).fillna("")

sub = sub[["image", "labels"]]
sub.to_csv("./submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print(
    "Unique label strings (first 10):", sub["labels"].value_counts().head(10).to_dict()
)
