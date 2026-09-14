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

0.1578947368421052

# 6. Current score

0.28627

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.28627) has done: 'Your code likely fails to yield a Kaggle score because the training generator uses a “dummy index label” (`__idx__`) but `flow_from_dataframe(..., class_mode="raw")` returns that column as float values (and can be out-of-sync when shuffled), which can lead to incorrect/invalid indexing into `y_array` (or silently wrong labels), producing a broken model and sometimes runtime errors. I make the smallest change that preserves your architecture/training loop: store the true global row index in `__idx__`, make it an integer label via `class_mode="sparse"`, and disable shuffling at the Keras generator level while keeping randomness via the DataFrame shuffle. This keeps the same core logic (same CNN, same loss, same epochs) but fixes label/target alignment so training is meaningful and a valid submission is reliably created. I also keep the submission writing exactly as required (`submission.csv` with `image,labels`).'
- What this solution (achieved 0.28627) has done: 'The timeout is dominated by slow Python-based image augmentation/loading in `ImageDataGenerator.flow_from_dataframe` plus per-batch Python `next()`/`yield` overhead in the wrapper; this keeps the GPU/CPU underutilized and burns wall time. I keep the same train/val split, model, loss, optimizer, and epochs, but replace the generators with an equivalent `tf.data` pipeline that performs the same decoding, resizing, rescaling, and augmentations using TensorFlow ops, with parallel mapping, caching, and prefetching. I also remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it forces a much slower protobuf backend) while preserving determinism via explicit seeds and deterministic dataset options. Finally, I speed up submission label formatting by vectorizing the thresholding and top-1 fallback without changing prediction semantics.'
- What this solution (achieved 0.28627) has done: 'The timeout is dominated by the input pipeline and augmentation: each epoch decodes/resizes every JPEG and then runs relatively expensive per-image Python/TensorFlow ops, and the current `.cache()` forces a full in-memory cache of the decoded 256×256 float tensors (large) which can thrash and slow training. I keep the exact model/training loop and the same deterministic augmentations, but restructure the `tf.data` pipeline so that only the *decoded* dataset is cached to a deterministic on-disk cache (no RAM blowup), and the augmentation runs after batching so the same math is applied vectorized on whole batches. I also wrap augmentation in `@tf.function` and use fused `Rescaling` for equivalent normalization, while preserving determinism and seeds. Finally, I speed up submission label formatting by vectorizing the “empty row -> top1” fix without changing thresholding semantics.'
- What this solution (achieved 0.28627) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation (this is a compatibility workaround for the `MessageFactory.GetPrototype` error in some Kaggle images), which is required for the notebook to run at all. Then I fix the `tf.data` augmentation seeding bug by making the `tf.fill` inputs scalars and ensuring shapes are consistent; this unblocks dataset construction so training actually runs and produces predictions. These changes are execution/stability fixes and do not change your model architecture, loss, optimizer, epochs, or basic prediction/thresholding semantics, so the score behavior should stay in the same ballpark (and may improve slightly simply because training no longer be broken). Finally, I keep the submission writing exactly as required (`submission.csv` with `image,labels`) and add a small safety check that all test image paths exist.'
- What this solution (achieved 0.28627) has done: 'I fix the TensorFlow import crash by removing the protobuf override that forces the slow/old pure-Python protobuf path, which is the direct cause of the `MessageFactory.GetPrototype` AttributeError in this environment. Then I fix the `tf.fill` shape/rank bug inside `_augment_batch` by ensuring `rep_i` is a scalar tensor before filling, which unblocks dataset creation so `train_ds` exists and training can run. These are execution/stability fixes that preserve your model architecture, loss, optimizer, epochs, and prediction formatting; score should stay in the same ballpark (and may change slightly only because training actually run correctly). The script still write a valid `submission.csv` with the required `image,labels` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

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

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB_CSV)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

print("Num classes:", len(classes))
print("Classes:", classes[:20], "..." if len(classes) > 20 else "")



## === cell 3
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_df = train.iloc[tr_idx].reset_index(drop=True).copy()
val_df = train.iloc[va_idx].reset_index(drop=True).copy()

train_y = y[tr_idx]
val_y = y[va_idx]

train_df["__idx__"] = np.arange(len(train_df), dtype=np.int32)
val_df["__idx__"] = np.arange(len(val_df), dtype=np.int32)

train_df.head()



## === cell 4
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

AUTO = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True

train_paths = (TRAIN_IMG_DIR + "/" + train_df["image"].values).astype(str)
val_paths = (TRAIN_IMG_DIR + "/" + val_df["image"].values).astype(str)

train_indices = train_df["__idx__"].values.astype(np.int32)
val_indices = val_df["__idx__"].values.astype(np.int32)

train_y_tf = tf.constant(train_y.astype(np.float32))
val_y_tf = tf.constant(val_y.astype(np.float32))

_random_rot = keras.layers.RandomRotation(
    factor=10.0 / 180.0, fill_mode="reflect", seed=SEED
)

_rescale = keras.layers.Rescaling(1.0 / 255.0)


@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = _rescale(tf.cast(img, tf.float32))
    return img


@tf.function
def _augment_one(img, seed_base):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_base)

    img = _random_rot(img, training=True)

    zoom = tf.random.stateless_uniform(
        shape=[],
        seed=seed_base + tf.constant([0, 1], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * zoom), tf.int32)
    img_zoom = tf.image.resize(
        img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.resize_with_crop_or_pad(img_zoom, IMG_SIZE[0], IMG_SIZE[1])

    max_dx = int(round(0.05 * IMG_SIZE[1]))
    max_dy = int(round(0.05 * IMG_SIZE[0]))
    dx = tf.random.stateless_uniform(
        shape=[],
        seed=seed_base + tf.constant([0, 2], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        shape=[],
        seed=seed_base + tf.constant([0, 3], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])
    return img


@tf.function
def _augment_batch(imgs, labels, idxs, rep_i):
    rep_i = tf.cast(rep_i, tf.int32)
    rep_i = tf.reshape(rep_i, [])  # ensure scalar
    idxs = tf.cast(idxs, tf.int32)

    n = tf.shape(idxs)[0]
    s0 = tf.fill([n], tf.cast(SEED, tf.int32))
    rep0 = tf.fill([n], rep_i)
    rep1 = tf.fill(
        [n], rep_i * tf.constant(1103515245, tf.int32) + tf.constant(12345, tf.int32)
    )

    seeds_base = tf.stack([s0, idxs], axis=1) ^ tf.stack([rep0, rep1], axis=1)

    def _fn(img, seedb):
        return _augment_one(img, seedb)

    imgs = tf.map_fn(
        lambda t: _fn(t[0], t[1]),
        (imgs, seeds_base),
        fn_output_signature=tf.float32,
        parallel_iterations=16,
    )
    return imgs, labels


def _make_ds(paths, indices, y_tf, training: bool, cache_tag: str):
    ds = tf.data.Dataset.from_tensor_slices((paths, indices))
    ds = ds.with_options(options)

    def _load_img_and_label(path, idx):
        img = _decode_resize_rescale(path)
        label = tf.gather(y_tf, idx)
        return img, label, idx

    ds = ds.map(_load_img_and_label, num_parallel_calls=AUTO, deterministic=True)

    cache_path = os.path.join(
        "/kaggle/working", f"pp2021_cache_{cache_tag}_{IMG_SIZE[0]}x{IMG_SIZE[1]}"
    )
    ds = ds.cache(cache_path)

    if training:
        ds = ds.repeat()
        rep = tf.data.Dataset.range(tf.int32.max).repeat()
        ds = tf.data.Dataset.zip((rep, ds))

        ds = ds.map(
            lambda rep_i, data: (data[0], data[1], data[2], rep_i),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

        ds = ds.map(
            lambda img, label, idx, rep_i: _augment_batch(img, label, idx, rep_i),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
    else:

        def _drop_idx(img, label, idx):
            return img, label

        ds = ds.map(_drop_idx, num_parallel_calls=AUTO, deterministic=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.prefetch(AUTO)
    return ds


train_ds = _make_ds(
    train_paths, train_indices, train_y_tf, training=True, cache_tag="train"
)
val_ds = _make_ds(val_paths, val_indices, val_y_tf, training=False, cache_tag="val")

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

steps_per_epoch, val_steps



## === cell 5
num_classes = len(classes)

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.25)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 6
EPOCHS = 3  # keep identical budget/loop

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2931669203.py in <cell line: 0>()
      1 EPOCHS = 3  # keep identical budget/loop
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node Reshape defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:10 transformation with iterator: Iterator::Root::Prefetch::Map: Input to reshape is a tensor with 16 values, but the requested shape has 1
	 [[{{node Reshape}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_2340]

## === cell 7
test_paths = (TEST_IMG_DIR + "/" + submissions["image"].values).astype(str)

missing = [p for p in test_paths[:50] if not tf.io.gfile.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing test image(s), e.g.: {missing[0]}")


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(options)
    ds = ds.map(_decode_resize_rescale, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(32, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_ds = _make_test_ds(test_paths)

preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)



## === cell 8
thresh = 0.5

preds_np = np.asarray(preds)
mask = preds_np >= thresh
top1 = np.argmax(preds_np, axis=1)

classes_arr = np.array(classes, dtype=object)

row_any = mask.any(axis=1)
mask_fixed = mask.copy()
mask_fixed[~row_any, :] = False
mask_fixed[~row_any, top1[~row_any]] = True

idxs = [np.flatnonzero(row_mask) for row_mask in mask_fixed]
pred_labels = [" ".join(classes_arr[sel].tolist()) for sel in idxs]

submissions["labels"] = pred_labels
submissions.to_csv("submission.csv", index=False)

submissions.head()



## === cell 9
assert os.path.exists("submission.csv"), "submission.csv was not created"
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image", "labels"], "Submission columns mismatch"
assert len(check) == len(submissions), "Submission row count mismatch"

print("Wrote submission.csv with", len(check), "rows")
print(check.head())
