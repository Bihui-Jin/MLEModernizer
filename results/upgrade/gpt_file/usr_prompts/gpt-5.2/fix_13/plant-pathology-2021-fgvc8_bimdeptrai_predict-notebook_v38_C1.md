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

0.8221237303785797

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

keras = tf.keras

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF:", tf.__version__)

from sklearn.preprocessing import MultiLabelBinarizer



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)

labels_df = pd.DataFrame(y, columns=class_names)
print("Classes:", class_names)
labels_df.head()



## === cell 3
train_df = train.copy()
for c in class_names:
    train_df[c] = labels_df[c].values

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df) * val_frac)
val_df = train_df.iloc[:n_val].reset_index(drop=True)
trn_df = train_df.iloc[n_val:].reset_index(drop=True)

print("Train/Val:", trn_df.shape, val_df.shape)



## === cell 4
h_target = 256
w_target = 256
batch_size = 32

AUTOTUNE = tf.data.AUTOTUNE


@tf.function(jit_compile=True)
def _read_jpeg(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, [h_target, w_target, 3])
    return img


@tf.function(jit_compile=True)
def _affine_transform(img, angle_rad, tx, ty, zx, zy, flip_lr):
    img = tf.cond(flip_lr, lambda: tf.image.flip_left_right(img), lambda: img)

    cx = (w_target - 1) / 2.0
    cy = (h_target - 1) / 2.0

    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)

    a00 = zx * cos_a
    a01 = -zx * sin_a
    a10 = zy * sin_a
    a11 = zy * cos_a

    b0 = cx + tx - (a00 * cx + a01 * cy)
    b1 = cy + ty - (a10 * cx + a11 * cy)

    det = a00 * a11 - a01 * a10
    inv00 = a11 / det
    inv01 = -a01 / det
    inv10 = -a10 / det
    inv11 = a00 / det

    t0 = -(inv00 * b0 + inv01 * b1)
    t1 = -(inv10 * b0 + inv11 * b1)

    transform = tf.stack([inv00, inv01, t0, inv10, inv11, t1, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


@tf.function(jit_compile=True)
def _augment(img, seed2):
    s = tf.cast(seed2, tf.int32)

    angle = tf.random.stateless_uniform(
        [], seed=[s[0], s[1]], minval=-15.0, maxval=15.0
    ) * (np.pi / 180.0)
    s = s + tf.constant([0, 1], tf.int32)

    tx = tf.random.stateless_uniform(
        [], seed=[s[0], s[1]], minval=-0.05, maxval=0.05
    ) * float(w_target)
    s = s + tf.constant([0, 1], tf.int32)

    ty = tf.random.stateless_uniform(
        [], seed=[s[0], s[1]], minval=-0.05, maxval=0.05
    ) * float(h_target)
    s = s + tf.constant([0, 1], tf.int32)

    zx = tf.random.stateless_uniform([], seed=[s[0], s[1]], minval=0.9, maxval=1.1)
    s = s + tf.constant([0, 1], tf.int32)

    zy = tf.random.stateless_uniform([], seed=[s[0], s[1]], minval=0.9, maxval=1.1)
    s = s + tf.constant([0, 1], tf.int32)

    flip_lr = (
        tf.random.stateless_uniform([], seed=[s[0], s[1]], minval=0.0, maxval=1.0) < 0.5
    )

    return _affine_transform(img, angle, tx, ty, zx, zy, flip_lr)


@tf.function
def _decode_with_labels(path, y_):
    return _read_jpeg(path), y_


@tf.function
def _decode_only(path):
    return _read_jpeg(path)


@tf.function
def _train_aug_from_seed(img, y_, seed2):
    img = _augment(img, seed2)
    return img, y_


def make_dataset_from_df(
    df, img_dir, training, shuffle, repeat, with_labels, seed, cache, cache_name=None
):
    paths = (img_dir.rstrip("/") + "/" + df["image"].values).astype(str)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    try:
        opts.experimental_threading.private_threadpool_size = 16
        opts.experimental_threading.max_intra_op_parallelism = 1
    except Exception:
        pass

    if with_labels:
        labels = df[class_names].values.astype(np.float32, copy=False)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.with_options(opts)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 16384), seed=seed, reshuffle_each_iteration=True
        )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if with_labels:
        ds = ds.map(
            _decode_with_labels, num_parallel_calls=AUTOTUNE, deterministic=True
        )

        if training:
            seed_ds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True)
            ds = tf.data.Dataset.zip((ds, seed_ds))

            @tf.function
            def _apply_aug(xy, s2):
                seed2 = tf.stack([tf.cast(seed, tf.int32), tf.cast(s2, tf.int32)])
                return _train_aug_from_seed(xy[0], xy[1], seed2)

            ds = ds.map(_apply_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
        else:
            if cache:
                ds = ds.cache()
    else:
        ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True)
        if cache:
            ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    drop_remainder = bool(training and repeat)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset_from_df(
    trn_df,
    TRAIN_IMG_DIR,
    training=True,
    shuffle=True,
    repeat=True,
    with_labels=True,
    seed=SEED,
    cache=False,  # keep disabled: training has augmentation; caching would alter per-epoch randomness
)
val_ds = make_dataset_from_df(
    val_df,
    TRAIN_IMG_DIR,
    training=False,
    shuffle=False,
    repeat=False,
    with_labels=True,
    seed=SEED,
    cache=True,  # safe and speeds up validation across epochs
    cache_name="val_decode_cache",
)
test_ds = make_dataset_from_df(
    submissions,
    TEST_IMG_DIR,
    training=False,
    shuffle=False,
    repeat=False,
    with_labels=False,
    seed=SEED,
    cache=True,  # safe and speeds up prediction
    cache_name="test_decode_cache",
)



## === cell 5
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
base.trainable = False  # keep training stable and within time

inp = keras.layers.Input(shape=(h_target, w_target, 3))
x = base(inp, training=False)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(len(class_names), activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="binary_crossentropy",
)

model.summary()



## === cell 6
epochs = 3

steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
val_steps = int(np.ceil(len(val_df) / batch_size))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1673331367.py in <cell line: 0>()
      4 val_steps = int(np.ceil(len(val_df) / batch_size))
      5 
----> 6 history = model.fit(
      7     train_ds,
      8     validation_data=val_ds,

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

Detected at node path defined at (most recent call last):
<stack traces unavailable>
Detected at node path defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::BatchV2::ForeverRepeat[0]::ParallelMapV2::Zip[0]::ParallelMapV2: Detected unsupported operations when trying to compile graph __inference__read_jpeg_37[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node path}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="path", index=0){{node path}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference__read_jpeg_37[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[PartitionedCall/PartitionedCall]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_30452]

## === cell 7
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1062792689.py in <cell line: 0>()
----> 1 preds = model.predict(test_ds, verbose=1)
      2 print("preds shape:", preds.shape)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:21 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Detected unsupported operations when trying to compile graph __inference__read_jpeg_37[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node path}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="path", index=0){{node path}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference__read_jpeg_37[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[PartitionedCall/PartitionedCall]] [Op:IteratorGetNext] name: 

## === cell 8
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}

thr_arr = np.array([thresh[c] for c in class_names], dtype=np.float32)
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None



## === cell 9
preds_np = np.asarray(preds, dtype=np.float32)
argmax_idx = preds_np.argmax(axis=1)
max_is_healthy = (healthy_idx is not None) & (argmax_idx == healthy_idx)

above = preds_np > thr_arr[None, :]

class_names_arr = np.array(class_names, dtype=object)

pred_labels = []
for i in range(preds_np.shape[0]):
    if max_is_healthy[i]:
        pred_labels.append("healthy")
        continue
    chosen_idx = np.flatnonzero(above[i])
    if chosen_idx.size:
        s = " ".join(class_names_arr[chosen_idx].tolist())
    else:
        s = ""
    if (s == "") or ("healthy" in s):
        s = class_names[int(argmax_idx[i])]
    pred_labels.append(s)

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels

sub_path = "submission.csv"
submissions_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
submissions_out.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1380034209.py in <cell line: 0>()
----> 1 preds_np = np.asarray(preds, dtype=np.float32)
      2 argmax_idx = preds_np.argmax(axis=1)
      3 max_is_healthy = (healthy_idx is not None) & (argmax_idx == healthy_idx)
      4 
      5 above = preds_np > thr_arr[None, :]

NameError: name 'preds' is not defined

## === cell 10
assert submissions_out.shape[0] == submissions.shape[0]
assert list(submissions_out.columns) == ["image", "labels"]
assert submissions_out["labels"].isna().sum() == 0
print(submissions_out["labels"].head(20).tolist())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2069190811.py in <cell line: 0>()
----> 1 assert submissions_out.shape[0] == submissions.shape[0]
      2 assert list(submissions_out.columns) == ["image", "labels"]
      3 assert submissions_out["labels"].isna().sum() == 0
      4 print(submissions_out["labels"].head(20).tolist())

NameError: name 'submissions_out' is not defined
