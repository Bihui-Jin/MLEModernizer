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

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'Main bottlenecks are (1) resizing every image to 512×512 on the fly for all train/val/test via `tf.image.resize`, and (2) an unnecessary training fallback path that could run if the external `.h5` isn’t present. To stay within 600s without changing the model or semantics, I (a) force inference-only behavior (same as when the pretrained `.h5` exists) and fail fast if it’s missing, (b) enable TF fast JPEG decode and dataset optimizations (same decoded pixels, just faster pipeline), (c) avoid redundant `num_parallel_calls` on `batch` and tune dataset prefetch/determinism safely, and (d) speed up label post-processing by vectorizing the Python loop while preserving identical thresholding logic. Paths, architecture, loss, preprocessing, and prediction thresholds remain unchanged.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by decoding and resizing 3,727 test images to 512×512 with a small batch size, plus the model inference itself. I (1) make the input pipeline faster while remaining deterministic by using fused `map_and_batch`, enabling TF data optimizations, and using file caching safely for repeated epochs, and (2) increase the inference batch size only for `test_ds` (this does not change model outputs, just throughput). I also eliminate avoidable Python overhead in the label post-processing by vectorizing the conversion from prediction mask to label strings. These changes keep the same model, preprocessing, thresholds, and evaluation semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_USE_CUDNN_AUTOTUNE", "1")

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

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
h_target = 512
w_target = 512



## === cell 3
label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

print("Num classes:", len(classes))
print("Classes:", classes)



## === cell 4
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BATCH_SIZE = 8
TEST_BATCH_SIZE = (
    32  # larger inference batch improves throughput; preserves identical outputs
)
AUTOTUNE = tf.data.AUTOTUNE

train_df = train.copy()
train_df["labels_list"] = label_split

class_to_idx = {c: i for i, c in enumerate(classes)}


def _build_multihot_matrix(labels_list):
    out = np.zeros((len(labels_list), len(classes)), dtype=np.float32)
    for i, labs in enumerate(labels_list):
        for lab in labs:
            j = class_to_idx.get(lab)
            if j is not None:
                out[i, j] = 1.0
    return out


val_frac = 0.1
n = len(train_df)
n_val = int(np.floor(n * val_frac))
n_train = n - n_val

train_df_train = train_df.iloc[:n_train].reset_index(drop=True)
train_df_val = train_df.iloc[n_train:].reset_index(drop=True)

y_train = _build_multihot_matrix(train_df_train["labels_list"].tolist())
y_val = _build_multihot_matrix(train_df_val["labels_list"].tolist())

train_paths = (TRAIN_DIR + "/" + train_df_train["image"].values).astype(object)
val_paths = (TRAIN_DIR + "/" + train_df_val["image"].values).astype(object)
test_paths = (TEST_DIR + "/" + submissions["image"].values).astype(object)


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(
        img_bytes, channels=3, dct_method="INTEGER_FAST"
    )  # uint8
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([h_target, w_target, 3])
    return img


def _ds_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True

    opt = opts.experimental_optimization
    try:
        opt.apply_default_optimizations = True
    except Exception:
        pass
    if hasattr(opt, "map_parallelization"):
        try:
            opt.map_parallelization = True
        except Exception:
            pass
    if hasattr(opt, "parallel_batch"):
        try:
            opt.parallel_batch = True
        except Exception:
            pass
    return opts


def _make_ds(
    paths, labels=None, training=False, cache_in_memory=False, batch_size=BATCH_SIZE
):
    opts = _ds_options()
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                _decode_resize_preprocess,
                batch_size=batch_size,
                num_parallel_calls=AUTOTUNE,
                drop_remainder=False,
                deterministic=True,
            )
        )
        if cache_in_memory:
            ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opts)

    if training:
        shuffle_buf = min(len(paths), 2048)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    def _map_xy(p, y_):
        return _decode_resize_preprocess(p), y_

    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            _map_xy,
            batch_size=batch_size,
            num_parallel_calls=AUTOTUNE,
            drop_remainder=False,
            deterministic=True,
        )
    )
    if cache_in_memory:
        ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds(
    train_paths, y_train, training=True, cache_in_memory=False, batch_size=BATCH_SIZE
)
val_ds = _make_ds(
    val_paths, y_val, training=False, cache_in_memory=False, batch_size=BATCH_SIZE
)
test_ds = _make_ds(
    test_paths,
    labels=None,
    training=False,
    cache_in_memory=False,
    batch_size=TEST_BATCH_SIZE,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2513006322.py in <cell line: 0>()
    129 
    130 
--> 131 train_ds = _make_ds(
    132     train_paths, y_train, training=True, cache_in_memory=False, batch_size=BATCH_SIZE
    133 )

/tmp/ipykernel_11/2513006322.py in _make_ds(paths, labels, training, cache_in_memory, batch_size)
    115 
    116     ds = ds.apply(
--> 117         tf.data.experimental.map_and_batch(
    118             _map_xy,
    119             batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    381               'in a future version' if date is None else ('after %s' % date),
    382               instructions)
--> 383       return func(*args, **kwargs)
    384 
    385     doc_controls.set_deprecated(new_func)

TypeError: map_and_batch() got an unexpected keyword argument 'deterministic'

## === cell 5
model_path = "../input/mobilenetv2-512/mobilenetv2_512.h5"

num_classes = len(classes)


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(h_target, w_target, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(h_target, w_target, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    m = tf.keras.Model(inputs, outputs)

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return m


if tf.io.gfile.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    try:
        model.compile(optimizer="adam", loss="binary_crossentropy")
    except Exception:
        pass
    print("Loaded model:", model_path)
else:
    print("Pretrained model not found at:", model_path)
    print(
        "Training fallback enabled to produce a valid submission within the runtime budget."
    )
    model = build_model(num_classes)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        verbose=1,
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1160467149.py in <cell line: 0>()
     38     model = build_model(num_classes)
     39     model.fit(
---> 40         train_ds,
     41         validation_data=val_ds,
     42         epochs=1,

NameError: name 'train_ds' is not defined

## === cell 6
preds = model.predict(test_ds, verbose=1)
print("Preds shape:", preds.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4228193762.py in <cell line: 0>()
      1 # Throughput-only: keep same computation but use the faster test_ds batching from above.
----> 2 preds = model.predict(test_ds, verbose=1)
      3 print("Preds shape:", preds.shape)
      4 

NameError: name 'test_ds' is not defined

## === cell 7
thresh = 0.4
healthy_idx = classes.index("healthy") if "healthy" in classes else None

p = preds
chosen = p >= thresh

none_chosen = ~chosen.any(axis=1)
if np.any(none_chosen):
    am = np.argmax(p[none_chosen], axis=1)
    chosen[none_chosen, :] = False
    chosen[none_chosen, am] = True

pred_labels = np.empty((p.shape[0],), dtype=object)

if healthy_idx is not None:
    healthy_mask = p[:, healthy_idx] >= thresh
    pred_labels[healthy_mask] = "healthy"

    other_mask = ~healthy_mask
    other_idx = np.flatnonzero(other_mask)

    cls_arr = np.asarray(classes, dtype=object)
    for i in other_idx:
        idxs = np.flatnonzero(chosen[i])
        if idxs.size:
            pred_labels[i] = " ".join(cls_arr[idxs].tolist())
        else:
            pred_labels[i] = "healthy"
else:
    cls_arr = np.asarray(classes, dtype=object)
    for i in range(p.shape[0]):
        idxs = np.flatnonzero(chosen[i])
        pred_labels[i] = " ".join(cls_arr[idxs].tolist()) if idxs.size else "healthy"

submissions = submissions.copy()
submissions["labels"] = pred_labels.tolist()

submissions.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3678753863.py in <cell line: 0>()
      4 healthy_idx = classes.index("healthy") if "healthy" in classes else None
      5 
----> 6 p = preds
      7 chosen = p >= thresh
      8 

NameError: name 'preds' is not defined

## === cell 8
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))



## === cell 9
submissions
