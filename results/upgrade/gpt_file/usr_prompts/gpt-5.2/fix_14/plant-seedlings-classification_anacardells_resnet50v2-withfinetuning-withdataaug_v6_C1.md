# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

print("Input root exists:", os.path.exists("/kaggle/input"))
print(
    "Train dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/train"),
)
print(
    "Test dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/test"),
)



## === cell 1
TRAIN_ROOT = "/kaggle/input/plant-seedlings-classification/train"
print("Train root:", TRAIN_ROOT)
print("Skipping train.csv generation (unused by training/prediction).")



## === cell 2
classes = sorted(
    [d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))]
)
print(f"Number of classes: {len(classes)}")
print("Classes:", classes)



## === cell 3
print("Skipping plots for runtime.")



## === cell 4
print("Skipping sample image visualization for runtime.")



## === cell 5
print("Skipping image-size histogram for runtime.")



## === cell 6
try:
    import google.protobuf  # noqa: F401
    import protobuf  # type: ignore # noqa: F401
except Exception:
    pass


def _ensure_protobuf_compatible():
    try:
        import google.protobuf as gp

        ver = getattr(gp, "__version__", "")
        print("Detected protobuf version:", ver)
        major = int(ver.split(".")[0]) if ver else 0
        if major >= 5:
            print("Downgrading protobuf to 4.25.3 for TF compatibility...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib

            importlib.invalidate_caches()
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf") or m == "protobuf":
                    sys.modules.pop(m, None)
    except Exception as e:
        print("[WARN] Could not validate/downgrade protobuf:", repr(e))


_ensure_protobuf_compatible()

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
)
from math import exp

print("TensorFlow:", tf.__version__)
print("Using tf.keras:", tf.keras.__name__)



## === cell 7
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
seed = 42
val_split = 0.2
image_size = (256, 256)

PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

train_list_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="int",
    class_names=classes,  # keep identical class->index mapping
    color_mode="rgb",
    batch_size=None,
    image_size=image_size,
    shuffle=True,
    seed=seed,
    validation_split=val_split,
    subset="training",
    interpolation="bilinear",
)

val_list_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="int",
    class_names=classes,
    color_mode="rgb",
    batch_size=None,
    image_size=image_size,
    shuffle=True,
    seed=seed,
    validation_split=val_split,
    subset="validation",
    interpolation="bilinear",
)

num_classes = len(classes)
print("Class indices:", {c: i for i, c in enumerate(classes)})
print("num_classes:", num_classes)

train_count = int(tf.data.experimental.cardinality(train_list_ds).numpy())
val_count = int(tf.data.experimental.cardinality(val_list_ds).numpy())

train_steps = train_count // batch_size
val_steps = val_count // batch_size
print("train_count:", train_count, "val_count:", val_count)
print("train_steps:", train_steps, "val_steps:", val_steps)


def _to_onehot(y_int):
    return tf.one_hot(tf.cast(y_int, tf.int32), depth=num_classes, dtype=tf.float32)


_rot_layer = layers.RandomRotation(
    factor=30.0 / 180.0,  # ±30 degrees
    fill_mode="reflect",
    interpolation="bilinear",
    seed=seed,
)


@tf.function
def _stateless_uniform_per_example(ex_seeds, minval, maxval, dtype=tf.float32):
    def _one(s):
        return tf.random.stateless_uniform(
            shape=[],
            seed=tf.cast(s, tf.int64),
            minval=tf.cast(minval, dtype),
            maxval=tf.cast(maxval, dtype),
            dtype=dtype,
        )

    return tf.map_fn(_one, ex_seeds, fn_output_signature=dtype)


@tf.function
def _stateless_uniform_int_per_example(ex_seeds, minval, maxval):
    def _one(s):
        return tf.random.stateless_uniform(
            shape=[],
            seed=tf.cast(s, tf.int64),
            minval=tf.cast(minval, tf.int32),
            maxval=tf.cast(maxval, tf.int32),
            dtype=tf.int32,
        )

    return tf.map_fn(_one, ex_seeds, fn_output_signature=tf.int32)


@tf.function
def _augment_batch(imgs_uint8_or_float, y_ints, idxs):
    imgs = tf.cast(imgs_uint8_or_float, tf.float32)
    y = _to_onehot(y_ints)

    ex_seeds = tf.stack(
        [tf.fill(tf.shape(idxs), tf.cast(seed, tf.int64)), tf.cast(idxs, tf.int64)],
        axis=1,
    )

    x = tf.clip_by_value(imgs / 255.0, 0.0, 1.0)

    p_lr = _stateless_uniform_per_example(
        ex_seeds + tf.constant([0, 10], tf.int64), 0.0, 1.0
    )
    x_lr = tf.image.flip_left_right(x)
    x = tf.where(p_lr[:, None, None, None] < 0.5, x_lr, x)

    p_ud = _stateless_uniform_per_example(
        ex_seeds + tf.constant([0, 11], tf.int64), 0.0, 1.0
    )
    x_ud = tf.image.flip_up_down(x)
    x = tf.where(p_ud[:, None, None, None] < 0.5, x_ud, x)

    br = _stateless_uniform_per_example(
        ex_seeds + tf.constant([0, 12], tf.int64), 0.7, 1.3, dtype=tf.float32
    )
    x = tf.clip_by_value(x * br[:, None, None, None], 0.0, 1.0)

    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    zoom = _stateless_uniform_per_example(
        ex_seeds + tf.constant([0, 13], tf.int64), 0.8, 1.2, dtype=tf.float32
    )
    crop_h = tf.cast(tf.cast(h, tf.float32) / zoom, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) / zoom, tf.int32)
    crop_h = tf.clip_by_value(crop_h, 1, h)
    crop_w = tf.clip_by_value(crop_w, 1, w)

    max_off_y = h - crop_h
    max_off_x = w - crop_w

    off_y = _stateless_uniform_int_per_example(
        ex_seeds + tf.constant([0, 4], tf.int64),
        0,
        tf.reduce_max(tf.maximum(max_off_y, 0)) + 1,
    )
    off_x = _stateless_uniform_int_per_example(
        ex_seeds + tf.constant([0, 5], tf.int64),
        0,
        tf.reduce_max(tf.maximum(max_off_x, 0)) + 1,
    )
    off_y = tf.minimum(off_y, tf.maximum(max_off_y, 0))
    off_x = tf.minimum(off_x, tf.maximum(max_off_x, 0))

    y1 = tf.cast(off_y, tf.float32) / tf.cast(h, tf.float32)
    x1 = tf.cast(off_x, tf.float32) / tf.cast(w, tf.float32)
    y2 = tf.cast(off_y + crop_h, tf.float32) / tf.cast(h, tf.float32)
    x2 = tf.cast(off_x + crop_w, tf.float32) / tf.cast(w, tf.float32)

    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    box_indices = tf.range(tf.shape(x)[0], dtype=tf.int32)
    x = tf.image.crop_and_resize(
        x,
        boxes,
        box_indices,
        crop_size=image_size,
        method="bilinear",
        extrapolation_value=0.0,
    )

    max_dx = tf.cast(tf.round(0.2 * tf.cast(image_size[1], tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.2 * tf.cast(image_size[0], tf.float32)), tf.int32)

    dx = _stateless_uniform_int_per_example(
        ex_seeds + tf.constant([0, 14], tf.int64), -max_dx, max_dx + 1
    )
    dy = _stateless_uniform_int_per_example(
        ex_seeds + tf.constant([0, 15], tf.int64), -max_dy, max_dy + 1
    )

    pad_x = tf.abs(dx)
    pad_y = tf.abs(dy)
    pad_x_max = tf.reduce_max(pad_x)
    pad_y_max = tf.reduce_max(pad_y)
    x_pad = tf.pad(
        x,
        [[0, 0], [pad_y_max, pad_y_max], [pad_x_max, pad_x_max], [0, 0]],
        mode="REFLECT",
    )

    off_y2 = pad_y_max - dy
    off_x2 = pad_x_max - dx
    Hp = tf.shape(x_pad)[1]
    Wp = tf.shape(x_pad)[2]

    y1p = tf.cast(off_y2, tf.float32) / tf.cast(Hp, tf.float32)
    x1p = tf.cast(off_x2, tf.float32) / tf.cast(Wp, tf.float32)
    y2p = tf.cast(off_y2 + image_size[0], tf.float32) / tf.cast(Hp, tf.float32)
    x2p = tf.cast(off_x2 + image_size[1], tf.float32) / tf.cast(Wp, tf.float32)

    boxes2 = tf.stack([y1p, x1p, y2p, x2p], axis=1)
    x = tf.image.crop_and_resize(
        x_pad,
        boxes2,
        box_indices,
        crop_size=image_size,
        method="bilinear",
        extrapolation_value=0.0,
    )

    x = _rot_layer(x, training=True)

    x = tf.clip_by_value(x, 0.0, 1.0) * 255.0
    x = preprocess_input(x)
    return x, y


@tf.function
def _preprocess_val_batch(imgs, y_ints):
    y = _to_onehot(y_ints)
    x = preprocess_input(tf.cast(imgs, tf.float32))
    return x, y


def _make_train_ds(ds_unbatched):
    ds = ds_unbatched.cache()
    ds = ds.enumerate()  # (idx, (img, y))
    ds = ds.batch(batch_size, drop_remainder=True)

    def _pack(idxs, xy):
        imgs, y_ints = xy
        return imgs, y_ints, tf.cast(idxs, tf.int64)

    ds = ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.map(
        lambda imgs, y_ints, idxs: _augment_batch(imgs, y_ints, idxs),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _make_val_ds(ds_unbatched):
    ds = ds_unbatched.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(
        _preprocess_val_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = _make_train_ds(train_list_ds)
val_ds = _make_val_ds(val_list_ds)



## === cell 8
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 9
for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)
pre_trained_model.add(layers.Flatten())
pre_trained_model.add(layers.Dense(512, activation="relu"))
pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())
pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))
pre_trained_model.summary()



## === cell 10
epochs = 200

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)

earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
)

best_path = file + ".weights.h5"
modelsave = ModelCheckpoint(
    filepath=best_path, save_best_only=True, save_weights_only=True, verbose=1
)

print("[INFO]: Training the network...")
H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## === cell 11
print("[INFO]: Training finished. Skipping curve plots for runtime.")
print("Epochs run:", len(H_pre.history.get("loss", [])))



## === cell 12
csv_testfile = "/kaggle/working/test.csv"
print("Skipping test.csv generation for runtime. (Not needed.)")



## === cell 13
print("Skipping loading test.csv for runtime.")



## === cell 14
if os.path.exists(best_path):
    print("[INFO]: Loading best checkpoint weights for inference:", best_path)
    pre_trained_model.load_weights(best_path)
else:
    print("[WARN]: Best checkpoint weights not found; using in-memory model weights.")

test_batch_size = 32
PROYECT_FOLDER_TEST = "/kaggle/input/plant-seedlings-classification/test"

test_files_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TEST,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode="rgb",
    batch_size=test_batch_size,
    image_size=image_size,
    shuffle=False,
    interpolation="bilinear",
)


@tf.function
def _preprocess_only(img):
    return preprocess_input(tf.cast(img, tf.float32))


test_ds = test_files_ds.map(
    _preprocess_only, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
).prefetch(tf.data.AUTOTUNE)

list_of_files = [os.path.basename(p) for p in test_files_ds.file_paths]
print("Example test filename from dataset:", list_of_files[0])
print("Num test files:", len(list_of_files))



## === cell 15
predicted_class = pre_trained_model.predict(
    test_ds,
    verbose=1,
)



## === cell 16
predicted_class_number = np.argmax(predicted_class, axis=1)

idx_to_class = {i: c for i, c in enumerate(classes)}
classes_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Ordered classes:", classes_ordered[:5], "...", len(classes_ordered))
print("Predicted class indices sample:", predicted_class_number[:10])



## === cell 17
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

sample = pd.read_csv(sample_path)
if "file" not in sample.columns or "species" not in sample.columns:
    raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")

pred_files = list_of_files
pred_species = [classes_ordered[i] for i in predicted_class_number]
pred_df = pd.DataFrame({"file": pred_files, "species": pred_species})

submission = sample[["file"]].merge(pred_df, on="file", how="left")
if submission["species"].isna().any():
    missing = submission.loc[submission["species"].isna(), "file"].head(10).tolist()
    raise ValueError(f"Missing predictions for some files, e.g.: {missing}")

submission = submission[["file", "species"]]

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.shape)
print(submission.head())
print("Submission columns:", submission.columns.tolist())
print("Done. Submission is ready at /kaggle/working/submission.csv")
