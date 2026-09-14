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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.8993

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



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

from zipfile import ZipFile



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
files_dataframe = pd.read_csv(path + "train.csv", dtype={"id": str, "has_cactus": int})
files_dataframe.head()



## === cell 3
os.makedirs("./train", exist_ok=True)
os.makedirs("./test", exist_ok=True)


def _maybe_extract(zip_path, out_dir):
    marker = os.path.join(out_dir, ".extracted")
    if os.path.exists(marker):
        return
    with ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    with open(marker, "w") as f:
        f.write("ok")


_maybe_extract(path + "train.zip", ".")
_maybe_extract(path + "test.zip", ".")

training_files = "train/" + files_dataframe["id"]
print("Training sample:")
print(training_files.head(2))

print(
    "Sanity check exists:",
    os.path.exists("./" + training_files.iloc[0]),
)



## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()



## === cell 5
total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts[1])
no_cactus_weight = total_samples / (2 * class_reparts[0])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights: ", class_weights)



## === cell 6
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (3, 97))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale




## === cell 7
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (32, 32)

files_df = files_dataframe.copy()
files_df["id"] = files_df["id"].astype(str)

perm = np.random.RandomState(SEED).permutation(len(files_df))
files_df = files_df.iloc[perm].reset_index(drop=True)

val_frac = 0.25
val_n = int(round(len(files_df) * val_frac))
val_df = files_df.iloc[:val_n].reset_index(drop=True)
train_df = files_df.iloc[val_n:].reset_index(drop=True)

print("Train size:", len(train_df), "Val size:", len(val_df))

_p_cache_path = "/kaggle/working/p2p98_cache.npz"


def _build_p2p98_cache(df, root_dir="./train/"):
    import imageio.v2 as imageio  # fast JPEG reader in Python

    ids = df["id"].to_numpy()
    p2 = np.empty(len(ids), dtype=np.float32)
    p98 = np.empty(len(ids), dtype=np.float32)
    for i, fn in enumerate(ids):
        img = imageio.imread(os.path.join(root_dir, fn))
        imgf = img.astype(np.float32)
        p2[i], p98[i] = np.percentile(imgf, (3, 97))
    return ids.astype(str), p2, p98


if os.path.exists(_p_cache_path):
    cache = np.load(_p_cache_path, allow_pickle=False)
    cache_ids = cache["id"].astype(str)
    cache_p2 = cache["p2"].astype(np.float32)
    cache_p98 = cache["p98"].astype(np.float32)
else:
    cache_ids, cache_p2, cache_p98 = _build_p2p98_cache(files_df, root_dir="./train/")
    np.savez_compressed(_p_cache_path, id=cache_ids, p2=cache_p2, p98=cache_p98)

keys = tf.constant(cache_ids, dtype=tf.string)
vals_p2 = tf.constant(cache_p2, dtype=tf.float32)
vals_p98 = tf.constant(cache_p98, dtype=tf.float32)

table_p2 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals_p2),
    default_value=tf.constant(0.0, dtype=tf.float32),
)
table_p98 = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals_p98),
    default_value=tf.constant(255.0, dtype=tf.float32),
)


def _decode_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0  # match skimage domain
    return img


def _rescale_intensity(img, p2, p98):
    p2 = tf.maximum(p2, 0.0)
    p98 = tf.minimum(p98, 255.0)
    denom = tf.maximum(p98 - p2, 1e-6)
    img = (img - p2) * (255.0 / denom)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def _samplewise_center_std(img):
    mean = tf.reduce_mean(img)
    std = tf.math.reduce_std(img)
    img = img - mean
    img = img / tf.maximum(std, 1e-6)
    return img


def _random_rotate_45(img):
    angle = tf.random.uniform([], minval=-45.0, maxval=45.0, seed=SEED) * (
        np.pi / 180.0
    )
    img = tf.image.rotate(
        img, angles=angle, interpolation="BILINEAR", fill_mode="REFLECT"
    )
    return img


def _random_shear_x(img, shear_deg=10.0):
    sh = tf.random.uniform([], -shear_deg, shear_deg, seed=SEED) * (np.pi / 180.0)
    tan = tf.math.tan(sh)
    transform = tf.stack([1.0, -tan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=0)
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(img, 0)


def _augment(img):
    img = tf.image.resize(img, IMG_SIZE, method="BILINEAR")
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    img = _random_rotate_45(img)
    img = _random_shear_x(img, shear_deg=10.0)
    return img


def _prep_example(filename, label, do_augment=True, use_preprocess=True):
    fpath = tf.strings.join(["./train/", filename])
    img = _decode_jpeg(fpath)

    if use_preprocess:
        p2 = table_p2.lookup(filename)
        p98 = table_p98.lookup(filename)
        img = _rescale_intensity(img, p2, p98)

    if do_augment:
        img = _augment(img)
    else:
        img = tf.image.resize(img, IMG_SIZE, method="BILINEAR")

    img = _samplewise_center_std(img)

    label = tf.cast(label, tf.int32)
    y = tf.one_hot(label, depth=2, dtype=tf.float32)
    return img, y


def make_ds(df, training=True):
    filenames = tf.constant(df["id"].to_numpy(dtype=str), dtype=tf.string)
    labels = tf.constant(df["has_cactus"].to_numpy(dtype=np.int32), dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda f, y: _prep_example(f, y, do_augment=True, use_preprocess=True),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda f, y: _prep_example(f, y, do_augment=False, use_preprocess=True),
            num_parallel_calls=AUTOTUNE,
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


training_ds = make_ds(train_df, training=True)
validation_ds = make_ds(val_df, training=False)

training_n = len(train_df)
validation_n = len(val_df)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2682795740.py in <cell line: 0>()
     38     cache_p98 = cache["p98"].astype(np.float32)
     39 else:
---> 40     cache_ids, cache_p2, cache_p98 = _build_p2p98_cache(files_df, root_dir="./train/")
     41     np.savez_compressed(_p_cache_path, id=cache_ids, p2=cache_p2, p98=cache_p98)
     42 

/tmp/ipykernel_11/2682795740.py in _build_p2p98_cache(df, root_dir)
     26     p98 = np.empty(len(ids), dtype=np.float32)
     27     for i, fn in enumerate(ids):
---> 28         img = imageio.imread(os.path.join(root_dir, fn))
     29         imgf = img.astype(np.float32)
     30         p2[i], p98[i] = np.percentile(imgf, (3, 97))

/usr/local/lib/python3.11/dist-packages/imageio/v2.py in imread(uri, format, **kwargs)
    357     imopen_args["legacy_mode"] = True
    358 
--> 359     with imopen(uri, "ri", **imopen_args) as file:
    360         result = file.read(index=0, **kwargs)
    361 

/usr/local/lib/python3.11/dist-packages/imageio/core/imopen.py in imopen(uri, io_mode, plugin, extension, format_hint, legacy_mode, **kwargs)
    111         request.format_hint = format_hint
    112     else:
--> 113         request = Request(uri, io_mode, format_hint=format_hint, extension=extension)
    114 
    115     source = "<bytes>" if isinstance(uri, bytes) else uri

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in __init__(self, uri, mode, extension, format_hint, **kwargs)
    247 
    248         # Parse what was given
--> 249         self._parse_uri(uri)
    250 
    251         # Set extension

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in _parse_uri(self, uri)
    407                 # Reading: check that the file exists (but is allowed a dir)
    408                 if not os.path.exists(fn):
--> 409                     raise FileNotFoundError("No such file: '%s'" % fn)
    410             else:
    411                 # Writing: check that the directory to write to does exist

FileNotFoundError: No such file: '/kaggle/working/train/5e8dedba4d068910a3a91e8ba6642880.jpg'

## === cell 8
print("training_ds samples:", training_n)
print("validation_ds samples:", validation_n)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3235180920.py in <cell line: 0>()
----> 1 print("training_ds samples:", training_n)
      2 print("validation_ds samples:", validation_n)
      3 

NameError: name 'training_n' is not defined

## === cell 9
pass



## === cell 10
from tf_keras import layers, models
from tf_keras.models import clone_model



## === cell 11
model = models.Sequential()

model.add(
    layers.Conv2D(
        32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 12
from tf_keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

checkpoint_path = "/tmp/checkpoint.keras"
save_best_model = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)



## === cell 13
steps_per_epoch = max(1, training_n // BATCH_SIZE)
val_steps = max(1, validation_n // BATCH_SIZE)

history = model.fit(
    training_ds,
    validation_data=validation_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
    epochs=10,
    callbacks=[reduce_lr, save_best_model],
    class_weight=class_weights,
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1430689189.py in <cell line: 0>()
----> 1 steps_per_epoch = max(1, training_n // BATCH_SIZE)
      2 val_steps = max(1, validation_n // BATCH_SIZE)
      3 
      4 history = model.fit(
      5     training_ds,

NameError: name 'training_n' is not defined

## === cell 14
sample_sub = pd.read_csv(path + "sample_submission.csv", dtype={"id": str})
test_df = sample_sub.copy()


def _prep_test_example(filename):
    fpath = tf.strings.join(["./test/", filename])
    img = _decode_jpeg(fpath)
    img = tf.image.resize(img, IMG_SIZE, method="BILINEAR")

    flat = tf.reshape(img, [-1])
    flat_sorted = tf.sort(flat)
    n = tf.shape(flat_sorted)[0]
    i2 = tf.cast(tf.round(0.03 * tf.cast(n - 1, tf.float32)), tf.int32)
    i98 = tf.cast(tf.round(0.97 * tf.cast(n - 1, tf.float32)), tf.int32)
    p2 = flat_sorted[i2]
    p98 = flat_sorted[i98]

    img = _rescale_intensity(img, p2, p98)
    img = _samplewise_center_std(img)
    return img


test_filenames = tf.constant(test_df["id"].to_numpy(dtype=str), dtype=tf.string)
test_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
test_ds = test_ds.map(_prep_test_example, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

print("test samples:", len(test_df))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/903989493.py in <cell line: 0>()
     24 test_filenames = tf.constant(test_df["id"].to_numpy(dtype=str), dtype=tf.string)
     25 test_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
---> 26 test_ds = test_ds.map(_prep_test_example, num_parallel_calls=AUTOTUNE)
     27 test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_file6zx7yk9g.py in tf___prep_test_example(filename)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 fpath = ag__.converted_call(ag__.ld(tf).strings.join, (['./test/', ag__.ld(filename)],), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_decode_jpeg), (ag__.ld(fpath),), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), ag__.ld(IMG_SIZE)), dict(method='BILINEAR'), fscope)
     13                 flat = ag__.converted_call(ag__.ld(tf).reshape, (ag__.ld(img), [-1]), None, fscope)

NameError: in user code:

    File "/tmp/ipykernel_11/903989493.py", line 7, in _prep_test_example  *
        img = _decode_jpeg(fpath)

    NameError: name '_decode_jpeg' is not defined


## === cell 15
from tf_keras.models import load_model

if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)



## === cell 16
proba = model.predict(test_ds, verbose=0)
if proba.ndim == 2 and proba.shape[1] == 2:
    has_cactus_proba = proba[:, 1]
else:
    has_cactus_proba = proba.reshape(-1)

output = sample_sub.copy()
output["has_cactus"] = has_cactus_proba.astype(float)
output["has_cactus"] = output["has_cactus"].clip(0.0, 1.0)

output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1121741776.py in <cell line: 0>()
----> 1 proba = model.predict(test_ds, verbose=0)
      2 if proba.ndim == 2 and proba.shape[1] == 2:
      3     has_cactus_proba = proba[:, 1]
      4 else:
      5     has_cactus_proba = proba.reshape(-1)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py in tf__predict_function(iterator)
     16                 except:
     17                     do_return = False
---> 18                     raise
     19                 return fscope.ret(retval_, do_return)
     20         return tf__predict_function

ValueError: in user code:

    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2436, in predict_function  *
        return step_function(self, iterator)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2421, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2409, in run_step  **
        outputs = model.predict_step(data)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2377, in predict_step
        return self(x, training=False)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py", line 70, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/input_spec.py", line 253, in assert_input_compatibility
        raise ValueError(

    ValueError: Exception encountered when calling layer 'sequential' (type Sequential).
    
    Input 0 of layer "conv2d" is incompatible with the layer: expected min_ndim=4, found ndim=0. Full shape received: ()
    
    Call arguments received by layer 'sequential' (type Sequential):
      • inputs=tf.Tensor(shape=(), dtype=string)
      • training=False
      • mask=None


## === cell 17
import shutil

for d in ["test", "train"]:
    try:
        shutil.rmtree(d)
    except OSError:
        print(f"{d} files already erased or not present")
