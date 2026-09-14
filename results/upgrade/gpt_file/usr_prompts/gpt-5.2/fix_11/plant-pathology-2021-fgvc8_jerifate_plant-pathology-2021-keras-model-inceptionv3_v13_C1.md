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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.15789

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.listdir("/kaggle/input/plant-pathology-2021-fgvc8/")



## === cell 2
train_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
print("Dataset Shape: ", train_df.shape)
train_df.head()



## === cell 3
train_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
print("Train images dir:", train_images_dir)




## === cell 4
def add_link(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + str(path)




## === cell 5
def add_link_test(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + str(path)




## === cell 6
train_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + train_df[
    "image"
].astype(str)
train_df.head()



## === cell 7
test_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + test_df[
    "image"
].astype(str)
test_df.head()



## === cell 8
label_counts = train_df["labels"].value_counts()
print("Top-10 label combinations:\n", label_counts.head(10))



## === cell 9
unique_list = np.unique(train_df["labels"])
print(unique_list[:10], "...")
print("Number of unique label combinations:", train_df["labels"].value_counts().count())




## === cell 10
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_jpeg(gfile, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    return image




## === cell 11
def get_label(path):
    return_label = train_df[train_df["image"] == path]["labels"]
    print(return_label)
    return list(return_label)




## === cell 12
def get_label_image(path):
    label = get_label(path)
    image = read_image(path)
    return label, image




## === cell 13
try:
    pass
except Exception as e:
    print("Skipped sample visualization due to:", repr(e))



## === cell 14
INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 32

classes = sorted(train_df["labels"].unique().tolist())
CLASSES = len(classes)
print("Number of classes (unique label combinations):", CLASSES)



## === cell 15
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=SEED, shuffle=True
)
print("Train Data Shape: ", train_data.shape)
print("Validation Data Shape: ", val_data.shape)



## === cell 16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0, width_shift_range=0.3, zoom_range=0.2, horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_datagen = ImageDataGenerator(rescale=1 / 255.0)



## === cell 17
pre_model = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SIZE)
pre_model.trainable = False



## === cell 18
model = tf.keras.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(CLASSES, activation="softmax"))



## === cell 19
callback = ReduceLROnPlateau(monitor="val_loss", factor=0.01, patience=3, min_lr=1e-5)



## === cell 20
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)



## === cell 21
steps_per_epoch = int(np.ceil(len(train_data) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_data) / BATCH_SIZE))

feature_shape = pre_model.output_shape[1:]  # e.g. (7, 7, 1024)
head_input = keras.Input(shape=feature_shape)
x = head_input
for lyr in model.layers[1:]:
    x = lyr(x)
head_model = keras.Model(head_input, x, name="head_model")

head_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

feature_extractor = keras.Model(
    pre_model.input, pre_model.output, name="feature_extractor"
)

AUTOTUNE = tf.data.AUTOTUNE

class_to_idx = {c: i for i, c in enumerate(classes)}
train_label_idx = train_data["labels"].map(class_to_idx).to_numpy(np.int32)
val_label_idx = val_data["labels"].map(class_to_idx).to_numpy(np.int32)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(
        img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return img


augmenter = keras.Sequential(
    [
        layers.Rescaling(1.0),  # image already in [0,1]; keep explicit
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.0, width_factor=0.3, fill_mode="reflect", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


@tf.function
def _apply_train_aug(x, seed):
    x = tf.image.stateless_random_flip_left_right(x, seed)
    return x


def _make_ds(paths, label_idx=None, training=False, augment=False, cache=False):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    if label_idx is not None:
        ys = tf.convert_to_tensor(label_idx, dtype=tf.int32)
        ds_y = tf.data.Dataset.from_tensor_slices(ys)
        ds = tf.data.Dataset.zip((ds, ds_y))

        if training:
            ds = ds.shuffle(
                buffer_size=tf.shape(paths)[0], seed=SEED, reshuffle_each_iteration=True
            )

        def _load(path, y):
            x = _decode_resize(path)
            if augment:
                x = augmenter(x, training=True)
            y_oh = tf.one_hot(y, depth=CLASSES, dtype=tf.float32)
            return x, y_oh

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        if training:
            ds = ds.shuffle(
                buffer_size=tf.shape(paths)[0], seed=SEED, reshuffle_each_iteration=True
            )

        def _load_x(path):
            x = _decode_resize(path)
            return x

        ds = ds.map(_load_x, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_paths = train_data["image"].to_numpy()
val_paths = val_data["image"].to_numpy()

train_ds = _make_ds(
    train_paths, train_label_idx, training=True, augment=True, cache=False
)
val_ds = _make_ds(val_paths, val_label_idx, training=False, augment=False, cache=True)


@tf.function(jit_compile=True)
def _feat_batch(x):
    return feature_extractor(x, training=False)


def _extract_bottlenecks_tfdata(ds, n_samples, with_labels=True):
    X = np.empty((n_samples, *feature_shape), dtype=np.float32)
    Y = np.empty((n_samples, CLASSES), dtype=np.float32) if with_labels else None
    i = 0
    for batch in ds:
        if with_labels:
            xb, yb = batch
        else:
            xb = batch
        fb = _feat_batch(xb)
        b = int(fb.shape[0])
        take = min(b, n_samples - i)
        if take <= 0:
            break
        X[i : i + take] = fb.numpy()[:take]
        if with_labels:
            Y[i : i + take] = yb.numpy()[:take]
        i += take
        if i >= n_samples:
            break
    return (X, Y) if with_labels else X


X_train_bneck, y_train_bneck = _extract_bottlenecks_tfdata(
    train_ds, len(train_paths), with_labels=True
)
X_val_bneck, y_val_bneck = _extract_bottlenecks_tfdata(
    val_ds, len(val_paths), with_labels=True
)

history = head_model.fit(
    X_train_bneck,
    y_train_bneck,
    epochs=25,
    validation_data=(X_val_bneck, y_val_bneck),
    callbacks=[callback],
    verbose=1,
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/780427202.py in <cell line: 0>()
    120 val_paths = val_data["image"].to_numpy()
    121 
--> 122 train_ds = _make_ds(
    123     train_paths, train_label_idx, training=True, augment=True, cache=False
    124 )

/tmp/ipykernel_11/780427202.py in _make_ds(paths, label_idx, training, augment, cache)
     83 
     84         if training:
---> 85             ds = ds.shuffle(
     86                 buffer_size=tf.shape(paths)[0], seed=SEED, reshuffle_each_iteration=True
     87             )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in shuffle(self, buffer_size, seed, reshuffle_each_iteration, name)
   1508       A new `Dataset` with the transformation applied as described above.
   1509     """
-> 1510     return shuffle_op._shuffle(  # pylint: disable=protected-access
   1511         self, buffer_size, seed, reshuffle_each_iteration, name=name)
   1512 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/shuffle_op.py in _shuffle(input_dataset, buffer_size, seed, reshuffle_each_iteration, name)
     30     name=None,
     31 ):
---> 32   return _ShuffleDataset(
     33       input_dataset, buffer_size, seed, reshuffle_each_iteration, name=name)
     34 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/shuffle_op.py in __init__(self, input_dataset, buffer_size, seed, reshuffle_each_iteration, name)
     47     """See `Dataset.shuffle()` for details."""
     48     self._input_dataset = input_dataset
---> 49     self._buffer_size = ops.convert_to_tensor(
     50         buffer_size, dtype=dtypes.int64, name="buffer_size")
     51     self._seed, self._seed2 = random_seed.get_seed(seed)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    207   overload = getattr(value, "__tf_tensor__", None)
    208   if overload is not None:
--> 209     return overload(dtype, name)  #  pylint: disable=not-callable
    210 
    211   for base_type, conversion_func in get(type(value)):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in __tf_tensor__(self, dtype, name)
    625                 name=name))
    626       return graph.capture(self, name=name)
--> 627     return super().__tf_tensor__(dtype, name)
    628 
    629   def _capture_as_const(self, name) -> Optional[tensor_lib.Tensor]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __tf_tensor__(self, dtype, name)
    759       ) -> "Tensor":
    760     if dtype is not None and not dtype.is_compatible_with(self.dtype):
--> 761       raise ValueError(
    762           _add_error_prefix(
    763               f"Tensor conversion requested dtype {dtype.name} "

ValueError: buffer_size: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=11924>

## === cell 22
try:
    pass
except Exception as e:
    print("Skipped training plots due to:", repr(e))



## === cell 23
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
submission.head()



## === cell 24
test_paths = test_df["image"].to_numpy()
test_n = len(test_paths)

test_ds = _make_ds(
    test_paths, label_idx=None, training=False, augment=False, cache=True
)

X_test_bneck = _extract_bottlenecks_tfdata(test_ds, test_n, with_labels=False)

preds = head_model.predict(
    X_test_bneck,
    verbose=1,
)

idx_to_class = np.asarray(classes, dtype=object)

test_pred_idx = np.argmax(preds, axis=-1)
test_pred_labels = idx_to_class[test_pred_idx]

submission["image"] = [os.path.basename(p) for p in test_paths]
submission["labels"] = test_pred_labels
submission = submission[["image", "labels"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2486986946.py in <cell line: 0>()
      8 )
      9 
---> 10 X_test_bneck = _extract_bottlenecks_tfdata(test_ds, test_n, with_labels=False)
     11 
     12 preds = head_model.predict(

NameError: name '_extract_bottlenecks_tfdata' is not defined
