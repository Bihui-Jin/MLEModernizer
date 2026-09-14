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

0.7699722991689769

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.2461) has done: 'I remove the failing `tensorflow_addons` import (it triggers the protobuf `MessageFactory.GetPrototype` error in this environment) since it isn’t used by your pipeline. I also fix the missing external model file path by building the same type of lightweight MobileNetV2-based model inside the notebook and running inference end-to-end, so a submission CSV is always produced. Finally, I correct the label post-processing bugs (`==` vs `=`, chained assignment, and the logic that accidentally always triggers) and ensure predictions are aligned with class names from `train.csv`, producing space-delimited labels in the required format.'
- What this solution (achieved 0.289) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the training `ValueError: None values not supported` by making the training/validation generators provide labels directly (instead of `y_col=None`) and by passing `valid_generator` normally to `validation_data`. Finally, to move the score upward toward your target without changing the core model/training approach, I train for a few epochs (still lightweight MobileNetV2) and use a slightly more standard multilabel threshold (0.5) with a safe fallback to at least one label.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by slow, single-threaded image loading/resizing inside a Python `Sequence`, plus 512×512 inputs which are expensive for MobileNetV2; we cannot change the latter without changing core logic, so we must remove Python/I/O overhead. I replace the custom `Sequence` with an equivalent `tf.data` pipeline that uses parallel JPEG decode/resize, caching (in-memory) for train/valid, and prefetching, while preserving the same train/valid split, normalization, batching, and shuffle semantics. I also enable deterministic TF execution and set `steps_per_execution` to reduce Python→TF call overhead without changing the training loop semantics. Prediction is likewise switched to `tf.data` with parallelism and prefetch to speed up inference.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/plant-pathology-2021-fgvc8"
if not os.path.exists(os.path.join(BASE_DIR, "train.csv")):
    BASE_DIR = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
print(train.head())



## === cell 2
h_target = 512
w_target = 512
batch_size = 32

label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
class_names = list(mlb.classes_)
n_classes = len(class_names)

print("n_classes:", n_classes)
print("classes:", class_names)



## === cell 3
train_df = train.copy()
train_df["labels_list"] = train_df["labels"].apply(lambda x: x.split())
Y = mlb.transform(train_df["labels_list"].values).astype("float32")

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_images = train_df.loc[trn_idx, "image"].values
valid_images = train_df.loc[val_idx, "image"].values
Y_train = Y[trn_idx]
Y_valid = Y[val_idx]


def _build_image_ds(
    image_names, y, image_dir, batch_size, target_size, shuffle, seed, cache
):
    image_names = tf.convert_to_tensor(image_names, dtype=tf.string)
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(image_names)
    else:
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((image_names, y))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=tf.shape(image_names)[0],
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _load_only(name):
        path = tf.strings.join([tf.constant(image_dir + os.sep), name])
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _load_xy(name, label):
        return _load_only(name), label

    if y is None:
        ds = ds.map(_load_only, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(_load_xy, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _build_image_ds(
    train_images,
    Y_train,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=True,
    seed=SEED,
    cache=True,
)
valid_ds = _build_image_ds(
    valid_images,
    Y_valid,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
    cache=True,
)
test_ds = _build_image_ds(
    submissions["image"].values,
    None,
    TEST_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
    cache=False,
)

train_steps = int(np.ceil(len(train_images) / batch_size))
valid_steps = int(np.ceil(len(valid_images) / batch_size))
test_steps = int(np.ceil(len(submissions) / batch_size))

print(
    "Train batches:",
    train_steps,
    "Valid batches:",
    valid_steps,
    "Test batches:",
    test_steps,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2145202779.py in <cell line: 0>()
     62 
     63 
---> 64 train_ds = _build_image_ds(
     65     train_images,
     66     Y_train,

/tmp/ipykernel_11/2145202779.py in _build_image_ds(image_names, y, image_dir, batch_size, target_size, shuffle, seed, cache)
     31     if shuffle:
     32         # Match epoch-wise reshuffle behavior of Sequence.on_epoch_end()
---> 33         ds = ds.shuffle(
     34             buffer_size=tf.shape(image_names)[0],
     35             seed=seed,

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

ValueError: buffer_size: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=13415>

## === cell 4
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
inputs = keras.Input(shape=(h_target, w_target, 3))
x = base(inputs, training=False)
outputs = keras.layers.Dense(n_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=8,
)

EPOCHS = 3
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=valid_ds,
    validation_steps=valid_steps,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2352540564.py in <cell line: 0>()
     20 EPOCHS = 3
     21 history = model.fit(
---> 22     train_ds,
     23     epochs=EPOCHS,
     24     steps_per_epoch=train_steps,

NameError: name 'train_ds' is not defined

## === cell 5
preds = model.predict(test_ds, steps=test_steps, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4003717277.py in <cell line: 0>()
      1 # --- Speed: predict from tf.data with prefetch/parallel decode already enabled above.
----> 2 preds = model.predict(test_ds, steps=test_steps, verbose=1)
      3 print("preds shape:", preds.shape)
      4 print(preds[:2])
      5 

NameError: name 'test_ds' is not defined

## === cell 6
thresh = 0.5

pred_labels = []
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

for i in range(preds.shape[0]):
    p = preds[i]
    top_idx = int(np.argmax(p))

    chosen = np.where(p >= thresh)[0]
    lab_list = [class_names[j] for j in chosen]

    if len(lab_list) == 0:
        lab_list = [class_names[top_idx]]

    if "healthy" in lab_list and len(lab_list) > 1:
        lab_list = [l for l in lab_list if l != "healthy"]
        if len(lab_list) == 0:
            lab_list = ["healthy"]

    lab = " ".join(lab_list)
    pred_labels.append(lab)

submissions = submissions.copy()
submissions["labels"] = pred_labels
print(submissions.head())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/938483319.py in <cell line: 0>()
      4 healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
      5 
----> 6 for i in range(preds.shape[0]):
      7     p = preds[i]
      8     top_idx = int(np.argmax(p))

NameError: name 'preds' is not defined

## === cell 7
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions.head(10))
