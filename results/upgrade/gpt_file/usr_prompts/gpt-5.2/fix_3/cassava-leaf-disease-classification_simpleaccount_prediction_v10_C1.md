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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.5480507706255666

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'Main runtime is dominated by CPU image decoding/resizing inside `Sequence.__getitem__` (Python loop + cv2 I/O) and by extra overhead during prediction. To keep identical model/training logic while cutting wall time, I switch both train/test pipelines to `tf.data` with parallel JPEG decode/resize, cached file-path tensors, deterministic options, and `prefetch(AUTOTUNE)` so the model is continuously fed. I also eliminate repeated pandas `.loc` lookups in inner loops by precomputing image paths/labels arrays (equivalent data, less Python overhead), and I keep batch size/epochs/model architecture unchanged. These changes are provably equivalent in semantics (same files, same resizing to 224×224, same normalization /255, same labels), but significantly reduce input pipeline bottlenecks to fit within 600s.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import cv2
from math import ceil

from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gm_exp = tf.Variable(3.0, dtype=tf.float32)


def generalized_mean_pool_2d(X):
    pool = (
        tf.reduce_mean(tf.abs(X ** (gm_exp)), axis=[1, 2], keepdims=False) + 1.0e-7
    ) ** (1.0 / gm_exp)
    return pool


def create_model(input_shape):
    inp = Input(shape=input_shape)

    x_model = ResNet50(
        weights=None, include_top=False, input_tensor=inp, pooling=None, classes=None
    )
    for layer in x_model.layers:
        layer.trainable = True

    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])

    x = lambda_layer(x_model.output)
    out = Dense(5, activation="softmax", name="plan_diseases")(x)
    model = Model(inputs=x_model.input, outputs=out)
    return model




## === cell 2
path = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(path, "train.csv")
sample_sub_path = os.path.join(path, "sample_submission.csv")
train_img_dir = os.path.join(path, "train_images") + "/"
test_img_dir = os.path.join(path, "test_images") + "/"

train_df = pd.read_csv(train_csv_path)
sample_submission = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape, "sample_submission:", sample_submission.shape)
print(train_df.head())




## === cell 3
def _read(path_):
    img = cv2.imread(path_)
    return img


def _build_dataset_from_paths(
    image_paths,
    labels=None,
    batch_size=16,
    img_size=(224, 224),
    shuffle=False,
    seed=42,
):
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)

    if labels is not None:
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(image_paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=tf.shape(image_paths)[0],
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _decode_resize_norm(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(
            img, img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    if labels is not None:

        def _map_fn(path, y):
            return _decode_resize_norm(path), y

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(
            _decode_resize_norm, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_set,
        ids,
        augmentation=None,
        img_dir="../input/cassava-leaf-disease-classification/test_images/",
        batch_size=8,
        img_size=(224, 224, 3),
        do_aug=False,
    ):
        self.X = X_set.reset_index(drop=True)
        self.batch_size = batch_size
        self.ids = np.array(list(ids))
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()
        self.augmentation = augmentation
        self.do_aug = do_aug

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        X = self.__generator__(batch_ids)
        return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))

    def __generator__(self, batch_ids):
        bsz = len(batch_ids)
        X = np.empty((bsz, *self.img_size), dtype=np.float32)
        for i, idx in enumerate(batch_ids):
            image_id = self.X.loc[idx, "image_id"]
            image = cv2.imread(self.img_dir + image_id)
            if image is None:
                image = np.zeros(
                    (self.img_size[0], self.img_size[1], 3), dtype=np.uint8
                )
            else:
                image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
            X[i] = image / 255.0
        return X


class TrainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, img_dir, batch_size=8, img_size=(224, 224, 3), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bsz = len(batch_idx)
        X = np.empty((bsz, *self.img_size), dtype=np.float32)
        y = np.empty((bsz,), dtype=np.int32)

        for i, di in enumerate(batch_idx):
            image_id = self.df.loc[di, "image_id"]
            label = int(self.df.loc[di, "label"])
            image = cv2.imread(self.img_dir + image_id)
            if image is None:
                image = np.zeros(
                    (self.img_size[0], self.img_size[1], 3), dtype=np.uint8
                )
            else:
                image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
            X[i] = image / 255.0
            y[i] = label
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)




## === cell 4
path_weights = "../input/weights/"
has_external_weights = os.path.isdir(path_weights) and len(os.listdir(path_weights)) > 0
print("External weights found:", has_external_weights)




## === cell 5
input_shape = (224, 224, 3)
batch_size = 16

X_test = sample_submission.copy()
if "label" in X_test.columns:
    X_test = X_test.drop(columns=["label"])

test_image_paths = (test_img_dir + X_test["image_id"].astype(str)).to_numpy()
data_generator_test = _build_dataset_from_paths(
    test_image_paths,
    labels=None,
    batch_size=batch_size,
    img_size=(input_shape[0], input_shape[1]),
    shuffle=False,
    seed=SEED,
)




## === cell 6
all_pred = []

if has_external_weights:
    models = sorted(os.listdir(path_weights))
    for w in models:
        model = create_model(input_shape)
        model.load_weights(os.path.join(path_weights, w))
        preds = model.predict(data_generator_test, verbose=1)
        preds = preds[: sample_submission.shape[0]]
        all_pred.append(preds)
else:
    model = create_model(input_shape)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    val_frac = 0.1
    n = len(train_df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - val_frac))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    tr_paths = (train_img_dir + tr_df["image_id"].astype(str)).to_numpy()
    tr_labels = tr_df["label"].astype(np.int32).to_numpy()
    va_paths = (train_img_dir + va_df["image_id"].astype(str)).to_numpy()
    va_labels = va_df["label"].astype(np.int32).to_numpy()

    train_gen = _build_dataset_from_paths(
        tr_paths,
        labels=tr_labels,
        batch_size=batch_size,
        img_size=(input_shape[0], input_shape[1]),
        shuffle=True,
        seed=SEED,
    )
    val_gen = _build_dataset_from_paths(
        va_paths,
        labels=va_labels,
        batch_size=batch_size,
        img_size=(input_shape[0], input_shape[1]),
        shuffle=False,
        seed=SEED,
    )

    model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=1)

    preds = model.predict(data_generator_test, verbose=1)
    preds = preds[: sample_submission.shape[0]]
    all_pred.append(preds)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3851770263.py in <cell line: 0>()
     33     va_labels = va_df["label"].astype(np.int32).to_numpy()
     34 
---> 35     train_gen = _build_dataset_from_paths(
     36         tr_paths,
     37         labels=tr_labels,

/tmp/ipykernel_11/1621275586.py in _build_dataset_from_paths(image_paths, labels, batch_size, img_size, shuffle, seed)
     30     if shuffle:
     31         # Same semantics as shuffling indices each epoch; seed keeps results stable.
---> 32         ds = ds.shuffle(
     33             buffer_size=tf.shape(image_paths)[0],
     34             seed=seed,

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

ValueError: buffer_size: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=16848>

## === cell 7
sum_pred = np.zeros_like(all_pred[0])
for p in all_pred:
    sum_pred += p
sum_pred = sum_pred / len(all_pred)

diagnos = [int(np.argmax(pr)) for pr in sum_pred]

sample_submission["label"] = diagnos
print(sample_submission.head())
print("Label value counts:\n", sample_submission["label"].value_counts())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3744347942.py in <cell line: 0>()
----> 1 sum_pred = np.zeros_like(all_pred[0])
      2 for p in all_pred:
      3     sum_pred += p
      4 sum_pred = sum_pred / len(all_pred)
      5 

IndexError: list index out of range

## === cell 8
sample_submission[["image_id", "label"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:", sample_submission[["image_id", "label"]].shape
)
print("submission.csv preview:")
print(pd.read_csv("submission.csv").head())
