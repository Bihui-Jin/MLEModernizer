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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.6613428307611109

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick optimal
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].fillna("")
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)

print("Num classes:", num_classes)
print("Example classes:", dataset_labels[:10])




## === cell 2
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=image_dims,
    pooling="avg",
)
base.trainable = False

inputs = tf.keras.Input(shape=image_dims, name="image")
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)
model.summary()




## === cell 3
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

y = one_hot.reindex(columns=dataset_labels, fill_value=0).astype("float32").values
x_paths = data_set["image"].apply(lambda x: os.path.join(train_img_dir, x)).values

idx = np.arange(len(x_paths))
np.random.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

x_tr, y_tr = x_paths[tr_idx], y[tr_idx]
x_va, y_va = x_paths[val_idx], y[val_idx]


@tf.function
def _decode_resize_float32(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_and_crop_jpeg(
        img_bytes, crop_window=[0, 0, 0x7FFFFFFF, 0x7FFFFFFF], channels=3
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(
        img,
        [image_dims[0], image_dims[1]],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    return img


@tf.function
def load_and_preprocess(path, label):
    return _decode_resize_float32(path), label


@tf.function
def preprocess_image_for_infer(path):
    return _decode_resize_float32(path)


batch_size = 32

options = tf.data.Options()
options.experimental_deterministic = True

ds_tr = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
ds_tr = ds_tr.with_options(options)
ds_tr = ds_tr.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
ds_tr = ds_tr.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_tr = ds_tr.batch(batch_size, drop_remainder=False)
ds_tr = ds_tr.prefetch(tf.data.AUTOTUNE)

ds_va = tf.data.Dataset.from_tensor_slices((x_va, y_va))
ds_va = ds_va.with_options(options)
ds_va = ds_va.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_va = ds_va.batch(batch_size, drop_remainder=False)
ds_va = ds_va.prefetch(tf.data.AUTOTUNE)

print("Train batches:", tf.data.experimental.cardinality(ds_tr))
print("Val batches:", tf.data.experimental.cardinality(ds_va))




## === cell 4
epochs = 2
history = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, verbose=2)

images_path_list = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
)
test_paths = [os.path.join(test_dir, f) for f in images_path_list]

ds_te = tf.data.Dataset.from_tensor_slices(test_paths)
ds_te = ds_te.with_options(
    tf.data.Options()
)  # keep default; inference order is already fixed by sorted filenames
ds_te = ds_te.map(preprocess_image_for_infer, num_parallel_calls=tf.data.AUTOTUNE)
ds_te = ds_te.batch(32, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

pred = model.predict(ds_te, verbose=0)  # shape: (N, num_classes)

thr = 0.6  # keep core behavior (thresholded multi-label, fallback to healthy)
mask = pred > thr  # (N, C) boolean
dataset_labels_arr = np.asarray(dataset_labels, dtype=object)

any_pos = mask.any(axis=1)
labels_out = np.full((mask.shape[0],), "healthy", dtype=object)
pos_idx = np.where(any_pos)[0]
for i in pos_idx.tolist():
    labels_out[i] = " ".join(dataset_labels_arr[mask[i]].tolist()).strip()

csv_pd = pd.DataFrame({"image": images_path_list, "labels": labels_out.tolist()})
sub_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print("Rows:", len(csv_pd), "Cols:", list(csv_pd.columns))
print(csv_pd.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_56/1460913794.py in <cell line: 0>()
      1 epochs = 2
----> 2 history = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, verbose=2)
      3 
      4 images_path_list = sorted(
      5     [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]

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

Detected at node DecodeAndCropJpeg defined at (most recent call last):
<stack traces unavailable>
Detected at node DecodeAndCropJpeg defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  jpeg::Uncompress failed. Invalid JPEG data or crop window.
	 [[{{node DecodeAndCropJpeg}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) INVALID_ARGUMENT:  jpeg::Uncompress failed. Invalid JPEG data or crop window.
	 [[{{node DecodeAndCropJpeg}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_16281]
