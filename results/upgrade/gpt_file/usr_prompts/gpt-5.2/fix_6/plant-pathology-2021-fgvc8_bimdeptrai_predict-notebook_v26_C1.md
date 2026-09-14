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

0.7977100646352737

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14534) has done: 'I remove the import that triggers the protobuf/TensorFlow Addons `GetPrototype` crash, and I add a safe model-loading fallback so the notebook can still run when the external `.h5` model file isn’t available in your `/kaggle/input` (the current error). I also fix the submission-generation logic bugs (`==` instead of `=`, chained indexing, and the always-true `if submissions['labels'][i] == ''or 'healthy' in ...`) and make it robust to any class ordering by using the MultiLabelBinarizer classes. Finally, I ensure the script always writes a valid `submission.csv` with the required `image,labels` columns and correct row alignment with the test generator.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash by removing the unnecessary TensorFlow import at the top and using the built-in `tf.keras` APIs only (this is the runtime blocker). Then I remove the untrained “fallback model” path (it produces near-random predictions and explains the very low 0.145 score) and instead load an existing Keras model from any available `.h5`/SavedModel path under `/kaggle/input`, failing fast with a clear error if none is found. Finally, I keep your label-thresholding logic but make it deterministic and aligned to the sample submission ordering, then always write a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.24507) has done: 'Main bottlenecks are (1) double-rescaling the images (decode converts to float in [0,1] and the model rescales again), and (2) expensive in-memory caching of full 256×256 float images for the entire train/val sets, which can cause memory pressure and slowdowns. I make decoding return uint8 and keep the model’s Rescaling layer as-is (same effective normalization), remove the in-memory `.cache()` to avoid RAM thrash, and enable deterministic but faster tf.data pipeline settings (AUTOTUNE parallelism, map/batch optimizations). I also vectorize the submission label post-processing to reduce Python-loop overhead while preserving identical decision logic. These changes keep the model, training loop, loss, thresholds, and semantics intact while reducing end-to-end runtime.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32



## === cell 4
import keras
from keras import layers

keras.utils.set_random_seed(SEED)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
print("Num classes:", len(class_names))
print("Classes:", class_names)
print("Y shape:", Y.shape)



## === cell 6
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_df = train.iloc[trn_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

Y_train = Y[trn_idx]
Y_val = Y[val_idx]

print("Train/Val sizes:", len(train_df), len(val_df))



## === cell 7
import tensorflow as tf

train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"

image_to_index = pd.Series(
    np.arange(len(train), dtype=np.int32), index=train["image"].values
)


def _decode_resize(path):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)  # uint8 [0,255]
    img = tf.image.resize(
        img, [h_target, w_target]
    )  # float32, values remain in [0,255]
    return img


def make_ds(df, image_dir, shuffle, batch_size, labels_arr=None, cache=False):
    filepaths = tf.constant(
        [os.path.join(image_dir, fn) for fn in df["image"].tolist()]
    )

    ds = tf.data.Dataset.from_tensor_slices(filepaths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if labels_arr is not None:
        labels_ds = tf.data.Dataset.from_tensor_slices(
            tf.constant(labels_arr, dtype=tf.float32)
        )
        ds = tf.data.Dataset.zip((ds, labels_ds))

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_indices = image_to_index.loc[train_df["image"].values].to_numpy()
val_indices = image_to_index.loc[val_df["image"].values].to_numpy()

train_labels = Y[train_indices].astype("float32", copy=False)
val_labels = Y[val_indices].astype("float32", copy=False)

train_ds = make_ds(
    train_df,
    train_img_dir,
    shuffle=True,
    batch_size=batch_size,
    labels_arr=train_labels,
    cache=False,
)
val_ds = make_ds(
    val_df,
    train_img_dir,
    shuffle=False,
    batch_size=batch_size,
    labels_arr=val_labels,
    cache=False,
)

test_filepaths = tf.constant(
    [os.path.join(test_img_dir, fn) for fn in submissions["image"].tolist()]
)
test_ds = tf.data.Dataset.from_tensor_slices(test_filepaths)
options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.autotune_buffers = True
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1027407521.py in <cell line: 0>()
     62 val_labels = Y[val_indices].astype("float32", copy=False)
     63 
---> 64 train_ds = make_ds(
     65     train_df,
     66     train_img_dir,

/tmp/ipykernel_11/1027407521.py in make_ds(df, image_dir, shuffle, batch_size, labels_arr, cache)
     34     options.experimental_optimization.map_parallelization = True
     35     options.experimental_optimization.parallel_batch = True
---> 36     options.experimental_optimization.autotune_buffers = True
     37     ds = ds.with_options(options)
     38 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 8
num_classes = len(class_names)

inputs = keras.Input(shape=(h_target, w_target, 3))
x = layers.Rescaling(1.0 / 255.0)(inputs)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 9
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2945222947.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     validation_data=val_ds,
      4     epochs=3,
      5     verbose=1,

NameError: name 'train_ds' is not defined

## === cell 10
preds = model.predict(test_ds, verbose=1)
preds = np.asarray(preds)
print("preds shape:", preds.shape)

if len(submissions) != preds.shape[0]:
    raise RuntimeError(
        f"Prediction rows ({preds.shape[0]}) do not match submission rows ({len(submissions)}). "
        "Check test dataset ordering."
    )



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4103557623.py in <cell line: 0>()
----> 1 preds = model.predict(test_ds, verbose=1)
      2 preds = np.asarray(preds)
      3 print("preds shape:", preds.shape)
      4 
      5 if len(submissions) != preds.shape[0]:

NameError: name 'test_ds' is not defined

## === cell 11
thresh = 0.25
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

argmax_idx = preds.argmax(axis=1).astype(np.int32)
mask_thresh = preds >= thresh  # (N, C) boolean

out_labels = []
for i in range(preds.shape[0]):
    row_argmax = int(argmax_idx[i])

    if healthy_idx is not None and row_argmax == healthy_idx:
        out_labels.append("healthy")
        continue

    chosen_idx = np.flatnonzero(mask_thresh[i])
    chosen = [class_names[j] for j in chosen_idx.tolist()]

    if (len(chosen) == 0) or ("healthy" in chosen):
        chosen = [class_names[row_argmax]]

    out_labels.append(" ".join(chosen))

submissions = submissions.copy()
submissions["labels"] = out_labels



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2943151528.py in <cell line: 0>()
      3 healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
      4 
----> 5 argmax_idx = preds.argmax(axis=1).astype(np.int32)
      6 mask_thresh = preds >= thresh  # (N, C) boolean
      7 

NameError: name 'preds' is not defined

## === cell 12
submissions[["image", "labels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
submissions.head()



## === cell 13
submissions
