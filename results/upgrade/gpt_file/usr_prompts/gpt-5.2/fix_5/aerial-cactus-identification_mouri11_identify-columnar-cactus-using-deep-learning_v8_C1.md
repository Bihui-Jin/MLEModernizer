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

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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

0.914

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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from os.path import join
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

BASE_DIR_CANDIDATES = [
    "../input/aerial-cactus-identification/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
]
BASE_DIR = None
for cand in BASE_DIR_CANDIDATES:
    if (
        os.path.isfile(join(cand, "train.csv"))
        and os.path.isdir(join(cand, "train"))
        and os.path.isdir(join(cand, "test"))
    ):
        BASE_DIR = cand
        break
if BASE_DIR is None:
    raise FileNotFoundError(
        f"Could not locate dataset root. Tried: {BASE_DIR_CANDIDATES}"
    )


def _resolve_img_dir(base_dir, split):
    cand1 = join(base_dir, split, split)  # e.g., train/train and test/test
    cand2 = join(base_dir, split)  # e.g., train/ and test/
    if os.path.isdir(cand1):
        return cand1
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(
        f"Could not find image dir for split='{split}'. Tried: {cand1} and {cand2}"
    )


train_img_dir = _resolve_img_dir(BASE_DIR, "train")
test_img_dir = _resolve_img_dir(BASE_DIR, "test")

train_data = pd.read_csv(join(BASE_DIR, "train.csv"))
sample_sub = pd.read_csv(join(BASE_DIR, "sample_submission.csv"))

img_size = 32

train_img_paths = [join(train_img_dir, img_id) for img_id in train_data["id"].values]
test_ids = sample_sub["id"].values.tolist()
test_img_paths = [join(test_img_dir, img_id) for img_id in test_ids]

assert os.path.isdir(train_img_dir), f"Train dir not found: {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Test dir not found: {test_img_dir}"

missing_train = [p for p in train_img_paths[:50] if not os.path.isfile(p)]
missing_test = [p for p in test_img_paths[:50] if not os.path.isfile(p)]
assert (
    len(missing_train) == 0
), f"Some train images missing (first 50 checked), e.g.: {missing_train[:3]}"
assert (
    len(missing_test) == 0
), f"Some test images missing (first 50 checked), e.g.: {missing_test[:3]}"

y_train = train_data["has_cactus"].astype(np.int64).values


def make_ds(paths, labels=None, batch_size=100, training=False):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)

    if labels is not None:
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((path_ds, label_ds))
    else:
        ds = path_ds

    def _load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    if labels is not None:

        def _map_fn(path, label):
            return _load_image(path), tf.cast(label, tf.int64)

    else:

        def _map_fn(path):
            return _load_image(path)

    if training:
        ds = ds.shuffle(
            buffer_size=min(10000, len(paths)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(train_img_paths, y_train, batch_size=100, training=True)

model = Sequential()
model.add(
    Conv2D(
        25,
        kernel_size=2,
        strides=2,
        activation="relu",
        input_shape=(img_size, img_size, 3),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(25, kernel_size=2, strides=2, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(250, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.summary()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
n = len(train_img_paths)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

split = int(n * 0.8)
train_idx = idx[:split]
val_idx = idx[split:]

train_paths_split = [train_img_paths[i] for i in train_idx]
val_paths_split = [train_img_paths[i] for i in val_idx]
y_train_split = y_train[train_idx]
y_val_split = y_train[val_idx]

train_ds_split = make_ds(
    train_paths_split, y_train_split, batch_size=100, training=True
)
val_ds_split = make_ds(val_paths_split, y_val_split, batch_size=100, training=False)

model.fit(train_ds_split, epochs=4, validation_data=val_ds_split)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3972354315.py in <cell line: 0>()
     10 train_paths_split = [train_img_paths[i] for i in train_idx]
     11 val_paths_split = [train_img_paths[i] for i in val_idx]
---> 12 y_train_split = y_train[train_idx]
     13 y_val_split = y_train[val_idx]
     14 

NameError: name 'y_train' is not defined

## === cell 2
test_ds = make_ds(test_img_paths, labels=None, batch_size=256, training=False)

preds_temp = model.predict(test_ds, verbose=1)
preds = preds_temp[:, 1].astype(float)

output = pd.DataFrame({"id": test_ids, "has_cactus": preds})
output = output.set_index("id").loc[sample_sub["id"].values].reset_index()

output.to_csv("submission.csv", index=False)

print(output.head())
print("Wrote submission.csv with shape:", output.shape)
assert os.path.isfile("submission.csv"), "submission.csv was not created"
assert list(output.columns) == [
    "id",
    "has_cactus",
], f"Wrong submission columns: {output.columns.tolist()}"
assert len(output) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission.csv"

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4042410649.py in <cell line: 0>()
----> 1 test_ds = make_ds(test_img_paths, labels=None, batch_size=256, training=False)
      2 
      3 preds_temp = model.predict(test_ds, verbose=1)
      4 preds = preds_temp[:, 1].astype(float)
      5 

NameError: name 'make_ds' is not defined
