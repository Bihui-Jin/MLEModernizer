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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.7423

# 6. Current score

0.972

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.972) has done: 'I update the notebook to be compatible with TensorFlow 2.18 by removing deprecated TF1 eager calls and replacing deprecated APIs (`tf.read_file`, `tf.image.resize_images`, `tf.data.experimental.shuffle_and_repeat`). I also fix the input paths to match your provided dataset layout (`/kaggle/input/aerial-cactus-identification/...`) so images are actually found, and ensure labels are the right dtype for `sparse_categorical_crossentropy`. Finally, I fix inference to output a proper probability for `has_cactus` (positive class) instead of an argmax class label, which should substantially improve ROC AUC toward your target while keeping the same model and training approach.'

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl

np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(SAMPLE_SUB)

train_image_names = train["id"].astype(str)
ytrain = train["has_cactus"].astype(np.int32)

train_image_paths = (TRAIN_DIR + "/" + train_image_names).values

assert len(train_image_paths) == len(train), "Train paths/labels length mismatch"
assert os.path.exists(
    train_image_paths[0]
), f"Train image not found: {train_image_paths[0]}"




## === cell 2
def preprocess_image(image_bytes):
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_and_preprocess_image(path):
    image_bytes = tf.io.read_file(path)
    return preprocess_image(image_bytes)




## === cell 3
path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
img_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

labels_ds = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(ytrain.values, dtype=tf.int32)
)
ds_label_ds = tf.data.Dataset.zip((img_ds, labels_ds))

ds_label_ds = ds_label_ds.shuffle(
    buffer_size=len(train), reshuffle_each_iteration=True
).repeat()
ds_label_ds = ds_label_ds.batch(30).prefetch(AUTOTUNE)



## === cell 4
model = km.Sequential(
    [
        kl.Conv2D(
            input_shape=(32, 32, 3),
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            filters=3,
        ),
        kl.Flatten(),
        kl.Dense(units=2500, activation=tf.nn.relu),
        kl.Dense(units=500, activation=tf.nn.relu),
        kl.Dense(units=500, activation=tf.nn.relu),
        kl.Dense(units=100, activation=tf.nn.relu),
        kl.Dropout(rate=0.2),
        kl.Dense(units=25, activation=tf.nn.relu),
        kl.Dense(units=10, activation=tf.nn.relu),
        kl.Dense(units=2, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"],
)



## === cell 5
history = model.fit(ds_label_ds, epochs=15, steps_per_epoch=len(train) // 60, verbose=1)



## === cell 6
test_image_names = test["id"].astype(str)
test_image_paths = (TEST_DIR + "/" + test_image_names).values
assert os.path.exists(
    test_image_paths[0]
), f"Test image not found: {test_image_paths[0]}"

test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = (
    test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(30)
    .prefetch(AUTOTUNE)
)



## === cell 7
pre = model.predict(test_ds, verbose=1)
has_cactus_prob = pre[:, 1].astype(np.float64)

submission = pd.DataFrame(
    {"id": test_image_names.values, "has_cactus": has_cactus_prob}
)
submission.to_csv("submission_2.csv", index=False)

print(submission.head())
print("Wrote:", pathlib.Path("submission_2.csv").resolve())
