# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf  # must be imported after the env var

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"jit_compile": True})

cpu_count = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical




## === cell 1
train_dir = "../input/aptos2019-blindness-detection/train_images"
train_csv = "../input/aptos2019-blindness-detection/train.csv"
test_csv = "../input/aptos2019-blindness-detection/test.csv"
sample_submission_path = "../input/aptos2019-blindness-detection/sample_submission.csv"

df = pd.read_csv(train_csv)
df["name"] = df["id_code"].apply(lambda x: f"{x}.png")
df["diagnosis"] = df["diagnosis"].astype(str)

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)




## === cell 2
BS = 16
IMG_SIZE = 300
SIZE = (IMG_SIZE, IMG_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def load_and_preprocess(path):
    """
    Read a PNG file, resize to IMG_SIZE, and apply EfficientNet preprocessing.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, SIZE)
    img = preprocess_input(img)
    return img


def make_dataset(df, directory, shuffle=False):
    """
    Build a TF dataset that streams images from disk, applies preprocessing,
    and pairs them with one‑hot labels. Caching is omitted to keep memory usage low.
    """
    paths = df["name"].apply(lambda x: os.path.join(directory, x)).values
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)

    images_ds = paths_ds.map(
        lambda p: load_and_preprocess(p), num_parallel_calls=AUTOTUNE
    )  # no .cache() to avoid huge RAM usage

    labels = to_categorical(df["diagnosis"].astype(int), num_classes=5).astype(
        np.float32
    )
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)  # no .cache()

    ds = tf.data.Dataset.zip((images_ds, labels_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    ds = ds.batch(BS).prefetch(AUTOTUNE)
    return ds


train_set = make_dataset(train_df, train_dir, shuffle=True)
val_set = make_dataset(val_df, train_dir, shuffle=False)




## === cell 3
base = tf.keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
output = tf.keras.layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_set,
    validation_data=val_set,
    epochs=5,
    verbose=1,
)




## === cell 4
test_df = pd.read_csv(test_csv)
test_dir = "../input/aptos2019-blindness-detection/test_images/"

test_df["name"] = test_df["id_code"].apply(lambda x: f"{x}.png")
test_paths = test_df["name"].apply(lambda x: os.path.join(test_dir, x)).values

test_paths_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_images_ds = test_paths_ds.map(
    lambda p: load_and_preprocess(p), num_parallel_calls=AUTOTUNE
)
test_ds = test_images_ds.batch(BS).prefetch(AUTOTUNE)

preds = model.predict(test_ds, verbose=0)
results = np.argmax(preds, axis=1).astype(int).tolist()




## === cell 5
submission = pd.read_csv(sample_submission_path)
submission["diagnosis"] = results
submission.to_csv("submission.csv", index=False)
submission
