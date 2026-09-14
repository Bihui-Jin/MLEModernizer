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

# 5. Target score

0.8998066739154611

# 6. Current score

0.76345

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.78588) has done: 'The timeout is dominated by Python-side image loading/resizing in `ImageDataGenerator.flow_from_dataframe` and by Keras generator overhead each step. I keep the exact same model, loss, preprocessing function, epochs, and data split, but switch the input pipeline to a deterministic `tf.data` pipeline that decodes PNGs, resizes, and applies the same `efficientnet.preprocess_input` with parallel mapping, caching, prefetching, and static shapes. This preserves evaluation semantics while removing most per-step Python overhead and improving CPU utilization. I also keep prediction logic identical but run it through the same fast `tf.data` pipeline with a larger batch.'
- What this solution (achieved 0.76864) has done: 'The crash is happening before any of your code really runs: setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` forces the pure-Python protobuf backend, which is incompatible with the protobuf version bundled in this Kaggle TensorFlow environment and triggers the `MessageFactory.GetPrototype` error during TensorFlow import. I remove that environment override so TensorFlow can use the default (C++) protobuf implementation, keeping the rest of your pipeline (tf.data loading, EfficientNetB0 frozen, loss/epochs, prediction and submission formatting) unchanged. I also add a small defensive check to fall back to the alternate `/kaggle/input/...` base path if `../input/...` isn’t present, without changing any I/O semantics when your current path works. This should run end-to-end and write a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.75715) has done: 'The crash happens during `import tensorflow as tf` due to an incompatible protobuf backend being selected implicitly in the environment; the minimal fix is to force TensorFlow to use the Python protobuf implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` attribute error in this Kaggle runtime. I keep your tf.data pipeline, model architecture (frozen EfficientNetB0 + dropout + dense softmax), training loop, and submission formatting unchanged. I also keep the existing base-path fallback logic and ensure the output is written to `submission.csv` with the required columns. These changes are execution/stability fixes and should also allow the model to train properly again, which is necessary to move the score upward toward your target.'
- What this solution (achieved 0.7443) has done: 'The crash is caused by forcing TensorFlow to use the pure-Python protobuf backend (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`), which is incompatible with the Kaggle TF/protobuf build and triggers the `MessageFactory.GetPrototype` import error. I remove those environment overrides so TensorFlow can import normally, keeping your tf.data pipeline, EfficientNetB0 frozen model, training loop, and submission formatting unchanged. This is the minimal stability fix needed to run end-to-end and should also improve your score simply by allowing training to actually execute correctly again. I also keep your existing base-path fallback and ensure `submission.csv` is written with `id_code,diagnosis`.'
- What this solution (achieved 0.75545) has done: 'I fix the root runtime failure by removing the protobuf environment-variable overrides that are forcing an incompatible protobuf implementation, which currently prevents TensorFlow from importing at all. Once TensorFlow imports successfully, the downstream `NameError`s (missing `df`, `tf`, and file paths) resolve automatically because those variables are defined in the first cell. I also keep a safe base-path fallback exactly as you intended so the CSV/image paths work in both `/kaggle/input/...` and relative setups. No model/training logic, architecture, loss, preprocessing, or submission semantics are changed—this is primarily an execution/stability fix to ensure a valid `submission.csv` is produced end-to-end.'
- What this solution (achieved 0.76345) has done: 'The immediate blocker is that TensorFlow can’t import because protobuf’s compiled extension (`google.protobuf.pyext._message`) is missing in this runtime; forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp` triggers that path. I remove that override and instead force the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the crash and restores downstream variables (`df`, `tf`, file paths) so later cells run. I keep your tf.data pipeline, EfficientNetB0 frozen head, loss, epochs, preprocessing, and submission formatting unchanged, only adding a small base-path fallback that matches your existing intent. Finally, the script write a valid `submission.csv` with `id_code,diagnosis`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2  # kept to preserve original environment/imports

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

BASE_PATH = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/kaggle/data/aptos2019-blindness-detection"

train_dir = os.path.join(BASE_PATH, "train_images")
test_dir = os.path.join(BASE_PATH, "test_images")

train_csv_path = os.path.join(BASE_PATH, "train.csv")
test_csv_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

df = pd.read_csv(train_csv_path)
df["name"] = df["id_code"].astype(str) + ".png"
df["diagnosis"] = df["diagnosis"].astype(str)

BS = 16
IMG_SIZE = 300
SIZE = (IMG_SIZE, IMG_SIZE)

if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
    raise FileNotFoundError(
        f"Image directories not found. BASE_PATH={BASE_PATH}, train_dir={train_dir}, test_dir={test_dir}"
    )

df.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

train, val = train_test_split(
    df, test_size=0.1, random_state=SEED, shuffle=True, stratify=df["diagnosis"]
)

preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

class_names = sorted(train["diagnosis"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(class_names)}
NUM_CLASSES = 5

train_paths = (train_dir + "/" + train["name"].values).astype(str)
val_paths = (train_dir + "/" + val["name"].values).astype(str)

train_labels = train["diagnosis"].map(class_to_index).astype(np.int32).values
val_labels = val["diagnosis"].map(class_to_index).astype(np.int32).values


def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


def _make_dataset(paths, labels=None, training=False, batch_size=BS):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    def _map_with_label(path, y):
        x = _decode_resize_preprocess(path)
        y = tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)
        return x, y

    if labels is None:
        ds = ds.map(
            _decode_resize_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    else:
        ds = ds.map(
            _map_with_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_set = _make_dataset(train_paths, train_labels, training=True, batch_size=BS)
val_set = _make_dataset(val_paths, val_labels, training=False, batch_size=BS)

print(
    f"Found {len(train_paths)} validated image filenames belonging to {NUM_CLASSES} classes."
)
print(
    f"Found {len(val_paths)} validated image filenames belonging to {NUM_CLASSES} classes."
)



## === cell 2
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    pooling="avg",
)
base.trainable = False  # keep training fast and stable

x = tf.keras.layers.Dropout(0.2)(base.output)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 2

train_steps = int(np.ceil(len(train_paths) / BS))
val_steps = int(np.ceil(len(val_paths) / BS))

history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 3
sub = pd.read_csv(test_csv_path)

test_df = sub.copy()
test_df["name"] = test_df["id_code"].astype(str) + ".png"

test_paths = (test_dir + "/" + test_df["name"].values).astype(str)

PRED_BS = BS * 4
test_gen = _make_dataset(test_paths, labels=None, training=False, batch_size=PRED_BS)

pred_probs = model.predict(
    test_gen,
    verbose=0,
)

results = pred_probs.argmax(axis=1).astype(int).tolist()

len(results), results[:10]



## === cell 4
pred_df = pd.read_csv(sample_sub_path)
pred_df = pred_df.set_index("id_code").reindex(sub["id_code"]).reset_index()

pred_df["diagnosis"] = results
pred_df.to_csv("submission.csv", index=False)

pred_df.head()
