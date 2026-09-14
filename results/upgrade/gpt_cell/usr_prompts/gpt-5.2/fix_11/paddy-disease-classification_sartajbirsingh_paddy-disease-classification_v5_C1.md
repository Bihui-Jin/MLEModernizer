# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.89976

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08801) has done: 'Diagnosis: The crash happens immediately on importing TensorFlow/Keras in cell 0, before any notebook logic runs. With TensorFlow 2.18.0 and protobuf 6.33.0 in this environment, a known incompatibility can trigger `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow’s protobuf initialization. The minimal reliable fix is to force TensorFlow to use the pure-Python protobuf implementation (instead of the C++ one) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **before** importing TensorFlow. This keeps all model/training logic unchanged and unblocks execution.

Patch summary: Edit only cell 0 to set the protobuf implementation environment variable prior to importing `tensorflow`/`keras`, leaving all other imports and logic intact.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: No variable names or outputs change; `train_df = pd.read_csv(...)` in cell 1 run as before once TensorFlow imports successfully.

Assumptions: It is acceptable to use the Python protobuf backend for compatibility; this does not change notebook semantics besides avoiding the import-time crash.'
- What this solution (achieved 0.01691) has done: 'The timeout is dominated by two things: (1) loading and holding the entire 7.8k-image training set and 2.6k test set in RAM as NumPy arrays, and (2) spending 30 epochs training a large dense network while feeding data from Python/NumPy. To keep the *same model and training semantics* but run faster, I switch to a fully streaming `tf.data` input pipeline that decodes/resizes images on the fly with parallelism, caching, and prefetch (so we never build `X`/`X_test` big arrays). I also remove the pip/protobuf install + forced kernel shutdown (very expensive and unnecessary here) and keep determinism by using deterministic dataset options and fixed seeds. Predictions are streamed as well and mapped to labels identically.'
- What this solution (achieved 0.01691) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0, before any model code runs. With `tensorflow==2.18.0` and `protobuf==6.33.0`, TensorFlow can fail at import time due to an incompatibility between TF’s generated protobuf code and protobuf 6.x, surfacing as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is an environment/API mismatch, not a logic bug in your notebook. The smallest reliable workaround in-notebook is to force TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow.

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version to `2`) before importing TensorFlow. This avoids the failing C++ protobuf path and lets TensorFlow import cleanly without changing any downstream model/training logic.

Updated cells: Only cell 0 is changed, and only to add the environment variable before the TensorFlow import.

Compatibility notes for cell k+1: No variables or interfaces used by cell 1 change; `tf`, `keras`, seeds, and determinism settings remain defined as before once TensorFlow imports successfully.

Assumptions: This runtime allows setting `os.environ[...]` prior to importing TensorFlow (standard in notebooks), and using the Python protobuf backend is acceptable for this workload (it is a compatibility fix; it does not change your model code).'
- What this solution (achieved 0.01691) has done: 'Diagnosis: The crash happens immediately in cell 0 during the `tensorflow` import due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, which triggers an internal call to `MessageFactory.GetPrototype` that no longer exists in protobuf 6.x. The current environment-variable workaround (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) does not fix this API mismatch. Since we cannot change installed packages, the only viable fix is to make TensorFlow use its bundled/compatible protobuf Python implementation via `google.protobuf.internal.api_implementation`, and to avoid setting the protobuf implementation in a way that triggers the failing code path.

Patch summary: In cell 0 only, remove the two `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION*` env-var settings and instead force protobuf to use the pure-Python implementation by calling `google.protobuf.internal.api_implementation._SetType("python")` before importing TensorFlow. This keeps the rest of the notebook logic unchanged and allows `import tensorflow as tf` to succeed under protobuf 6.x.

Updated cells: cell 0 only (below).

Compatibility notes for cell k+1: No variables, imports, or semantics used by cell 1 are changed; `np`, `pd`, `tf`, `keras`, etc. are still imported and seeded exactly as before once TensorFlow imports successfully.

Assumptions: `google.protobuf` is available (it is, given `protobuf==6.33.0`) and its internal API setter exists in this version; this is a standard workaround to force the Python protobuf runtime when C++/default runtime is incompatible.'
- What this solution (achieved 0.01691) has done: 'Diagnosis: The crash occurs in cell 0 during the attempt to force the pure-Python protobuf implementation via `google.protobuf.internal.api_implementation._SetType("python")`. With protobuf 6.x, internal APIs and message factory behavior changed, and this call can trigger an `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow/protobuf initialization. The fix is to avoid calling this unstable internal protobuf API and instead set the supported environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before* importing TensorFlow, which preserves the original intent without relying on removed internals.  

Patch summary: Remove the `_SetType("python")` call and replace it with a safe environment-variable switch applied prior to importing TensorFlow/Keras. Keep all other logic (seeding, determinism settings, GPU config, JIT, etc.) unchanged.  

Updated cells: Only cell 0 is modified.  

Compatibility notes for cell k+1: All imports (`np`, `pd`, `tf`, `keras`, etc.) and variables (`SEED`) remain defined exactly as before, so cell 1 (`pd.read_csv(...)`) and later cells run unchanged.  

Assumptions: The goal of the protobuf block was to prefer the Python implementation for compatibility; using the documented env var provides the same effect without depending on protobuf internals.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import google.protobuf  # noqa: F401
    from packaging.version import Version
    import google.protobuf as _pb

    if Version(_pb.__version__).major >= 5:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

tf.get_logger().setLevel("ERROR")

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
except Exception:
    pass

tf.config.optimizer.set_jit(True)
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

tf.config.experimental.enable_op_determinism()


## === cell 1
train_df = pd.read_csv("../input/paddy-disease-classification/train.csv")
train_df.head()



## === cell 2
train_images_path = "../input/paddy-disease-classification/train_images/"
all_images = {
    row.image_id: os.path.join(train_images_path, row.label, row.image_id)
    for row in train_df.itertuples(index=False)
}



## === cell 3
categories = sorted(os.listdir(train_images_path))



## === cell 4
label_dict = {category: idx for idx, category in enumerate(categories)}
label_dict




## === cell 5
def get_label(label):
    return label_dict[label]




## === cell 6
num_label = {
    0: "tungro",
    1: "hispa",
    2: "downy_mildew",
    3: "bacterial_leaf_streak",
    4: "bacterial_leaf_blight",
    5: "brown_spot",
    6: "blast",
    7: "normal",
    8: "dead_heart",
    9: "bacterial_panicle_blight",
}




## === cell 7
def get_name(x):
    return num_label[x]




## === cell 8
num_label



## === cell 9
img_size = 128



## === cell 10
from concurrent.futures import ThreadPoolExecutor


def _load_resize(path_size):
    path, size = path_size
    img_array = cv2.imread(path)  # BGR (unchanged)
    if img_array is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.resize(img_array, (size, size), interpolation=cv2.INTER_LINEAR)


n = len(train_df)
paths = np.array(
    [all_images[img_id] for img_id in train_df["image_id"].to_numpy()], dtype=object
)
labels_str = train_df["label"].to_numpy()

label_index = pd.Index(categories)
y = label_index.get_indexer(labels_str).astype(np.int64, copy=False)

idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
paths = paths[idx]
y = y[idx]

training0 = [paths[0], y[0]]
training0



## === cell 11
Y = tf.one_hot(y, depth=10, dtype=tf.float32).numpy()
print(Y[100])
print(Y.shape)



## === cell 12
from sklearn.model_selection import train_test_split

paths_train, paths_valid, y_train_int, y_valid_int, y_train, y_valid = train_test_split(
    paths, y, Y, test_size=0.2, random_state=42, stratify=y
)



## === cell 13
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam



## === cell 14
model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8192, activation="relu"),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)



## === cell 15
model.summary()



## === cell 16
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 17
AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.deterministic = True


def _decode_resize_scale(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((paths_train, y_train))
    .with_options(options)
    .map(_decode_resize_scale, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(128, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((paths_valid, y_valid))
    .with_options(options)
    .map(_decode_resize_scale, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(128, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
)



## === cell 18
history = model.fit(train_ds, epochs=30, validation_data=valid_ds)



## === cell 19
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Training Data")
plt.plot(history.history["val_accuracy"], label="Validation Data")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.show()



## === cell 20
model.evaluate(valid_ds, verbose=0)



## === cell 21
submission_df = pd.read_csv(
    "../input/paddy-disease-classification/sample_submission.csv"
)
submission_df



## === cell 22
test_path = "../input/paddy-disease-classification/test_images/"

test_ids = submission_df["image_id"].to_numpy()
test_paths = np.array(
    [os.path.join(test_path, img_id) for img_id in test_ids], dtype=object
)


def _decode_resize_scale_test(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(_decode_resize_scale_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(128, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_images0 = next(iter(test_ds.unbatch().take(1))).numpy()
test_images0



## === cell 23
X_test = None



## === cell 24
test_images0.shape



## === cell 25
pass



## === cell 26
y_pred = model.predict(test_ds, verbose=0)



## === cell 27
y_pred



## === cell 28
labels = np.argmax(y_pred, axis=1).tolist()
labels



## === cell 29
submission_df["label"] = labels



## === cell 30
submission_df["label"] = submission_df["label"].apply(get_name)



## === cell 31
submission_df



## === cell 32
submission_df.to_csv("submission.csv", index=False)



## === cell 33
model.save("paddy_classification.h5")
