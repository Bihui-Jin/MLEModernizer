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

0.1695290858725762

# 6. Current score

0.24237

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14468) has done: 'I fix the TensorFlow import crash by forcing the standalone `tf_keras` backend (common in Kaggle when `tensorflow` + protobuf versions clash) and only fall back to `tensorflow` if needed. Since the referenced pretrained `.h5` model file path does not exist, I replace the load step with a small TF-Keras model that matches the same multi-label setup so the notebook runs end-to-end and can generate predictions. I also fix the label post-processing to always output a valid space-delimited label string (no trailing spaces, never empty—defaults to `healthy`) and ensure the submission rows align with `sample_submission.csv` ordering. These are minimal changes to unblock execution and produce a valid `submission.csv` while keeping the same overall pipeline (build/load model → predict on test dataset → round → write labels).'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/Keras import crash by avoiding the problematic `tf_keras`/protobuf combination and using the built-in `tensorflow.keras` consistently. Then I fix the F1 metric implementation to use `tf.keras.backend` (your current `keras.backend` points to a different Keras package that lacks needed ops), which is what’s causing the training-time AttributeError. Finally, I make a minimal score-improving calibration change by predicting labels with a tuned probability threshold (instead of hard rounding at 0.5) while keeping the same model and training loop; this should nudge the mean F1 upward toward your target. The script still write a valid `submission.csv` with the required space-delimited labels.'
- What this solution (achieved 0.24237) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which is the smallest change that typically unblocks Kaggle TF environments without changing your model/training logic. I also make the pipeline more robust by setting seeds for determinism and by ensuring images are decoded safely for both `.jpg`/`.png` cases (score-neutral, prevents runtime failures). The rest of your core logic (model, training loop, thresholding, submission formatting) be preserved so behavior and score stay close to your current 0.24507 (already above target). Finally, it always write `./submission.csv` with the required `image,labels` columns in the same order as `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

print("Using tensorflow:", tf.__version__, "| keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras import backend as K


def f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val




## === cell 2
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TEST_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isfile(TRAIN_CSV), f"Missing train csv: {TRAIN_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample submission: {SAMPLE_SUB}"



## === cell 3
IMSIZE = 128
NUM_CLASSES = 7


def build_model(input_shape=(IMSIZE, IMSIZE, 3), num_classes=NUM_CLASSES):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[f1])
    return model


model = build_model()



## === cell 4
label_names = [
    "healthy",
    "scab",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "powdery_mildew",
    "rust",
    "complex",
]

train_df = pd.read_csv(TRAIN_CSV)
for ln in label_names:
    train_df[ln] = (
        train_df["labels"].fillna("").apply(lambda s, c=ln: int(c in s.split()))
    )


def _decode_image(image_bytes):
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def _parse_train(image_name, y):
    image_path = tf.strings.join([TRAIN_DIR, "/", image_name])
    image_string = tf.io.read_file(image_path)
    image_decoded = _decode_image(image_string)
    img = tf.image.resize(image_decoded, [IMSIZE, IMSIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(y, tf.float32)


N_TRAIN = min(len(train_df), 1024)
train_small = train_df.sample(N_TRAIN, random_state=42).reset_index(drop=True)

y_small = train_small[label_names].values.astype(np.float32)
ds_train = (
    tf.data.Dataset.from_tensor_slices((train_small["image"].values, y_small))
    .map(_parse_train, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(512, seed=42, reshuffle_each_iteration=True)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

model.fit(ds_train, epochs=1, verbose=1)



## === cell 5
sub_df = pd.read_csv(SAMPLE_SUB)
names = sub_df["image"].tolist()


def _parse_test(name):
    image_path = tf.strings.join([TEST_DIR, "/", name])
    image_string = tf.io.read_file(image_path)
    image_decoded = _decode_image(image_string)
    img = tf.image.resize(image_decoded, [IMSIZE, IMSIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


dataset = (
    tf.data.Dataset.from_tensor_slices(tf.constant(names))
    .map(_parse_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 6
y_pred = model.predict(dataset, verbose=1)



## === cell 7
THRESH = 0.35
y = (y_pred >= THRESH).astype(int)

labels = []
for i in range(len(y)):
    chosen = [label_names[j] for j in range(len(label_names)) if y[i][j] == 1]
    if len(chosen) == 0:
        chosen = ["healthy"]
    labels.append(" ".join(chosen))



## === cell 8
df = pd.DataFrame({"image": names, "labels": labels})
df.to_csv("./submission.csv", index=False)
print(df.head())
print("Wrote submission to ./submission.csv with shape:", df.shape)
assert df.shape[0] == len(names)
assert list(df.columns) == ["image", "labels"]
