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

3.10

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

0.1341184834637552

# 6. Current score

0.74457

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.74457) has done: 'I remove the failing `pip install tensorflow-addons` (it triggers a protobuf incompatibility in this environment) and drop the unused `tensorflow_addons` import so imports succeed. I also fix the broken cell ordering/numbering and ensure `keras/pandas` are defined before use, which resolves the `NameError` crashes. Because the referenced external pretrained model path (`../input/eff-b0-model-kappa/eff_b0_model`) is not available in your provided filesystem, I replace that load with a small DenseNet121-based classifier (same overall “CNN classifier” core logic) trained briefly on the provided train set so the notebook can run end-to-end and output a valid `submission.csv`. Finally, I make inference deterministic and ensure the submission matches `id_code,diagnosis` exactly.'
- What this solution (achieved 0.74457) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing Python protobuf mode before TensorFlow is imported, which resolves the common TF↔protobuf binary mismatch in Kaggle-like images. I also make the dataset base path robust by falling back to `/kaggle/input/...` if `/kaggle/data/...` isn’t present, so the notebook runs in the standard Kaggle filesystem. Since your current score (0.74457) is far above the target (0.1341) and higher is better, I avoid any modeling changes that might further improve score; the changes are score-neutral and focused on stability and producing a valid `submission.csv`. The rest of the pipeline (DenseNet121 backbone, training loop, preprocessing, and submission formatting) is preserved.'
- What this solution (achieved 0.74457) has done: 'I fix the TensorFlow/protobuf crash by switching to the safer “python” protobuf implementation without pinning an incompatible implementation version (the `*_VERSION=3` setting is a known trigger for `MessageFactory.GetPrototype` errors in some Kaggle images). I also add a defensive fallback that removes those protobuf env vars entirely if TensorFlow still fails to import, ensuring the notebook runs end-to-end. The rest of the pipeline (DenseNet121 classifier, preprocessing, tf.data input pipeline, training epochs, and submission formatting) is kept identical to preserve core logic and keep score changes minimal. Finally, I keep the robust dataset path selection and ensure `submission.csv` is written with the exact required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import gc
import random
import numpy as np
import pandas as pd
import cv2

try:
    import tensorflow as tf
except AttributeError:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
    import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""
Config + image preprocessing (kept from original logic).
"""
IMG_SIZE = 224
BATCH_SIZE = 16


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32")


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
"""
Data paths (use the provided dataset location).
Fix: Make BASE_DIR robust for both /kaggle/data and standard Kaggle /kaggle/input layouts.
"""
BASE_CANDIDATES = [
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
]
BASE_DIR = next((p for p in BASE_CANDIDATES if os.path.exists(p)), BASE_CANDIDATES[0])

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("BASE_DIR:", BASE_DIR)
print(train_df.shape, test_df.shape)
print(train_df.head())
print(test_df.head())



## === cell 3
"""
Build tf.data pipelines (minimal, avoids Keras generator pitfalls and keeps runtime reasonable).
"""
from sklearn.model_selection import train_test_split

train_df = train_df.copy()
train_df["path"] = train_df["id_code"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png")
)

test_df = test_df.copy()
test_df["path"] = test_df["id_code"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, f"{x}.png")
)

tr_df, va_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["diagnosis"]
)


def _read_preprocess(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)
    img = tf.numpy_function(
        func=lambda x: load_ben_color(x),
        inp=[img],
        Tout=tf.float32,
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    img = img / 255.0
    if label is None:
        return img
    label = tf.cast(label, tf.int32)
    return img, label


def make_ds(df, training=True):
    paths = df["path"].values
    labels = df["diagnosis"].values if "diagnosis" in df.columns else None

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            lambda p: _read_preprocess(p, None), num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda p, y: _read_preprocess(p, y), num_parallel_calls=tf.data.AUTOTUNE
        )

    if training:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(tr_df, training=True)
val_ds = make_ds(va_df, training=False)
test_ds = make_ds(test_df, training=False)



## === cell 4
"""
Model definition.
The original notebook tried to load an external pretrained model that isn't present in the provided filesystem.
To keep the same overall core idea (CNN image classifier) and make the run end-to-end, we train a DenseNet121 head.
"""
base = DenseNet121(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # keep training quick and stable

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 5
"""
Train briefly to produce non-random predictions and a valid submission within time limits.
(No early stopping / approximations introduced; just a small fixed epoch count.)
"""
EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

gc.collect()



## === cell 6
"""
Inference + submission creation (guarantee correct columns and .csv suffix).
"""
pred = model.predict(test_ds, verbose=1)
test_prediction = np.argmax(pred, axis=1).astype(np.int64)

sub = pd.read_csv(SAMPLE_SUB)
id_to_pred = dict(zip(test_df["id_code"].values, test_prediction))
sub["diagnosis"] = sub["id_code"].map(id_to_pred).astype(np.int64)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

unique, counts = np.unique(sub["diagnosis"].values, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print(f"Wrote {sub_path} with shape {sub.shape}")
print(sub.head())
