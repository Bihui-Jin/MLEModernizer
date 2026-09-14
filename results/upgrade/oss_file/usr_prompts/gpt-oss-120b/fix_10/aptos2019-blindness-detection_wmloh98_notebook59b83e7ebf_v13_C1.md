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

0.7702645650369125

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I added a safeguard for the protobuf issue, wrapped the model loading in a try‑except that builds a small EfficientNet‑B0 model when the .h5 file is missing, and trained this model briefly on the provided training data. I also replaced the deprecated `predict_generator` call with `model.predict`, ensured all variables are defined before use, and kept the original preprocessing pipeline. The script now always creates a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'The changes reorder the dataset pipeline to cache the expensive image‑preprocessing step **before** shuffling and batching, which avoids re‑doing the costly OpenCV work each epoch and reduces memory pressure. The cache is then shuffled (if requested) and batched, followed by prefetching. This preserves the exact preprocessing, model architecture, and training loops, so the predictions remain identical while cutting runtime dramatically.'
- What this solution (achieved 0.0) has done: 'The timeout is caused mainly by training the fallback EfficientNetB0 model on CPU; the heavy backbone is still trainable, which makes each epoch slow. By freezing the EfficientNet base we keep the exact architecture and training loop but eliminate costly back‑propagation through the large backbone, dramatically reducing compute while preserving inference behavior. The only code change is adding `base.trainable = False` right after creating the EfficientNet model.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

tf.config.threading.set_intra_op_parallelism_threads(
    tf.config.threading.get_intra_op_parallelism_threads()
)
tf.config.threading.set_inter_op_parallelism_threads(
    tf.config.threading.get_inter_op_parallelism_threads()
)

DATA_PATH = "../input/aptos2019-blindness-detection/"

DIM_X = 256
DIM_Y = 256
BATCH_SIZE = 32


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop_v2(img):
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img


def preprocess_image(image, sigmaX=25, DIM_X=256, DIM_Y=256):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = circle_crop_v2(image)
    image = cv2.resize(image, (DIM_X, DIM_Y))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def _load_and_preprocess(path_bytes):
    """Load a PNG file and apply the original preprocess pipeline."""
    path = path_bytes.decode()
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = preprocess_image(img, sigmaX=25, DIM_X=DIM_X, DIM_Y=DIM_Y)
    img = img.astype(np.float32) / 255.0
    return img


def _map_fn_image_label(x, y):
    img = tf.numpy_function(_load_and_preprocess, [x], tf.float32)
    img = tf.ensure_shape(img, (DIM_X, DIM_Y, 3))
    return img, tf.cast(y, tf.int32)


def _map_fn_image(x):
    img = tf.numpy_function(_load_and_preprocess, [x], tf.float32)
    img = tf.ensure_shape(img, (DIM_X, DIM_Y, 3))
    return img


def build_dataset(filenames, labels=None, shuffle=False):
    """Create a tf.data.Dataset that yields (image, label) batches when labels are given."""
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
        ds = ds.map(
            _map_fn_image_label,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.map(
            _map_fn_image,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)


def build_fallback_model(num_classes=5):
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(DIM_X, DIM_Y, 3), weights="imagenet", include_top=False
    )
    base.trainable = False  # <<< freeze weights to avoid heavy back‑propagation

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = FixedDropout(0.2)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


FALLBACK_MODEL_PATH = "../working/effnet_fallback.h5"

try:
    model = load_model(
        "../input/effnet-8/effnet_09-12_2.h5",
        custom_objects={"FixedDropout": FixedDropout},
    )
except Exception:
    if os.path.exists(FALLBACK_MODEL_PATH):
        model = load_model(
            FALLBACK_MODEL_PATH, custom_objects={"FixedDropout": FixedDropout}
        )
    else:
        model = build_fallback_model()
        train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
        train_df["filename"] = train_df["id_code"].astype(str) + ".png"
        train_paths = (
            train_df["filename"]
            .apply(lambda f: os.path.join(DATA_PATH, "train_images", f))
            .values
        )
        train_labels = train_df["diagnosis"].values
        train_ds = build_dataset(train_paths, labels=train_labels, shuffle=True)
        model.fit(train_ds, epochs=2, verbose=1)
        model.save(FALLBACK_MODEL_PATH)




## === cell 2
submission_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
submission_df["filename"] = submission_df["id_code"].astype(str) + ".png"

test_paths = (
    submission_df["filename"]
    .apply(lambda f: os.path.join(DATA_PATH, "test_images", f))
    .values
)

test_ds = build_dataset(test_paths, labels=None, shuffle=False)

pred_probs = model.predict(test_ds, verbose=1)




## === cell 3
pred_classes = np.argmax(pred_probs, axis=1)
labels = pred_classes.tolist()

submission_df.drop(columns=["filename"], inplace=True)
submission_df["diagnosis"] = labels
submission_df.to_csv("submission.csv", index=False)




## === cell 4
from collections import Counter

cnt = Counter(labels)
cnt
