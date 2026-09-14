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

0.5332928807844208

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking import issue by removing the standalone `keras` imports and using `tf.keras` consistently, which avoids the protobuf `MessageFactory.GetPrototype` crash in Kaggle. Since the external pretrained `.h5` model file is missing, I replace the model-loading step with a minimal, deterministic fallback model so the notebook runs end-to-end and produces a valid `submission.csv` with the required columns. I also update deprecated `predict_generator` to `predict` and ensure paths/filenames match the competition’s `test_images` folder. These changes are strictly to unblock execution and generate a valid submission file (score may be low without the missing pretrained weights, but it run correctly).'
- What this solution (achieved 0.0) has done: 'I fix the environment-breaking protobuf/keras crash by forcing TensorFlow to use the Python protobuf implementation before importing `tensorflow`, which avoids the `MessageFactory.GetPrototype` error in Kaggle. Then I make the preprocessing function compatible with `ImageDataGenerator` by ensuring it accepts and returns float images in the expected range (so it doesn’t receive a file path or produce uint8 unexpectedly). Finally, I keep your existing model logic (load if present, otherwise fallback) and ensure the prediction/output pipeline always produces a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.006) has done: 'The crash happens before any training/inference because the Kaggle environment’s protobuf/TensorFlow combination is incompatible with forcing the pure-Python protobuf implementation; removing that environment override fixes the `MessageFactory.GetPrototype` error. I keep your exact preprocessing and submission pipeline, but replace the (currently score-0.0) “fallback model” with a tiny deterministic image-feature baseline that uses the already-computed preprocessed images (no training loop change needed, and still fast) so predictions are no longer essentially random. This should move the score upward toward the target without introducing extra packages or changing I/O paths/format. The submission writing is kept identical (`submission.csv` with `id_code,diagnosis` aligned to `test.csv`).'
- What this solution (achieved 0.006) has done: 'We fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the most common stable workaround in Kaggle for that exact attribute error). Then we keep your existing preprocessing, generator, and model selection logic unchanged, but ensure inference runs end-to-end and always writes a valid `submission.csv` with the correct columns aligned to `test.csv`. No architecture/training loop changes are introduced; the only functional change is the import/environment ordering to unblock execution.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by removing the environment override that forces the pure-Python protobuf implementation, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle runtime. Then I keep your preprocessing, generator, and submission-writing logic intact, but (when the external `.h5` model is absent) I lightly improve the existing heuristic by calibrating its thresholds from the training label distribution (no new model/training loop, still deterministic) so the score moves upward toward the target. Finally, I make the test image directory resolution a bit more robust across the provided folder layouts and ensure the submission is always written as `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import cv2

DATA_PATH = "../input/aptos2019-blindness-detection/"

DIM_X = 256
DIM_Y = 256
BATCH_SIZE = 32

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def crop_image_from_gray(img, tol=7):
    if img is None:
        return img
    if img.ndim == 2:
        mask = img > tol
        if mask.any():
            return img[np.ix_(mask.any(1), mask.any(0))]
        return img
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
    else:
        return img


def circle_crop_v2(img):
    if img is None:
        return img
    height, width, depth = img.shape
    largest_side = int(np.max((height, width)))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = int(np.amin((x, y)))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img


def preprocess_image(image, sigmaX=25, DIM_X=256, DIM_Y=256):
    """
    ImageDataGenerator(preprocessing_function=...) passes a numpy image array (RGB) by default.
    Make this robust to float/uint8 inputs and always return float32 in [0,255] (before rescale).
    """
    if image is None:
        return np.zeros((DIM_Y, DIM_X, 3), dtype=np.float32)

    image = np.asarray(image)

    if image.dtype != np.uint8:
        mx = float(np.max(image)) if image.size else 0.0
        if mx <= 1.5:
            image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image_u8 = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image_u8 = image

    img = crop_image_from_gray(image_u8)
    img = circle_crop_v2(img)
    img = cv2.resize(img, (DIM_X, DIM_Y))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)

    return img.astype(np.float32)




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




## === cell 2
def build_fallback_model(input_shape=(DIM_Y, DIM_X, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model_path = "../input/effnet-7/effnet_09-12.h5"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(
        model_path, custom_objects={"FixedDropout": FixedDropout}, compile=False
    )
    use_heuristic = False
else:
    model = build_fallback_model()
    use_heuristic = True



## === cell 3
train_csv_path = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_csv_path)
label_counts = train_df["diagnosis"].value_counts().sort_index()
label_probs = (
    (label_counts / label_counts.sum()).reindex(range(5), fill_value=0.0).values
)
cum_probs = np.cumsum(label_probs)

calib_thresholds = np.clip(cum_probs[:4], 1e-3, 1 - 1e-3).astype(np.float32)
if not np.all(np.diff(calib_thresholds) > 0):
    calib_thresholds = np.array([0.35, 0.45, 0.55, 0.65], dtype=np.float32)

print("Using heuristic thresholds:", calib_thresholds)



## === cell 4
submission_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
submission_df["filename"] = submission_df["id_code"].astype(str) + ".png"

candidate_test_dirs = [
    os.path.join(DATA_PATH, "test_images"),
    "../input/test_images",
    "../input/aptos2019-blindness-detection/test_images",
]
test_dir = None
for d in candidate_test_dirs:
    if os.path.isdir(d):
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {candidate_test_dirs}"
    )

submission_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1 / 255.0,
)

gen = submission_datagen.flow_from_dataframe(
    dataframe=submission_df,
    directory=test_dir,
    x_col="filename",
    y_col=None,
    batch_size=22,
    shuffle=False,
    class_mode=None,
    target_size=(DIM_Y, DIM_X),
    validate_filenames=False,
)

if not use_heuristic:
    pred = model.predict(gen, verbose=1)
else:
    preds = []
    thresholds = calib_thresholds
    for i in range(len(gen)):
        batch = gen[i]  # shape (bs, H, W, 3), float32 in [0,1]
        g_mean = batch[:, :, :, 1].mean(axis=(1, 2))
        rgb_std = batch.std(axis=(1, 2, 3))

        score = 0.85 * g_mean + 0.15 * (1.0 - np.clip(rgb_std, 0.0, 1.0))
        score = np.clip(score, 0.0, 1.0)

        cls = np.digitize(score, thresholds, right=False).astype(np.int64)
        onehot = np.zeros((batch.shape[0], 5), dtype=np.float32)
        onehot[np.arange(batch.shape[0]), cls] = 1.0
        preds.append(onehot)
    pred = np.concatenate(preds, axis=0)[: len(submission_df)]



## === cell 5
pred = np.argmax(pred, axis=1).astype(int)

if len(pred) != len(submission_df):
    raise RuntimeError(
        f"Prediction length {len(pred)} != test rows {len(submission_df)}"
    )

out_df = submission_df[["id_code"]].copy()
out_df["diagnosis"] = pred
out_df.to_csv("submission.csv", index=False)

print(out_df.head())
print("Wrote submission.csv with shape:", out_df.shape)



## === cell 6
from collections import Counter

cnt = Counter(out_df["diagnosis"].tolist())
print(cnt)
