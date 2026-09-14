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

0.6376212321223429

# 6. Current score

-0.02911

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.19606) has done: 'I remove the failing `tensorflow-addons` installation/import (it triggers a known protobuf `GetPrototype` error) and keep the rest of the pipeline intact. I also fix the missing `keras/pd` definitions by consolidating imports in the first cell so later cells can run. Since the referenced external pretrained model path (`../input/eff-b2-model/eff_b2_model`) is not available in your provided dataset paths, I switch to loading a model from the same competition input directory if present; otherwise, I fall back to a standard Keras application model (DenseNet121) to ensure the notebook runs end-to-end and produces `submission.csv`. Finally, I fix inference preprocessing to match the expected float input range (0–1) to avoid obvious prediction issues and ensure the submission format is correct.'
- What this solution (achieved 0.03816) has done: 'I fix the immediate runtime crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I make model loading robust by compiling the fallback model (if no external pretrained model is found) and loading weights if present, so predictions are not random; this should move the QWK score upward toward your target while keeping the same overall pipeline (single-image loop inference, same preprocessing, same DenseNet121 fallback architecture). I also add a safe path fallback for the dataset directory because Kaggle can mount it under `/kaggle/input/...` rather than `../input/...`, without changing the I/O semantics. Finally, I ensure the submission is aligned to `test.csv` ordering (not the sample submission ordering) to avoid silent id/prediction mismatches.'
- What this solution (achieved 0.09953) has done: 'To fix the runtime crash, I force TensorFlow to use the pure-Python protobuf *and* proactively remove the incompatible `google.protobuf` C-extension modules from `sys.modules` before importing TensorFlow (this is the usual root cause of the `MessageFactory.GetPrototype` error). To move the score up toward your target without changing the modeling approach, I add a minimal, standard “Ben Graham” preprocessing step at inference (the same one you already defined for training-style pipelines) so the DenseNet/ImageNet features see a more appropriate input distribution than raw resized RGB. I also make image path resolution a bit more robust (nested `test_images/test_images` appears in your directory listing) to prevent missing-file failures while keeping I/O semantics identical. The rest of the logic (single-image loop inference, same fallback DenseNet121 architecture/head, same argmax-to-class submission) is preserved.'
- What this solution (achieved -0.02911) has done: 'The crash happens before any modeling because TensorFlow is importing an incompatible protobuf build; setting the environment variable inside Python is too late in this Kaggle image. I fix this by switching to a TF import strategy that forces the pure-Python protobuf implementation from the very start of the process and avoids the `GetPrototype` call path, while keeping all modeling/inference logic identical. I also add a safe fallback to run inference without TensorFlow if TF still can’t be imported (using a deterministic, simple image-statistics classifier) so you always get a valid `submission.csv`. This is primarily a stability fix; if TF loads correctly you keep the same model path/Ben-Graham preprocessing/argmax submission behavior as before.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import gc
import numpy as np
import pandas as pd
import cv2

np.random.seed(42)

TF_AVAILABLE = False
tf = None
keras = None
load_model = None
GlobalAveragePooling2D = Dropout = Dense = None
Adam = None
DenseNet121 = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    from tensorflow.keras.models import load_model  # noqa: F401
    from tensorflow.keras.layers import (
        GlobalAveragePooling2D,
        Dropout,
        Dense,
    )  # noqa: F401
    from tensorflow.keras.optimizers import Adam  # noqa: F401
    from tensorflow.keras.applications import DenseNet121  # noqa: F401

    tf.random.set_seed(42)
    TF_AVAILABLE = True
    print("TF:", tf.__version__)
    print("Eager:", tf.executing_eagerly())
except Exception as e:
    print(
        "WARNING: TensorFlow could not be imported; will use non-TF fallback for predictions."
    )
    print("TF import error:", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


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
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32")


"""
    Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
def build_fallback_model(img_size=224, n_classes=5):
    base = DenseNet121(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inputs=base.input, outputs=out)
    return m


DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/aptos2019-blindness-detection"

model = None
if TF_AVAILABLE:
    candidate_paths = [
        "../input/eff-b2-model/eff_b2_model",
        "../input/eff-b2-model/eff_b2_model.h5",
        os.path.join(DATA_ROOT, "eff_b2_model"),
        os.path.join(DATA_ROOT, "eff_b2_model.h5"),
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                model = load_model(p, compile=False)
                print(f"Loaded model from: {p}")
                break
            except Exception as e:
                print(f"Found but failed to load {p}: {e}")

    if model is None:
        print(
            "No external pretrained model found in input paths; using fallback DenseNet121 model."
        )
        model = build_fallback_model(IMG_SIZE, 5)

    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy")

    _ = model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
    print("Model output shape:", model.output_shape)




## === cell 3
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TEST_DIR_NESTED = os.path.join(TEST_DIR, "test_images")

test_csv_path = os.path.join(DATA_ROOT, "test.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
else:
    test_df = pd.read_csv(sample_sub_path)[["id_code"]]

id_code = test_df["id_code"].astype(str).values
test_prediction = np.empty(len(id_code), dtype=np.int64)


def heuristic_predict_class(img_rgb_float01: np.ndarray) -> int:
    gray = cv2.cvtColor((img_rgb_float01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    m = float(np.mean(gray)) / 255.0
    if m < 0.22:
        return 4
    elif m < 0.30:
        return 3
    elif m < 0.38:
        return 2
    elif m < 0.50:
        return 1
    else:
        return 0


for i in range(len(id_code)):
    img_path = os.path.join(TEST_DIR, f"{id_code[i]}.png")
    if not os.path.exists(img_path):
        alt = os.path.join(TEST_DIR_NESTED, f"{id_code[i]}.png")
        if os.path.exists(alt):
            img_path = alt

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = load_ben_color(img)  # float32 in [~0..255] after enhancement
    X = (img / 255.0)[None, ...].astype(np.float32)

    if TF_AVAILABLE:
        pred = model.predict(X, verbose=0)
        test_prediction[i] = int(np.argmax(pred, axis=1)[0])
    else:
        test_prediction[i] = heuristic_predict_class(X[0])




## === cell 4
sub = test_df[["id_code"]].copy()
sub["diagnosis"] = test_prediction.astype(np.int64)
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print("Done!")
