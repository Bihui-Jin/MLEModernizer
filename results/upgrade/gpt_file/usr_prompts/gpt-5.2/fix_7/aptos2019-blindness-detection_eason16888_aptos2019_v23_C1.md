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

0.7657290256664656

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing `tensorflow-addons` install/import (it’s not used for inference here and is triggering the protobuf `MessageFactory` error), and consolidate imports so `tf`, `pd`, etc. are always defined before later cells run. I also fix the image normalization mismatch: your model is fed raw 0–255 images while EfficientNet expects normalized inputs; we reuse your existing `preprocessing()` logic for test-time so predictions are meaningful (this should improve score vs. random). Finally, I make file paths robust to this environment by using the provided `/kaggle/data/aptos2019-blindness-detection/...` paths (with a safe fallback), ensure weights loading is optional (so the notebook still runs even if the weights file isn’t present), and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.15492) has done: 'The crash in your first cell is happening before your code runs due to an incompatibility between the installed protobuf runtime and TensorFlow’s internal proto usage; we can fix this safely by forcing the pure-Python protobuf implementation *before* importing TensorFlow. After that, the rest of your pipeline can run end-to-end and always write a valid `submission.csv`. To move the score up from 0.0 toward your target, the smallest legitimate improvement is to ensure EfficientNetB2 receives inputs in the expected scale and normalization: keep your existing BEN preprocessing but additionally apply `tf.keras.applications.efficientnet.preprocess_input` (this is consistent with EfficientNet evaluation semantics and doesn’t change your model architecture). I also keep your weight-loading logic but make it robust (if weights exist, they be used; otherwise it still runs and submits).'
- What this solution (achieved -0.05813) has done: 'We fix the crash happening before your code runs by forcing TensorFlow to use the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow (this directly targets the `MessageFactory.GetPrototype` error). Next, we make the EfficientNet preprocessing consistent and non-destructive by avoiding the redundant RGB↔BGR conversion mismatch (so the BEN preprocessing is applied correctly) and then applying `efficientnet.preprocess_input` exactly once in the expected scale. Finally, we keep your weight-loading logic but expand it to also search common Kaggle input directories so it actually finds the provided model file when present, which should move your score up substantially toward the target without changing the model architecture.'
- What this solution (achieved 0.02537) has done: 'We fix the TensorFlow import crash by ensuring the protobuf runtime is compatible before TensorFlow loads, using a safe monkey-patch for the missing `MessageFactory.GetPrototype` symbol (this is the direct cause of your current failure). Then we keep your model and preprocessing logic unchanged, but make the environment and paths deterministic and robust so the notebook runs end-to-end and always writes `submission.csv`. Finally, we keep the existing weight-search logic (so you can actually use the trained weights when present), which is the main lever to move the score up toward the target without changing core modeling.'
- What this solution (achieved -0.03952) has done: 'The current failure happens before your code runs because TensorFlow’s protobuf expects `MessageFactory.GetPrototype` but in this environment it’s missing; your patch only adds it to the class, while TensorFlow can be calling it on an already-created instance. I make the protobuf compatibility patch robust by adding `GetPrototype` to both the class and any existing `MessageFactory` instances (and fall back safely if protobuf internals differ), which should unblock the TensorFlow import. After that, I keep your model, preprocessing, and inference logic unchanged, only ensuring the submission is always written as `submission.csv` with the required columns. This is score-neutral in intent (it mainly fixes the crash) and allows your existing weight-loading logic (the main score driver) to actually run.'
- What this solution (achieved 0.0) has done: 'Your current score (-0.03952) is far below the target (0.7657), so we should improve predictions without changing the model architecture or training logic. The biggest score issue here is likely a preprocessing mismatch: you apply BEN-style preprocessing twice (once in `load_ben_color`, then again in `preprocessing`), which can distort images and make the trained weights (if found) behave poorly. I make inference preprocessing consistent by running BEN preprocessing exactly once (keep your existing function), then apply EfficientNet’s `preprocess_input` once in the expected scale. I also read `test.csv` (instead of `sample_submission.csv`) to guarantee the exact test ordering used by Kaggle, while still writing the required `id_code,diagnosis` submission.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    from google.protobuf import message_factory as _message_factory

    def _patch_message_factory_getprototype():
        MF = getattr(_message_factory, "MessageFactory", None)
        if MF is None:
            return

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            raise AttributeError(
                "MessageFactory has no GetMessageClass; cannot emulate GetPrototype"
            )

        if not hasattr(MF, "GetPrototype"):
            setattr(MF, "GetPrototype", _GetPrototype)

        for attr in ("_DEFAULT", "default_pool", "_FACTORY", "_factory", "factory"):
            obj = getattr(_message_factory, attr, None)
            if (
                obj is not None
                and hasattr(obj, "__class__")
                and obj.__class__.__name__.endswith("MessageFactory")
            ):
                if not hasattr(obj, "GetPrototype") and hasattr(obj, "GetMessageClass"):
                    try:
                        setattr(
                            obj,
                            "GetPrototype",
                            _GetPrototype.__get__(obj, obj.__class__),
                        )
                    except Exception:
                        pass

    _patch_message_factory_getprototype()
except Exception:
    pass

import gc
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm

tf.random.set_seed(0)
np.random.seed(0)



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16  # kept for compatibility; not used in this single-image loop


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


def load_ben_color(image_bgr, sigmaX=10):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def preprocessing(image_rgb_uint8, sigmaX=10):
    image = crop_image_from_gray(image_rgb_uint8).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
base_model = tf.keras.applications.efficientnet.EfficientNetB2(
    include_top=False,
    weights=None,  # preserve original setting (weights loaded via .load_weights if provided)
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

flatten_layer = tf.keras.layers.Flatten()
dense_layer_1 = tf.keras.layers.Dense(4096, activation="relu")
Dropout_1 = tf.keras.layers.Dropout(0.6)
dense_layer_2 = tf.keras.layers.Dense(2048, activation="relu")
Dropout_2 = tf.keras.layers.Dropout(0.6)
dense_layer_3 = tf.keras.layers.Dense(1024, activation="relu")
prediction_layer = tf.keras.layers.Dense(5, activation="softmax")

model = tf.keras.models.Sequential(
    [
        base_model,
        flatten_layer,
        dense_layer_1,
        Dropout_1,
        dense_layer_2,
        Dropout_2,
        dense_layer_3,
        prediction_layer,
    ]
)

_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32))



## === cell 3
candidate_weight_paths = [
    "../input/eff-b2-model/eff_b2_model.h5",
    "/kaggle/input/eff-b2-model/eff_b2_model.h5",
    "/kaggle/data/eff-b2-model/eff_b2_model.h5",
    "/kaggle/input/eff-b2-model/eff_b2_model/eff_b2_model.h5",
    "/kaggle/input/eff-b2-model/eff_b2_model.h5/eff_b2_model.h5",
]

try:
    for root, _, files in os.walk("/kaggle/input"):
        if "eff_b2_model.h5" in files:
            candidate_weight_paths.append(os.path.join(root, "eff_b2_model.h5"))
except Exception:
    pass

weight_path = next((p for p in candidate_weight_paths if os.path.exists(p)), None)

if weight_path is not None:
    model.load_weights(weight_path)
    print(f"Loaded weights: {weight_path}")
else:
    print(
        "WARNING: Weight file not found. Running with random weights will yield poor score, "
        "but the pipeline will still produce a valid submission."
    )




## === cell 4
def resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


test_csv_path = resolve_path(
    "../input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/test.csv",
)
sample_sub_path = resolve_path(
    "../input/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/data/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)
test_img_dir = resolve_path(
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/test_images",
)

if test_csv_path is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle paths.")
if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle paths."
    )
if test_img_dir is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected Kaggle paths."
    )

test_df = pd.read_csv(test_csv_path)
id_code = test_df["id_code"].astype(str).values
print(f"Loaded {len(id_code)} test ids from: {test_csv_path}")
print(f"Using test image dir: {test_img_dir}")



## === cell 5
eff_preprocess = tf.keras.applications.efficientnet.preprocess_input

test_prediction = np.empty(len(id_code), dtype=np.int64)

for i, img_id in enumerate(tqdm(id_code, total=len(id_code))):
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        test_prediction[i] = 0
        continue

    img_rgb_uint8 = load_ben_color(
        img_bgr
    )  # RGB uint8, already cropped/resized/sharpened
    img_255 = img_rgb_uint8.astype(np.float32)  # [0,255]
    img_input = eff_preprocess(img_255)  # EfficientNet normalized

    X = np.expand_dims(img_input, axis=0)
    pred = model.predict(X, verbose=0)
    test_prediction[i] = int(np.argmax(pred, axis=1)[0])



## === cell 6
sub = pd.read_csv(sample_sub_path)[["id_code"]].copy()
pred_map = dict(zip(id_code.tolist(), test_prediction.astype(np.int64).tolist()))
sub["diagnosis"] = sub["id_code"].map(pred_map).fillna(0).astype(np.int64)

sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(sub["diagnosis"].values, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv")

gc.collect()
