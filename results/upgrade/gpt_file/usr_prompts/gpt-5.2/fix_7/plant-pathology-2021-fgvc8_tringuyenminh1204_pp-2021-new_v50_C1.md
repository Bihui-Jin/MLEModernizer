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

0.7871283471837496

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the import crash in the first cell by avoiding the `kaggle_datasets` import (it isn’t used and triggers a protobuf incompatibility in this environment). Then I fix the missing model file issue by loading the provided `sample_submission.csv` and generating a valid, correctly ordered submission even when the external `.h5` model input folder is unavailable (so the notebook always yields `submission.csv`). Finally, I fix the submission length mismatch by ensuring the `image` list used in the DataFrame matches exactly the prediction list, and I preserve your existing thresholding-to-label-string logic unchanged.'
- What this solution (achieved 0.24507) has done: 'You’re hitting a TensorFlow import crash caused by an incompatible protobuf runtime in this Kaggle environment, so the first fix is to force TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`. Next, your label mapping is missing class index `5` (which corresponds to `multiple_diseases` in this competition), which makes your post-processing ignore an entire class and severely hurts mean F1; adding this missing class preserves your existing thresholding logic while making predictions semantically correct. Finally, I keep your model-loading/fallback behavior intact but ensure paths are robust and the submission is always aligned to `sample_submission.csv` ordering and written as `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow import, plus clearing any pre-imported protobuf modules that can keep the bad state. Then I remove the redundant second TensorFlow import to avoid re-triggering the same issue. Finally, I keep your model loading and label-thresholding logic intact, but make the fallback probabilities predict “healthy” (instead of all-zeros), which legitimately improves mean F1 when the external model file is missing and moves your score toward the target.'
- What this solution (achieved 0.24507) has done: 'We need to fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) so the notebook can actually run end-to-end and generate predictions instead of always falling back to “healthy”. The minimal, robust fix in Kaggle is to remove the protobuf env overrides and explicitly pin the `protobuf` Python package to a TensorFlow-compatible version at runtime (no internet needed), then import TensorFlow once. I keep your dataset/model/prediction/threshold-to-label-string logic unchanged, but ensure the test image path discovery is robust and the submission is written in the sample submission order with the required `.csv` suffix. This should move the score up substantially toward the target because the model load and be used.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the most likely reason is that you are not actually using the intended pretrained model (the code falls back to all-healthy when the `.h5` isn’t found), which tanks mean F1. I keep your exact model, dataset pipeline, and threshold-to-label-string logic, but make the model discovery robust by searching all standard Kaggle input locations for `5SMResNet50.h5` (including nested dataset folders) and failing only if it truly doesn’t exist. I also force deterministic ordering/alignment by building the submission strictly from `sample_submission.csv` image order (then loading images by that same order), which avoids any filename ordering mismatch harming F1. These are minimal changes focused on ensuring the real model is used and predictions align with the evaluation rows.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the most likely issue is that predictions and `sample_submission.csv` rows are misaligned when the code falls back to `os.listdir()` ordering (or when some expected files aren’t found), causing many labels to be assigned to the wrong images and crushing mean F1. I make image path construction strictly follow `sample_submission.csv` order and, if any files are missing, still keep that order by filtering to existing files while preserving indices, then merging back to the full sample order (so alignment is always correct). I also fix a small but critical path typo (`fgvcvc8`) and ensure the dataset is built from exactly the same ordered filenames used to build the final submission. These are minimal changes that preserve your model, preprocessing, thresholds, and label-string logic, but should materially improve the score toward the target.'

# 9. Code solution

## === cell 0
import os, re, math, random, sys, importlib, pathlib, subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    target = "protobuf==3.20.3"
    try:
        if pb_ver is None or tuple(int(x) for x in pb_ver.split(".")[:2]) >= (4, 0):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", target]
            )
            importlib.invalidate_caches()
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    sys.modules.pop(m, None)
    except Exception as e:
        print("WARNING: protobuf pin attempt failed:", repr(e))


_ensure_compatible_protobuf()

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)




## === cell 1
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 2
BATCH_SIZE = 32



## === cell 3
sample_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path_abs = "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if os.path.exists(sample_path_abs):
        sample_path = sample_path_abs
if not os.path.exists(sample_path):
    sample_path_alt = "/kaggle/data/plant-pathology-2021-fgvc8/sample_submission.csv"
    if os.path.exists(sample_path_alt):
        sample_path = sample_path_alt
if not os.path.exists(sample_path):
    sample_path_alt2 = "/kaggle/data/plant-pathology-2021-fgvc8/sample_submission.csv"
    if os.path.exists(sample_path_alt2):
        sample_path = sample_path_alt2

if not os.path.exists(sample_path):
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle paths."
    )

sample = pd.read_csv(sample_path)
test_images_order = sample["image"].tolist()

source = "../input/plant-pathology-2021-fgvc8/test_images"
if not os.path.isdir(source):
    alt = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
    if os.path.isdir(alt):
        source = alt
if not os.path.isdir(source):
    alt2 = "/kaggle/data/plant-pathology-2021-fgvc8/test_images"
    if os.path.isdir(alt2):
        source = alt2

if not os.path.isdir(source):
    raise FileNotFoundError(
        "Could not locate test_images directory in expected Kaggle paths."
    )

IMAGE_PATHS = [os.path.join(source, fn) for fn in test_images_order]
exists_mask = [os.path.exists(p) for p in IMAGE_PATHS]
if not all(exists_mask):
    print(
        "WARNING: Some test images listed in sample_submission.csv were not found on disk. "
        "Will predict only for existing files and merge back to full sample order as 'healthy' for missing."
    )
    IMAGE_PATHS = [p for p, ok in zip(IMAGE_PATHS, exists_mask) if ok]



## === cell 4
IMAGE_PATHS[:5], len(IMAGE_PATHS)



## === cell 5
AUTO = tf.data.experimental.AUTOTUNE



## === cell 6
test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)



## === cell 7
from tensorflow import keras




## === cell 8
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




## === cell 9
def _find_model_file(filename="5SMResNet50.h5"):
    candidates = [
        "../input/finalpp2021/5SMResNet50.h5",
        "/kaggle/input/finalpp2021/5SMResNet50.h5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    roots = ["/kaggle/input", "../input"]
    for root in roots:
        if os.path.isdir(root):
            for dirpath, _, files in os.walk(root):
                if filename in files:
                    return os.path.join(dirpath, filename)
    return None


model_path = _find_model_file("5SMResNet50.h5")

model = None
if model_path is not None and os.path.exists(model_path):
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    print("Loaded model:", model_path)
else:
    print(
        "WARNING: Model file 5SMResNet50.h5 not found under /kaggle/input (or ../input)."
    )
    print("Will write a valid fallback submission using 'healthy' for all test images.")



## === cell 10
if model is not None:
    probs = model.predict(test_dataset, verbose=1)
    temp_probs = probs
else:
    n = len(IMAGE_PATHS)
    temp_probs = np.zeros((n, 7), dtype=np.float32)
    temp_probs[:, 6] = 1.0



## === cell 11
temp_probs.shape



## === cell 12
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "multiple_diseases",
    6: "healthy",
}

threshold = {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.35, 5: 0.35}
threshold2 = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25}


def get_key(val):
    for key, value in name.items():
        if val == value:
            return key
    return "key doesn't exist"


pred_string = []
for line in temp_probs:
    s = ""
    count = 0

    for i in range(6):
        if line[i] > threshold[i]:
            s = s + name[i] + " "

    for i in range(6):
        if line[i] > threshold2[i]:
            count += 1

    if count >= 2:
        notComplex = True
        for i in range(6):
            if line[i] > threshold[i] and name[i] == "complex":
                notComplex = False
                break
        if notComplex == True:
            s = s + "complex" + " "

    if s == "":
        s = name[6]
    else:
        s = s.strip()
    pred_string.append(s)



## === cell 13
pred_string[:5], len(pred_string)



## === cell 14
image_names = [os.path.basename(p) for p in IMAGE_PATHS]

if len(image_names) != len(pred_string):
    m = min(len(image_names), len(pred_string))
    image_names = image_names[:m]
    pred_string = pred_string[:m]

df_pred = pd.DataFrame({"image": image_names, "labels": pred_string})

df = sample[["image"]].merge(df_pred, on="image", how="left")
df["labels"] = df["labels"].fillna("healthy")

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
