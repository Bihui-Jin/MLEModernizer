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

0.7575253924284395

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, math, random, pathlib
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__ if hasattr(tf.keras, "__version__") else "bundled")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib




## === cell 2
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(
        bits, channels=3
    )  # faster than generic decoder for known JPEGs
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32




## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"

IMAGE_PATHS = tf.io.gfile.glob(os.path.join(source, "*.jpg"))
if not IMAGE_PATHS:
    IMAGE_PATHS = tf.io.gfile.glob(os.path.join(source, "*.JPG"))
IMAGE_PATHS = sorted(IMAGE_PATHS)

IMAGE_FILES = [os.path.basename(p) for p in IMAGE_PATHS]

print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", IMAGE_FILES[:3])




## === cell 5
IMAGE_PATHS[:5]




## === cell 6
AUTO = tf.data.experimental.AUTOTUNE




## === cell 7
options = tf.data.Options()
options.experimental_deterministic = True  # preserve determinism
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_fusion = True
options.experimental_optimization.map_parallelization = True

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()  # safe for test set size; speeds up any repeated iteration inside predict()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 8
import tensorflow as tf
from tensorflow import keras




## === cell 9
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




## === cell 10
def find_model_file():
    candidates = [
        "../input/3smresnet50/3SMResNet50.h5",
        "/kaggle/input/3smresnet50/3SMResNet50.h5",
        "../input/3smresnet50/3SMResNet50.hdf5",
        "/kaggle/input/3smresnet50/3SMResNet50.hdf5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    shallow_roots = [
        "../input/3smresnet50",
        "/kaggle/input/3smresnet50",
    ]
    for root in shallow_roots:
        if os.path.isdir(root):
            for fn in (
                "3SMResNet50.h5",
                "3SMResNet50.hdf5",
                "3smresnet50.h5",
                "3smresnet50.hdf5",
            ):
                p = os.path.join(root, fn)
                if os.path.exists(p):
                    return p
    return None


model_path = find_model_file()
if model_path is None:
    raise FileNotFoundError(
        "Pretrained model file not found under ../input/3smresnet50 or /kaggle/input/3smresnet50. "
        "Fallback training is disabled to meet the 600s timeout constraint."
    )

print("Loading model from:", model_path)
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={"FixedDropout": FixedDropout},
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2714791033.py in <cell line: 0>()
     32 model_path = find_model_file()
     33 if model_path is None:
---> 34     raise FileNotFoundError(
     35         "Pretrained model file not found under ../input/3smresnet50 or /kaggle/input/3smresnet50. "
     36         "Fallback training is disabled to meet the 600s timeout constraint."

FileNotFoundError: Pretrained model file not found under ../input/3smresnet50 or /kaggle/input/3smresnet50. Fallback training is disabled to meet the 600s timeout constraint.

## === cell 11
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs

print("Pred shape:", temp_probs.shape)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3017807206.py in <cell line: 0>()
----> 1 probs = model.predict(test_dataset, verbose=1)
      2 temp_probs = probs
      3 
      4 print("Pred shape:", temp_probs.shape)
      5 

NameError: name 'model' is not defined

## === cell 12
temp_probs[:2]




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1439111443.py in <cell line: 0>()
----> 1 temp_probs[:2]
      2 
      3 

NameError: name 'temp_probs' is not defined

## === cell 13
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}
threshold = {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.35}

p = np.asarray(temp_probs, dtype=np.float32)
thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)[None, :]
mask = p[:, :5] > thr  # (N,5) boolean

cnt = mask.sum(axis=1)
need_complex = (cnt >= 2) & (~mask[:, 2])

labels_per_row = []
for row_mask, add_c in zip(mask, need_complex):
    labs = [name[i] for i in np.flatnonzero(row_mask)]
    if add_c:
        labs.append("complex")
    if not labs:
        labs = [name[6]]
    labels_per_row.append(" ".join(labs))

pred_string = labels_per_row




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/878974204.py in <cell line: 0>()
     13 threshold = {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.35}
     14 
---> 15 p = np.asarray(temp_probs, dtype=np.float32)
     16 thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)[None, :]
     17 mask = p[:, :5] > thr  # (N,5) boolean

NameError: name 'temp_probs' is not defined

## === cell 14
pred_string[:10], len(pred_string)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2264494190.py in <cell line: 0>()
----> 1 pred_string[:10], len(pred_string)
      2 
      3 

NameError: name 'pred_string' is not defined

## === cell 15
df = pd.DataFrame({"image": IMAGE_FILES, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2754545659.py in <cell line: 0>()
----> 1 df = pd.DataFrame({"image": IMAGE_FILES, "labels": pred_string})
      2 df.to_csv("submission.csv", index=False)
      3 print(df.head())
      4 print("Wrote submission.csv with shape:", df.shape)

NameError: name 'pred_string' is not defined
