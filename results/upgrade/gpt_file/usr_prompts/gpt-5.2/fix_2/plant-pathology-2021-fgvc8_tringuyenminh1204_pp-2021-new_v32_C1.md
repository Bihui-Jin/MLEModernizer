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

0.7773961218836576

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, random, math, glob
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib




## === cell 2
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
INPUT_DIR = "../input/plant-pathology-2021-fgvc8"
TEST_DIR = os.path.join(INPUT_DIR, "test_images")

sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    alt = "../input/sample_submission.csv"
    if os.path.exists(alt):
        sample_sub_path = alt
    else:
        raise FileNotFoundError(
            f"Could not find sample_submission.csv at {sample_sub_path} or {alt}"
        )

sample_sub = pd.read_csv(sample_sub_path)
test_images = sample_sub["image"].astype(str).tolist()

source = TEST_DIR
IMAGE_PATHS = [os.path.join(source, img) for img in test_images]

missing = [p for p in IMAGE_PATHS if not tf.io.gfile.exists(p)]
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} test image paths do not exist (showing up to 5): {missing[:5]}"
    )

print("Num test images (from sample_submission):", len(IMAGE_PATHS))



## === cell 5
IMAGE_PATHS[:5]



## === cell 6
AUTO = tf.data.experimental.AUTOTUNE



## === cell 7
test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




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
def find_model_file():
    candidates = []
    candidates += glob.glob("../input/**/SMResNet50.h5", recursive=True)
    candidates += glob.glob("../input/**/*.h5", recursive=True)
    candidates += glob.glob("../input/**/*.keras", recursive=True)

    candidates = [p for p in candidates if os.path.isfile(p)]
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq[0] if len(uniq) else None


model_path = find_model_file()
if model_path is None:
    raise FileNotFoundError(
        "No .h5/.keras model file found under ../input. "
        "Please add the pretrained model dataset to this notebook."
    )

print("Loading model:", model_path)
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={"FixedDropout": FixedDropout},
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4218978520.py in <cell line: 0>()
     23 model_path = find_model_file()
     24 if model_path is None:
---> 25     raise FileNotFoundError(
     26         "No .h5/.keras model file found under ../input. "
     27         "Please add the pretrained model dataset to this notebook."

FileNotFoundError: No .h5/.keras model file found under ../input. Please add the pretrained model dataset to this notebook.

## === cell 10
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs

print("Raw probs shape:", temp_probs.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4158147877.py in <cell line: 0>()
----> 1 probs = model.predict(test_dataset, verbose=1)
      2 temp_probs = probs
      3 
      4 print("Raw probs shape:", temp_probs.shape)
      5 

NameError: name 'model' is not defined

## === cell 11
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {
    0: 0.27,
    1: 0.5,
    2: 0.3,
    3: 0.5,
    4: 0.5,
}

pred_string = []

has_healthy_logit = temp_probs.ndim == 2 and temp_probs.shape[1] >= 6

for line in temp_probs:
    s = ""
    count = 0

    for i in range(5):
        if i < len(line) and line[i] > threshold[i]:
            s = s + name[i] + " "
            count += 1

    if count >= 2:
        notComplex = True
        for i in range(5):
            if i < len(line) and line[i] > threshold[i] and name[i] == "complex":
                notComplex = False
                break
        if notComplex is True:
            s = s + "complex" + " "

    if s.strip() == "":
        if has_healthy_logit:
            s = name[6]
        else:
            s = name[6]

    pred_string.append(s.strip())

print("Num predictions:", len(pred_string))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2792095871.py in <cell line: 0>()
     21 # If model outputs 6 classes including healthy (common), map index 5 to healthy.
     22 # Otherwise keep original behavior: no predicted labels => healthy.
---> 23 has_healthy_logit = temp_probs.ndim == 2 and temp_probs.shape[1] >= 6
     24 
     25 for line in temp_probs:

NameError: name 'temp_probs' is not defined

## === cell 12
sub = pd.DataFrame({"image": test_images, "labels": pred_string})

if len(sub) != len(sample_sub):
    raise ValueError(
        f"Submission length mismatch: got {len(sub)} rows, expected {len(sample_sub)}"
    )

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote", sub_path)
print(sub.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1955677734.py in <cell line: 0>()
      1 # Build submission with exact ordering/length from sample_submission to avoid mismatches.
----> 2 sub = pd.DataFrame({"image": test_images, "labels": pred_string})
      3 
      4 # Safety: enforce exact length match
      5 if len(sub) != len(sample_sub):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
