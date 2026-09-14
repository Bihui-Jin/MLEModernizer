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

0.1503231763619575

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
WORK_ROOT = "/kaggle/working"
TMP_ROOT = "/kaggle/tmp"

train_csv_path = f"{INPUT_ROOT}/train.csv"
sample_sub_path = f"{INPUT_ROOT}/sample_submission.csv"
train_img_dir = f"{INPUT_ROOT}/train_images"
test_img_dir = f"{INPUT_ROOT}/test_images"

assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.isdir(test_img_dir), f"Missing test_images dir at {test_img_dir}"

print("TensorFlow:", tf.__version__)
print("Test images:", len(list(Path(test_img_dir).glob("*.jpg"))))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_path = "/kaggle/input/dlcv-projekt/model-best.h5"

custom_objects = {}


class _DummyObject:
    def __init__(self, *args, **kwargs):
        pass


try:
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects=custom_objects
    )
except Exception as e1:
    custom_objects.update(
        {
            "RectifiedAdam": _DummyObject,
            "Lookahead": _DummyObject,
            "AdamW": _DummyObject,
            "F1Score": _DummyObject,
            "SigmoidFocalCrossEntropy": _DummyObject,
        }
    )
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects=custom_objects
    )

print("Loaded model from:", model_path)
print("Model output shape:", model.output_shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1724068733.py in <cell line: 0>()
     16 try:
---> 17     model = tf.keras.models.load_model(
     18         model_path, compile=False, custom_objects=custom_objects

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/dlcv-projekt/model-best.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1724068733.py in <cell line: 0>()
     29         }
     30     )
---> 31     model = tf.keras.models.load_model(
     32         model_path, compile=False, custom_objects=custom_objects
     33     )

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/dlcv-projekt/model-best.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 2
os.makedirs(f"{TMP_ROOT}/test_dataset/test", exist_ok=True)

dst_dir = Path(f"{TMP_ROOT}/test_dataset/test/test_images")
if not dst_dir.exists():
    os.system(f"cp -r {test_img_dir} {TMP_ROOT}/test_dataset/test/")

copied_test_dir = Path(f"{TMP_ROOT}/test_dataset/test/test_images")
n_imgs = len(list(copied_test_dir.glob("*.jpg")))
print("Copied test images:", n_imgs, "from", copied_test_dir)



## === cell 3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator()

TARGET_SIZE = (380, 380)
BATCH_SIZE = 32

test_generator = test_datagen.flow_from_directory(
    f"{TMP_ROOT}/test_dataset/test",
    classes=["test_images"],  # ensure deterministic class folder
    class_mode=None,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Generator samples:", test_generator.samples)
print("First 3 filenames:", test_generator.filenames[:3])



## === cell 4
x = model.predict(test_generator, verbose=1)
x = np.asarray(x)

print("Preds shape:", x.shape)
print("Preds range:", float(np.min(x)), float(np.max(x)))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2858691299.py in <cell line: 0>()
      1 # Predict
----> 2 x = model.predict(test_generator, verbose=1)
      3 x = np.asarray(x)
      4 
      5 print("Preds shape:", x.shape)

NameError: name 'model' is not defined

## === cell 5
labels = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]
threshold = 0.4

if x.ndim == 1:
    x = x.reshape(-1, 1)

if x.shape[1] != len(labels):
    if x.shape[1] > len(labels):
        x_use = x[:, : len(labels)]
    else:
        pad = np.zeros((x.shape[0], len(labels) - x.shape[1]), dtype=x.dtype)
        x_use = np.concatenate([x, pad], axis=1)
else:
    x_use = x

z = (x_use > threshold).astype(np.int32)

predictions = [[labels[i] for i, flag in enumerate(row) if flag == 1] for row in z]
predictions_str = [" ".join(p) if len(p) > 0 else "healthy" for p in predictions]

print("Example prediction strings:", predictions_str[:5])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/508234024.py in <cell line: 0>()
      4 
      5 # If model outputs extra columns unexpectedly, align defensively without changing core semantics.
----> 6 if x.ndim == 1:
      7     x = x.reshape(-1, 1)
      8 

NameError: name 'x' is not defined

## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
sample_images = sample_sub["image"].tolist()

gen_images = [Path(p).name for p in test_generator.filenames]

pred_df = pd.DataFrame({"image": gen_images, "labels": predictions_str})

sub = sample_sub[["image"]].merge(pred_df, on="image", how="left")

sub["labels"] = sub["labels"].fillna("healthy")

assert sub.shape[0] == sample_sub.shape[0], "Submission row count mismatch."
assert list(sub.columns) == ["image", "labels"], "Submission columns mismatch."

sub.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3017099252.py in <cell line: 0>()
      6 gen_images = [Path(p).name for p in test_generator.filenames]
      7 
----> 8 pred_df = pd.DataFrame({"image": gen_images, "labels": predictions_str})
      9 
     10 # Merge to enforce exact submission ordering and to avoid missing/extra rows issues.

NameError: name 'predictions_str' is not defined

## === cell 7
out_path = f"{WORK_ROOT}/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Submission preview:")
print(sub.head(10).to_string(index=False))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3876425436.py in <cell line: 0>()
      1 # Write valid .csv submission
      2 out_path = f"{WORK_ROOT}/submission.csv"
----> 3 sub.to_csv(out_path, index=False)
      4 print("Wrote:", out_path)
      5 print("Submission preview:")

NameError: name 'sub' is not defined
