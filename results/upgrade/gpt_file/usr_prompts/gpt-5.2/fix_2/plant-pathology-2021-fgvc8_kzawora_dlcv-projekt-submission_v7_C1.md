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

0.1917451523545707

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

INPUT_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
WORK_ROOT = "/kaggle/working"

train_csv_path = f"{INPUT_ROOT}/train.csv"
sample_sub_path = f"{INPUT_ROOT}/sample_submission.csv"
test_img_dir = f"{INPUT_ROOT}/test_images"

print("Exists train_csv:", os.path.exists(train_csv_path))
print("Exists sample_submission:", os.path.exists(sample_sub_path))
print("Exists test_img_dir:", os.path.isdir(test_img_dir))

sub_df = pd.read_csv(sample_sub_path)
print(sub_df.head())
print("n_test:", len(sub_df))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import tensorflow as tf

model_path = "/kaggle/input/dlcv-projekt/model-best.h5"

custom_objects = {}
try:
    class _DummyMetric(tf.keras.metrics.Metric):
        def __init__(self, name="dummy_metric", **kwargs):
            super().__init__(name=name, **kwargs)
            self.v = self.add_weight(name="v", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            return

        def result(self):
            return self.v

        def reset_states(self):
            self.v.assign(0.0)

    custom_objects.update(
        {
            "F1Score": _DummyMetric,
            "f1_score": _DummyMetric,
        }
    )
except Exception as e:
    print("Warning: could not create dummy metric:", e)

model = tf.keras.models.load_model(
    model_path, compile=False, custom_objects=custom_objects
)
print("Model loaded:", type(model))
print("Model input shape:", model.inputs[0].shape)
print("Model output shape:", model.outputs[0].shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2163375640.py in <cell line: 0>()
     36 # Load model (compile=False avoids needing training-time objects/metrics)
     37 # This is the key fix to avoid the tensorflow_addons/protobuf crash and get 'model' defined.
---> 38 model = tf.keras.models.load_model(
     39     model_path, compile=False, custom_objects=custom_objects
     40 )

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
import shutil

tmp_root = "/kaggle/tmp/test_dataset"
tmp_test_dir = f"{tmp_root}/test"
os.makedirs(tmp_test_dir, exist_ok=True)

for p in Path(tmp_test_dir).glob("*"):
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    except Exception:
        pass

missing = 0
for img_name in sub_df["image"].tolist():
    src = Path(test_img_dir) / img_name
    dst = Path(tmp_test_dir) / img_name
    if not src.exists():
        missing += 1
        continue
    try:
        os.symlink(str(src), str(dst))
    except FileExistsError:
        pass
    except OSError:
        shutil.copy2(str(src), str(dst))

print("Missing test images:", missing)
print("Prepared files:", len(list(Path(tmp_test_dir).glob("*.jpg"))))



## === cell 3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    tmp_root,
    class_mode=None,
    target_size=(380, 380),
    shuffle=False,  # critical: stable filename order
    batch_size=32,
)

print("Generator samples:", test_generator.samples)
print("First 5 generator filenames:", test_generator.filenames[:5])



## === cell 4
x = model.predict(test_generator, verbose=1)
print("Pred shape:", x.shape)
print("Pred sample row:", x[0])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4133250198.py in <cell line: 0>()
      1 # Predict for all samples
----> 2 x = model.predict(test_generator, verbose=1)
      3 print("Pred shape:", x.shape)
      4 print("Pred sample row:", x[0])
      5 

NameError: name 'model' is not defined

## === cell 5
labels = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]
threshold = 0.7

z = (x > threshold).astype(np.int32)
print("Binary preds shape:", z.shape)
print("Binary sample row:", z[0])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3373225537.py in <cell line: 0>()
      4 
      5 # Bugfix: use != 0 not "is not 0" (identity vs equality)
----> 6 z = (x > threshold).astype(np.int32)
      7 print("Binary preds shape:", z.shape)
      8 print("Binary sample row:", z[0])

NameError: name 'x' is not defined

## === cell 6
gen_filenames = [Path(f).name for f in test_generator.filenames]

predictions = [[labels[i] for i, j in enumerate(row) if j != 0] for row in z]
predictions_str = [" ".join(lst) if len(lst) > 0 else "complex" for lst in predictions]

pred_df = pd.DataFrame({"image": gen_filenames, "labels": predictions_str})

out_df = sub_df[["image"]].merge(pred_df, on="image", how="left")

out_df["labels"] = out_df["labels"].fillna("complex")

print(out_df.head())
print("Output rows:", len(out_df))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/970078235.py in <cell line: 0>()
      3 gen_filenames = [Path(f).name for f in test_generator.filenames]
      4 
----> 5 predictions = [[labels[i] for i, j in enumerate(row) if j != 0] for row in z]
      6 predictions_str = [" ".join(lst) if len(lst) > 0 else "complex" for lst in predictions]
      7 

NameError: name 'z' is not defined

## === cell 7
out_path = "submission.csv"
out_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
print("Submission columns:", list(pd.read_csv(out_path, nrows=1).columns))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3128097521.py in <cell line: 0>()
      1 # Write valid submission
      2 out_path = "submission.csv"
----> 3 out_df.to_csv(out_path, index=False)
      4 print("Wrote:", out_path)
      5 print(pd.read_csv(out_path).head())

NameError: name 'out_df' is not defined
