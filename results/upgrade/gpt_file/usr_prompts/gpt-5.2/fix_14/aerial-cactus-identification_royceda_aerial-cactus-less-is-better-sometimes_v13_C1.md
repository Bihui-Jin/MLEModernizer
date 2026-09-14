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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.8928

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9832) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime (your current `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting triggers the `MessageFactory.GetPrototype` error in this Kaggle image). I switch to the default C++ protobuf implementation and force it to be set before importing TensorFlow, which fixes the runtime error and lets training/prediction complete. To fix the “Invalid submission: ... same number of rows” issue, I make the data root detection deterministic (use the correct Kaggle folder), and I always build predictions in exactly the same row order as `sample_submission.csv` and validate against it before writing `submission.csv`. These are execution/format fixes and should be score-neutral beyond negligible numeric differences.'
- What this solution (achieved 0.99344) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow import* (the current “pop env vars” still leaves the runtime picking an incompatible protobuf path in this Kaggle image). I also make dataset root detection more robust by checking for both CSVs and ZIPs so we always unzip the intended files, and I ensure the test directory is the one actually containing exactly the IDs in `sample_submission.csv`. Finally, I keep the core model/training logic unchanged, but I hard-validate that prediction length and ID order exactly match `sample_submission.csv` before writing `submission.csv`, preventing the “same number of rows” submission error.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf setting (it’s what triggers the `MessageFactory.GetPrototype` error in this Kaggle image) and explicitly forcing the default C++ implementation before importing TensorFlow. This is an execution-only fix and should be score-neutral (negligible numeric differences at most). I also keep your dataset root detection, generators, training loop, and submission alignment checks unchanged, so the pipeline still trains and writes a valid `submission.csv` matching `sample_submission.csv` row order.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf environment variables (they trigger the `MessageFactory.GetPrototype` error in this Kaggle image) and explicitly forcing the default C++ protobuf implementation before TensorFlow is imported. Then I make the data root detection deterministic for this dataset layout and ensure we always unzip into a clean, known folder so we don’t accidentally pick up nested `.../test/test` folders that can cause row-count mismatches. Finally, I keep your generators/model/training intact but tighten submission alignment: predictions be generated in exactly the `sample_submission.csv` row order and we validate count and IDs before writing `submission.csv` to guarantee Kaggle accepts it.'
- What this solution (achieved 0.5) has done: 'I fix the data extraction/directory discovery so `TRAIN_DIR` and `TEST_DIR` are correctly found after unzipping (the current `rglob('train')` misses the typical `train/` folder because it’s searching for an exact-name directory without a trailing slash match). I also fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation (and disabling C descriptors) *before* importing TensorFlow, which avoids the missing `google.protobuf.pyext._message` error in this environment. These changes unblock training and prediction, and should move your AUC up from 0.5 toward the target by producing real model probabilities instead of failing/degenerate output. Finally, I keep your model, generators, training loop, and submission alignment checks intact so the evaluation semantics remain the same and the submission is guaranteed valid.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_C_DESCRIPTORS", "1")

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

for dirname, _, filenames in os.walk("/kaggle/input/aerial-cactus-identification"):
    for i, filename in enumerate(filenames[:5]):
        print(os.path.join(dirname, filename))
    break



## === cell 1
import subprocess, pathlib, sys, shutil

workdir = "/kaggle/working"
os.makedirs(workdir, exist_ok=True)


def run(cmd):
    print(cmd)
    subprocess.check_call(cmd, shell=True)


CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
]

DATA_ROOT = None
for p in CANDIDATES:
    if (
        os.path.isfile(os.path.join(p, "train.csv"))
        and os.path.isfile(os.path.join(p, "sample_submission.csv"))
        and os.path.isfile(os.path.join(p, "train.zip"))
        and os.path.isfile(os.path.join(p, "test.zip"))
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(f"Could not find dataset root in candidates: {CANDIDATES}")

print("Using DATA_ROOT:", DATA_ROOT)

EXTRACT_ROOT = os.path.join(workdir, "aerial_cactus_extracted")
os.makedirs(EXTRACT_ROOT, exist_ok=True)

for leaf in ["train", "test", "aerial-cactus-identification"]:
    p = os.path.join(EXTRACT_ROOT, leaf)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)

run(f"cp -f {DATA_ROOT}/train.csv {workdir}/train.csv")
run(f"cp -f {DATA_ROOT}/sample_submission.csv {workdir}/sample_submission.csv")
run(f"unzip -o {DATA_ROOT}/train.zip -d {EXTRACT_ROOT}")
run(f"unzip -o {DATA_ROOT}/test.zip -d {EXTRACT_ROOT}")


def find_image_dir(root, leaf):
    """
    FIX: Previous implementation searched rglob(leaf) which may miss typical folders
    depending on extraction structure; robustly search all directories named leaf
    AND also accept paths that end with '/leaf' among all subfolders, then require jpgs.
    Prefer non-duplicated .../leaf/leaf and shallower paths.
    """
    root_p = pathlib.Path(root)
    candidates = []
    for p in root_p.rglob("*"):
        if p.is_dir() and p.name == leaf and any(p.glob("*.jpg")):
            candidates.append(p)

    if not candidates:
        return None

    def score(p):
        parts = p.parts
        dup_penalty = int(len(parts) >= 2 and parts[-1] == leaf and parts[-2] == leaf)
        return (dup_penalty, len(parts), str(p))

    candidates.sort(key=score)
    return str(candidates[0])


TRAIN_DIR = find_image_dir(EXTRACT_ROOT, "train")
TEST_DIR = find_image_dir(EXTRACT_ROOT, "test")

print("Detected TRAIN_DIR:", TRAIN_DIR)
print("Detected TEST_DIR :", TEST_DIR)
print("Exists train.csv:", os.path.isfile(os.path.join(workdir, "train.csv")))
print(
    "Exists sample_submission.csv:",
    os.path.isfile(os.path.join(workdir, "sample_submission.csv")),
)

if TRAIN_DIR is None or TEST_DIR is None:
    print("Debug: top-level extracted dirs:", os.listdir(EXTRACT_ROOT)[:50])
    raise FileNotFoundError(
        f"Could not locate extracted image directories. TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}. "
        f"Check unzip output under {EXTRACT_ROOT}."
    )



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2649627043.py in <cell line: 0>()
     84     # Provide debug listing for the extracted structure
     85     print("Debug: top-level extracted dirs:", os.listdir(EXTRACT_ROOT)[:50])
---> 86     raise FileNotFoundError(
     87         f"Could not locate extracted image directories. TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}. "
     88         f"Check unzip output under {EXTRACT_ROOT}."

FileNotFoundError: Could not locate extracted image directories. TRAIN_DIR=None, TEST_DIR=None. Check unzip output under /kaggle/working/aerial_cactus_extracted.

## === cell 2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

print("tf version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

tf.keras.utils.set_random_seed(42)
np.random.seed(42)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv("/kaggle/working/train.csv")
print(df.head())
df.has_cactus.value_counts().plot.bar()
plt.show()



## === cell 4
from tensorflow.keras.utils import load_img

filename = df.id.iloc[10]
print("Example file:", filename)
image_path = os.path.join(TRAIN_DIR, filename)
print("Resolved path:", image_path, "exists:", os.path.isfile(image_path))
image = load_img(image_path)
plt.imshow(image)
plt.axis("off")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2647921589.py in <cell line: 0>()
      3 filename = df.id.iloc[10]
      4 print("Example file:", filename)
----> 5 image_path = os.path.join(TRAIN_DIR, filename)
      6 print("Resolved path:", image_path, "exists:", os.path.isfile(image_path))
      7 image = load_img(image_path)

/usr/lib/python3.11/posixpath.py in join(a, *p)

TypeError: expected str, bytes or os.PathLike object, not NoneType

## === cell 5
try:
    train_test_split
except NameError:
    from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

print(train_df.shape, validate_df.shape)

train_exists = (
    train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
)
valid_exists = (
    validate_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
)
print("Train files present ratio:", train_exists)
print("Valid files present ratio:", valid_exists)
if train_exists < 0.99 or valid_exists < 0.99:
    missing = (
        train_df.loc[
            ~train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))),
            "id",
        ]
        .head(5)
        .tolist()
    )
    raise FileNotFoundError(
        f"Some training images are missing under TRAIN_DIR={TRAIN_DIR}. Example missing: {missing}"
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1989387007.py in <cell line: 0>()
     13 
     14 train_exists = (
---> 15     train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
     16 )
     17 valid_exists = (

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/1989387007.py in <lambda>(x)
     13 
     14 train_exists = (
---> 15     train_df["id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_DIR, x))).mean()
     16 )
     17 valid_exists = (

/usr/lib/python3.11/posixpath.py in join(a, *p)

TypeError: expected str, bytes or os.PathLike object, not NoneType

## === cell 6
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255.0,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 7
BATCH_SIZE = 2**10
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    seed=42,
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)

print("train_generator batches:", len(train_generator))
print("validation_generator batches:", len(validation_generator))
if len(train_generator) == 0 or len(validation_generator) == 0:
    raise ValueError(
        f"Generator length is 0. TRAIN_DIR={TRAIN_DIR}. "
        f"Check that images exist and dataframe ids match filenames."
    )



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/360972142.py in <cell line: 0>()
     31 print("validation_generator batches:", len(validation_generator))
     32 if len(train_generator) == 0 or len(validation_generator) == 0:
---> 33     raise ValueError(
     34         f"Generator length is 0. TRAIN_DIR={TRAIN_DIR}. "
     35         f"Check that images exist and dataframe ids match filenames."

ValueError: Generator length is 0. TRAIN_DIR=None. Check that images exist and dataframe ids match filenames.

## === cell 8
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    AveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential(
    [
        Conv2D(
            filters=64,
            kernel_size=(4, 4),
            strides=(1, 1),
            activation="relu",
            input_shape=INPUT_SHAPE,
            padding="same",
        ),
        BatchNormalization(),
        AveragePooling2D(pool_size=(3, 3)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.45),
        Dense(1, activation="sigmoid"),
    ]
)

earlystop = EarlyStopping(patience=4, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.summary()



## === cell 9
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[earlystop],
    verbose=2,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/731527035.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_generator,
      3     epochs=30,
      4     validation_data=validation_generator,
      5     callbacks=[earlystop],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 10
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/206874453.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).plot()
      2 plt.grid(True)
      3 plt.show()
      4 

NameError: name 'history' is not defined

## === cell 11
sample_sub_path = os.path.join("/kaggle/working", "sample_submission.csv")
sub_df = pd.read_csv(sample_sub_path)

assert (
    "id" in sub_df.columns and "has_cactus" in sub_df.columns
), "sample_submission.csv must contain columns: id, has_cactus"


def best_dir_for_ids(initial_dir, ids):
    candidates = []
    search_roots = [pathlib.Path(EXTRACT_ROOT)]
    for r in search_roots:
        for p in r.rglob("test"):
            if p.is_dir():
                candidates.append(p)
    if initial_dir is not None:
        initial = pathlib.Path(initial_dir)
        if initial.is_dir():
            candidates = [initial] + candidates

    uniq = []
    seen = set()
    for p in candidates:
        sp = str(p)
        if sp not in seen:
            uniq.append(p)
            seen.add(sp)

    ids_list = ids.tolist()
    best = None
    best_cov = -1.0
    best_depth = 10**9
    for p in uniq:
        cov = float(np.mean([os.path.isfile(p / i) for i in ids_list]))
        depth = len(p.parts)
        if (cov > best_cov) or (cov == best_cov and depth < best_depth):
            best, best_cov, best_depth = p, cov, depth
        if cov >= 0.9999:
            break
    return str(best), best_cov


TEST_DIR, coverage = best_dir_for_ids(TEST_DIR, sub_df["id"])
print("Adjusted TEST_DIR :", TEST_DIR)
print("Coverage for sample ids in TEST_DIR:", coverage)

test_exists = (
    sub_df["id"].apply(lambda x: os.path.isfile(os.path.join(TEST_DIR, x))).mean()
)
print("Test files present ratio:", test_exists)
if test_exists < 0.99:
    missing = (
        sub_df.loc[
            ~sub_df["id"].apply(lambda x: os.path.isfile(os.path.join(TEST_DIR, x))),
            "id",
        ]
        .head(10)
        .tolist()
    )
    raise FileNotFoundError(
        f"Some test images are missing under TEST_DIR={TEST_DIR}. Example missing: {missing}"
    )

test_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_gen.flow_from_dataframe(
    dataframe=sub_df[["id"]],  # preserve exact row order of sample_submission
    directory=TEST_DIR,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("test_generator batches:", len(test_generator))
if len(test_generator) == 0:
    raise ValueError(f"Test generator length is 0. TEST_DIR={TEST_DIR}.")

pred = model.predict(test_generator, verbose=0).reshape(-1)

n = len(sub_df)
if len(pred) != n:
    raise ValueError(
        f"Prediction length mismatch: got {len(pred)} predictions for {n} test ids."
    )

sub_df["has_cactus"] = pred.astype(np.float32)

print(sub_df.head())
print(
    "Pred min/max:",
    float(sub_df["has_cactus"].min()),
    float(sub_df["has_cactus"].max()),
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1612888316.py in <cell line: 0>()
     60         .tolist()
     61     )
---> 62     raise FileNotFoundError(
     63         f"Some test images are missing under TEST_DIR={TEST_DIR}. Example missing: {missing}"
     64     )

FileNotFoundError: Some test images are missing under TEST_DIR=None. Example missing: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '134f04305c795d6d202502c2ce3578f3.jpg', '41fad8d145e6c41868ce3617e30a2545.jpg', '35f8a11352c8d41b6231bb33d8d09f7e.jpg', 'b77dc902b035887cbbc01920ce0e3151.jpg', '33baa477c94f9171566f52c4a8f53ec8.jpg', 'bc6f04ed633c3e3e52125f9b6c078bf3.jpg', '9f988c7fcb21f35f523b0faecc590cb7.jpg', 'ca3a10b0f34219a40f6d6d5a2d863826.jpg', '49d51f2e7dc3376b91ed29144bb66df5.jpg']

## === cell 12
submission = sub_df[["id", "has_cactus"]].copy()
out_path = "/kaggle/working/submission.csv"

expected = pd.read_csv(sample_sub_path)
if len(submission) != len(expected):
    raise ValueError(f"Row mismatch: {len(submission)} vs {len(expected)}")
if list(submission.columns) != ["id", "has_cactus"]:
    raise ValueError(f"Column mismatch: {submission.columns.tolist()}")
if submission["id"].tolist() != expected["id"].tolist():
    raise ValueError("ID order mismatch vs sample_submission.csv")

submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission))



## === cell 13
print(submission.columns.tolist())
print(submission.isna().sum())
print(submission.head())
print(submission["has_cactus"].describe())



## === cell 14
print(os.listdir("/kaggle/working")[:50])
print("Submission exists:", os.path.isfile("/kaggle/working/submission.csv"))

expected = pd.read_csv(sample_sub_path)
assert len(submission) == len(
    expected
), f"Row mismatch: {len(submission)} vs {len(expected)}"
assert list(submission.columns) == ["id", "has_cactus"]
assert (
    submission["id"].tolist() == expected["id"].tolist()
), "ID order mismatch vs sample_submission.csv"
print("Submission validated with rows:", len(expected))
