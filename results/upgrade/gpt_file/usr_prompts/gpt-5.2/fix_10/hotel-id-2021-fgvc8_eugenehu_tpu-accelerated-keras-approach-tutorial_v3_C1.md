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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.0463500884061631

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00186) has done: 'Main bottlenecks are (1) the per-path existence check using `tf.map_fn(... .numpy() ...)` which forces slow Python execution and (2) caching full decoded/resized train/val/test image tensors to disk, which adds huge IO/serialization overhead and can easily dominate runtime. I replace the existence check with a fast vectorized `tf.io.gfile.exists` loop in Python (no TF graph/`map_fn`) and keep only lightweight caching (file-path lists) while relying on parallel decode/resize + prefetch for throughput. I also make the input pipeline more efficient but equivalent by using `num_parallel_calls=AUTO`, `deterministic=True`, adding `prefetch(AUTO)` before expensive stages where appropriate, and avoiding redundant dataset materialization; model/training/prediction logic and semantics remain unchanged.'
- What this solution (achieved 0.00245) has done: 'I fix the crash at import time caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing protobuf to use the pure-Python implementation before importing TensorFlow. Then I keep your training/inference logic the same, but make one minimal, score-improving change: switch ResNet50 to ImageNet pretrained weights (still frozen) so predictions aren’t effectively random, which should move MAP@5 substantially toward your target. I also make the train/val split happen after dropping missing image paths to avoid mismatched indexing, and ensure the submission is aligned to `sample_submission.csv` and always written as `submission.csv`. All other architecture, loss, and loops remain unchanged.'
- What this solution (achieved 0.00245) has done: 'We fix the import-time crash by setting both protobuf environment variables early enough and by making the TensorFlow import robust to the protobuf 6.x / TF 2.18 incompatibility. Then we fix a path bug where `DIR="../input/hotel-id-2021-fgvc8"` may not exist in this environment by auto-selecting the first existing dataset root from the provided paths (without changing the rest of the pipeline). Finally, to improve MAP@5 toward your target with minimal semantic change, we ensure test images are predicted in the exact `sample_submission.csv` order (instead of filesystem glob order), preventing misalignment that can severely depress the score.'
- What this solution (achieved 0.00245) has done: 'You’re hitting a TensorFlow import crash caused by the protobuf 6.x runtime API change (`MessageFactory.GetPrototype`), so the pipeline never reaches training/inference or writes `submission.csv`. I fix this by forcing TensorFlow to use the C++ protobuf implementation (instead of the pure-Python one) and by importing TensorFlow first (before importing `tensorflow_hub/tfds` etc.), which resolves this specific error in TF 2.18 + protobuf 6 in Kaggle environments. After that, I keep your model/training/prediction logic the same, but ensure the test image paths are constructed from `sample_submission.csv` order (already done) and that any missing test files still produce valid 5-id strings so a valid `submission.csv` is always written. These changes are execution-unblocking and score-positive (the model can actually run), without altering your core architecture/training semantics.'
- What this solution (achieved 0.00245) has done: 'I fix the TensorFlow import crash by setting protobuf environment variables before any TensorFlow-related import so TF 2.18 can work with protobuf 6.x in this environment. Then I keep your model/training/prediction logic unchanged, but make the pipeline robust by ensuring the dataset path resolution is stable and the test predictions remain aligned to `sample_submission.csv` order (which strongly affects MAP@5). Finally, I keep the submission generation identical but ensure it always writes a valid `submission.csv` with the required columns even if some test files are missing.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed. Env vars were set before TF import to avoid known "
        "TF 2.18 + protobuf 6.x compatibility crashes."
    ) from e

from sklearn import preprocessing

print("TF:", tf.__version__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism setting skipped:", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/52812684.py in <cell line: 0>()
     14 try:
---> 15     import tensorflow as tf
     16 except Exception as e:

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/52812684.py in <cell line: 0>()
     15     import tensorflow as tf
     16 except Exception as e:
---> 17     raise RuntimeError(
     18         "TensorFlow import failed. Env vars were set before TF import to avoid known "
     19         "TF 2.18 + protobuf 6.x compatibility crashes."

RuntimeError: TensorFlow import failed. Env vars were set before TF import to avoid known TF 2.18 + protobuf 6.x compatibility crashes.

## === cell 1
def intialize_accel(hardware):
    """
    input:
    str: GPU or TPU for hardware accelerator

    output:
    strategy -- used later for model definition and fitting
    """
    if hardware == "TPU":
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print("TPU Initialized")
            print("TPU Units:", strategy.num_replicas_in_sync)
            return strategy
        except Exception as e:
            print("TPU Initialization Failed:", repr(e))
            print("Falling back to default strategy.")
            return tf.distribute.get_strategy()

    elif hardware == "GPU":
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
        try:
            strategy = tf.distribute.MirroredStrategy()
        except Exception as e:
            print("MirroredStrategy init failed:", repr(e))
            strategy = tf.distribute.get_strategy()
        return strategy

    print("Unknown hardware option; using default strategy.")
    return tf.distribute.get_strategy()


strategy = intialize_accel("TPU")
AUTO = tf.data.AUTOTUNE



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3167848309.py in intialize_accel(hardware)
     10         try:
---> 11             resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
     12             tf.config.experimental_connect_to_cluster(resolver)

NameError: name 'tf' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3167848309.py in <cell line: 0>()
     34 
     35 
---> 36 strategy = intialize_accel("TPU")
     37 AUTO = tf.data.AUTOTUNE
     38 

/tmp/ipykernel_55/3167848309.py in intialize_accel(hardware)
     19             print("TPU Initialization Failed:", repr(e))
     20             print("Falling back to default strategy.")
---> 21             return tf.distribute.get_strategy()
     22 
     23     elif hardware == "GPU":

NameError: name 'tf' is not defined

## === cell 2
CANDIDATE_DIRS = [
    "../input/hotel-id-2021-fgvc8",
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "../input",
    "/kaggle/input",
]
DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(d) and os.path.isdir(d):
        if os.path.basename(d) == "hotel-id-2021-fgvc8":
            DIR = d
            break
        if os.path.exists(os.path.join(d, "hotel-id-2021-fgvc8")):
            DIR = os.path.join(d, "hotel-id-2021-fgvc8")
            break

if DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset directory. Tried: " + ", ".join(CANDIDATE_DIRS)
    )

Train_PATH = os.path.join(DIR, "train_images")
Test_PATH = os.path.join(DIR, "test_images")

train_csv_path = os.path.join(DIR, "train.csv")
sample_sub_path = os.path.join(DIR, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
train_df = train_df.drop_duplicates(subset=["image"]).reset_index(drop=True)

print("Using DIR:", DIR)
print("Number of unique hotel chains: ", train_df.chain.nunique())
print("Number of unique hotels: ", train_df.hotel_id.nunique())
print("Number of Training Samples: ", train_df.shape[0])
print(train_df.head())



## === cell 3
Classes = train_df.hotel_id.nunique()
Channels = 3
size = (200, 200)



## === cell 4
le = preprocessing.LabelEncoder()
train_df["label"] = le.fit_transform(train_df["hotel_id"])

train_df["image_path"] = (
    Train_PATH
    + os.sep
    + train_df["chain"].astype(str).to_numpy()
    + os.sep
    + train_df["image"].to_numpy()
)

paths_list = train_df["image_path"].tolist()
exists_mask_np = np.fromiter(
    (tf.io.gfile.exists(p) for p in paths_list), dtype=bool, count=len(paths_list)
)
missing = int((~exists_mask_np).sum())
if missing:
    print(f"Warning: {missing} train image paths missing; dropping them.")
train_df = train_df.loc[exists_mask_np].reset_index(drop=True)

print(train_df[["hotel_id", "label", "image_path"]].head())

Split = int(0.9 * train_df.shape[0])




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2017689556.py in <cell line: 0>()
----> 1 le = preprocessing.LabelEncoder()
      2 train_df["label"] = le.fit_transform(train_df["hotel_id"])
      3 
      4 train_df["image_path"] = (
      5     Train_PATH

NameError: name 'preprocessing' is not defined

## === cell 5
def image_proces(path, labels):
    """
    Reads, decodes, resizes and scales image to [0,1].
    """
    data = tf.io.read_file(path)
    data = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_FAST")
    data = tf.image.resize(data, size)
    data = tf.cast(data, tf.float32) / 255.0
    return data, labels


_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.deterministic = True  # keep stable order for reproducibility
try:
    _DS_OPTIONS.threading.private_threadpool_size = 16
except Exception:
    pass


def import_image(paths, labels, cache=False, cache_path=None):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.with_options(_DS_OPTIONS)
    dataset = dataset.map(image_proces, num_parallel_calls=AUTO)
    if cache:
        dataset = (
            dataset.cache(cache_path) if cache_path is not None else dataset.cache()
        )
    return dataset


def data_augment(image, labels):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, labels


sample = pd.read_csv(sample_sub_path)
sample_images = sample["image"].tolist()

Paths_Test = [os.path.join(Test_PATH, img) for img in sample_images]

exists_test = np.fromiter(
    (tf.io.gfile.exists(p) for p in Paths_Test), dtype=bool, count=len(Paths_Test)
)
if not exists_test.all():
    missing_test = int((~exists_test).sum())
    print(
        f"Warning: {missing_test} test image paths missing; will still create submission."
    )
print("Test images (from sample_submission):", len(Paths_Test))

dataset_Test = import_image(
    Paths_Test,
    np.arange(len(Paths_Test), dtype=np.int32),
    cache=False,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1821746698.py in <cell line: 0>()
     10 
     11 
---> 12 _DS_OPTIONS = tf.data.Options()
     13 _DS_OPTIONS.deterministic = True  # keep stable order for reproducibility
     14 try:

NameError: name 'tf' is not defined

## === cell 6
print("Number of Test Samples:", dataset_Test.cardinality().numpy())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2239041905.py in <cell line: 0>()
----> 1 print("Number of Test Samples:", dataset_Test.cardinality().numpy())
      2 

NameError: name 'dataset_Test' is not defined

## === cell 7
paths = train_df["image_path"].values
labels = train_df["label"].values.astype(np.int32)

train_paths, val_paths = paths[:Split], paths[Split:]
train_labels, val_labels = labels[:Split], labels[Split:]

BATCH_SIZE = 32

train_dataset_base = import_image(
    train_paths,
    train_labels,
    cache=False,
)
train_dataset = train_dataset_base.map(data_augment, num_parallel_calls=AUTO)
train_dataset = train_dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

val_dataset = import_image(
    val_paths,
    val_labels,
    cache=False,
)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_dataset).numpy())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'image_path'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3356196781.py in <cell line: 0>()
----> 1 paths = train_df["image_path"].values
      2 labels = train_df["label"].values.astype(np.int32)
      3 
      4 train_paths, val_paths = paths[:Split], paths[Split:]
      5 train_labels, val_labels = labels[:Split], labels[Split:]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'image_path'

## === cell 8
def create_model(Base, input_shape):
    inputs = tf.keras.Input(shape=tuple(input_shape))

    norm = tf.keras.layers.Normalization()
    x = norm(inputs)

    x = Base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(50, activation="relu", dtype="float32")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(Classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    model._norm_layer = norm  # keep reference for later adapt
    return model




## === cell 9
def compile_model(model, lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, steps_per_execution=8
    )
    return model




## === cell 10
EPOCHS = 1
VERBOSE = 1

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.1,
    patience=3,
    mode="max",
    min_delta=0.0001,
    verbose=1,
)

checkpoint_filepath = "./best_model.weights.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, mode="max", min_delta=0.0001, verbose=1
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1969134924.py in <cell line: 0>()
      2 VERBOSE = 1
      3 
----> 4 reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
      5     monitor="val_accuracy",
      6     factor=0.1,

NameError: name 'tf' is not defined

## === cell 11
input_shape = [200, 200, Channels]

with strategy.scope():
    Base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=tuple(input_shape)
    )
    Base.trainable = False
    model = create_model(Base, input_shape)
    model = compile_model(model, lr=0.001)

try:
    adapt_ds = (
        train_dataset_base.map(lambda x, y: x, num_parallel_calls=AUTO)
        .take(512)
        .prefetch(AUTO)
    )
    model._norm_layer.adapt(adapt_ds)
    print("Normalization layer adapted.")
except Exception as e:
    print("Normalization adapt skipped due to:", repr(e))

print("Fitting")
History = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr, model_checkpoint_callback, callback],
    validation_data=val_dataset,
    verbose=VERBOSE,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3560007624.py in <cell line: 0>()
      1 input_shape = [200, 200, Channels]
      2 
----> 3 with strategy.scope():
      4     Base = tf.keras.applications.ResNet50(
      5         weights="imagenet", include_top=False, input_shape=tuple(input_shape)

NameError: name 'strategy' is not defined

## === cell 12
if os.path.exists(checkpoint_filepath):
    model.load_weights(checkpoint_filepath)
    print("Loaded best weights from:", checkpoint_filepath)
else:
    print("Best weights file not found; using last epoch weights.")

best_model = model



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1782022114.py in <cell line: 0>()
----> 1 if os.path.exists(checkpoint_filepath):
      2     model.load_weights(checkpoint_filepath)
      3     print("Loaded best weights from:", checkpoint_filepath)
      4 else:
      5     print("Best weights file not found; using last epoch weights.")

NameError: name 'checkpoint_filepath' is not defined

## === cell 13
dataset_Test_batched = dataset_Test.batch(32, drop_remainder=False).prefetch(AUTO)
predictions = best_model.predict(dataset_Test_batched, verbose=1)
print("Pred shape:", predictions.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1878213060.py in <cell line: 0>()
----> 1 dataset_Test_batched = dataset_Test.batch(32, drop_remainder=False).prefetch(AUTO)
      2 predictions = best_model.predict(dataset_Test_batched, verbose=1)
      3 print("Pred shape:", predictions.shape)
      4 

NameError: name 'dataset_Test' is not defined

## === cell 14
topk = 5
k = topk
part = np.argpartition(-predictions, kth=k - 1, axis=1)[:, :k]
row_idx = np.arange(predictions.shape[0])[:, None]
part_scores = predictions[row_idx, part]
order_within = np.argsort(-part_scores, axis=1)
topk_idx = part[row_idx, order_within]

topk_hotel_ids = le.inverse_transform(topk_idx.reshape(-1)).reshape(-1, topk)

images_names = [os.path.basename(p) for p in Paths_Test]
pred_strings = [" ".join(map(str, row)) for row in topk_hotel_ids]

submission = pd.DataFrame({"image": images_names, "hotel_id": pred_strings})

sample = pd.read_csv(sample_sub_path)
submission = sample[["image"]].merge(submission, on="image", how="left")
if submission["hotel_id"].isna().any():
    fallback_ids = le.inverse_transform(np.arange(min(topk, Classes))).tolist()
    if len(fallback_ids) < topk:
        fallback_ids = (fallback_ids * (topk // len(fallback_ids) + 1))[:topk]
    fallback = " ".join(map(str, fallback_ids))
    submission["hotel_id"] = submission["hotel_id"].fillna(fallback)

print(submission.head())
print("Submission rows:", len(submission))

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3657097791.py in <cell line: 0>()
      1 topk = 5
      2 k = topk
----> 3 part = np.argpartition(-predictions, kth=k - 1, axis=1)[:, :k]
      4 row_idx = np.arange(predictions.shape[0])[:, None]
      5 part_scores = predictions[row_idx, part]

NameError: name 'predictions' is not defined
