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

0.3105632502308408

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import gc
import pickle
import random
import hashlib
import time

import numpy as np
import pandas as pd

from PIL import Image as PILImage

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

DATA_ROOT = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"




## === cell 2
sub_df = pd.read_csv(SAMPLE_SUB)
test_images = sub_df["image"].astype(str).tolist()
print("sample_submission rows:", len(test_images), "example:", test_images[:3])

test_paths = [os.path.join(TEST_IMG_DIR, img) for img in test_images]
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))
print("First path:", test_paths[0])




## === cell 3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

base = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
model_f = Model(inputs=base.input, outputs=base.output)
model_f.trainable = False

print("Feature dim:", model_f.output_shape)




## === cell 4
def _build_image_dataset(image_paths, batch_size, cache=False):
    paths = tf.constant(image_paths)

    def _load_one(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_and_crop_jpeg(
            img_bytes, crop_window=[0, 0, 0, 0], channels=3
        )  # no-op crop
        img = tf.image.resize(
            img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32)
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _feats_step(batch):
    batch = preprocess_input(batch)
    return model_f(batch, training=False)


def _paths_fingerprint(paths):
    h = hashlib.sha1()
    h.update(str(len(paths)).encode())
    if paths:
        h.update(paths[0].encode("utf-8", "ignore"))
        h.update(paths[-1].encode("utf-8", "ignore"))
    h.update(("\n".join(paths)).encode("utf-8", "ignore"))
    h.update(f"{IMG_HEIGHT}x{IMG_WIDTH}x{NR_CHANNELS}".encode())
    h.update(model_f.name.encode())
    h.update(str(model_f.output_shape[-1]).encode())
    return h.hexdigest()


def extract_features(image_paths, batch_size=32, cache_ds=False, cache_dir="./_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    n = len(image_paths)
    fp = _paths_fingerprint(image_paths)
    cache_path = os.path.join(cache_dir, f"resnet50_avg_{fp}.npy")

    if os.path.exists(cache_path):
        feats = np.load(cache_path, mmap_mode="r")
        if feats.shape[0] == n:
            return np.asarray(feats)

    ds = _build_image_dataset(image_paths, batch_size=batch_size, cache=cache_ds)

    feat_dim = int(model_f.output_shape[-1])
    feats = np.empty((n, feat_dim), dtype=np.float32)

    offset = 0
    for X in ds:
        out = _feats_step(X).numpy()
        bs = out.shape[0]
        feats[offset : offset + bs] = out
        offset += bs

    feats = feats[:n]  # safety; preserves ordering/shape
    np.save(cache_path, feats)
    return feats




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
train_df["image"] = train_df["image"].astype(str)
train_df["labels"] = train_df["labels"].astype(str)

all_labels = []
for labels in pd.unique(train_df["labels"]):
    all_labels.extend(labels.split())
tagnames = np.unique(np.array(all_labels, dtype=object))

print("Num train:", len(train_df), "Num tags:", len(tagnames))
print("Tags:", tagnames)




## === cell 6
train_paths = [os.path.join(TRAIN_IMG_DIR, img) for img in train_df["image"].tolist()]
for p in train_paths[:5]:
    assert os.path.exists(p), f"Missing train image: {p}"

avail_mask = np.fromiter(
    (os.path.exists(p) for p in test_paths), count=len(test_paths), dtype=bool
)
avail_idx = np.where(avail_mask)[0]
print("Available test images in this runtime:", len(avail_idx), "of", len(test_paths))

all_paths = train_paths + [test_paths[i] for i in avail_idx]

all_feats = extract_features(all_paths, batch_size=128, cache_ds=False)

Xf_train = all_feats[: len(train_paths)]
Xf_test = all_feats[len(train_paths) :]

print("Xf_train:", Xf_train.shape)
print("Xf_test:", Xf_test.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2468993198.py in <cell line: 0>()
     13 # Speedup: increase batch size for inference to improve GPU/CPU utilization; inference semantics unchanged.
     14 # This does not change the model, preprocessing, or feature values aside from negligible FP accumulation.
---> 15 all_feats = extract_features(all_paths, batch_size=128, cache_ds=False)
     16 
     17 Xf_train = all_feats[: len(train_paths)]

/tmp/ipykernel_11/3331533651.py in extract_features(image_paths, batch_size, cache_ds, cache_dir)
     69 
     70     offset = 0
---> 71     for X in ds:
     72         out = _feats_step(X).numpy()
     73         bs = out.shape[0]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} jpeg::Uncompress failed. Invalid JPEG data or crop window.
	 [[{{node DecodeAndCropJpeg}}]] [Op:IteratorGetNext] name: 

## === cell 7
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_avail = scaler.transform(Xf_test)

print("Scaled train/test:", trainXn.shape, testXn_avail.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3317925762.py in <cell line: 0>()
      1 scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
----> 2 trainXn = scaler.fit_transform(Xf_train)
      3 testXn_avail = scaler.transform(Xf_test)
      4 
      5 print("Scaled train/test:", trainXn.shape, testXn_avail.shape)

NameError: name 'Xf_train' is not defined

## === cell 8
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=list(tagnames))
Y = mlb.fit_transform(train_df["labels"].str.split()).astype(np.int32, copy=False)

from joblib import Parallel, delayed


def _fit_one_tag(i, t):
    y = Y[:, i]
    if y.min() == y.max():
        return t, None
    lr = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        random_state=SEED,
    )
    lr.fit(trainXn, y)
    return t, lr


n_jobs = max(1, min(os.cpu_count() or 1, 8))
fit_results = Parallel(n_jobs=n_jobs, backend="threading")(
    delayed(_fit_one_tag)(i, t) for i, t in enumerate(tagnames)
)
tagmodels1 = dict(fit_results)

skipped = [t for t, m in tagmodels1.items() if m is None]
for t in skipped:
    print(f"Tag {t}: skipped (only one class present in y)")

print(
    "Trained models:",
    sum(m is not None for m in tagmodels1.values()),
    "of",
    len(tagnames),
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/_utils.py", line 72, in __call__
    return self.func(**kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2889177498.py", line 19, in _fit_one_tag
    lr.fit(trainXn, y)
           ^^^^^^^
NameError: name 'trainXn' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2889177498.py in <cell line: 0>()
     22 
     23 n_jobs = max(1, min(os.cpu_count() or 1, 8))
---> 24 fit_results = Parallel(n_jobs=n_jobs, backend="threading")(
     25     delayed(_fit_one_tag)(i, t) for i, t in enumerate(tagnames)
     26 )

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

NameError: name 'trainXn' is not defined

## === cell 9
def _pred_one_tag(i, t):
    mdl = tagmodels1[t]
    if mdl is None:
        return i, None
    return i, mdl.predict_proba(testXn_avail)[:, 1].astype(np.float32, copy=False)


testKaggle_ppredscore1_avail = np.zeros(
    (testXn_avail.shape[0], len(tagnames)), dtype=np.float32
)

pred_results = Parallel(n_jobs=n_jobs, backend="threading")(
    delayed(_pred_one_tag)(i, t) for i, t in enumerate(tagnames)
)
for i, col in pred_results:
    if col is not None:
        testKaggle_ppredscore1_avail[:, i] = col

print("Pred score shape:", testKaggle_ppredscore1_avail.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4242491926.py in <cell line: 0>()
      7 
      8 testKaggle_ppredscore1_avail = np.zeros(
----> 9     (testXn_avail.shape[0], len(tagnames)), dtype=np.float32
     10 )
     11 

NameError: name 'testXn_avail' is not defined

## === cell 10
def class2tags(classes, tagnames):
    tags = []
    for row in classes:
        idx = np.flatnonzero(row)
        if idx.size:
            tags.append(" ".join(tagnames[idx]))
        else:
            tags.append("")
    return tags




## === cell 11
TH = 0.90
test_predclass_avail = testKaggle_ppredscore1_avail > TH
test_predtags_avail = class2tags(test_predclass_avail, tagnames)

test_predtags_avail = [
    s if len(s.strip()) > 0 else "healthy" for s in test_predtags_avail
]

print("Example preds:", test_predtags_avail[:10])




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3986552311.py in <cell line: 0>()
      1 TH = 0.90
----> 2 test_predclass_avail = testKaggle_ppredscore1_avail > TH
      3 test_predtags_avail = class2tags(test_predclass_avail, tagnames)
      4 
      5 test_predtags_avail = [

NameError: name 'testKaggle_ppredscore1_avail' is not defined

## === cell 12
full_predtags = ["healthy"] * len(test_images)
for out_pos, idx in enumerate(avail_idx):
    full_predtags[idx] = test_predtags_avail[out_pos]

submission = pd.DataFrame({"image": test_images, "labels": full_predtags})
print(submission.head())
print("Submission shape:", submission.shape)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3994478496.py in <cell line: 0>()
      1 full_predtags = ["healthy"] * len(test_images)
      2 for out_pos, idx in enumerate(avail_idx):
----> 3     full_predtags[idx] = test_predtags_avail[out_pos]
      4 
      5 submission = pd.DataFrame({"image": test_images, "labels": full_predtags})

NameError: name 'test_predtags_avail' is not defined

## === cell 13
out_path = "./submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path), "bytes")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2888061638.py in <cell line: 0>()
      1 out_path = "./submission.csv"
----> 2 submission.to_csv(out_path, index=False)
      3 print("Wrote:", out_path, "size:", os.path.getsize(out_path), "bytes")

NameError: name 'submission' is not defined
