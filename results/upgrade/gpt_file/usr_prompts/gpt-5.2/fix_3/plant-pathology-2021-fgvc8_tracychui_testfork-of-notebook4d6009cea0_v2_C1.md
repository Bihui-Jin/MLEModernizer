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

0.3428175702413932

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
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.applications.resnet50 as resnet

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"



## === cell 2
imglist_test = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
print("n_test_images:", len(imglist_test))
assert len(imglist_test) > 0, "No test images found - check TEST_DIR path."



## === cell 3
training_csv = pd.read_csv(TRAIN_CSV)
training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
tagnames = np.unique(training_class)
print("n_train:", len(training_csv), "n_tags:", len(tagnames))
print("tags:", tagnames)




## === cell 4
def labels_to_multihot(labels_series, tagnames):
    tag2idx = {t: i for i, t in enumerate(tagnames)}
    Y = np.zeros((len(labels_series), len(tagnames)), dtype=np.int8)
    for n, lab in enumerate(labels_series):
        for t in str(lab).split():
            if t in tag2idx:
                Y[n, tag2idx[t]] = 1
    return Y


Y = labels_to_multihot(training_csv["labels"], tagnames)



## === cell 5
train_paths = [os.path.join(TRAIN_DIR, fn) for fn in training_csv["image"].tolist()]
exists_mask = np.array([os.path.exists(p) for p in train_paths])
if not np.all(exists_mask):
    missing = np.sum(~exists_mask)
    print(f"Warning: {missing} training images missing; filtering them out.")
    training_csv = training_csv.loc[exists_mask].reset_index(drop=True)
    train_paths = [p for p, ok in zip(train_paths, exists_mask) if ok]
    Y = Y[exists_mask]

print("n_train_after_filter:", len(train_paths), "Y shape:", Y.shape)



## === cell 6
base = resnet.ResNet50(
    weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base(inp, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
feat_model = keras.Model(inp, x)
feat_dim = feat_model.output_shape[-1]
print("feature dim:", feat_dim)




## === cell 7
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    return img


def extract_features(paths, batch_size=32):
    paths_tf = tf.constant(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths_tf)
    opts = tf.data.Options()
    try:
        opts.experimental_deterministic = True
    except Exception:
        pass
    ds = ds.with_options(opts)
    ds = ds.map(_decode_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    feats = np.zeros((len(paths), feat_dim), dtype=np.float32)
    start = 0
    for batch in ds:
        f = feat_model.predict_on_batch(batch).numpy()
        bs = f.shape[0]
        feats[start : start + bs] = f
        start += bs
    return feats




## === cell 8
Xf_train = extract_features(train_paths, batch_size=64)
print("Xf_train:", Xf_train.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2952937231.py in <cell line: 0>()
----> 1 Xf_train = extract_features(train_paths, batch_size=64)
      2 print("Xf_train:", Xf_train.shape)
      3 

/tmp/ipykernel_11/182032998.py in extract_features(paths, batch_size)
     28     start = 0
     29     for batch in ds:
---> 30         f = feat_model.predict_on_batch(batch).numpy()
     31         bs = f.shape[0]
     32         feats[start : start + bs] = f

AttributeError: 'numpy.ndarray' object has no attribute 'numpy'

## === cell 9
Xf_test = extract_features(imglist_test, batch_size=64)
print("Xf_test:", Xf_test.shape)

gc.collect()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1421650118.py in <cell line: 0>()
----> 1 Xf_test = extract_features(imglist_test, batch_size=64)
      2 print("Xf_test:", Xf_test.shape)
      3 
      4 gc.collect()
      5 

/tmp/ipykernel_11/182032998.py in extract_features(paths, batch_size)
     28     start = 0
     29     for batch in ds:
---> 30         f = feat_model.predict_on_batch(batch).numpy()
     31         bs = f.shape[0]
     32         feats[start : start + bs] = f

AttributeError: 'numpy.ndarray' object has no attribute 'numpy'

## === cell 10
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_test = scaler.transform(Xf_test)
print("trainXn:", trainXn.shape, "testXn_test:", testXn_test.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/953458219.py in <cell line: 0>()
      1 scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
----> 2 trainXn = scaler.fit_transform(Xf_train)
      3 testXn_test = scaler.transform(Xf_test)
      4 print("trainXn:", trainXn.shape, "testXn_test:", testXn_test.shape)
      5 

NameError: name 'Xf_train' is not defined

## === cell 11
from joblib import Parallel, delayed


def _fit_one_tag(i, t):
    y = Y[:, i]
    if y.sum() == 0 or y.sum() == len(y):
        return t, None
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,  # keep single-thread per model to avoid oversubscription; parallelism is outside
    )
    lr.fit(trainXn, y)
    return t, lr


results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(_fit_one_tag)(i, t) for i, t in enumerate(tagnames)
)
tagmodels1 = dict(results)

print(
    "trained tag models:",
    sum(m is not None for m in tagmodels1.values()),
    "/",
    len(tagnames),
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/3605888351.py", line 17, in _fit_one_tag
NameError: name 'trainXn' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3605888351.py in <cell line: 0>()
     20 
     21 # Use all available cores; sklearn/joblib is deterministic given fixed data and random_state.
---> 22 results = Parallel(n_jobs=-1, prefer="processes")(
     23     delayed(_fit_one_tag)(i, t) for i, t in enumerate(tagnames)
     24 )

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

## === cell 12
testKaggle_ppredscore1 = np.zeros((len(testXn_test), len(tagnames)), dtype=np.float32)
for i, t in enumerate(tagnames):
    model = tagmodels1[t]
    if model is None:
        y = Y[:, i]
        const = 1.0 if y.sum() == len(y) else 0.0
        testKaggle_ppredscore1[:, i] = const
    else:
        testKaggle_ppredscore1[:, i] = model.predict_proba(testXn_test)[:, 1]

print("pred score matrix:", testKaggle_ppredscore1.shape)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1622876929.py in <cell line: 0>()
----> 1 testKaggle_ppredscore1 = np.zeros((len(testXn_test), len(tagnames)), dtype=np.float32)
      2 for i, t in enumerate(tagnames):
      3     model = tagmodels1[t]
      4     if model is None:
      5         y = Y[:, i]

NameError: name 'testXn_test' is not defined

## === cell 13
def class2tags(classes, tagnames):
    tagnames = np.asarray(tagnames)
    return [" ".join(tagnames[row]) for row in classes]




## === cell 14
test_predclass = testKaggle_ppredscore1 > 0.5
test_predtags = class2tags(test_predclass, tagnames)

for i in range(len(test_predtags)):
    if test_predtags[i] == "":
        test_predtags[i] = "healthy"

print("example preds:", test_predtags[:5])



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2578001954.py in <cell line: 0>()
----> 1 test_predclass = testKaggle_ppredscore1 > 0.5
      2 test_predtags = class2tags(test_predclass, tagnames)
      3 
      4 for i in range(len(test_predtags)):
      5     if test_predtags[i] == "":

NameError: name 'testKaggle_ppredscore1' is not defined

## === cell 15
df1 = pd.DataFrame({"image": [os.path.basename(p) for p in imglist_test]})
df2 = pd.DataFrame({"labels": test_predtags})
submission = pd.concat([df1, df2], axis=1)

sample = pd.read_csv(SAMPLE_SUB)
assert list(sample.columns) == ["image", "labels"]
assert list(submission.columns) == ["image", "labels"]
assert len(submission) == len(
    sample
), f"Submission rows {len(submission)} != sample rows {len(sample)}"

submission.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4293374412.py in <cell line: 0>()
      1 df1 = pd.DataFrame({"image": [os.path.basename(p) for p in imglist_test]})
----> 2 df2 = pd.DataFrame({"labels": test_predtags})
      3 submission = pd.concat([df1, df2], axis=1)
      4 
      5 sample = pd.read_csv(SAMPLE_SUB)

NameError: name 'test_predtags' is not defined

## === cell 16
submission.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3759179968.py in <cell line: 0>()
----> 1 submission.to_csv("./submission.csv", index=False)
      2 print("Wrote ./submission.csv with shape:", submission.shape)
      3 print(submission.head())

NameError: name 'submission' is not defined
