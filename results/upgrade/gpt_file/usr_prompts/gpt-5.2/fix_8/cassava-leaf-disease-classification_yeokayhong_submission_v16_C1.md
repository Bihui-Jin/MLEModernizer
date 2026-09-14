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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8644605621033545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.31091) has done: 'You’re not getting a score because the current pipeline is likely failing to run end-to-end in the Kaggle environment due to (a) missing PyTorch/TorchVision packages despite using them, and (b) hard-coded model weight paths that are not present in the provided dataset tree. The minimal fix that preserves your overall “single pretrained model inference over test images” core logic is to switch to a TensorFlow/Keras pretrained classifier (available inside TensorFlow) and keep the same submission schema. I also ensure the submission rows exactly match `sample_submission.csv` order (Kaggle-safe alignment) rather than relying on arbitrary filesystem listing order. This should produce a valid `submission.csv` and yield a reasonable accuracy baseline that can be iterated toward your target.'
- What this solution (achieved 0.45329) has done: 'I fix the immediate runtime crash caused by a TensorFlow/Protobuf incompatibility in this Kaggle environment by switching the core model from `tf.keras.applications` (which triggers the protobuf issue on import) to a lightweight, locally-available TFRecord-based baseline that does not require TensorFlow at all. This preserves the end-to-end logic (read official files → generate predictions for every `image_id` in `sample_submission.csv` → write `submission.csv`), while ensuring the notebook runs reliably under Python 3.13. To move the score up from ~0.31 toward your ~0.864 target (a large gap), I implement a simple but stronger image-classification baseline using scikit-learn (available in Kaggle) with HOG-like gradient features and a linear classifier trained on `train.csv` images. Finally, I guarantee strict submission alignment with `sample_submission.csv` ordering and enforce integer labels 0–4.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog

from joblib import Parallel, delayed

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")

IMG_SIZE = 128
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_cpu = os.cpu_count() or 2
N_JOBS_FEAT = max(1, min(8, _cpu - 1))

print("Python:", sys.version)
print("Data dir exists:", os.path.isdir(DATA_DIR))
print(
    "Train images:", os.path.isdir(TRAIN_DIR), "Test images:", os.path.isdir(TEST_DIR)
)
print(
    "Train CSV exists:",
    os.path.isfile(TRAIN_CSV),
    "Sample sub exists:",
    os.path.isfile(SAMPLE_SUB_PATH),
)

CACHE_DIR = os.path.join("/kaggle/working", "hog_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
print("Cache dir:", CACHE_DIR)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)

num_classes = train_df["label"].nunique()
print(
    "Train rows:",
    len(train_df),
    "Num classes:",
    num_classes,
    "Label counts:\n",
    train_df["label"].value_counts().sort_index(),
)

for fn in train_df["image_id"].head(3):
    p = os.path.join(TRAIN_DIR, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(p)

test_image_ids = sample_sub["image_id"].tolist()
test_paths = (TEST_DIR + "/" + sample_sub["image_id"].values).tolist()

missing = [p for p in test_paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. Example: {missing[0]}"
    )

print("Test rows:", len(test_image_ids))
print("Feature extraction jobs:", N_JOBS_FEAT)



## === cell 2
import hashlib


def _hog_cache_key(image_path: str) -> str:
    st = os.stat(image_path)
    payload = (
        image_path,
        int(st.st_mtime_ns),
        int(st.st_size),
        int(IMG_SIZE),
        int(HOG_PIXELS_PER_CELL[0]),
        int(HOG_PIXELS_PER_CELL[1]),
        int(HOG_CELLS_PER_BLOCK[0]),
        int(HOG_CELLS_PER_BLOCK[1]),
        int(HOG_ORIENTATIONS),
        "L2-Hys",
    )
    return hashlib.md5(repr(payload).encode("utf-8")).hexdigest()


def extract_hog_feature(image_path: str) -> np.ndarray:
    """Read image -> grayscale -> resize -> HOG feature vector."""
    img = imread(image_path)
    if img.ndim == 2:  # already grayscale
        gray = img.astype(np.float32) / 255.0
    else:
        gray = rgb2gray(img).astype(np.float32)

    gray = resize(gray, (IMG_SIZE, IMG_SIZE), anti_aliasing=True).astype(
        np.float32, copy=False
    )

    feat = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32, copy=False)
    return feat


def _atomic_save_npy(final_path: str, arr: np.ndarray) -> None:
    """
    Bugfix: avoid a global thread lock (not picklable by joblib processes).
    Use an atomic temp-write then replace. If two workers race, one replace wins.
    """
    tmp_path = final_path + f".tmp.{os.getpid()}.npy"
    np.save(tmp_path, arr, allow_pickle=False)
    os.replace(tmp_path, final_path)


def extract_hog_feature_cached(image_path: str) -> np.ndarray:
    key = _hog_cache_key(image_path)
    cache_path = os.path.join(CACHE_DIR, f"{key}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path, allow_pickle=False, mmap_mode="r")

    feat = extract_hog_feature(image_path)

    if not os.path.exists(cache_path):
        try:
            _atomic_save_npy(cache_path, feat)
        except FileNotFoundError:
            _atomic_save_npy(cache_path, feat)
        except PermissionError:
            pass
        except OSError:
            pass

    if os.path.exists(cache_path):
        return np.load(cache_path, allow_pickle=False, mmap_mode="r")

    return feat


def build_features_from_paths(paths, n_jobs: int) -> np.ndarray:
    feats = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
        batch_size=64,
        pre_dispatch="2*n_jobs",
        temp_folder="/kaggle/working",
        max_nbytes=None,
    )(delayed(extract_hog_feature_cached)(p) for p in paths)
    return np.vstack(feats)


train_paths = (TRAIN_DIR + "/" + train_df["image_id"].values).tolist()
y = train_df["label"].values

print("Extracting HOG features for train set (parallel CPU)...")
X = build_features_from_paths(train_paths, n_jobs=N_JOBS_FEAT)
print("Train feature matrix:", X.shape, "dtype:", X.dtype)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 688, in wait_result_broken_or_wakeup
    result_item = result_reader.recv()
                  ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/connection.py", line 251, in recv
    return _ForkingPickler.loads(buf.getbuffer())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/_memmapping_reducer.py", line 259, in _strided_from_memmap
    return make_memmap(
           ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/backports.py", line 140, in make_memmap
    mm = np.memmap(
         ^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/core/memmap.py", line 268, in __new__
    mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 24] Too many open files
"""

The above exception was the direct cause of the following exception:

BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_55/3640474287.py in <cell line: 0>()
     97 
     98 print("Extracting HOG features for train set (parallel CPU)...")
---> 99 X = build_features_from_paths(train_paths, n_jobs=N_JOBS_FEAT)
    100 print("Train feature matrix:", X.shape, "dtype:", X.dtype)
    101 

/tmp/ipykernel_55/3640474287.py in build_features_from_paths(paths, n_jobs)
     82 
     83 def build_features_from_paths(paths, n_jobs: int) -> np.ndarray:
---> 84     feats = Parallel(
     85         n_jobs=n_jobs,
     86         prefer="processes",

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

BrokenProcessPool: A result has failed to un-serialize. Please ensure that the objects returned by the function are always picklable.

## === cell 3
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                max_iter=800,
                multi_class="multinomial",
                solver="lbfgs",
                n_jobs=1,
                class_weight="balanced",
                C=2.0,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

print("Fitting classifier (train split for sanity-check)...")
clf.fit(X_tr, y_tr)
va_acc = clf.score(X_va, y_va)
print("Validation accuracy (sanity check):", va_acc)

print("Refitting classifier on full training data for final test inference...")
clf.fit(X, y)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1856034114.py in <cell line: 0>()
      1 X_tr, X_va, y_tr, y_va = train_test_split(
----> 2     X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
      3 )
      4 
      5 clf = Pipeline(

NameError: name 'X' is not defined

## === cell 4
print("Extracting HOG features for test set (parallel CPU)...")
X_test = build_features_from_paths(test_paths, n_jobs=N_JOBS_FEAT)
print("Test feature matrix:", X_test.shape, "dtype:", X_test.dtype)

pred_labels = clf.predict(X_test).astype(int)
pred_labels = np.clip(pred_labels, 0, 4)

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
print("Label distribution:\n", submission_df["label"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 688, in wait_result_broken_or_wakeup
    result_item = result_reader.recv()
                  ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/connection.py", line 251, in recv
    return _ForkingPickler.loads(buf.getbuffer())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/_memmapping_reducer.py", line 259, in _strided_from_memmap
    return make_memmap(
           ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/backports.py", line 140, in make_memmap
    mm = np.memmap(
         ^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/core/memmap.py", line 268, in __new__
    mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 24] Too many open files
"""

The above exception was the direct cause of the following exception:

BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_55/4230274832.py in <cell line: 0>()
      1 print("Extracting HOG features for test set (parallel CPU)...")
----> 2 X_test = build_features_from_paths(test_paths, n_jobs=N_JOBS_FEAT)
      3 print("Test feature matrix:", X_test.shape, "dtype:", X_test.dtype)
      4 
      5 pred_labels = clf.predict(X_test).astype(int)

/tmp/ipykernel_55/3640474287.py in build_features_from_paths(paths, n_jobs)
     82 
     83 def build_features_from_paths(paths, n_jobs: int) -> np.ndarray:
---> 84     feats = Parallel(
     85         n_jobs=n_jobs,
     86         prefer="processes",

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

BrokenProcessPool: A result has failed to un-serialize. Please ensure that the objects returned by the function are always picklable.
