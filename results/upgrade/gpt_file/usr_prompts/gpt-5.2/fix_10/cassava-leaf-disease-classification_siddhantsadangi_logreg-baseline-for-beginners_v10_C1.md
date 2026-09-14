# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")



## === cell 1
df_train = pd.read_csv(r"/kaggle/input/cassava-leaf-disease-classification/train.csv")
df_train.head()



## === cell 2
import json

with open(
    r"/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
) as json_file:
    label_map = json.load(json_file)

label_map = {int(k): v for k, v in label_map.items()}
label_map



## === cell 3
df_train["disease"] = df_train["label"].map(label_map)
df_train



## === cell 4
import glob

train_img_dir = r"/kaggle/input/cassava-leaf-disease-classification/train_images"
sample_paths = [
    os.path.join(train_img_dir, df_train["image_id"].iloc[i])
    for i in (0, len(df_train) // 2, len(df_train) - 1)
]
for p in sample_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected training image not found: {p}")
print("Train images dir OK; example count check skipped for speed.")



## === cell 5
train_img_dir = r"/kaggle/input/cassava-leaf-disease-classification/train_images"
df_train["path"] = train_img_dir + "/" + df_train["image_id"].astype(str)

df_train



## === cell 6
pass



## === cell 7
import matplotlib.pyplot as plt
from PIL import Image



## === cell 8
pass



## === cell 9
pass



## === cell 10
from tqdm import tqdm  # to monitor progress

np.random.seed(42)  # to get reproducible results



## === cell 11
df_samp = pd.concat([df_train.sample(2000, random_state=42)], ignore_index=True)
df_samp.groupby(by="disease").count()



## === cell 12
from sklearn.utils import shuffle

df_samp = shuffle(df_samp, random_state=42).reset_index(drop=True)



## === cell 13
from sklearn.model_selection import train_test_split

X = df_samp.drop(columns=["label"])
y = df_samp["label"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(X_train.shape)
print(len(y_train))
print(X_valid.shape)
print(len(y_valid))



## === cell 14
compressed_size = (200, 150)



## === cell 15
import tensorflow as tf

tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_DIR = r"/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = r"/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

if not train_tfrecs:
    raise FileNotFoundError(f"No train tfrecords found in: {TRAIN_TFREC_DIR}")
if not test_tfrecs:
    raise FileNotFoundError(f"No test tfrecords found in: {TEST_TFREC_DIR}")

_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_resize_flat(image_bytes, size_wh):
    w, h = size_wh
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [h, w], method=tf.image.ResizeMethod.LANCZOS3, antialias=True
    )
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [h * w * 3])
    return img


def _parse_train(rec_bytes):
    ex = tf.io.parse_single_example(rec_bytes, _FEATURE_DESC_TRAIN)
    x = _decode_resize_flat(ex["image"], compressed_size)
    name = ex["image_name"]
    y_ = tf.cast(ex["target"], tf.int64)
    return name, x, y_


def _parse_test(rec_bytes):
    ex = tf.io.parse_single_example(rec_bytes, _FEATURE_DESC_TEST)
    x = _decode_resize_flat(ex["image"], compressed_size)
    name = ex["image_name"]
    return name, x


def _make_ds_train(tfrecs):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE).with_options(opts)
    ds = ds.map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds


def _make_ds_test(tfrecs):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE).with_options(opts)
    ds = ds.map(_parse_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds


train_ids = set(X_train["image_id"].astype(str).tolist())
valid_ids = set(X_valid["image_id"].astype(str).tolist())




## === cell 16
def _collect_split_features_from_tfrecords_single_pass():
    w, h = compressed_size
    d = w * h * 3

    train_id_list = X_train["image_id"].astype(str).tolist()
    valid_id_list = X_valid["image_id"].astype(str).tolist()

    train_index = {img_id: i for i, img_id in enumerate(train_id_list)}
    valid_index = {img_id: i for i, img_id in enumerate(valid_id_list)}

    n_train = len(train_id_list)
    n_valid = len(valid_id_list)

    Xtr = np.empty((n_train, d), dtype=np.float32)
    Xva = np.empty((n_valid, d), dtype=np.float32)
    ytr = np.empty((n_train,), dtype=np.int64)
    yva = np.empty((n_valid,), dtype=np.int64)

    filled_tr = np.zeros(n_train, dtype=np.bool_)
    filled_va = np.zeros(n_valid, dtype=np.bool_)

    ds = _make_ds_train(train_tfrecs).batch(256).prefetch(AUTOTUNE)

    tr_count = 0
    va_count = 0
    total_needed = n_train + n_valid

    for name_b, x_b, y_b in tqdm(
        ds.as_numpy_iterator(), desc="Decode train/valid TFRecords"
    ):
        names = [nb.decode("utf-8") for nb in name_b]

        tr_pos = []
        tr_rows = []
        va_pos = []
        va_rows = []

        for i, img_id in enumerate(names):
            j = train_index.get(img_id)
            if j is not None:
                if not filled_tr[j]:
                    tr_pos.append(i)
                    tr_rows.append(j)
                continue
            j = valid_index.get(img_id)
            if j is not None and (not filled_va[j]):
                va_pos.append(i)
                va_rows.append(j)

        if tr_rows:
            tr_rows = np.asarray(tr_rows, dtype=np.int64)
            tr_pos = np.asarray(tr_pos, dtype=np.int64)
            Xtr[tr_rows] = x_b[tr_pos]
            ytr[tr_rows] = y_b[tr_pos]
            newly = ~filled_tr[tr_rows]
            filled_tr[tr_rows] = True
            tr_count += int(newly.sum())

        if va_rows:
            va_rows = np.asarray(va_rows, dtype=np.int64)
            va_pos = np.asarray(va_pos, dtype=np.int64)
            Xva[va_rows] = x_b[va_pos]
            yva[va_rows] = y_b[va_pos]
            newly = ~filled_va[va_rows]
            filled_va[va_rows] = True
            va_count += int(newly.sum())

        if (tr_count + va_count) >= total_needed:
            break

    if tr_count != n_train or va_count != n_valid:
        miss_tr = int((~filled_tr).sum())
        miss_va = int((~filled_va).sum())
        raise RuntimeError(
            f"Did not collect all requested samples from TFRecords. "
            f"train filled: {tr_count}/{n_train} (missing {miss_tr}), "
            f"valid filled: {va_count}/{n_valid} (missing {miss_va})"
        )

    return Xtr, Xva, ytr, yva


train_array, valid_array, y_train_arr, y_valid_arr = (
    _collect_split_features_from_tfrecords_single_pass()
)

y_train = pd.Series(y_train_arr)
y_valid = pd.Series(y_valid_arr)

print(train_array.shape)
print(valid_array.shape)



## === cell 17
pass



## === cell 18
pass



## === cell 19
print(f"Length of the training array is {len(train_array)}")
print(f"Shape of the training array is {train_array.shape}")
print(
    f"Shape of each training image array is {(compressed_size[1], compressed_size[0], 3)}"
)



## === cell 20
print(f"New length of the training array is {len(train_array)}")
print(f"New shape of the training array is {train_array.shape}")
print(f"New shape of each training image array is {train_array[0].shape}")



## === cell 21
print(f"New length of the validation array is {len(valid_array)}")
print(f"New shape of the validation array is {valid_array.shape}")
print(f"New shape of each validation image array is {valid_array[0].shape}")



## === cell 22
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("scikit-learn-intelex patch applied.")
except Exception as e:
    print("sklearn-intelex patch not applied (fallback to sklearn). Reason:", repr(e))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    class_weight="balanced", verbose=1, n_jobs=-1, solver="saga", max_iter=300
)

train_array = np.ascontiguousarray(train_array, dtype=np.float32)
valid_array = np.ascontiguousarray(valid_array, dtype=np.float32)



## === cell 23
lr.fit(train_array, y_train)



## === cell 24
preds = lr.predict(valid_array)
preds



## === cell 25
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print(classification_report(y_valid, preds))



## === cell 26
pass




## === cell 27
def _collect_test_features_from_tfrecords():
    w, h = compressed_size
    d = w * h * 3

    sample_sub = pd.read_csv(
        r"/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
    test_id_list = sample_sub["image_id"].astype(str).tolist()
    n_test = len(test_id_list)

    test_index = {img_id: i for i, img_id in enumerate(test_id_list)}

    Xte = np.empty((n_test, d), dtype=np.float32)
    names = np.array(test_id_list, dtype=object)
    filled = np.zeros(n_test, dtype=np.bool_)

    ds = _make_ds_test(test_tfrecs).batch(256).prefetch(AUTOTUNE)

    filled_count = 0
    for name_b, x_b in tqdm(ds.as_numpy_iterator(), desc="Decode test TFRecords"):
        batch_names = [nb.decode("utf-8") for nb in name_b]

        pos = []
        rows = []
        for i, img_id in enumerate(batch_names):
            j = test_index.get(img_id)
            if j is None or filled[j]:
                continue
            pos.append(i)
            rows.append(j)

        if rows:
            rows = np.asarray(rows, dtype=np.int64)
            pos = np.asarray(pos, dtype=np.int64)
            Xte[rows] = x_b[pos]
            newly = ~filled[rows]
            filled[rows] = True
            filled_count += int(newly.sum())

        if filled_count >= n_test:
            break

    if filled_count != n_test:
        missing_ids = [test_id_list[i] for i in np.where(~filled)[0][:5].tolist()]
        raise RuntimeError(
            f"Did not collect all test samples from TFRecords: {filled_count}/{n_test}. "
            f"Example missing ids: {missing_ids}"
        )

    return Xte, names, sample_sub


test_array, test_names, sample_sub = _collect_test_features_from_tfrecords()
test_array = np.ascontiguousarray(test_array, dtype=np.float32)



## === cell 28
submission = lr.predict(test_array)
submission[:10]



## === cell 29
pred_by_id = dict(zip(test_names.tolist(), submission.astype(int)))

submission_df = sample_sub.copy()
submission_df["label"] = submission_df["image_id"].map(pred_by_id)

if submission_df["label"].isna().any():
    missing_ids = (
        submission_df.loc[submission_df["label"].isna(), "image_id"].head(5).tolist()
    )
    raise RuntimeError(f"Missing predictions for some test image_id(s): {missing_ids}")

submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", submission_df.shape)
print(submission_df.head())
