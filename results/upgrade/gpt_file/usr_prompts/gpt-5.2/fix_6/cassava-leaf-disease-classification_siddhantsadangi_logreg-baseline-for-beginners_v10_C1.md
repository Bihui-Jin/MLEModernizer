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
import os
from concurrent.futures import ThreadPoolExecutor

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def _load_one_resize_rgb_uint8_cv2(args):
    i, p, size = args
    w, h = size
    im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    im_bgr = cv2.resize(im_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return i, im_rgb


def _load_one_resize_rgb_uint8_pil(args):
    i, p, size, resample = args
    with Image.open(p) as im:
        im = im.convert("RGB").resize(size, resample)
        arr = np.asarray(im, dtype=np.uint8)
    return i, arr


_GLOBAL_EXECUTOR = None


def _get_executor():
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, max(4, cpu))
        _GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers)
    return _GLOBAL_EXECUTOR


def load_resize_rgb_to_array(paths, size, resample):
    paths = list(paths)
    n = len(paths)
    w, h = size
    out = np.empty((n, h, w, 3), dtype=np.uint8)

    if _HAS_CV2:
        worker = _load_one_resize_rgb_uint8_cv2
        gen = ((i, paths[i], size) for i in range(n))
    else:
        worker = _load_one_resize_rgb_uint8_pil
        gen = ((i, paths[i], size, resample) for i in range(n))

    chunksize = 256 if n >= 2048 else (128 if n >= 1024 else 32)

    ex = _get_executor()
    it = ex.map(worker, gen, chunksize=chunksize)
    for i, arr in tqdm(it, total=n):
        out[i] = arr
    return out


train_array = load_resize_rgb_to_array(X_train.path.values, compressed_size, RESAMPLE)
valid_array = load_resize_rgb_to_array(X_valid.path.values, compressed_size, RESAMPLE)

print(train_array.shape)
print(valid_array.shape)



## === cell 16
pass



## === cell 17
pass



## === cell 18
print(f"Length of the training array is {len(train_array)}")
print(f"Shape of the training array is {train_array.shape}")
print(f"Shape of each training image array is {train_array[0].shape}")



## === cell 19
train_array = np.ascontiguousarray(train_array).reshape(len(train_array), -1)

print(f"New length of the training array is {len(train_array)}")
print(f"New shape of the training array is {train_array.shape}")
print(f"New shape of each training image array is {train_array[0].shape}")



## === cell 20
valid_array = np.ascontiguousarray(valid_array).reshape(len(valid_array), -1)

print(f"New length of the validation array is {len(valid_array)}")
print(f"New shape of the validation array is {valid_array.shape}")
print(f"New shape of each validation image array is {valid_array[0].shape}")



## === cell 21
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



## === cell 22
lr.fit(train_array, y_train)



## === cell 23
preds = lr.predict(valid_array)
preds



## === cell 24
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print(classification_report(y_valid, preds))



## === cell 25
test_img_dir = r"/kaggle/input/cassava-leaf-disease-classification/test_images"
test_path = [
    e.path for e in os.scandir(test_img_dir) if e.is_file() and e.name.endswith(".jpg")
]
test_path.sort()
print(len(test_path))
test_path[:3]



## === cell 26
pass



## === cell 27
test_array = load_resize_rgb_to_array(
    np.array(test_path, dtype=object), compressed_size, RESAMPLE
)
test_array = np.ascontiguousarray(test_array).reshape(len(test_array), -1)
test_array = np.ascontiguousarray(test_array, dtype=np.float32)



## === cell 28
submission = lr.predict(test_array)
submission[:10]



## === cell 29
sample_sub = pd.read_csv(
    r"/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

pred_by_id = dict(zip((os.path.basename(p) for p in test_path), submission.astype(int)))

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
