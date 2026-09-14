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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.86442

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.65584) has done: 'Outline:
I set the protobuf implementation environment variable before importing TensorFlow to avoid the `MessageFactory` error.  
All `Conv2D` layers use `padding='same'` so the spatial dimensions stay valid after repeated pooling, fixing the negative size error during model building and training.  
The rest of the pipeline remains unchanged, ensuring the script runs end‑to‑end and produces a correctly formatted `submission.csv`.'
- What this solution (achieved 0.56575) has done: 'I added a robust import guard for TensorFlow: if the protobuf‑related error occurs, the script falls back to a pure scikit‑learn solution that trains a separate RandomForest classifier for each label on flattened image pixels. This keeps the original image preprocessing, ensures a valid `submission.csv` is written, and provides a stronger baseline than the previous constant prediction, moving the AUC toward the target. All other logic and file paths remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import random

use_tf = False

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

possible_root = os.path.join(os.getcwd(), "data", "plant-pathology-2020-fgvc7")
if not os.path.isdir(possible_root):
    possible_root = os.path.abspath(
        os.path.join(".", "data", "plant-pathology-2020-fgvc7")
    )
DATA_ROOT = possible_root

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
y_train = train[target_cols].values.astype("float32")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1261102658.py in <cell line: 0>()
     24 test_path = os.path.join(DATA_ROOT, "test.csv")
     25 
---> 26 train = pd.read_csv(train_path)
     27 test = pd.read_csv(test_path)
     28 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/plant-pathology-2020-fgvc7/train.csv'

## === cell 1
img_size = 128  # reasonable size for quick training


def read_img(fname):
    path = os.path.join(DATA_ROOT, "images", fname)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image {path} not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_to_square(img, size=img_size):
    h, w = img.shape[:2]
    if h > w:
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(
            img, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=0
        )
    elif w > h:
        pad = (w - h) // 2
        img = cv2.copyMakeBorder(
            img, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0
        )
    img = cv2.resize(img, (size, size))
    return img.astype("float32") / 255.0


train_imgs = np.zeros((train.shape[0], img_size, img_size, 3), dtype="float32")
for i, fid in enumerate(tqdm.tqdm(train["image_id"], desc="Loading train images")):
    img = read_img(f"{fid}.jpg")
    train_imgs[i] = resize_to_square(img)

test_imgs = np.zeros((test.shape[0], img_size, img_size, 3), dtype="float32")
for i, fid in enumerate(tqdm.tqdm(test["image_id"], desc="Loading test images")):
    img = read_img(f"{fid}.jpg")
    test_imgs[i] = resize_to_square(img)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3131144717.py in <cell line: 0>()
     28 
     29 # Load training images
---> 30 train_imgs = np.zeros((train.shape[0], img_size, img_size, 3), dtype="float32")
     31 for i, fid in enumerate(tqdm.tqdm(train["image_id"], desc="Loading train images")):
     32     img = read_img(f"{fid}.jpg")

NameError: name 'train' is not defined

## === cell 2
if use_tf:
    pass  # TensorFlow path not used in this minimal fix
else:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier

    X_flat = train_imgs.reshape(train_imgs.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    pca = PCA(n_components=200, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    rf_models = []
    for col in range(y_train.shape[1]):
        rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_pca, y_train[:, col])
        rf_models.append(rf)

    X_test_flat = test_imgs.reshape(test_imgs.shape[0], -1)
    X_test_scaled = scaler.transform(X_test_flat)
    X_test_pca = pca.transform(X_test_scaled)

    pred_list = [rf.predict_proba(X_test_pca)[:, 1] for rf in rf_models]
    test_preds = np.stack(pred_list, axis=1)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1216926166.py in <cell line: 0>()
      7 
      8     # Flatten training images
----> 9     X_flat = train_imgs.reshape(train_imgs.shape[0], -1)
     10 
     11     # Standardize

NameError: name 'train_imgs' is not defined

## === cell 3
submission = pd.DataFrame(test_preds, columns=target_cols)
submission.insert(0, "image_id", test["image_id"])
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2761123889.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(test_preds, columns=target_cols)
      2 submission.insert(0, "image_id", test["image_id"])
      3 submission_path = "submission.csv"
      4 submission.to_csv(submission_path, index=False)
      5 print(f"Submission saved to {submission_path}")

NameError: name 'test_preds' is not defined
