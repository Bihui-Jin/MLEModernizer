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

3.7

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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.8791

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, cv2, concurrent.futures

np.random.seed(42)
cv2.setNumThreads(0)  # disable OpenCV internal threading to avoid oversubscription


def load_image_cv2(path):
    """Fast load with OpenCV; images are already 32×32 RGB‑equivalent,
    so we only read and normalize."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # returns BGR
    if img is None:
        raise ValueError(f"Unable to read image {path}")
    return img.astype(np.float32) / 255.0


def load_images_parallel(id_list, folder):
    """
    Load images in parallel using a process pool (CPU‑bound I/O).
    Returns a NumPy array of shape (n, 32, 32, 3) with float32 pixel values.
    Order of images matches the order of ``id_list``.
    """

    def _load_one(img_id):
        path = os.path.join(folder, img_id)
        return load_image_cv2(path)

    max_workers = min(8, (os.cpu_count() or 1))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        images_iter = executor.map(_load_one, id_list)

    return np.stack(list(images_iter), axis=0)




## === cell 1
train_img_folder = "../input/train/train/"
test_img_folder = "../input/test/test/"

train_csv = pd.read_csv("../input/train.csv", dtype={"id": str, "has_cactus": np.int8})
test_submission = pd.read_csv("../input/sample_submission.csv", dtype={"id": str})

train_ids = train_csv["id"].tolist()
test_ids = test_submission["id"].tolist()

train_imgs = load_images_parallel(train_ids, train_img_folder)
test_imgs = load_images_parallel(test_ids, test_img_folder)

y = train_csv["has_cactus"].values

print(f"Loaded {train_imgs.shape[0]} training images, {test_imgs.shape[0]} test images")
print("Training data shape:", train_imgs.shape, "Labels shape:", y.shape)
print("Test data shape:", test_imgs.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object 'load_images_parallel.<locals>._load_one'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1095845774.py in <cell line: 0>()
      8 test_ids = test_submission["id"].tolist()
      9 
---> 10 train_imgs = load_images_parallel(train_ids, train_img_folder)
     11 test_imgs = load_images_parallel(test_ids, test_img_folder)
     12 

/tmp/ipykernel_55/3711103328.py in load_images_parallel(id_list, folder)
     33 
     34     # Stack results into a single NumPy array (float32).
---> 35     return np.stack(list(images_iter), axis=0)
     36 
     37 

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object 'load_images_parallel.<locals>._load_one'

## === cell 2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

X_flat = train_imgs.reshape((train_imgs.shape[0], -1))
X_test_flat = test_imgs.reshape((test_imgs.shape[0], -1))

X_tr, X_val, y_tr, y_val = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(X_tr, y_tr)

val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.4f}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1778682559.py in <cell line: 0>()
      3 from sklearn.metrics import roc_auc_score
      4 
----> 5 X_flat = train_imgs.reshape((train_imgs.shape[0], -1))
      6 X_test_flat = test_imgs.reshape((test_imgs.shape[0], -1))
      7 

NameError: name 'train_imgs' is not defined

## === cell 3
test_pred = model.predict_proba(X_test_flat)[:, 1]

submission_df = test_submission.copy()
submission_df["has_cactus"] = test_pred
submission_path = "sample_submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1043146377.py in <cell line: 0>()
----> 1 test_pred = model.predict_proba(X_test_flat)[:, 1]
      2 
      3 submission_df = test_submission.copy()
      4 submission_df["has_cactus"] = test_pred
      5 submission_path = "sample_submission.csv"

NameError: name 'model' is not defined

## === cell 4
print(submission_df.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2356174036.py in <cell line: 0>()
----> 1 print(submission_df.head())

NameError: name 'submission_df' is not defined
