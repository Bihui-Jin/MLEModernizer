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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.43059) has done: 'I fix the runtime failures by removing the TensorFlow import that triggers the protobuf `MessageFactory.GetPrototype` crash and by replacing the missing `tf.io.decode_dicom_image` call with a stable DICOM reader based on `pydicom` (available in the Kaggle RSNA environment). I keep your core approach identical: read T1wCE slices, train the same small CNN on per-slice labels, predict per-slice probabilities, then average per patient to form `MGMT_value`. I also add small guards so unreadable/corrupt DICOMs are skipped (rather than crashing), ensuring an end-to-end run and a valid `submission.csv`. No score-tuning changes are introduced beyond making the pipeline actually execute and produce probabilistic outputs for ROC-AUC.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import pydicom
from pydicom.pixels import pixel_array as _pydicom_pixel_array

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

IMAGE_SIZE = 224


_RESIZE_CACHE = {}


def _resize_nn(img2d: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize for 2D arrays (keeps dependencies minimal and fast)."""
    h, w = img2d.shape
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=img2d.dtype)
    key = (h, w, size)
    idx = _RESIZE_CACHE.get(key)
    if idx is None:
        ys = (np.linspace(0, h - 1, size)).astype(np.int32)
        xs = (np.linspace(0, w - 1, size)).astype(np.int32)
        idx = (ys, xs)
        _RESIZE_CACHE[key] = idx
    ys, xs = idx
    return img2d[ys[:, None], xs[None, :]]


def load_dicom_np(path, size=224):
    """
    pydicom-based reader.
    Returns uint8 image (H,W) scaled to [0,255].
    """
    try:
        dcm = pydicom.dcmread(path, force=True, stop_before_pixels=True)
        img = _pydicom_pixel_array(dcm).astype(np.float32)

        if img.ndim == 3:
            img = img[0]
        if img.ndim != 2:
            return None

        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        lo = float(np.min(img))
        hi = float(np.max(img))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None

        img = (img - lo) / (hi - lo)
        img = np.clip(img, 0.0, 1.0)

        img = _resize_nn(img, size=size)
        img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
        return img_u8
    except Exception:
        return None




## === cell 2
from concurrent.futures import ThreadPoolExecutor

_PATHS_CACHE = {}


def _slice_num_from_name(p: str) -> int:
    b = os.path.basename(p)
    dash = b.rfind("-")
    dot = b.rfind(".")
    if dash == -1:
        dash = 0
    if dot == -1:
        dot = len(b)
    try:
        return int(b[dash + 1 : dot])
    except Exception:
        return 0


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns a subset of slice paths for a patient and modality.
    Keeps the original "middle 50% if enough slices" logic.
    """
    assert image_type in TYPES

    pid = str(int(brats21id)).zfill(5)
    cache_key = (folder, pid, image_type)
    paths = _PATHS_CACHE.get(cache_key)
    if paths is None:
        patient_path = os.path.join(DATA_ROOT, folder, pid)
        g = glob.glob(os.path.join(patient_path, image_type, "*"))
        if not g:
            paths = []
        else:
            paths = sorted(g, key=_slice_num_from_name)
        _PATHS_CACHE[cache_key] = paths

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    if num_images > 10:
        start = int(num_images * 0.25)
        end = int(num_images * 0.75)
    else:
        start = 0
        end = num_images

    interval = 1
    return np.array(paths[start:end:interval], dtype=object)


def get_all_images(brats21id, image_type, folder="train", size=224, max_workers=None):
    paths = get_all_image_paths(brats21id, image_type, folder)
    if len(paths) == 0:
        return []
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 2))

    imgs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        loaded = list(ex.map(lambda p: load_dicom_np(p, size=size), paths))
    for im in loaded:
        if im is not None:
            imgs.append(im)
    return imgs


def get_all_data_for_train(image_type):
    X, y, train_ids = [], [], []
    ids = train_df["BraTS21ID"].to_numpy()
    labels = train_df["MGMT_value"].to_numpy()
    for brats_id, label in zip(ids, labels):
        brats_id = int(brats_id)
        label = float(label)
        images = get_all_images(brats_id, image_type, folder="train", size=IMAGE_SIZE)

        if len(images) == 0:
            continue

        X += images
        y += [label] * len(images)
        train_ids += [brats_id] * len(images)
    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type):
    X, test_ids = [], []
    ids = test_df["BraTS21ID"].to_numpy()
    for brats_id in ids:
        brats_id = int(brats_id)
        images = get_all_images(brats_id, image_type, folder="test", size=IMAGE_SIZE)

        if len(images) == 0:
            continue

        X += images
        test_ids += [brats_id] * len(images)

    return np.array(X), np.array(test_ids)


X_train, y_train, train_ids = get_all_data_for_train("T1wCE")
X_test, test_ids = get_all_data_for_test("T1wCE")

X_train = (X_train.astype(np.float32, copy=False) / 255.0).reshape(len(X_train), -1)
X_test = (X_test.astype(np.float32, copy=False) / 255.0).reshape(len(X_test), -1)

print("Train slices:", X_train.shape, "Labels:", y_train.shape)
print("Test slices:", X_test.shape, "Test ids:", test_ids.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/471848058.py in <cell line: 0>()
    114 
    115 # Speed: avoid an extra large temporary by converting/reshaping in one pass with astype(copy=False).
--> 116 X_train = (X_train.astype(np.float32, copy=False) / 255.0).reshape(len(X_train), -1)
    117 X_test = (X_test.astype(np.float32, copy=False) / 255.0).reshape(len(X_test), -1)
    118 

ValueError: cannot reshape array of size 0 into shape (0,newaxis)

## === cell 3
idx = np.arange(len(X_train))
np.random.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

model_best = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=2,
    random_state=SEED,
    verbose=True,
)

model_best.fit(X_train[tr_idx], y_train[tr_idx])

try:
    from sklearn.metrics import roc_auc_score

    va_pred = model_best.predict_proba(X_train[va_idx])[:, 1]
    auc = roc_auc_score(y_train[va_idx], va_pred)
    print("Validation ROC-AUC (slice-level):", float(auc))
except Exception as e:
    print("Validation AUC not computed:", repr(e))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3009929043.py in <cell line: 0>()
     16 )
     17 
---> 18 model_best.fit(X_train[tr_idx], y_train[tr_idx])
     19 
     20 try:

/usr/local/lib/python3.11/dist-packages/sklearn/neural_network/_multilayer_perceptron.py in fit(self, X, y)
    747         self._validate_params()
    748 
--> 749         return self._fit(X, y, incremental=False)
    750 
    751     def _check_solver(self):

/usr/local/lib/python3.11/dist-packages/sklearn/neural_network/_multilayer_perceptron.py in _fit(self, X, y, incremental)
    435         )
    436 
--> 437         X, y = self._validate_input(X, y, incremental, reset=first_pass)
    438 
    439         n_samples, n_features = X.shape

/usr/local/lib/python3.11/dist-packages/sklearn/neural_network/_multilayer_perceptron.py in _validate_input(self, X, y, incremental, reset)
   1087 
   1088     def _validate_input(self, X, y, incremental, reset):
-> 1089         X, y = self._validate_data(
   1090             X,
   1091             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    900             # If input is 1D raise error
    901             if array.ndim == 1:
--> 902                 raise ValueError(
    903                     "Expected 2D array, got 1D array instead:\narray={}.\n"
    904                     "Reshape your data either using array.reshape(-1, 1) if "

ValueError: Expected 2D array, got 1D array instead:
array=[].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

## === cell 4
y_pred = model_best.predict_proba(X_test)[:, 1].reshape(-1)

result = pd.DataFrame(
    {"BraTS21ID": test_ids.astype(int), "MGMT_value": y_pred.astype(float)}
)
result2 = result.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

result2 = sample[["BraTS21ID"]].merge(result2, on="BraTS21ID", how="left")

fill_value = (
    float(np.nanmean(result2["MGMT_value"].values))
    if result2["MGMT_value"].notna().any()
    else 0.5
)
result2["MGMT_value"] = result2["MGMT_value"].fillna(fill_value).clip(0.0, 1.0)


result2 = result2[["BraTS21ID", "MGMT_value"]]
result2.to_csv("submission.csv", index=False)
print(result2.head())
print("Wrote submission.csv with shape:", result2.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/408882383.py in <cell line: 0>()
----> 1 y_pred = model_best.predict_proba(X_test)[:, 1].reshape(-1)
      2 
      3 result = pd.DataFrame(
      4     {"BraTS21ID": test_ids.astype(int), "MGMT_value": y_pred.astype(float)}
      5 )

/usr/local/lib/python3.11/dist-packages/sklearn/neural_network/_multilayer_perceptron.py in predict_proba(self, X)
   1239             model, where classes are ordered as they are in `self.classes_`.
   1240         """
-> 1241         check_is_fitted(self)
   1242         y_pred = self._forward_pass_fast(X)
   1243 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This MLPClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
