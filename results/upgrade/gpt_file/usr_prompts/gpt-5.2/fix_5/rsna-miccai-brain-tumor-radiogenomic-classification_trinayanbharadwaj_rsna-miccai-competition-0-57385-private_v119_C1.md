# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

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

# 5. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
import pandas as pd
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
LABELS_CSV = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))
print("Labels csv exists:", os.path.exists(LABELS_CSV))

try:
    import SimpleITK as sitk  # type: ignore

    _HAS_SITK = True
except Exception:
    sitk = None
    _HAS_SITK = False

print("SimpleITK available:", _HAS_SITK)




## === cell 1
def _list_cases(path_root):
    return sorted([f.path for f in os.scandir(path_root) if f.is_dir()])


def _case_id_from_path(case_path):
    return os.path.basename(case_path)


def _t2w_dir_for_case(case_path):
    subdirs = [f.path for f in os.scandir(case_path) if f.is_dir()]
    if not subdirs:
        return None
    by_name = {os.path.basename(p): p for p in subdirs}
    if "T2w" in by_name:
        return by_name["T2w"]
    return sorted(subdirs)[-1]


def _read_dcm_pixel(path):
    """
    Bugfix: pydicom import crashes (protobuf MessageFactory.GetPrototype).
    Use SimpleITK to read a single DICOM slice robustly. Return float32 2D array or None.
    """
    if not _HAS_SITK:
        return None
    try:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # typically (1, H, W) for single slice
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        return arr.astype(np.float32, copy=False)
    except Exception:
        return None


def _try_get_instance_number(path):
    """
    Score-stability improvement: sort slices by DICOM InstanceNumber when possible.
    Uses SimpleITK metadata without adding new dependencies.
    """
    if not _HAS_SITK:
        return None
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        if reader.HasMetaDataKey("0020|0013"):
            v = reader.GetMetaData("0020|0013")
            return int(str(v).strip())
    except Exception:
        return None
    return None


def _sorted_dicom_paths(dcm_paths):
    inst = []
    for p in dcm_paths:
        n = _try_get_instance_number(p)
        inst.append(n)
    if sum(x is not None for x in inst) >= max(1, len(dcm_paths) // 2):
        keyed = []
        for p, n in zip(dcm_paths, inst):
            keyed.append((1_000_000_000 if n is None else n, os.path.basename(p), p))
        keyed.sort()
        return [k[2] for k in keyed]
    return sorted(dcm_paths)


def load_T2W_images(
    path_root,
    max_slices=6,
    img_px_size=150,
    sum_threshold=100000.0,
    norm_sum_threshold=2000.0,
):
    arrays = [[] for _ in range(max_slices)]
    case_paths = _list_cases(path_root)

    for case_path in case_paths:
        t2w_dir = _t2w_dir_for_case(case_path)
        if t2w_dir is None or (not os.path.isdir(t2w_dir)):
            continue

        img_paths = [
            f.path
            for f in os.scandir(t2w_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
        img_paths = _sorted_dicom_paths(img_paths)

        count = 0
        for p in img_paths:
            px = _read_dcm_pixel(p)
            if px is None:
                continue
            if px.sum() <= sum_threshold:
                continue

            resized_img = resize(
                px, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((resized_img,) * 3, axis=-1)

            mx = np.max(stacked)
            if mx <= 0:
                continue
            stacked_norm = stacked / mx

            if stacked_norm.sum() <= norm_sum_threshold:
                continue

            if count < max_slices:
                arrays[count].append(stacked_norm)
                count += 1
            if count >= max_slices:
                break

    out = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.size > 0:
            a = a / np.max(a)
        out.append(a)

    print("Loaded slices per position:", [len(x) for x in out])
    return out




## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(str).str.zfill(5)

bad_cases = {"00109", "00123", "00709"}
labels = labels[~labels["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

train_case_paths = _list_cases(TRAIN_DIR)
train_ids = [_case_id_from_path(p) for p in train_case_paths]
train_ids_set = set(train_ids)

labels = labels[labels["BraTS21ID"].isin(train_ids_set)].reset_index(drop=True)
print("Train labels after filtering:", labels.shape)




## === cell 3
def build_cnn(input_shape=(150, 150, 3)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def load_case_slices_for_ids(path_root, ids, max_slices=6, img_px_size=150):
    case_paths = [os.path.join(path_root, _id) for _id in ids]
    arrays = [[] for _ in range(max_slices)]
    kept_ids_per_slot = [[] for _ in range(max_slices)]

    for case_path, case_id in zip(case_paths, ids):
        if not os.path.isdir(case_path):
            continue
        t2w_dir = _t2w_dir_for_case(case_path)
        if t2w_dir is None or (not os.path.isdir(t2w_dir)):
            continue

        img_paths = [
            f.path
            for f in os.scandir(t2w_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
        img_paths = _sorted_dicom_paths(img_paths)

        valid = []
        for p in img_paths:
            px = _read_dcm_pixel(p)
            if px is None:
                continue
            if px.sum() <= 100000.0:
                continue
            resized_img = resize(
                px, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked = np.stack((resized_img,) * 3, axis=-1)
            mx = np.max(stacked)
            if mx <= 0:
                continue
            stacked_norm = stacked / mx
            if stacked_norm.sum() <= 2000.0:
                continue
            valid.append(stacked_norm.astype(np.float32))
            if len(valid) >= max_slices:
                break

        for s in range(min(max_slices, len(valid))):
            arrays[s].append(valid[s])
            kept_ids_per_slot[s].append(case_id)

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    for i in range(max_slices):
        if arrays[i].size > 0:
            arrays[i] = arrays[i] / np.max(arrays[i])
    return arrays, kept_ids_per_slot




## === cell 4
all_ids = labels["BraTS21ID"].tolist()
y_map = dict(zip(labels["BraTS21ID"], labels["MGMT_value"].astype(np.float32)))

X_slices, kept_ids = load_case_slices_for_ids(
    TRAIN_DIR, all_ids, max_slices=6, img_px_size=150
)

models = []
for s in range(6):
    X = X_slices[s]
    ids_s = kept_ids[s]
    if len(ids_s) == 0:
        models.append(None)
        continue
    y = np.asarray([y_map[i] for i in ids_s], dtype=np.float32)

    m = build_cnn(input_shape=(150, 150, 3))
    m.fit(X, y, epochs=2, batch_size=16, verbose=0)
    models.append(m)

print("Trained models:", sum(m is not None for m in models), "/ 6")




## === cell 5
test_case_paths = _list_cases(TEST_DIR)
test_ids = [_case_id_from_path(p) for p in test_case_paths]

X_test_slices, kept_test_ids = load_case_slices_for_ids(
    TEST_DIR, test_ids, max_slices=6, img_px_size=150
)

pred_sum = {i: 0.0 for i in test_ids}
pred_cnt = {i: 0 for i in test_ids}

for s in range(6):
    m = models[s]
    X = X_test_slices[s]
    ids_s = kept_test_ids[s]
    if m is None or len(ids_s) == 0:
        continue
    p = m.predict(X, batch_size=32, verbose=0).reshape(-1)
    for _id, _p in zip(ids_s, p):
        pred_sum[_id] += float(_p)
        pred_cnt[_id] += 1

final_preds = []
for _id in test_ids:
    if pred_cnt[_id] > 0:
        final_preds.append(pred_sum[_id] / pred_cnt[_id])
    else:
        final_preds.append(0.5)




## === cell 6
sub_df = pd.DataFrame(
    {
        "BraTS21ID": [str(i).zfill(5) for i in test_ids],
        "MGMT_value": np.clip(np.asarray(final_preds, dtype=np.float32), 0.0, 1.0),
    }
)

sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("MGMT_value stats:", sub_df["MGMT_value"].describe())




## === cell 7
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
