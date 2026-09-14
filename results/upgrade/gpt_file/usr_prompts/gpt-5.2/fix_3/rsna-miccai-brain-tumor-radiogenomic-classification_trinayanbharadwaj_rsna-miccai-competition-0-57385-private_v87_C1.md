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

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import layers

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf.random.set_seed(0)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

BAD_CASES = set([109, 123, 709])
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

labels_df.head(), sample_sub.head()




## === cell 2
def _sorted_case_dirs(root_dir):
    dirs = []
    for d in os.scandir(root_dir):
        if d.is_dir() and d.name.isdigit():
            dirs.append(d.name)
    dirs = sorted(dirs)
    return [os.path.join(root_dir, d) for d in dirs]


def _read_dicom_cv2(dcm_path):
    """
    Read a DICOM into a float32 numpy array using OpenCV.
    OpenCV builds on Kaggle commonly include GDCM, so this works without pydicom.
    """
    img = cv2.imread(dcm_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cv2 could not read DICOM: {dcm_path}")
    img = img.astype(np.float32)
    return img


def _resize_to_rgb(img2d, img_px_size=150):
    img2d = cv2.resize(img2d, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA)
    mx = float(np.max(img2d))
    if mx > 0:
        img2d = img2d / mx
    else:
        img2d = img2d * 0.0
    img3 = np.stack([img2d, img2d, img2d], axis=-1).astype(np.float32)
    return img3


def _case_t2w_dcm_paths(case_dir):
    t2_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2_dir):
        candidates = [
            p.path
            for p in os.scandir(case_dir)
            if p.is_dir() and ("t2" in p.name.lower())
        ]
        if len(candidates) == 0:
            return []
        t2_dir = sorted(candidates)[0]
    dcm_paths = [
        p.path
        for p in os.scandir(t2_dir)
        if p.is_file() and p.name.lower().endswith(".dcm")
    ]
    return sorted(dcm_paths)




## === cell 3
def load_T2W_images_fixed(
    path_root,
    max_slices=15,
    img_px_size=150,
    pixel_sum_thresh=100000,
    norm_sum_thresh=2500,
):
    """
    Returns a list of length max_slices, each element is a float32 array (N, H, W, 3).
    Deterministic: uses sorted directories and sorted dicom filenames.
    """
    case_dirs = _sorted_case_dirs(path_root)
    slice_buckets = [[] for _ in range(max_slices)]

    for case_dir in case_dirs:
        count = 0
        dcm_paths = _case_t2w_dcm_paths(case_dir)

        for dcm_path in dcm_paths:
            if count >= max_slices:
                break
            try:
                img2d = _read_dicom_cv2(dcm_path)
            except Exception:
                continue

            if float(np.sum(img2d)) <= pixel_sum_thresh:
                continue

            img3 = _resize_to_rgb(img2d, img_px_size=img_px_size)

            if float(np.sum(img3)) <= norm_sum_thresh:
                continue

            slice_buckets[count].append(img3)
            count += 1

        if count == 0:
            pad_img = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for j in range(max_slices):
                slice_buckets[j].append(pad_img)
        else:
            last_img = slice_buckets[count - 1][-1]
            for j in range(count, max_slices):
                slice_buckets[j].append(last_img)

    arrays = [np.stack(bucket, axis=0).astype(np.float32) for bucket in slice_buckets]
    print("Number of T2 images loaded are ", ", ".join(str(a.shape[0]) for a in arrays))
    return arrays




## === cell 4
pixels_list_test = load_T2W_images_fixed(TEST_PATH, max_slices=15, img_px_size=150)
[p.shape for p in pixels_list_test[:3]]




## === cell 5
def build_slice_model(input_shape=(150, 150, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


model_T2 = build_slice_model()




## === cell 6
def make_training_slice_dataset(
    train_root, labels_df, max_slices=15, img_px_size=150, cases_limit=120
):
    id_to_label = dict(
        zip(
            labels_df["BraTS21ID"].astype(int).tolist(),
            labels_df["MGMT_value"].astype(int).tolist(),
        )
    )

    available_dirs = {
        int(os.path.basename(p)): p for p in _sorted_case_dirs(train_root)
    }

    ids = [i for i in sorted(id_to_label.keys()) if i in available_dirs]
    ids = ids[:cases_limit]

    X = []
    y = []
    for brats_id in ids:
        case_dir = available_dirs[brats_id]
        dcm_paths = _case_t2w_dcm_paths(case_dir)
        count = 0
        for dcm_path in dcm_paths:
            if count >= max_slices:
                break
            try:
                img2d = _read_dicom_cv2(dcm_path)
            except Exception:
                continue
            if float(np.sum(img2d)) <= 100000:
                continue
            img3 = _resize_to_rgb(img2d, img_px_size=img_px_size)
            if float(np.sum(img3)) <= 2500:
                continue
            X.append(img3)
            y.append(id_to_label[brats_id])
            count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


X_train, y_train = make_training_slice_dataset(
    TRAIN_PATH, labels_df, max_slices=15, img_px_size=150, cases_limit=120
)
X_train.shape, y_train.mean() if len(y_train) else None



## === cell 7
if len(y_train) == 0:
    raise RuntimeError("No training slices could be loaded; cannot proceed.")

history = model_T2.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.2,
    shuffle=True,
    verbose=2,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1829831639.py in <cell line: 0>()
      1 if len(y_train) == 0:
----> 2     raise RuntimeError("No training slices could be loaded; cannot proceed.")
      3 
      4 history = model_T2.fit(
      5     X_train,

RuntimeError: No training slices could be loaded; cannot proceed.

## === cell 8
predictions_per_slice = []
for pixels in pixels_list_test:
    preds = model_T2.predict(pixels, batch_size=16, verbose=0)
    predictions_per_slice.append(preds[:, 1].astype(np.float32))

pred_matrix = np.stack(predictions_per_slice, axis=0)
avg_pred = pred_matrix.mean(axis=0)

avg_pred.shape, float(avg_pred.min()), float(avg_pred.max())




## === cell 9
def create_sub_fixed(path_test, avg_prediction):
    case_dirs = _sorted_case_dirs(path_test)
    cases = [int(os.path.basename(p)) for p in case_dirs]
    if len(cases) != len(avg_prediction):
        raise ValueError(
            f"Mismatch: {len(cases)} test cases but {len(avg_prediction)} predictions"
        )

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": avg_prediction.astype(float)})
    df["BraTS21ID"] = df["BraTS21ID"].map(lambda x: f"{int(x):05d}")
    return df


sub_df = create_sub_fixed(TEST_PATH, avg_pred)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/560107959.py in <cell line: 0>()
     15 
     16 # Align ordering to sample_submission just in case
---> 17 sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
     18 sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)
     19 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    805         # validate the merge keys dtypes. We may need to coerce
    806         # to avoid incompatible dtypes
--> 807         self._maybe_coerce_merge_keys()
    808 
    809         # If argument passed to validate,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _maybe_coerce_merge_keys(self)
   1506                     inferred_right in string_types and inferred_left not in string_types
   1507                 ):
-> 1508                     raise ValueError(msg)
   1509 
   1510             # datetimelikes must match exactly

ValueError: You are trying to merge on int64 and object columns for key 'BraTS21ID'. If you wish to proceed you should use pd.concat
