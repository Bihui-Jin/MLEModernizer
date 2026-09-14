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
import numpy as np
import pandas as pd

import pydicom as dicom
import cv2

SEED = 42
np.random.seed(SEED)

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    TF_AVAILABLE = True
except Exception as e:
    TF_IMPORT_ERROR = repr(e)
    TF_AVAILABLE = False

print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print(
        "TensorFlow import failed; will use fallback predictions. Error:",
        TF_IMPORT_ERROR,
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Test directory not found: {TEST_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Sample submission not found: {SAMPLE_SUB_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print(sample_sub.head())
print("Sample rows:", len(sample_sub))




## === cell 2
def _safe_resize_to_rgb(img2d: np.ndarray, img_px_size: int = 150) -> np.ndarray:
    """Resize a single-channel image to (img_px_size, img_px_size) and stack to 3 channels."""
    img2d = img2d.astype(np.float32)
    resized = cv2.resize(
        img2d, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA
    )
    mx = float(np.max(resized))
    if mx > 0:
        resized = resized / mx
    stacked = np.stack([resized, resized, resized], axis=-1)  # (H,W,3)
    return stacked


def _case_id_from_path(case_path: str) -> str:
    """Return the folder name (e.g. '00019') from a case path."""
    return os.path.basename(case_path.rstrip("/"))


def load_test_T2W_images(path_test: str, img_px_size: int = 150, max_slices: int = 6):
    """
    Fix: Guarantee exactly one selected image per slice-slot (1..max_slices) per case.
    The original code appended only when it found 'good' slices; some cases yield fewer
    than max_slices, causing length mismatch vs number of cases.
    Here, we select up to max_slices "good" slices; if fewer are found, we pad by repeating
    the last selected slice, or use a zero image if none were found.
    """
    arrays = [[] for _ in range(max_slices)]

    path_cases = sorted(
        [f.path for f in os.scandir(path_test) if f.is_dir()], key=_case_id_from_path
    )

    for case_path in path_cases:
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 4:
            blank = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for k in range(max_slices):
                arrays[k].append(blank)
            continue

        img_dir = mri_type[3]
        img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        chosen = []
        for p in img_paths:
            try:
                ds = dicom.dcmread(p)
                px = ds.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                stacked = _safe_resize_to_rgb(px, img_px_size=img_px_size)
                if stacked.sum() > 2000:
                    chosen.append(stacked)
                    if len(chosen) >= max_slices:
                        break

        if len(chosen) == 0:
            blank = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            chosen = [blank] * max_slices
        elif len(chosen) < max_slices:
            chosen = chosen + [chosen[-1]] * (max_slices - len(chosen))

        for k in range(max_slices):
            arrays[k].append(chosen[k])

    out = []
    for k in range(max_slices):
        x = np.asarray(arrays[k], dtype=np.float32)
        if x.size:
            mx = float(np.max(x))
            if mx > 0:
                x = x / mx
        out.append(x)

    print("Number of T2 images loaded per slice-slot:", [len(a) for a in out])
    return tuple(out)


pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    TEST_DIR, img_px_size=150
)

for i, px in enumerate(
    [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6], start=1
):
    assert px.ndim == 4 and px.shape[1:] == (
        150,
        150,
        3,
    ), f"pixels_{i} shape unexpected: {px.shape}"




## === cell 3
def build_fallback_model(input_shape=(150, 150, 3), seed=42):
    tf.random.set_seed(seed)
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0)(inputs)  # already normalized, but keep stable
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


if TF_AVAILABLE:
    model_T2 = build_fallback_model(seed=SEED + 1)
    model_T2_2 = build_fallback_model(seed=SEED + 2)
    model_T2_3 = build_fallback_model(seed=SEED + 3)
    model_T2_4 = build_fallback_model(seed=SEED + 4)
else:
    model_T2 = model_T2_2 = model_T2_3 = model_T2_4 = None




## === cell 4
def _predict_prob1(model, x):
    """
    Fix: If TF isn't available, return deterministic 0.5 probabilities (valid submission).
    If TF is available, preserve original behavior.
    """
    if x is None or getattr(x, "size", 0) == 0:
        return np.asarray([], dtype=np.float32)

    if not TF_AVAILABLE or model is None:
        return np.full((x.shape[0],), 0.5, dtype=np.float32)

    preds = model.predict(x, verbose=0)
    return preds[:, 1].astype(np.float32)


prediction_1 = _predict_prob1(model_T2, pixels_1)
prediction_2 = _predict_prob1(model_T2, pixels_2)
prediction_3 = _predict_prob1(model_T2, pixels_3)
prediction_4 = _predict_prob1(model_T2, pixels_4)
prediction_5 = _predict_prob1(model_T2, pixels_5)
prediction_6 = _predict_prob1(model_T2, pixels_6)

prediction_101 = _predict_prob1(model_T2_2, pixels_1)
prediction_102 = _predict_prob1(model_T2_2, pixels_2)
prediction_103 = _predict_prob1(model_T2_2, pixels_3)
prediction_104 = _predict_prob1(model_T2_2, pixels_4)
prediction_105 = _predict_prob1(model_T2_2, pixels_5)
prediction_106 = _predict_prob1(model_T2_2, pixels_6)

prediction_201 = _predict_prob1(model_T2_3, pixels_1)
prediction_202 = _predict_prob1(model_T2_3, pixels_2)
prediction_203 = _predict_prob1(model_T2_3, pixels_3)
prediction_204 = _predict_prob1(model_T2_3, pixels_4)
prediction_205 = _predict_prob1(model_T2_3, pixels_5)
prediction_206 = _predict_prob1(model_T2_3, pixels_6)

prediction_301 = _predict_prob1(model_T2_4, pixels_1)
prediction_302 = _predict_prob1(model_T2_4, pixels_2)
prediction_303 = _predict_prob1(model_T2_4, pixels_3)
prediction_304 = _predict_prob1(model_T2_4, pixels_4)
prediction_305 = _predict_prob1(model_T2_4, pixels_5)
prediction_306 = _predict_prob1(model_T2_4, pixels_6)

print(
    "Pred lengths:",
    [
        len(prediction_1),
        len(prediction_2),
        len(prediction_3),
        len(prediction_4),
        len(prediction_5),
        len(prediction_6),
    ],
)




## === cell 5
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
):
    path_cases = sorted(
        [f.path for f in os.scandir(path_test) if f.is_dir()], key=_case_id_from_path
    )
    n = len(path_cases)

    pred_arrays = [
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p101,
        p102,
        p103,
        p104,
        p105,
        p106,
        p201,
        p202,
        p203,
        p204,
        p205,
        p206,
        p301,
        p302,
        p303,
        p304,
        p305,
        p306,
    ]
    for idx, arr in enumerate(pred_arrays):
        if len(arr) != n:
            raise ValueError(
                f"Prediction array {idx} length {len(arr)} does not match number of cases {n}."
            )

    cases, preds = [], []
    for i, case_path in enumerate(path_cases):
        case_number = _case_id_from_path(case_path)  # '00019'
        cases.append(int(case_number))

        pred_i = (
            float(p1[i])
            + float(p2[i])
            + float(p3[i])
            + float(p4[i])
            + float(p5[i])
            + float(p6[i])
            + float(p101[i])
            + float(p102[i])
            + float(p103[i])
            + float(p104[i])
            + float(p105[i])
            + float(p106[i])
            + float(p201[i])
            + float(p202[i])
            + float(p203[i])
            + float(p204[i])
            + float(p205[i])
            + float(p206[i])
            + float(p301[i])
            + float(p302[i])
            + float(p303[i])
            + float(p304[i])
            + float(p305[i])
            + float(p306[i])
        ) / 24.0
        preds.append(pred_i)

    df = pd.DataFrame(
        {"BraTS21ID": cases, "MGMT_value": np.asarray(preds, dtype=np.float32)}
    )
    return df




## === cell 6
sub_df = create_sub(
    TEST_DIR,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sample_sub_ids = sample_sub.copy()
sample_sub_ids["BraTS21ID"] = sample_sub_ids["BraTS21ID"].astype(int)

sub_df = sample_sub_ids[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(sample_sub)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1334146076.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     TEST_DIR,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/2346823466.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p301, p302, p303, p304, p305, p306)
     67     for i, case_path in enumerate(path_cases):
     68         case_number = _case_id_from_path(case_path)  # '00019'
---> 69         cases.append(int(case_number))
     70 
     71         pred_i = (

ValueError: invalid literal for int() with base 10: 'test'

## === cell 7
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
print("submission.csv exists:", os.path.isfile("submission.csv"))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3340605069.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 
      3 print("Wrote submission.csv with shape:", sub_df.shape)
      4 print(sub_df.head())
      5 print("submission.csv exists:", os.path.isfile("submission.csv"))

NameError: name 'sub_df' is not defined
