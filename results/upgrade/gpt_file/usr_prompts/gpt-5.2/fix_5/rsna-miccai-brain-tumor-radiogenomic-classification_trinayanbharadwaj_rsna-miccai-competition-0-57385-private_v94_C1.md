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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the runtime import error by avoiding `pydicom` (which is triggering a protobuf incompatibility) and instead reading DICOM pixel data using the already-available `SimpleITK` (this keeps the same core image-loading logic: read slices → resize → normalize → select 6 slices). Then I fix the submission row-count mismatch by ensuring we generate exactly one prediction per test `BraTS21ID` even when some slices fail filtering, by padding missing slice predictions with a default value (0.5) and averaging across the six slices. Finally, I ensure the output `submission.csv` matches `sample_submission.csv` ordering and row count exactly so Kaggle accepts it.'
- What this solution (achieved 0.5) has done: 'I fix the runtime import crash caused by a protobuf incompatibility (triggered when TensorFlow loads and imports protobuf) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I keep your existing SimpleITK-based DICOM loading and the 6-slice averaging/padding logic intact, only adding a safe fallback so the script still produces a valid `submission.csv` even if TensorFlow cannot be imported or the pretrained model is missing. These changes are execution/stability focused and should preserve your current scoring behavior (0.5 baseline) while ensuring the notebook runs end-to-end and writes a valid CSV with the correct row count and ordering.'
- What this solution (achieved 0.5) has done: 'I fix the crash happening at import time by removing the protobuf environment override that is triggering the `MessageFactory.GetPrototype` incompatibility in this Kaggle image, and instead make TensorFlow an optional dependency (so the pipeline still runs even if TF can’t import). I also avoid importing `skimage` (often not available by default) by switching the resize step to `SimpleITK.Resample`, which keeps the same core logic (read slice → resize to 150×150 → normalize → pick 6 slices). Finally, I keep your existing 6-slice averaging/padding and sample-submission alignment so we always write a valid `submission.csv` with the correct rows/ordering. These changes are primarily to restore end-to-end execution; score improve only if TensorFlow can load the provided pretrained model in this environment, otherwise it safely fall back to 0.5.'

# 9. Code solution

## === cell 0
import os

import numpy as np
import pandas as pd

import SimpleITK as sitk

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print(
        "WARNING: TensorFlow import failed; will fall back to constant predictions (0.5)."
    )
    print("TensorFlow import error:", repr(e))

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
if TF_AVAILABLE:
    try:
        tf.random.set_seed(0)
    except Exception:
        pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_PATH = "../input/trained-model-for-rsnamiccai/rsna_miccai_11_epochs_T2W_6k_.h5"
model_T2 = None

if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model_T2 = keras.models.load_model(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        model_T2 = None
        print(
            "WARNING: Failed to load pretrained model (will fall back to constant predictions)."
        )
        print("Model load error:", repr(e))
elif not os.path.exists(MODEL_PATH):
    print(
        f"WARNING: Pretrained model not found at {MODEL_PATH}. "
        "Falling back to constant predictions (0.5) to produce a valid submission."
    )
else:
    print(
        "WARNING: TensorFlow unavailable; cannot load pretrained model. "
        "Falling back to constant predictions (0.5)."
    )




## === cell 2
def _read_dicom_pixel_array_sitk(dcm_path):
    """
    Read a single-slice DICOM and return a 2D numpy array.
    SimpleITK avoids the pydicom dependency.
    """
    img = sitk.ReadImage(dcm_path)
    arr = sitk.GetArrayFromImage(img)  # often shape (1, H, W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _resize_2d_with_sitk(arr2d, out_size=(150, 150)):
    """
    Fix: avoid skimage.transform.resize (may be unavailable). Use SimpleITK resampling.
    Keeps core behavior: resize image to 150x150 while preserving intensity scale.
    """
    arr2d = np.asarray(arr2d)
    if arr2d.ndim != 2:
        raise ValueError("Expected a 2D array for resizing.")
    img = sitk.GetImageFromArray(arr2d.astype(np.float32))
    in_h, in_w = arr2d.shape
    out_h, out_w = out_size

    if in_h <= 0 or in_w <= 0:
        raise ValueError("Invalid input size.")

    scale_x = float(in_w) / float(out_w)
    scale_y = float(in_h) / float(out_h)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetSize([int(out_w), int(out_h)])
    resampler.SetOutputSpacing([scale_x, scale_y])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetDefaultPixelValue(0.0)

    out_img = resampler.Execute(img)
    out_arr = sitk.GetArrayFromImage(out_img)  # (H, W)
    return out_arr.astype(np.float32)


def load_test_T2W_images(path_test):
    IMG_PX_SIZE = 150
    arrays = [[] for _ in range(6)]

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_types) < 4:
            continue

        img_paths = sorted([f.path for f in os.scandir(mri_types[3]) if f.is_file()])

        for img_p in img_paths:
            if count >= 6:
                break
            try:
                px = _read_dicom_pixel_array_sitk(img_p)
            except Exception:
                continue

            if px is None or np.asarray(px).ndim != 2:
                continue

            if float(np.sum(px)) > 100000:
                try:
                    resized_img = _resize_2d_with_sitk(
                        px, out_size=(IMG_PX_SIZE, IMG_PX_SIZE)
                    )
                except Exception:
                    continue

                img = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img,) * 3, axis=-1)

                mx = float(np.max(stacked_img))
                if mx <= 0:
                    continue
                stacked_img_normalize = stacked_img / mx

                if float(np.sum(stacked_img_normalize)) > 2500:
                    arrays[count].append(stacked_img_normalize)
                    count += 1

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of T2 images loaded are ", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)




## === cell 5
def _predict_or_constant(model, x, default=0.5):
    """
    Predict on x (N, H, W, C). Returns length-N vector of probs.
    If model or x is missing, returns an empty vector (caller will pad to match cases).
    """
    if model is None:
        return np.array([], dtype=np.float32)
    if x is None or len(x) == 0:
        return np.array([], dtype=np.float32)

    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1].astype(np.float32)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0].astype(np.float32)
    if preds.ndim == 1:
        return preds.astype(np.float32)
    return np.full((len(x),), default, dtype=np.float32)


def _pad_to_n(pred, n, default=0.5):
    """
    Ensure slice-predictions are exactly n cases (pad with default).
    """
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if pred.size >= n:
        return pred[:n]
    out = np.full((n,), default, dtype=np.float32)
    out[: pred.size] = pred
    return out




## === cell 6
path_cases = sorted([f.path for f in os.scandir(test) if f.is_dir()])
n_cases = len(path_cases)

prediction_1 = _pad_to_n(_predict_or_constant(model_T2, pixels_1), n_cases, default=0.5)
prediction_2 = _pad_to_n(_predict_or_constant(model_T2, pixels_2), n_cases, default=0.5)
prediction_3 = _pad_to_n(_predict_or_constant(model_T2, pixels_3), n_cases, default=0.5)
prediction_4 = _pad_to_n(_predict_or_constant(model_T2, pixels_4), n_cases, default=0.5)
prediction_5 = _pad_to_n(_predict_or_constant(model_T2, pixels_5), n_cases, default=0.5)
prediction_6 = _pad_to_n(_predict_or_constant(model_T2, pixels_6), n_cases, default=0.5)




## === cell 7
def create_sub(path_test, p1, p2, p3, p4, p5, p6):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [os.path.basename(p) for p in path_cases]  # keep as strings

    n = len(cases)
    p1 = np.asarray(p1, dtype=np.float32).reshape(-1)[:n]
    p2 = np.asarray(p2, dtype=np.float32).reshape(-1)[:n]
    p3 = np.asarray(p3, dtype=np.float32).reshape(-1)[:n]
    p4 = np.asarray(p4, dtype=np.float32).reshape(-1)[:n]
    p5 = np.asarray(p5, dtype=np.float32).reshape(-1)[:n]
    p6 = np.asarray(p6, dtype=np.float32).reshape(-1)[:n]

    prediction = (p1 + p2 + p3 + p4 + p5 + p6) / 6.0
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 8
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)

sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
sample_df = pd.read_csv(sample_path, dtype={"BraTS21ID": str})

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str)
merged = sample_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

merged["MGMT_value"] = merged["MGMT_value"].astype(np.float32).fillna(0.5)
sub_df = merged

print("Submission preview:")
print(sub_df.head())
print("Submission shape:", sub_df.shape)



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.dtypes)
print(
    "Min/Max MGMT_value:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)
