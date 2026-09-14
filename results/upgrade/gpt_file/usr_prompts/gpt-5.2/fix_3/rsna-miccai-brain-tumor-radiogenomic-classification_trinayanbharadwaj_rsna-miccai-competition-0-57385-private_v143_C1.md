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

- What this solution (achieved 0.5) has done: 'I fix the import/runtime crash by removing the protobuf-triggering `pympler` import (it’s unused) and by making the `resize` dependency explicit inside the image loader so it can’t be missing at runtime. Because the referenced pre-trained `.h5` models are not available in this environment, I keep the exact “ensemble-averaging” submission semantics but fall back to a deterministic baseline predictor (global mean of the training labels) so the notebook always produces a valid `submission.csv`. I also fix logic bugs in `create_sub` (it was overwriting `prediction` inside a loop and producing mismatched lengths) by computing predictions aligned 1:1 with sorted test case IDs. Finally, I enforce the required submission format (5-digit `BraTS21ID` strings) and ensure the file is written with a `.csv` suffix.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash causing the protobuf `MessageFactory.GetPrototype` error by removing heavy/unused visualization imports and using `tensorflow.keras` consistently (this avoids the known standalone-keras/protobuf incompatibility in some Kaggle images). I keep your exact inference/submission semantics (use provided `.h5` models if present; otherwise fall back to the deterministic baseline mean label), but make the environment robust by optionalizing plotting and unused imports. I also keep the existing submission formatting/alignment safeguards so it always writes a valid `submission.csv` with the required columns and 5-digit IDs. These changes are score-neutral except that they allow the code to run end-to-end reliably (your current 0.5 baseline remains).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # keep API available; core logic unchanged

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

np.random.seed(42)
tf.random.set_seed(42)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

MODEL_DIR_CANDIDATES = [
    "../input/trained-model-for-rsnamiccai",
    "../input/trained-model-for-rsnamiccai/",
]


def _existing_model_paths():
    paths = []
    for d in MODEL_DIR_CANDIDATES:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".h5", ".keras")):
                    paths.append(os.path.join(d, fn))
    return sorted(paths)


existing_models = _existing_model_paths()
print(f"Found {len(existing_models)} model files in provided model directories.")
if existing_models:
    print("Example model file:", existing_models[0])



## === cell 2
model_T2 = None
model_T2_2 = None
model_T2_3 = None
model_T2_4 = None

model_paths = {
    "model_T2": "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "model_T2_2": "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "model_T2_3": "../input/trained-model-for-rsnamiccai/rsna_miccai_83_b600_T2W_7k_imgs.h5",
    "model_T2_4": "../input/trained-model-for-rsnamiccai/rsna_miccai_28_b50_T2W_7k_imgs.h5",
}


def _safe_load_model(path):
    if os.path.exists(path):
        return keras.models.load_model(path)
    return None


model_T2 = _safe_load_model(model_paths["model_T2"])
model_T2_2 = _safe_load_model(model_paths["model_T2_2"])
model_T2_3 = _safe_load_model(model_paths["model_T2_3"])
model_T2_4 = _safe_load_model(model_paths["model_T2_4"])

loaded = [m is not None for m in [model_T2, model_T2_2, model_T2_3, model_T2_4]]
print("Models loaded:", loaded)




## === cell 3
def load_test_T2W_images(path_test):
    """
    Original core idea preserved: load up to 7 T2 slices per case after basic filtering and resizing to 150x150x3.

    Bugfixes / robustness:
      - Ensure resize is imported/available at runtime.
      - Convert lists to numpy arrays before normalization.
      - Guard against empty arrays to prevent np.max(empty) crash.
      - Use dtype float32 for TF predict compatibility.
    """
    try:
        from skimage.transform import resize as sk_resize
    except Exception as e:
        raise ImportError(
            "skimage is required for resize but could not be imported."
        ) from e

    arrays = [[] for _ in range(7)]
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 4:
            continue
        img_folder = mri_type[
            3
        ]  # preserve original logic: use the 4th folder (typically T2w)
        img_paths = sorted([f.path for f in os.scandir(img_folder) if f.is_file()])

        for p in img_paths:
            if count >= 7:
                break
            try:
                img = dicom.dcmread(p)
                px = img.pixel_array
            except Exception:
                continue

            if px is None:
                continue

            if px.sum() > 100000:
                resized_img = sk_resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img2 = np.array(resized_img, dtype=np.float32)
                stacked = np.stack((img2,) * 3, axis=-1)
                mx = float(np.max(stacked)) if stacked.size else 0.0
                if mx > 0:
                    stacked_norm = stacked / mx
                else:
                    continue
                if stacked_norm.sum() > 2000:
                    arrays[count].append(stacked_norm)
                    count += 1

    out = []
    for a in arrays:
        a_np = np.array(a, dtype=np.float32)
        if a_np.size == 0:
            out.append(a_np)
            continue
        m = float(np.max(a_np))
        if m > 0:
            a_np = a_np / m
        out.append(a_np)

    print("Number of T2 images loaded are ", ", ".join(str(len(x)) for x in out))
    return tuple(out)




## === cell 4
have_all_models = all(
    m is not None for m in [model_T2, model_T2_2, model_T2_3, model_T2_4]
)

if have_all_models:
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = (
        load_test_T2W_images(TEST_DIR)
    )
else:
    pixels_1 = pixels_2 = pixels_3 = pixels_4 = pixels_5 = pixels_6 = pixels_7 = None
    print(
        "Skipping DICOM loading because required model files were not found; will use baseline predictions."
    )



## === cell 5
if have_all_models:
    preds_1 = model_T2.predict(pixels_1, verbose=0)
    prediction_1 = preds_1[:, 1]
    preds_2 = model_T2.predict(pixels_2, verbose=0)
    prediction_2 = preds_2[:, 1]
    preds_3 = model_T2.predict(pixels_3, verbose=0)
    prediction_3 = preds_3[:, 1]
    preds_4 = model_T2.predict(pixels_4, verbose=0)
    prediction_4 = preds_4[:, 1]
    preds_5 = model_T2.predict(pixels_5, verbose=0)
    prediction_5 = preds_5[:, 1]
    preds_6 = model_T2.predict(pixels_6, verbose=0)
    prediction_6 = preds_6[:, 1]
    preds_7 = model_T2.predict(pixels_7, verbose=0)
    prediction_7 = preds_7[:, 1]

    preds_101 = model_T2_2.predict(pixels_1, verbose=0)
    prediction_101 = preds_101[:, 1]
    preds_102 = model_T2_2.predict(pixels_2, verbose=0)
    prediction_102 = preds_102[:, 1]
    preds_103 = model_T2_2.predict(pixels_3, verbose=0)
    prediction_103 = preds_103[:, 1]
    preds_104 = model_T2_2.predict(pixels_4, verbose=0)
    prediction_104 = preds_104[:, 1]
    preds_105 = model_T2_2.predict(pixels_5, verbose=0)
    prediction_105 = preds_105[:, 1]
    preds_106 = model_T2_2.predict(pixels_6, verbose=0)
    prediction_106 = preds_106[:, 1]
    preds_107 = model_T2_2.predict(pixels_7, verbose=0)
    prediction_107 = preds_107[:, 1]

    preds_201 = model_T2_3.predict(pixels_1, verbose=0)
    prediction_201 = preds_201[:, 1]
    preds_202 = model_T2_3.predict(pixels_2, verbose=0)
    prediction_202 = preds_202[:, 1]
    preds_203 = model_T2_3.predict(pixels_3, verbose=0)
    prediction_203 = preds_203[:, 1]
    preds_204 = model_T2_3.predict(pixels_4, verbose=0)
    prediction_204 = preds_204[:, 1]
    preds_205 = model_T2_3.predict(pixels_5, verbose=0)
    prediction_205 = preds_205[:, 1]
    preds_206 = model_T2_3.predict(pixels_6, verbose=0)
    prediction_206 = preds_206[:, 1]
    preds_207 = model_T2_3.predict(pixels_7, verbose=0)
    prediction_207 = preds_207[:, 1]

    preds_301 = model_T2_4.predict(pixels_1, verbose=0)
    prediction_301 = preds_301[:, 1]
    preds_302 = model_T2_4.predict(pixels_2, verbose=0)
    prediction_302 = preds_302[:, 1]
    preds_303 = model_T2_4.predict(pixels_3, verbose=0)
    prediction_303 = preds_303[:, 1]
    preds_304 = model_T2_4.predict(pixels_4, verbose=0)
    prediction_304 = preds_304[:, 1]
    preds_305 = model_T2_4.predict(pixels_5, verbose=0)
    prediction_305 = preds_305[:, 1]
    preds_306 = model_T2_4.predict(pixels_6, verbose=0)
    prediction_306 = preds_306[:, 1]
    preds_307 = model_T2_4.predict(pixels_7, verbose=0)
    prediction_307 = preds_307[:, 1]
else:
    prediction_1 = prediction_2 = prediction_3 = prediction_4 = prediction_5 = (
        prediction_6
    ) = prediction_7 = None
    prediction_101 = prediction_102 = prediction_103 = prediction_104 = (
        prediction_105
    ) = prediction_106 = prediction_107 = None
    prediction_201 = prediction_202 = prediction_203 = prediction_204 = (
        prediction_205
    ) = prediction_206 = prediction_207 = None
    prediction_301 = prediction_302 = prediction_303 = prediction_304 = (
        prediction_305
    ) = prediction_306 = prediction_307 = None




## === cell 6
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p107,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p207,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
    p307,
):
    """
    Bugfix: ensure predictions are stored per-case and aligned 1:1 with sorted test case IDs.
    Keeps original ensemble semantics: mean over 28 slice/model predictions.
    """
    case_dirs = sorted([f.name for f in os.scandir(path_test) if f.is_dir()])
    cases = [d for d in case_dirs]

    preds = [
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p101,
        p102,
        p103,
        p104,
        p105,
        p106,
        p107,
        p201,
        p202,
        p203,
        p204,
        p205,
        p206,
        p207,
        p301,
        p302,
        p303,
        p304,
        p305,
        p306,
        p307,
    ]
    preds = [np.asarray(p, dtype=np.float32) for p in preds]
    n = len(cases)
    for idx, p in enumerate(preds):
        if len(p) != n:
            raise ValueError(
                f"Prediction vector {idx} has length {len(p)} but expected {n} to match test cases."
            )

    prediction = np.mean(np.stack(preds, axis=1), axis=1).astype(np.float32)
    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()

if have_all_models:
    sub_df = create_sub(
        TEST_DIR,
        prediction_1,
        prediction_2,
        prediction_3,
        prediction_4,
        prediction_5,
        prediction_6,
        prediction_7,
        prediction_101,
        prediction_102,
        prediction_103,
        prediction_104,
        prediction_105,
        prediction_106,
        prediction_107,
        prediction_201,
        prediction_202,
        prediction_203,
        prediction_204,
        prediction_205,
        prediction_206,
        prediction_207,
        prediction_301,
        prediction_302,
        prediction_303,
        prediction_304,
        prediction_305,
        prediction_306,
        prediction_307,
    )
    sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
    sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
else:
    train_labels = pd.read_csv(TRAIN_LABELS_CSV)
    baseline_prob = float(train_labels["MGMT_value"].mean())
    sub_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": baseline_prob})

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)

print(sub_df.head())
print("submission rows:", len(sub_df), "unique ids:", sub_df["BraTS21ID"].nunique())



## === cell 8
if sns is not None and plt is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)
else:
    print("Plotting libraries not available; skipping plot.")



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print("Columns:", list(sub_df.columns))
print("Preview:\n", sub_df.head().to_string(index=False))
