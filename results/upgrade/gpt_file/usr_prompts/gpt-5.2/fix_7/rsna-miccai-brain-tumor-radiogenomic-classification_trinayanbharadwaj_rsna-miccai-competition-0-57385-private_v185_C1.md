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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the initial import crash by removing/guarding non-essential imports that trigger the protobuf `MessageFactory.GetPrototype` error, while keeping the core inference logic intact. Then I make model loading robust: if the external pretrained models are not available in this Kaggle environment, the code fall back to a safe baseline that still produces a valid submission (using the sample submission probabilities). I also fix the `resize` NameError and correct the submission creation logic (it currently recomputes `prediction` inside the loop incorrectly and has indentation issues), ensuring predictions align 1:1 with the sorted test IDs. Finally, I guarantee `submission.csv` is written with the exact required columns and formatting.'
- What this solution (achieved 0.5) has done: 'I fix the crash happening at import time by pinning protobuf to the compatible pure-Python implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in Kaggle’s environment. I also make the DICOM modality selection robust (choose folders by name like `T2w/FLAIR/T1wCE` instead of relying on fixed indices), which prevents silent mis-ordering and improves the chance that model inference runs correctly. Finally, I make the submission ID order deterministic by deriving `test_ids` from the actual `test/` directory (and then reindexing to match the sample submission order), ensuring the output is valid and aligned 1:1 with the required rows. Core model usage and averaging logic are preserved.'
- What this solution (achieved 0.5) has done: 'I fix the protobuf/TensorFlow import crash that currently stops execution by forcing the pure-Python protobuf implementation early and (if needed) falling back cleanly without breaking the rest of the pipeline. I also make the model-loading block robust to “missing trained model directory” (common in this environment), so it deterministically uses the fallback submission rather than partially-initialized variables. Finally, I keep your core prediction/averaging logic unchanged, but ensure the submission is always produced as a valid `submission.csv` with IDs aligned to the sample submission order. Since the target score is -1.0 (not meaningful for AUC) and the current score is already a reasonable baseline (0.5), I not introduce score-changing modeling changes—only stability/correctness fixes.'
- What this solution (achieved 0.5) has done: 'I fix the import-time protobuf crash that prevents the notebook from running by forcing the pure-Python protobuf implementation and (critically) reloading the `google.protobuf` module before importing TensorFlow. This is a correctness/stability change only and preserves your existing model loading/inference logic unchanged. I also add a robust fallback to guarantee a valid `submission.csv` is always written even if TensorFlow, pydicom, skimage, or the external model directory are unavailable. Since the target score (-1.0) is not meaningful for AUC and the current score is already in a reasonable baseline range, I not introduce score-changing modeling edits.'
- What this solution (achieved 0.5) has done: 'I fix the runtime crash coming from the protobuf/TensorFlow incompatibility by forcing the Python protobuf implementation *and* setting `protobuf` to use the pure-Python backend via an additional environment variable before any TensorFlow import occurs, and by cleanly skipping TF import if it still fails. I also make the script robust to cases where `pydicom`/`skimage`/TensorFlow/models are unavailable by reliably falling back to the sample submission probabilities (score-neutral baseline ~0.5 AUC), ensuring end-to-end execution and a valid `submission.csv`. Finally, I keep your core inference/averaging logic unchanged, but add small guards to avoid `None` placeholders being appended as “loaded models” and to guarantee the test ID ordering matches the sample submission.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by forcing the pure-Python protobuf implementation early and ensuring TensorFlow is never imported unless it can be safely loaded; this removes the `MessageFactory.GetPrototype` failure that currently stops the pipeline. I also make the TensorFlow import fully optional and keep the existing behavior of falling back to the sample submission probabilities when models/DICOM tooling aren’t available (score-neutral vs your current 0.5 baseline). Finally, I keep the submission alignment logic but harden it with a couple of small guards to always write a valid `submission.csv` with the exact required columns and row order.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

for mod in list(sys.modules.keys()):
    if mod.startswith("google.protobuf"):
        del sys.modules[mod]

try:
    import google.protobuf  # noqa: F401

    importlib.reload(google.protobuf)
except Exception as e:
    print("WARNING: google.protobuf reload failed (continuing):", repr(e))

try:
    import pydicom as dicom
except Exception as e:
    dicom = None
    print(
        "WARNING: pydicom import failed; DICOM loading will be unavailable. Error:",
        repr(e),
    )

try:
    from skimage.transform import resize  # used by the original image-loading code
except Exception as e:
    resize = None
    print(
        "WARNING: skimage.transform.resize import failed; image resizing will be unavailable. Error:",
        repr(e),
    )

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
except Exception as e:
    tf = None
    keras = None
    print(
        "WARNING: TensorFlow/Keras import failed; pretrained model inference will be unavailable. Error:",
        repr(e),
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_DIR = "../input/trained-model-for-rsnamiccai"
MODEL_PATHS = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
]


def _try_load_model(path):
    if keras is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        return keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"WARNING: failed to load model at {path}: {repr(e)}")
        return None


models = []
if os.path.isdir(MODEL_DIR):
    for mp in MODEL_PATHS:
        full = os.path.join(MODEL_DIR, mp)
        models.append(_try_load_model(full))
else:
    models = [None] * len(MODEL_PATHS)

(model_T2, model_T2_2, model_T2_3, model_T2_4, model_T2_5, model_T2_6) = models

loaded_count = sum(m is not None for m in models)
print(f"Loaded {loaded_count}/{len(models)} pretrained models.")




## === cell 2
def _get_modality_dir(case_path: str, modality_name: str) -> str:
    modality_dirs = [f.path for f in os.scandir(case_path) if f.is_dir()]
    by_name = {os.path.basename(p).lower(): p for p in modality_dirs}
    key = modality_name.lower()
    if key in by_name:
        return by_name[key]
    for k, p in by_name.items():
        if key in k:
            return p
    raise FileNotFoundError(
        f"Could not find modality '{modality_name}' in {case_path}. Found: {sorted(by_name.keys())}"
    )


def _load_7_images_per_case(path_test: str, modality_name: str):
    array_1, array_2, array_3, array_4, array_5, array_6, array_7 = (
        [] for _ in range(7)
    )
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    if dicom is None or resize is None:
        raise RuntimeError(
            "DICOM loading/resizing is unavailable (pydicom or skimage is missing)."
        )

    for case_path in path_cases:
        count = 0
        modality_dir = _get_modality_dir(case_path, modality_name)
        img_path = sorted([f.path for f in os.scandir(modality_dir) if f.is_file()])
        for p in img_path:
            img = dicom.dcmread(p)
            if img.pixel_array.sum() > 100000:
                resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                img_arr = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img_arr,) * 3, axis=-1)
                mx = np.max(stacked_img)
                if mx <= 0:
                    continue
                stacked_img_normalize = stacked_img / mx
                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 6:
                        array_7.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 7:
                        break

    out = []
    for a in [array_1, array_2, array_3, array_4, array_5, array_6, array_7]:
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            out.append(a)
        else:
            mx = np.max(a)
            out.append(a / mx if mx > 0 else a)
    return tuple(out)


def load_test_T2W_images(path_test):
    out = _load_7_images_per_case(path_test, "T2w")
    print("Number of T2 images loaded are ", ", ".join(str(len(x)) for x in out))
    return out


def load_test_flair_images(path_test):
    out = _load_7_images_per_case(path_test, "FLAIR")
    print("Number of flair images loaded are ", ", ".join(str(len(x)) for x in out))
    return out


def load_test_T1wce_images(path_test):
    out = _load_7_images_per_case(path_test, "T1wCE")
    print("Number of T1wce images loaded are ", ", ".join(str(len(x)) for x in out))
    return out




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_sub_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)

sample_df = pd.read_csv(sample_sub_path)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

fs_test_ids = sorted([f.name for f in os.scandir(test) if f.is_dir()])
fs_test_ids = [str(x).zfill(5) for x in fs_test_ids]

sample_ids = sample_df["BraTS21ID"].tolist()
if set(sample_ids) == set(fs_test_ids) and len(sample_ids) == len(fs_test_ids):
    test_ids = sample_ids
else:
    test_ids = fs_test_ids

print("Test IDs:", len(test_ids))



## === cell 4
use_models = all(
    m is not None
    for m in [model_T2, model_T2_2, model_T2_3, model_T2_4, model_T2_5, model_T2_6]
)
use_dicom = (dicom is not None) and (resize is not None)

print("use_models:", use_models, "use_dicom:", use_dicom)

if use_models and use_dicom:
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = (
        load_test_T2W_images(test)
    )
    pixels_7b, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12, pixels_13 = (
        load_test_flair_images(test)
    )
    pixels_13b, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18, pixels_19 = (
        load_test_T1wce_images(test)
    )

    n = len(test_ids)
    for idx, arr in enumerate(
        [
            pixels_1,
            pixels_2,
            pixels_3,
            pixels_4,
            pixels_5,
            pixels_6,
            pixels_7,
            pixels_7b,
            pixels_8,
            pixels_9,
            pixels_10,
            pixels_11,
            pixels_12,
            pixels_13,
            pixels_13b,
            pixels_14,
            pixels_15,
            pixels_16,
            pixels_17,
            pixels_18,
            pixels_19,
        ],
        start=1,
    ):
        if len(arr) != n:
            raise RuntimeError(
                f"Loaded image count mismatch at block {idx}: got {len(arr)}, expected {n}"
            )

    def _pred_pos(model, x):
        preds = model.predict(x, verbose=0)
        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[1] >= 2:
            return preds[:, 1].astype(np.float32)
        return preds.reshape(-1).astype(np.float32)

    prediction_1 = _pred_pos(model_T2, pixels_1)
    prediction_2 = _pred_pos(model_T2, pixels_2)
    prediction_3 = _pred_pos(model_T2, pixels_3)
    prediction_4 = _pred_pos(model_T2, pixels_4)
    prediction_5 = _pred_pos(model_T2, pixels_5)
    prediction_6 = _pred_pos(model_T2, pixels_6)
    prediction_7 = _pred_pos(model_T2, pixels_7)

    prediction_101 = _pred_pos(model_T2_2, pixels_1)
    prediction_102 = _pred_pos(model_T2_2, pixels_2)
    prediction_103 = _pred_pos(model_T2_2, pixels_3)
    prediction_104 = _pred_pos(model_T2_2, pixels_4)
    prediction_105 = _pred_pos(model_T2_2, pixels_5)
    prediction_106 = _pred_pos(model_T2_2, pixels_6)
    prediction_107 = _pred_pos(model_T2_2, pixels_7)

    prediction_201 = _pred_pos(model_T2_3, pixels_7b)
    prediction_202 = _pred_pos(model_T2_3, pixels_8)
    prediction_203 = _pred_pos(model_T2_3, pixels_9)
    prediction_204 = _pred_pos(model_T2_3, pixels_10)
    prediction_205 = _pred_pos(model_T2_3, pixels_11)
    prediction_206 = _pred_pos(model_T2_3, pixels_12)
    prediction_207 = _pred_pos(model_T2_3, pixels_13)

    prediction_301 = _pred_pos(model_T2_4, pixels_13b)
    prediction_302 = _pred_pos(model_T2_4, pixels_14)
    prediction_303 = _pred_pos(model_T2_4, pixels_15)
    prediction_304 = _pred_pos(model_T2_4, pixels_16)
    prediction_305 = _pred_pos(model_T2_4, pixels_17)
    prediction_306 = _pred_pos(model_T2_4, pixels_18)
    prediction_307 = _pred_pos(model_T2_4, pixels_19)

    prediction_401 = _pred_pos(model_T2_5, pixels_1)
    prediction_402 = _pred_pos(model_T2_5, pixels_2)
    prediction_403 = _pred_pos(model_T2_5, pixels_3)
    prediction_404 = _pred_pos(model_T2_5, pixels_4)
    prediction_405 = _pred_pos(model_T2_5, pixels_5)
    prediction_406 = _pred_pos(model_T2_5, pixels_6)
    prediction_407 = _pred_pos(model_T2_5, pixels_7)

    prediction_501 = _pred_pos(model_T2_6, pixels_1)
    prediction_502 = _pred_pos(model_T2_6, pixels_2)
    prediction_503 = _pred_pos(model_T2_6, pixels_3)
    prediction_504 = _pred_pos(model_T2_6, pixels_4)
    prediction_505 = _pred_pos(model_T2_6, pixels_5)
    prediction_506 = _pred_pos(model_T2_6, pixels_6)
    prediction_507 = _pred_pos(model_T2_6, pixels_7)
else:
    prediction_fallback = (
        sample_df.set_index("BraTS21ID")
        .reindex(test_ids)["MGMT_value"]
        .fillna(0.5)
        .astype(np.float32)
        .values
    )




## === cell 5
def create_sub_from_predictions(prediction_vector):
    prediction_vector = np.asarray(prediction_vector, dtype=np.float32).reshape(-1)
    if len(prediction_vector) != len(test_ids):
        raise ValueError(
            f"Prediction length {len(prediction_vector)} does not match test_ids length {len(test_ids)}"
        )
    df = pd.DataFrame(
        {"BraTS21ID": test_ids, "MGMT_value": prediction_vector.astype(np.float32)}
    )
    df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
    df["MGMT_value"] = df["MGMT_value"].clip(0.0, 1.0)
    return df


if use_models and use_dicom:
    prediction = (
        prediction_1.astype(np.float32)
        + prediction_2.astype(np.float32)
        + prediction_3.astype(np.float32)
        + prediction_4.astype(np.float32)
        + prediction_5.astype(np.float32)
        + prediction_6.astype(np.float32)
        + prediction_101.astype(np.float32)
        + prediction_102.astype(np.float32)
        + prediction_103.astype(np.float32)
        + prediction_104.astype(np.float32)
        + prediction_105.astype(np.float32)
        + prediction_106.astype(np.float32)
        + prediction_201.astype(np.float32)
        + prediction_202.astype(np.float32)
        + prediction_203.astype(np.float32)
        + prediction_204.astype(np.float32)
        + prediction_205.astype(np.float32)
        + prediction_206.astype(np.float32)
        + prediction_301.astype(np.float32)
        + prediction_302.astype(np.float32)
        + prediction_303.astype(np.float32)
        + prediction_304.astype(np.float32)
        + prediction_305.astype(np.float32)
        + prediction_306.astype(np.float32)
        + prediction_307.astype(np.float32)
        + prediction_401.astype(np.float32)
        + prediction_402.astype(np.float32)
        + prediction_403.astype(np.float32)
        + prediction_404.astype(np.float32)
        + prediction_405.astype(np.float32)
        + prediction_406.astype(np.float32)
        + prediction_501.astype(np.float32)
        + prediction_502.astype(np.float32)
        + prediction_503.astype(np.float32)
        + prediction_504.astype(np.float32)
        + prediction_505.astype(np.float32)
        + prediction_506.astype(np.float32)
    ) / 36.0

    sub_df = create_sub_from_predictions(prediction)
else:
    sub_df = create_sub_from_predictions(prediction_fallback)

assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(test_ids)

sub_df = sub_df.set_index("BraTS21ID")
sub_df = sub_df.reindex(sample_df["BraTS21ID"]).reset_index()

sub_df["MGMT_value"] = (
    sub_df["MGMT_value"].astype(np.float32).fillna(0.5).clip(0.0, 1.0)
)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with", len(sub_df), "rows")
