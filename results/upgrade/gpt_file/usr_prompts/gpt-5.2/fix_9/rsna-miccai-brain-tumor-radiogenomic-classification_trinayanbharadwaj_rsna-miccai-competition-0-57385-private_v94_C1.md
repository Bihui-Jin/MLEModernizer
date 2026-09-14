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

0.58

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the runtime import error by avoiding `pydicom` (which is triggering a protobuf incompatibility) and instead reading DICOM pixel data using the already-available `SimpleITK` (this keeps the same core image-loading logic: read slices → resize → normalize → select 6 slices). Then I fix the submission row-count mismatch by ensuring we generate exactly one prediction per test `BraTS21ID` even when some slices fail filtering, by padding missing slice predictions with a default value (0.5) and averaging across the six slices. Finally, I ensure the output `submission.csv` matches `sample_submission.csv` ordering and row count exactly so Kaggle accepts it.'
- What this solution (achieved 0.5) has done: 'I fix the runtime import crash caused by a protobuf incompatibility (triggered when TensorFlow loads and imports protobuf) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I keep your existing SimpleITK-based DICOM loading and the 6-slice averaging/padding logic intact, only adding a safe fallback so the script still produces a valid `submission.csv` even if TensorFlow cannot be imported or the pretrained model is missing. These changes are execution/stability focused and should preserve your current scoring behavior (0.5 baseline) while ensuring the notebook runs end-to-end and writes a valid CSV with the correct row count and ordering.'
- What this solution (achieved 0.5) has done: 'I fix the crash happening at import time by removing the protobuf environment override that is triggering the `MessageFactory.GetPrototype` incompatibility in this Kaggle image, and instead make TensorFlow an optional dependency (so the pipeline still runs even if TF can’t import). I also avoid importing `skimage` (often not available by default) by switching the resize step to `SimpleITK.Resample`, which keeps the same core logic (read slice → resize to 150×150 → normalize → pick 6 slices). Finally, I keep your existing 6-slice averaging/padding and sample-submission alignment so we always write a valid `submission.csv` with the correct rows/ordering. These changes are primarily to restore end-to-end execution; score improve only if TensorFlow can load the provided pretrained model in this environment, otherwise it safely fall back to 0.5.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by preventing TensorFlow from being imported at all in this environment (where protobuf incompatibility is causing `MessageFactory.GetPrototype` errors) and cleanly falling back to constant predictions. This keeps your core inference logic (6-slice load → per-slice predict → pad → average → sample-submission alignment) unchanged while guaranteeing the notebook runs end-to-end and always writes a valid `submission.csv`. I also move seeding/environment setup to occur before any optional TF import attempt (even though we skip TF), to avoid side effects. No score-tuning changes are introduced since your target score is -1.0 (nonsensical for AUC), and the priority here is producing a valid submission reliably.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC comes from producing constant predictions because TensorFlow is disabled, so the smallest legitimate way to move score upward toward a meaningful target is to re-enable TensorFlow safely and load the provided pretrained `.h5` model. I keep your exact preprocessing (SimpleITK DICOM read → resample to 150×150 → normalize → select 6 slices → average/pad) and only change the TF import gating so it attempts to import/load, while still cleanly falling back to 0.5 if it fails. I also make the DICOM slice selection stable by sorting filenames numerically (so “Image-10” doesn’t come before “Image-2”), which preserves the same slice-count logic but reduces accidental randomness and typically improves AUC without changing the approach. The script still always write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by forcing protobuf to use the pure-Python backend *before* attempting to import TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle image. I keep your exact SimpleITK-based DICOM loading, 6-slice selection, padding/averaging, and submission alignment logic unchanged. I also make the TensorFlow fallback path deterministic and safe so it always writes a valid `submission.csv` even if TF/model loading still fails. This should restore the intended pretrained-model inference (raising AUC above the constant 0.5 baseline when the model can load), while remaining within your core logic constraints.'
- What this solution (achieved 0.58) has done: 'I fix the crash in the first cell by preventing the TensorFlow/protobuf incompatibility from killing the run: TensorFlow be imported only if it can be safely loaded, otherwise we cleanly fall back to constant predictions (so a valid submission is always produced). To move the AUC score upward from the current constant-0.5 baseline toward a more meaningful level, I keep your exact preprocessing and 6-slice averaging logic, but switch the inference backend to a lightweight, dependency-free classical model (sklearn LogisticRegression) trained on simple intensity statistics extracted from the same 6 selected T2w slices. This preserves the “load slices → normalize → select 6 slices → aggregate → predict probability” semantics while avoiding TensorFlow entirely (and thus avoiding the protobuf crash). The submission is still aligned exactly to `sample_submission.csv` ordering and row count, and always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import SimpleITK as sitk

np.random.seed(0)

TF_AVAILABLE = False
tf = None
keras = None
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401

    TF_AVAILABLE = True
    try:
        tf.random.set_seed(0)
    except Exception:
        pass
    print("INFO: TensorFlow available (optional).")
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print(
        "WARNING: TensorFlow import failed; will proceed without it. "
        f"Import error: {repr(e)}"
    )

from sklearn.linear_model import LogisticRegression



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_PATH = "../input/trained-model-for-rsnamiccai/rsna_miccai_11_epochs_T2W_6k_.h5"
model_T2 = None

if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model_T2 = keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded TF model:", MODEL_PATH)
    except Exception as e:
        model_T2 = None
        print("WARNING: Failed to load TF model; proceeding without it.")
        print("Model load error:", repr(e))
else:
    if not os.path.exists(MODEL_PATH):
        print(f"INFO: TF model not found at {MODEL_PATH}; proceeding without it.")
    else:
        print("INFO: TensorFlow unavailable; proceeding without TF model.")




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
    Use SimpleITK resampling to resize a 2D array.
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


def _numeric_image_key(path):
    """
    Ensure slice filenames are sorted numerically so Image-10 doesn't come before Image-2.
    """
    base = os.path.basename(path)
    digits = "".join([c for c in base if c.isdigit()])
    if digits == "":
        return (base, 10**12)
    return (base, int(digits))


def load_T2W_images_for_cases(path_root, case_ids=None, verbose=False):
    """
    Load exactly up to 6 filtered/normalized/resized T2w slices per case, preserving the
    original slice selection logic.
    Returns:
      case_ids_out: list[str]
      arrays: list of 6 arrays, each shape (num_loaded_slices_at_index, 150,150,3)
      mask_loaded: list of 6 boolean lists aligned to case_ids_out indicating whether that slice index exists for that case
    """
    IMG_PX_SIZE = 150
    arrays = [[] for _ in range(6)]
    loaded_mask = [[] for _ in range(6)]

    if case_ids is None:
        case_paths = sorted([f.path for f in os.scandir(path_root) if f.is_dir()])
    else:
        case_paths = [os.path.join(path_root, cid) for cid in case_ids]

    case_ids_out = []
    for case_path in case_paths:
        cid = os.path.basename(case_path)
        if not os.path.isdir(case_path):
            continue

        count = 0
        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_types) < 4:
            case_ids_out.append(cid)
            for k in range(6):
                loaded_mask[k].append(False)
            continue

        img_dir = mri_types[3]
        img_paths = sorted(
            [f.path for f in os.scandir(img_dir) if f.is_file()],
            key=_numeric_image_key,
        )

        case_ids_out.append(cid)
        per_case_slices = [None] * 6

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
                    per_case_slices[count] = stacked_img_normalize
                    count += 1

        for k in range(6):
            if per_case_slices[k] is None:
                loaded_mask[k].append(False)
            else:
                arrays[k].append(per_case_slices[k])
                loaded_mask[k].append(True)

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    if verbose:
        print(
            "Loaded T2 images per slice index:", ", ".join(str(len(a)) for a in arrays)
        )
        print("Total cases:", len(case_ids_out))
    return case_ids_out, arrays, loaded_mask




## === cell 3
train_root = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
test_root = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
labels_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

train_labels = pd.read_csv(labels_path, dtype={"BraTS21ID": str})
bad_ids = set(["00109", "00123", "00709"])
train_labels = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(
    drop=True
)

print("Train labels shape (after excluding bad cases):", train_labels.shape)




## === cell 4
def _slice_stats(img_150_150_3):
    """
    Compute simple, stable intensity statistics from a normalized (0..1) 150x150x3 image.
    Keep lightweight to stay within time/memory constraints.
    """
    x = np.asarray(img_150_150_3, dtype=np.float32)
    if x.ndim != 3:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ch = x[..., 0]
    mean = float(np.mean(ch))
    std = float(np.std(ch))
    q10 = float(np.quantile(ch, 0.10))
    q90 = float(np.quantile(ch, 0.90))
    return np.array([mean, std, q10, q90], dtype=np.float32)


def build_case_features(case_ids, arrays6, loaded_mask6, default_fill=0.0):
    """
    Turn the per-slice loaded arrays + masks into fixed-length per-case features.
    For each of 6 slices, compute 4 stats -> 24 features per case.
    Missing slices are filled with default_fill.
    """
    n = len(case_ids)
    feats = np.full((n, 6 * 4), default_fill, dtype=np.float32)

    for k in range(6):
        mask_k = loaded_mask6[k]
        arr_k = arrays6[k]
        ptr = 0
        for i in range(n):
            if mask_k[i]:
                stats = _slice_stats(arr_k[ptr])
                feats[i, k * 4 : (k + 1) * 4] = stats
                ptr += 1
            else:
                pass
    return feats




## === cell 5
train_case_ids = train_labels["BraTS21ID"].tolist()
case_ids_out_tr, arrays6_tr, mask6_tr = load_T2W_images_for_cases(
    train_root, case_ids=train_case_ids, verbose=True
)

id_to_y = dict(
    zip(
        train_labels["BraTS21ID"].tolist(),
        train_labels["MGMT_value"].astype(int).tolist(),
    )
)
y_train = np.array([id_to_y[cid] for cid in case_ids_out_tr], dtype=np.int64)

X_train = build_case_features(case_ids_out_tr, arrays6_tr, mask6_tr, default_fill=0.0)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("y_train mean:", float(np.mean(y_train)))



## === cell 6
clf = LogisticRegression(
    solver="lbfgs",
    max_iter=500,
    random_state=0,
    class_weight=None,
)

clf.fit(X_train, y_train)
print("Trained LogisticRegression on intensity-stat features.")



## === cell 7
test_case_ids_out, arrays6_te, mask6_te = load_T2W_images_for_cases(
    test_root, case_ids=None, verbose=True
)
X_test = build_case_features(test_case_ids_out, arrays6_te, mask6_te, default_fill=0.0)
print("X_test shape:", X_test.shape)

proba_test = clf.predict_proba(X_test)[:, 1].astype(np.float32)
proba_test = np.clip(proba_test, 0.0, 1.0)

sub_df = pd.DataFrame({"BraTS21ID": test_case_ids_out, "MGMT_value": proba_test})



## === cell 8
sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
sample_df = pd.read_csv(sample_path, dtype={"BraTS21ID": str})

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str)
merged = sample_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

merged["MGMT_value"] = merged["MGMT_value"].astype(np.float32).fillna(0.5)
merged["MGMT_value"] = np.clip(merged["MGMT_value"].to_numpy(np.float32), 0.0, 1.0)

sub_df = merged

print("Submission preview:")
print(sub_df.head())
print("Submission shape:", sub_df.shape)
print(
    "Min/Max MGMT_value:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.dtypes)
