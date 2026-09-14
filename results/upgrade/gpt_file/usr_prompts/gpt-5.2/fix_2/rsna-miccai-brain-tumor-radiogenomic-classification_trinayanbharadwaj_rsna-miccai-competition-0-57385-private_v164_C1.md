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

0.46471

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46471) has done: 'I remove/replace imports that are breaking the notebook at import-time (the protobuf `GetPrototype` error) and make the script independent of non-existent external model files by falling back to a simple, deterministic baseline when those `.h5` models aren’t available. I also fix the `resize` NameError by importing it safely, and make the DICOM reading robust so it doesn’t crash on occasional bad slices. Finally, I fix the submission creation logic so it generates one probability per test subject (aligned to `BraTS21ID`) and always writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

import pydicom as dicom

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
test = os.path.join(DATA_ROOT, "test")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

sub_template = pd.read_csv(sample_sub_path)
sub_template["BraTS21ID"] = sub_template["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sub_template["BraTS21ID"].tolist()

print("Test subjects in sample_submission:", len(test_ids))
print("Test folder exists:", os.path.isdir(test))




## === cell 2
def _safe_read_dcm(dcm_path):
    try:
        ds = dicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def _slice_to_img(arr2d, img_px_size=150):
    resized = resize(
        arr2d, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    stacked = np.stack((resized,) * 3, axis=-1)
    mx = float(np.max(stacked))
    if mx <= 0:
        return None
    stacked_norm = stacked / mx
    return stacked_norm


def load_test_modality_images(path_test, modality_name, n_slices=6, img_px_size=150):
    """
    Returns: list_of_cases_imgs where each element is a list of up to n_slices images (H,W,3) for that case.
    We also return the case ids in the same order.
    """
    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in case_dirs]

    all_cases = []
    for case_path in case_dirs:
        modality_dir = os.path.join(case_path, modality_name)
        if not os.path.isdir(modality_dir):
            all_cases.append([])
            continue

        dcm_files = sorted(
            [
                f.path
                for f in os.scandir(modality_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        case_imgs = []
        count = 0

        for dcm_path in dcm_files:
            arr = _safe_read_dcm(dcm_path)
            if arr is None:
                continue

            if float(np.sum(arr)) <= 100000:
                continue

            img3 = _slice_to_img(arr, img_px_size=img_px_size)
            if img3 is None:
                continue

            if float(np.sum(img3)) <= 2000:
                continue

            case_imgs.append(img3)
            count += 1
            if count >= n_slices:
                break

        all_cases.append(case_imgs)

    return case_ids, all_cases




## === cell 3
t2_case_ids, t2_cases = load_test_modality_images(
    test, modality_name="T2w", n_slices=6, img_px_size=150
)
flair_case_ids, flair_cases = load_test_modality_images(
    test, modality_name="FLAIR", n_slices=6, img_px_size=150
)

print("Loaded T2w cases:", len(t2_case_ids), "FLAIR cases:", len(flair_case_ids))
if t2_case_ids != flair_case_ids:
    print("Warning: modality case order mismatch; will align by ID.")
case_ids_from_disk = t2_case_ids



## === cell 4
MODEL_PATHS = [
    "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b400_flair_5.5k_0.68auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
]


def try_load_model(path):
    if os.path.exists(path):
        try:
            return keras.models.load_model(path, compile=False)
        except Exception as e:
            print(f"Failed to load model {path}: {e}")
            return None
    return None


models = [try_load_model(p) for p in MODEL_PATHS]
available_models = [m for m in models if m is not None]
print(f"Models found: {len(available_models)}/{len(MODEL_PATHS)}")




## === cell 5
def _predict_model_on_slices(model, slice_imgs):
    """
    model: keras model expecting (N,150,150,3).
    slice_imgs: list of images for one case.
    Returns: float probability for the case (mean over slices).
    """
    if model is None or len(slice_imgs) == 0:
        return None

    x = np.asarray(slice_imgs, dtype=np.float32)
    preds = model.predict(x, verbose=0)

    if preds.ndim == 2 and preds.shape[1] >= 2:
        p = preds[:, 1].astype(np.float32)
    else:
        p = preds.reshape(-1).astype(np.float32)

    p = np.clip(p, 0.0, 1.0)
    return float(np.mean(p))


def _baseline_prob_from_slices(t2_slices, flair_slices):
    """
    Deterministic fallback: use mean intensity as a weak signal and map through a sigmoid.
    This is only used when pretrained models are unavailable.
    """
    feats = []
    for sl in t2_slices[:6] if t2_slices else []:
        feats.append(float(np.mean(sl)))
    for sl in flair_slices[:6] if flair_slices else []:
        feats.append(float(np.mean(sl)))

    if len(feats) == 0:
        return 0.5

    m = float(np.mean(feats))
    z = (m - 0.5) * 6.0
    p = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(p, 0.0, 1.0))




## === cell 6
t2_model_idxs = [0, 1, 4, 5]
flair_model_idxs = [2, 3, 6]


def predict_case(case_idx):
    t2_slices = t2_cases[case_idx] if case_idx < len(t2_cases) else []
    fl_slices = flair_cases[case_idx] if case_idx < len(flair_cases) else []

    per_model_preds = []

    for mi in t2_model_idxs:
        if mi < len(models) and models[mi] is not None:
            p = _predict_model_on_slices(models[mi], t2_slices)
            if p is not None:
                per_model_preds.append(p)

    for mi in flair_model_idxs:
        if mi < len(models) and models[mi] is not None:
            p = _predict_model_on_slices(models[mi], fl_slices)
            if p is not None:
                per_model_preds.append(p)

    if len(per_model_preds) > 0:
        return float(np.mean(per_model_preds))

    return _baseline_prob_from_slices(t2_slices, fl_slices)


pred_by_id = {}
for i, cid in enumerate(case_ids_from_disk):
    pred_by_id[str(cid).zfill(5)] = predict_case(i)

print("Predictions computed for cases:", len(pred_by_id))



## === cell 7
sub_df = sub_template.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_by_id).astype(np.float32)

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("MGMT_value stats:", sub_df["MGMT_value"].describe())



## === cell 8
if sns is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
    except Exception:
        pass



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(sub_df), "rows")
