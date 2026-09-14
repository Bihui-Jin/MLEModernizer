# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.40824

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42941) has done: 'I remove/avoid the imports that trigger the `MessageFactory.GetPrototype` crash and the unused heavy dependencies, while keeping the same DICOM-to-299×299×3 preprocessing logic. Since the referenced pre-trained `.h5` files are not available in your environment, I keep the “single-slice-per-case” image loading but replace the missing model with a simple, deterministic baseline that outputs a probability from the same loaded pixels so the notebook runs end-to-end and produces a valid `submission.csv`. I also fix the `resize`/`randrange` name errors by using a local resize implementation (OpenCV) and `random.randrange`, and fix submission ID formatting to match `sample_submission.csv` (`00002` style strings) with proper alignment between IDs and predictions. The result run within the Kaggle environment and write a correct submission file.'
- What this solution (achieved 0.42941) has done: 'I fix the crash in the submission-creation cell by making sure `BraTS21ID` has the same dtype/format in both `sample_sub` and the predicted dataframe before merging (the current failure is an int-vs-string merge key mismatch). I also enforce zero-padded 5-digit string IDs (`'00002'` style) everywhere to guarantee alignment with `sample_submission.csv`. These changes are score-neutral (they don’t change predictions except ensuring they map to the correct IDs) and make the notebook run end-to-end and reliably write a valid `submission.csv`.'
- What this solution (achieved 0.40824) has done: 'Your current pipeline is a very weak baseline (single slice from a single modality + raw mean intensity), so the smallest legitimate improvement toward a higher ROC-AUC is to keep the same overall approach but make the per-case score slightly more informative and less noisy. I (1) read the DICOM slice more robustly (handle MONOCHROME1, apply rescale slope/intercept when present), (2) extract a few additional intensity statistics from the same already-loaded pixels (mean/std/high-percentile fraction) and combine them with fixed weights into a probability, and (3) ensure predictions stay well-calibrated by clipping away exact 0/1. This preserves your “one slice per case” core logic and still runs fast, but should generally move AUC up from ~0.43 toward a more reasonable baseline without changing any file paths or submission semantics. The script still writes a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pydicom as dicom
import cv2

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
sample_sub.head()




## === cell 2
def _sorted_case_dirs(path_test: str):
    case_dirs = [f.path for f in os.scandir(path_test) if f.is_dir()]
    case_dirs = sorted(case_dirs, key=lambda p: os.path.basename(p))
    return case_dirs


def _sorted_modality_dirs(case_dir: str):
    mods = [f.path for f in os.scandir(case_dir) if f.is_dir()]
    mods = sorted(mods, key=lambda p: os.path.basename(p))
    return mods


def _sorted_dcm_files(modality_dir: str):
    files = [f.path for f in os.scandir(modality_dir) if f.is_file()]
    files = sorted(files, key=lambda p: os.path.basename(p))
    return files


def _resize_to_299(img2d: np.ndarray, size: int = 299) -> np.ndarray:
    img = img2d.astype(np.float32)
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return resized


def _normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / mx).astype(np.float32)


def _read_dcm_pixels_float(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    return arr


def load_test_images_one_slice_per_case(
    path_test: str, modality_index: int, img_px_size: int = 299
):
    """
    Minimal bug-fix version of original loaders:
    - Uses cv2 resize instead of skimage.resize (avoids missing import/runtime issues).
    - Keeps original logic: for each case, scan slices and take first slice passing thresholds.
    - If none pass, fall back to middle slice.
    Returns:
      pixels: np.ndarray of shape (N, 299, 299, 3)
      ids: list of case folder names (e.g. '00002')
    """
    pixels = []
    ids = []

    case_dirs = _sorted_case_dirs(path_test)
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        mods = _sorted_modality_dirs(case_dir)
        if modality_index >= len(mods):
            continue
        dcm_files = _sorted_dcm_files(mods[modality_index])
        if len(dcm_files) == 0:
            continue

        chosen = None
        for fp in dcm_files:
            ds = dicom.dcmread(fp)
            arr = _read_dcm_pixels_float(ds)
            if arr.sum() > 100000:
                resized = _resize_to_299(arr, img_px_size)
                stacked = np.stack((resized,) * 3, axis=-1)
                stacked_norm = _normalize01(stacked)
                if stacked_norm.sum() > 5000:
                    chosen = stacked_norm
                    break

        if chosen is None:
            mid_fp = dcm_files[len(dcm_files) // 2]
            ds = dicom.dcmread(mid_fp)
            arr = _read_dcm_pixels_float(ds)
            resized = _resize_to_299(arr, img_px_size)
            stacked = np.stack((resized,) * 3, axis=-1)
            chosen = _normalize01(stacked)

        pixels.append(chosen)
        ids.append(case_id)

    pixels = np.asarray(pixels, dtype=np.float32)
    print(
        f"Loaded modality_index={modality_index} images: {pixels.shape[0]} cases, tensor shape={pixels.shape}"
    )
    return pixels, ids




## === cell 3
pixels_1, test_ids = load_test_images_one_slice_per_case(
    TEST_DIR, modality_index=0, img_px_size=299
)



## === cell 4
plt.figure(figsize=(18, 12))
n_show = min(6, len(pixels_1))
for i in range(n_show):
    plt.subplot(3, 2, i + 1)
    random_number = random.randrange(len(pixels_1))
    plt.imshow(pixels_1[random_number])
    plt.axis("off")
plt.tight_layout()




## === cell 5
def baseline_predict_proba_from_pixels(pixels: np.ndarray) -> np.ndarray:
    """
    Compute a per-case probability from simple intensity statistics on already-normalized pixels.
    Output shape: (N,)
    """
    x = pixels[..., 0].astype(
        np.float32
    )  # (N,H,W), channels are identical by construction

    mean_intensity = x.mean(axis=(1, 2))
    std_intensity = x.std(axis=(1, 2))

    frac_hi = (x > 0.80).mean(axis=(1, 2)).astype(np.float32)

    score = (1.25 * mean_intensity) + (0.55 * std_intensity) + (1.10 * frac_hi) - 0.65
    p = 1.0 / (1.0 + np.exp(-score))

    p = np.clip(p, 1e-4, 1.0 - 1e-4).astype(np.float32)
    return p


prediction_1 = baseline_predict_proba_from_pixels(pixels_1)
prediction_1[:10], prediction_1.shape




## === cell 6
def create_sub_from_ids(ids, preds):
    if len(ids) != len(preds):
        raise ValueError(f"Length mismatch: ids={len(ids)} preds={len(preds)}")

    ids = pd.Series(ids, dtype="string").str.zfill(5).tolist()
    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": preds.astype(float)})
    return df


sub_pred_df = create_sub_from_ids(test_ids, prediction_1)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_pred_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

sub_df.head(), sub_df.shape



## === cell 7
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub_df.describe(include="all"))
print("Submission dtypes:", sub_df.dtypes.to_dict())
print("Any NA in MGMT_value:", sub_df["MGMT_value"].isna().any())
