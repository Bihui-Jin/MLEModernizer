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

0.43765

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.43765) has done: 'I remove/avoid the imports that trigger the protobuf/pydicom crash and replace the missing external pretrained model files with a minimal fallback that still outputs valid probabilities. I also fix the missing `resize` reference by using a small local resize routine (no skimage dependency), and make the DICOM reading robust (handle missing/invalid slices and sort modalities by folder name rather than index). Finally, I fix submission alignment: generate exactly one prediction per test subject, ensure `BraTS21ID` formatting matches the sample submission, and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


RNG_SEED = 42
np.random.seed(RNG_SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("TEST_DIR exists:", os.path.isdir(TEST_DIR))
print("SAMPLE_SUB exists:", os.path.isfile(SAMPLE_SUB_PATH))



## === cell 1
try:
    import pydicom

    _PYDICOM_OK = True
except Exception as e:
    _PYDICOM_OK = False
    _PYDICOM_IMPORT_ERR = repr(e)

print("pydicom available:", _PYDICOM_OK)
if not _PYDICOM_OK:
    print("pydicom import error:", _PYDICOM_IMPORT_ERR)


def _resize_nn(img2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize for 2D arrays, avoids external deps (cv2/skimage)."""
    if img2d.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img2d.shape}")
    in_h, in_w = img2d.shape
    if in_h == 0 or in_w == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    y_idx = (np.linspace(0, in_h - 1, out_h)).astype(np.int64)
    x_idx = (np.linspace(0, in_w - 1, out_w)).astype(np.int64)
    return img2d[y_idx[:, None], x_idx[None, :]].astype(np.float32)


def _read_dicom_pixel_array(dcm_path: str) -> np.ndarray:
    """Read DICOM pixel array robustly. Returns float32 2D array or raises."""
    if not _PYDICOM_OK:
        raise RuntimeError("pydicom not available in this environment.")
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    return arr




## === cell 2
def _sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _case_probability_from_slices(slices: np.ndarray) -> float:
    """
    slices: (n_slices, H, W, 3) float in [0,1] (approximately)
    Produce a single probability per case.
    """
    if slices.size == 0:
        return 0.5
    x = float(slices.mean())
    s = float(slices.std())
    logit = (x - 0.35) * 8.0 + (s - 0.15) * 4.0
    return float(_sigmoid(logit))




## === cell 3
def _list_case_dirs(path_test: str):
    case_dirs = [f.path for f in os.scandir(path_test) if f.is_dir()]
    case_dirs = sorted(case_dirs, key=lambda p: os.path.basename(p))
    return case_dirs


def _find_modality_dir(case_dir: str, modality_name: str) -> str:
    mod_dir = os.path.join(case_dir, modality_name)
    return mod_dir if os.path.isdir(mod_dir) else ""


def _load_case_slices(
    case_dir: str, modality: str, img_px_size: int, max_slices: int = 6
):
    """
    Load up to max_slices informative slices for a case/modality.
    Returns (n, img_px_size, img_px_size, 3) float32.
    """
    mod_dir = _find_modality_dir(case_dir, modality)
    if not mod_dir:
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = [
        f.path
        for f in os.scandir(mod_dir)
        if f.is_file() and f.name.lower().endswith(".dcm")
    ]
    dcm_files = sorted(dcm_files, key=lambda p: os.path.basename(p))

    chosen = []
    for fp in dcm_files:
        try:
            arr = _read_dicom_pixel_array(fp)  # 2D float32
        except Exception:
            continue

        if float(arr.sum()) <= 100000.0:
            continue

        arr_rs = _resize_nn(arr, img_px_size, img_px_size)

        mx = float(np.max(arr_rs))
        if mx <= 0:
            continue
        arr_norm = arr_rs / mx

        stacked = np.stack([arr_norm, arr_norm, arr_norm], axis=-1).astype(np.float32)

        if float(stacked.sum()) < (2500.0 if modality == "FLAIR" else 1900.0):
            continue

        chosen.append(stacked)
        if len(chosen) >= max_slices:
            break

    if len(chosen) == 0:
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    out = np.stack(chosen, axis=0).astype(np.float32)
    denom = float(np.max(out))
    if denom > 0:
        out = out / denom
    return out




## === cell 4
def predict_test_set(path_test: str):
    """
    For each case, load up to 6 slices from T2w and FLAIR and average probabilities
    across slices and modalities (mirrors original 'average many predictions' idea,
    but without missing external models).
    """
    case_dirs = _list_case_dirs(path_test)
    case_ids = [os.path.basename(p) for p in case_dirs]

    probs = []
    for cd in case_dirs:
        t2_slices = _load_case_slices(cd, modality="T2w", img_px_size=150, max_slices=6)
        flair_slices = _load_case_slices(
            cd, modality="FLAIR", img_px_size=299, max_slices=6
        )

        if not _PYDICOM_OK:
            probs.append(0.5)
            continue

        p_t2 = _case_probability_from_slices(t2_slices)
        p_flair = _case_probability_from_slices(flair_slices)
        p = 0.5 * (p_t2 + p_flair)
        probs.append(float(np.clip(p, 0.0, 1.0)))

    return case_ids, np.array(probs, dtype=np.float32)


case_ids, case_probs = predict_test_set(TEST_DIR)
print("Predicted cases:", len(case_ids), "Probs shape:", case_probs.shape)
print(
    "Probs summary:",
    float(case_probs.min()),
    float(case_probs.mean()),
    float(case_probs.max()),
)



## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, dtype={"BraTS21ID": str})
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].str.zfill(5)

pred_map = {cid.zfill(5): float(p) for cid, p in zip(case_ids, case_probs)}

sample_sub["MGMT_value"] = (
    sample_sub["BraTS21ID"].map(pred_map).fillna(0.5).astype(float)
)

assert list(sample_sub.columns) == ["BraTS21ID", "MGMT_value"]
assert sample_sub["MGMT_value"].between(0.0, 1.0).all()

sample_sub.head()



## === cell 6
out_path = "submission.csv"
sample_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sample_sub))
print(sample_sub.describe(include="all"))
