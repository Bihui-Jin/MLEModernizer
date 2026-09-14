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

0.46

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.46) has done: 'Your script already creates `submission.csv`, but your `BraTS21ID` values are strings like `"00002"` while the competition expects 5-digit IDs and Kaggle align/parse more reliably if the column is numeric or consistently zero-padded; we force exact 5-digit formatting and ensure ordering matches `sample_submission.csv`. To move the AUC up from an uncalibrated heuristic without changing the “core logic” (still intensity-based, deterministic, no training), we (1) average probabilities across the 6 slices first, then (2) apply the same sigmoid calibration once on the averaged score instead of per-slice (reduces noise and tends to improve ranking/AUC). Finally, we clip predictions to a safe range and write the submission in the exact row order of `sample_submission.csv` to avoid any accidental misalignment.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize




## === cell 1
def _safe_stack_and_normalize(img2d: np.ndarray) -> np.ndarray:
    img = np.asarray(img2d, dtype=np.float32)
    stacked = np.stack((img,) * 3, axis=-1)
    mx = float(np.max(stacked))
    if mx <= 0:
        return stacked
    return stacked / mx


def _safe_div_by_max(arr_list):
    arr = np.asarray(arr_list, dtype=np.float32)
    mx = float(np.max(arr)) if arr.size else 0.0
    if mx > 0:
        arr = arr / mx
    return arr


def _get_modality_dir(case_dir: str, modality_name: str) -> str:
    candidate = os.path.join(case_dir, modality_name)
    if os.path.isdir(candidate):
        return candidate
    for f in os.scandir(case_dir):
        if f.is_dir() and f.name.lower() == modality_name.lower():
            return f.path
    return ""


def _list_dcm_files(folder: str):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        [
            f.path
            for f in os.scandir(folder)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )


def _zero_image(img_px_size: int = 299) -> np.ndarray:
    return np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)


def _load_case_slices(
    case_dir: str, modality_name: str, n_slices: int, img_px_size: int = 299
):
    """
    Load up to n_slices per case for a modality. If fewer are found, pad with zeros.
    This preserves the original core logic (thresholding + resize + normalization) but prevents length mismatches.
    """
    modality_dir = _get_modality_dir(case_dir, modality_name)
    img_paths = _list_dcm_files(modality_dir)

    out = []
    for p in img_paths:
        try:
            img = dicom.dcmread(p)
            arr = img.pixel_array
        except Exception:
            continue

        if arr.sum() > 100000:
            resized_img = resize(
                arr,
                (img_px_size, img_px_size),
                preserve_range=True,
                anti_aliasing=True,
            )
            stacked_img_normalize = _safe_stack_and_normalize(resized_img)
            if stacked_img_normalize.sum() > 10000:
                out.append(stacked_img_normalize)
                if len(out) >= n_slices:
                    break

    while len(out) < n_slices:
        out.append(_zero_image(img_px_size))

    return out


def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_dir in path_cases:
        slices = _load_case_slices(
            case_dir, "FLAIR", n_slices=6, img_px_size=IMG_PX_SIZE
        )
        array_1.append(slices[0])
        array_2.append(slices[1])
        array_3.append(slices[2])
        array_4.append(slices[3])
        array_5.append(slices[4])
        array_6.append(slices[5])

    array_1 = _safe_div_by_max(array_1)
    array_2 = _safe_div_by_max(array_2)
    array_3 = _safe_div_by_max(array_3)
    array_4 = _safe_div_by_max(array_4)
    array_5 = _safe_div_by_max(array_5)
    array_6 = _safe_div_by_max(array_6)

    print(
        "Number of flair images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        "and",
        len(array_6),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6


def load_test_T1W_images(path_test):
    array = []
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_dir in path_cases:
        sl = _load_case_slices(case_dir, "T1w", n_slices=1, img_px_size=IMG_PX_SIZE)[0]
        array.append(sl)

    array = _safe_div_by_max(array)
    print("Number of T1W images loaded are ", len(array))
    return array


def load_test_T1wCE_images(path_test):
    array = []
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_dir in path_cases:
        sl = _load_case_slices(case_dir, "T1wCE", n_slices=1, img_px_size=IMG_PX_SIZE)[
            0
        ]
        array.append(sl)

    array = _safe_div_by_max(array)
    print("Number of T1wCE images loaded are ", len(array))
    return array


def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_dir in path_cases:
        slices = _load_case_slices(case_dir, "T2w", n_slices=6, img_px_size=IMG_PX_SIZE)
        array_1.append(slices[0])
        array_2.append(slices[1])
        array_3.append(slices[2])
        array_4.append(slices[3])
        array_5.append(slices[4])
        array_6.append(slices[5])

    array_1 = _safe_div_by_max(array_1)
    array_2 = _safe_div_by_max(array_2)
    array_3 = _safe_div_by_max(array_3)
    array_4 = _safe_div_by_max(array_4)
    array_5 = _safe_div_by_max(array_5)
    array_6 = _safe_div_by_max(array_6)

    print(
        "Number of T2W images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        "and",
        len(array_6),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 2
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 3
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_flair_images(
    test
)




## === cell 4
class SimpleDeterministicModel:
    def predict(self, x, batch_size=32, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input of shape (N,H,W,C), got {x.shape}")
        mean_intensity = x.mean(axis=(1, 2, 3))
        std_intensity = x.std(axis=(1, 2, 3))
        score = 0.7 * mean_intensity + 0.3 * std_intensity
        p1 = 1.0 / (1.0 + np.exp(-(score - np.median(score)) / (np.std(score) + 1e-6)))
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


model_1 = SimpleDeterministicModel()

if len(pixels_1) == 0:
    raise RuntimeError("No test images were loaded. Check test path and DICOM reading.")

preds_1 = model_1.predict(pixels_1)
prediction_1 = preds_1[:, 1]
preds_2 = model_1.predict(pixels_2)
prediction_2 = preds_2[:, 1]
preds_3 = model_1.predict(pixels_3)
prediction_3 = preds_3[:, 1]
preds_4 = model_1.predict(pixels_4)
prediction_4 = preds_4[:, 1]
preds_5 = model_1.predict(pixels_5)
prediction_5 = preds_5[:, 1]
preds_6 = model_1.predict(pixels_6)
prediction_6 = preds_6[:, 1]




## === cell 5
def create_sub(path_test, p1, p2, p3, p4, p5, p6):
    sample_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
    sample = pd.read_csv(sample_path)
    sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

    cases = []
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        case_number = os.path.basename(path_cases[i])  # e.g., "00002"
        cases.append(str(case_number).zfill(5))

    raw_avg = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
    ) / 6.0

    med = float(np.median(raw_avg))
    sd = float(np.std(raw_avg)) + 1e-6
    prediction = 1.0 / (1.0 + np.exp(-((raw_avg - med) / sd)))
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)

    if len(cases) != len(prediction):
        raise ValueError(
            f"Mismatch: {len(cases)} cases but {len(prediction)} predictions"
        )

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})

    df = sample[["BraTS21ID"]].merge(df, on="BraTS21ID", how="left")
    if df["MGMT_value"].isna().any():
        missing = df.loc[df["MGMT_value"].isna(), "BraTS21ID"].tolist()[:10]
        raise ValueError(
            f"Missing predictions for some IDs (showing up to 10): {missing}"
        )

    return df




## === cell 6
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)



## === cell 7
sub_df.head()



## === cell 8
sub_df["MGMT_value"].describe()



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
