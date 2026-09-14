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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58588) has done: 'I remove/avoid the problematic imports that trigger the `MessageFactory.GetPrototype` protobuf error and also drop the dependency on missing external pre-trained `.h5` files by keeping the same inference flow but using a small CNN defined in-code. I fix `resize`/array-type bugs in the DICOM loader (lists were being divided by scalars, and an off-by-one prevented `array_12` from ever filling) and make sure every test case always yields exactly 11 slices so predictions align by `BraTS21ID`. Finally, I correct `create_sub` so it computes one probability per case (not the same vector for all cases) and ensure the submission is written as `submission.csv` with the required columns and formatting.'
- What this solution (achieved 0.5) has done: 'I fix the immediate runtime crash by avoiding the `protobuf`/`MessageFactory.GetPrototype` issue that is triggered when importing TensorFlow in this Kaggle environment, while keeping the rest of your pipeline (DICOM loading, 11-slice extraction, per-slice inference, averaging, and CSV formatting) intact. Since your code currently doesn’t train and uses an untrained CNN for inference, I replace only the model/predict portion with a score-neutral fallback that outputs a constant probability (0.5) per case so the notebook runs end-to-end and always produces a valid `submission.csv`. This is the smallest change that guarantees execution and a correctly formatted submission without adding new dependencies or changing the data flow. The submission alignment/reindexing logic is preserved to ensure `BraTS21ID` ordering matches `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

SEED = 42
np.random.seed(SEED)



## === cell 1
BASE_INPUT_CANDIDATES = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
]
BASE_INPUT = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        BASE_INPUT = p
        break

if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not locate competition dataset folder under expected Kaggle input paths."
    )

TEST_DIR = os.path.join(BASE_INPUT, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()
print("Found test cases:", len(test_ids), "Example:", test_ids[:5])




## === cell 2
def _read_dicom_pixels(dcm_path: str) -> np.ndarray:
    """Read DICOM and return pixel array as float32."""
    ds = dicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    return arr


def _normalize01(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1] with safe guard."""
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)


def load_test_T2W_images(path_test, img_px_size=150, n_slices=11):
    """
    Load T2w images for each case and return a list of n_slices arrays:
      pixels_slices[j] has shape (N_cases, img_px_size, img_px_size, 3)
    Ensures every case contributes exactly n_slices (pads by repeating last valid slice).
    """
    pixels_by_slice = [[] for _ in range(n_slices)]

    for case_id in test_ids:
        case_dir = os.path.join(path_test, case_id)
        t2_dir = os.path.join(case_dir, "T2w")
        if not os.path.isdir(t2_dir):
            modality_dirs = [f.path for f in os.scandir(case_dir) if f.is_dir()]
            t2_candidates = [
                d for d in modality_dirs if "t2" in os.path.basename(d).lower()
            ]
            if len(t2_candidates) == 0:
                raise FileNotFoundError(f"No T2w directory found for case {case_id}")
            t2_dir = sorted(t2_candidates)[0]

        dcm_files = sorted(
            [
                f.path
                for f in os.scandir(t2_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        if len(dcm_files) == 0:
            raise FileNotFoundError(f"No DICOM files found in {t2_dir}")

        chosen = []
        for fp in dcm_files:
            arr = _read_dicom_pixels(fp)
            if arr.sum() > 100000:
                arr_r = resize(
                    arr,
                    (img_px_size, img_px_size),
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.float32)
                arr_n = _normalize01(arr_r)
                stacked = np.stack([arr_n, arr_n, arr_n], axis=-1)  # (H,W,3)
                if stacked.sum() > 2000:
                    chosen.append(stacked)
            if len(chosen) >= n_slices:
                break

        if len(chosen) == 0:
            blank = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            chosen = [blank] * n_slices
        elif len(chosen) < n_slices:
            chosen = chosen + [chosen[-1]] * (n_slices - len(chosen))

        for j in range(n_slices):
            pixels_by_slice[j].append(chosen[j])

    pixels_by_slice = [np.asarray(lst, dtype=np.float32) for lst in pixels_by_slice]

    print(
        "Loaded T2w slices per index:",
        [x.shape[0] for x in pixels_by_slice],
        "cases total:",
        pixels_by_slice[0].shape[0],
    )
    return pixels_by_slice




## === cell 3
pixels_slices = load_test_T2W_images(TEST_DIR, img_px_size=150, n_slices=11)
(
    pixels_1,
    pixels_2,
    pixels_3,
    pixels_4,
    pixels_5,
    pixels_6,
    pixels_7,
    pixels_8,
    pixels_9,
    pixels_10,
    pixels_11,
) = pixels_slices



## === cell 4
n_cases = pixels_1.shape[0]
prediction_1 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_2 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_3 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_4 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_5 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_6 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_7 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_8 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_9 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_10 = np.full((n_cases,), 0.5, dtype=np.float32)
prediction_11 = np.full((n_cases,), 0.5, dtype=np.float32)




## === cell 5
def create_sub_from_predictions(
    case_ids_zeropad5, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11
):
    """
    Create submission dataframe with one prediction per case, averaged over 11 slice predictions.
    """
    preds = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p7.astype(np.float32)
        + p8.astype(np.float32)
        + p9.astype(np.float32)
        + p10.astype(np.float32)
        + p11.astype(np.float32)
    ) / 11.0

    df = pd.DataFrame(
        {
            "BraTS21ID": [int(x) for x in case_ids_zeropad5],
            "MGMT_value": np.clip(preds, 0.0, 1.0),
        }
    )
    return df


sub_df = create_sub_from_predictions(
    test_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
    prediction_8,
    prediction_9,
    prediction_10,
    prediction_11,
)



## === cell 6
print(sub_df.head())
print("Shape:", sub_df.shape)
print(sub_df["MGMT_value"].describe())



## === cell 7
sub_df_sorted = sub_df.copy()
sub_df_sorted["BraTS21ID"] = sub_df_sorted["BraTS21ID"].astype(int)
sub_df_sorted = (
    sub_df_sorted.set_index("BraTS21ID")
    .reindex(sample_sub["BraTS21ID"].astype(int).values)
    .reset_index()
)

assert list(sub_df_sorted.columns) == ["BraTS21ID", "MGMT_value"]
assert sub_df_sorted.shape[0] == sample_sub.shape[0]
assert sub_df_sorted["MGMT_value"].notna().all()



## === cell 8
sub_df_sorted.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df_sorted.shape)
print(sub_df_sorted.head())
