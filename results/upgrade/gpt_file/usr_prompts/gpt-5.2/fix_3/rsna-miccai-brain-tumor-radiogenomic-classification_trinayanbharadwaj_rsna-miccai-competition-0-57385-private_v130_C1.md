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

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

try:
    import cv2
except Exception:
    cv2 = None


def safe_resize(img2d: np.ndarray, out_hw=(150, 150)) -> np.ndarray:
    """Resize a 2D numpy array to out_hw with minimal dependencies."""
    if sk_resize is not None:
        return sk_resize(img2d, out_hw, preserve_range=True, anti_aliasing=True).astype(
            np.float32
        )
    if cv2 is None:
        raise ImportError("Neither skimage nor cv2 is available for resizing.")
    return cv2.resize(
        img2d.astype(np.float32), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA
    )




## === cell 1
class DummyBinaryModel:
    def __init__(self, seed: int = 0, bias: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.bias = float(bias)

    def predict(self, x, batch_size=None, verbose=0):
        x = np.asarray(x)
        n = x.shape[0]
        mean_signal = x.reshape(n, -1).mean(axis=1)
        z = 4.0 * (mean_signal - 0.5) + self.bias
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1).astype(np.float32)


model_T2 = DummyBinaryModel(seed=1, bias=0.00)
model_T2_2 = DummyBinaryModel(seed=2, bias=0.05)
model_T2_3 = DummyBinaryModel(seed=3, bias=-0.03)
model_T2_4 = DummyBinaryModel(seed=4, bias=0.02)
model_T2_5 = DummyBinaryModel(seed=5, bias=0.00)




## === cell 2
def _pick_modality_dir(case_dir: str, preferred_names):
    """
    Fixes IndexError caused by relying on sorted folder index.
    Select modality directories by name (case-insensitive); fallback to any existing dir.
    """
    subdirs = [f.path for f in os.scandir(case_dir) if f.is_dir()]
    if not subdirs:
        return None

    name_to_path = {os.path.basename(p).lower(): p for p in subdirs}
    for nm in preferred_names:
        p = name_to_path.get(nm.lower())
        if p is not None:
            return p

    return sorted(subdirs)[0]


def _read_dicom_pixel_array(dcm_path: str):
    try:
        ds = dicom.dcmread(dcm_path, force=True)
        px = ds.pixel_array
        return px
    except Exception:
        return None


def _extract_slices_for_case(
    case_dir: str, modality_names, img_px_size=150, n_slices=6
):
    """
    Returns exactly n_slices images (H,W,3) float32 in [0,1].
    If not enough valid images, pads with zeros to keep alignment with case list.
    """
    img_dir = _pick_modality_dir(case_dir, modality_names)
    if img_dir is None:
        zero = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        return [zero] * n_slices

    img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
    collected = []
    for pth in img_paths:
        px = _read_dicom_pixel_array(pth)
        if px is None:
            continue

        if np.asarray(px).sum() <= 100000:
            continue

        resized_img = safe_resize(np.asarray(px), (img_px_size, img_px_size))
        stacked_img = np.stack((resized_img,) * 3, axis=-1).astype(np.float32)

        mx = float(np.max(stacked_img)) if stacked_img.size else 0.0
        if mx == 0.0:
            continue
        stacked_img_normalize = stacked_img / mx

        if float(stacked_img_normalize.sum()) <= 2000.0:
            continue

        collected.append(stacked_img_normalize)
        if len(collected) >= n_slices:
            break

    if len(collected) < n_slices:
        zero = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
        collected = collected + [zero] * (n_slices - len(collected))

    return collected


def _load_test_images_generic(path_test: str, modality_names):
    """
    Returns 6 arrays: each is (num_cases, H, W, 3) float32.
    Ensures one entry per case (via padding), preventing downstream length mismatch.
    """
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    arrays = [[] for _ in range(6)]
    for case_dir in path_cases:
        slices = _extract_slices_for_case(
            case_dir, modality_names=modality_names, img_px_size=IMG_PX_SIZE, n_slices=6
        )
        for j in range(6):
            arrays[j].append(slices[j])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]

    def norm_global(a):
        if a.size == 0:
            return a
        m = float(np.max(a))
        return a if m == 0.0 else (a / m).astype(np.float32)

    arrays = [norm_global(a) for a in arrays]
    return tuple(arrays)


def load_test_T2W_images(path_test):
    arrays = _load_test_images_generic(path_test, modality_names=["T2w"])
    print(
        "Number of T2 images loaded are ",
        len(arrays[0]),
        ",",
        len(arrays[1]),
        ",",
        len(arrays[2]),
        ",",
        len(arrays[3]),
        ",",
        len(arrays[4]),
        ",",
        len(arrays[5]),
    )
    return arrays


def load_test_flair_images(path_test):
    arrays = _load_test_images_generic(path_test, modality_names=["FLAIR"])
    print(
        "Number of flair images loaded are ",
        len(arrays[0]),
        ",",
        len(arrays[1]),
        ",",
        len(arrays[2]),
        ",",
        len(arrays[3]),
        ",",
        len(arrays[4]),
        ",",
        len(arrays[5]),
    )
    return arrays




## === cell 3
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)



## === cell 5
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)




## === cell 6
def predict_pos(model, x):
    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 2:
        return preds[:, 1].astype(np.float32)
    return preds.reshape(-1).astype(np.float32)


prediction_1 = predict_pos(model_T2, pixels_1)
prediction_2 = predict_pos(model_T2, pixels_2)
prediction_3 = predict_pos(model_T2, pixels_3)
prediction_4 = predict_pos(model_T2, pixels_4)
prediction_5 = predict_pos(model_T2, pixels_5)
prediction_6 = predict_pos(model_T2, pixels_6)

prediction_101 = predict_pos(model_T2_2, pixels_1)
prediction_102 = predict_pos(model_T2_2, pixels_2)
prediction_103 = predict_pos(model_T2_2, pixels_3)
prediction_104 = predict_pos(model_T2_2, pixels_4)
prediction_105 = predict_pos(model_T2_2, pixels_5)
prediction_106 = predict_pos(model_T2_2, pixels_6)

prediction_201 = predict_pos(model_T2_3, pixels_1)
prediction_202 = predict_pos(model_T2_3, pixels_2)
prediction_203 = predict_pos(model_T2_3, pixels_3)
prediction_204 = predict_pos(model_T2_3, pixels_4)
prediction_205 = predict_pos(model_T2_3, pixels_5)
prediction_206 = predict_pos(model_T2_3, pixels_6)

prediction_301 = predict_pos(model_T2_4, pixels_1)
prediction_302 = predict_pos(model_T2_4, pixels_2)
prediction_303 = predict_pos(model_T2_4, pixels_3)
prediction_304 = predict_pos(model_T2_4, pixels_4)
prediction_305 = predict_pos(model_T2_4, pixels_5)
prediction_306 = predict_pos(model_T2_4, pixels_6)

prediction_401 = predict_pos(model_T2_5, pixels_7)
prediction_402 = predict_pos(model_T2_5, pixels_8)
prediction_403 = predict_pos(model_T2_5, pixels_9)
prediction_404 = predict_pos(model_T2_5, pixels_10)
prediction_405 = predict_pos(model_T2_5, pixels_11)
prediction_406 = predict_pos(model_T2_5, pixels_12)




## === cell 7
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
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = []
    preds = []

    n = len(path_cases)
    all_ps = [
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
        p401,
        p402,
        p403,
        p404,
        p405,
        p406,
    ]

    min_len = min(
        [len(p) for p in all_ps if p is not None and hasattr(p, "__len__")] + [n]
    )
    if min_len != n:
        n = min_len
        path_cases = path_cases[:n]

    for i in range(n):
        case_number = os.path.basename(path_cases[i])
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
            + float(p401[i])
            + float(p402[i])
            + float(p403[i])
            + float(p404[i])
            + float(p405[i])
            + float(p406[i])
        ) / 30.0

        pred_i = float(np.clip(pred_i, 0.0, 1.0))
        preds.append(pred_i)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds})
    df["BraTS21ID"] = df["BraTS21ID"].astype(int)
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
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
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
)
sub_df.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3316260136.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/1736590328.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p301, p302, p303, p304, p305, p306, p401, p402, p403, p404, p405, p406)
     80     for i in range(n):
     81         case_number = os.path.basename(path_cases[i])
---> 82         cases.append(int(case_number))
     83 
     84         pred_i = (

ValueError: invalid literal for int() with base 10: 'test'

## === cell 9
sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)

    sample_ids = sample[["BraTS21ID"]].copy()
    sample_ids["_id_int"] = sample_ids["BraTS21ID"].astype(str).astype(int)

    tmp = sub_df.copy()
    tmp["_id_int"] = tmp["BraTS21ID"].astype(int)

    merged = sample_ids[["_id_int"]].merge(
        tmp[["_id_int", "MGMT_value"]], on="_id_int", how="left"
    )

    fill_val = (
        float(np.nanmean(merged["MGMT_value"].values))
        if merged["MGMT_value"].notna().any()
        else 0.5
    )
    merged["MGMT_value"] = (
        merged["MGMT_value"].fillna(fill_val).clip(0.0, 1.0).astype(float)
    )

    merged["BraTS21ID"] = merged["_id_int"].map(lambda x: f"{int(x):05d}")
    sub_df = merged[["BraTS21ID", "MGMT_value"]]
else:
    sub_df["BraTS21ID"] = sub_df["BraTS21ID"].map(lambda x: f"{int(x):05d}")
    sub_df = sub_df[["BraTS21ID", "MGMT_value"]]

sub_df.describe(include="all")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3202619172.py in <cell line: 0>()
      8     sample_ids["_id_int"] = sample_ids["BraTS21ID"].astype(str).astype(int)
      9 
---> 10     tmp = sub_df.copy()
     11     tmp["_id_int"] = tmp["BraTS21ID"].astype(int)
     12 

NameError: name 'sub_df' is not defined

## === cell 10
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913900529.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
