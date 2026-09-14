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

0.40706

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.37176) has done: 'I fix the immediate runtime/import issues by removing the problematic TensorFlow/Keras dependency (which triggers the `MessageFactory.GetPrototype` crash in this environment) and the missing pretrained model file reference. To preserve the end-to-end pipeline and produce a valid `submission.csv`, I keep your test-case enumeration logic but replace the unavailable model predictions with a stable, deterministic baseline probability derived from a simple image-intensity feature extracted from the same T2w slices you already load. I also fix logic bugs in the image loaders (list/array division, `resize` scoping) and make the submission IDs match Kaggle’s expected 5-digit `BraTS21ID` strings. This run within the Kaggle constraints and generate a correctly formatted submission file.'
- What this solution (achieved 0.40706) has done: 'I fix the `ValueError` in `create_sub` by ensuring we only enumerate actual case folders (numeric IDs) inside the `test/` directory and ignore any stray non-numeric directories like an extra nested `test` folder. I also remove the brittle “min_len alignment” that can silently drop many test IDs due to slice-loading variability; instead we compute one probability per case and keep the submission rows exactly aligned to all test cases. To preserve the existing core logic (same slice loading + same `_slice_feature`), I aggregate available slice-features per case (mean over up to 6 slices) and default to 0.5 if a case yields no valid slices. Finally, the script always write a valid `submission.csv` with correct columns and 5-digit `BraTS21ID`.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

np.random.seed(0)




## === cell 1
def load_test_T2W_images(path_test):
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])

        t2w_dir = None
        for d in mri_type:
            if os.path.basename(d).lower() == "t2w":
                t2w_dir = d
                break
        if t2w_dir is None:
            continue

        img_path = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])

        for k in range(len(img_path)):
            img = dicom.dcmread(img_path[k])
            px = img.pixel_array

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img_arr = np.array(resized_img, dtype=np.float32)

                stacked_img = np.stack((img_arr,) * 3, axis=-1)
                mx = float(np.max(stacked_img))
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
                        break

    def _to_norm(a_list):
        arr = np.asarray(a_list, dtype=np.float32)
        if arr.size == 0:
            return arr
        mx = float(np.max(arr))
        if mx > 0:
            arr = arr / mx
        return arr

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)

    print(
        "Number of T2 images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )

    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 2
def load_test_T1w_images(path_test):
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])
        if len(mri_type) < 2:
            continue

        img_path = sorted([f.path for f in os.scandir(mri_type[1]) if f.is_file()])

        for k in range(len(img_path)):
            img = dicom.dcmread(img_path[k])
            px = img.pixel_array

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img_arr = np.array(resized_img, dtype=np.float32)

                stacked_img = np.stack((img_arr,) * 3, axis=-1)
                mx = float(np.max(stacked_img))
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
                        break

    def _to_norm(a_list):
        arr = np.asarray(a_list, dtype=np.float32)
        if arr.size == 0:
            return arr
        mx = float(np.max(arr))
        if mx > 0:
            arr = arr / mx
        return arr

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)

    print(
        "Number of T1w images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )

    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"




## === cell 4
def _slice_feature(pixels):
    """
    pixels: (N, H, W, 3) float32 in [0,1] approximately.
    Returns: (N,) feature in [0,1].
    """
    if pixels is None or getattr(pixels, "size", 0) == 0:
        return np.zeros((0,), dtype=np.float32)
    x = pixels[..., 0]
    flat = x.reshape((x.shape[0], -1))
    k = max(1, int(flat.shape[1] * 0.05))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    feat = part.mean(axis=1).astype(np.float32)
    z = (feat - np.median(feat)) / (np.std(feat) + 1e-6)
    prob = 1.0 / (1.0 + np.exp(-z))
    return prob.astype(np.float32)




## === cell 5
def _list_case_dirs(path_test):
    case_dirs = []
    for f in os.scandir(path_test):
        if not f.is_dir():
            continue
        name = os.path.basename(f.path)
        if re.fullmatch(r"\d+", name):
            case_dirs.append(f.path)
    case_dirs = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
    return case_dirs


def _load_case_t2w_slices(case_dir, max_slices=6, IMG_PX_SIZE=150):
    t2w_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2w_dir):
        for d in os.scandir(case_dir):
            if d.is_dir() and os.path.basename(d.path).lower() == "t2w":
                t2w_dir = d.path
                break
    if not os.path.isdir(t2w_dir):
        return []

    img_paths = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])
    out = []
    for p in img_paths:
        try:
            img = dicom.dcmread(p)
            px = img.pixel_array
        except Exception:
            continue

        if px.sum() > 100000:
            resized_img = resize(
                px,
                (IMG_PX_SIZE, IMG_PX_SIZE),
                preserve_range=True,
                anti_aliasing=True,
            )
            img_arr = np.array(resized_img, dtype=np.float32)

            stacked_img = np.stack((img_arr,) * 3, axis=-1)
            mx = float(np.max(stacked_img))
            if mx <= 0:
                continue
            stacked_img_normalize = stacked_img / mx

            if stacked_img_normalize.sum() > 2000:
                out.append(stacked_img_normalize.astype(np.float32))
                if len(out) >= max_slices:
                    break
    return out


case_dirs = _list_case_dirs(test)
print("Detected test case folders:", len(case_dirs))
print("First 5 case IDs:", [os.path.basename(p) for p in case_dirs[:5]])



## === cell 6
ids = []
preds = []

for case_dir in case_dirs:
    case_id = f"{int(os.path.basename(case_dir)):05d}"
    slices = _load_case_t2w_slices(case_dir, max_slices=6, IMG_PX_SIZE=150)

    if len(slices) == 0:
        prob = 0.5
    else:
        pixels = np.asarray(slices, dtype=np.float32)
        slice_probs = _slice_feature(pixels)
        prob = float(np.mean(slice_probs)) if slice_probs.size else 0.5

    ids.append(case_id)
    preds.append(prob)

sub_df = pd.DataFrame(
    {
        "BraTS21ID": ids,
        "MGMT_value": np.clip(np.array(preds, dtype=np.float32), 0.0, 1.0),
    }
)
sub_df.head(), sub_df.shape



## === cell 7
print(sub_df["MGMT_value"].describe())
print("Nulls:", sub_df.isna().sum().to_dict())

sample_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    print("Sample rows:", sample.shape[0], "Our rows:", sub_df.shape[0])
    print(
        "ID overlap:",
        len(
            set(sample["BraTS21ID"].astype(str)).intersection(
                set(sub_df["BraTS21ID"].astype(str))
            )
        ),
    )



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print("First rows:\n", sub_df.head(3).to_string(index=False))
print("Last rows:\n", sub_df.tail(3).to_string(index=False))
