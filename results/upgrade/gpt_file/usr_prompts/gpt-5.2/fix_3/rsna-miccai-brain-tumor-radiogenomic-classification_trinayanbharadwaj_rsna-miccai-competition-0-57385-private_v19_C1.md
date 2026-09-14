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

0.50588

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove/replace imports that trigger the protobuf `MessageFactory.GetPrototype` crash and unnecessary heavy dependencies, keeping only what’s needed for inference. Since the referenced pretrained `.h5` model file is not available in your environment, I keep the same overall “load model → predict” semantics by falling back to a tiny Keras model if loading fails, so the notebook always runs end-to-end and produces a valid `submission.csv`. I also fix the `resize` `NameError` by switching to a lightweight OpenCV resize helper (with a safe NumPy fallback) and ensure pixel normalization is correct. Finally, I make the submission IDs zero-padded to 5 digits to match the competition format and align prediction length to the number of test cases.'
- What this solution (achieved 0.50588) has done: 'I fix the immediate crash caused by importing TensorFlow in this Kaggle environment (the protobuf `MessageFactory.GetPrototype` error) by removing the TensorFlow/Keras dependency entirely. To preserve the core “load model → predict probabilities” semantics and keep the pipeline end-to-end, I replace the unavailable `.h5` model with a deterministic, lightweight NumPy-based predictor that outputs valid probabilities for each test case. I also make sure the number/order of predictions matches the sorted test folder IDs and that the submission file is written as `submission.csv` with the exact required columns. These changes are focused on unblocking execution and producing a valid submission reliably.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import cv2
import pydicom as dicom

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)



## === cell 1
MODEL_PATH = "../input/trained-model-for-rsnamiccai/model_rsna_miccai_100epochs.h5"


class NumpyBaselineModel:
    """
    Deterministic probability predictor using simple intensity statistics.
    Keeps the same high-level inference semantics: model.predict(batch) -> (N, 1).
    """

    def predict(self, x, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected input shape (N, H, W, 3), got {x.shape}")

        g = x.mean(axis=-1)

        mean = g.mean(axis=(1, 2))
        std = g.std(axis=(1, 2))
        p90 = np.quantile(g.reshape(g.shape[0], -1), 0.90, axis=1)

        z = (1.8 * (p90 - 0.35)) + (0.9 * (mean - 0.25)) + (0.6 * (std - 0.15))
        prob = 1.0 / (1.0 + np.exp(-z))
        return prob.reshape(-1, 1).astype(np.float32)


def get_model():
    return NumpyBaselineModel()


model = get_model()
print("Using model:", type(model).__name__)
print("MODEL_PATH exists:", os.path.exists(MODEL_PATH))




## === cell 2
def _resize_img(img2d: np.ndarray, out_size: int = 299) -> np.ndarray:
    """Resize 2D image to (out_size, out_size) as float32."""
    img2d = img2d.astype(np.float32)
    try:
        return cv2.resize(
            img2d, (out_size, out_size), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
    except Exception:
        h, w = img2d.shape[:2]
        y_idx = (np.linspace(0, h - 1, out_size)).astype(np.int32)
        x_idx = (np.linspace(0, w - 1, out_size)).astype(np.int32)
        return img2d[np.ix_(y_idx, x_idx)].astype(np.float32)


def load_test_images(path_test):
    array = []
    count = 0
    IMG_PX_SIZE = 299

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if not mri_type:
            continue

        img_path = sorted(
            [
                f.path
                for f in os.scandir(mri_type[0])
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        if not img_path:
            continue

        found = False
        for p in img_path:
            try:
                img = dicom.dcmread(p)
                px = img.pixel_array
            except Exception:
                continue

            if np.asarray(px).sum() > 100000:
                resized_img = _resize_img(np.asarray(px), IMG_PX_SIZE)
                array.append(resized_img)
                count += 1
                found = True
                break

        if not found:
            mid = img_path[len(img_path) // 2]
            try:
                img = dicom.dcmread(mid)
                px = img.pixel_array
                resized_img = _resize_img(np.asarray(px), IMG_PX_SIZE)
                array.append(resized_img)
                count += 1
            except Exception:
                array.append(np.zeros((IMG_PX_SIZE, IMG_PX_SIZE), dtype=np.float32))
                count += 1

    array = np.asarray(array, dtype=np.float32)
    maxv = float(np.max(array)) if array.size else 1.0
    if maxv > 0:
        array = array / maxv

    print("Number of images loaded are ", count)
    return array




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
pixels = load_test_images(test)
pixels.shape



## === cell 4
pixels_reshape = pixels.reshape(len(pixels), 299, 299)
rgb_batch_test = np.repeat(pixels_reshape[..., np.newaxis], 3, axis=-1).astype(
    np.float32
)
rgb_batch_test.shape



## === cell 5
preds = model.predict(rgb_batch_test, verbose=0)

preds = np.asarray(preds)
if preds.ndim == 2 and preds.shape[1] == 1:
    prediction = preds[:, 0]
elif preds.ndim == 2 and preds.shape[1] > 1:
    prediction = np.max(preds, axis=1)
else:
    prediction = preds.reshape(-1)

prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)
prediction[:5], prediction.shape




## === cell 6
def create_sub(path_test, prediction):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [
        os.path.basename(p) for p in path_cases
    ]  # already zero-padded like '00002'

    n = min(len(cases), len(prediction))
    cases = cases[:n]
    pred = prediction[:n].astype(float)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": pred})
    return df


sub_df = create_sub(test, prediction)
sub_df.head(), sub_df.shape



## === cell 7
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
