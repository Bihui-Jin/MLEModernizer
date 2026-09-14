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

0.45882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48941) has done: 'I fix the immediate runtime issues by removing/import-guarding problematic optional packages (which trigger the protobuf `MessageFactory` error) and by ensuring `resize` is always available inside the image-loading functions. Because the referenced pretrained `.h5` models are not present in your `/kaggle/input` (causing `FileNotFoundError`), I keep the overall “load images → model predicts probabilities → write submission.csv” pipeline but replace missing model loads with a lightweight Keras fallback model so the notebook runs end-to-end and outputs a valid submission. I also fix the submission creation logic bug where it was averaging full prediction arrays instead of producing one prediction per case (and ensure alignment with `BraTS21ID` from `sample_submission.csv`). These changes are necessary for correctness and to yield a non-trivial AUC vs. a constant baseline, without adding new external dependencies or changing the data paths.'
- What this solution (achieved 0.45882) has done: 'I fix the TensorFlow/protobuf import crash by avoiding the `tensorflow` dependency entirely (it’s the root cause of the `MessageFactory.GetPrototype` error in this environment). To preserve the end-to-end pipeline and submission semantics with minimal changes, I replace the Keras fallback models with a deterministic, lightweight, non-TF fallback “model” that outputs probabilities from simple image statistics. I also ensure predictions always align to the `sample_submission.csv` ordering by iterating cases in that ID order (instead of filesystem scan order, which can silently misalign). These changes are primarily stability/correctness fixes; they should also improve AUC versus the current miscalibrated/mostly-random fallback predictions.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

import pydicom as dicom

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

import cv2

np.random.seed(42)

print("Imports OK. Using non-TensorFlow fallback to avoid protobuf/TensorFlow crash.")



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")
labels_csv = os.path.join(DATA_ROOT, "train_labels.csv")
sample_csv = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(test_dir), f"Missing test dir: {test_dir}"
assert os.path.isfile(sample_csv), f"Missing sample submission: {sample_csv}"

sample_sub = pd.read_csv(sample_csv)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
print(sample_sub.head())
print("n_test:", len(sample_sub))




## === cell 2
def _resize_2d(img2d: np.ndarray, out_size: int) -> np.ndarray:
    if sk_resize is not None:
        return sk_resize(
            img2d, (out_size, out_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    return cv2.resize(
        img2d.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_AREA
    )


def _to_3ch(img2d: np.ndarray) -> np.ndarray:
    return np.stack([img2d, img2d, img2d], axis=-1).astype(np.float32)


def _collect_case_slices(
    case_path: str, modality_folder_name: str, img_px_size: int, max_slices: int = 6
):
    """Return up to max_slices slices for a given case+modality, filtered similarly to original logic."""
    modality_path = os.path.join(case_path, modality_folder_name)
    if not os.path.isdir(modality_path):
        return []

    img_files = sorted(
        [
            f.path
            for f in os.scandir(modality_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    out = []
    for fp in img_files:
        img2d = dicom.dcmread(fp).pixel_array
        if img2d.sum() <= 100000:
            continue
        img2d = _resize_2d(img2d, img_px_size)
        img3 = _to_3ch(img2d)
        maxv = float(np.max(img3)) if np.max(img3) != 0 else 1.0
        img3 = img3 / maxv
        if img3.sum() <= 2000:
            continue
        out.append(img3.astype(np.float32))
        if len(out) >= max_slices:
            break
    return out




## === cell 3
def _case_paths_in_sample_order(base_test_dir: str, sample_ids):
    paths = []
    for sid in sample_ids:
        p = os.path.join(base_test_dir, str(sid))
        if os.path.isdir(p):
            paths.append(p)
        else:
            p2 = os.path.join(base_test_dir, str(int(sid)))
            paths.append(p2)
    return paths


def load_test_T2W_images(path_test):
    IMG_PX_SIZE = 150
    arrays = [[] for _ in range(6)]
    path_cases = _case_paths_in_sample_order(
        path_test, sample_sub["BraTS21ID"].tolist()
    )
    for case_path in path_cases:
        slices = _collect_case_slices(case_path, "T2w", IMG_PX_SIZE, max_slices=6)
        for j in range(min(6, len(slices))):
            arrays[j].append(slices[j])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of T2 images loaded are", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)


def load_test_flair_images(path_test):
    IMG_PX_SIZE = 150
    arrays = [[] for _ in range(6)]
    path_cases = _case_paths_in_sample_order(
        path_test, sample_sub["BraTS21ID"].tolist()
    )
    for case_path in path_cases:
        slices = _collect_case_slices(case_path, "FLAIR", IMG_PX_SIZE, max_slices=6)
        for j in range(min(6, len(slices))):
            arrays[j].append(slices[j])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of flair images loaded are", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)


def load_test_T1wce_images(path_test):
    IMG_PX_SIZE = 150
    arrays = [[] for _ in range(6)]
    path_cases = _case_paths_in_sample_order(
        path_test, sample_sub["BraTS21ID"].tolist()
    )
    for case_path in path_cases:
        slices = _collect_case_slices(case_path, "T1wCE", IMG_PX_SIZE, max_slices=6)
        for j in range(min(6, len(slices))):
            arrays[j].append(slices[j])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of T1wce images loaded are", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 4


class SimpleStatModel:
    """
    Deterministic lightweight "model" that maps image stats to a probability.
    Returns Nx2 softmax-like outputs to preserve original prediction API.
    """

    def __init__(self, w_mean=2.0, w_std=1.0, bias=-1.0, noise=0.0, seed=42):
        self.w_mean = float(w_mean)
        self.w_std = float(w_std)
        self.bias = float(bias)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, x, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input (N,H,W,C), got shape {x.shape}")
        mean = x.mean(axis=(1, 2, 3))
        std = x.std(axis=(1, 2, 3))
        z = self.w_mean * mean + self.w_std * std + self.bias
        if self.noise > 0:
            z = z + self.rng.normal(0.0, self.noise, size=z.shape).astype(np.float32)
        p1 = self._sigmoid(z).astype(np.float32)
        p0 = (1.0 - p1).astype(np.float32)
        return np.stack([p0, p1], axis=1)


def safe_load_model(path):
    return None


model_paths = [
    "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_20_b600_t1wce_7k_0.73auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
    "../input/trained-model-for-rsnamiccai/rsna_miccai_25_b600_t1wce_7k_0.78auc_imgs.h5",
]

loaded = [safe_load_model(p) for p in model_paths]
for i in range(len(loaded)):
    if loaded[i] is None:
        loaded[i] = SimpleStatModel(
            w_mean=1.5 + 0.15 * i,
            w_std=0.8 + 0.05 * i,
            bias=-0.9,
            noise=0.0,
            seed=42 + i,
        )

(
    model_T2,
    model_T2_2,
    model_T2_3,
    model_T2_4,
    model_T2_5,
    model_T2_6,
    model_T2_7,
    model_T2_8,
) = loaded

print("Models ready (SimpleStatModel fallback; TensorFlow avoided).")



## === cell 5
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test_dir
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test_dir
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(test_dir)
)

n_cases = len(sample_sub)
print("Expected n_cases:", n_cases)




## === cell 6
def _predict_col(model, x):
    if x is None or len(x) == 0:
        return np.array([], dtype=np.float32)
    preds = model.predict(x, verbose=0)
    return preds[:, 1].astype(np.float32)


prediction_1 = _predict_col(model_T2, pixels_1)
prediction_2 = _predict_col(model_T2, pixels_2)
prediction_3 = _predict_col(model_T2, pixels_3)
prediction_4 = _predict_col(model_T2, pixels_4)
prediction_5 = _predict_col(model_T2, pixels_5)
prediction_6 = _predict_col(model_T2, pixels_6)

prediction_101 = _predict_col(model_T2_2, pixels_1)
prediction_102 = _predict_col(model_T2_2, pixels_2)
prediction_103 = _predict_col(model_T2_2, pixels_3)
prediction_104 = _predict_col(model_T2_2, pixels_4)
prediction_105 = _predict_col(model_T2_2, pixels_5)
prediction_106 = _predict_col(model_T2_2, pixels_6)

prediction_201 = _predict_col(model_T2_3, pixels_7)
prediction_202 = _predict_col(model_T2_3, pixels_8)
prediction_203 = _predict_col(model_T2_3, pixels_9)
prediction_204 = _predict_col(model_T2_3, pixels_10)
prediction_205 = _predict_col(model_T2_3, pixels_11)
prediction_206 = _predict_col(model_T2_3, pixels_12)

prediction_301 = _predict_col(model_T2_4, pixels_13)
prediction_302 = _predict_col(model_T2_4, pixels_14)
prediction_303 = _predict_col(model_T2_4, pixels_15)
prediction_304 = _predict_col(model_T2_4, pixels_16)
prediction_305 = _predict_col(model_T2_4, pixels_17)
prediction_306 = _predict_col(model_T2_4, pixels_18)

prediction_401 = _predict_col(model_T2_5, pixels_1)
prediction_402 = _predict_col(model_T2_5, pixels_2)
prediction_403 = _predict_col(model_T2_5, pixels_3)
prediction_404 = _predict_col(model_T2_5, pixels_4)
prediction_405 = _predict_col(model_T2_5, pixels_5)
prediction_406 = _predict_col(model_T2_5, pixels_6)

prediction_501 = _predict_col(model_T2_6, pixels_1)
prediction_502 = _predict_col(model_T2_6, pixels_2)
prediction_503 = _predict_col(model_T2_6, pixels_3)
prediction_504 = _predict_col(model_T2_6, pixels_4)
prediction_505 = _predict_col(model_T2_6, pixels_5)
prediction_506 = _predict_col(model_T2_6, pixels_6)

prediction_601 = _predict_col(model_T2_7, pixels_7)
prediction_602 = _predict_col(model_T2_7, pixels_8)
prediction_603 = _predict_col(model_T2_7, pixels_9)
prediction_604 = _predict_col(model_T2_7, pixels_10)
prediction_605 = _predict_col(model_T2_7, pixels_11)
prediction_606 = _predict_col(model_T2_7, pixels_12)

prediction_701 = _predict_col(model_T2_8, pixels_13)
prediction_702 = _predict_col(model_T2_8, pixels_14)
prediction_703 = _predict_col(model_T2_8, pixels_15)
prediction_704 = _predict_col(model_T2_8, pixels_16)
prediction_705 = _predict_col(model_T2_8, pixels_17)
prediction_706 = _predict_col(model_T2_8, pixels_18)

print(
    "Example prediction lengths:",
    len(prediction_1),
    len(prediction_201),
    len(prediction_301),
)




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
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
    p701,
    p702,
    p703,
    p704,
    p705,
    p706,
):
    sub = pd.read_csv(sample_csv)
    sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

    preds_list = [
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
        p501,
        p502,
        p503,
        p504,
        p505,
        p506,
        p601,
        p602,
        p603,
        p604,
        p605,
        p606,
        p701,
        p702,
        p703,
        p704,
        p705,
        p706,
    ]

    n = len(sub)
    mat = np.full((len(preds_list), n), np.nan, dtype=np.float32)
    for i, p in enumerate(preds_list):
        p = np.asarray(p, dtype=np.float32).reshape(-1)
        m = min(len(p), n)
        if m > 0:
            mat[i, :m] = p[:m]

    pred = np.nanmean(mat, axis=0)
    pred = np.where(np.isnan(pred), 0.5, pred)
    pred = np.clip(pred, 1e-6, 1 - 1e-6)

    out = pd.DataFrame(
        {"BraTS21ID": sub["BraTS21ID"], "MGMT_value": pred.astype(float)}
    )
    return out




## === cell 8
sub_df = create_sub(
    test_dir,
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
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
    prediction_701,
    prediction_702,
    prediction_703,
    prediction_704,
    prediction_705,
    prediction_706,
)

print(sub_df.head())
print(sub_df.shape)
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(sample_sub)
assert (
    sub_df["BraTS21ID"].values == sample_sub["BraTS21ID"].values
).all(), "ID order mismatch vs sample_submission"



## === cell 9
if sns is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
    except Exception as e:
        print("Plot skipped:", repr(e))

print("MGMT_value stats:", sub_df["MGMT_value"].describe())



## === cell 10
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(sub_df), "rows")
print("Saved at:", os.path.abspath("submission.csv"))
