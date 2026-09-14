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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
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
MODEL_DIR = "../input/trained-model-for-rsnamiccai"

MODEL_PATHS = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_10_b800_t1wce_7k_0.64auc_imgs.h5",
    "rsna_miccai_10_b800_t1wce_7k_0.65auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
]


def _safe_load_model(path):
    try:
        if os.path.exists(path):
            return keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"Warning: failed to load model {path}: {e}")
    return None


models = []
for fname in MODEL_PATHS:
    models.append(_safe_load_model(os.path.join(MODEL_DIR, fname)))

use_pretrained = all(m is not None for m in models)
if use_pretrained:
    (
        model_T2,
        model_T2_2,
        model_T2_3,
        model_T2_4,
        model_T2_5,
        model_T2_6,
        model_T2_7,
    ) = models
    print("All pretrained models loaded successfully.")
else:
    print(
        "Pretrained model files not found/failed to load. Using deterministic fallback predictor."
    )




## === cell 2
def _to_float32_image(img2d, img_px_size=150):
    """Resize 2D to (img_px_size,img_px_size), stack to 3ch, normalize to [0,1]."""
    resized_img = resize(
        img2d, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    stacked = np.stack([resized_img, resized_img, resized_img], axis=-1)
    mx = float(np.max(stacked))
    if mx > 0:
        stacked = stacked / mx
    return stacked


def _get_series_dir(subject_dir, series_name):
    p = os.path.join(subject_dir, series_name)
    return p if os.path.isdir(p) else None


def _read_best_slices(
    series_dir, n_slices=6, img_px_size=150, sum_thr=100000, normsum_thr=2000
):
    """
    Reads DICOMs in series_dir, returns up to n_slices normalized RGB images (H,W,3).
    Selection logic preserved from original: pixel sum threshold + normalized sum threshold.
    If fewer found, pads by repeating the last found slice or zeros.
    """
    if series_dir is None:
        imgs = []
    else:
        files = sorted([f.path for f in os.scandir(series_dir) if f.is_file()])
        imgs = []
        for fp in files:
            try:
                d = dicom.dcmread(fp, force=True)
                arr = d.pixel_array
            except Exception:
                continue
            if arr is None:
                continue
            if float(np.sum(arr)) > sum_thr:
                im = _to_float32_image(arr, img_px_size=img_px_size)
                if float(np.sum(im)) > normsum_thr:
                    imgs.append(im)
                    if len(imgs) >= n_slices:
                        break

    if len(imgs) == 0:
        imgs = [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)]
    while len(imgs) < n_slices:
        imgs.append(imgs[-1].copy())
    return np.stack(imgs[:n_slices], axis=0)


def load_test_images_by_modality(path_test, modality, n_slices=6, img_px_size=150):
    """
    Returns list of per-slice batches: [slice0_batch, slice1_batch, ... slice5_batch]
    where each slice batch is (N, H, W, 3).
    """
    subject_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    per_slice = [[] for _ in range(n_slices)]

    for subj in subject_dirs:
        series_dir = _get_series_dir(subj, modality)
        stack = _read_best_slices(
            series_dir, n_slices=n_slices, img_px_size=img_px_size
        )
        for s in range(n_slices):
            per_slice[s].append(stack[s])

    per_slice = [np.stack(x, axis=0).astype(np.float32) for x in per_slice]
    out = []
    for arr in per_slice:
        mx = float(np.max(arr))
        out.append(arr / mx if mx > 0 else arr)
    return out




## === cell 3
BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_PATH = os.path.join(BASE_PATH, "test")

if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError(f"Test folder not found: {TEST_PATH}")

outer_dirs = [d.name for d in os.scandir(TEST_PATH) if d.is_dir()]
has_numeric_outer = any(name.isdigit() for name in outer_dirs)
nested_test = os.path.join(TEST_PATH, "test")
if (not has_numeric_outer) and os.path.isdir(nested_test):
    TEST_PATH = nested_test

print("Using TEST_PATH =", TEST_PATH)

pixels_T2 = load_test_images_by_modality(TEST_PATH, "T2w", n_slices=6, img_px_size=150)
pixels_flair = load_test_images_by_modality(
    TEST_PATH, "FLAIR", n_slices=6, img_px_size=150
)
pixels_t1wce = load_test_images_by_modality(
    TEST_PATH, "T1wCE", n_slices=6, img_px_size=150
)

print(
    "Loaded test batches per slice:",
    "T2:",
    [p.shape for p in pixels_T2],
    "FLAIR:",
    [p.shape for p in pixels_flair],
    "T1wCE:",
    [p.shape for p in pixels_t1wce],
)




## === cell 4
def _predict_positive_class(model, x):
    """
    Returns probability of positive class.
    Handles both (N,2) softmax and (N,1) sigmoid outputs robustly.
    """
    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 2:
        return preds[:, 1].astype(np.float32)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0].astype(np.float32)
    if preds.ndim == 1:
        return preds.astype(np.float32)
    return preds.reshape((preds.shape[0], -1))[:, 0].astype(np.float32)


def _fallback_prob_from_images(pixels_T2, pixels_flair, pixels_t1wce):
    """
    Deterministic fallback: combine per-subject mean intensities across modalities/slices,
    then squash to (0,1) via logistic after standardization.
    """
    feats = []
    for modality in (pixels_T2, pixels_flair, pixels_t1wce):
        slice_means = [arr.mean(axis=(1, 2, 3)) for arr in modality]
        feats.append(np.mean(np.stack(slice_means, axis=1), axis=1))
    X = np.stack(feats, axis=1)  # (N,3)

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Z = (X - mu) / sd

    w = np.array([0.6, 0.3, 0.4], dtype=np.float32)
    b = np.float32(0.0)
    logits = (Z * w.reshape(1, -1)).sum(axis=1) + b

    prob = 1.0 / (1.0 + np.exp(-logits))
    return prob.astype(np.float32)




## === cell 5
if use_pretrained:
    pred_T2_m1 = [_predict_positive_class(model_T2, x) for x in pixels_T2]
    pred_T2_m2 = [_predict_positive_class(model_T2_2, x) for x in pixels_T2]
    pred_T2_m5 = [_predict_positive_class(model_T2_5, x) for x in pixels_T2]
    pred_T2_m6 = [_predict_positive_class(model_T2_6, x) for x in pixels_T2]

    pred_flair_m7 = [_predict_positive_class(model_T2_7, x) for x in pixels_flair]

    pred_t1wce_m3 = [_predict_positive_class(model_T2_3, x) for x in pixels_t1wce]
    pred_t1wce_m4 = [_predict_positive_class(model_T2_4, x) for x in pixels_t1wce]

    all_preds = (
        pred_T2_m1
        + pred_T2_m2
        + pred_t1wce_m3
        + pred_t1wce_m4
        + pred_T2_m5
        + pred_T2_m6
        + pred_flair_m7
    )
    prediction = np.mean(np.stack(all_preds, axis=0), axis=0).astype(np.float32)
else:
    prediction = _fallback_prob_from_images(pixels_T2, pixels_flair, pixels_t1wce)

print(
    "Final prediction stats:",
    "N=",
    int(prediction.shape[0]),
    "min=",
    float(np.min(prediction)),
    "max=",
    float(np.max(prediction)),
    "mean=",
    float(np.mean(prediction)),
)




## === cell 6
def _list_test_ids(path_test):
    ids = []
    for f in os.scandir(path_test):
        if f.is_dir():
            name = f.name
            if name.isdigit():
                ids.append(int(name))
    return sorted(ids)


def create_sub(path_test, prediction):
    cases = _list_test_ids(path_test)
    if len(cases) == 0:
        raise RuntimeError(f"No numeric case folders found under: {path_test}")

    if len(prediction) != len(cases):
        raise ValueError(
            f"Prediction length ({len(prediction)}) != number of test cases ({len(cases)}). "
            f"Check test path and image loading order."
        )

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(float)})
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
    return df


sub_df = create_sub(TEST_PATH, prediction)

sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    if len(sample) == len(sub_df):
        sub_df = (
            sub_df.set_index("BraTS21ID")
            .reindex(sample["BraTS21ID"].values)
            .reset_index()
        )
        sub_df.columns = ["BraTS21ID", "MGMT_value"]

sub_df["MGMT_value"] = sub_df["MGMT_value"].clip(0.0, 1.0)

print("Submission preview:")
print(sub_df.head())
print("Submission shape:", sub_df.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/53977498.py in <cell line: 0>()
     25 
     26 
---> 27 sub_df = create_sub(TEST_PATH, prediction)
     28 
     29 # Align to sample submission order if available (keeps exact expected ordering)

/tmp/ipykernel_11/53977498.py in create_sub(path_test, prediction)
     15 
     16     if len(prediction) != len(cases):
---> 17         raise ValueError(
     18             f"Prediction length ({len(prediction)}) != number of test cases ({len(cases)}). "
     19             f"Check test path and image loading order."

ValueError: Prediction length (60) != number of test cases (59). Check test path and image loading order.

## === cell 7
if sns is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
    except Exception:
        pass



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913900529.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
